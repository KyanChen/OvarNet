import re

import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from ..builder import BACKBONES


@BACKBONES.register_module()
class OFAPromptLearner(BaseModule):
    def __init__(self,
                 classnames,
                 model,
                 task,
                 n_ctx=16,
                 ctx_init='',
                 c_specific=False,
                 class_token_position='end'
                 ):
        super().__init__()

        n_cls = len(classnames)

        ctx_dim = model.encoder.embed_tokens.weight.shape[1]  # 256 model.embed_tokens.weight.shape[1]

        if ctx_init:
            ctx_init = self.pre_question(ctx_init, model.max_src_length)
            ctx_init = ctx_init + '?' if not ctx_init.endswith('?') else ctx_init
            prompt = self.encode_text(' {}'.format(ctx_init), append_bos=True, append_eos=True)
            n_ctx = prompt.ne(model.pad).long().sum()

            with torch.no_grad():
                embedding = model.embed_tokens(prompt)  # 1x9x256
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if c_specific:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        # ctx_init = self.pre_question(ctx_init, model.max_src_length)
        # ctx_init = ctx_init + '?' if not ctx_init.endswith('?') else ctx_init
        # prompt = self.encode_text(' {}'.format(ctx_init), append_bos=True, append_eos=True)
        # n_ctx = (prompt.ne(model.eos) & prompt.ne(model.pad)).long().sum()

        self.max_src_length = task.max_src_length
        classnames = [self.pre_question(name, self.max_src_length) for name in classnames]
        self.name_lens = [len(x.split()) for x in classnames]
        prompts = [prompt_prefix + " " + name + "?" for name in classnames]

        self.tgt_dict = task.tgt_dict
        self.src_dict = task.src_dict
        self.bpe = task.bpe

        tokenized_prompts = torch.stack(
            [self.encode_text(' {}'.format(p),
                              pad=True,
                              context_length=self.max_src_length,
                              append_bos=True,
                              append_eos=True
                              ) for p in prompts])

        # self.prev_output_item = torch.cat(
        #     [self.encode_text(' {}'.format(x), length=model.cfg.max_src_length, append_bos=True, append_eos=False)
        #      for x in classnames])
        # self.target_item = torch.cat(
        #     [self.encode_text(' {}'.format(x), length=model.cfg.max_src_length, append_bos=False, append_eos=True)
        #      for x in classnames])

        with torch.no_grad():
            embedding = model.encoder.embed_tokens(tokenized_prompts)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.class_token_position = class_token_position

    def encode_text(self, texts, pad=True, context_length=128, truncate=False, append_bos=False, append_eos=False):
        tokens = self.tgt_dict.encode_line(
            line=self.bpe.encode(texts),
            add_if_not_exist=False,
            append_eos=False
        ).long()
        if append_bos:
            bos_item = torch.LongTensor([self.src_dict.bos()])  # [0]
            tokens = torch.cat([bos_item, tokens])
        if append_eos:
            eos_item = torch.LongTensor([self.src_dict.eos()])
            tokens = torch.cat([tokens, eos_item])
        if pad:
            result = torch.ones(context_length, dtype=torch.long) * self.src_dict.pad()

            if len(tokens) > context_length:
                if truncate:
                    tokens = tokens[:context_length]
                    tokens[-1] = self.src_dict.eos()
                else:
                    raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
            result[:len(tokens)] = tokens
        else:
            result = tokens

        return result

    def pre_question(self, question, max_ques_words):
        question = question.lower().lstrip(",.!?*#:;~").replace('-', ' ').replace('/', ' ')
        question = re.sub(
            r"\s{2,}",
            ' ',
            question,
        )
        question = question.rstrip('\n')
        question = question.strip(' ')
        # truncate question
        question_words = question.split(' ')
        if len(question_words) > max_ques_words:
            question = ' '.join(question_words[:max_ques_words])
        return question

    def forward(self):
        ctx = self.ctx  # 4x512
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)  # 620x4x512

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

