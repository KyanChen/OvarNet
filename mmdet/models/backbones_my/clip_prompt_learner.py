import json

import numpy as np
import warnings
from collections import OrderedDict

import torch
import torch.nn as nn
from mmcv.runner import BaseModule, get_dist_info
from torch.nn import TransformerEncoderLayer, LayerNorm, TransformerEncoder

from ..builder import BACKBONES
from .clip import tokenize, _Tokenizer

_tokenizer = _Tokenizer()


@BACKBONES.register_module()
class PromptLearner(BaseModule):
    def __init__(self,
                 attribute_list,
                 clip_model,
                 n_ctx=16,
                 ctx_init='',
                 c_specific=False,
                 class_token_position='end',
                 load_ckpt_from=None
                 ):
        super().__init__()
        n_cls = len(attribute_list)
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if c_specific:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        rank, world_size = get_dist_info()
        if rank == 0:
            print(f'Initial context: "{prompt_prefix}"')
            print(f"Number of context words (tokens): {n_ctx}")

        # self.prompt_vectors = nn.Parameter(ctx_vectors)  # to be optimized
        # self.ctx = self.prompt_vectors
        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in attribute_list]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = class_token_position
        if load_ckpt_from is not None:
            state_dict = torch.load(load_ckpt_from, map_location="cpu")['state_dict']
            ctx_data = state_dict['prompt_learner.ctx']
            self.ctx.data.copy_(ctx_data)

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


@BACKBONES.register_module()
class PromptAttributes(BaseModule):
    def __init__(self,
                 attribute_list,
                 clip_model,
                 prompt_config=dict(
                     n_prompt=16,
                     is_att_specific=False,
                     att_position='mid',
                     att2type=None,
                     context_length=77,
                     n_prompt_type=8,
                     generated_context=False,
                     pos_emb=True,
                 ),
                 load_ckpt_from=None,
                 shared_prompt_vectors=None,
                 self_name='',
                 ):
        # import pdb
        # pdb.set_trace()
        super(PromptAttributes, self).__init__()
        self.prompt_config = prompt_config
        n_att = len(attribute_list)
        word_dim = clip_model.ln_final.weight.shape[0]
        # clip_imsize = clip_model.visual.input_resolution
        n_prompt_vec = prompt_config.get('n_prompt', 16)
        att_position = prompt_config.get('att_position', 16)
        is_att_specific = prompt_config.get('is_att_specific', False)
        self.att2type = prompt_config.get('att2type', None)
        n_prompt_type = prompt_config.get('n_prompt_type', None)
        self.generated_context = prompt_config.get('generated_context', False)
        pos_emb = prompt_config.get('pos_emb', False)
        if shared_prompt_vectors is None:
            if is_att_specific:
                print("Initializing att-specific contexts")
                prompt_vectors = torch.empty(n_att, n_prompt_vec, word_dim, dtype=torch.float32)
            else:
                prompt_vectors = torch.empty(n_prompt_vec, word_dim, dtype=torch.float32)
                nn.init.normal_(prompt_vectors, std=0.02)
        if n_prompt_type and not self.generated_context:
            assert n_prompt_type == n_prompt_vec
            att2types = json.load(open(self.att2type, 'r'))
            id2type = att2types['id2type']
            prompt_vectors = torch.empty(1+len(id2type), n_prompt_vec, word_dim, dtype=torch.float32)
            nn.init.normal_(prompt_vectors, std=0.02)

            att2typeid = att2types['att2typeid']
            self.att_type_id = [att2typeid[attribute] for attribute in attribute_list]

        # prompt_prefix = " ".join(["X"] * n_ctx)
        # print(f'Initial context: "{prompt_prefix}"')
        rank, world_size = get_dist_info()
        if rank == 0:
            print(f"Number of all-shared prompt (tokens): {n_prompt_vec}")
            print(f"Number of type-shared prompt (tokens): {n_prompt_type}")
            print('generated context: ', self.generated_context)
            print('is pos emb: ', pos_emb)
            print('att position: ', att_position)
            print('att type: ', self.att2type)

        # self.ctx = nn.Parameter(prompt_vectors)  # to be optimized
        if shared_prompt_vectors is not None:
            self.prompt_vectors = shared_prompt_vectors
        else:
            if len(prompt_vectors):
                self.prompt_vectors = nn.Parameter(prompt_vectors)
            else:
                self.prompt_vectors = prompt_vectors
        sot_token = torch.tensor([_tokenizer.encoder["<|startoftext|>"]], dtype=torch.long)
        eot_token = torch.tensor([_tokenizer.encoder["<|endoftext|>"]], dtype=torch.long)
        pad_token = torch.tensor([0], dtype=torch.long)
        if self.att2type is not None:
            att2types = json.load(open(self.att2type, 'r'))
            id2type = att2types['id2type']
            att2typeid = att2types['att2typeid']
            att_type_list = []
            for attribute in attribute_list:
                att_group_member_num = len(attribute.split(':')[-1].split('/'))
                att_type_list += [id2type[str(att2typeid[attribute])]] * att_group_member_num
                # att_type_list += [attribute] * att_group_member_num
            # att_type_list = [f'And it is a {att_type}.' for att_type in att_type_list]
            att_type_list = [att_type.replace("_", " ") for att_type in att_type_list]
            type_tokens = [torch.tensor(_tokenizer.encode(att_type)) for att_type in att_type_list]
            self.type_embeddings = [clip_model.token_embedding(x).detach() for x in type_tokens]

        # attribute_list = [attribute.replace("_", " ") for attribute in attribute_list]
        # attribute_list = [f'It is a photo of {attribute}.' for attribute in attribute_list]
        # attribute_list = [f'The attribute of the object is {attribute}.' for attribute in attribute_list]

        attribute_list = [attribute.split(':')[-1].split('/') for attribute in attribute_list]
        self.att_group_member_num = [len(x) for x in attribute_list]
        attribute_list = [t.replace("_", " ") for x in attribute_list for t in x]
        if rank == 0:
            print('att group total: ', len(attribute_list))
        attribute_tokens = [torch.tensor(_tokenizer.encode(attribute)) for attribute in attribute_list]
        self.register_buffer('sot_embedding', clip_model.token_embedding(sot_token).detach())
        self.register_buffer('eot_embedding', clip_model.token_embedding(eot_token).detach())
        self.register_buffer('pad_embedding', clip_model.token_embedding(pad_token).detach())
        self.attribute_embeddings = [clip_model.token_embedding(x).detach() for x in attribute_tokens]

        if self.generated_context:
            self.att_len = [len(x) for x in self.attribute_embeddings]
            if self.with_att_type:
                self.type_len = [len(x) for x in self.type_embeddings]

            self.max_att_len = max(self.att_len) + 2
            if self.with_att_type:
                self.max_att_len = max(self.att_len) + max(self.type_len)
            self.transformer_layer = self.build_transformer_encoder(
                embed_dim=word_dim,
                num_encoder_layers=2,
                dim_feedforward=1024
            )
            if pos_emb:
                self.pos_embed = nn.Parameter(torch.randn(1, self.max_att_len+n_prompt_vec, word_dim) * .02)

        if load_ckpt_from is not None:
            state_dict = torch.load(load_ckpt_from, map_location="cpu")['state_dict']
            prompt_vectors = state_dict[f'{self_name}.prompt_vectors']
            if rank == 0:
                print(f'load prompt vectors from {load_ckpt_from}')
            self.prompt_vectors.data.copy_(prompt_vectors)

        # prompt_context, eot_index = self.rearrange_context(**prompt_config)
        # self.register_buffer("prompt_context", prompt_context)  # SOS
        # self.register_buffer("eot_index", eot_index)  # CLS, EOS

        # prompts = [prompt_prefix + " " + name + "." for name in classnames]
        # tokenized_prompts = torch.cat([tokenize(p) for p in prompts])
        #
        # self.tokenized_prompts = tokenized_prompts  # torch.Tensor

    def build_transformer_encoder(
            self, embed_dim, num_encoder_layers=3, dim_feedforward=2048
    ):
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            norm_first=True,
            activation='gelu',
            batch_first=True
        )
        # encoder_norm = LayerNorm(embed_dim)
        encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        return encoder

    def rearrange_context(
            self,
            inds=None,
            context_length=77,
            is_att_specific=False,
            att_position='mid',
            att2type=None,
            n_prompt_type=None,
            *args,
            **kwargs
    ):
        if att2type is not None:
            self.type_embeddings = [x.to(self.sot_embedding.device) for x in self.type_embeddings]
            assert att_position == 'mid'
        if inds is not None:
            attribute_embeddings = [self.attribute_embeddings[idx] for idx in inds]
        else:
            attribute_embeddings = self.attribute_embeddings
        attribute_embeddings = [x.to(self.sot_embedding.device) for x in attribute_embeddings]
        rearranged_context = []
        eot_index = []
        for i in range(len(attribute_embeddings)):
            rearranged_context_tmp = [self.sot_embedding]

            if is_att_specific:
                prompt_vectors = self.prompt_vectors[i]
            else:
                prompt_vectors = self.prompt_vectors.to(self.sot_embedding.device)

            if att_position == 'end':
                rearranged_context_tmp.append(prompt_vectors)
                rearranged_context_tmp.append(attribute_embeddings[i])
                rearranged_context_tmp.append(self.eot_embedding)
            elif att_position == 'front':
                rearranged_context_tmp.append(attribute_embeddings[i])
                rearranged_context_tmp.append(prompt_vectors)
                rearranged_context_tmp.append(self.eot_embedding)
            elif att_position == 'mid':
                if n_prompt_type:
                    n_part = prompt_vectors.size(1) // 2
                    all_shared_part_1 = prompt_vectors[0, :n_part]
                    all_shared_part_2 = prompt_vectors[0, n_part:]
                    type_shared_vec = prompt_vectors[self.att_type_id[i]+1]
                    type_shared_part_1 = type_shared_vec[:n_part]
                    type_shared_part_2 = type_shared_vec[n_part:]
                    rearranged_context_tmp.append(all_shared_part_1)
                    rearranged_context_tmp.append(type_shared_part_1)
                    rearranged_context_tmp.append(attribute_embeddings[i])
                    rearranged_context_tmp.append(type_shared_part_2)
                    rearranged_context_tmp.append(all_shared_part_2)
                    rearranged_context_tmp.append(self.eot_embedding)
                elif att2type is not None:
                    if len(prompt_vectors):
                        n_part = len(prompt_vectors) // 3
                        part_1 = prompt_vectors[:n_part]
                        part_2 = prompt_vectors[n_part:n_part * 2]
                        part_3 = prompt_vectors[n_part * 2:]
                        rearranged_context_tmp.append(part_1)
                        rearranged_context_tmp.append(attribute_embeddings[i])
                        rearranged_context_tmp.append(part_2)
                        rearranged_context_tmp.append(self.type_embeddings[i])
                        rearranged_context_tmp.append(part_3)
                        rearranged_context_tmp.append(self.eot_embedding)
                    else:
                        rearranged_context_tmp.append(attribute_embeddings[i])
                        rearranged_context_tmp.append(self.type_embeddings[i])
                        rearranged_context_tmp.append(self.eot_embedding)
                else:
                    n_part = len(prompt_vectors) // 2
                    part_1 = prompt_vectors[:n_part]
                    part_2 = prompt_vectors[n_part:]
                    rearranged_context_tmp.append(part_1)
                    rearranged_context_tmp.append(attribute_embeddings[i])
                    rearranged_context_tmp.append(part_2)
                    rearranged_context_tmp.append(self.eot_embedding)
            else:
                raise NotImplementedError
            rearranged_context_tmp = torch.cat(rearranged_context_tmp, dim=0)
            eot_index.append(len(rearranged_context_tmp) - 1)
            rearranged_context_tmp = [rearranged_context_tmp] + \
                                     [self.pad_embedding] * (context_length - len(rearranged_context_tmp))
            rearranged_context_tmp = torch.cat(rearranged_context_tmp, dim=0)
            rearranged_context.append(rearranged_context_tmp)
        return torch.stack(rearranged_context, dim=0), torch.tensor(eot_index, dtype=torch.long, device=self.prompt_vectors.device)

    def rearrange_generated_context(
            self,
            prompt_vectors,
            context_length=77,
            att_position='none',
            with_att_type=False,
            *args,
            **kwargs
    ):
        rearranged_context = []
        eot_index = []
        for i in range(len(prompt_vectors)):
            rearranged_context_tmp = [self.sot_embedding]
            if att_position == 'end':
                rearranged_context_tmp.append(prompt_vectors[i])
                rearranged_context_tmp.append(self.attribute_embeddings[i])
                rearranged_context_tmp.append(self.eot_embedding)
            elif att_position == 'front':
                rearranged_context_tmp.append(self.attribute_embeddings[i])
                rearranged_context_tmp.append(prompt_vectors[i])
                rearranged_context_tmp.append(self.eot_embedding)
            elif att_position == 'mid':
                # import pdb
                # pdb.set_trace()
                if with_att_type:
                    n_part = prompt_vectors.size(1) // 3
                    part_1 = prompt_vectors[i, :n_part]
                    part_2 = prompt_vectors[i, n_part:n_part * 2]
                    part_3 = prompt_vectors[i, n_part * 2:]
                    rearranged_context_tmp.append(part_1)
                    rearranged_context_tmp.append(self.attribute_embeddings[i])
                    rearranged_context_tmp.append(part_2)
                    rearranged_context_tmp.append(self.type_embeddings[i])
                    rearranged_context_tmp.append(part_3)
                    rearranged_context_tmp.append(self.eot_embedding)
                else:
                    n_part = prompt_vectors.size(1) // 2
                    part_1 = prompt_vectors[i, :n_part]
                    part_2 = prompt_vectors[i, n_part:]
                    rearranged_context_tmp.append(part_1)
                    rearranged_context_tmp.append(self.attribute_embeddings[i])
                    rearranged_context_tmp.append(part_2)
                    rearranged_context_tmp.append(self.eot_embedding)
            elif att_position == 'none':
                rearranged_context_tmp.append(prompt_vectors[i])
                rearranged_context_tmp.append(self.eot_embedding)
            else:
                raise NotImplementedError
            rearranged_context_tmp = torch.cat(rearranged_context_tmp, dim=0)
            eot_index.append(len(rearranged_context_tmp) - 1)
            rearranged_context_tmp = [rearranged_context_tmp] + \
                                     [self.pad_embedding] * (context_length - len(rearranged_context_tmp))
            rearranged_context_tmp = torch.cat(rearranged_context_tmp, dim=0)
            rearranged_context.append(rearranged_context_tmp)
        return torch.stack(rearranged_context, dim=0), torch.tensor(eot_index, dtype=torch.long)

    def forward(self, inds=None):
        if self.generated_context:
            self.attribute_embeddings = [x.to(self.prompt_vectors.device) for x in self.attribute_embeddings]
            if self.with_att_type:
                self.type_embeddings = [x.to(self.prompt_vectors.device) for x in self.type_embeddings]

            rearranged_context = []
            for i in range(len(self.attribute_embeddings)):
                rearranged_context_tmp = []
                # if self.with_att_type:
                #     n_part = len(self.prompt_vectors) // 3
                #     part_1 = self.prompt_vectors[:n_part]
                #     part_2 = self.prompt_vectors[n_part:n_part * 2]
                #     part_3 = self.prompt_vectors[n_part * 2:]
                #     rearranged_context_tmp.append(part_1)
                #     rearranged_context_tmp.append(self.attribute_embeddings[i])
                #     rearranged_context_tmp.append(part_2)
                #     rearranged_context_tmp.append(self.type_embeddings[i])
                #     rearranged_context_tmp.append(part_3)
                # else:
                rearranged_context_tmp.append(self.prompt_vectors)
                rearranged_context_tmp.append(self.attribute_embeddings[i])
                if self.with_att_type:
                    rearranged_context_tmp.append(self.type_embeddings[i])
                    rearranged_context_tmp += [self.pad_embedding] * (self.max_att_len - self.att_len[i] - self.type_len[i])
                else:
                    rearranged_context_tmp += [self.pad_embedding] * (
                                self.max_att_len - self.att_len[i])
                rearranged_context_tmp = torch.cat(rearranged_context_tmp, dim=0)
                rearranged_context.append(rearranged_context_tmp)
            rearranged_context = torch.stack(rearranged_context, dim=0)
            if hasattr(self, 'pos_embed'):
                rearranged_context = rearranged_context + self.pos_embed
            prompt = self.transformer_layer(rearranged_context)
            prompt = prompt[:, :len(self.prompt_vectors), :]
            prompt_context, eot_index = self.rearrange_generated_context(prompt, **self.prompt_config)
        else:
            prompt_context, eot_index = self.rearrange_context(inds, **self.prompt_config)
        return prompt_context, eot_index, self.att_group_member_num
