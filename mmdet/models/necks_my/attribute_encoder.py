import json

import torch
from einops import repeat, rearrange
from mmcv.cnn import ConvModule
from torch import nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder, LayerNorm, Transformer, TransformerDecoderLayer, \
    TransformerDecoder
from torch.nn.init import xavier_uniform_
import torch.nn.functional as F
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import NECKS, build_neck, build_roi_extractor, build_shared_head
from mmcv.runner import BaseModule

import gzip
import html
import os
from functools import lru_cache

import ftfy
import regex as re


@NECKS.register_module()
class AttributeEncoder(BaseModule):
    def __init__(
            self,
            attribute_id_map,
            n_ctx=16,
            prompt_num=8,
            class_token_position='mid',
            context_length=32,
            model_dim=512,
            out_channels=1024,
            train_cfg=None,
            test_cfg=None,
            pretrained=None,
            init_cfg=None
    ):
        super(AttributeEncoder, self).__init__(init_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.class_token_position = class_token_position
        self.prompt_num = prompt_num
        self.context_length = context_length
        self.model_dim = model_dim

        self.tokenizer = SimpleTokenizer()
        vocab_size = len(self.tokenizer.encoder)
        ctx_vectors = torch.empty(prompt_num, n_ctx, model_dim)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized, 8x16x512
        self.token_embedding = nn.Embedding(vocab_size, model_dim)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, model_dim))
        nn.init.normal_(self.positional_embedding, std=0.01)

        self.attribute_encoder = self.build_attribute_encoder(
            num_encoder_layers=3, dim_feedforward=self.model_dim*2
        )

        self.seq_squeezer = self.build_seq_squeezer(num_decoder_layers=2, dim_feedforward=self.model_dim*2)
        self.seq_representation = nn.Parameter(torch.empty(1, model_dim))
        nn.init.normal_(self.seq_representation, std=0.02)

        self.prompt_squeezer = self.build_prompt_squeezer(
            num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=self.model_dim*2
        )
        self.prompt_attention_map = nn.Parameter(torch.empty(prompt_num, model_dim))
        nn.init.normal_(self.prompt_attention_map, std=0.02)
        self.prompt_attention_proj = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(model_dim // 2, 1)
        )

        self.attribute_decoder = self.build_attribute_decoder(num_decoder_layers=6, dim_feedforward=self.model_dim * 2)
        self.attribute_representation = nn.Parameter(torch.empty(1, model_dim))
        nn.init.normal_(self.attribute_representation, std=0.02)
        self.attribute_proj_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim*2),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(model_dim*2, out_channels),
        )

        self.attribute_id_map = json.load(open(attribute_id_map, 'r'))

        sub_attributes = self.attribute_id_map['attribute2id'].keys()
        self.sub_attributes2id = {sub_attribute: idx for idx, sub_attribute in enumerate(sub_attributes)}
        self.num_sub_attributes = self.attribute_id_map['num_sub_attributes']
        self.num_attributes = self.attribute_id_map['num_attributes']

        per_attribute_lens = [len(self.tokenizer.encode(attribute)) for attribute in sub_attributes]
        prompts = [prompt_prefix + " " + attribute + "?" for attribute in sub_attributes]
        self.tokenized_prompts = torch.cat([self.tokenize(p) for p in prompts])  # N_Attribute len_seq

        self.src_key_padding_mask = self.tokenized_prompts == 0  # regard 0 as PAD

        self.n_ctx = n_ctx
        self.per_attribute_lens = torch.tensor(per_attribute_lens)

    def tokenize(self, texts, context_length=32, truncate=False):
        """
        Returns the tokenized representation of given input string(s)

        Parameters
        ----------
        texts : Union[str, List[str]]
            An input string or a list of input strings to tokenize

        context_length : int
            The context length to use; all CLIP models use 77 as the context length

        truncate: bool
            Whether to truncate the text in case its encoding is longer than the context length

        Returns
        -------
        A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
        """
        if isinstance(texts, str):
            texts = [texts]

        sot_token = self.tokenizer.encoder["<|startoftext|>"]
        eot_token = self.tokenizer.encoder["<|endoftext|>"]
        all_tokens = [[sot_token] + self.tokenizer.encode(text) + [eot_token] for text in texts]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                if truncate:
                    tokens = tokens[:context_length]
                    tokens[-1] = eot_token
                else:
                    raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result

    def build_attribute_encoder(
            self, num_encoder_layers=3, dim_feedforward=2048
    ):
        encoder_layer = TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=8,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        encoder_norm = LayerNorm(self.model_dim)
        encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        return encoder

    def build_seq_squeezer(
            self, num_decoder_layers=4, dim_feedforward=2048
    ):
        decoder_layer = TransformerDecoderLayer(
            d_model=self.model_dim,
            nhead=8,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        decoder_norm = LayerNorm(self.model_dim)
        decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        return decoder

    def build_prompt_squeezer(
            self, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=2048):
        transformer_layers = Transformer(
            d_model=self.model_dim,
            nhead=8,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        return transformer_layers

    def build_attribute_decoder(
            self, num_decoder_layers=4, dim_feedforward=2048
    ):
        decoder_layer = TransformerDecoderLayer(
            d_model=self.model_dim,
            nhead=8,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        decoder_norm = LayerNorm(self.model_dim)
        decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        return decoder

    def attribute2subattribute_idxs(self, attribute_idxs):
        selected_attribute_idxs = []
        for attribute_idx in attribute_idxs:
            attributes = self.attribute_id_map['id2attribute'][repr(attribute_idx.item())]
            attribute = attributes[torch.randint(len(attributes), []).item()]
            # attribute = attributes[0]
            selected_attribute_idxs.append(self.sub_attributes2id[attribute])
        return torch.tensor(selected_attribute_idxs)

    def forward(self, attribute_idxs, **kwargs):
        return self.forward_train(attribute_idxs, **kwargs)

    def get_prompts_embeddings(self, device):
        embedding = self.token_embedding(self.tokenized_prompts.to(device))

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer('token_prefix', embedding[:, :1, :])  # SOS, N_Att 1 model_dim
        self.register_buffer('token_suffix', embedding[:, 1 + self.n_ctx:, :]) # ATT, EOS

    def forward_train(self, attribute_idxs, device, **kwargs):
        # attribute_idxs is attribute not sub_attribute

        # if not hasattr(self, 'token_prefix'):
        #     self.get_prompts_embeddings(device)

        attribute_idxs = self.attribute2subattribute_idxs(attribute_idxs)

        tokenized_prompts = self.tokenized_prompts[attribute_idxs]
        embedding = self.token_embedding(tokenized_prompts.to(device))
        prefix = embedding[:, :1, :]  # SOS, N_Att 1 model_dim
        suffix = embedding[:, 1 + self.n_ctx:, :]

        # # attribute_ids B
        # prefix = self.token_prefix[attribute_idxs]  # BxLxC
        # suffix = self.token_suffix[attribute_idxs]  # # BxLxC
        attribute_lens = self.per_attribute_lens[attribute_idxs]
        ctx = self.ctx  # 8x16x512

        prefix = repeat(prefix, 'B L C -> B prompt_num L C', prompt_num=self.prompt_num)
        suffix = repeat(suffix, 'B L C -> B prompt_num L C', prompt_num=self.prompt_num)
        ctx = repeat(ctx, 'prompt_num L C -> B prompt_num L C', B=attribute_idxs.size(0))

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (B prompt_num 1, dim)
                    ctx,  # (B prompt_num n_ctx, dim)
                    suffix,  # (B prompt_num *, dim)
                ],
                dim=-2,
            )

        elif self.class_token_position == "mid":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(len(attribute_idxs)):
                attribute_len = attribute_lens[i]
                prefix_i = prefix[i: i + 1, ...]
                att_i = suffix[i: i + 1, :, :attribute_len, :]
                suffix_i = suffix[i: i + 1, :, attribute_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, :, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        att_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=-2,
                )

                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in attribute_idxs:
                attribute_len = self.per_attribute_lens[i]
                prefix_i = prefix[i: i + 1, ...]
                att_i = suffix[i: i + 1, :, :attribute_len, :]
                suffix_i = suffix[i: i + 1, :, attribute_len:, :]
                ctx_i = ctx[i: i + 1, ...]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        att_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=-2,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError
        # prompts : B prompt_num L C

        x = prompts + self.positional_embedding  # broadcast, B prompt_num L C

        x_encoded = rearrange(x, 'B prompt_num L C -> (B prompt_num) L C')
        src_key_padding_mask = self.src_key_padding_mask[attribute_idxs].to(x.device)  # BxLxC
        src_key_padding_mask_expand = repeat(
            src_key_padding_mask, 'B L -> (B prompt_num) L', prompt_num=self.prompt_num
        )
        x_memory = self.attribute_encoder(
            x_encoded, src_key_padding_mask=src_key_padding_mask_expand
        )  # (B prompt_num) L C
        seq_representation = repeat(
            self.seq_representation, 'L C -> (B prompt_num) L C', B=attribute_idxs.size(0), prompt_num=self.prompt_num
        )  # (B prompt_num) 1 C

        squeezed_seq_x = self.seq_squeezer(
            seq_representation, x_memory, memory_key_padding_mask=src_key_padding_mask_expand
        )  # (B prompt_num) 1 C
        squeezed_seq_x = rearrange(squeezed_seq_x, '(B prompt_num) L C -> B (prompt_num L) C', prompt_num=self.prompt_num)

        prompt_attention_map = repeat(
            self.prompt_attention_map, 'prompt_num C -> B prompt_num C', B=attribute_idxs.size(0)
        )
        prompt_attention_map = self.prompt_squeezer(squeezed_seq_x, prompt_attention_map)  # B prompt_num C
        prompt_attention_vector = self.prompt_attention_proj(prompt_attention_map)  # B prompt_num 1
        prompt_attention_vector = F.softmax(prompt_attention_vector.squeeze(dim=-1), dim=-1)  # B prompt_num

        x_memory = rearrange(x_memory, '(B prompt_num) L C -> B prompt_num L C', B=attribute_idxs.size(0))
        # x = x * prompts_weight[..., None, None]
        x_memory = x_memory * prompt_attention_vector[..., None, None]  # (B prompt_num) L C
        x_memory = torch.sum(x_memory, dim=1).squeeze(dim=1)  # B L C

        attribute_representation = repeat(self.attribute_representation, 'L C -> B L C', B=attribute_idxs.size(0))
        attribute_representation = torch.cat((attribute_representation, x_memory), dim=1)
        x = self.attribute_decoder(
            attribute_representation, x_memory, memory_key_padding_mask=src_key_padding_mask
        )  # B 1 model_dim

        text_feature = self.attribute_proj_head(x[:, 0])

        return text_feature

    def simple_test(self, x, proposal_list, **kwargs):
        return 0


# 预先读入内存缓存
@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class SimpleTokenizer:
    def __init__(self, bpe_path="bpe_simple_vocab_16e6.txt.gz"):
        if not os.path.exists(bpe_path):
            bpe_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")
        assert os.path.exists(bpe_path)
        self.byte_encoder = bytes_to_unicode()  # dict int:char
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}  # dict char:int
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))  # 49408
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text
