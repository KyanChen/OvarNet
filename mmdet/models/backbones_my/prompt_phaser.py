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
class PromptAttributeWords(BaseModule):
    def __init__(self,
                 clip_model,
                 prompt_config=dict(
                     n_prompt=30,
                     att_position='mid',
                     context_length=77,
                 ),
                 load_ckpt_from=None,
                 self_name='',
                 ):
        # import pdb
        # pdb.set_trace()
        super(PromptAttributeWords, self).__init__()
        self.prompt_config = prompt_config
        word_dim = clip_model.ln_final.weight.shape[0]
        # clip_imsize = clip_model.visual.input_resolution
        n_prompt_vec = prompt_config.get('n_prompt', 30)
        self.att_position = prompt_config.get('att_position', 'mid')
        self.context_length = prompt_config.get('context_length', '77')

        prompt_vectors = torch.empty(n_prompt_vec, word_dim, dtype=torch.float32)
        nn.init.normal_(prompt_vectors, std=0.02)

        rank, world_size = get_dist_info()
        if rank == 0:
            print(f"Number of attribute prompt (tokens): {n_prompt_vec}")
            print('att position: ', self.att_position)

        if len(prompt_vectors):
            self.prompt_vectors = nn.Parameter(prompt_vectors)
        else:
            self.prompt_vectors = prompt_vectors

        sot_token = torch.tensor([_tokenizer.encoder["<|startoftext|>"]], dtype=torch.long)
        eot_token = torch.tensor([_tokenizer.encoder["<|endoftext|>"]], dtype=torch.long)
        pad_token = torch.tensor([0], dtype=torch.long)

        self.register_buffer('sot_embedding', clip_model.token_embedding(sot_token).detach())
        self.register_buffer('eot_embedding', clip_model.token_embedding(eot_token).detach())
        self.register_buffer('pad_embedding', clip_model.token_embedding(pad_token).detach())
        self.token_emb = clip_model.token_embedding

        if load_ckpt_from is not None:
            state_dict = torch.load(load_ckpt_from, map_location="cpu")['state_dict']
            prompt_vectors = state_dict[f'{self_name}.prompt_vectors']
            if rank == 0:
                print(f'load prompt vectors from {load_ckpt_from}')
            self.prompt_vectors.data.copy_(prompt_vectors)

    def forward(self, attribute_word, device):
        words = [x.replace("_", " ") for x in attribute_word]
        word_tokens = [torch.tensor(_tokenizer.encode(x), device=device) for x in words]
        word_embeddings = [self.token_emb(x).detach() for x in word_tokens]
        prompt_vectors = self.prompt_vectors.to(device)

        rearranged_context = []
        eot_index = []
        for i in range(len(word_embeddings)):
            rearranged_context_tmp = [self.sot_embedding]
            if self.att_position == 'end':
                rearranged_context_tmp.append(prompt_vectors)
                rearranged_context_tmp.append(self.attribute_embeddings[i])
                rearranged_context_tmp.append(self.eot_embedding)
            elif self.att_position == 'front':
                rearranged_context_tmp.append(self.attribute_embeddings[i])
                rearranged_context_tmp.append(prompt_vectors)
                rearranged_context_tmp.append(self.eot_embedding)
            elif self.att_position == 'mid':
                n_part = len(prompt_vectors) // 2
                part_1 = prompt_vectors[:n_part]
                part_2 = prompt_vectors[n_part:]
                rearranged_context_tmp.append(part_1)
                rearranged_context_tmp.append(word_embeddings[i])
                rearranged_context_tmp.append(part_2)
                rearranged_context_tmp.append(self.eot_embedding)
            else:
                raise NotImplementedError
            rearranged_context_tmp = torch.cat(rearranged_context_tmp, dim=0)
            eot_index.append(len(rearranged_context_tmp) - 1)
            rearranged_context_tmp = [rearranged_context_tmp] + \
                                     [self.pad_embedding] * (self.context_length - len(rearranged_context_tmp))
            rearranged_context_tmp = torch.cat(rearranged_context_tmp, dim=0)
            rearranged_context.append(rearranged_context_tmp)
        return torch.stack(rearranged_context, dim=0), torch.tensor(eot_index, dtype=torch.long, device=device), None


@BACKBONES.register_module()
class PromptPhases(BaseModule):
    def __init__(self,
                 clip_model,
                 prompt_config=dict(
                     n_prompt=16,
                     att_position='mid',
                     context_length=77,
                 ),
                 load_ckpt_from=None,
                 self_name='',
                 ):
        # import pdb
        # pdb.set_trace()
        super(PromptPhases, self).__init__()
        self.prompt_config = prompt_config
        word_dim = clip_model.ln_final.weight.shape[0]
        # clip_imsize = clip_model.visual.input_resolution
        n_prompt_vec = prompt_config.get('n_prompt', 16)
        self.att_position = prompt_config.get('att_position', 'mid')
        self.context_length = prompt_config.get('context_length', '77')

        prompt_vectors = torch.empty(n_prompt_vec, word_dim, dtype=torch.float32)
        nn.init.normal_(prompt_vectors, std=0.02)

        rank, world_size = get_dist_info()
        if rank == 0:
            print(f"Number of phase prompt (tokens): {n_prompt_vec}")
            print('att position: ', self.att_position)

        if len(prompt_vectors):
            self.prompt_vectors = nn.Parameter(prompt_vectors)
        else:
            self.prompt_vectors = prompt_vectors

        sot_token = torch.tensor([_tokenizer.encoder["<|startoftext|>"]], dtype=torch.long)
        eot_token = torch.tensor([_tokenizer.encoder["<|endoftext|>"]], dtype=torch.long)
        pad_token = torch.tensor([0], dtype=torch.long)

        self.register_buffer('sot_embedding', clip_model.token_embedding(sot_token).detach())
        self.register_buffer('eot_embedding', clip_model.token_embedding(eot_token).detach())
        self.register_buffer('pad_embedding', clip_model.token_embedding(pad_token).detach())
        self.token_emb = clip_model.token_embedding

        if load_ckpt_from is not None:
            state_dict = torch.load(load_ckpt_from, map_location="cpu")['state_dict']
            prompt_vectors = state_dict[f'{self_name}.prompt_vectors']
            if rank == 0:
                print(f'load prompt vectors from {load_ckpt_from}')
            self.prompt_vectors.data.copy_(prompt_vectors)

    def forward(self, phases, device):
        phases = [x.replace("_", " ") for x in phases]
        phase_tokens = [torch.tensor(_tokenizer.encode(x), device=device).long() for x in phases]
        phase_embeddings = [self.token_emb(x).detach() for x in phase_tokens]
        prompt_vectors = self.prompt_vectors.to(device)

        rearranged_context = []
        eot_index = []
        for i in range(len(phase_embeddings)):
            rearranged_context_tmp = [self.sot_embedding]
            if self.att_position == 'end':
                rearranged_context_tmp.append(prompt_vectors)
                rearranged_context_tmp.append(self.attribute_embeddings[i])
                rearranged_context_tmp.append(self.eot_embedding)
            elif self.att_position == 'front':
                rearranged_context_tmp.append(self.attribute_embeddings[i])
                rearranged_context_tmp.append(prompt_vectors)
                rearranged_context_tmp.append(self.eot_embedding)
            elif self.att_position == 'mid':
                n_part = len(prompt_vectors) // 2
                part_1 = prompt_vectors[:n_part]
                part_2 = prompt_vectors[n_part:]
                rearranged_context_tmp.append(part_1)
                rearranged_context_tmp.append(phase_embeddings[i])
                rearranged_context_tmp.append(part_2)
                rearranged_context_tmp.append(self.eot_embedding)
            else:
                raise NotImplementedError
            rearranged_context_tmp = torch.cat(rearranged_context_tmp, dim=0)
            eot_index.append(len(rearranged_context_tmp) - 1)
            rearranged_context_tmp = [rearranged_context_tmp] + \
                                     [self.pad_embedding] * (self.context_length - len(rearranged_context_tmp))
            rearranged_context_tmp = torch.cat(rearranged_context_tmp, dim=0)
            rearranged_context.append(rearranged_context_tmp)
        return torch.stack(rearranged_context, dim=0), torch.tensor(eot_index, dtype=torch.long, device=device), None


@BACKBONES.register_module()
class PromptCaption(BaseModule):
    def __init__(self,
                 clip_model,
                 prompt_config=dict(
                     context_length=77,
                 ),
                 load_ckpt_from=None,
                 self_name='',
                 ):
        super(PromptCaption, self).__init__()
        self.prompt_config = prompt_config
        # clip_imsize = clip_model.visual.input_resolution
        self.context_length = prompt_config.get('context_length', '77')
        self.token_emb = clip_model.token_embedding

    def forward(self, captions, device):
        caption_tokens = tokenize(captions, self.context_length, truncate=True).to(device)
        caption_embs = self.token_emb(caption_tokens)
        return caption_embs, caption_tokens.argmax(dim=-1), None

