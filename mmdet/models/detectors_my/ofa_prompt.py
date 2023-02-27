import json
import math
import pickle

import torch

from ..builder import DETECTORS, build_backbone, build_head, build_neck
from ..detectors.base import BaseDetector
import warnings
from mmcv import ConfigDict
from utils_my import Trie
import utils_my.data_utils as data_utils
from fairseq import search
from fairseq.data.encoders import build_bpe
from omegaconf import DictConfig

@DETECTORS.register_module()
class OFA_Prompter(BaseDetector):
    def __init__(self,
                 classname_path,
                 backbone,
                 prompt_learner,
                 model_config=dict(
                     bpe_dir='../utils_my/BPE',
                     valid_batch_size=20,
                     max_src_length=32,
                     max_tgt_length=8
                 ),
                 n_sample_attr=5,
                 prompt_learner_weights='',
                 ofa_pretrained_weights='',
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(OFA_Prompter, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.cfg = self.load_cfg()
        # self.cfg.update(model_config)
        self.ans2label_dict = self.cfg.ans2label_dict
        if ofa_pretrained_weights:
            state_dict = torch.load(ofa_pretrained_weights, map_location="cpu")
            self.cfg.update(state_dict['cfg']['generation'])
            self.cfg.update(vars(state_dict['cfg']['model']))
            self.cfg.update(state_dict['cfg']['task'])
            # self.model.load_state_dict(state_dict['model'])
        self.cfg.update(model_config)
        self.setup_task(self.cfg)

        classname_maps = json.load(open(classname_path))
        classnames = list(classname_maps.keys())
        self.n_classnames = len(classnames)
        self.n_sample_attr = n_sample_attr

        self.task = ConfigDict(
            dict(
                src_dict=self.src_dict,
                tgt_dict=self.tgt_dict
            )
        )
        backbone.update(model_cfg=self.cfg, task=self.task)
        self.model = build_backbone(backbone).model
        if ofa_pretrained_weights:
            self.model.load_state_dict(state_dict['model'])

        bpe_dict = {
            "_name": "gpt2",
            "gpt2_encoder_json": self.cfg.bpe_dir+"/encoder.json",
            "gpt2_vocab_bpe": self.cfg.bpe_dir+"/vocab.bpe"
        }
        bpe_dict = DictConfig(bpe_dict)
        self.bpe = build_bpe(bpe_dict)

        tgt_list = []
        prev_output_list = []
        self.index2ans = {}
        self.ans2index = {}
        self.constraint_trie = Trie(self.tgt_dict.eos())
        for i, answer in enumerate(self.ans2label_dict.keys()):
            answer_item = self.tgt_dict.encode_line(
                line=self.bpe.encode(' ' + answer),
                add_if_not_exist=False,
                append_eos=False
            ).long()
            tgt_list += [torch.cat([answer_item, torch.LongTensor([self.tgt_dict.eos()])])]
            prev_output_list += [torch.cat([torch.LongTensor([self.tgt_dict.bos()]), answer_item])]
            self.index2ans[i] = answer
            self.ans2index[answer] = i
            self.constraint_trie.insert([self.tgt_dict.bos()] + answer_item.tolist() + [self.tgt_dict.eos()])

        constraint_mask_list = []
        for prev_output_item in prev_output_list:
            constraint_mask = torch.zeros((len(prev_output_item), len(self.tgt_dict))).bool()
            for i in range(len(prev_output_item)):
                constraint_prefix_token = prev_output_item[:i + 1].tolist()
                constraint_nodes = self.constraint_trie.get_next_layer(constraint_prefix_token)
                constraint_mask[i][constraint_nodes] = True
            constraint_mask_list.append(constraint_mask)

        eos = self.src_dict.eos()
        pad = self.src_dict.pad()
        self.valid_tgt_list = []
        self.valid_prev_output_list = []
        self.valid_constraint_masks_list = []
        for i in range(0, len(tgt_list), self.cfg.valid_batch_size):
            tgt_item = tgt_list[i:i + self.cfg.valid_batch_size]
            prev_output_item = prev_output_list[i:i + self.cfg.valid_batch_size]
            constrain_mask = constraint_mask_list[i:i + self.cfg.valid_batch_size]
            self.valid_tgt_list.append(
                data_utils.collate_tokens(tgt_item, pad_idx=pad, eos_idx=eos, left_pad=False)
            )
            self.valid_prev_output_list.append(
                data_utils.collate_tokens(prev_output_item, pad_idx=pad, eos_idx=eos, left_pad=False)
            )
            self.valid_constraint_masks_list.append(
                data_utils.collate_tokens(constrain_mask, pad_idx=pad, left_pad=False)
            )

        self.task.update(max_src_length=self.cfg.max_src_length, bpe=self.bpe)
        prompt_learner.update(dict(classnames=classnames, model=self.model, task=self.task))
        self.prompt_learner = build_backbone(prompt_learner)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        if prompt_learner_weights:
            state_dict = torch.load(prompt_learner_weights, map_location="cpu")
            self.prompt_learner.load_state_dict(state_dict)

        if neck is not None:
            self.neck = build_neck(neck)

        self.generator = self.build_generator(self.model, self.cfg)

        self.idx2tokens = torch.cat(self.idx2tokens())
        self.bos_token_tensor = torch.LongTensor([self.src_dict.bos()])
        self.eos_token_tensor = torch.LongTensor([self.src_dict.eos()])

        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        bbox_head.update(task=self.task)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        print('max_src_length: ', self.cfg.max_src_length)
        print('max_tgt_length: ', self.cfg.max_tgt_length)
        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

    def load_cfg(self):
        cfg = ConfigDict(dict(
            ans2label_dict={"no": 0, "yes": 1},
            ans2label_file=None,
            valid_batch_size=20,
            uses_ema=False,
            data=None,
            selected_cols=None,
            bpe_dir=None,
            max_source_positions=1024,
            max_target_positions=1024,
            max_src_length=80,
            max_tgt_length=30,
            code_dict_size=8192,
            patch_image_size=256,
            num_bins=1000,
            imagenet_default_mean_and_std=False,
            constraint_range=None,
        ))
        return cfg

    def setup_task(self, cfg, **kwargs):
        from fairseq.data import Dictionary
        src_dict = Dictionary.load(cfg.bpe_dir + "/dict.txt")
        tgt_dict = Dictionary.load(cfg.bpe_dir + "/dict.txt")
        src_dict.add_symbol("<mask>")
        tgt_dict.add_symbol("<mask>")
        for i in range(cfg.code_dict_size):  # default, code_dict_size=8192
            src_dict.add_symbol("<code_{}>".format(i))
            tgt_dict.add_symbol("<code_{}>".format(i))
        # quantization
        for i in range(cfg.num_bins):  # default, num_bins=1000
            src_dict.add_symbol("<bin_{}>".format(i))
            tgt_dict.add_symbol("<bin_{}>".format(i))
        print("source dictionary: {} types".format(len(src_dict)))
        print("target dictionary: {} types".format(len(tgt_dict)))

        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    def idx2tokens(self):
        idx2tokens = []
        for idx, ans in self.index2ans.items():
            idx2tokens.append(
                self.prompt_learner.encode_text(
                ' {}'.format(ans),
                pad=False,
                append_bos=False,
                append_eos=False
                )
            )
        return idx2tokens

    def build_generator(self,
                        models,
                        args,
                        seq_gen_cls=None,
                        extra_gen_cls_kwargs=None,
                        prefix_allowed_tokens_fn=None
                        ):
        if getattr(args, "score_reference", False):
            from fairseq.sequence_scorer import SequenceScorer

            return SequenceScorer(
                self.target_dictionary,
                compute_alignment=getattr(args, "print_alignment", False),
            )

        from fairseq.sequence_generator import (
            # SequenceGenerator,
            SequenceGeneratorWithAlignment,
        )
        from utils_my.sequence_generator import SequenceGenerator

        # Choose search strategy. Defaults to Beam Search.
        sampling = getattr(args, "sampling", False)
        sampling_topk = getattr(args, "sampling_topk", -1)
        sampling_topp = getattr(args, "sampling_topp", -1.0)
        diverse_beam_groups = getattr(args, "diverse_beam_groups", -1)
        diverse_beam_strength = getattr(args, "diverse_beam_strength", 0.5)
        match_source_len = getattr(args, "match_source_len", False)
        diversity_rate = getattr(args, "diversity_rate", -1)
        constrained = getattr(args, "constraints", False)
        if prefix_allowed_tokens_fn is None:
            prefix_allowed_tokens_fn = getattr(args, "prefix_allowed_tokens_fn", None)
        if (
            sum(
                int(cond)
                for cond in [
                    sampling,
                    diverse_beam_groups > 0,
                    match_source_len,
                    diversity_rate > 0,
                ]
            )
            > 1
        ):
            raise ValueError("Provided Search parameters are mutually exclusive.")
        assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
        assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

        if sampling:
            search_strategy = search.Sampling(
                self.target_dictionary, sampling_topk, sampling_topp
            )
        elif diverse_beam_groups > 0:
            search_strategy = search.DiverseBeamSearch(
                self.target_dictionary, diverse_beam_groups, diverse_beam_strength
            )
        elif match_source_len:
            # this is useful for tagging applications where the output
            # length should match the input length, so we hardcode the
            # length constraints for simplicity
            search_strategy = search.LengthConstrainedBeamSearch(
                self.target_dictionary,
                min_len_a=1,
                min_len_b=0,
                max_len_a=1,
                max_len_b=0,
            )
        elif diversity_rate > -1:
            search_strategy = search.DiverseSiblingsSearch(
                self.target_dictionary, diversity_rate
            )
        elif constrained:
            search_strategy = search.LexicallyConstrainedBeamSearch(
                self.target_dictionary, args.constraints
            )
        elif prefix_allowed_tokens_fn:
            search_strategy = search.PrefixConstrainedBeamSearch(
                self.target_dictionary, prefix_allowed_tokens_fn
            )
        else:
            search_strategy = search.BeamSearch(self.target_dictionary)

        extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
        if seq_gen_cls is None:
            if getattr(args, "print_alignment", False):
                seq_gen_cls = SequenceGeneratorWithAlignment
                extra_gen_cls_kwargs["print_alignment"] = args.print_alignment
            else:
                seq_gen_cls = SequenceGenerator

        seq_generator = seq_gen_cls(
            models,
            self.target_dictionary,
            beam_size=getattr(args, "beam", 5),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search_strategy,
            constraint_range=self.cfg.constraint_range,
            **extra_gen_cls_kwargs,
        )

        seq_generator.constraint_trie = self.constraint_trie

        return seq_generator

    def extract_feat(self, img):
        return img

    def train(self, mode=True):
        for name, module in self.named_children():
            if 'prompt_learner' in name:
                module.train(mode)
            else:
                module.eval()

    def forward(self, img, img_metas, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.cfg.max_source_positions, self.cfg.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    def train_step(self, data, optimizer):
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def val_step(self, data, optimizer=None):

        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)

        if self.uses_ema:
            assert 'ema_model' in extra_kwargs and extra_kwargs['ema_model'] is not None
        if self.uses_ema:
            eval_model = extra_kwargs['ema_model']
        else:
            eval_model = model

        eval_model.eval()
        with torch.no_grad():
            batch_size = sample["net_input"]["src_tokens"].size(0)
            encoder_out = eval_model.encoder(
                sample["net_input"]["src_tokens"],
                src_lengths=sample["net_input"]["src_lengths"],
                patch_images=sample["net_input"]["patch_images"],
                patch_masks=sample["net_input"]["patch_masks"]
            )
            device = sample["net_input"]["src_tokens"].device
            valid_result = []
            for valid_tgt, valid_prev_output, valid_constraint_masks in zip(self.valid_tgt_list,
                                                                            self.valid_prev_output_list,
                                                                            self.valid_constraint_masks_list):
                valid_tgt_size = valid_tgt.size(0)
                valid_tgt = valid_tgt.repeat(batch_size, 1).to(device)
                valid_prev_output = valid_prev_output.repeat(batch_size, 1).to(device)
                valid_constraint_masks = valid_constraint_masks.repeat(batch_size, 1, 1).to(device)
                new_encoder_out = {}
                new_encoder_out["encoder_out"] = [
                    encoder_out["encoder_out"][0].repeat_interleave(valid_tgt_size, dim=1)
                ]
                new_encoder_out["encoder_padding_mask"] = [
                    encoder_out["encoder_padding_mask"][0].repeat_interleave(valid_tgt_size, dim=0)
                ]
                new_encoder_out["position_embeddings"] = [
                    encoder_out["position_embeddings"][0].repeat_interleave(valid_tgt_size, dim=0)
                ]

                decoder_out = eval_model.decoder(valid_prev_output, encoder_out=new_encoder_out)
                decoder_out[0].masked_fill_(~valid_constraint_masks, -math.inf)
                lprobs = eval_model.get_normalized_probs(decoder_out, log_probs=True)
                scores = lprobs.gather(dim=-1, index=valid_tgt.unsqueeze(-1)).squeeze(-1)
                scores = scores.masked_fill(valid_tgt.eq(self.tgt_dict.pad()), 0)
                scores = scores.sum(1)
                scores = scores.view(-1, valid_tgt_size)
                valid_result.append(scores)

        valid_result = torch.cat(valid_result, dim=-1)
        predicts = valid_result.argmax(1).tolist()
        hyps = [self.index2ans[predict_index] for predict_index in predicts]
        scores = [ref_dict.get(hyp, 0) for ref_dict, hyp in zip(sample['ref_dict'], hyps)]
        logging_output["_score_sum"] = sum(scores)
        logging_output["_score_cnt"] = len(scores)

        return loss, sample_size, logging_output


        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        def sum_logs(key):
            import torch
            result = sum(log.get(key, 0) for log in logging_outputs)
            if torch.is_tensor(result):
                result = result.cpu()
            return result

        def compute_score(meters):
            score = meters["_score_sum"].sum / meters["_score_cnt"].sum
            score = score if isinstance(score, float) else score.item()
            return round(score, 3)

        if sum_logs("_score_cnt") > 0:
            metrics.log_scalar("_score_sum", sum_logs("_score_sum"))
            metrics.log_scalar("_score_cnt", sum_logs("_score_cnt"))
            metrics.log_derived("score", compute_score)

    def forward_train(self, img, img_metas, gt_labels, gt_bboxes_ignore=None):

        prompts_embeddings = self.prompt_learner().to(img.device)
        self.tokenized_prompts = self.tokenized_prompts.to(img.device)
        input_patch_images = []
        input_tokenized_prompts = []
        input_prompts_embeddings = []
        input_prev_output_item = []
        input_target_item = []
        input_constraint_mask = []
        for i_img in range(len(img)):
            gt_mask = gt_labels[i_img] < 2
            gt_mask_idx = torch.nonzero(gt_mask).squeeze(-1)
            if len(gt_mask_idx) > 0:
                sample_id = torch.randint(0, len(gt_mask_idx), size=[self.n_sample_attr])
                sample_id = gt_mask_idx[sample_id]
                labels = gt_labels[i_img][sample_id]

                input_patch_images.append(img[i_img].unsqueeze(0).expand(self.n_sample_attr, -1, -1, -1))
                input_tokenized_prompts.append(self.tokenized_prompts[sample_id])
                input_prompts_embeddings.append(prompts_embeddings[sample_id])
                label_tokens = torch.index_select(self.idx2tokens.to(labels.device), 0, labels).view(-1, 1)
                prev_output_item = torch.cat([
                    self.bos_token_tensor.expand(self.n_sample_attr, 1).to(labels.device), label_tokens],
                    dim=1
                )
                target_item = torch.cat([
                    label_tokens, self.eos_token_tensor.expand(self.n_sample_attr, 1).to(labels.device)],
                    dim=1
                )

                if self.constraint_trie is not None:
                    input_constraint_mask.append(
                        torch.index_select(self.valid_constraint_masks_list[0].to(labels.device), 0, labels)
                    )
                # constraint_mask = torch.zeros((len(prev_output_item), len(self.tgt_dict))).bool()
                # for i in range(len(prev_output_item)):
                #     constraint_prefix_token = prev_output_item[:i + 1].tolist()
                #     constraint_nodes = self.constraint_trie.get_next_layer(constraint_prefix_token)
                #     constraint_mask[i][constraint_nodes] = True

                input_prev_output_item.append(prev_output_item)
                input_target_item.append(target_item)

        input_patch_images = torch.cat(input_patch_images)
        input_tokenized_prompts = torch.cat(input_tokenized_prompts)
        input_prompts_embeddings = torch.cat(input_prompts_embeddings)
        input_prev_output_item = torch.cat(input_prev_output_item)
        input_target_item = torch.cat(input_target_item)
        input_constraint_mask = torch.cat(input_constraint_mask)

        patch_masks = torch.tensor(len(input_patch_images) * [True]).to(img.device)
        src_lengths = torch.sum(input_tokenized_prompts.ne(self.tgt_dict.pad()).long(), dim=1)  # N

        sample = {
            'net_input': dict(
                src_tokens=input_tokenized_prompts,
                src_lengths=src_lengths,
                prev_output_tokens=input_prev_output_item,
                patch_images=input_patch_images.contiguous(),
                patch_images_2=None,
                patch_masks=patch_masks,
                code_masks=None,
                sample_patch_num=None,
                features_only=False,
                classification_head_name=None,
                token_embeddings=input_prompts_embeddings,
                return_all_hiddens=False,
                alignment_layer=None,
                alignment_heads=None
            ),
            'constraint_masks': input_constraint_mask,
            'target': input_target_item,
            "conf": None,
        }

        # if self.constraint_trie is not None:
        #     # constraint_mask = torch.zeros((len(prev_output_item), len(self.tgt_dict))).bool()
        #     # for i in range(len(prev_output_item)):
        #     #     constraint_prefix_token = prev_output_item[:i + 1].tolist()
        #     #     constraint_nodes = self.constraint_trie.get_next_layer(constraint_prefix_token)
        #     #     constraint_mask[i][constraint_nodes] = True
        #     sample["constraint_mask"] = torch.cat(self.valid_constraint_masks_list)

        # self.model = self.model.cpu()
        # for k, v in sample.items():
        #     try:
        #         sample[k] = v.cpu()
        #     except:
        #         if k == 'net_input':
        #             for k, v in sample['net_input'].items():
        #                 try:
        #                     sample['net_input'][k] = v.cpu()
        #                 except:
        #                     pass
        losses = self.bbox_head.forward_train(self.model, sample, update_num=0)

        return losses

    def forward_test(self, imgs, img_metas, **kwargs):
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            assert imgs[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{imgs[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(imgs, img_metas, **kwargs)

    def simple_test(self, img, img_metas, rescale=False):
        batch_size = len(img)
        prompts_embeddings = self.prompt_learner()

        input_patch_images = []
        input_tokenized_prompts = []
        input_prompts_embeddings = []
        # input_prev_output_item = []
        # input_target_item = []
        # input_constraint_mask = []
        for i_img in range(len(img)):
            input_patch_images.append(img[i_img].unsqueeze(0).expand(self.n_classnames, -1, -1, -1))
            input_tokenized_prompts.append(self.tokenized_prompts.to(img.device))
            input_prompts_embeddings.append(prompts_embeddings.to(img.device))
                # if self.constraint_trie is not None:
                #     input_constraint_mask.append(torch.stack([self.valid_constraint_masks_list[x] for x in labels]))
                # # constraint_mask = torch.zeros((len(prev_output_item), len(self.tgt_dict))).bool()
                # # for i in range(len(prev_output_item)):
                # #     constraint_prefix_token = prev_output_item[:i + 1].tolist()
                # #     constraint_nodes = self.constraint_trie.get_next_layer(constraint_prefix_token)
                # #     constraint_mask[i][constraint_nodes] = True
                #
                # input_prev_output_item.append(prev_output_item)
                # input_target_itemem.append(target_item)

        input_patch_images = torch.cat(input_patch_images)
        input_tokenized_prompts = torch.cat(input_tokenized_prompts)
        input_prompts_embeddings = torch.cat(input_prompts_embeddings)
        # input_prev_output_item = torch.cat(input_prev_output_item)
        # input_target_item = torch.cat(input_target_item)
        # input_constraint_mask = torch.cat(input_constraint_mask)

        patch_masks = torch.tensor(len(input_patch_images) * [True]).to(img.device)
        src_lengths = torch.sum(input_tokenized_prompts.ne(self.tgt_dict.pad()).long(), dim=1)  # N

        sample = {
            'net_input': dict(
                src_tokens=input_tokenized_prompts,
                src_lengths=src_lengths,
                prev_output_tokens=None,
                patch_images=input_patch_images,
                patch_images_2=None,
                patch_masks=patch_masks,
                code_masks=None,
                sample_patch_num=None,
                features_only=False,
                classification_head_name=None,
                token_embeddings=input_prompts_embeddings,
                return_all_hiddens=False,
                alignment_layer=None,
                alignment_heads=None
            ),
            'constraint_masks': None,
            'target': None,
            "conf": None,
        }

        # self.model = self.model.cpu()
        # for k, v in sample.items():
        #     try:
        #         sample[k] = v.cpu()
        #     except:
        #         if k == 'net_input':
        #             for k, v in sample['net_input'].items():
        #                 try:
        #                     sample['net_input'][k] = v.cpu()
        #                 except:
        #                     pass

        encoder_out = self.model.encoder(
            sample["net_input"]["src_tokens"],
            src_lengths=sample["net_input"]["src_lengths"],
            patch_images=sample["net_input"]["patch_images"],
            patch_masks=sample["net_input"]["patch_masks"],
            token_embeddings=sample["net_input"]["token_embeddings"],
        )
        device = sample["net_input"]["src_tokens"].device
        valid_result = []
        # self.valid_tgt_list [tensor([[ 117,    2], [4420,    2]])]  [2x2]
        # self.valid_prev_output_list [tensor([[   0,  117], [   0, 4420]])]  [2x2]
        # self.valid_constraint_masks_list [2,2,59457]
        for valid_tgt, valid_prev_output, valid_constraint_masks in zip(self.valid_tgt_list,
                                                                        self.valid_prev_output_list,
                                                                        self.valid_constraint_masks_list):
            valid_tgt_size = valid_tgt.size(0)
            valid_tgt = valid_tgt.repeat(batch_size*self.n_classnames, 1).to(device)
            valid_prev_output = valid_prev_output.repeat(batch_size*self.n_classnames, 1).to(device)
            valid_constraint_masks = valid_constraint_masks.repeat(batch_size*self.n_classnames, 1, 1).to(device)
            new_encoder_out = {}
            new_encoder_out["encoder_out"] = [
                encoder_out["encoder_out"][0].repeat_interleave(valid_tgt_size, dim=1)
            ]
            new_encoder_out["encoder_padding_mask"] = [
                encoder_out["encoder_padding_mask"][0].repeat_interleave(valid_tgt_size, dim=0)
            ]
            new_encoder_out["position_embeddings"] = [
                encoder_out["position_embeddings"][0].repeat_interleave(valid_tgt_size, dim=0)
            ]

            decoder_out = self.model.decoder(valid_prev_output, encoder_out=new_encoder_out)
            decoder_out[0].masked_fill_(~valid_constraint_masks, -math.inf)
            lprobs = self.model.get_normalized_probs(decoder_out, log_probs=True)
            scores = lprobs.gather(dim=-1, index=valid_tgt.unsqueeze(-1)).squeeze(-1)
            scores = scores.masked_fill(valid_tgt.eq(self.tgt_dict.pad()), 0)
            scores = scores.sum(1)
            scores = scores.view(-1, valid_tgt_size)
            valid_result.append(scores)
        # import pdb
        # pdb.set_trace()
        valid_result = torch.cat(valid_result, dim=-1)
        # predicts = valid_result.argmax(1).tolist()
        # hyps = [self.index2ans[predict_index] for predict_index in predicts]
        # scores = [ref_dict.get(hyp, 0) for ref_dict, hyp in zip(sample['ref_dict'], hyps)]

        # pred = list(scores.detach().cpu().numpy())
        pred_score = torch.softmax(valid_result, dim=-1)[..., 1].view(-1, self.n_classnames).cpu().numpy()
        pred_score = list(pred_score)
        return pred_score


    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        pass

