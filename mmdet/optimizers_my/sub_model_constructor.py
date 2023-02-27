import torch.nn
from mmcv.utils import _BatchNorm, _InstanceNorm, build_from_cfg, is_list_of
from mmcv.runner.optimizer import OPTIMIZER_BUILDERS, OPTIMIZERS
from mmcv.runner.optimizer import DefaultOptimizerConstructor
from mmcv.runner import get_dist_info


@OPTIMIZER_BUILDERS.register_module()
class SubModelConstructor(DefaultOptimizerConstructor):
    def __call__(self, model):
        if hasattr(model, 'module'):
            model = model.module
        # import pdb
        # pdb.set_trace()

        optimizer_cfg = self.optimizer_cfg.copy()
        sub_models = optimizer_cfg.pop('sub_model', None)

        if isinstance(sub_models, str):
            sub_models = {sub_models: {}}
        if isinstance(sub_models, list):
            sub_models = {x: {} for x in sub_models}

        # set training parameters and lr
        for sub_model_name, value in sub_models.items():
            if hasattr(model, sub_model_name):
                sub_model_ = getattr(model, sub_model_name)
                if isinstance(sub_model_, torch.nn.Parameter):
                    # filter(lambda p: p.requires_grad, model.parameters())
                    sub_models[sub_model_name]['params'] = filter(lambda p: p.requires_grad, [sub_model_])
                else:
                    sub_models[sub_model_name]['params'] = filter(lambda p: p.requires_grad, sub_model_.parameters())

                lr_mult = value.pop('lr_mult', 1.)
                sub_models[sub_model_name]['lr'] = self.base_lr * lr_mult
                if self.base_wd is not None:
                    decay_mult = value.pop('decay_mult', 1.)
                    sub_models[sub_model_name]['weight_decay'] = self.base_wd * decay_mult
            else:
                raise ModuleNotFoundError(f'{sub_model_name} not in model')

        _rank, _word_size = get_dist_info()
        if _rank == 0:
            print()
            print('All sub models:')
            for name, module in model.named_children():
                print(name, end=', ')
            print()
            print('Needed train models:')
            for needed_train_sub_model in sub_models.keys():
                print(needed_train_sub_model, end=', ')
            print()

        # if no paramwise option is specified, just use the global setting
        if not self.paramwise_cfg:
            optimizer_cfg['params'] = [value for key, value in sub_models.items()]
            return build_from_cfg(optimizer_cfg, OPTIMIZERS)
        # if not self.paramwise_cfg:
        #     optimizer_cfg['params'] = model.parameters()
        #     return build_from_cfg(optimizer_cfg, OPTIMIZERS)

        # set param-wise lr and weight decay recursively
        params = []
        self.add_params(params, model)
        optimizer_cfg['params'] = params

        return build_from_cfg(optimizer_cfg, OPTIMIZERS)
