from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class SetSubModelEvalHook(Hook):
    def __init__(self, train_module=[]):
        self.train_module = train_module

    def _eval_model(self, runner):
        """To resume model with it's ema parameters more friendly.

        Register ema parameter as ``named_buffer`` to model.
        """
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        for name, module in model.named_children():
            flag = [True for x in self.train_module if x in name]
            if len(flag):
                module.train()
            else:
                module.eval()

    def before_train_epoch(self, runner):
        self._eval_model()
