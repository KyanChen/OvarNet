# Copyright (c) OpenMMLab. All rights reserved.
import collections

from mmcv.utils import build_from_cfg

from ..builder import PIPELINES


@PIPELINES.register_module()
class Compose:
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict | callable]): Sequence of transform object or
            config dict to be composed.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

    def __call__(self, data, idx=None):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """
        if idx is None:
            transforms = self.transforms
        elif isinstance(idx, int):
            transforms = self.transforms[idx:idx+1]
        elif isinstance(idx, tuple) or isinstance(idx, list):
            assert len(idx) == 2
            if idx[0] == ':':
                transforms = self.transforms[: idx[1]]
            elif idx[1] == ':':
                transforms = self.transforms[idx[0]:]
            else:
                transforms = self.transforms[idx[0]: idx[1]]
        else:
            raise NotImplementedError

        for t in transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            str_ = t.__repr__()
            if 'Compose(' in str_:
                str_ = str_.replace('\n', '\n    ')
            format_string += '\n'
            format_string += f'    {str_}'
        format_string += '\n)'
        return format_string
