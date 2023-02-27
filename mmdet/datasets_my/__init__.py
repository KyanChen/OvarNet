from .vaw_datasset import VAWDataset
from .nwpu_detection_dataset import NWPUDataset
from .piplines_my import *
from .coco_clip import CocoCLIPDataset
from .coco_clip_annotated import CocoCLIPAnnDataset
from .vaw_rpn_datasset import VAWRPNDataset
from .vaw_proposal_datasset import VAWProposalDataset
from .vaw_region_att_pred_datasset import VAWRegionDataset


from .vg_rpn_datasset import VGRPNDataset
from .class_agnostic_rpn_infer_datasets import RPNOnCocoCapDataset, RPNOnCC3MDataset
from .ovarnet_attribute_dataset import OvarNetAttributeDataset

from .boost_clip_with_cap_crop_img_datasset import BoostCLIPWithCapCropDataset
from .lsa_crop_img_datasset import LSACropDataset
from .lsa_rpn_attribute_datasset import LSARPNAttributeDataset

from .eval_results import InferenceRPNAttributeDataset


from .class_agnostic_rpn_detection_dataset import ClsAgnosticRPNDataset
from .clipattr_img_crop_dataset import CLIPAttrImgCropDataset
__all__ = [
    'VAWDataset', 'NWPUDataset', 'CocoCLIPDataset',
    'CocoCLIPAnnDataset', 'VAWRPNDataset', 'VAWProposalDataset',
    'VAWRegionDataset', 'CLIPAttrImgCropDataset', 'RPNOnCocoCapDataset',
    'VGRPNDataset', 'OvarNetAttributeDataset', 'BoostCLIPWithCapCropDataset',
    'RPNOnCC3MDataset', 'LSACropDataset', 'LSARPNAttributeDataset',
    'InferenceRPNAttributeDataset', 'ClsAgnosticRPNDataset'
]