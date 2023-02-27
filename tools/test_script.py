import os
import time

os.system("CUDA_VISIBLE_DEVICES=0,1,2,3 "
            "sh dist_test.sh "
              # "../configs_my/RPN_CLIPPrompt_Region_KD_COCO_VAW.py "
              # '../configs_my/rpn_r50_fpn_mstrain_vg.py '
              # '../configs_my/rpn_r50_fpn_mstrain_vaw.py '
              # "../configs_my/rpn_r50_fpn_mstrain_coco.py "
              # "../configs_my/Op3_RPN_coco.py "
              # "../configs_my/Op3_RPN_cc3M.py "
              # f"../pretrain/faster_rcnn_x101_64x4d_fpn_mstrain_3x_coco_20210524_124528-26c63de6.pth "
              # "../configs_my/Op3_2_CLIPPrompt_Crop_Img_Prob_cc3M.py "
              # f"results/EXP20221006_0/epoch_20.pth "
              "../configs_my/Op1_CLIPPrompt_Crop_Img_VAW.py "
              # "../configs_my/Op2_RPN_CLIPPrompt_Region_KD_COCO_VAW.py "
            # "../configs_my/Op3_2_CLIPPrompt_Crop_Img_Prob_COCO.py "
            # "../configs_my/Op4_1_Train_Crop_Img_LSA.py "
            # "../configs_my/Op4_2_CLIPPrompt_Crop_Img_VAW.py "
            # "../configs_my/Op2_fasterrcnn_CLIPPrompt_Region_KD_COCO_VAW.py "
            f"results/EXP20221103_1/epoch_28.pth "
            "4"
    )

# for i in range(20, 150, 20):
#     os.system("CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "
#               "sh dist_test.sh "
#               # "../configs_my/Op2_RPN_CLIPPrompt_Region_KD_COCO_VAW.py "
#               # '../configs_my/rpn_r50_fpn_mstrain_vg.py '
#               # '../configs_my/rpn_r50_fpn_mstrain_vaw.py '
#               # "../configs_my/rpn_r50_fpn_mstrain_coco.py "
#               "../configs_my/Op1_CLIPPrompt_Crop_Img_VAW.py "
#               f"results/EXP20220903_0/epoch_{i}.pth "
#               "8"
#     )

