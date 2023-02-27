import os
import time

for i_epoch in range(10, 60, 10):
    os.system("CUDA_VISIBLE_DEVICES=0,1,2,3 "
              "sh dist_test.sh "
              # f"../configs_my/Op1_CLIPPrompt_Crop_Img_VAW.py "
              # f"../configs_my/Op2_RPN_CLIPPrompt_Region_KD_COCO_VAW.py "
              # f"results/EXP20221024_2/epoch_{i_epoch}.pth "
              # "../configs_my/Op2_fasterrcnn_CLIPPrompt_Region_KD_COCO_VAW.py "
              # "../configs_my/Op2_2_RPN_CLIPPrompt_Region_LSA.py "
              "../configs_my/OvarNet/Op2_1_OvarNetP_w_KD.py "
              f"results/EXP20230210_1/epoch_{i_epoch}.pth "
              "4"
              )

