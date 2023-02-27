import os
import time

while True:
    # os.system("CUDA_VISIBLE_DEVICES=0,1,2,3 "
    #           "sh dist_train.sh "
    #           # "../configs_my/Op3_3_Boost_CLIPPrompt_Crop_Img_COCO_VAW.py "
    #           "../configs_my/Op2_2_RPN_CLIPPrompt_Region_LSA.py "
    #           "results/EXP20230102_0 "
    #           "4")

    # # Class agnostic RPN MStrain
    # os.system("CUDA_VISIBLE_DEVICES=0,1,2,3 "
    #           "sh dist_train.sh "
    #           "../configs_my/OvarNet/Op1_2_class_agnostic_rpn_mstrain.py "
    #           "results/EXP20230207_0 "
    #           "4")

    # OvarNetP R50 train with KD
    os.system("CUDA_VISIBLE_DEVICES=0,1,2,3 "
              "sh dist_train.sh "
              "../configs_my/OvarNet/Op2_1_OvarNetP_w_KD.py "
              "results/EXP20230210_1 "
              "4")


    # os.system("CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "
    #           "sh dist_train.sh "
    #           "../configs_my/Op1_CLIPPrompt_Crop_Img_VAW.py "
    #           "results/EXP20221218_0 "
    #           "8")
    # os.system("CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "
    #           "sh dist_train.sh "
    #           "../configs_my/Op2_RPN_CLIPPrompt_Region_KD_COCO_VAW.py "
    #           "results/EXP20221107_0 "
    #           "8")
    # os.system("CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "
    #           "sh dist_train.sh "
    #           "../configs_my/Op2_fasterrcnn_CLIPPrompt_Region_KD_COCO_VAW.py "
    #           "results/EXP20221219_0 "
    #           "8")
    # os.system("CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "
    #           "sh dist_train.sh "
    #           "../configs_my/Op1_CLIPPrompt_Crop_Img_VAW.py "
    #           "results/EXP20221030_0 "
    #           "8")
    # os.system("CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "
    #           "sh dist_train.sh "
    #           "../configs_my/CLIPPrompt_Crop_Img_VAW3.py "
    #           "results/EXP20220903_3 "
    #           "8")

    # os.system("CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "
    #           "sh dist_train.sh "
    #           "../configs_my/CLIPPrompt_Crop_Img_VAW2.py "
    #           "results/EXP20220901_5 "
    #           "8")

    # os.system("CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "
    #           "sh dist_train.sh "
    #           "../configs_my/CLIPPrompt_Crop_Img_VAW1.py "
    #           # "../configs_my/CLIPPrompt_Region_KD_VAW.py "
    #           # "../configs_my/CLIPPrompt_Region_VAW.py "
    #           # "../configs_my/MAEPrompt_Crop_Img_VAW.py "
    #           # "../configs_my/Op1_2_class_agnostic_rpn_mstrain.py "
    #           # "../configs_my/rpn_r50_fpn_mstrain_coco.py "
    #           # "../configs_my/rpn_r50_fpn_mstrain_vaw.py "
    #           # "../configs_my/rpn_r50_fpn_mstrain_vg.py "
    #           # "../configs_my/CLIPPrompt_Region_FasterRcnn_KD_VAW.py "
    #           # "../configs_my/Op2_RPN_CLIPPrompt_Region_KD_COCO_VAW.py "
    #           "results/EXP20220828_5 "
    #           "8")
    #
    # os.system("CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "
    #           "sh dist_train.sh "
    #           "../configs_my/CLIPPrompt_Crop_Img_VAW2.py "
    #           "results/EXP20220828_6 "
    #           "8")
    # # os.system("CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "
    # #           "sh dist_train.sh "
    # #           "../configs_my/CLIPPrompt_Crop_Img_VAW3.py "
    # #           "results/EXP20220828_3 "
    # #           "8")
    time.sleep(60*2)

