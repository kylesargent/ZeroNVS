#!/bin/bash
CKPT_PATH='zeronvs.ckpt'
CONFIG_PATH='zeronvs_config.yaml'

scene_uids=(8 21 30 31 34 38 40 41 45 55 63 82 103 110 114)
scales=(1.2 1.4 1.5 1.4 1.4 1.3 1.2 1.3 1.4 1.2 1.5 1.5 1.5 1.3 1.3)

for ((i=0; i<15; i++)); do
    
    scene_uid=${scene_uids[$i]}
    exp_root_dir=eval_outputs/DTU/$scene_uid
    scale=${scales[$i]}

    echo $exp_root_dir
    echo $scene_uid
    echo $scale

    python launch.py --config configs/zero123_scene.yaml --train --gpu 0 \
        system.guidance.cond_image_path="/tmp/input_image_dtu_guidance.png" \
        data.image_path="/tmp/input_image_dtu.png" \
        system.guidance.pretrained_model_name_or_path=$CKPT_PATH \
        system.guidance.pretrained_config=$CONFIG_PATH \
        data.view_synthesis.dataset='DTU' \
        data.view_synthesis.excluded_views='[3,4,5,6,7,16,17,18,19,20,21,36,37,38,39]' \
        exp_root_dir=$exp_root_dir \
        data.view_synthesis.scene_uid=$scene_uid \
        system.loss.lambda_opaque=0.1 \
        system.background.color='[0.0,0.0,0.0]' \
        system.background.random_aug=false \
        data.view_synthesis.manual_gt_to_pred_scale=$scale \
        data.random_camera.eval_width=400 \
        data.random_camera.eval_height=300 \
        data.width='[144,400]' \
        data.height='[108,300]'
done
