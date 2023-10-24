#!/bin/bash
CKPT_PATH='zeronvs.ckpt'
CONFIG_PATH='zeronvs_config.yaml'

scene_uids=(bicycle bonsai counter garden kitchen room stump)
precomputed_scales=(0.9 0.9 0.9 0.9 0.9 2.0 0.9)
view_idxs=(2 0 20 0 0 0 11)
view_scales=(1.0 1.0 0.9 1.0 0.9 0.9 1.1)

for ((i=0; i<7; i++)); do
    
    scene_uid=${scene_uids[$i]}
    exp_root_dir=eval_outputs/mipnerf360/$scene_uid
    view_scale=${view_scales[$i]}
    view_idx=${view_idxs[$i]}
    precomputed_scale=${precomputed_scales[$i]}

    echo $scene_uid
    echo $exp_root_dir
    echo $view_scale
    echo $view_idx
    echo $precomputed_scale

    python launch.py --config configs/zero123_scene.yaml --train --gpu 0 \
        system.guidance.cond_image_path="/tmp/input_image_mipnerf360_guidance.png" \
        data.image_path="/tmp/input_image_mipnerf360.png" \
        system.guidance.pretrained_model_name_or_path=$CKPT_PATH \
        system.guidance.pretrained_config=$CONFIG_PATH \
        data.view_synthesis.dataset='mipnerf360' \
        exp_root_dir=$exp_root_dir \
        data.view_synthesis.scene_uid=$scene_uid \
        system.guidance.precomputed_scale=$precomputed_scale \
        data.view_synthesis.manual_gt_to_pred_scale=$view_scale \
        data.view_synthesis.input_view_idx=$view_idx \
        system.loss.lambda_opaque=0.0 \
        system.background.color='[0.5,0.5,0.5]' \
        system.background.random_aug=true \
        system.background.random_aug_prob=1.0 \
        system.guidance.guidance_scale=9.5 \
        system.renderer.near_plane=0.6 \
        system.renderer.far_plane=1000.0 \
        system.guidance.use_anisotropic_schedule=true \
        system.guidance.anisotropic_offset=1000
done
