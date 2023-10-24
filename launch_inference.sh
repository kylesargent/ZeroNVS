#!/bin/bash
CKPT_PATH='zeronvs.ckpt'
CONFIG_PATH='zeronvs_config.yaml'

# Set these yourself!
IMAGE_PATH='motorcycle.png'
FOV=52.55
ELEVATION_DEG=31.0

# How close is the content to the camera (smaller is closer)
# See paper for details.
SCALE=0.7


python launch.py --config configs/zero123_scene.yaml --train --gpu 0 \
    system.guidance.cond_image_path=$IMAGE_PATH \
    data.image_path=$IMAGE_PATH \
    system.guidance.pretrained_model_name_or_path=$CKPT_PATH \
    system.guidance.pretrained_config=$CONFIG_PATH \
    data.view_synthesis=null \
    data.default_elevation_deg=$ELEVATION_DEG \
    data.default_fovy_deg=$FOV \
    data.random_camera.fovy_range="[$FOV,$FOV]" \
    data.random_camera.eval_fovy_deg=$FOV \
    system.loss.lambda_opaque=0.0 \
    system.background.color='[0.5,0.5,0.5]' \
    system.background.random_aug=true \
    system.background.random_aug_prob=1.0 \
    system.guidance.guidance_scale=9.5 \
    system.renderer.near_plane=0.5 \
    system.renderer.far_plane=1000.0 \
    system.guidance.precomputed_scale=$SCALE \
    system.guidance.use_anisotropic_schedule=true \
    system.guidance.anisotropic_offset=1000
    
    