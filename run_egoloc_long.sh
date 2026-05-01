#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python egoloc3d_long.py \
  --video_path ManiTIL/cabinet/video/video1.mp4 \
  --speed_json ManiTIL/cabinet/speed/video1_speed.json \
  --video_type long
