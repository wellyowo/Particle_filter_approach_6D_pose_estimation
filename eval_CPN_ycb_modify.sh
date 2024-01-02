#!/bin/bash
python3 CenterFindNet/eval_CPN_modify.py --dataset ycb\
  --dataset_root_dir /home/uwe/Particle_filter_approach_6D_pose_estimation/dataset/ycb \
  --model_path /trained_model/test/1_1_50epoch/CPN_model_43_0.00020250714047467634.pth\
  --visualization True \
  --show_time 10