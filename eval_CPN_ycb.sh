#!/bin/bash
python3 CenterFindNet/eval_CPN.py --dataset ycb\
  --dataset_root_dir /home/uwe/Particle_filter_approach_6D_pose_estimation/dataset/ycb \
  --model_path /trained_model/CPN_model_91_0.00023821471899932882.pth\
  --visualization True \
  --show_time 10
