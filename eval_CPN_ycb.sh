#!/bin/bash
python CenterFindNet/eval_CPN.py --dataset ycb\
  --dataset_root_dir /home/welly/Particle_filter_approach_6D_pose_estimation/dataset/ycb/data1 \
  --model_path /trained_model/CPN_model_91_0.00023821471899932882.pth\
  --visualization True
