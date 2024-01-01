#!/bin/bash
python3 demo.py --dataset lmo\
  --dataset_root_dir /home/welly/Particle_filter_approach_6D_pose_estimation/dataset/LMO/lmo_test_all/test/000002\
  --max_iteration 30 \
  --visualization True\
  --w_o_Scene_occlusion False \
  --w_o_CPN False \
  --save_path results/lmo/ \
  --num_particles 300

