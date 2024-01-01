#!/bin/bash
echo "Running  Partical Filter to get object pose"
python3 testing_dataset_check.py --dataset ycb\
  --dataset_root_dir /home/welly/Particle_filter_approach_6D_pose_estimation/dataset/ycb/ \
  --max_iteration 30 \
  --visualization False\
  --w_o_Scene_occlusion False \
  --w_o_CPN False \
  --save_path results/ycb_test/ \
  --num_particles 300 \
  --nth_photo "$1"

echo "Evaluate Particle filter result"

python3 eval_test.py --dataset ycb\
  --dataset_root_dir /home/welly/Particle_filter_approach_6D_pose_estimation/dataset/ycb\
  --visualization True \
  --save_path results/ycb_test/ \
  --nth_photo "$1"

