#!/bin/bash
# python3 demo.py --dataset ycb\
#   --dataset_root_dir /home/welly/Particle_filter_approach_6D_pose_estimation/dataset/ycb/ \
#   --max_iteration 30 \
#   --visualization True\
#   --w_o_Scene_occlusion False \
#   --w_o_CPN False \
#   --save_path results/ycb_300_test/30/ \
#   --num_particles 300

# sleep 5

# python3 demo.py --dataset ycb\
#   --dataset_root_dir /home/welly/Particle_filter_approach_6D_pose_estimation/dataset/ycb/ \
#   --max_iteration 50 \
#   --visualization True\
#   --w_o_Scene_occlusion False \
#   --w_o_CPN False \
#   --save_path results/ycb_300_test/50/ \
#   --num_particles 300


# # sleep 5

# python3 demo.py --dataset ycb\
#   --dataset_root_dir /home/welly/Particle_filter_approach_6D_pose_estimation/dataset/ycb/ \
#   --max_iteration 30 \
#   --visualization False\
#   --w_o_Scene_occlusion False \
#   --w_o_CPN False \
#   --save_path results/ycb_600_test/30/ \
#   --num_particles 600


# sleep 5

# python3 demo.py --dataset ycb\
#   --dataset_root_dir /home/welly/Particle_filter_approach_6D_pose_estimation/dataset/ycb/ \
#   --max_iteration 30 \
#   --visualization False\
#   --w_o_Scene_occlusion False \
#   --w_o_CPN False \
#   --save_path results/ycb_300_para/30/ \
#   --num_particles 300

  python3 demo.py --dataset ycb\
  --dataset_root_dir /home/welly/Particle_filter_approach_6D_pose_estimation/dataset/ycb/ \
  --max_iteration 50 \
  --visualization False\
  --w_o_Scene_occlusion False \
  --w_o_CPN False \
  --save_path results/ycb_300_para/50/ \
  --num_particles 300