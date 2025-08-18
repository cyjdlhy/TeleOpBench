# cd teleop
# CUDA_VISIBLE_DEVICES=0 python -m debugpy \
#        --listen 0.0.0.0:4689 \
#        --wait-for-client \
#         teleop_test_gym_debug.py


CUDA_VISIBLE_DEVICES=0 python -m debugpy \
       --listen 0.0.0.0:4690 \
       --wait-for-client \
       fused_input.py 

    