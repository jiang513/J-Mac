CUDA_VISIBLE_DEVICES=0 python3 /J-Mac/DMC-GB/src/train.py \
    --algorithm svea \
    --seed 1 \
    --domain_name walker \
    --task_name walk \
    --action_repeat 4 \
    --use_aux \
    --use_jacobian
