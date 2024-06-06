python3 ./src/train.py \
    --dataset cifar10 \
    --model resnet18 \
    --batch_size 128 \
    --num_epochs 20 \
    --num_snapshots 2 \
    --scheduler_type cyclic_cosine \
    --max_lr 0.1 \
    --min_lr 0.0 \
    --optimizer sgd \
    --cycle_length -1 \
    --cycle_length_decay 1 \
    --device cuda \
    --seed 42 \
    --log_interval 100 \
    --proj_name snapshot_ensemble
