#!/bin/bash


# for embedding in avg idf cnn
for embedding in avg
do 
    python main.py \
        --dataset huffpos \
        --mode train \
        --num_way 2 \
        --num_way 1 \
        --cls rr \
        --emb $embedding
done

# https://github.com/YujiaBao/Distributional-Signatures/blob/master/bin/our.sh 
# if [ "$dataset" = "fewrel" ]; then
#     python src/main.py \
#         --cuda 0 \
#         --way 5 \
#         --shot 1 \
#         --query 25 \
#         --mode train \
#         --embedding meta \
#         --classifier r2d2 \
#         --dataset=$dataset \
#         --data_path=$data_path \
#         --n_train_class=$n_train_class \
#         --n_val_class=$n_val_class \
#         --n_test_class=$n_test_class \
#         --auxiliary pos \
#         --meta_iwf \
#         --meta_w_target
# else
#     python src/main.py \
#         --cuda 0 \
#         --way 5 \
#         --shot 1 \
#         --query 25 \
#         --mode train \
#         --embedding meta \
#         --classifier r2d2 \
#         --dataset=$dataset \
#         --data_path=$data_path \
#         --n_train_class=$n_train_class \
#         --n_val_class=$n_val_class \
#         --n_test_class=$n_test_class \
#         --meta_iwf \
#         --meta_w_target
# fi