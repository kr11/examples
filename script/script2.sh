#!/usr/bin/env bash
# 第一轮
epoch_size=10
use_GPU='--cuda'
# use_GPU=''
test_model='GRU'
# 1. baseline:(之后都在GPU上测）
#    1. GPU-LSTM
# python train.py $use_GPU --model LSTM --epochs=20
#    2. GPU-GRU: 已经做了,但这个是bash line：
# python train.py $use_GPU --model $test_model --epochs=$epoch_size --batch_size=20 --lr=20 --bptt=35 --nhid=200 --nembed=200 --dropout=0.2 --nlayers=2
#    3. CPU-GRU-mine： 已经做了
# 2. batch size 10分钟
python train.py $use_GPU --model $test_model --epochs=$epoch_size --batch_size=5
python train.py $use_GPU --model $test_model --epochs=$epoch_size --batch_size=40
# 3. lr 10分钟
python train.py $use_GPU --model $test_model --epochs=$epoch_size --lr=5
python train.py $use_GPU --model $test_model --epochs=$epoch_size --lr=40
# 4. bptt 10分钟
python train.py $use_GPU --model $test_model --epochs=$epoch_size --bptt=15
python train.py $use_GPU --model $test_model --epochs=$epoch_size --bptt=55
# 5. nhid 10分钟
python train.py $use_GPU --model $test_model --epochs=$epoch_size --nhid=50
python train.py $use_GPU --model $test_model --epochs=$epoch_size --nhid=400
# 6. embedding size
python train.py $use_GPU --model $test_model --epochs=$epoch_size --nembed=50
python train.py $use_GPU --model $test_model --epochs=$epoch_size --nembed=400
# 7. dropout
# 8. clip
# 9. layers 


# 记得找 ./script2.sh这一行

