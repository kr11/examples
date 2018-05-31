# 第一轮
epoch_size=10
use_GPU='--cuda'
# use_GPU=''
test_model='GRU'
# 1. baseline:(之后都在GPU上测）
#    1. GPU-LSTM
python train.py $use_GPU --model LSTM --epochs=20
#    2. GPU-GRU: 已经做了,但这个是bash line：
python train.py $use_GPU --model $test_model --epochs=$epoch_size --batch_size=20 --lr=20 --bptt=35 --nhid=200 --nembed=200 --dropout=0.2 --nlayers=2
#    3. CPU-GRU-mine： 已经做了
# 2. batch size 10分钟
python train.py $use_GPU --model $test_model --epochs=$epoch_size --batch_size=10
python train.py $use_GPU --model $test_model --epochs=$epoch_size --batch_size=30
# 3. lr 10分钟
python train.py $use_GPU --model $test_model --epochs=$epoch_size --lr=10
python train.py $use_GPU --model $test_model --epochs=$epoch_size --lr=30
# 4. bptt 10分钟
python train.py $use_GPU --model $test_model --epochs=$epoch_size --bptt=25
python train.py $use_GPU --model $test_model --epochs=$epoch_size --bptt=45
# 5. nhid 10分钟
python train.py $use_GPU --model $test_model --epochs=$epoch_size --nhid=100
python train.py $use_GPU --model $test_model --epochs=$epoch_size --nhid=300
# 6. embedding size
python train.py $use_GPU --model $test_model --epochs=$epoch_size --nembed=100
python train.py $use_GPU --model $test_model --epochs=$epoch_size --nembed=300
# 7. dropout
# 8. clip
# 9. layers 


