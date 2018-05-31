# 第四轮：LSTM的全套实验
epoch_size=10
use_GPU='--cuda'
# use_GPU=''
test_model='LSTM'
#    2. GPU-GRU: 已经做了,但这个是bash line：
python train.py $use_GPU --model $test_model --epochs=$epoch_size --batch_size=20 --lr=20 --bptt=35 --nhid=200 --nembed=200 --dropout=0.2 --nlayers=2
#    3. CPU-GRU-mine： 已经做了
# 2. batch size 10分钟
python train.py $use_GPU --model $test_model --epochs=$epoch_size --batch_size=5
python train.py $use_GPU --model $test_model --epochs=$epoch_size --batch_size=10
python train.py $use_GPU --model $test_model --epochs=$epoch_size --batch_size=30
python train.py $use_GPU --model $test_model --epochs=$epoch_size --batch_size=40
# 3. lr 10分钟
python train.py $use_GPU --model $test_model --epochs=$epoch_size --lr=5
python train.py $use_GPU --model $test_model --epochs=$epoch_size --lr=10
python train.py $use_GPU --model $test_model --epochs=$epoch_size --lr=30
python train.py $use_GPU --model $test_model --epochs=$epoch_size --lr=40
# 4. bptt 10分钟
python train.py $use_GPU --model $test_model --epochs=$epoch_size --bptt=15
python train.py $use_GPU --model $test_model --epochs=$epoch_size --bptt=25
python train.py $use_GPU --model $test_model --epochs=$epoch_size --bptt=45
python train.py $use_GPU --model $test_model --epochs=$epoch_size --bptt=55
# 5. nhid 10分钟
python train.py $use_GPU --model $test_model --epochs=$epoch_size --nhid=50
python train.py $use_GPU --model $test_model --epochs=$epoch_size --nhid=100
python train.py $use_GPU --model $test_model --epochs=$epoch_size --nhid=300
python train.py $use_GPU --model $test_model --epochs=$epoch_size --nhid=400
# 6. embedding size
python train.py $use_GPU --model $test_model --epochs=$epoch_size --nembed=50
python train.py $use_GPU --model $test_model --epochs=$epoch_size --nembed=100
python train.py $use_GPU --model $test_model --epochs=$epoch_size --nembed=300
python train.py $use_GPU --model $test_model --epochs=$epoch_size --nembed=400
# 7. dropout
python train.py $use_GPU --model $test_model --epochs=$epoch_size --dropout=0.1
python train.py $use_GPU --model $test_model --epochs=$epoch_size --dropout=0.3
python train.py $use_GPU --model $test_model --epochs=$epoch_size --dropout=0.4
python train.py $use_GPU --model $test_model --epochs=$epoch_size --dropout=0.5
# 8. clip
# python train.py $use_GPU --model $test_model --epochs=$epoch_size --nembed=50
# 9. layers 
python train.py $use_GPU --model $test_model --epochs=$epoch_size --nlayers=1
python train.py $use_GPU --model $test_model --epochs=$epoch_size --nlayers=3
python train.py $use_GPU --model $test_model --epochs=$epoch_size --nlayers=4
python train.py $use_GPU --model $test_model --epochs=$epoch_size --nlayers=5


# 记得找 ./script2.sh这一行

