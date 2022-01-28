import os
import itertools
import argparse
import time
'''parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0) # also data seed
parser.add_argument('--data_path', type=str, default='data/wn18rr')
args = parser.parse_args()'''
def run_commands():
    for idx in range(0, 16):
        # param = param_range[idx]
        # data_path, dim, hd, g, init, a, lr, bt, neg, r, model = param

        # gpu_id = (idx+args.gpu_id) % 16
        tmp=0+idx

        command = f'''CUDA_VISIBLE_DEVICES={idx} nohup python -u codes/run.py --do_train --cuda --do_valid --do_test --data_path data/FB15k --model HousD -n 256 -b 1024 -d 2000 -hd 18 -g 9.0 -a 1.0 -adv -lr 0.00005 --max_steps 200000 --warm_up_steps 50000 -save models/fb15k_pd_{tmp} --test_batch_size 16 > ./logs/fb15k_pd_last_{tmp}.log 2>&1 &
        '''
        print('Command:', command)
        os.system(command)
        time.sleep(2)


'''data_path = [args.data_path]
dims = [400]
hds = [4]
gs = [5.0, 6.0, 9.0, 12.0]
inits = [0.01, 0.015, 0.02]
a_list = [0.3, 0.5, 0.7]
lrs = [0.00005]
bts = [512, 1024]
negs = [256, 512, 1024]
rs = [0.2]
models = ['HousH']
param_range = list(itertools.product(data_path, dims, hds, gs, inits, a_list, lrs, bts, negs, rs, models))'''
run_commands()
# print(len(param_range))
