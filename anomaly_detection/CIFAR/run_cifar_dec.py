import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lrd', type=float, default=1e-3)
parser.add_argument('--lrg', type=float, default=1e-3)
parser.add_argument('--loss', type=str, default='hinge')
parser.add_argument('--gpu', default='0' ,type=str)
parser.add_argument('--opt', type=str, default='rms')
parser.add_argument('--gw', default=1.0 ,type=float)
parser.add_argument('--L', default=10 ,type=int)
parser.add_argument('--fxp', default=0.1 ,type=float)
parser.add_argument('--del_we', default=1 ,type=float)
parser.add_argument('--max_e', default=100 ,type=int)
parser.add_argument('--alpha', default=0 ,type=float)
parser.add_argument('--eta', default=0.03,type=float)
parser.add_argument('--sf', type=str, default='')
parser.add_argument('--load', type=str, default='')
parser.add_argument('--cof', default=0,type=float)
parser.add_argument('--sig', default=0,type=float)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--method', type=int, default=1)
parser.add_argument('--no', type=str, default='0')   #表示训练集只使用某个标签
args = parser.parse_args()
print(args)


if __name__ == '__main__':
    import pickle
    import numpy as np
    import os, sys
    import cifar_dec
    if not os.path.exists('cifar_result'):
        os.mkdir('cifar_result')
    if args.method==1:
        # 对给定的一个标签与seed进行实验
        args.no=int(args.no)
        best_score=cifar_dec.main(args.no,args.seed,args)
        print("num:",args.no,"seed:",args.seed,"best:",best_score)
        sys.stdout.flush()
    elif args.method==2:
        #对给定的一组标签以及seed 1-10进行实验
        if os.path.exists('cifar_result/cifar_nrf_dec_%s'%args.sf):
            b_s=pickle.load(open('cifar_result/cifar_nrf_dec_%s'%args.sf,'rb' ))
        else:b_s=np.zeros((10,10))
        num_all=[int(num) for num in args.no.split(',')]

        for num in num_all:
            for seed in range(1,11):
                best_score=cifar_dec.main(num,seed,args)
                print("num:",num,"seed:",seed,"best:",best_score)
                sys.stdout.flush()
                b_s[num,seed-1]=best_score
        print(b_s)
        sys.stdout.flush()
        pickle.dump(b_s,open('cifar_result/cifar_nrf_dec_%s'%args.sf,'wb' ))
        print(np.mean(b_s,1))

