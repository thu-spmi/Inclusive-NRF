import argparse

# settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1000)     #random seed for theano operation
parser.add_argument('--seed_data', type=int, default=1000)  #random seed for picking labeled data
parser.add_argument('--count', type=int, default=10)   #how much data one class
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--lrd', type=float, default=3e-3)
parser.add_argument('--lrg', type=float, default=1e-3)
parser.add_argument('--beta', default=0.5 ,type=float)   #beta for SGHMC
parser.add_argument('--gradient_coefficient', default=0.003,type=float)  #coefficient for gradient term of SGLD/SGHMC
parser.add_argument('--noise_coefficient', default=0,type=float)   #coefficient for noise term of SGLD/SGHMC
parser.add_argument('--fxp', default=1,type=float)   #coefficient for noise term of SGLD/SGHMC
parser.add_argument('--sig', default=0.1,type=float)   #coefficient for noise term of SGLD/SGHMC
parser.add_argument('--L', default=20 ,type=int)   #revision steps
parser.add_argument('--max_e', default=50 ,type=int)   #max number of epochs
parser.add_argument('--no', default=0 ,type=int)   #训练集只包含某个标签
parser.add_argument('--revison_method', default='revision_x_sghmc' ,type=str)   #revision method
parser.add_argument('--sf', default='' ,type=str)   #revision method
parser.add_argument('--load', default='' ,type=str)    #file name to load trained model
parser.add_argument('--data_root', type=str, default='../../data/mnist.npz')   #data folder to load
parser.add_argument('--method', default=1 ,type=int)
args = parser.parse_args()
print(args)


if __name__ == '__main__':
    import pickle
    import numpy as np
    import os,sys
    import mnist_dec
    if not os.path.exists('result'):
        os.mkdir('result')
    if args.method==1:
        # 对于给定的某个标签和seed进行实验
        best_score=mnist_dec.main(args.no,args.seed,args)
        print("num:",args.no,"seed:",args.seed,"best:",best_score)
        sys.stdout.flush()
    elif args.method==2:
        # 对于0-9所有标签，seed 1-10进行实验
        b_s=np.zeros((10,10))
        for num in range(10):
            for seed in range(1,11):
                best_score=mnist_dec.main(num,seed,args)
                print("num:",num,"seed:",seed,"best:",best_score)
                sys.stdout.flush()
                b_s[num,seed-1]=best_score
        print(b_s)
        sys.stdout.flush()

        pickle.dump(b_s,open('result/mnist_nrf_dec_%s'%args.sf,'wb' ))
        print(np.mean(b_s,1))


