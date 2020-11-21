import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--lrd', type=float, default=1e-4)
parser.add_argument('--lrg', type=float, default=3e-4)
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--gw', default=1.0, type=float)
parser.add_argument('--L', default=10, type=int)
parser.add_argument('--fxp', default=0.1, type=float)
parser.add_argument('--del_we', default=1, type=float)
parser.add_argument('--max_e', default=30, type=int)
parser.add_argument('--alpha', default=0, type=float)
parser.add_argument('--eta', default=0.003, type=float)
parser.add_argument('--sf', type=str, default='')
parser.add_argument('--load', type=str, default='')
parser.add_argument('--cof', default=0, type=float)
parser.add_argument('--sig', default=0, type=float)
parser.add_argument('--beta1', default=0.5, type=float)
parser.add_argument('--beta2', default=0.999, type=float)
parser.add_argument('--seed', type=int, default=1)

args = parser.parse_args()
print(args)

if __name__ == '__main__':
    import pickle
    import numpy as np
    import os,sys
    import kdd_dec
    os.makedirs("result",exist_ok=True)
    b_s=np.zeros((20,3))

    for seed in range(1,21):
        best_score=kdd_dec.main(seed,args)
        print("seed:",seed,"best:",best_score)
        sys.stdout.flush()

        b_s[seed-1]=best_score
    print(b_s)
    sys.stdout.flush()

    pickle.dump(b_s,open('result/kdd_20_%s'%args.sf,'wb' ))
    print(np.mean(b_s,0))
