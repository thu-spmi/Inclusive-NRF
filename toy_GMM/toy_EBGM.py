import numpy as np
import theano as th
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
import lasagne
import lasagne.layers as ll
import nn
import sys
import time
import argparse
import datautil

# settings
parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--lrd', type=float, default=1e-3)
parser.add_argument('--lrg', type=float, default=1e-3)
parser.add_argument('--suffix', type=str, default='')
parser.add_argument('--max_iterations', default=500000 ,type=int) #max number of epochs
args = parser.parse_args()
print (args)

# fixed random seeds
theano_rng = MRG_RandomStreams(np.random.randint(2 ** 15))
lasagne.random.set_rng(np.random.RandomState(np.random.randint(2 ** 15)))

# specify generator
h =  theano_rng.normal(size=(args.batch_size,2))
gen_layers = [ll.InputLayer(shape=(None, 2))]
gen_layers.append(nn.batch_norm(ll.DenseLayer(gen_layers[-1], num_units=50, nonlinearity=nn.relu,name='g_1' ),name='g_b1'))
gen_layers.append(nn.batch_norm(ll.DenseLayer(gen_layers[-1], num_units=50, nonlinearity=nn.relu,name='g_2' ),name='g_b2'))
gen_layers.append(nn.l2normalize(ll.DenseLayer(gen_layers[-1], num_units=2, nonlinearity=None,name='g_3' )))
gen_dat = ll.get_output(gen_layers[-1],h,deterministic=False)

# specify random field
layers = [ll.InputLayer(shape=(None,2))]
layers.append(nn.DenseLayer(layers[-1], num_units=100, nonlinearity=nn.lrelu,name='d_1' ))
layers.append(nn.DenseLayer(layers[-1], num_units=100, nonlinearity=nn.lrelu,name='d_2' ))
layers.append(nn.DenseLayer(layers[-1], num_units=1, nonlinearity=None,train_scale=True,name='d_3' ))


labels = T.ivector()
x = T.matrix()

#weight norm initalization
temp = ll.get_output(layers[-1], x, deterministic=False, init=True)
init_updates = [u for l in layers for u in getattr(l,'init_updates',[])]
init_param = th.function(inputs=[x], outputs=None, updates=init_updates)

#generate samples
noise_sample =  T.matrix()
sample_dat= ll.get_output(gen_layers[-1],noise_sample,deterministic=False)
generate=th.function(inputs=[noise_sample], outputs=sample_dat)

#loss for random field
u_x = ll.get_output(layers[-1], x, deterministic=False)
u_sam = ll.get_output(layers[-1], gen_dat, deterministic=False)
loss_RF = T.mean(u_sam)-T.mean(u_x)
#theano function to train random field
lr = T.scalar()
RF_params = ll.get_all_params(layers, trainable=True)
RF_param_updates =lasagne.updates.adam(loss_RF, RF_params, learning_rate=lr,beta1=0)
train_RF = th.function(inputs=[x,lr], outputs=None, updates=RF_param_updates)
#loss for generator

#theano function to train generator
gen_params = ll.get_all_params(gen_layers, trainable=True)
gen_bn = [p for p in gen_params if '.g' in p.name]
h_loss=0
for p in gen_bn:
    h_loss += T.sum(T.log(p**2))
# h_loss is for entropy loss as in paper [Deep Directed Generative Models with Energy-Based Probability Estimation]
loss_gen= - T.mean(u_sam) - h_loss * 0.01
gen_param_updates = lasagne.updates.adam(loss_gen, gen_params, learning_rate=lr,beta1=0)
train_gen = th.function(inputs=[lr], outputs=None, updates=gen_param_updates)
#compute potential fuction of input x
potential_function=th.function(inputs=[x], outputs=u_x)

#get training data
trainx=datautil.produce_train_data()

init_param(trainx)
lr_D=args.lrd
lr_G=args.lrg

for it in range(args.max_iterations):

    start_id = (it*args.batch_size)%trainx.shape[0]
    train_RF(trainx[start_id:start_id+args.batch_size],lr_D)
    train_gen(lr_G)

    if (it+1)%10000==0:
        #compute "covered modes" and "realistic ratio" for generated and revised data
        gen_covered_modes_all=np.zeros(100)
        gen_realistic_ratio_all = 0.
        for j in range(100):
            h = np.cast[th.config.floatX](np.random.normal(size=(100, 2)))
            gen_dat = generate(h)
            gen_covered_modes,gen_realistic_number = datautil.get_modes(gen_dat)
            gen_covered_modes_all[j]=np.sum(gen_covered_modes)
            gen_realistic_ratio_all+=np.sum(gen_realistic_number)
        gen_realistic_ratio_all/=10000.
        print ('generation covered modes: %.4f, realistic ratio: %.4f'%(np.mean(gen_covered_modes_all),gen_realistic_ratio_all))
        sys.stdout.flush()
        if (it + 1) % 100000 == 0:
            h = np.cast[th.config.floatX](np.random.normal(size=(1000, 2)))
            gen_dat = generate(h)
            #plot generated and revised data, and potentials
            datautil.plot_data(gen_dat,'ebgm_it%d_%s'%(it+1,args.suffix))
            datautil.plot_ux(potential_function,'ebgm_ux_it%d_%s'%(it+1,args.suffix))






