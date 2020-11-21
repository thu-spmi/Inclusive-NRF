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
parser.add_argument('--gradient_coefficient', default=0.01 ,type=float)  #coefficient for gradient term of SGLD/SGHMC
parser.add_argument('--noise_coefficient', default=0.01,type=float)  #coefficient for noise term of SGLD/SGHMC
parser.add_argument('--sigma', default=0,type=float)   #noise std on generated data
parser.add_argument('--L', default=10 ,type=int)   #revision steps
parser.add_argument('--max_iterations', default=500000 ,type=int) #max number of epochs
args = parser.parse_args()
print (args)

# fixed random seeds
theano_rng = MRG_RandomStreams(np.random.randint(2 ** 15))
lasagne.random.set_rng(np.random.RandomState(np.random.randint(2 ** 15)))

# specify generator
y = T.matrix()

h =  T.matrix()
gen_layers = [ll.InputLayer(shape=(None, 2))]
# gen_layers.append(nn.MLPConcatLayer(ll.DenseLayer(gen_layers[-1])))
gen_layers.append(nn.batch_norm(ll.DenseLayer(gen_layers[-1], num_units=50, nonlinearity=nn.relu,name='g_1' ),name='g_b1'))
gen_layers.append(nn.batch_norm(ll.DenseLayer(gen_layers[-1], num_units=50, nonlinearity=nn.relu,name='g_2' ),name='g_b2'))
gen_layers.append(nn.l2normalize(ll.DenseLayer(gen_layers[-1], num_units=2, nonlinearity=None,name='g_3' )))
gen_dat = ll.get_output(gen_layers[-1],h,deterministic=False)

# specify random field
layers = [ll.InputLayer(shape=(None,2))]
layers.append(nn.DenseLayer(layers[-1], num_units=100, nonlinearity=nn.lrelu,name='d_1' ))
layers.append(nn.DenseLayer(layers[-1], num_units=100, nonlinearity=nn.lrelu,name='d_2' ))
layers.append(nn.DenseLayer(layers[-1], num_units=1, nonlinearity=None,train_scale=True,name='d_3' ))


#we simply use SGLD revision only on x in toy experiment
x_revised=gen_dat+args.sigma * theano_rng.normal(size=T.shape(gen_dat))
for i in range(args.L):
    loss_revision=T.sum(nn.log_sum_exp(ll.get_output(layers[-1], x_revised)))
    gradient = T.grad(loss_revision, [x_revised])[0]
    x_revised = x_revised + args.gradient_coefficient * gradient + args.noise_coefficient * theano_rng.normal(size=T.shape(gen_dat))
revision=th.function(inputs=[h], outputs=[x_revised,gen_dat])

x_revised= T.matrix()
labels = T.ivector()
x = T.matrix()

#weight norm initalization
temp = ll.get_output(layers[-1], x, deterministic=False, init=True)
init_updates = [u for l in layers for u in getattr(l,'init_updates',[])]
init_param = th.function(inputs=[x], outputs=None, updates=init_updates)

#loss for random field
u_x = ll.get_output(layers[-1], x, deterministic=False)
u_revised = ll.get_output(layers[-1], x_revised, deterministic=False)
loss_RF = T.mean(u_revised)-T.mean(u_x)
#theano function to train random field
lr = T.scalar()
RF_params = ll.get_all_params(layers, trainable=True)
RF_param_updates =lasagne.updates.adam(loss_RF, RF_params, learning_rate=lr,beta1=0)
train_RF = th.function(inputs=[x_revised,x,lr], outputs=None, updates=RF_param_updates)
#loss for generator
loss_gen=T.sum(T.square(x_revised-gen_dat))
#theano function to train generator
gen_params = ll.get_all_params(gen_layers, trainable=True)
gen_param_updates = lasagne.updates.adam(loss_gen, gen_params, learning_rate=lr,beta1=0)
train_gen = th.function(inputs=[h,x_revised,lr], outputs=None, updates=gen_param_updates)
#compute potential fuction of input x
potential_function=th.function(inputs=[x], outputs=u_x)

#get training data
trainx=datautil.produce_train_data()

init_param(trainx)
lr_D=args.lrd
lr_G=args.lrg

for it in range(args.max_iterations):

    start_id = (it*args.batch_size)%trainx.shape[0]
    h = np.cast[th.config.floatX](np.random.normal(size=(args.batch_size, 2)))
    x_revised, gen_dat = revision(h)
    train_RF(x_revised,trainx[start_id:start_id+args.batch_size],lr_D)
    train_gen(h,x_revised,lr_G)

    if (it+1)%10000==0:
        #compute "covered modes" and "realistic ratio" for generated and revised data
        gen_covered_modes_all=np.zeros(100)
        gen_realistic_ratio_all = 0.
        revised_covered_modes_all=np.zeros(100)
        revised_realistic_ratio_all = 0.
        for j in range(100):
            h = np.cast[th.config.floatX](np.random.normal(size=(100, 2)))
            x_revised, gen_dat = revision(h)
            gen_covered_modes,gen_realistic_number = datautil.get_modes(gen_dat)
            gen_covered_modes_all[j]=np.sum(gen_covered_modes)
            gen_realistic_ratio_all+=np.sum(gen_realistic_number)
            revised_covered_modes, revised_realistic_number= datautil.get_modes(x_revised)
            revised_covered_modes_all[j]=np.sum(revised_covered_modes)
            revised_realistic_ratio_all += np.sum(revised_realistic_number)
        gen_realistic_ratio_all/=10000.
        revised_realistic_ratio_all /= 10000.
        print ('generation covered modes: %.4f, realistic ratio: %.4f'%(np.mean(gen_covered_modes_all),gen_realistic_ratio_all))
        print('revision covered modes: %.4f, realistic ratio: %.4f' % (
        np.mean(revised_covered_modes_all), revised_realistic_ratio_all))
        sys.stdout.flush()
        if (it + 1) % 100000 == 0:
            h = np.cast[th.config.floatX](np.random.normal(size=(1000, 2)))
            x_revised, gen_dat = revision(h)
            #plot generated and revised data, and potentials
            datautil.plot_data(gen_dat,'nrf_gen_it%d_%s'%(it+1,args.suffix))
            datautil.plot_data(x_revised,'nrf_rev_it%d_%s'%(it+1,args.suffix))
            datautil.plot_ux(potential_function,'nrf_ux_it%d_%s'%(it+1,args.suffix))


