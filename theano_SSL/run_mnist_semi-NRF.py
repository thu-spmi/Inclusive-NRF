import argparse
import time
import numpy as np
import theano as th
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
import lasagne
import lasagne.layers as ll
from lasagne.init import Normal
from lasagne.layers import dnn
import nn
import sys
from checkpoints import save_weights,load_weights

# settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1000)     #random seed for theano operation
parser.add_argument('--seed_data', type=int, default=1000)  #random seed for picking labeled data
parser.add_argument('--count', type=int, default=10)   #how much data one class
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--supervised_loss_weight', type=float, default=10.)
parser.add_argument('--lrd', type=float, default=1e-3)
parser.add_argument('--lrg', type=float, default=3e-3)
parser.add_argument('--entropy_loss_weight', default=10 ,type=float)    #weight for confidence loss
parser.add_argument('--beta', default=0.5 ,type=float)   #beta for SGHMC
parser.add_argument('--gradient_coefficient', default=0.003,type=float)  #coefficient for gradient term of SGLD/SGHMC
parser.add_argument('--noise_coefficient', default=0,type=float)   #coefficient for noise term of SGLD/SGHMC
parser.add_argument('--L', default=20 ,type=int)   #revision steps
parser.add_argument('--max_e', default=200 ,type=int)   #max number of epochs
parser.add_argument('--revison_method', default='revision_x_sghmc' ,type=str)   #revision method
parser.add_argument('--load', default='' ,type=str)    #file name to load trained model
parser.add_argument('--data_root', type=str, default='../data/mnist.npz')   #data folder to load
args = parser.parse_args()
print(args)

# fixed random seeds
rng = np.random.RandomState(args.seed)
theano_rng = MRG_RandomStreams(rng.randint(2 ** 15))
lasagne.random.set_rng(np.random.RandomState(rng.randint(2 ** 15)))

#logsoftmax for computing entropy
def logsoftmax(x):
    xdev = x - T.max(x, 1, keepdims=True)
    lsm = xdev - T.log(T.sum(T.exp(xdev), 1, keepdims=True))
    return lsm

#load MNIST data
data = np.load(args.data_root)
trainx = np.concatenate([data['x_train'], data['x_valid']], axis=0).astype(th.config.floatX)
trainy = np.concatenate([data['y_train'], data['y_valid']]).astype(np.int32)
nr_batches_train = int(trainx.shape[0]/args.batch_size)
testx = data['x_test'].astype(th.config.floatX)
testy = data['y_test'].astype(np.int32)
nr_batches_test = int(testx.shape[0]/args.batch_size)
trainx_unl = trainx.copy()

# specify generator
h = T.matrix()
gen_layers = [ll.InputLayer(shape=(None, 100))]
gen_layers.append(nn.batch_norm(ll.DenseLayer(gen_layers[-1], num_units=500, W=Normal(0.05),nonlinearity=T.nnet.softplus,name='g1'), g=None,name='g_b1'))
gen_layers.append(nn.batch_norm(ll.DenseLayer(gen_layers[-1], num_units=500,  W=Normal(0.05),nonlinearity=T.nnet.softplus,name='g2'), g=None,name='g_b2'))
gen_layers.append(nn.l2normalize(ll.DenseLayer(gen_layers[-1], num_units=28**2,  W=Normal(0.05),nonlinearity=T.nnet.sigmoid,name='g3')))
gen_dat = ll.get_output(gen_layers[-1],h, deterministic=False)

# specify random field
layers = [ll.InputLayer(shape=(None, 28**2))]
layers.append(nn.DenseLayer(layers[-1], num_units=1000,theta=Normal(0.05),name='d_1'))
layers.append(nn.DenseLayer(layers[-1], num_units=500,theta=Normal(0.05),name='d_2'))
layers.append(nn.DenseLayer(layers[-1], num_units=250,theta=Normal(0.05),name='d_3'))
layers.append(nn.DenseLayer(layers[-1], num_units=250,theta=Normal(0.05),name='d_4'))
layers.append(nn.DenseLayer(layers[-1], num_units=250,theta=Normal(0.05),name='d_5'))
layers.append(nn.DenseLayer(layers[-1], num_units=10,theta=Normal(0.05), nonlinearity=None, train_scale=True))
RF_params = ll.get_all_params(layers, trainable=True)

#revision method
if args.revison_method=='revision_x_sgld':    #only x will be revised, SGLD
    x_revised = gen_dat
    gradient_coefficient = T.scalar()
    noise_coefficient = T.scalar()
    for i in range(args.L):
        loss_revision=T.sum(nn.log_sum_exp(ll.get_output(layers[-1], x_revised, deterministic=False)))
        gradient_x = T.grad(loss_revision, [x_revised])[0]
        x_revised = x_revised + gradient_coefficient * gradient_x + noise_coefficient * theano_rng.normal(size=T.shape(x_revised))

    revision = th.function(inputs=[h, gradient_coefficient, noise_coefficient], outputs=x_revised)

elif args.revison_method=='revision_x_sghmc':  #only x will be revised, SGHMC
    x_revised = gen_dat
    gradient_coefficient = T.scalar()
    beta = T.scalar()
    noise_coefficient = T.scalar()
    v_x = 0.
    for i in range(args.L):
        loss_revision=T.sum(nn.log_sum_exp(ll.get_output(layers[-1], x_revised, deterministic=False)))
        gradient_x = T.grad(loss_revision, [x_revised])[0]
        v_x = beta * v_x + gradient_coefficient * gradient_x + noise_coefficient * theano_rng.normal(size=T.shape(x_revised))
        x_revised = x_revised + v_x

    revision = th.function(inputs=[h, beta,gradient_coefficient, noise_coefficient], outputs=x_revised)
elif args.revison_method=='revision_joint_sgld':  #x and h will be revised jointly, SGLD
    x_revised = gen_dat
    h_revised = h
    gradient_coefficient = T.scalar()
    noise_coefficient = T.scalar()
    for i in range(args.L):

        loss_x_revision=T.sum(nn.log_sum_exp(ll.get_output(layers[-1], x_revised, deterministic=False)))
        gradient_x = T.grad(loss_x_revision, [x_revised])[0]
        x_revised = x_revised + gradient_coefficient * gradient_x + noise_coefficient * theano_rng.normal(size=T.shape(x_revised))
        if i==0:
            loss_h_revision = T.sum(T.square(x_revised - gen_dat)) + T.sum(T.square(h))/args.batch_size
            gradient_h = T.grad(loss_h_revision, [h])[0]
            h_revised= h - gradient_coefficient * gradient_h + noise_coefficient * theano_rng.normal(size=T.shape(h))
        else:
            loss_h_revision = T.sum(T.square(x_revised - gen_dat_h_revised))+ T.sum(T.square(h_revised))/args.batch_size
            gradient_h = T.grad(loss_h_revision, [h_revised])[0]
            h_revised = h_revised - gradient_coefficient * gradient_h + noise_coefficient * theano_rng.normal(size=T.shape(h))
        gen_dat_h_revised=ll.get_output(gen_layers[-1],h_revised, deterministic=False)

    revision = th.function(inputs=[h,gradient_coefficient, noise_coefficient], outputs=[x_revised,h_revised])
elif args.revison_method=='revision_joint_sghmc':   #x and h will be revised jointly, SGHMC
    x_revised = gen_dat
    h_revised = h
    beta=T.scalar()
    gradient_coefficient = T.scalar()
    noise_coefficient = T.scalar()
    v_x=0.
    for i in range(args.L):

        loss_x_revision=T.sum(nn.log_sum_exp(ll.get_output(layers[-1], x_revised, deterministic=False)))
        gradient_x = T.grad(loss_x_revision, [x_revised])[0]
        v_x=v_x*beta + gradient_coefficient * gradient_x + noise_coefficient * theano_rng.normal(size=T.shape(x_revised))
        x_revised = x_revised + v_x

        if i==0:
            loss_h_revision = T.sum(T.square(x_revised - gen_dat))+ T.sum(T.square(h))/args.batch_size
            gradient_h = T.grad(loss_h_revision, [h])[0]
            v_h= gradient_coefficient * gradient_h + noise_coefficient * theano_rng.normal(size=T.shape(h))
            h_revised= h - v_h

        else:
            loss_h_revision = T.sum(T.square(x_revised - gen_dat_h_revised))+ T.sum(T.square(h_revised))/args.batch_size
            gradient_h = T.grad(loss_h_revision, [h_revised])[0]
            v_h=v_h*beta+gradient_coefficient * gradient_h + noise_coefficient * theano_rng.normal(size=T.shape(h))
            h_revised = h_revised - v_h
            gen_dat_h_revised=ll.get_output(gen_layers[-1],h_revised, deterministic=False)

    revision = th.function(inputs=[h, beta,gradient_coefficient, noise_coefficient], outputs=[x_revised,h_revised])

supervised_loss_weight = T.scalar()
entropy_loss_weight = T.scalar()
x_revised = T.matrix()
labels = T.ivector()
x_lab = T.matrix()
x_unl = T.matrix()
temp = ll.get_output(layers[-1], x_lab, deterministic=False, init=True)
init_updates = [u for l in layers for u in getattr(l,'init_updates',[])]

output_before_softmax_lab = ll.get_output(layers[-1], x_lab, deterministic=False)
output_before_softmax_unl = ll.get_output(layers[-1], x_unl, deterministic=False)
output_before_softmax_revised = ll.get_output(layers[-1], x_revised, deterministic=False)

logit_lab = output_before_softmax_lab[T.arange(T.shape(x_lab)[0]),labels]

u_lab = T.mean(nn.log_sum_exp(output_before_softmax_lab))
u_unl = T.mean(nn.log_sum_exp(output_before_softmax_unl))
u_revised = T.mean(nn.log_sum_exp(output_before_softmax_revised))
#cross entropy loss of labeled data
loss_lab = -T.mean(logit_lab) + u_lab

#entropy of unlabeded data
entropy_unl=-T.sum(T.nnet.softmax(output_before_softmax_unl)*logsoftmax(output_before_softmax_unl))/args.batch_size
#unsupervised loss
loss_unl = u_revised-u_unl + entropy_unl*entropy_loss_weight
#train_err
train_err = T.mean(T.neq(T.argmax(output_before_softmax_lab,axis=1),labels))
#loss on random field
loss_RF=loss_lab*supervised_loss_weight+loss_unl

# Theano functions for training the random field
lr = T.scalar()
RF_params = ll.get_all_params(layers, trainable=True)
RF_param_updates = lasagne.updates.rmsprop(loss_RF, RF_params, learning_rate=lr)
train_RF = th.function(inputs=[x_revised,x_lab,labels,x_unl,lr,supervised_loss_weight,entropy_loss_weight], outputs=[loss_lab, loss_unl, train_err], updates=RF_param_updates)
#weight norm initalization
init_param = th.function(inputs=[x_lab], outputs=None, updates=init_updates)
#predition on test data
output_before_softmax = ll.get_output(layers[-1], x_lab, deterministic=True)
test_batch = th.function(inputs=[x_lab], outputs=output_before_softmax)

#loss on generator
loss_G = T.sum(T.square(x_revised - gen_dat))
# Theano functions for training the generator
gen_params = ll.get_all_params(gen_layers, trainable=True)
gen_param_updates =lasagne.updates.rmsprop(loss_G, gen_params, learning_rate=lr)
train_G = th.function(inputs=[h,x_revised,lr], outputs=None, updates=gen_param_updates)

# select labeled data

rng_data = np.random.RandomState(args.seed_data)
inds = rng_data.permutation(trainx.shape[0])
trainx = trainx[inds]
trainy = trainy[inds]
txs = []
tys = []
for j in range(10):
    txs.append(trainx[trainy==j][:args.count])
    tys.append(trainy[trainy==j][:args.count])
txs = np.concatenate(txs, axis=0)
tys = np.concatenate(tys, axis=0)


# //////////// perform training //////////////
lr_D=args.lrd
lr_G=args.lrg
beta=args.beta
gradient_coefficient=args.gradient_coefficient
noise_coefficient=args.noise_coefficient
supervised_loss_weight = args.supervised_loss_weight
entropy_loss_weight=0.
acc_all=[]
best_acc=1
for epoch in range(args.max_e):
    begin = time.time()
    # construct randomly permuted minibatches
    trainx = []
    trainy = []
    for t in range(int(np.ceil(trainx_unl.shape[0]/float(txs.shape[0])))):
        inds = rng.permutation(txs.shape[0])
        trainx.append(txs[inds])
        trainy.append(tys[inds])
    trainx = np.concatenate(trainx, axis=0)
    trainy = np.concatenate(trainy, axis=0)
    trainx_unl = trainx_unl[rng.permutation(trainx_unl.shape[0])]
    if (epoch+1)==40:
        # adding confidence loss after burn-in
        entropy_loss_weight=args.entropy_loss_weight
    if epoch == 60:
        #drop learning rate
        lr_D /= 10
        lr_G /= 3
    if epoch==0:
        init_param(trainx[:500]) # data based initialization
        if args.load:
            load_weights('mnist_model/mnist_jrf_'+args.load+'.npy', layers + gen_layers)
    # train
    loss_lab = 0.
    loss_unl = 0.
    train_err = 0.

    for t in range(nr_batches_train):
        h = np.cast[th.config.floatX](rng.uniform(size=(args.batch_size, 100)))
        if args.revison_method=='revision_x_sgld':
            x_revised = revision(h, gradient_coefficient, noise_coefficient)
        elif args.revison_method=='revision_x_sghmc':
            x_revised= revision(h, beta, gradient_coefficient, noise_coefficient)
        elif args.revison_method == 'revision_joint_sgld':
            x_revised,h = revision(h, gradient_coefficient, noise_coefficient)
        elif args.revison_method == 'revision_joint_sghmc':
            x_revised,h = revision(h, beta, gradient_coefficient, noise_coefficient)
        ran_from = t * args.batch_size
        ran_to = (t + 1) * args.batch_size
        #updata random field
        lo_lab, lo_unl, tr_er = train_RF(x_revised, trainx[ran_from:ran_to], trainy[ran_from:ran_to],
                                      trainx_unl[ran_from:ran_to], lr_D,supervised_loss_weight,entropy_loss_weight)
        loss_lab += lo_lab
        loss_unl += lo_unl
        train_err += tr_er
        #updata generator
        train_G(h,x_revised, lr_G)
    loss_lab /= nr_batches_train
    loss_unl /= nr_batches_train
    train_err /= nr_batches_train
    # test
    test_pred = np.zeros((len(testy), 10), dtype=th.config.floatX)
    for t in range(nr_batches_test):
        last_ind = np.minimum((t + 1) * args.batch_size, len(testy))
        first_ind = last_ind - args.batch_size
        test_pred[first_ind:last_ind] = test_batch(testx[first_ind:last_ind])
    test_err = np.mean(np.argmax(test_pred, axis=1) != testy)
    print("epoch %d, time = %ds, loss_lab = %.4f, loss_unl = %.4f, train err = %.4f, test err = %.4f, best_err = %.4f" % (
        epoch+1, time.time() - begin, loss_lab, loss_unl,train_err, test_err,best_acc))
    sys.stdout.flush()
    acc_all.append(test_err)

    if acc_all[-1] < best_acc:
        best_acc = acc_all[-1]
    if (epoch+1)%50==0:
        import os
        if not os.path.exists('mnist_model'):
            os.mkdir('mnist_model')
        params = ll.get_all_params(layers + gen_layers)
        save_weights('mnist_model/semi_nrf_data%d_ep%d.npy'%(args.seed_data,epoch+1), params)
    if loss_unl<-100:
        break








