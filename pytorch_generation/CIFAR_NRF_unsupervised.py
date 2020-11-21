import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import os,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lrd', type=float, default=3e-4)
parser.add_argument('--lrg', type=float, default=1e-4)
parser.add_argument('--L', default=1 ,type=int)   #revision steps
parser.add_argument('--potential_control_weight', default=0.1 ,type=float)  #weight for base random field loss, i.e. f-E[f]
parser.add_argument('--max_iterations', default=500000 ,type=int)    #maximal iterations
parser.add_argument('--gradient_coefficient', default=0.003,type=float)  #coefficient for gradient term of SGLD/SGHMC
parser.add_argument('--noise_coefficient', default=0,type=float)   #coefficient for noise term of SGLD/SGHMC
parser.add_argument('--suffix', type=str, default='')
parser.add_argument('--data_root', type=str, default='../data/cifar-10-python')   #CIFAR10 data folder
parser.add_argument('--stat_flie_path', type=str, default='../data/chainer_stat_CIFAR.npz')   #stat file to compute FID
parser.add_argument('--evalution_model', type=str, default='chainer')
parser.add_argument('--load', type=str, default='')
args = parser.parse_args()
print(args)

#load CIFAR data
loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(args.data_root,train=True,download=True,
    transform=transforms.Compose([transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
    batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True,drop_last=True)

def cycle(iterable):
    while True:
        for x in iterable:
            yield x
loader = iter(cycle(loader))

h_dim = 128
#initialize random field and generator,using ResNet
import model_unsupervised
RF = model_unsupervised.Discriminator().cuda()
generator = model_unsupervised.Generator(h_dim).cuda()
print(RF)
sys.stdout.flush()

# because the spectral normalization module creates parameters that don't require gradients (u and v), we don't want to 
# optimize these using sgd. We only let the optimizer operate on parameters that _do_ require gradients
optim_RF = optim.Adam(filter(lambda p: p.requires_grad, RF.parameters()), lr=args.lrd, betas=(0.,0.9))
optim_G  = optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()), lr=args.lrg, betas=(0.,0.9))

#for CIFAR generation, we simply use one step SGLD for revision
def revision(gen_images):
    x_revised = gen_images.data
    x_revised = Variable(x_revised, requires_grad=True)
    for i in range(args.L):
        loss_revision = torch.mean(RF(x_revised))
        gradient = torch.autograd.grad(outputs=loss_revision, inputs=x_revised)[0]

        x_revised = x_revised + args.gradient_coefficient * gradient + torch.randn(x_revised.size(0),3,32,32).cuda()* args.noise_coefficient
        x_revised.data.clamp_(-1, 1)

    return x_revised.data

#one alternant update for random field and generator
def train():
    #compute random field loss
    data, _ = next(loader)
    data=data.cuda()
    h = Variable(torch.randn(args.batch_size, h_dim).cuda())
    gen_images=generator(h)
    x_revised=revision(gen_images)
    u_unl = RF(data)

    u_revised = RF(x_revised)
    loss_RF = torch.mean(u_revised) - torch.mean(u_unl) + args.potential_control_weight * torch.mean(u_unl ** 2)
    #update random field
    optim_RF.zero_grad()
    loss_RF.backward()
    optim_RF.step()

    #for loss of generator, see Section 11 of our paper
    h = Variable(torch.randn(args.batch_size, h_dim).cuda())
    gen_images = generator(h)
    loss_G = -torch.mean(RF(gen_images))
    #update generator
    optim_G.zero_grad()
    loss_G.backward()
    optim_G.step()

#generate 100 images and store
fixed_h = Variable(torch.randn(100, h_dim).cuda())

def generate():
    samples = generator(fixed_h).cpu().data.numpy()
    fig = plt.figure(figsize=(10,10))
    gs = gridspec.GridSpec(10,10)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.transpose((1,2,0)) * 0.5 + 0.5)

    if not os.path.exists('cifar_image/'):
        os.makedirs('cifar_image/')

    plt.savefig('cifar_image/NRF_unsupervised_{}.png'.format(args.suffix), bbox_inches='tight')
    plt.close(fig)

#evaluate inception score and FID for generator
def evaluate():

    samples_eval = []
    for i in range(50):
        samples = generator(Variable(torch.randn(100, h_dim).cuda()))
        samples=samples.cpu().data.numpy()
        samples_eval.append(samples)
    samples_eval = np.concatenate(samples_eval)

    if args.evalution_model=='tf':
        import evaluations.tf.inception_score as inception_score
        m, s = inception_score.get_inception_score(samples_eval)
        print('inception score mean and std::', m, s)
        sys.stdout.flush()
        import evaluations.tf.fid as fid
        samples_eval = ((samples_eval+1.)*(255./2)).astype('int32')
        samples_eval = samples_eval.transpose(0,2,3,1)
        FID=fid.calculate_fid_given_path(samples_eval, args.stat_flie_path)   #the stat file path should be changed to '../data/tf_stat_CIFAR.npz' if evaluation_model=tf
        print("FID:",FID)
        sys.stdout.flush()
    # to compare with SNGAN ([SPECTRAL NORMALIZATION FOR GENERATIVE ADVERSARIAL NETWORKS]),
    # we use their chainer code (https://github.com/pfnet-research/sngan_projection) to compute inception score and FID
    elif args.evalution_model=='chainer':
        samples_eval = np.clip(samples_eval * 127.5 + 127.5, 0.0, 255.0).astype(np.uint8).astype('f')
        import evaluations.chainer.evaluation as evaluation
        m, s = evaluation.calc_inception(samples_eval)
        print('inception score mean and std:', m, s)
        sys.stdout.flush()
        FID=evaluation.calc_FID(samples_eval,stat_file= args.stat_flie_path)
        print("FID:", FID)
        sys.stdout.flush()
    return m,FID

os.makedirs('cifar_model', exist_ok=True)

IS_all=[]
FID_all=[]
for it in range(args.max_iterations):
    train()
    if (it+1)%5000==0:
        generate()
        torch.save([RF.state_dict(),generator.state_dict()],  'cifar_model/NRF_unsupervised_it%d_%s.pkl'%(it+1,args.suffix))
        m,FID=evaluate()
        IS_all.append(m)
        FID_all.append(FID)
        print('iteration:%d best inception score:%.4f best FID:%.4f' % (it+1, max(IS_all), min(FID_all)))
        sys.stdout.flush()



