def main(num,seed,args):
    import torch
    torch.backends.cudnn.deterministic = True
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.optim.lr_scheduler import ExponentialLR
    from torchvision import datasets, transforms
    from torch.autograd import Variable
    
    import numpy as np
    
    import os,sys


    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    
    #处理训练数据，只保留num这一类
    dst=datasets.CIFAR10('../../data/cifar-10-python/', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    dst.train_labels=np.array(dst.train_labels)
    dst.train_data=dst.train_data[dst.train_labels==num]
    dst.train_labels=dst.train_labels[dst.train_labels==num]
    print(len(dst))
    loader = torch.utils.data.DataLoader(
        dst,batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    dev_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../../data/cifar-10-python/', train=False, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),batch_size=1000)
    
    Z_dim = 100
    #number of updates to discriminator for every update to generator 
    
    import model
    discriminator = model.Discriminator().cuda()
    generator = model.Generator(Z_dim).cuda()
    
    
    print(discriminator)
    sys.stdout.flush()


    if args.opt=='adam':
        optim_disc = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lrd, betas=(0.5,0.999))
        optim_gen  = optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()), lr=args.lrg, betas=(0.5,0.999))
    elif args.opt=='rms':
        optim_disc = optim.RMSprop(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lrd)
        optim_gen = optim.RMSprop(filter(lambda p: p.requires_grad, generator.parameters()), lr=args.lrg)

    
    def get_rev(gen_images):
        x_sam = gen_images.detach()+args.sig*torch.randn(gen_images.size(0),3,32,32).cuda()
        x_sam = Variable(x_sam, requires_grad=True)
        v = 0

        for i in range(args.L):
            loss_s = torch.sum(discriminator(x_sam))
            gra = torch.autograd.grad(outputs=loss_s, inputs=x_sam)[0]
            v = v * args.alpha + args.eta * gra
            x_sam = x_sam + v + torch.randn(x_sam.size(0),3,32,32).cuda()* args.cof
            x_sam.data.clamp_(-1, 1)


        return x_sam.detach()
    
    def train(epoch):
        discriminator.train()
        for batch_idx ,(data,target) in enumerate(loader):
    
            data=data.cuda()
            z = Variable(torch.randn(args.batch_size, Z_dim).cuda())
            gen_x=generator(z)
            x_sam=get_rev(gen_x)
    
            unl_logits = discriminator(data)
            sam_logits = discriminator(x_sam)
            del_loss = torch.mean(sam_logits) - torch.mean(unl_logits)
            unl_loss = del_loss * args.del_we + args.fxp * torch.mean(unl_logits ** 2)
    
            optim_disc.zero_grad()
            unl_loss.backward()
            optim_disc.step()
    
    
            g_loss=torch.sum((x_sam-gen_x)**2)*args.gw

            optim_gen.zero_grad()
            g_loss.backward()
            optim_gen.step()
    


    def eval():
        # 测试模型在测试集上的AUROC指标
        discriminator.eval()
        test_score=[]
        real_labels=[]
        for idx, (data,target) in enumerate(dev_loader):
            data=data.cuda()
            pred=discriminator(data)
            test_score.append(pred.cpu().data.numpy())
            real_labels.append(target.numpy())
        test_score=np.concatenate(test_score,0)
        real_labels=np.concatenate(real_labels,0)
        from sklearn.metrics import roc_auc_score
    
        test_err = roc_auc_score(real_labels == num, test_score)
        return test_err
    
    os.makedirs('cifar_model', exist_ok=True)
    
    import time
    begin=time.time()
    if args.load:
        dp,gp=torch.load('cifar_model/jrf_sn_cnn_'+args.load+'.pkl')
        discriminator.load_state_dict(dp)
        generator.load_state_dict(gp)
    auc_all=[]
    
    for epoch in range(args.max_e):
        train(epoch)
        auc=eval()
        auc_all.append(auc)
        print ('epoch: %d time cost: %.2f auc: %.6f best:%.6f'%(epoch+1,time.time()-begin,auc,max(auc_all)))
        sys.stdout.flush()
        begin=time.time()
        if (epoch+1)%50==0:
            if not os.path.exists('cifar_model'):
                os.mkdir('cifar_model')
            torch.save([discriminator.state_dict(),generator.state_dict()],"cifar_model/cifar_dec_num%d_seed%d_%s"%(num,seed,args.sf))

    return max(auc_all)


