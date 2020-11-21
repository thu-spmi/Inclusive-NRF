
def main(seed=None,args=None):
    import numpy as np

    if seed!=None:
        args.seed=seed
        print("seed:",args.seed)

    import torch
    torch.backends.cudnn.benchmark = True
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.autograd import Variable

    torch.manual_seed(args.seed)            # 为CPU设置随机种子
    torch.cuda.manual_seed(args.seed)       # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(args.seed)   # 为所有GPU设置随机种子

    import model

    data_size=118
    Z_dim=5

    dis=model.Dis(data_size).cuda()
    gen=model.Gen(data_size,Z_dim).cuda()

    optim_disc = optim.Adam(filter(lambda p: p.requires_grad, dis.parameters()), lr=args.lrd, betas=(args.beta1,args.beta2))
    optim_gen  = optim.Adam(filter(lambda p: p.requires_grad, gen.parameters()), lr=args.lrg, betas=(args.beta1,args.beta2))

    import data_loader
    loader,dev_loader=data_loader.get_loader('kdd_cup.npz',args.batch_size,np.random.RandomState(args.seed))

    def get_rev(gen_images):
        x_sam = gen_images.detach() + torch.randn(gen_images.size()).cuda()* args.sig
        x_sam = Variable(x_sam, requires_grad=True)
        v = 0
        for i in range(args.L):
            loss_s = torch.sum(dis(x_sam))
            gra = torch.autograd.grad(outputs=loss_s, inputs=x_sam)[0]

            v = v * args.alpha + args.eta * gra
            x_sam = x_sam + v + torch.randn(x_sam.size()).cuda()* args.cof
        return x_sam.detach()


    def train():
        for batch_idx , (data,label) in enumerate(loader):
            data=data.cuda()
            z = Variable(torch.randn(args.batch_size, Z_dim).cuda())
            gen_x=gen(z)
            x_sam=get_rev(gen_x)

            unl_logits = dis(data)
            sam_logits = dis(x_sam)
            del_loss = torch.mean(sam_logits) - torch.mean(unl_logits)
            unl_loss = del_loss * args.del_we + args.fxp * torch.mean(unl_logits ** 2)

            optim_disc.zero_grad()
            unl_loss.backward()
            optim_disc.step()

            loss_g = torch.sum((x_sam-gen_x)**2)

            optim_gen.zero_grad()
            loss_g.backward()
            optim_gen.step()

    def eval():
        test_energy = []
        test_labels = []
        for it, (input_data, labels) in enumerate(dev_loader):
            input_data = input_data.cuda()
            pred = dis(input_data)
            test_energy.append(pred.data.cpu().numpy())
            test_labels.append(labels.numpy())

        test_energy = np.concatenate(test_energy, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)
        test_energy=-test_energy
        thresh = np.percentile(test_energy, 80)
        pred = (test_energy > thresh).astype(int)
        gt = test_labels.astype(int)
        from sklearn.metrics import precision_recall_fscore_support as prf
        precision, recall, f_score, _ = prf(gt, pred, average='binary')
        return precision, recall, f_score


    pre_all=[]
    rec_all=[]
    f1_all=[]


    for ep in range(args.max_e):
        train()
        pre,rec,f1=eval()
        pre_all.append(pre)
        rec_all.append(rec)
        f1_all.append(f1)
        print(ep+1,(f1,pre,rec),max(zip(f1_all,pre_all,rec_all)))
        import sys
        sys.stdout.flush()
        if (ep+1)%10==0:
            import os
            os.makedirs("kdd_model",exist_ok=True)

            torch.save([dis.state_dict(),gen.state_dict()],"kdd_model/kdd_seed%d_%s"%(seed,args.sf))
    return max(zip(f1_all,pre_all,rec_all))
