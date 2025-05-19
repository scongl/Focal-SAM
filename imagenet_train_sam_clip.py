import argparse
import os
import random
import time
import warnings
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import models
import os
from models import resnet
from models.resnet_cifar import NormedLinear
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
from utils import *
from datasets.imagenet_clip import ImageNetLTDataLoader
from losses import LDAMLoss, FocalLoss, LALoss
# import wandb
from sam import SAM
from scipy.interpolate import interp1d
from focal_sam import FocalSAM
from train_step import focal_sam_step

wandb = None


parser = argparse.ArgumentParser(description='PyTorch Cifar Training')
parser.add_argument('--dataset', default='cifar10', help='dataset setting')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50'),
parser.add_argument('--loss_type', default="CE", type=str, help='loss type')
parser.add_argument('--train_rule', default='None', type=str, help='data sampling strategy for train loader')
parser.add_argument('--rand_number', default=0, type=int, help='fix random number for data sampling')
parser.add_argument('--exp_str', default='0', type=str, help='additional info to indicate which experiment it is')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=2e-4, type=float,
                    metavar='W', help='weight decay (default: 2e-4)', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=str,
                    help='GPU id to use.')
parser.add_argument('--root_log',type=str, default='log')
parser.add_argument('--root_model', type=str, default='checkpoint')
parser.add_argument('--log_results', action='store_true',
                    help='to log results on wandb')
parser.add_argument('--rho', type=float, default=0.05)
parser.add_argument('--name', type=str, default='test')
parser.add_argument('--data_path', default='data/ImageNet_LT', type=str, metavar='PATH',
                    help='path to latest dataset ')
parser.add_argument('--cos_lr', action='store_true', help='Using cosine lr')
parser.add_argument('--end_lr_cos', default=0.0, type=float, metavar='M',
                    help='End lr for cos learning schedule')
parser.add_argument('--min_rho', type=float, default=0.05)
parser.add_argument('--max_rho', type=float, default=0.8)
parser.add_argument('--rho_schedule', default='none',
                    choices=('none','linear', 'step'))
parser.add_argument('--rho_steps', metavar = 'N', type = float, nargs = '+', help = 'rho values at various steps')
parser.add_argument('--margin', default=0.5, type=float, metavar='M',
                    help='Margin value for LDAM')
parser.add_argument('--drop_last', action='store_true')
parser.add_argument('--SAM_type', type=str, choices=('Focal-SAM', 'SAM', 'None'))
parser.add_argument('--flat_gamma', type=float, default=1.0)
parser.add_argument('--sharpness', type=float, default=0.5)
parser.add_argument('--save_freq', type=int, default=40)
parser.add_argument('--prec', type=str, default='fp16')
parser.add_argument('--resolution', type=int, default=224)
parser.add_argument('--full_tuning', action='store_true')
parser.add_argument('--bias_tuning', action='store_true')
parser.add_argument('--ln_tuning', action='store_true')
parser.add_argument('--vpt_shallow', action='store_true')
parser.add_argument('--vpt_deep', action='store_true')
parser.add_argument('--adapter', action='store_true')
parser.add_argument('--adaptformer', action='store_true')
parser.add_argument('--lora', action='store_true')
parser.add_argument('--lora_mlp', action='store_true')
parser.add_argument('--ssf_attn', action='store_true')
parser.add_argument('--ssf_mlp', action='store_true')
parser.add_argument('--ssf_ln', action='store_true')
parser.add_argument('--mask', action='store_true')
parser.add_argument('--partial', nargs='+', type=int, default=None)
parser.add_argument('--vpt_len', type=int, default=None)
parser.add_argument('--adapter_dim', type=int, default=None)
parser.add_argument('--mask_ratio', type=float, default=None)
parser.add_argument('--mask_seed', type=int, default=None)


best_acc1 = 0


def load_clip_to_cpu(backbone_name, prec):
    backbone_name = backbone_name.lstrip("CLIP-")
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cuda:0").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cuda:0").eval()

    model = clip.build_model(state_dict or model.state_dict()).cuda()

    assert prec in ["fp16", "fp32", "amp"]
    if prec == "fp32" or prec == "amp":
        # CLIP's default precision is fp16
        model.float()

    return model


def init_head_text_feat(args, classnames, model):
    def get_tokenized_prompts(classnames, template):
        prompts = [template.format(c.replace("_", " ")) for c in classnames]
        # print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.cuda()
        return prompts

    print("Initialize head with text features")
    template = "a photo of a {}."
    prompts = get_tokenized_prompts(classnames, template)

    text_features = model.encode_text(prompts)
    text_features = F.normalize(text_features, dim=-1)

    if args.arch.startswith("CLIP-ViT"):
        text_features = text_features @ model.image_encoder.proj.t()
        text_features = F.normalize(text_features, dim=-1)
    
    model.head.apply_weight(text_features)


def main(args):

    name_list = [args.dataset, args.arch, args.loss_type, args.train_rule, args.SAM_type, str(args.rho), 'lr', str(args.lr),'wd', str(args.weight_decay), 'seed', str(args.seed), args.exp_str]

    if args.SAM_type == "SAM" or args.SAM_type == "None":
        pass
    elif args.SAM_type == "Focal-SAM":
        name_list.extend(['flat_gamma', str(args.flat_gamma), 'sharpness', str(args.sharpness)])
    

    args.store_name = '_'.join(name_list)

    print("The args.store name is", args.store_name)
      
    prepare_folders(args)
    
    if os.path.exists(os.path.join(args.root_log, args.store_name, '{}_ckpt.pth.tar'.format(args.epochs))):
        print('Already trained')
        exit(0)
    
    if args.seed is not None:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    main_worker(args.gpu, args)


def main_worker(gpu, args):
    global best_acc1
    args.gpu = gpu
    

    if args.log_results:
        wandb.init(project="long-tail", name=args.store_name)
        wandb.config.update(args)
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
        
    # create model
    print("[INFORMATION] creating model '{}'".format(args.arch))
    num_classes = 1000

    # Data loading code
    if args.dataset == 'imagenet':
        data_loader = ImageNetLTDataLoader(data_dir=args.data_path, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, drop_last=args.drop_last)

        args.cls_num_list = data_loader.cls_num_list
        cls_num_list = data_loader.cls_num_list

    else:
        raise ValueError("Dataset not supported")

    train_loader = data_loader
    val_loader = data_loader.split_validation()
    cls_num_list = data_loader.cls_num_list


    many_shot_classes = []
    medium_shot_classes = [] 
    few_shot_classes = []
    
    for i in range(len(cls_num_list)):
        if cls_num_list[i] > 100:
            many_shot_classes.append(i)
        elif cls_num_list[i] < 20:
            few_shot_classes.append(i)
        else:
            medium_shot_classes.append(i)

    args.head_class_idx = many_shot_classes
    args.med_class_idx = medium_shot_classes
    args.tail_class_idx = few_shot_classes


    clip_model = load_clip_to_cpu(args.arch, args.prec)
    model = PeftModelFromCLIP(args, clip_model, num_classes)
    tuner = model.tuner
    head = model.head

    for name, param in model.named_parameters():
        param.requires_grad_(False)
    for name, param in tuner.named_parameters():
        param.requires_grad_(True)
    for name, param in head.named_parameters():
        param.requires_grad_(True)

    init_head_text_feat(args, data_loader.classnames, model)

    params = [{"params": tuner.parameters()}, {"params": head.parameters()}]

    model = model.cuda()
        
    if torch.cuda.device_count() > 1:
        print("[INFORMATION] Using ", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model).cuda()

    base_optimizer = torch.optim.SGD
    
    if args.SAM_type == 'SAM':
        optimizer = SAM(base_optimizer=base_optimizer, rho=args.rho, params=params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.SAM_type == 'Focal-SAM':
        optimizer = FocalSAM(params=params, base_optimizer=base_optimizer, rho=args.rho, adaptive=False,  lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(params, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.cos_lr == True:
        print("[INFORMATION] Using cosine lr_scheduler")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.end_lr_cos)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("[INFORMATION] loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)

            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.cuda()
            
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            
            if args.cos_lr == True:
                scheduler.load_state_dict(checkpoint['scheduler'])
            print("[INFORMATION] loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("[INFORMATION] no checkpoint found at '{}'".format(args.resume))

    # init log for training
    log_training = open(os.path.join(args.root_log, args.store_name, 'log_train.csv'), 'w')
    log_testing = open(os.path.join(args.root_log, args.store_name, 'log_test.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    
    
    for epoch in range(args.start_epoch, args.epochs):

        if args.train_rule == 'None':
            train_sampler = None  
            per_cls_weights = None
        elif args.train_rule == 'Reweight':
            train_sampler = None
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, cls_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
        elif args.train_rule == 'DRW':
            train_sampler = None
            idx = epoch // 160
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
        else:
            warnings.warn('Sample rule is not listed')


        if args.loss_type == 'CE':
            criterion = nn.CrossEntropyLoss(weight=per_cls_weights, reduction='none').cuda()
        elif args.loss_type == 'LDAM':
            print("[INFORMATION] LDAM is being used")
            print("[INFORMATION] margin value being used is ", args.margin)
            criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=args.margin, s=30, weight=per_cls_weights, reduction='none').cuda()
        elif args.loss_type == 'LA':
            criterion = LALoss(cls_num_list=cls_num_list, reduction='none').cuda()
        else:
            warnings.warn('Loss type is not listed')
            return

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, log_training, few_shot_classes, medium_shot_classes, many_shot_classes, cls_num_list)
        
        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, epoch, args, log_testing)

        if args.cos_lr == True:
            scheduler.step()
        
        if args.log_results:
            wandb.log({'epoch':epoch, 'val_acc':acc1})
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        # tf_writer.add_scalar('acc/test_top1_best', best_acc1, epoch)
        output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
        # print(output_best)
        log_testing.write(output_best + '\n')
        log_testing.flush()
        
        if args.cos_lr == True:
            save_checkpoint_new(args, {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, args.save_freq)

        else:
            save_checkpoint_new(args, {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, args.save_freq)

    if args.log_results:
        wandb.log({'best_acc':best_acc1})

def train(train_loader, model, criterion, optimizer, epoch, args, log, few_shot_class, medium_shot_class, many_shot_class, cls_num_list):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    # switch to train mode
    try: 
        model.tuner.train()
        model.head.train()
    except:
        model.module.tuner.train()
        model.module.head.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        if args.SAM_type == "Focal-SAM":
            output, loss = focal_sam_step(args, model, criterion, input, target, optimizer, cls_num_list)
        elif args.SAM_type == 'SAM':
            # compute output
            output = model(input)
            loss = criterion(output, target)
            loss = loss.mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.first_step(zero_grad=True)

            output = model(input)
            loss = criterion(output, target)
            loss = loss.mean()
            loss.backward()
            optimizer.second_step(zero_grad=True)
            
        else:
            # compute output
            output = model(input)
            loss = criterion(output, target)
            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        if args.log_results:
            wandb.log({'loss':loss, 'top1_acc':acc1, 'top5_acc':acc5})
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr']))
            print(output)
            log.write(output + '\n')
            log.flush()


def validate(val_loader, model, criterion, epoch, args, log=None, tf_writer=None, flag='val'):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    # switch to evaluate mode
    try:
        model.tuner.eval()
        model.head.eval()
    except:
        model.module.tuner.eval()
        model.module.head.eval()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
        
            if args.gpu is not None:
                input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input) #bs, num_classes
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.mean().item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            _, pred = torch.max(output, 1) #bs
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print(output)
        cf = confusion_matrix(all_targets, all_preds).astype(float)
        # print("[INFORMATION] The size of the cf is", cf.shape)
        
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf) #num of correct preds
        cls_acc = cls_hit / cls_cnt

        if args.dataset == 'imagenet':
            head_acc = cls_acc[args.head_class_idx].mean() * 100
            med_acc = cls_acc[args.med_class_idx].mean() * 100
            tail_acc = cls_acc[args.tail_class_idx].mean() * 100

            if args.log_results:
                wandb.log({'head_acc':head_acc, 'med_acc':med_acc, 'tail_acc':tail_acc})
        
        output = ('{flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
                .format(flag=flag, top1=top1, top5=top5, loss=losses))
        out_cls_acc = '%s Class Accuracy: %s'%(flag,(np.array2string(cls_acc, separator=',', formatter={'float_kind':lambda x: "%.3f" % x})))
        # print(output)
       
        # if args.dataset != 'imagenet':
            # print(out_cls_acc)

        if log is not None:
            log.write(output + '\n')
            log.write(out_cls_acc + '\n')
            log.flush()

    return top1.avg


def adjust_rho(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    epoch = epoch + 1
    
    if args.rho_schedule == 'step':
        if epoch < 0.8 * args.epochs:
            rho = args.rho_steps[0]
        else:
            rho = args.rho_steps[1]

        for param_group in optimizer.param_groups:
            param_group['rho'] = rho
        
        args.rho = rho

    elif args.rho_schedule == 'linear':
        X = [1, args.epochs]
        Y = [args.min_rho, args.max_rho]
        y_interp = interp1d(X, Y)
        rho = y_interp(epoch)

        for param_group in optimizer.param_groups:
            param_group['rho'] = np.float16(rho)

        args.rho = rho

    elif args.rho_schedule == 'none':
        rho = args.rho
        for param_group in optimizer.param_groups:
            param_group['rho'] = rho

    else:
        raise ValueError("The rho schedule is not supported")


if __name__ == '__main__':
    args = parser.parse_args()
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    from clip import clip
    from models_clip import PeftModelFromCLIP
    main(args)