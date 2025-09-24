import torch
import os , sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import numpy as np
 
from SL_CT_MR_data_pan_v1 import *
from DualNet_pan_online import  PrivilegedDistillModel
from DualNet_pretrain_v2 import (Dynamic_DistillationLoss, Static_DistillationLoss)
import argparse, datetime, logging, cv2, timeit
from torchvision import datasets, transforms

from tqdm import tqdm
from utils.loss import *
from utils.vis import *
from data_aug import *
from utils.configs import get_config
start = timeit.default_timer()

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def parse_option():
    """
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Model for 3D Medical Image Classification.")
    parser.add_argument('--cfg', 
                        type=str,
                        # required=True,
                        metavar="FILE", 
                        default='.yaml',
                        help='path to config file', )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--start_iters", type=int, default=0)
    parser.add_argument("--start_save_iter", type=int, default=8000)
    
    parser.add_argument("--weight_std", type=str2bool, default=True)
    parser.add_argument("--momentum", type=float, default=0.9)
     
    parser.add_argument("--weight_decay", type=float, default=0.0005)

    parser.add_argument("--random_seed", type=int, default=999)
    parser.add_argument("--print_freq", type=int, default=10)

    parser.add_argument("--norm_cfg", type=str, default='GN')  # normalization
    parser.add_argument("--activation_cfg", type=str, default='LeakyReLU')  # activation
    parser.add_argument("--train_only", action="store_true")
    parser.add_argument("--mode", type=str, default='0,1,2')
    # parser.add_argument("--input_size", type=str, default='224,224,32')
    # todo add
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument('--start_train_epoch', type=int, default=0) 
    parser.add_argument('--start_compute_train', type=int, default=20) 
    parser.add_argument("--val_every_iter", type=int, default=20)
    parser.add_argument("--task", type=str, default='3cls_bm_CT-T1-T2')
    
    parser.add_argument("--trans", type=str, default='center_crop')
    
    parser.add_argument("--resume", type=str, default=None)
    
    parser.add_argument("--num_classes_texture", type=int, default=3)
    parser.add_argument("--num_classes_duct", type=int, default=3)
    
    parser.add_argument("--device", type=str, default='cuda')
    
    parser.add_argument('--init', default='k', help='resume from checkpoint',type=str) 
    parser.add_argument("--power", type=float, default=0.9)
    
    # todo important 
    parser.add_argument("--p_mode", type=str, default='T1')
    parser.add_argument("--best_val_acc", type=float, default= 0.7) 
    
    parser.add_argument("--feature_dim", type=int, default=256) # encoder dim
    parser.add_argument("--optimizer", type=str, default='adam') 
    parser.add_argument("--base_lr", type=float, default=1e-4) 
    parser.add_argument("--lr_mode", type=str, default='reduce')
    parser.add_argument("--lr_set", type=str, default='5-0.1')
    
    parser.add_argument("--warmup_epoch", type=int, default=0) # todo start distilling
    # parser.add_argument("--kd_w", type=float, default=0.5)
    parser.add_argument("--kd_temp", type=float, default=1.0)
    parser.add_argument("--kd_mode", type=str, default='d')
    # parser.add_argument("--fkd_mode", type=str, default='kl')
    
    parser.add_argument("--fkd_w", type=float, default=1.5)
    parser.add_argument("--fkd_mode", type=str, default='kl') # kl ; info
    parser.add_argument("--rkd_w", type=float, default=0.5)
    parser.add_argument("--update_mode", type=str, default='ema') # relative ; eam; combined ; exp
    
    parser.add_argument("--auxi_w", type=float, default=0.1)
    
    parser.add_argument("--num_heads", type=int, default=5)
    parser.add_argument("--head_indice", type=int, default=3)
    parser.add_argument('--add_tag', default='', help='resume from checkpoint',type=str)
    parser.add_argument("--test_code", type=int, default=1)
    parser.add_argument("--vis", type=int, default=0) 
    parser.add_argument("--vis_per_epoch", type=int, default=1)
    parser.add_argument("--input_size", type=str, default='224,224,32')
    parser.add_argument("--mr_size", type=str, default='224,224,32')
    parser.add_argument("--texture_net", type=str, default='mobile')
    parser.add_argument("--duct_net", type=str, default='mobile')
    parser.add_argument("--ct_net", type=str, default='mobile')
    
    parser.add_argument("--clinical", type=int, default=0) 
    parser.add_argument("--clinical_w1", type=float, default=1.0) 
    parser.add_argument("--clinical_w2", type=float, default=1.0) 
    parser.add_argument("--clinical_mode", type=str, default='mse') 
    
    args, unparsed = parser.parse_known_args()
    args.p_mode = args.p_mode.upper()
    args.num_classes = int(args.task.split('cls')[0])
    config = get_config(args.cfg)
    args.snapshot_dir = config['data']['snapshot_dir']
    args.root_path = config['data']['root_path']
    
    return args,  parser

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter, lr, num_steps, power):
    lr = lr_poly(lr, i_iter, num_steps, power)
    optimizer.param_groups[0]['lr'] = lr
    return lr

def train(model, trainloader, valloader, loss_fn, optimizer,  args, writer,scheduler=None):
    model.train()
    total_loss = 0
    pred_list = []
    target_list = []
    prob_list = []
    if args.clinical != 1:
        loss_details = {k: 0 for k in ['loss_main_ct',  'loss_auxi_cls','loss_texture', 'loss_duct', 'loss_distill', 'loss_fkd','loss_pkd', 'weights']}
    else:
        loss_details = {k: 0 for k in ['loss_main_ct', 'loss_main_hard', 'num_fixed_by_aux','loss_auxi_cls','loss_texture', 'loss_duct', 'loss_distill', 'loss_fkd','loss_pkd', 'weights']}
    
    all_mr_direct_feats = []
    all_mr_indirect_feats = []
    
    for i_iter, batch in enumerate(trainloader):
        
        idx, ct  = batch['idx'], batch['ct']
        
        if len(args.p_mode) == 2:
            mr = [batch[args.p_mode.lower()].to(args.device) ]
        else:
            mr = [batch['t1'].to(args.device), batch['t2'].to(args.device)]
        ct = ct.to(args.device)
        # mr = mr.to(args.device)
        labels = {'cls':batch['cls'].cuda(), 'texture':batch['texture'].cuda(), 'duct':batch['duct'].cuda()}
        ids = batch['ID']
        
        optimizer.zero_grad()
        if scheduler is None:
            lr = adjust_learning_rate(optimizer, args.now_iteration, args.base_lr, args.max_iterations, args.power)
        else:
            lr = optimizer.param_groups[0]['lr']
        
        outputs = model([ct, mr], head_indice=args.head_indice)
        #todo cmopute train performance
        if args.curr_epoch >= args.start_compute_train and args.curr_epoch % 5 == 0:
            probs = outputs['logits_ct'][-1].softmax(dim=-1)
            preds = torch.argmax(probs,dim=1) 
            
            pred_list.append(preds)
            target_list.append(labels['cls'])
            prob_list.append(probs)
 
        loss, batch_loss_details = loss_fn(outputs, labels, idx)
        
        if args.curr_epoch >= args.start_compute_train and args.curr_epoch % 5 == 0:
            train_results = evaluate_multiclass_metrics(prob_list, pred_list, target_list)
            log_val_results(train_results, args.now_iteration + 1, name = 'Train')
            for i in train_results['metrics'].keys():
                writer.add_scalars('Train_acc', {i:train_results['metrics'][i]['acc']}, args.curr_epoch)
                
        loss.backward()
        optimizer.step()
        # current_lr = get_lr(optimizer)
        writer.add_scalars('LR', {'lr':lr}, args.now_iteration)
        total_loss += loss.item()
        for k in loss_details:
            loss_details[k] += batch_loss_details.get(k, 0)
            
        if args.test_code == 1:
            validate_iter = 0
        else:
            validate_iter = 500
        if args.now_iteration >= validate_iter :
            if (args.now_iteration + 1) % args.val_every_iter == 0:
            
                val_results = validate(model, valloader, args)
                val_acc_avg =  val_results['metrics']['avg']['acc'] 
                if scheduler is not None:
                    scheduler.step(val_acc_avg)
                log_val_results(val_results, args.now_iteration + 1, name = 'Val')
                for i in val_results['metrics'].keys():
                    writer.add_scalars('Val_acc', {i:val_results['metrics'][i]['acc']}, args.now_iteration)
                    
                if val_acc_avg > args.best_val_acc:
                    args.best_val_acc = val_acc_avg
                    args.best_iteration = args.now_iteration
                    
                    checkpoint = {'model':model.state_dict(),
                                    # 'scheduler': scheduler.state_dict(),
                                    'optimizer': optimizer.state_dict(),
                                    'iter': args.now_iteration
                    }
                    
                    torch.save(checkpoint, os.path.join(args.chpk_dir,f"{args.tag}_best_val_Iter{args.now_iteration}_acc{args.best_val_acc:.2f}.pth"))
                    
                    logging.info(f'Save Iter_{args.now_iteration} Val model ; Acc is {args.best_val_acc} ...')
        model.train()
        args.now_iteration += 1
    
    avg_loss = total_loss / len(trainloader)
    for k in loss_details:
        if k == 'num_fixed_by_aux':
            loss_details[k] /= 1
        else:
            loss_details[k] /= len(trainloader)
    
    if args.clinical == 1:
        # rectify_num +=batch_loss_details['num_fixed_by_aux'] 
        logging.info(f'=====>>>>> Training completed {args.curr_epoch} epochs; Best val acc is {args.best_val_acc} in iter_{args.best_iteration} ; Rectify num : {loss_details["num_fixed_by_aux"]}<<<<<======')
    else:
        logging.info(f'=====>>>>>Training completed {args.curr_epoch} epochs; Best val acc is {args.best_val_acc} in iter_{args.best_iteration} <<<<<======')
    
    return avg_loss, loss_details
  
def validate(model, dataloader, args):
    model.eval()
    correct_main = 0
    total = 0
    pred_list = []
    target_list = []
    prob_list = []
    
    with torch.no_grad():
        for i_iter, batch in enumerate(dataloader):
            id, ct = batch['ID'], batch['ct']
            labels = {'cls':batch['cls'].cuda(), 'texture':batch['texture'].cuda(), 'duct':batch['duct'].cuda()}
            ct = ct.to(args.device)
            # mr = mr.to(args.device)
            
            # 测试时只使用CT图像
            outputs = model([ct], ct_only =True)['logits_ct'][-1]
            outputs = outputs.softmax(dim=-1)
            pred = torch.argmax(outputs,dim=1) 
            
            pred_list.append(pred)
            target_list.append(labels['cls'])
            prob_list.append(outputs)
            
        results = evaluate_multiclass_metrics(prob_list, pred_list, target_list)
    return results

def generate_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)   
def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']   
 
def main():
    now_time = datetime.datetime.now().strftime("%m-%d-%H-%M")
    args, parser = parse_option()
    args.patch_size = [int(x) for x in args.input_size.split(',')] 
    args.mr_size = [int(x) for x in args.mr_size.split(',')] 
    model_name = f'Online_KD_CT_{args.ct_net}_{args.mr_size[0]}_{args.texture_net}_{args.duct_net}'

    args.tag = f'{args.p_mode}_{args.task.split("_CT")[0]}_head{args.num_heads}{args.head_indice}_{args.clinical_tag}_{args.auxi_w}Auxi_{args.kd_tag}_{args.club_tag}_{args.optimizer}_Lr_{args.base_lr}_{args.lr_tag}_{now_time}{args.add_tag}'
    
    if args.resume:
        resume_tag = args.resume.split('/')[-1].split('.pth')[0]
        if args.test_code == 0:
            args.tag = 'Resume_from_' + resume_tag 
        else:
            args.tag = 'Test==========Resume_from_' + resume_tag 
        
    snapshot_dir = os.path.join(args.snapshot_dir, model_name)
    generate_folder(snapshot_dir)
    generate_folder(os.path.join(snapshot_dir,'log'))
    args.chpk_dir = os.path.join(snapshot_dir,'checkpoint')
    args.vis_dir = os.path.join(snapshot_dir,'vis')
    generate_folder(args.chpk_dir)
    generate_folder(args.vis_dir)
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
 
    logging.basicConfig(filename = os.path.join(snapshot_dir,'log') + f"/{args.tag}.txt", level=logging.INFO,   # TODO logging file path
                                format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    s = print_options(args, parser)
    # logging.info(s)
    logging.info(f'=========== {args.tag}===========')
    tb_dir = os.path.join(snapshot_dir,f'tb')
    tb_tag_dir = os.path.join(tb_dir, args.tag)
    generate_folder(tb_tag_dir)
    writer = SummaryWriter(tb_tag_dir)
    # todo load data
    if args.trans == "random_crop":
        trans = transforms.Compose([RandomRotFlip(), RandomCrop(args.patch_size),ToTensor(), ])
        mr_trans = transforms.Compose([RandomRotFlip(), RandomCrop(args.mr_size),ToTensor(), ])
    elif args.trans == "center_crop":
        trans = transforms.Compose([RandomRotFlip(),CenterCrop(args.patch_size),ToTensor(), ])
        mr_trans = transforms.Compose([RandomRotFlip(), CenterCrop(args.mr_size),ToTensor(), ])
    elif args.trans =='center_aug':
        trans = transforms.Compose([RandomRotFlip(),CenterCrop(args.patch_size),  ToTensor(), Random_Bright_rotation((0.8, 1.2), (-15, 15)), ])
        trans = transforms.Compose([RandomRotFlip(),CenterCrop(args.mr_size),  ToTensor(), Random_Bright_rotation((0.8, 1.2), (-15, 15)), ])
    elif args.trans =='random_aug':
        trans = transforms.Compose([RandomRotFlip(),RandomCrop(args.patch_size),  ToTensor(), Random_Bright_rotation((0.8, 1.2), (-15, 15)), ])
        trans = transforms.Compose([RandomRotFlip(),RandomCrop(args.mr_size),  ToTensor(), Random_Bright_rotation((0.8, 1.2), (-15, 15)), ])
    elif args.trans =='random_crop2':   
        trans = transforms.Compose([RandomRotFlip(), RandomCrop2(args.patch_size),ToTensor(), ])    
        trans = transforms.Compose([RandomRotFlip(), RandomCrop2(args.mr_size),ToTensor(), ])         
    val_trans = transforms.Compose([CenterCrop(args.patch_size),ToTensor(), ])
    val_mr_trans = transforms.Compose([CenterCrop(args.mr_size),ToTensor(), ])
    
    train_dataset = CLS_202503(args,  split ='train',transform=trans, mr_transform = mr_trans)
    val_dataset = CLS_202503(args, split ='val',transform=val_trans, mr_transform = val_mr_trans)
    
    trainloader = torch.utils.data.DataLoader(train_dataset, 
                                batch_size=args.batch_size,
                                shuffle=True, 
                                num_workers=0, 
                                pin_memory=False,
                                drop_last = True
                                )
    valloader = torch.utils.data.DataLoader(val_dataset, 
                                batch_size=2, 
                                shuffle=False, 
                                num_workers=0,
                                pin_memory=False,
                                drop_last = True
                                )
    
    # todo load model 
    model = PrivilegedDistillModel(args).cuda()
    
    def init_weights(m):
        if isinstance(m, nn.Conv3d):
            # logging.info(f'==========={args.init.lower()} Init weights============')
            if args.init.lower() == 'x':
                torch.nn.init.xavier_normal_(m.weight)
            elif args.init.lower() == 'k':
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    # todo loss     
    if args.kd_mode == 'dynamic'  :    
        loss_fn = Dynamic_DistillationLoss(args, len(train_dataset), update_mode = args.update_mode)
    else:
        loss_fn = Static_DistillationLoss(args)
    
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.base_lr, alpha=0.99)
    elif args.optimizer == 'sgd':
        # optimizer = torch.optim.SGD(model.parameters(), lr = args.learning_rate)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.99, nesterov=True)
    elif args.optimizer == "adamW" :
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr)    
        
    if args.lr_mode == 'multistep':
        step_list = [int(x) for x in  args.lr_set.split('-') ] 
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=step_list, gamma=0.1)
    elif args.lr_mode == 'cosine':
        step_list = [int(x) for x in  args.lr_set.split('-') ] 
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=step_list[0], T_mult=step_list[1])    
    elif args.lr_mode == 'reduce':
        step_list =  [int(x) if float(x) > 1 else float(x) for x in args.lr_set.split('-')]
        scheduler = ReduceLROnPlateau(optimizer, 'max', patience=step_list[0], factor=step_list[1])

    if args.resume:
        logging.info('loading from checkpoint: {}'.format(args.resume))
        if os.path.exists(args.resume):
            checkpoint = torch.load(args.resume, weights_only=True)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_iters = checkpoint['iter']
            # scheduler.load_state_dict(checkpoint['scheduler'])
            logging.info(f"Loaded model trained for {args.start_iters} iters")
        else:
            logging.info('File not exists in the reload path: {}'.format(args.resume))
            exit(0)
    else:
        model.apply(init_weights)
        if args.club == 1:
            club_estimator.apply(init_weights)
    
    start_train_epoch = args.start_iters // len(trainloader) 
    args.max_iterations = args.epochs * len(trainloader) 
    args.now_iteration = args.start_iters
    logging.info('Start training from epoch: %d' % start_train_epoch)
    # val_acc_best = 0.0
    args.best_iteration = 0
    for epoch in tqdm(range(start_train_epoch, args.epochs), ncols=50):
        args.curr_epoch = epoch
        
        # todo train 
        if args.lr_mode != 'no':
            train_loss, train_details = train(model, trainloader, valloader, loss_fn, optimizer,  args, writer, scheduler=scheduler )
        else:
            train_loss, train_details = train(model, trainloader, valloader, loss_fn, optimizer,  args, writer)
        
        log_msg = (
            f"[Epoch {epoch+1:03d}/{args.epochs}]\n"
            f"Train Loss: {train_loss:.4f}\n"
            f"  ├─ CT cls loss       : {train_details['loss_main_ct']:.4f}\n"
            f"  ├─ Auxi cls loss     : {train_details['loss_auxi_cls']:.4f}\n"
            f"  │     ├─ Texture     : {train_details['loss_texture']:.4f}\n"
            f"  │     └─ Duct        : {train_details['loss_duct']:.4f}\n"
        )


        log_msg += (
            f"  └─ Distill loss      : {train_details['loss_distill']:.4f} ; [weight : {train_details['weights']:.4f}]\t"
            f"(FKD: {train_details.get('loss_fkd', 0.0):.4f}, PKD: {train_details.get('loss_pkd', 0.0):.4f})"
        )

        logging.info(log_msg)
        
        for i in train_details.keys():
            writer.add_scalars('Train', {i:train_details[i]}, epoch)
        # todo validate
        if args.curr_epoch +1 ==  args.epochs:
            val_results = validate(model, valloader, args)
            val_acc_avg =  val_results['metrics']['avg']['acc'] 
            for i in val_results['metrics'].keys():
                writer.add_scalars('Val_acc', {i:val_results['metrics'][i]['acc']}, args.now_iteration)
            log_val_results(val_results, args.now_iteration + 1, name = 'Val')

            if val_acc_avg > args.best_val_acc:
                args.best_val_acc = val_acc_avg
                checkpoint = {'model':model.state_dict(),
                                #   'scheduler': scheduler.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'iter': args.now_iteration
                }
                
                torch.save(checkpoint, os.path.join(args.chpk_dir,f"{args.tag}_val_Iter{args.now_iteration}_acc{args.best_val_acc:.2f}.pth"))
                
                logging.info(f'save Iter_{args.now_iteration} val model (acc is {args.best_val_acc}) ...')
    end = timeit.default_timer()
    logging.info(f"Training completed (take {end - start} seconds). Best validation accuracy: {args.best_val_acc:.2f}%")

if __name__ == "__main__":
    
    main()
