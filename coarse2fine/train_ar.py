import gc
import os
import sys
import dist
import time
import torch
import shutil
import warnings
import torch.nn as nn
from functools import partial
from utils import arg_util, misc
from utils.misc import auto_resume
from torch.utils.data import DataLoader
from utils.data_sampler import DistInfiniteBatchSampler, EvalDistributedSampler
from train_util import load_shape_meta,load_sep_vae_model,load_obs_encoder,load_realworld_image_dataset

class NullDDP(torch.nn.Module):
    """
    :func: the ddp for non-distributed
    """
    def __init__(self, module, *args, **kwargs):
        super(NullDDP, self).__init__()
        self.module = module
        self.require_backward_grad_sync = False
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
def build_everything(args: arg_util.Args):
    """
    :func: build everthing for training process
    """
    
    ### resume the model
    auto_resume_info, start_ep, start_it, trainer_state, args_state = auto_resume(args, 'ar-ckpt*.pth')
    
    ### create tensorboard logger
    tb_lg: misc.TensorboardLogger
    with_tb_lg = dist.is_master()
    if with_tb_lg:
        os.makedirs(args.tb_log_dir_path, exist_ok=True)
        # noinspection PyTypeChecker
        tb_lg = misc.DistLogger(misc.TensorboardLogger(log_dir=args.tb_log_dir_path, filename_suffix=f'__{misc.time_str("%m%d_%H%M")}'), verbose=True)
        tb_lg.flush()
    else:
        # noinspection PyTypeChecker
        tb_lg = misc.DistLogger(None, verbose=False)
    dist.barrier()
    
    ### log args
    print(f'global bs={args.glb_batch_size}, local bs={args.batch_size}')
    print(f'initial args:\n{str(args)}')
    
    ### build data
    print(f'[build PT data] ...\n')
    
    ### 💦 load all of the data | actions has shape [16,act_dim] 💦
    shape_meta = load_shape_meta()
    dataset_train, dataset_val, normalizer = load_realworld_image_dataset(args, shape_meta, n_obs_steps=args.tnobs)
    types = str((type(dataset_train).__name__, type(dataset_val).__name__))    
    
    ## build the distributed validation dataset
    ld_val = DataLoader(
        dataset_val, # DatasetFolder obj
        num_workers=0,
        pin_memory=True,
        batch_size=round(args.batch_size*1.5),
        # distributed sample
        sampler=EvalDistributedSampler(
            dataset_val, 
            num_replicas=dist.get_world_size(), 
            rank=dist.get_rank()),
        shuffle=False, 
        drop_last=False,
    )
    del dataset_val
    
    ## build the distributed training dataset
    ld_train = DataLoader(
        dataset=dataset_train, # DatasetFolder obj
        num_workers=args.workers,
        pin_memory=True,
        generator=args.get_different_generator_for_each_rank(), # worker_init_fn=worker_init_fn,
        # distributed sample
        batch_sampler=DistInfiniteBatchSampler(
            dataset_len=len(dataset_train), 
            glb_batch_size=args.glb_batch_size, # the global batch size which contains all of the gpus
            same_seed_for_all_ranks=args.same_seed_for_all_ranks, # the seed of all of the ranks
            shuffle=True, 
            fill_last=True, 
            rank=dist.get_rank(), # the rank
            world_size=dist.get_world_size(), # the total number of gpus
            start_ep=start_ep, 
            start_it=start_it, 
        ),
    )
    del dataset_train
    
    [print(line) for line in auto_resume_info]
    print(f'[dataloader multi processing] ...', end='', flush=True)
    stt = time.time()
    iters_train = len(ld_train)
    ld_train = iter(ld_train)
    # noinspection PyArgumentList
    print(f'[dataloader multi processing](*) finished! ({time.time()-stt:.2f}s)', flush=True, clean=True)
    print(f'[dataloader] gbs={args.glb_batch_size}, lbs={args.batch_size}, iters_train={iters_train}, types(tr, va)={types}')
    
    ### build models
    from torch.nn.parallel import DistributedDataParallel as DDP
    from vqvae.vqvae import VQVAE
    from trainer_ar import VARTrainer
    from optim.amp_opt import AmpOptimizer
    from optim.lr_control import filter_params
    from autoreg import build_vae_var
    from autoreg import VAR
    
    ### load obs encoder
    obs_encoder = load_obs_encoder(shape_meta)
    
    ### load vae and var
    vae_local, var_wo_ddp = build_vae_var(
        device=dist.get_device(),
        patch_nums=args.patch_nums,
        ## VAE
        V=args.vocab_size, 
        Cvae=args.vocab_ch, 
        ch=args.vch, 
        num_actions=16,
        dropout=args.vdrop,
        beta=args.vqbeta,
        using_znorm=args.vqnorm,
        quant_conv_ks=3,
        quant_resi=args.vqresi,
        share_quant_resi=4,
        ## VAR
        obs_encoder = obs_encoder,      # 🔥 🔥 🔥
        depth=args.tdepth,              # 🔥 🔥 🔥
        n_obs_steps=args.tnobs,         # 🔥 🔥 🔥
        embed_dim=args.tembed,          # 🔥 🔥 🔥
        shared_aln=args.saln,           # whether to use shared adaln
        attn_l2_norm=args.anorm,        # whether to use L2 normalized attention
        init_adaln=args.taln,           # for var
        init_adaln_gamma=args.talng,    # for var
        init_head=args.thd,             # for var
        init_std=args.tini,             # for var
    )
    
    ### load the model of vae
    if hasattr(vae_local, '_orig_mod'):
        vae_local = vae_local._orig_mod
    vae_local = load_sep_vae_model(vae_local, args.task_name)
    
    ### load models
    vae_local: VQVAE = args.compile_model(vae_local, args.vfast)    #
    var_wo_ddp: VAR = args.compile_model(var_wo_ddp, args.tfast)    #
    assert all(p.requires_grad is False for p in vae_local.parameters())
    assert all(p.requires_grad is True for p in var_wo_ddp.parameters())
    
    ### multi-gpu training | load model for each gpu
    var: DDP = (DDP if dist.initialized() else NullDDP)(var_wo_ddp, device_ids=[dist.get_local_rank()], find_unused_parameters=False, broadcast_buffers=False)
    
    ### showcase
    print(f'[INIT] VAR model = {var_wo_ddp}\n\n')
    count_p = lambda m: f'{sum(p.numel() for p in m.parameters())/1e6:.4f}M'
    print(f'[INIT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (('VAE', vae_local), 
                                                                          ('VAE.enc', vae_local.encoders), 
                                                                          ('VAE.dec', vae_local.decoders), 
                                                                          ('VAE.quant', vae_local.quantizers))]))
    print(f'[INIT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (('VAR', var_wo_ddp),
                                                                          ('VAR.enc', var_wo_ddp.obs_encoder))]) + '\n\n')
    
    ### construct the params for building optimizer
    names, paras, para_groups = filter_params(var_wo_ddp, nowd_keys={
        ###
        'cls_token', 
        'start_token', 
        'task_token', 
        'cfg_uncond',
        'pos_embed', 
        'pos_1LC', 
        'pos_start', 
        'start_pos', 
        'lvl_embed',
        'gamma', 
        'beta',
        'ada_gss', 
        'moe_bias',
        'scale_mul',
        ### 
        'class_emb', 
        'embedding',
        'norm_scale',
    })
    opt_clz = {
        'adam':  partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=args.afuse),
        'adamw': partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=args.afuse),
    }[args.opt.lower().strip()]
    opt_kw = dict(lr=args.tlr, weight_decay=0)
    print(f'[INIT] optim={opt_clz}, opt_kw={opt_kw}\n')
    
    ### build the optimizer
    var_opt = AmpOptimizer(
        mixed_precision=args.fp16, 
        optimizer=opt_clz(params=para_groups, **opt_kw), 
        names=names,
        paras=paras,
        grad_clip=args.tclip, # whether to utilize the gradient clipping
        n_gradient_accumulation=args.ac
    )
    del names, paras, para_groups
    
    ### build trainer
    trainer = VARTrainer(
        device=args.device, # cpu or gpu
        patch_nums=args.patch_nums, # pns
        resos=args.resos, # resolutions | pns * patch_size
        vae_local=vae_local, # vae
        var_wo_ddp=var_wo_ddp, # var
        var=var, # ddp'version var
        var_opt=var_opt, # optimizer 
        label_smooth=args.tls, # smooth the labels in CrossEntropyLoss to avoid overfitting
        normalizer=normalizer,
    )
    if trainer_state is not None and len(trainer_state):
        trainer.load_state_dict(trainer_state, strict=False, skip_vae=True) # don't load vae again
    del vae_local, var_wo_ddp, var, var_opt
    
    ### synchronize each distribution
    dist.barrier()
    
    ### retrun the building results
    return (
        tb_lg,          # the tensorboard object
        trainer,        # the trainer
        start_ep,       # the start idx of epoch
        start_it,       # the start idx of iteration of a the above epoch
        iters_train,    # the total iterative number of training dataloader
        ld_train,       # the iterator of training dataloader
        ld_val          # the iterator of validation dataloader
    )
    
def train_one_ep(
        ep: int, 
        is_first_ep: bool, 
        start_it: int, 
        args: arg_util.Args, 
        tb_lg: misc.TensorboardLogger, 
        ld_or_itrt, 
        iters_train: int, 
        trainer):
    """
    @func: 
    train for one epoch
    """
    
    ### import heavy packages after Dataloader object creation
    from trainer_ar import VARTrainer
    from optim.lr_control import lr_wd_annealing
    trainer: VARTrainer
    
    ### 
    step_cnt = 0
    me_lg = misc.MetricLogger(delimiter='  ')
    me_lg.add_meter('tlr', misc.SmoothedValue(window_size=1, fmt='{value:.2g}'))
    me_lg.add_meter('tnm', misc.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    [me_lg.add_meter(x, misc.SmoothedValue(fmt='{median:.3f} ({global_avg:.3f})')) for x in ['Lm', 'Lt']]
    [me_lg.add_meter(x, misc.SmoothedValue(fmt='{median:.2f} ({global_avg:.2f})')) for x in ['Accm', 'Acct']]
    header = f'[ep]: [{ep:4d}/{args.ep}]'
    
    if is_first_ep:
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', category=UserWarning)

    ### current iter | max iter
    g_it, max_it = ep * iters_train, args.ep * iters_train
    
    ### run every batch
    for it, obj in me_lg.log_every(start_it, iters_train, ld_or_itrt, 30 if iters_train > 8000 else 5, header):
        
        ## params        
        g_it = ep * iters_train + it
        if it < start_it: continue # skip the start iter
        if is_first_ep and it == start_it: warnings.resetwarnings()
        
        ## move to the device
        args.cur_it = f'{it+1}/{iters_train}' # '1/iters_train' = '1/695'
        
        ## set the 'learning rate (lr)' and the 'weight decay (wd)' | the warm up
        wp_it = args.twp * iters_train # args.wp = args.ep * 1/50
        
        ## get learning rate and weight decay
        min_tlr, max_tlr, min_twd, max_twd = lr_wd_annealing(args.tsche, 
                                                             trainer.var_opt.optimizer, 
                                                             args.tlr, 
                                                             args.twd, 
                                                             args.twde, 
                                                             g_it, 
                                                             wp_it, 
                                                             max_it, 
                                                             wp0=args.twp0, 
                                                             wpe=args.twpe)
        args.cur_lr, args.cur_wd = max_tlr, max_twd # wd is used for the normalization of the weight

        ## gradient accumulation for a huge batch training
        stepping = (g_it + 1) % args.ac == 0 # whether finish the gradient update
        step_cnt += int(stepping) # step_cnt update according to the gradient accumulation
        
        ### train one step 
        grad_norm, scale_log2 = trainer.train_step(
            it=it,              # local iter
            g_it=g_it,          # global iter
            stepping=stepping,  # whether doing the Gradient Accumulation
            me_lg=me_lg,        # metric logger
            tb_lg=tb_lg,        # tensorboard logger
            nactions=obj['action'],     # original input images ｜ [batch_size, 3, data_load_reso, data_load_reso]
            nobs=obj['obs'],            # class labels ｜ [batch_size, ]
        )
        
        ### write into the tensorboard
        # metric logger update
        me_lg.update(tlr=max_tlr)
        me_lg.update(grad_norm=grad_norm)
        # tensorboard logger update
        tb_lg.update(head='AR_opt_lr/lr_min', sche_tlr=min_tlr, step=g_it)      # opt_lr
        tb_lg.update(head='AR_opt_lr/lr_max', sche_tlr=max_tlr, step=g_it) 
        tb_lg.update(head='AR_opt_wd/wd_max', sche_twd=max_twd, step=g_it)      # opt_wd
        tb_lg.update(head='AR_opt_wd/wd_min', sche_twd=min_twd, step=g_it)
        tb_lg.update(head='AR_opt_grad/fp16', scale_log2=scale_log2, step=g_it) # opt_grad
        tb_lg.update(head='AR_opt_grad/grad', grad_norm=grad_norm, step=g_it)
        tb_lg.update(head='AR_opt_grad/grad', grad_clip=args.tclip, step=g_it)
    
    me_lg.synchronize_between_processes()

    # [acverage value during one epoch] | [remain_secs, remain_time, finish_time]
    return {k: meter.global_avg for k, meter in me_lg.meters.items()}, me_lg.iter_time.time_preds(max_it - (g_it + 1) + (args.ep - ep) * 15)  # +15: other cost
    
def main_training():
    """
    @func: the entry for main training process
    """

    print("🔥🔥🔥🔥 BEGIN 🔥🔥🔥🔥")

    ### get args and update the original args
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    
    ### build everything
    (
        tb_lg, 
        trainer,
        start_ep, 
        start_it,
        iters_train, 
        ld_train, 
        ld_val
    ) = build_everything(args)
    
    ### params
    start_time = time.time()
    best_L_mean, best_L_tail, best_acc_mean, best_acc_tail = 999., 999., -1., -1.
    best_val_loss_mean, best_val_loss_tail, best_val_acc_mean, best_val_acc_tail = 999, 999, -1, -1
    L_mean, L_tail = -1, -1
    
    ### start to train
    for ep in range(start_ep, args.ep):
        
        ## make sure the sample different between epochs
        if hasattr(ld_train, 'sampler') and hasattr(ld_train.sampler, 'set_epoch'):
            ld_train.sampler.set_epoch(ep)
            if ep < 3:
                # noinspection PyArgumentList
                print(f'[{type(ld_train).__name__}] [ld_train.sampler.set_epoch({ep})]', flush=True, force=True)
        
        ## set steps
        tb_lg.set_step(ep * iters_train)
        
        ## train for one epoch
        stats, (sec, remain_time, finish_time) = train_one_ep(
            ep,                                 # which epoch are we in
            ep == start_ep,                     # whether the start epoch
            start_it if ep == start_ep else 0,  # which iter are we in of the specific epoch
            args,                               #
            tb_lg,                              # tensorboard log
            ld_train,                           # 
            iters_train,                        # the total number of each epoch
            trainer                             # 
        )
        
        ## training data collection
        L_mean, L_tail, acc_mean, acc_tail, grad_norm = stats['Lm'], stats['Lt'], stats['Accm'], stats['Acct'], stats['tnm'] # 
        best_L_mean, best_acc_mean = min(best_L_mean, L_mean), max(best_acc_mean, acc_mean) #
        if L_tail != -1: best_L_tail, best_acc_tail = min(best_L_tail, L_tail), max(best_acc_tail, acc_tail) #
        args.L_mean, args.L_tail, args.acc_mean, args.acc_tail, args.grad_norm = L_mean, L_tail, acc_mean, acc_tail, grad_norm #
        args.cur_ep = f'{ep+1}/{args.ep}'
        args.remain_time, args.finish_time = remain_time, finish_time
        AR_ep_loss = dict(L_mean=L_mean, L_tail=L_tail, acc_mean=acc_mean, acc_tail=acc_tail) # 💦 training process 💦
        
        ## validation and weight saving
        saving_interval = args.saving_interval
        is_val_and_also_saving = (ep + 1) % saving_interval == 0 or (ep + 1) == args.ep
        if is_val_and_also_saving:
            
            ## evaluate
            val_loss_mean, val_loss_tail, val_acc_mean, val_acc_tail, tot, cost = trainer.eval_ep(ld_val)
            
            ## update
            best_updated = best_val_loss_tail > val_loss_tail
            best_val_loss_mean, best_val_loss_tail = min(best_val_loss_mean, val_loss_mean), min(best_val_loss_tail, val_loss_tail)
            best_val_acc_mean, best_val_acc_tail = max(best_val_acc_mean, val_acc_mean), max(best_val_acc_tail, val_acc_tail)
            AR_ep_loss.update(vL_mean=val_loss_mean, vL_tail=val_loss_tail, vacc_mean=val_acc_mean, vacc_tail=val_acc_tail) # 💦 validation process 💦
            args.vL_mean, args.vL_tail, args.vacc_mean, args.vacc_tail = val_loss_mean, val_loss_tail, val_acc_mean, val_acc_tail
            print(f'[EP{ep}]  (validating {tot})  Lm: {val_loss_mean:.4f}, Lt: {val_loss_tail:.4f}, Acc m&t: {val_acc_mean:.2f} {val_acc_tail:.2f},  Val cost: {cost:.2f}s')
            
            ## saving
            if dist.is_local_master():                
                local_out_ckpt = os.path.join(args.local_out_dir_path, f'ar-ep_{ep+1}-accmean_{val_acc_mean:.2f}-acctail_{val_acc_tail:.2f}.pth')
                torch.save({
                        'epoch': ep + 1,
                        'iter': 0,
                        'trainer': trainer.state_dict(),
                        'args': args.state_dict(),
                    }, local_out_ckpt)
                print(f'[saving ckpt](*) finished! @ {local_out_ckpt}', flush=True, clean=True)

            # waiting for all gpus
            dist.barrier()
        
        ## show
        print(f'[EP{ep}]  (training)  Lm: {best_L_mean:.3f} ({L_mean:.3f}), Lt: {best_L_tail:.3f} ({L_tail:.3f}),  Acc m&t: {best_acc_mean:.2f} {best_acc_tail:.2f},  Remain: {remain_time},  Finish: {finish_time}', flush=True)
        tb_lg.update(head='AR_ep_loss', step=ep+1, **AR_ep_loss)
        tb_lg.update(head='AR_z_burnout', step=ep+1, rest_hours=round(sec / 60 / 60, 2))

        ## flush
        args.dump_log(); tb_lg.flush()

    ## final
    total_time = f'{(time.time() - start_time) / 60 / 60:.1f}h'
    print('\n\n')
    print(f'[PT finished]  Total cost: {total_time},   Lm: {best_L_mean:.3f} ({L_mean}),   Lt: {best_L_tail:.3f} ({L_tail})')
    print('\n\n')
    
    del stats
    del iters_train, ld_train
    time.sleep(3), gc.collect(), torch.cuda.empty_cache(), time.sleep(3)
    
    args.remain_time, args.finish_time = '-', time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() - 60))
    print(f'final args:\n\n{str(args)}')
    args.dump_log(); tb_lg.flush(); tb_lg.close()
    dist.barrier()

if __name__ == '__main__':
    try: main_training()
    finally:
        dist.finalize()
        if isinstance(sys.stdout, misc.SyncPrint) and isinstance(sys.stderr, misc.SyncPrint):
            sys.stdout.close(), sys.stderr.close()
