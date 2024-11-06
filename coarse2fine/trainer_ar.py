import time
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import dist
import copy
from vqvae.vqvae import VQVAE
from vqvae.quant import VectorQuantizer2
from autoreg import VAR
from optim.amp_opt import AmpOptimizer
from utils.misc import MetricLogger, TensorboardLogger

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor

from robodata.normalizer import LinearNormalizer
from roboenv.ema_model import EMAModel

class VARTrainer(object):
    def __init__(
        self, 
        ###
        device, 
        normalizer : LinearNormalizer,
        ### VAR
        patch_nums: Tuple[int, ...], 
        resos: Tuple[int, ...],
        vae_local: VQVAE, 
        var_wo_ddp: VAR, 
        var: DDP,
        var_opt: AmpOptimizer,
        label_smooth: float,
    ):
        super(VARTrainer, self).__init__()
        
        ### 🔥 🔥 🔥
        self.action_dim = 10
        
        ### models
        self.var, self.vae_local = var, vae_local # 🔥 🔥 🔥
        self.var_wo_ddp: VAR = var_wo_ddp  # after torch.compile
        self.var_opt = var_opt
        
        ### 🔥 ema var 🔥
        self.ema_var_wo_ddp: VAR = copy.deepcopy(self.var_wo_ddp)
        self.ema_func = EMAModel(inv_gamma=1.0, 
                                 max_value=0.9999, 
                                 min_value=0.0, 
                                 power=0.75,
                                 update_after_step=0,
                                 model = self.ema_var_wo_ddp)
        
        ### loss
        self.patch_nums, self.resos = [x * self.action_dim for x in patch_nums], resos # 🔥 🔥 🔥
        self.label_smooth = label_smooth # whetehr smooth the gt labels
        self.train_loss = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction='none') # reduction -> untouch value | [0.5, 1.0, 1.5]
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean') # reduction -> mean value | (0.5 + 1.0 + 1.5) / 3 = 1.0
        self.L = sum(pn * 1 for pn in self.patch_nums) # length for all of the scale | 70 pathes ｜ 🔥 🔥 🔥
        self.last_l = self.patch_nums[-1] * 1 # the length of the last scale | 4*1*7 patches | 🔥 🔥 🔥
        self.loss_weight = torch.ones(1, self.L, device=device) / self.L # for training loss | [1,L]
        
        #### get the section for each scale
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):        # 🔥 🔥 🔥
            self.begin_ends.append((cur, cur + pn * 1)) # 🔥 🔥 🔥
            cur += pn*1                                 # 🔥 🔥 🔥
        
        ### normalizer
        self.var_norm = normalizer
        self.var_norm.to(device)
    
    @torch.no_grad()
    def eval_ep(self, ld_val: DataLoader):
        tot = 0
        L_mean, L_tail, acc_mean, acc_tail = 0, 0, 0, 0
        stt = time.time()
        training = self.var_wo_ddp.training
        self.var_wo_ddp.eval()
        for obj in ld_val:
            
            ### input
            nactions: FTen = self.var_norm['action'].normalize(obj['action']) # [B,L,action_dim]
            nactions: FTen = nactions.view(nactions.shape[0],1,nactions.shape[1],nactions.shape[2]).contiguous() # [B,1,L,action_dim]
            nobs: List[FTen] = self.var_norm.normalize(obj['obs']) # [obs,...]
            B, V = nactions.shape[0], self.vae_local.vocab_size # batch size | vocabulary size

            ### transform
            idxBls: List[List[ITen]] = self.vae_local.inp_to_idxBl(nactions)    # List[List([B,1or2or3or4])]
            var_inputs: List[FTen] = self.vae_local.idxBl_to_var_input(idxBls)  # List[[B,2+3+4,c])]
            gt_BL = self.trains_sep_idxBls(idxBls)                              # [B，10*action_dim]
            x_BLCv_wo_first_l = self.trains_sep_var_inputs(var_inputs)          # [B, 9*action_dim, 8]
            
            ### run inference
            self.var_wo_ddp.forward
            logits_BLV = self.var_wo_ddp(nobs, x_BLCv_wo_first_l)
            
            ### 
            L_mean += self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)) * B
            L_tail += self.val_loss(logits_BLV.data[:, -self.last_l:].reshape(-1, V), gt_BL[:, -self.last_l:].reshape(-1)) * B
            acc_mean += (logits_BLV.data.argmax(dim=-1) == gt_BL).sum() * (100/gt_BL.shape[1])
            acc_tail += (logits_BLV.data[:, -self.last_l:].argmax(dim=-1) == gt_BL[:, -self.last_l:]).sum() * (100 / self.last_l)
            tot += B
            
        self.var_wo_ddp.train(training)

        stats = L_mean.new_tensor([L_mean.item(), L_tail.item(), acc_mean.item(), acc_tail.item(), tot])
        dist.allreduce(stats)
        tot = round(stats[-1].item())
        stats /= tot
        L_mean, L_tail, acc_mean, acc_tail, _ = stats.tolist()
        return L_mean, L_tail, acc_mean, acc_tail, tot, time.time()-stt
    
    def trains_sep_idxBls(self, idxBls):
        """
        :func: transform the separate idxBls into the format of the input of var
        """
        B = idxBls[0][0].shape[0]
        gt_BL = torch.cat([torch.cat(idxBl, dim=1).unsqueeze(1) for idxBl in idxBls], dim=1) # [B,action_dim,10]
        gt_BL = torch.transpose(gt_BL, 1, 2) # shape->[B, 10, action_dim]
        gt_BL = gt_BL.reshape(B,-1) # shape->[B, 10*action_dim]
        return gt_BL
    
    def trains_sep_var_inputs(self, var_inputs):
        """
        :func: transform the separate var_input into the format of the input of var
        """
        B,_,C = var_inputs[0].shape
        input = torch.cat([x.unsqueeze(1) for x in var_inputs], dim=1) # [B,action_dim,9,C]
        input = torch.transpose(input, 1, 2) # shape->[B,9,action_dim,C]
        input = input.reshape(B,-1, C) # shape->[B,9*action_dim,C]
        return input
    
    def train_step(
        self, 
        it: int, 
        g_it: int, 
        stepping: bool, 
        me_lg: MetricLogger, 
        tb_lg: TensorboardLogger,
        nactions: torch.Tensor, 
        nobs: torch.Tensor, 
    ) -> Tuple[Optional[Union[FTen, float]], Optional[float]]:
        """
        :func: train the var with one step
        """
        
        ### input
        nactions: FTen = self.var_norm['action'].normalize(nactions) # [B,L,action_dim]
        nactions: FTen = nactions.view(nactions.shape[0],1,nactions.shape[1],nactions.shape[2]).contiguous() # [B,1,L,action_dim]
        nobs: List[FTen] = self.var_norm.normalize(nobs) # [obs,...]
        
        ### params
        B, V = nactions.shape[0], self.vae_local.vocab_size # batch_size, vocab_size
        self.var.require_backward_grad_sync = stepping # whether update the weight with the gradient
        idxBls: List[List[ITen]] = self.vae_local.inp_to_idxBl(nactions) # List[List([B,1or2or3or4])]
        var_inputs: List[FTen] = self.vae_local.idxBl_to_var_input(idxBls) # List[[B,2+3+4,c])]
        
        ### transition
        gt_BL = self.trains_sep_idxBls(idxBls)                          # [B，10*action_dim]
        x_BLCv_wo_first_l = self.trains_sep_var_inputs(var_inputs)      # [B, 9*action_dim, 8]
        
        ### automatic mixed precision
        with self.var_opt.amp_ctx:
            ## forward
            self.var_wo_ddp.forward # forward
            logits_BLV = self.var(nobs, x_BLCv_wo_first_l) # transformer inference | BLV
            ## loss
            loss = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1) # calc the corss-entropy loss |  [B,L,V] and [B,L] | get [B,L]
            # final loss
            lw = self.loss_weight # 1L
            loss = loss.mul(lw).sum(dim=-1).mean() # get [B,]
        
        ### backward
        grad_norm, scale_log2 = self.var_opt.backward_clip_step(loss=loss, stepping=stepping)
        
        ### 🔥 ema 🔥
        self.ema_func.step(self.var_wo_ddp)
        
        ### log to metric
        pred_BL = logits_BLV.data.argmax(dim=-1)
        if it == 0 or it in me_lg.log_iters:
            ## all tokens loss
            Lmean = self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)).item()
            acc_mean = (pred_BL == gt_BL).float().mean().item() * 100
            ## last scale tokens loss
            Ltail = self.val_loss(logits_BLV.data[:, -self.last_l:].reshape(-1, V), gt_BL[:, -self.last_l:].reshape(-1)).item()
            acc_tail = (pred_BL[:, -self.last_l:] == gt_BL[:, -self.last_l:]).float().mean().item() * 100
            ## grad norm
            grad_norm = grad_norm.item()
            ## update
            me_lg.update(Lm=Lmean, Lt=Ltail, Accm=acc_mean, Acct=acc_tail, tnm=grad_norm)
        
        ### log to tensorboard
        if g_it == 0 or (g_it + 1) % 500 == 0:
            # the rate of vocabularies we often use
            prob_per_class_is_chosen = pred_BL.view(-1).bincount(minlength=V).float()
            dist.allreduce(prob_per_class_is_chosen)
            prob_per_class_is_chosen /= prob_per_class_is_chosen.sum()
            cluster_usage = (prob_per_class_is_chosen > 0.001 / V).float().mean().item() * 100
            # update once
            if dist.is_master():
                # z_voc_usage
                if g_it == 0:
                    tb_lg.update(head='AR_iter_loss', z_voc_usage=cluster_usage, step=-10000)
                    tb_lg.update(head='AR_iter_loss', z_voc_usage=cluster_usage, step=-1000)
                kw = dict(z_voc_usage=cluster_usage)
                # accuracy and loss
                for si, (bg, ed) in enumerate(self.begin_ends):
                    pred, tar = logits_BLV.data[:, bg:ed].reshape(-1, V), gt_BL[:, bg:ed].reshape(-1) # get ans
                    acc = (pred.argmax(dim=-1) == tar).float().mean().item() * 100 # the accuracy between predicted idx and target idx
                    ce = self.val_loss(pred, tar).item() # cross-entropy loss between [L,V] and [L,]
                    kw[f'acc_{self.resos[si]}'] = acc # 
                    kw[f'L_{self.resos[si]}'] = ce  # 

                tb_lg.update(head='AR_iter_loss', **kw, step=g_it)
        
        return grad_norm, scale_log2
    
    def get_config(self):
        return {
            'patch_nums':   self.patch_nums, 
            'resos': self.resos,
            'label_smooth': self.label_smooth,
        }
    
    def state_dict(self):
        state = {'config': self.get_config()}
        for k in ('var_wo_ddp', 'vae_local', 'var_opt', 'var_norm', 'ema_var_wo_ddp'):
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                state[k] = m.state_dict()
        return state
    
    def load_state_dict(self, state, strict=True, skip_vae=False):
        for k in ('var_wo_ddp', 'vae_local', 'var_opt', 'var_norm', 'ema_var_wo_ddp'):
            if skip_vae and 'vae' in k: continue
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                ret = m.load_state_dict(state[k], strict=strict)
                if ret is not None:
                    missing, unexpected = ret
                    print(f'[VARTrainer.load_state_dict] {k} missing:  {missing}')
                    print(f'[VARTrainer.load_state_dict] {k} unexpected:  {unexpected}')
        
        config: dict = state.pop('config', None)
        self.prog_it = config.get('prog_it', 0)
        self.last_prog_si = config.get('last_prog_si', -1)
        self.first_prog = config.get('first_prog', True)
        if config is not None:
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f'[VAR.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                    if strict: raise AttributeError(err)
                    else: print(err)
