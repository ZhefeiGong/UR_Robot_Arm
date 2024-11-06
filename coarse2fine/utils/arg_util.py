import json
import os
import random
import re
import subprocess
import sys
import time
from collections import OrderedDict
from typing import Optional, Union
import numpy as np
import torch
try:
    from tap import Tap
except ImportError as e:
    print(f'>>>>>>>> `from tap import Tap` failed, please run:      pip3 install typed-argument-parser     <<<<<<<<', file=sys.stderr, flush=True)
    print(f'>>>>>>>> `from tap import Tap` failed, please run:      pip3 install typed-argument-parser     <<<<<<<<', file=sys.stderr, flush=True)
    time.sleep(5)
    raise e
import dist

class Args(Tap):
    
    #### 🔥Setting🔥
    data_path: str = '/path/to/dataset'                                     # 🌞 the data we're gonna use 🌞
    vae_ckpt_path: str = 'ckpt/act_vq_tool-hang_0916/vae-ckpt-220.pth'      # 🌞 the path of vae 🌞 | useless
    obs_dim: int = 46                                                       # 🌞 tool-hang:106(53) / can:46(23) / square:46(23) 🌞
    model_name: str = "ar"                                                  # 🌞 the name of the experiment 🌞
    task_name: str = 'can'                                                  # 🌞 the class name of the model 🌞
    exp_name: str = 'ar'                                                    # 🌞 the name of this experiment 🌞
    
    saving_interval: int = 100              # 🌞 the rollout and evaluation time interval 🌞
    topk: int = 5                           # 🌞 the topk ckpts we need save🌞
    
    #### 🔥VAE🔥
    # vae training
    act_dim_sep: int=0          # 🌞 the dimension where we're going to build 🌞
    vfast: int = 0              # torch.compile VAE; =0: not compile; 1: compile with 'reduce-overhead'; 2: compile with 'max-autotune'
    vblr: float = 3e-4          # learning rate
    vlr: float = None           # lr = base lr * (bs / 256)
    vwd: float = 0.005          # weight decay
    vwde: float = 0             # weight decay end
    vclip: float = 10.          # 
    vema: float=0.9999          # 🌞 whether to do the ema operation 🌞
    vwp: float = 0              # warm up 
    vwp0: float = 0.005         # initial lr ratio at the begging of lr warm up
    vwpe: float = 0.3           # final lr ratio at the end of training
    vsche: str = 'cos'          #
    vdrop: float = 0.0          # 🌞 0.05 : image | 0.0 : action 🌞
    vopt_beta: str = '0.5_0.9'  # 
    vae_init: float = -0.5      # <0: xavier_normal_(gain=abs(init)); >0: trunc_normal_(std=init)
    vocab_init: float = -1      # <0: uniform(-abs(init)*base, abs(init)*base), where base = 20/vocab_size; >0: trunc_normal_(std=init)
    # vae structure
    vocab_size: int = 512     # 🌞 4096 : image | 512 : action 🌞
    vocab_ch: int = 8         # 🌞 32 : image | 8 : action | 7 : action🌞
    vch: int = 2              # 🌞 160 : image | 8 : action | 2 : action 🌞
    vqresi: float = 0.5       #
    vqbeta: float = 0.25      #
    vqnorm: bool = True       # 🌞 True : action 🌞 
    
    #### 🔥AR🔥
    # ar structure
    tdepth: int = 16        # 🌞 16 : image | 8 : action 🌞 | 4 : action 🌞 | 🔥 🔥 🔥
    tembed: int = 160       # 🌞 160 : the embedding of ar | 🔥 🔥 🔥
    tnobs: int = 1          # 🌞 1 : the number of observations | 🔥 🔥 🔥
    # ar transformer initialization
    tini: float = -1        # -1: automated model parameter initialization
    thd: float = 0.02       # head.w *= hd
    taln: float = 0.5       # the multiplier of ada_lin.w's initialization
    talng: float = 1e-3     # the multiplier of ada_lin.w[gamma channels]'s initialization | 1e-5
    # ar transformer optimization
    tfast: int = 0          # torch.compile VAR; =0: not compile; 1: compile with 'reduce-overhead'; 2: compile with 'max-autotune'
    tblr: float = 1e-4      # base lr
    tlr: float = None       # lr = base lr * (bs / 256)
    twd: float = 0.05       # initial wd
    tclip: float = 2.       # <=0 for not using grad clip
    twde: float = 0         # final wd
    tls: float = 0.0        # label smooth
    twp: float = 0          # warm up 
    twp0: float = 0.005     # initial lr ratio at the begging of lr warm up
    twpe: float = 0.1       # final lr ratio at the end of training | 0.01
    tsche: str = 'lin0'     # lr schedule
    
    # gpu calculation
    tf32: bool = True       # whether to use TensorFloat32 | automatically for Ampere
    fp16: int = 0           # 0: using fp32, 1: using fp16, 2: using bf16
    
    # training
    seed: int = None        # 🌞 seed 🌞
    bs: int = 768           # 🌞 global batch size 🌞
    ac: int = 1             # gradient accumulation
    ep: int = 250           # 🌞 the number of epochs 🌞
    device: str = 'cpu'     # [automatically set; don't specify this]
    same_seed_for_all_ranks: int = 0     # this is only for distributed sampler
    batch_size: int = 0     # [automatically set; don't specify this] batch size per GPU = round(args.bs / args.ac / dist.get_world_size() / 8) * 8
    glb_batch_size: int = 0 # [automatically set; don't specify this] global batch size = args.batch_size * dist.get_world_size()
    opt: str = 'adamw'      # 
    afuse: bool = True      # 
    
    # other hps
    saln: bool = False      # whether to use shared adaln
    anorm: bool = True      # whether to use L2 normalized attention
    
    # scale
    pn: str = '1_2_3_4'         # 🌞 '1_2_3_4_5_6_8_10_13_16' : image | '1_2_3_4' : action 🌞
    patch_size: int = 1         # 🌞 16 : image | 1 : action 🌞
    patch_nums: tuple = None    # [automatically set; don't specify this] = tuple(map(int, args.pn.replace('-', '_').split('_')))
    resos: tuple = None         # [automatically set; don't specify this] = tuple(pn * args.patch_size for pn in args.patch_nums)
    
    # data
    workers: int = 0            # 🌞 num workers; 0: auto, -1: don't use multiprocessing in DataLoader 🌞
    
    # would be automatically set in runtime | [automatically set; don't specify this]
    cmd: str = ' '.join(sys.argv[1:])  # [automatically set; don't specify this] | save all of the args
    acc_mean: float = None      # [automatically set; don't specify this]
    acc_tail: float = None      # [automatically set; don't specify this]
    L_mean: float = None        # [automatically set; don't specify this]
    L_tail: float = None        # [automatically set; don't specify this]
    vacc_mean: float = None     # [automatically set; don't specify this]
    vacc_tail: float = None     # [automatically set; don't specify this]
    vL_mean: float = None       # [automatically set; don't specify this]
    vL_tail: float = None       # [automatically set; don't specify this]
    grad_norm: float = None     # [automatically set; don't specify this]
    cur_lr: float = None        # [automatically set; don't specify this]
    cur_wd: float = None        # [automatically set; don't specify this]
    cur_it: str = ''            # [automatically set; don't specify this]
    cur_ep: str = ''            # [automatically set; don't specify this]
    remain_time: str = ''       # [automatically set; don't specify this]
    finish_time: str = ''       # [automatically set; don't specify this]
    
    # environment | [automatically set; don't specify this]
    local_out_dir_path: str = '...'     # [automatically set; don't specify this]
    tb_log_dir_path: str = '...tb-...'  # [automatically set; don't specify this]
    log_txt_path: str = '...'           # [automatically set; don't specify this]
    last_ckpt_path: str = '...'         # [automatically set; don't specify this]
    
    def seed_everything(self, benchmark: bool):
        """
        @func :
        seed for whole progress in order to reproduce easily
        """
        
        # Dynamically select an optimal convolution algorithm
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = benchmark
        
        if self.seed is None:
            torch.backends.cudnn.deterministic = False
        else:
            torch.backends.cudnn.deterministic = True

            # ensure different process dealing with different random seed🔥 🔥 🔥
            seed = self.seed * dist.get_world_size() + dist.get_rank() 
            print(f"[INFO] 🔥 the seed 🔥 is : {seed}")
            
            os.environ['PYTHONHASHSEED'] = str(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

            # # Disable parallelization to ensure reproducibility
            # os.environ['OMP_NUM_THREADS'] = '1'
            # os.environ['MKL_NUM_THREADS'] = '1'

    def get_different_generator_for_each_rank(self) -> Optional[torch.Generator]:
        """
        @func: 
        construct the generator for random variables
        """
        if self.seed is None:
            return None
        g = torch.Generator()
        g.manual_seed(self.seed * dist.get_world_size() + dist.get_rank())
        return g
    
    def compile_model(self, m, fast):
        """
        @func: 
        optimize the compiling process of models
        """
        if fast == 0:
            return m
        return torch.compile(m, mode={
            1: 'reduce-overhead',
            2: 'max-autotune',
            3: 'default',
        }[fast]) if hasattr(torch, 'compile') else m
    
    def state_dict(self, key_ordered=True) -> Union[OrderedDict, dict]:
        """
        @func : 
        """
        d = (OrderedDict if key_ordered else dict)()
        # self.as_dict() would contain methods, but we only need variables
        for k in self.class_variables.keys():
            if k not in {'device'}:     # these are not serializable
                d[k] = getattr(self, k)
        return d
    
    def load_state_dict(self, d: Union[OrderedDict, dict, str]):
        """
        @func : 
        """
        if isinstance(d, str):  # for compatibility with old version
            d: dict = eval('\n'.join([l for l in d.splitlines() if '<bound' not in l and 'device(' not in l]))
        for k in d.keys():
            try:
                setattr(self, k, d[k])
            except Exception as e:
                print(f'k={k}, v={d[k]}')
                raise e
    
    @staticmethod
    def set_tf32(tf32: bool):
        """
        @func : 
        """
        if torch.cuda.is_available():
            torch.backends.cudnn.allow_tf32 = bool(tf32)
            torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision('high' if tf32 else 'highest')
                print(f'[tf32] [precis] torch.get_float32_matmul_precision(): {torch.get_float32_matmul_precision()}')
            print(f'[tf32] [ conv ] torch.backends.cudnn.allow_tf32: {torch.backends.cudnn.allow_tf32}')
            print(f'[tf32] [matmul] torch.backends.cuda.matmul.allow_tf32: {torch.backends.cuda.matmul.allow_tf32}')
    
    def dump_log(self):
        """
        @func: 
        output something into the log.txt
        """

        # only record for local master
        if not dist.is_local_master():
            return
        
        # first time to dump log
        if '1/' in self.cur_ep: 
            with open(self.log_txt_path, 'w') as fp:
                json.dump({'is_master': dist.is_master(), 
                           'name': self.exp_name, 
                           'cmd': self.cmd, 
                           'tb_log_dir_path': self.tb_log_dir_path}, fp, indent=0)
                fp.write('\n')
        
        # 
        log_dict = {}
        for k, v in {
            'it': self.cur_it, 
            'ep': self.cur_ep,
            'lr': self.cur_lr, 
            'wd': self.cur_wd, 
            'grad_norm': self.grad_norm,
        }.items():
            if hasattr(v, 'item'): v = v.item()
            log_dict[k] = v
        
        # write
        with open(self.log_txt_path, 'a') as fp:
            fp.write(f'{log_dict}\n')
        
    def __str__(self):
        """
        @func : 
        """

        s = []
        for k in self.class_variables.keys():
            if k not in {'device', 'dbg_ks_fp'}:     # these are not serializable
                s.append(f'  {k:20s}: {getattr(self, k)}')
        s = '\n'.join(s)
        return f'{{\n{s}\n}}\n'


def init_dist_and_get_args():
    """
    @func: 
    initialize all of the dist and args
    """
    
    ### delete local_rank param
    for i in range(len(sys.argv)):
        if sys.argv[i].startswith('--local-rank=') or sys.argv[i].startswith('--local_rank='):
            del sys.argv[i]
            break
    
    ### get all params
    args = Args(explicit_bool=True).parse_args(known_only=True)
    
    ### warn args.extra_args
    if len(args.extra_args) > 0:
        print(f'======================================================================================')
        print(f'=========================== WARNING: UNEXPECTED EXTRA ARGS ===========================\n{args.extra_args}')
        print(f'=========================== WARNING: UNEXPECTED EXTRA ARGS ===========================')
        print(f'======================================================================================\n\n')
    
    ### init torch distributed
    from utils import misc
    from datetime import datetime
    current_time = datetime.now().strftime("%m%d%H")
    args.local_out_dir_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), f'local_output/act_{args.exp_name}_{args.model_name}_v{args.vocab_size}_{current_time}')
    os.makedirs(args.local_out_dir_path, exist_ok=True)
    misc.init_distributed_mode(local_out_path=args.local_out_dir_path, timeout=30)
    
    ### set env
    args.set_tf32(args.tf32)                # tf32 | args.tf32
    args.seed_everything(benchmark=True)    # the random seed for the process | args.seed
    
    ### update args: data loading
    args.device = dist.get_device()
    
    ### update args: multi scales
    args.patch_nums = tuple(map(int, args.pn.replace('-', '_').split('_')))
    args.resos = tuple(pn * args.patch_size for pn in args.patch_nums) # pn * patch_size
    
    ### update args: bs
    bs_per_gpu = round(args.bs / args.ac / dist.get_world_size())
    args.batch_size = bs_per_gpu
    args.bs = args.glb_batch_size = args.batch_size * dist.get_world_size() # 
    args.workers = min(max(0, args.workers), args.batch_size) # make workers \in [0, batch_size]
    
    ### update args: lr
    args.tlr = args.ac * args.tblr * args.glb_batch_size / 256
    args.vlr = args.ac * args.vblr * args.glb_batch_size / 256
    
    ### update args: warm up epoches
    if args.twp == 0:
        args.twp = args.ep * 1/50
    if args.vwp == 0:
        args.vwp = args.ep * 1/50
    
    ### update args: paths
    args.log_txt_path = os.path.join(args.local_out_dir_path, 'log.txt')
    args.last_ckpt_path = os.path.join(args.local_out_dir_path, f'{args.model_name}-ckpt-last.pth')
    _reg_valid_name = re.compile(r'[^\w\-+,.]')
    tb_name = _reg_valid_name.sub(
        '_',
        f'tb-{args.model_name}_ch{args.vch}'
        f'__pn{args.pn}'
        f'__b{args.bs}ep{args.ep}{args.opt[:4]}lr{args.vblr:g}wd{args.vwd:g}'
    )
    args.tb_log_dir_path = os.path.join(args.local_out_dir_path, tb_name)
    
    return args

