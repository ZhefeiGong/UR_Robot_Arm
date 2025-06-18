import torch
import torch.nn as nn
import torch.nn.functional as F

# this file only provides the 2 modules used in VQVAE
__all__ = ['Encoder', 'Decoder',]

def nonlinearity(x):
    """
    @func: 
    change to non linear network
    """
    return x * torch.sigmoid(x)

def Normalize(num_channels, num_groups):
    """
    @func: 
    normalize each group along with the feature dimension | num_fea_each_goup = num_groups / num_channels
    """
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=1e-6, affine=True) # group normalization

# class Upsample2x_IP(nn.Module):
#     """
#     @func: 
#     upsample | *2
#     """
#     def __init__(self, in_channels):
#         super().__init__()
#         self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=(3,1), stride=1, padding=(1,0))
#     def forward(self, x):
#         H,W = x.shape[-2], x.shape[-1]
#         return self.conv(F.interpolate(x, size=(2*H,W), mode='nearest'))

class Upsample2x_TF(nn.Module):
    """
    @func: 
    upsample | *2
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=(3, 1), stride=(2, 1), padding=(1,0), output_padding=(1, 0))
    def forward(self, x):
        return self.conv_transpose(x)
    
class Downsample2x(nn.Module):
    """
    @func: 
    downsample | /2
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=(3,1), stride=(2,1), padding=0) # /2
    def forward(self, x):
        return self.conv(F.pad(x, pad=(0, 0, 0, 1), mode='constant', value=0)) # F.pad: [left,right,top,bottom]

# class ActionStretchMLP(nn.Module):
#     """
#     @func: 
#     transform the dimension of the action
#     """
#     def __init__(self, input_dim, output_dim, hidden_dim=16):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             torch.nn.Linear(input_dim, hidden_dim),     # 
#             torch.nn.SiLU(inplace=True),                # 
#             torch.nn.Linear(hidden_dim, hidden_dim),    # 
#             torch.nn.SiLU(inplace=True),                # 
#             torch.nn.Linear(hidden_dim, output_dim)     # 
#         )
#     def forward(self, x):
#         return self.mlp(x)

class ConvBlock(nn.Module):
    def __init__(self, *, in_channels, num_groups, out_channels=None, dropout=None):
        super().__init__()
        # init
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        # layer1
        self.norm1 = Normalize(num_channels=in_channels, num_groups=num_groups)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=(3,1), stride=(1,1), padding=(1,0)) # only on time-dimension
        # layer2
        self.norm2 = Normalize(num_channels=out_channels, num_groups=num_groups)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 1e-6 else nn.Identity() # 
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=(3,1), stride=(1,1), padding=(1,0)) # only on time-dimension
        # residual
        if self.in_channels != self.out_channels:
            self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.nin_shortcut = nn.Identity()
    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x), inplace=True))                 # F.silu is the Sigmoid Linear Unit 
        h = self.conv2(self.dropout(F.silu(self.norm2(h), inplace=True)))   # F.silu is the Sigmoid Linear Unit 
        return self.nin_shortcut(x) + h

# class AttnBlock(nn.Module):
#     def __init__(self, in_channels, z_channels):
#         super().__init__()
#         self.C = in_channels
#         self.norm = Normalize(num_channels=in_channels, num_groups=z_channels) # group normalization
#         self.qkv = torch.nn.Conv2d(in_channels, 3*in_channels, kernel_size=1, stride=1, padding=0) # channel communication | up channel to 3*in_channels
#         self.w_ratio = int(in_channels) ** (-0.5) # weight
#         self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0) # channel communication
#     def forward(self, x):
#         """
#         @func : 
#         V(K^TQ)
#         """
#         # init
#         qkv = self.qkv(self.norm(x))
#         B, _, H, W = qkv.shape  # should be B,3C,H,W
#         C = self.C # 
#         q, k, v = qkv.reshape(B, 3, C, H, W).unbind(1) # unbind | get a tuple of all slices along a given dimension
#         # compute attention matrix
#         q = q.view(B, C, H * W).contiguous()    # B,C,HW
#         q = q.permute(0, 2, 1).contiguous()     # B,HW,C - query
#         k = k.view(B, C, H * W).contiguous()    # B,C,HW - key
#         w = torch.bmm(q, k).mul_(self.w_ratio)  # B,HW(q),HW(k)    w[B,i,j]=sum_c q[B,i,C]k[B,C,j]
#         w = F.softmax(w, dim=2) # in HW(k) dim
#         # attend to values
#         v = v.view(B, C, H * W).contiguous()    # B,C,HW
#         w = w.permute(0, 2, 1).contiguous()     # B,HW(k),HW(q) (first HW of k, second of q)
#         h = torch.bmm(v, w)                     # B,C,HW(q) | h[B,C,j] = sum_i v[B,C,i] w[B,i,j]
#         h = h.view(B, C, H, W).contiguous()     # B,C,H,W
#         return x + self.proj_out(h)

class Encoder(nn.Module):
    """
    @func: 
    from [B,1,16,7] to [B,7,4,1]
    @feature: 
    conv | attn | linear
    """
    def __init__(
        self, 
        *, 
        ch=2, 
        ch_mult=(2, 4), 
        in_channels=1,
        z_channels=8, 
        # action_dim=1, 
        num_actions=16, 
        dropout=0.0,
    ):
        super().__init__()
        
        # init
        self.ch = ch
        self.ch_mult = ch_mult
        self.in_channels = in_channels
        self.z_channels = z_channels # 🔥 🔥 🔥
        # self.action_dim = 1
        self.num_actions = num_actions

        # conv in
        self.conv_in = torch.nn.Conv2d(self.in_channels, self.ch, kernel_size=(3,1), stride=(1,1), padding=(1,0)) # B,ch,16,7
        # down - block 1
        self.down_1 = nn.Module()
        self.down_1.block_1 = ConvBlock(in_channels=self.ch, num_groups=1, out_channels=self.ch*self.ch_mult[0], dropout=dropout) # B,2*ch,16,7
        self.down_1.downsample = Downsample2x(in_channels=self.ch*self.ch_mult[0]) # B,2*ch,8,7
        self.down_1.block_2 = ConvBlock(in_channels=self.ch*self.ch_mult[0], num_groups=1, out_channels=self.ch*self.ch_mult[0], dropout=dropout) # B,2*ch,8,7
        
        # # down - attn 1
        # self.attn_1 = AttnBlock(in_channels=self.ch*self.ch_mult[0], z_channels=self.z_channels) # B,2*ch,8,7

        # down - block 2
        self.down_2 = nn.Module()
        self.down_2.block_1 = ConvBlock(in_channels=self.ch*self.ch_mult[0], num_groups=1, out_channels=self.ch*self.ch_mult[1], dropout=dropout) # B,4*ch,4,7
        self.down_2.downsample = Downsample2x(in_channels=self.ch*self.ch_mult[1]) # B,4*ch,4,7
        self.down_2.block_2 = ConvBlock(in_channels=self.ch*self.ch_mult[1], num_groups=1, out_channels=self.ch*self.ch_mult[1], dropout=dropout) # B,4*ch,4,7

        # # down - attn 2
        # self.attn_2 = AttnBlock(in_channels=self.ch * self.ch_mult[1], z_channels=self.z_channels) # B,4*ch,4,7

        # conv out
        self.norm_out = Normalize(num_channels=self.ch*self.ch_mult[1], num_groups=1) # B,4*ch,4,7
        self.conv_out = torch.nn.Conv2d(self.ch*self.ch_mult[1], self.z_channels, kernel_size=(3,1), stride=(1,1), padding=(1,0)) # B,z_channels,4,7
    
        # # action
        # self.action_mlp = ActionStretchMLP(input_dim=action_dim,output_dim=1) # B,z_channels,4,1
    
    def forward(self, x):
        """
        @input:
        x has shape [batch_size, 1, num_actions, action_dim]
        """

        # begin
        h = self.conv_in(x) # [B,ch,16,act_dim]

        # down-1
        h = self.down_1.block_2(self.down_1.downsample(self.down_1.block_1(h))) # [B,2ch,8,act_dim]
        
        # # attn-1
        # h = self.attn_1(h) # [B,2ch,8,act_dim]
        
        # down-2
        h = self.down_2.block_2(self.down_2.downsample(self.down_2.block_1(h))) # [B,4ch,4,act_dim]
        
        # # attn-1
        # h = self.attn_2(h)  # [B,4ch,4,act_dim]

        # end
        h = self.conv_out(F.silu(self.norm_out(h), inplace=True)) # [B,z_channels,4,act_dim]

        # # action
        # h = self.action_mlp(h) # [B,z_channels,4,1]
        # # change dim
        # h = torch.transpose(h, 1, 3) # [B,7,4,z_channels]
        
        # output
        return h # h has shape [B,z_channels,4,act_dim]

class Decoder(nn.Module):
    """
    @func: 
    from [B,7,4,1] to [B,1,16,7]
    @feature: 
    conv | attn | linear
    """
    def __init__(
        self, 
        *, 
        ch=8, 
        ch_mult=(2, 4), 
        in_channels=1,
        z_channels=8, 
        # action_dim=1, 
        num_actions=16, 
        dropout=0.0,
    ):
        super().__init__()
        # init
        self.ch = ch
        self.ch_mult = ch_mult
        self.in_channels = in_channels
        self.z_channels = z_channels # 🔥 🔥 🔥
        # self.action_dim = 1
        self.num_actions = num_actions

        # self.action_mlp = ActionStretchMLP(input_dim=1,output_dim=action_dim) # B,z_channels,4,7

        # z to block_in
        self.norm_in = Normalize(num_channels=self.z_channels, num_groups=1) # B,z_channels,4,7
        self.conv_in = torch.nn.Conv2d(self.z_channels, self.ch * self.ch_mult[-1], kernel_size=(3,1), stride=(1,1), padding=(1,0)) # B,4ch,4,7

        # # up - attn 1
        # self.attn_1 = AttnBlock(in_channels=self.ch*self.ch_mult[-1], z_channels=self.z_channels) # B,4ch,4,7

        # up - block 1
        self.up_1 = nn.Module()
        self.up_1.block_1 = ConvBlock(in_channels=self.ch*self.ch_mult[-1], num_groups=1, out_channels=self.ch*self.ch_mult[-1], dropout=dropout) # B,4ch,4,7
        self.up_1.upsample = Upsample2x_TF(in_channels=self.ch*self.ch_mult[-1]) # B,4ch,8,7
        self.up_1.block_2 = ConvBlock(in_channels=self.ch*self.ch_mult[-1], num_groups=1, out_channels=self.ch*self.ch_mult[-2], dropout=dropout) # B,2ch,8,7

        # # up - attn 2
        # self.attn_2 = AttnBlock(in_channels=self.ch*self.ch_mult[-2], z_channels=self.z_channels) # B,2ch,8,7
        
        # up - block 2
        self.up_2 = nn.Module()
        self.up_2.block_1 = ConvBlock(in_channels=self.ch*self.ch_mult[-2], num_groups=1, out_channels=self.ch*self.ch_mult[-2], dropout=dropout) # B,2ch,8,7
        self.up_2.upsample = Upsample2x_TF(in_channels=self.ch*self.ch_mult[-2]) # B,2ch,16,7
        self.up_2.block_2 = ConvBlock(in_channels=self.ch*self.ch_mult[-2], num_groups=1, out_channels=self.ch, dropout=dropout) # B,ch,16,7

        # end
        self.conv_out = torch.nn.Conv2d(self.ch, self.in_channels, kernel_size=(3,1), stride=(1,1), padding=(1,0)) # B,1,16,7

    def forward(self, z):
        """
        @input:
        h has shape [batch_size, z_channels, 4, act_dim]
        """

        # # action
        # h = self.action_mlp(h) # [B,z_channels,4,act_dim]
        
        # # change dim
        # h = torch.transpose(h, 1, 3) # [B,z_channels,4,act_dim]

        # begin
        h = self.conv_in(F.silu(self.norm_in(z), inplace=True)) # [B,4ch,4,act_dim]
        
        # # up-1
        # h = self.attn_1(h) # [B,4ch,4,act_dim]

        # up-1
        h = self.up_1.block_2(self.up_1.upsample(self.up_1.block_1(h))) # [B,2ch,8,act_dim]

        # # up-2
        # h = self.attn_2(h) # [B,2ch,8,act_dim]

        # up-2
        h = self.up_2.block_2(self.up_2.upsample(self.up_2.block_1(h))) # [B,ch,16,act_dim]

        # end
        h = self.conv_out(h) # [B,1,16,act_dim]

        return h # h has shape [B,1,16,act_dim]

