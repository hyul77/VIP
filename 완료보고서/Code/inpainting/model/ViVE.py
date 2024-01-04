import numpy as np
import time
import math
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from core.spectral_norm import spectral_norm as _spectral_norm



# STTN 모델 그대로 Network 활용
class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).' % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)


'''
Image를 받았을 때 Encoder 부분을 추가해야 feature를 더 뽑을 수 있음
'''
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ELU(inplace=True)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.relu3 = nn.ELU(inplace=True)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ELU(inplace=True)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ELU(inplace=True)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu6 = nn.ELU(inplace=True)
        self.conv7 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.relu7 = nn.ELU(inplace=True)
        self.conv8 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.relu8 = nn.ELU(inplace=True)

    def forward(self, x):
        batch, channels, height, width = x.size()
        height, width = height // 4, width // 4
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        x = self.relu6(self.conv6(x))
        x = self.relu7(self.conv7(x))
        x = self.relu8(self.conv8(x))
        return x
    

class InpaintingGenerator_V7(BaseNetwork):
    def __init__(self, init_weights=True):
        super(InpaintingGenerator_V7, self).__init__()
        channel = 256
        hidden = 512
        stack_num = 8
        num_head = 4
        kernel_size = (7, 7)
        padding = (3, 3)
        stride = (3, 3)
        output_size = (60, 108)
        blocks = []
        dropout = 0.
        t2t_params = {'kernel_size': kernel_size, 'stride': stride, 'padding': padding, 'output_size': output_size}
        n_vecs = 1
        for i, d in enumerate(kernel_size):
            n_vecs *= int((output_size[i] + 2 * padding[i] - (d - 1) - 1) / stride[i] + 1)
        for _ in range(stack_num):
            blocks.append(TransGanBlock(hidden=hidden, num_head=num_head, dropout=dropout, n_vecs=n_vecs,
                                           t2t_params=t2t_params))
        self.transformer = nn.Sequential(*blocks)
        self.Split = Split(channel // 2, hidden, kernel_size, stride, padding, dropout=dropout)
        self.add_pos_emb = AddPositionalEmbedding(n_vecs, hidden)
        self.sc = Completion(channel // 2, hidden, output_size, kernel_size, stride, padding)

        self.encoder = Encoder()

        # decoder
        self.decoder = nn.Sequential(
            deconv_block(channel // 2, 128, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            deconv_block(64, 64, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        )

        if init_weights:
            self.init_weights()

    def forward(self, masked_frames):
        # extracting features
        b, t, c, h, w = masked_frames.size()
        time0 = time.time()
        enc_feat = self.encoder(masked_frames.view(b * t, c, h, w))
        _, c, h, w = enc_feat.size()
        # 추가
        trans_feat = self.Split(enc_feat, b)
        trans_feat = self.add_pos_emb(trans_feat)
        trans_feat = self.transformer(trans_feat)
        trans_feat = self.sc(trans_feat, t)

        enc_feat = enc_feat + trans_feat
        output = self.decoder(enc_feat)
        output = torch.tanh(output)
        return output
    

    def infer(self, feat, masks):
        t, c, h, w = masks.size()
        masks = masks.view(t, c, h, w)
        masks = F.interpolate(masks, scale_factor=1.0/4)
        t, c, _, _ = feat.size()
        enc_feat = self.transformer(
            {'x': feat, 'm': masks, 'b': 1, 'c': c})['x']
        return enc_feat
'''    
class InpaintingGenerator_V10(BaseNetwork):
    def __init__(self, init_weights=True):
        super(InpaintingGenerator_V10, self).__init__()
        channel = 256
        hidden = 512
        stack_num = 8
        num_head = 4
        kernel_size = (10, 10)
        padding = (3, 3)
        stride = (3, 3)
        output_size = (60, 108)
        blocks = []
        dropout = 0.
        t2t_params = {'kernel_size': kernel_size, 'stride': stride, 'padding': padding, 'output_size': output_size}
        n_vecs = 1
        for i, d in enumerate(kernel_size):
           n_vecs *= int((output_size[i] + 2 * padding[i] - (d - 1) - 1) / stride[i] + 1) if d != 1 else 1
        for _ in range(stack_num):
            blocks.append(TransGanBlock(hidden=hidden, num_head=num_head, dropout=dropout, n_vecs=n_vecs,
                                        t2t_params=t2t_params))
        self.transformer = nn.Sequential(*blocks)
        self.Split = Split(channel // 2, hidden, kernel_size, stride, padding, dropout=dropout)
        self.add_pos_emb = AddPositionalEmbedding(n_vecs, hidden)
        self.sc = Completion(channel // 2, hidden, output_size, kernel_size, stride, padding)

        self.encoder = Encoder()
'''

class deconv_block(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel,
                              kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                          align_corners=True)
        return self.conv(x)


# #############################################################################
# ############################# PATCH  ########################################
# #############################################################################


class AddPositionalEmbedding(nn.Module):
    def __init__(self, n, c):
        super(AddPositionalEmbedding, self).__init__()
        self.pos_emb = nn.Parameter(torch.zeros(1, 1, n, c).float().normal_(mean=0, std=0.02), requires_grad=True)
        self.num_vecs = n

    def forward(self, x):
        b, n, c = x.size()
        x = x.view(b, -1, self.num_vecs, c)
        x = x + self.pos_emb
        x = x.view(b, n, c)
        return x


class Split(nn.Module):
    def __init__(self, channel, hidden, kernel_size, stride, padding, dropout=0.1):
        super(Split, self).__init__()
        self.kernel_size = kernel_size
        self.t2t = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding)
        c_in = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(c_in, hidden)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, b):
        feat = self.t2t(x)
        feat = feat.permute(0, 2, 1)
        feat = self.embedding(feat)
        feat = feat.view(b, -1, feat.size(2))
        feat = self.dropout(feat)
        return feat


class Completion(nn.Module):
    def __init__(self, channel, hidden, output_size, kernel_size, stride, padding):
        super(Completion, self).__init__()
        self.relu = nn.ELU(inplace=True)
        c_out = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(hidden, c_out)
        self.t2t = torch.nn.Fold(output_size=output_size, kernel_size=kernel_size, stride=stride, padding=padding)
        h, w = output_size
        self.bias = nn.Parameter(torch.zeros((channel, h, w), dtype=torch.float32), requires_grad=True)

    def forward(self, x, t):
        feat = self.embedding(x)
        b, n, c = feat.size()
        feat = feat.view(b * t, -1, c).permute(0, 2, 1)
        feat = self.t2t(feat) + self.bias[None]
        return feat



# #############################################################################
# ############################# Transformer  ##################################
# #############################################################################



class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def __init__(self, p=0.1):
        super(Attention, self).__init__()
        self.dropout = nn.Dropout(p=p)

    def forward(self, query, key, value, m=None):
        scores = self.compute_attention_scores(query, key, m)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        p_val = torch.matmul(p_attn, value)
        return p_val, p_attn

    def compute_attention_scores(self, query, key, m=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if m is not None:
            scores.masked_fill_(m, -1e9)
        return scores
    

class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, d_model, head, p=0.1):
        super().__init__()
        self.query_embedding = nn.Linear(d_model, d_model)
        self.value_embedding = nn.Linear(d_model, d_model)
        self.key_embedding = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention(p=p)
        self.head = head

    def forward(self, x):
        b, n, c = x.size()
        c_h = c // self.head
        key, query, value = map(lambda emb: emb(x).view(b, n, self.head, c_h).permute(0, 2, 1, 3),
                               [self.key_embedding, self.query_embedding, self.value_embedding])
        att, _ = self.attention(query, key, value)
        output = self.output_linear(att.permute(0, 2, 1, 3).contiguous().view(b, n, c))
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, p=0.1):
        super(FeedForward, self).__init__()
        # We set d_ff as a default to 2048
        self.conv = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(p=p))

    def forward(self, x):
        x = self.conv(x)
        return x


class MixForward(nn.Module):
    def __init__(self, d_model, p=0.1, n_vecs=None, t2t_params=None):
        super(MixForward, self).__init__()
        
        hd = 1960
        self.conv1 = nn.Sequential(
            nn.Linear(d_model, hd))
        self.conv2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
            nn.Linear(hd, d_model),
            nn.Dropout(p=p))
        assert t2t_params is not None and n_vecs is not None
        tp = t2t_params.copy()
        self.fold = nn.Fold(**tp)
        del tp['output_size']
        self.unfold = nn.Unfold(**tp)
        self.n_vecs = n_vecs

    def forward(self, x):
        x = self.conv1(x)
        b, n, c = x.size()
        normalizer = x.new_ones(b, n, 49).view(-1, self.n_vecs, 49).permute(0, 2, 1)
        x = self.unfold(self.fold(x.view(-1, self.n_vecs, c).permute(0, 2, 1)) / self.fold(normalizer)).permute(0, 2,
                                                                                                                1).contiguous().view(
            b, n, c)
        x = self.conv2(x)
        return x


class TransGanBlock(nn.Module):

    def __init__(self, hidden=128, num_head=4, dropout=0.1, n_vecs=None, t2t_params=None):
        super().__init__()
        self.attention = MultiHeadedAttention(d_model=hidden, head=num_head, p=dropout)
        self.ffn = MixForward(hidden, p=dropout, n_vecs=n_vecs, t2t_params=t2t_params)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input):
        x = self.norm1(input)
        x = input + self.dropout(self.attention(x))
        y = self.norm2(x)
        x = x + self.ffn(y)
        return x


# ######################################################################
# ####################   Discriminator   ###############################
# ######################################################################

class Discriminator(BaseNetwork):
    def __init__(self, in_channels=3, use_sigmoid=False, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        nf = 32

        D_block = lambda in_channels, out_channels, kernel_size, stride, padding: nn.Sequential(
            spectral_norm(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=not use_spectral_norm), use_spectral_norm),
            nn.ELU(inplace=True)
        )
        # in_channels, out_channels, kernel_size, stride, padding
        self.conv = nn.Sequential(
            D_block(in_channels, nf * 1, (3, 5, 5), (1, 2, 2), (1, 2, 2)),
            D_block(nf * 1, nf * 2, (3, 5, 5), (1, 2, 2), (1, 2, 2)),
            D_block(nf * 2, nf * 4, (3, 5, 5), (1, 2, 2), (1, 2, 2)),
            D_block(nf * 4, nf * 4, (3, 5, 5), (1, 2, 2), (1, 2, 2)),
            D_block(nf * 4, nf * 4, (3, 5, 5), (1, 2, 2), (1, 2, 2)),
            D_block(nf * 4, nf * 4, (3, 5, 5), (1, 2, 2), (1, 2, 2)),
            nn.Conv3d(nf * 4, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2))
        )

        if init_weights:
            self.init_weights()

    def forward(self, xs):
        xs_t = xs.transpose(0, 1).unsqueeze(0)  # B, C, T, H, W
        feat = self.conv(xs_t)
        if self.use_sigmoid:
            feat = torch.sigmoid(feat)
        out = feat.transpose(1, 2) 
        return out



def spectral_norm(module, mode=True):
    if mode:
        return _spectral_norm(module)
    return module


# #############################################################################
# ############################# Test More  ####################################
# #############################################################################

'''
TransGan으로 형성된 이미지는 kernel크기의 고정으로 인해 패턴이 생성되는걸 볼 수 있었다.
패턴의 형성으로 생성이미지에 대한 완성도가 크게 떨어졌다.
kernel 크기를 2개로, (7,7) (10,10)으로 만들어서 2개의 경우에 돌아가게끔 형성을 하면
패턴 형성을 회피할 수 있다. 

패턴이 형성되면 그 이후로 train을 해도 결과는 유지 되기 때문에 그 패턴을 깨야만
더 좋은 성능을 구사할 수 있다.
'''

'''
discriminator = Discriminator()
input_tensor = torch.rand((1, 3, 64, 128, 128))
output_tensor = discriminator(input_tensor)
print("Input Tensor Shape:", input_tensor.shape)
print("Output Tensor Shape:", output_tensor.shape)
'''


