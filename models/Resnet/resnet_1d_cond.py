import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, emb_dim = 32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv1d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv1d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv1d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv1d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv1d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.cond = nn.Linear(emb_dim, nf)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x, emb):
        emb_out = self.cond(emb)
        while len(emb_out.shape) < len(x.shape):
            emb_out = emb_out[..., None]
        x1 = self.lrelu(self.conv1(x + emb_out))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5 * 0.2 + x


class RRDB_cond(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32, emb_dim = 32):
        super(RRDB_cond, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc, emb_dim)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc, emb_dim)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc, emb_dim)

    def forward(self, x, emb):
        out = self.RDB1(x, emb)
        out = self.RDB2(out, emb)
        out = self.RDB3(out, emb)
        return out * 0.2 + x


class Resnet_1D_cond(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, emb_dim=32, gc=32):
        super(Resnet_1D_cond, self).__init__()
        RRDB_block_f = functools.partial(RRDB_cond, nf=nf, gc=gc, emb_dim=emb_dim)

        self.conv_first = nn.Conv1d(in_nc, nf, 3, 1, 1, bias=True)

        self.RRDB_trunk = nn.ModuleList([RRDB_block_f() for _ in range(nb)])
        self.trunk_conv = nn.Conv1d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.HRconv = nn.Conv1d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv1d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, emb):
        fea = self.conv_first(x)
        
        for i, layer in enumerate(self.RRDB_trunk):
            if i == 0:
                trunk = layer(fea, emb)
            else:
                trunk = layer(trunk, emb)

        trunk = self.trunk_conv(trunk)
        fea = fea + trunk
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out