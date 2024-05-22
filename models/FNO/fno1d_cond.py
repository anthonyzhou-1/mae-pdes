import torch
from torch import nn

def batchmul1d(input, weights, emb):
    '''
    args:
        input: (batch, c_in, modes)
        weights: (c_in, c_out, modes)
        emb: (batch, modes) ? unclear
    returns:
        out: (batch, c_out, x)
    '''

    temp = input * emb.unsqueeze(1)

    # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
    out = torch.einsum("bix,iox->box", temp, weights)
    return out

class FreqLinear(nn.Module):
    def __init__(self, in_channel, modes1):
        super().__init__()
        self.modes1 = modes1
        scale = 1 / (in_channel + 2 * modes1)
        self.weights = nn.Parameter(scale * torch.randn(in_channel, 2 * modes1, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(1, 2 * modes1, dtype=torch.float32))

    def forward(self, x):
        # x in shape (batch, cond_channels)
        # returns spectral embedding in shape (batch, modes1) 

        B = x.shape[0]
        h = torch.einsum("tc,cm->tm", x, self.weights) + self.bias
        h = h.reshape(B, self.modes1, 2)
        return torch.view_as_complex(h)


class SpectralConv1d_cond(nn.Module):
    def __init__(self, in_channels, out_channels, cond_channels, modes1):
        super().__init__()
        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        @author: Zongyi Li
        [paper](https://arxiv.org/pdf/2010.08895.pdf)
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

        self.cond_emb = FreqLinear(cond_channels, self.modes1)

    def forward(self, x, emb):
        '''
        args:
            x: (batch, in_channels, x)
            emb: (batch, cond_channels)
        '''
        emb = self.cond_emb(emb) # (batch, cond_channels) -> (batch, modes1)
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x) # (batch, in_channels, x) -> (batch, in_channels, x//2 + 1)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize, 
            self.out_channels, 
            x.size(-1)//2 + 1,  
            device=x.device, 
            dtype=torch.cfloat
        )
        out_ft[:, :, :self.modes1] = batchmul1d(x_ft[:, :, :self.modes1], self.weights1, emb) 

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x
    
class FourierBasicBlock(nn.Module):

    def __init__(
        self,
        in_planes: int,
        planes: int,
        cond_channels: int,
        modes1: int = 16,
    ) -> None:
        super().__init__()
        self.modes1 = modes1
        self.activation = nn.GELU()

        self.fourier1 = SpectralConv1d_cond(in_planes, planes, cond_channels, modes1=self.modes1)
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=1, padding=0, padding_mode="zeros", bias=True)
        self.fourier2 = SpectralConv1d_cond(planes, planes, cond_channels, modes1=self.modes1)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=1, padding=0, padding_mode="zeros", bias=True)
        self.cond_emb = nn.Linear(cond_channels, planes)

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        x1 = self.fourier1(x, emb)
        x2 = self.conv1(x)
        emb_out = self.cond_emb(emb)
        while len(emb_out.shape) < len(x2.shape):
            emb_out = emb_out[..., None]

        out = self.activation(x1 + x2 + emb_out)
        x1 = self.fourier2(out, emb)
        x2 = self.conv2(out)
        out = x1 + x2
        out = self.activation(out)
        return out
    
class FNO1d_bundled_cond(nn.Module):
    def __init__(self,
                time_window: int = 25,
                 modes: int = 32,
                 width: int = 256,
                 cond_channels: int = 64,
                 num_layers: int = 5):
        super(FNO1d_bundled_cond, self).__init__()
        """
        Args:
            time_window (int): input/output timesteps of the trajectory
            modes (int): low frequency Fourier modes considered for multiplication in the Fourier space
            width (int): hidden channel dimension
            num_layers (int): number of FNO layers
        """
        self.modes = modes
        self.width = width
        self.in_channels = time_window
        self.out_channels = time_window

        self.conv_in1 = nn.Conv1d(
            self.in_channels,
            self.width,
            kernel_size=1,
            bias=True,
        )
        self.conv_in2 = nn.Conv1d(
            self.width,
            self.width,
            kernel_size=1,
            bias=True,
        )
        self.conv_out1 = nn.Conv1d(
            self.width,
            self.width,
            kernel_size=1,
            bias=True,
        )
        self.conv_out2 = nn.Conv1d(
            self.width,
            self.out_channels,
            kernel_size=1,
            bias=True,
        )

        self.layers = nn.ModuleList(
            [
                FourierBasicBlock(self.width, self.width, cond_channels, modes1=self.modes)
                for i in range(num_layers)
            ]
        )

        self.activation = nn.GELU()

    def forward(self, 
                x: torch.Tensor,
                emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor of shape [batch, time_window, x]
            emb (torch.Tensor): input tensor of shape [batch, cond_channels]
        Returns: torch.Tensor: output has the shape [batch, time_window x]
        """
        x = self.activation(self.conv_in1(x)) # (batch, time_window, x) -> (batch, width, x)
        x = self.activation(self.conv_in2(x)) # (batch, width, x) -> (batch, width, x)

        for layer in self.layers:
            x = layer(x, emb) # (batch, width, x) -> (batch, width, x)

        x = self.activation(self.conv_out1(x))
        x = self.conv_out2(x)

        return x