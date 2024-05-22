import torch
import torch.nn as nn
import torch.nn.functional as F
##########################################################s######
#  1d fourier layer
################################################################

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, modes ), (in_channel, out_channel, modes) -> (batch, out_channel, modes)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x) # (batch, in_channels, x) -> (batch, in_channels, x//2 + 1)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FourierBasicBlock(nn.Module):
    """Basic block for Fourier Neural Operators

    Args:
        in_planes (int): number of input channels
        planes (int): number of output channels
        modes1 (int, optional): number of modes for the first spatial dimension. Defaults to 16.
    """

    def __init__(
        self,
        in_planes: int,
        planes: int,
        modes1: int = 16,
    ):
        super().__init__()
        self.modes1 = modes1
        self.fourier1 = SpectralConv1d(in_planes, planes, modes1=self.modes1)
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=1, padding=0, padding_mode="zeros", bias=True)
        self.fourier2 = SpectralConv1d(planes, planes, modes1=self.modes1)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=1, padding=0, padding_mode="zeros", bias=True)

        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.fourier1(x) # (batch, in_planes, x) -> (batch, planes, x)
        x2 = self.conv1(x)
        out = self.activation(x1 + x2)

        x1 = self.fourier2(out)
        x2 = self.conv2(out)
        out = x1 + x2
        out = self.activation(out)
        return out

class FNO1d_bundled(nn.Module):
    def __init__(self,
                time_window: int = 25,
                 modes: int = 32,
                 width: int = 256,
                 num_layers: int = 5):
        super(FNO1d_bundled, self).__init__()
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
                FourierBasicBlock(self.width, self.width, modes1=self.modes)
                for i in range(num_layers)
            ]
        )

        self.activation = nn.GELU()

    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor of shape [batch, time_window, x]
        Returns: torch.Tensor: output has the shape [batch, time_window x]
        """
        x = self.activation(self.conv_in1(x)) # (batch, time_window, x) -> (batch, width, x)
        x = self.activation(self.conv_in2(x)) # (batch, width, x) -> (batch, width, x)

        for layer in self.layers:
            x = layer(x) # (batch, width, x) -> (batch, width, x)

        x = self.activation(self.conv_out1(x))
        x = self.conv_out2(x)

        return x



        

      