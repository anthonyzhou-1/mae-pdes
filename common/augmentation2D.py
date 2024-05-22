import numpy as np
import torch
import random
from typing import Optional, Tuple
from einops import repeat
import torch.nn.functional as F

def fourier_shift(u: torch.Tensor, eps: float=0., dim: int=-1, order: int=0) -> torch.Tensor:
    """
    Shift in Fourier space.
    Args:
        u (torch.Tensor): input tensor, usually of shape [batch, t, x]
        eps (float): shift parameter
        dim (int): dimension which is used for shifting
        order (int): derivative order
    Returns:
        torch.Tensor: Fourier shifted input
    """
    n = u.shape[dim]
    u_hat = torch.fft.rfft(u, dim=dim, norm='ortho')
    # Fourier modes
    omega = torch.arange(n // 2 + 1)
    if n % 2 == 0:
        omega[-1] *= 0
    # Applying Fourier shift according to shift theorem
    fs = torch.exp(- 2 * np.pi * 1j * omega * eps)
    # For order>0 derivative is taken
    if order > 0:
        fs = (- 2 * np.pi * 1j * omega) ** order * fs

    if dim == -2:
        extra_size = u_hat.shape[-1]
        fs = repeat(fs, 'b n ->b 1 n x', x=extra_size)
    elif dim == -1:
        extra_size = u_hat.shape[-2]
        fs = repeat(fs, 'b n ->b 1 x n', x=extra_size)

    output =  torch.fft.irfft(fs * u_hat, n=n, dim=dim, norm='ortho')
    return output

def fourier_shift_batched(u: torch.Tensor, eps: torch.Tensor, dim:int=-1, order = 0) -> torch.Tensor:
    '''
    Shift in Fourier space for batched inputs.
    Args:
        u (torch.Tensor): input tensor, usually of shape [batch, t, x]
        eps (float): shift parameter in shape [batch, t]
        dim (int): dimension which is used for shifting
        order (int): derivative order
    Returns:
        torch.Tensor: Fourier shifted input
    '''

    n = u.shape[dim]
    u_hat = torch.fft.rfft(u, dim=dim, norm='ortho')
    # Fourier modes
    omega = torch.arange(n // 2 + 1, device=u.device, dtype=torch.float32)
    if n % 2 == 0:
        omega[-1] *= 0
    # Applying Fourier shift according to shift theorem
    fs = torch.exp(- 2 * np.pi * 1j * torch.matmul(eps[..., None], omega.unsqueeze(0))) # [batch, nt, n//2 + 1]

    # For order>0 derivative is taken
    if order > 0:
        fs = (- 2 * np.pi * 1j * omega) ** order * fs

    if dim == -2:
        extra_size = u_hat.shape[-1]
        fs = repeat(fs, 'b 1 n -> b 1 n x', x=extra_size)
    elif dim == -1:
        extra_size = u_hat.shape[-2]
        fs = repeat(fs, 'b 1 n -> b 1 x n', x=extra_size)

    out = torch.fft.irfft(fs * u_hat, n=n, dim=dim, norm='ortho')
    return out 

def linear_shift(u: torch.Tensor, eps: float=0., dim:int=-1) -> torch.Tensor:
    """
    Linear shift.
    Args:
        u (torch.Tensor): input tensor, usually of shape [batch, t, x]
        eps (float): shift parameter
        dim (int): dimension which is used for shifting
    Returns:
        Linear shifted input
    """
    n = u.shape[dim]
    # Shift to the left and to the right and interpolate linearly
    q, r = torch.div(eps*n, 1, rounding_mode='floor'), (eps * n) % 1
    q_left, q_right = q/n, (q+1)/n
    u_left = fourier_shift(u, eps=q_left, dim=-1)
    u_right = fourier_shift(u, eps=q_right, dim=-1)
    return (1-r) * u_left + r * u_right

class SpaceTranslate:
    def __init__(self, max_shift: float=1., dim: str = 'x'):
        """
        Instantiate sub-pixel space translation.
        Translations are drawn from the distribution.
        Uniform(-max_shift/2, max_shift/2) where max_shift is in units of input side length.
        Args:
            max_shift (float): maximum shift length (rotations)
            dim (str): Spatial dimension to shift
        """
        self.max_shift = max_shift
        self.dim = dim

    def __call__(self, u: torch.Tensor, eps: Optional[float]=None, shift: str='fourier', batched=False) -> torch.Tensor:
        """
        Sub-pixel space translation shift.
        Args:
            sample (torch.Tensor): input tensor of the form [u, X]
            eps (float): shift parameter
            shift (str): fourier or linear shift
        Returns:
            torch.Tensor: sub-pixel shifted tensor of the form [u, X]
        """
        if self.dim == 'x':
            dim = 2
        elif self.dim == 'y':
            dim = 1

        if not batched:
            if eps is None:
                eps = self.max_shift * (torch.rand((), device=u.device) - 0.5)
            else:
                eps = eps * torch.ones((), device=u.device)

            if shift == 'fourier':
                output = fourier_shift(u, eps=eps, dim=-1 * dim)
            elif shift == 'linear':
                output = linear_shift(u, eps=eps, dim=-1 * dim)
        
        else:
            batch_size = u.shape[0]
            nt = u.shape[1]
            if eps is None:
                eps = self.max_shift * (torch.rand((batch_size, 1), device=u.device) - 0.5)
            else:
                eps = eps * torch.ones((batch_size, 1), device=u.device)

            #eps = repeat(eps, 'b 1 -> b nt', nt=nt)

            if shift == 'fourier':
                output = fourier_shift_batched(u, eps=eps, dim=-1 * dim)
            else:
                raise NotImplementedError

        return output
    
class NodalScale:
    def __init__(self, max_scale: float=1.) -> torch.Tensor:
        """
        Instantiate nodal multiplication generator.
        Nodal scales are drawn from the distribution
            Uniform(1-max_scale, 1+max_scale) where max_scale is in units of input side length.
        Args:
            max_scale: float for maximum scale
        """
        self.max_scale = max_scale

    def __call__(self, u: torch.Tensor, eps: Optional[float]=None, shift: str='fourier', batched=False) -> torch.Tensor:
        """
        Nodal scale.
        Args:
            sample (torch.Tensor): input tensor of the form [u, X]
            eps (float): scale parameter
        Returns:
            torch.Tensor: Nodal scaled tensor of the form [u, X]
        """

        if eps is None:
            eps = self.max_scale * (torch.rand((), device=u.device) - 0.5)
        else:
            eps = eps * torch.ones((), device=u.device)

        # operation is already batched
        output = torch.exp(eps) * u

        return output

class NodalShift:
    def __init__(self, max_shift: float=1.) -> torch.Tensor:
        """
        Instantiate nodal shift generator.
        Nodal shifts are drawn from the distribution
            Uniform(-max_shift/2, max_shift/2) where max_shift is in units of input side length.
        Args:
            max_shift: float for maximum shift length
            dim: Spatial dimension to shift
        """
        self.max_shift = max_shift

    def __call__(self, u: torch.Tensor, eps: Optional[float]=None, shift: str='fourier', batched=False) -> torch.Tensor:
        """
        Nodal shift.
        Args:
            sample (torch.Tensor): input tensor of the form [u, X]
            eps (float): shift parameter
            shift (str): fourier or linear shift
        Returns:
            torch.Tensor: Nodal shifted tensor of the form [u, X]
        """
        if eps is None:
            eps = self.max_shift * (torch.rand((), device=u.device) - 0.5)
        else:
            eps = eps * torch.ones((), device=u.device)

        # operation is already batched
        output = u + eps

        return output

class Heat2D_Augmentation:
    def __init__(self, max_shift: float = 1.0, max_scale: float = 1.0, max_shift_nodal: float = 1.0):
        """
        Instantiate 2D Heat data augmentation.
        Args:
            max_x_shift (float): parameter of sub-pixel space translation
            max_velocity (float): parameter of Galilean transformation
            max_scale (float): parameter of scaling transformation

        Use a subset of Heat Lie group transformations, namely:
            - dx : shift x
            - dy : shift y
            - b(t, x) * du
                - if b = 1, this is a nodal shift
            - u * du : nodal scale

        """
        self.generators = [SpaceTranslate(max_shift, dim = 'x'),
                           SpaceTranslate(max_shift, dim = 'y'),
                           NodalScale(max_scale),
                           NodalShift(max_shift_nodal)]

    def __call__(self, u: torch.Tensor, shift: str='fourier', batched=False) -> torch.Tensor:
        """
        Data augmentation, evoking one generator after each other.
        Args:
            u (torch.Tensor): input tensor of the form [u, X]
            shift (str): fourier or linear shift (not used, only for consistency w.r.t. other generators)
        Returns:
            torch.Tensor
        """
        for g in self.generators:
            u = g(u, shift=shift, batched=batched)
        return u
    
class Advection2D_Augmentation:
    def __init__(self, max_shift: float = 1.0, max_scale: float = 1.0, max_shift_nodal: float = 1.0):
        """
        Instantiate 2D Advection data augmentation.
        Args:
            max_shift (float): parameter of sub-pixel space translation
            max_velocity (float): parameter of Galilean transformation
            max_scale (float): parameter of scaling transformation
            max_shift_nodal (float): parameter of nodal shift

        Use a subset of Heat Lie group transformations defined by:
        theta*dt + [psi - c*theta]dx + [chi - c*theta]dy + phi*du
            - dx : shift x
                - set theta = phi = chi = 0, psi = 1
            - dy : shift y
                - set theta = phi = psi = 0, chi = 1
            - du : shift u
                - set theta = chi = psi = 0, phi = 1
            - u * du : nodal scale
                - set theta = chi = psi = 0, phi = u

        """
        self.generators = [SpaceTranslate(max_shift, dim = 'x'),
                           SpaceTranslate(max_shift, dim = 'y'),
                           NodalScale(max_scale),
                           NodalShift(max_shift_nodal)]

    def __call__(self, u: torch.Tensor, shift: str='fourier', batched=False) -> torch.Tensor:
        """
        Data augmentation, evoking one generator after each other.
        Args:
            u (torch.Tensor): input tensor of the form [u, X]
            shift (str): fourier or linear shift (not used, only for consistency w.r.t. other generators)
        Returns:
            torch.Tensor
        """
        for g in self.generators:
            u = g(u, shift=shift, batched=batched)
        return u
    
class Burgers2D_Augmentation:
    def __init__(self, max_shift: float = 1.0):
        """
        Instantiate 2D Advection data augmentation.
        Args:
            max_shift (float): parameter of sub-pixel space translation
            max_velocity (float): parameter of Galilean transformation
            max_scale (float): parameter of scaling transformation
            max_shift_nodal (float): parameter of nodal shift

        Use a subset of Burgers Lie group transformations defined by:
            - dx : shift x
            - dy : shift y
        Not many implemented, but can add more Lie groups if needed
        """

        self.generators = [SpaceTranslate(max_shift, dim = 'x'),
                           SpaceTranslate(max_shift, dim = 'y')]

    def __call__(self, u: torch.Tensor, shift: str='fourier', batched=False) -> torch.Tensor:
        """
        Data augmentation, evoking one generator after each other.
        Args:
            u (torch.Tensor): input tensor of the form [u, X]
            shift (str): fourier or linear shift (not used, only for consistency w.r.t. other generators)
        Returns:
            torch.Tensor
        """
        for g in self.generators:
            u = g(u, shift=shift, batched=batched)
        return u
    
class Combined2D_Augmentation:
    def __init__(self, max_shift:float = 1.0, max_scale:float = 1.0, max_shift_nodal:float = 1.0):
        """
        Instantiate 2D combined data augmentation
        Args:
            heat_aug: Heat2D_Augmentation
            advection_aug: Advection2D_Augmentation
            burgers_aug: Burgers2D_Augmentation
        """
        # self.heat_aug = Heat2D_Augmentation(max_shift, max_scale, max_shift_nodal)
        # self.advection_aug = Advection2D_Augmentation(max_shift, max_scale, max_shift_nodal)
        # only use burgers aug since doing batched augmentations (can't augment each sample individually!)
        self.burgers_aug = Burgers2D_Augmentation(max_shift)
    
    def __call__(self, u: torch.Tensor, shift: str='fourier', batched = False) -> torch.Tensor:
        """
        Data augmentation, evoking one generator after each other.
        Args:
            u (torch.Tensor): input tensor of the form [u, X]
            pde (str): pde string
            shift (str): fourier or linear shift (not used, only for consistency w.r.t. other generators)
        Returns:
            torch.Tensor
        """
        u = self.burgers_aug(u, shift=shift, batched=batched)
        return u
    
class AugmentationWrapper2D:
    def __init__(self, augmentation, augmentation_ratio, shift='fourier', batched=True, sizes=[]) -> None:
        self.augmentation = augmentation
        self.augmentation_ratio = augmentation_ratio
        self.shift = shift
        self.batched = batched 
        self.sizes = sizes

    def map_size(self, size, batch_size):
        if size[0] == 48:
            cls = 0
        elif size[0] == 52:
            cls = 1
        elif size[0] == 56:
            cls = 2
        elif size[0] == 60:
            cls = 3
        elif size[0] == 64:
            cls = 4
        else:
            raise ValueError("Size not found")
        labels = cls * torch.ones(batch_size, dtype=torch.long)
        return labels

    def __call__(self, u: torch.Tensor, label: bool = False) -> torch.Tensor:
        """
        Augmentation wrapper.
        Args:
            u (torch.Tensor): input tensor 
            variables (list): list of PDE variables
        Returns:
            torch.Tensor: augmented trajectory
        """

        if self.augmentation_ratio > random.random(): 
            u = self.augmentation(u, self.shift, batched=self.batched)

        if len(self.sizes) > 0:
            # u is in shape [b, nt, nx, ny]
            # Goal is to downsample nx, ny to a size 

            rand_idx = random.randint(0, len(self.sizes) - 1)
            new_size = self.sizes[rand_idx] #[newx, newy]

            u = F.interpolate(u, size=new_size, mode='bicubic')

        if label == True and len(self.sizes) > 0:
            labels = self.map_size(new_size, u.shape[0])
            return u, labels

        return u
