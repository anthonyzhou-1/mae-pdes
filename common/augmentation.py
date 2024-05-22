import numpy as np
import torch
from typing import Optional
import random
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
    assert dim < 0
    n = u.shape[dim]
    u_hat = torch.fft.rfft(u, dim=dim, norm='ortho')
    # Fourier modes
    omega = torch.arange(n // 2 + 1, device=u.device, dtype=torch.float32)
    if n % 2 == 0:
        omega[-1] *= 0
    # Applying Fourier shift according to shift theorem
    fs = torch.exp(- 2 * np.pi * 1j * omega * eps)
    # For order>0 derivative is taken
    if order > 0:
        fs = (- 2 * np.pi * 1j * omega) ** order * fs
    for _ in range(-dim - 1):
        fs = fs[..., None]
    return torch.fft.irfft(fs * u_hat, n=n, dim=dim, norm='ortho')

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
    for _ in range(-dim - 1):
        fs = fs[..., None]
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
    def __init__(self, max_x_shift: float=1.):
        """
        Instantiate sub-pixel space translation.
        Translations are drawn from the distribution.
        Uniform(-max_shift/2, max_shift/2) where max_shift is in units of input side length.
        Args:
            max_shift (float): maximum shift length (rotations)
        """
        self.max_x_shift = max_x_shift

    def __call__(self, sample: torch.Tensor, eps: Optional[float]=None, shift: str='fourier', batched=False) -> torch.Tensor:
        """
        Sub-pixel space translation shift.
        Args:
            sample (torch.Tensor): input tensor of the form [u, X]
            eps (float): shift parameter
            shift (str): fourier or linear shift
        Returns:
            torch.Tensor: sub-pixel shifted tensor of the form [u, X]
        """
        u, dx, dt = sample
        if not batched:
            if eps is None:
                eps = self.max_x_shift * (torch.rand((), device=u.device) - 0.5)
            else:
                eps = eps * torch.ones((), device=u.device)

            if shift == 'fourier':
                output = (fourier_shift(u, eps=eps, dim=-1), dx, dt)
            elif shift == 'linear':
                output = (linear_shift(u, eps=eps, dim=-1), dx, dt)
        else:
            batch_size = u.shape[0]
            nt = u.shape[1]
            if eps is None:
                eps = self.max_x_shift * (torch.rand((batch_size, 1), device=u.device) - 0.5)
            else:
                eps = eps * torch.ones((batch_size, 1), device=u.device)

            eps = repeat(eps, 'b 1 -> b nt', nt=nt)
            if shift == 'fourier':
                output = (fourier_shift_batched(u, eps=eps, dim=-1), dx, dt)
            else:
                raise NotImplementedError

        return output


class Galileo:
    def __init__(self, max_velocity: float=1) -> torch.Tensor:
        """
        Instantiate Galileo generator.
        Galilean transformations are drawn from the distribution
            Uniform(-max_velocity, max_velocity) where max_velocity is in units of m/s.
        Args:
            max_velocity: float for maximum velocity in m/s.
        """
        self.max_velocity = max_velocity

    def __call__(self, sample: torch.Tensor, eps: Optional[float]=None, shift: str='fourier', batched:bool = False) -> torch.Tensor:
        """
        Galilean shift.
        Args:
            sample (torch.Tensor): input tensor of the form [u, X]
            eps (float): shift parameter
            shift (str): fourier or linear shift (not used, only for consistency w.r.t. other generators)
        Returns:
            torch.Tensor: Galilean shifted tensor of the form [u, X]
        """
        u, dx, dt = sample

        T = u.shape[-2]
        N = u.shape[-1]
        t = dt * torch.arange(T, device=u.device)
        L = dx * N

        if not batched:
            if eps is None:
                eps = 2 * self.max_velocity * (torch.rand((), device=u.device) - 0.5)
            else:
                eps = eps * torch.ones((), device=u.device)
            # shift in pixel
            d = -(eps * t) / L

            if shift == 'fourier':
                output = (fourier_shift(u, eps=d[:, None], dim=-1), dx, dt)
            elif shift == 'linear':
                output = (linear_shift(u, eps=d[:, None], dim=-1), dx, dt)
        
        else:
            batch_size = u.shape[0]
            if eps is None:
                eps = 2 * self.max_velocity * (torch.rand((batch_size, 1), device=u.device) - 0.5)
            else:
                eps = eps * torch.ones((batch_size, 1), device=u.device)
            
            t = t.unsqueeze(0) # [1, nt]
            # shift in pixel
            d = -(torch.matmul(eps, t)) / L # [batch, nt]

            if shift == 'fourier':
                output = (fourier_shift_batched(u, eps=d, dim=-1), dx, dt)
            else:
                raise NotImplementedError

        return output

class KdVBurgers_augmentation:
    def __init__(self, max_x_shift: float = 1.0, max_velocity: float = 1.0):
        """
        Instantiate KS data augmentation.
        Args:
            max_x_shift (float): parameter of sub-pixel space translation
            max_velocity (float): parameter of Galilean transformation
        """
        self.generators = [SpaceTranslate(max_x_shift),
                           Galileo(max_velocity)]

    def __call__(self, u: torch.Tensor, shift: str='fourier', batched: bool=False) -> torch.Tensor:
        """
        KS data augmentation, evoking one generator after each other.
        Args:
            u (torch.Tensor): input tensor of the form [u, dx ,dt]
            shift (str): fourier or linear shift (not used, only for consistency w.r.t. other generators)
        Returns:
            torch.Tensor: new space shifted and Galilean transformed trajectory
        """

        for g in self.generators:
            u = g(u, shift=shift, batched=batched)
        return u
    
class AugmentationWrapper1D:
    def __init__(self, augmentation, augmentation_ratio, dx, dt, shift='fourier', batched=True, sizes=[]) -> None:
        self.augmentation = augmentation
        self.augmentation_ratio = augmentation_ratio
        self.shift = shift
        self.dx = dx
        self.dt = dt
        self.batched = batched
        self.sizes = sizes

    def map_size(self, size, batch_size):
        cls = size//10 - 5 # [50, 60, 70, 80, 90, 100] -> [0, 1, 2, 3, 4, 5]
        labels = cls * torch.ones(batch_size, dtype=torch.long)
        return labels 

    def __call__(self, u: torch.Tensor, label:bool = False) -> torch.Tensor:
        """
        Augmentation wrapper.
        Args:
            u (torch.Tensor): input tensor 
        Returns:
            torch.Tensor: augmented trajectory
        """

        if self.augmentation_ratio > random.random(): # augment data w/ probability augmentation_ratio
            dt = self.dt
            dx = self.dx
            sol = (u, dx, dt)
            sol = self.augmentation(sol, self.shift, batched=self.batched)
            u = sol[0]
        
        if len(self.sizes) > 0:
            # u is in shape [b, nt, nx]
            # Goal is to downsample nx to a size 

            rand_idx = random.randint(0, len(self.sizes) - 1)
            new_size = self.sizes[rand_idx]

            u = F.interpolate(u, size=new_size, mode='linear')

        if label == True and len(self.sizes) > 0:
            labels = self.map_size(new_size, u.shape[0])
            return u, labels

        return u
    
class Identity_Aug:
    def __init__(self) -> None:
        pass

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        return u
