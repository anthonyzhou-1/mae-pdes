import numpy as np
import torch
from utils import RandomSin
from tqdm import tqdm

def initial_conditions(A: torch.Tensor,
                       omega: torch.Tensor,
                       phi: torch.Tensor,
                       l: torch.Tensor,
                       L: float):
    """
    Return initial conditions for combined equation based on initial parameters
    Args:
        A (torch.Tensor): amplitude of different sine waves
        omega (torch.Tensor): time-dependent frequency
        phi (torch.Tensor): phase shift of different sine waves
        l (torch.Tensor): frequency of sine waves
    Returns:
        Callable[[torch.Tensor], torch.Tensor]: function which initializes for chosen set of parameters
    """
    def fnc(x, t=0):
        u = torch.zeros_like(x)
        for a, o, p, l_i in zip(A, omega, phi, l):
            u += a * torch.sin(o*t + (2 * np.pi * l_i * x / L) + p)

        return u
    return fnc

def params(J: int,
           seed: int = 0,
           device: torch.cuda.device="cpu",):
    """
    Get initial parameters for combined equation
    Args:
        pde (PDE): PDE at hand
        batch_size (int): batch size
        device (torch.cuda.device): device (cpu/gpu)
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A, omega, phi, l
    """
    torch.manual_seed(seed)
    A = torch.rand(J) - 0.5
    omega = 0.8 * (torch.rand(J) - 0.5)
    phi = 2.0 * np.pi * torch.rand(J)
    l = torch.randint(1, 3, (J,))
    return A.to(device), omega.to(device), phi.to(device), l.to(device)

def simulate_advection(args, a, idx):
    SAVE_STEPS = args.nt
    TOTAL_TIME = args.total_time
    nx = args.nx
    J = args.J
    length = args.length

    dt = TOTAL_TIME/SAVE_STEPS
    nt = int(np.ceil(TOTAL_TIME/dt))

    x = torch.linspace(0, length, nx)

    # Generate initial condition
    A, omega, phi, l = params(J, seed=idx, device='cpu')
    f = initial_conditions(A, omega, phi, l, length)
    u = f(x)

    all_us = torch.empty((nt, nx))
    times = torch.empty(nt)
    all_us[0] = u.clone()
    times[0] = 0

    for n in range(1, nt): ##loop across number of time steps -1 because we already have the initial condition
        x_adv = -dt*n*a
        new_x = x - x_adv

        # Sample function at new grid
        new_u = f(new_x)
        
        all_us[n] = new_u.clone()
        times[n] = TOTAL_TIME*(n)/nt

    return all_us.float(), x.float(), times.float()