import numpy as np
import torch
from typing import Tuple

class RandomSin():

    def __init__(self, 
                 shape:Tuple,
                 num_waves:int =5, 
                 max_wavenum:int = 3, 
                 length:float = 2.0,
                 device: str = 'cpu'):
        '''
        shape: tuple, shape of the grid
        num_waves: int, number of waves to sum
        max_wavenum: int, maximum wavenumber
        length: float, length of the grid
        '''
        self.shape = shape
        self.num_waves = num_waves
        self.max_wavenum = max_wavenum
        self.length = length
        self.device = device

    def sample(self, grid = None, seed = None):
        # Samples random summation of sine waves according to MP-PDE Solvers (and also loosely PDEBench)

        # Set seed if given
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Setting constants 
        J = self.num_waves
        lj = self.max_wavenum + 1
        nx = self.shape[0]
        ny = self.shape[1]
        L = self.length

        # Generate grid if not given
        if grid is None:
            x = torch.linspace(-1, 1, nx, device=self.device)
            y = torch.linspace(-1, 1, ny, device=self.device)
            [xx, yy] = torch.meshgrid(x, y)   
        else:
            xx = grid[0]
            yy = grid[1]

        u = torch.zeros((nx, ny), device=self.device)

        A = np.random.uniform(-0.5, 0.5, J)
        Kx = 2*np.pi*np.random.randint(1, lj, J)/L
        Ky = 2*np.pi*np.random.randint(1, lj, J)/L
        phi = np.random.uniform(0, 2*np.pi, J)

        A = torch.from_numpy(A).to(self.device)
        Kx = torch.from_numpy(Kx).to(self.device)
        Ky = torch.from_numpy(Ky).to(self.device)
        phi = torch.from_numpy(phi).to(self.device)

        for i in range(J):
            u = u + A[i]*torch.sin(Kx[i] * xx + Ky[i] * yy + phi[i])
        return u