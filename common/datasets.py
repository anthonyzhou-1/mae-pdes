import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from typing import Tuple
import random

class PDEDataset(Dataset):
    """Load samples of an PDE Dataset, get items according to PDE"""

    def __init__(self,
                 path: str,
                 pde: str,
                 mode: str,
                 resolution: list=None,
                 load_all: bool=False,
                 n_samples: int=-1,
                 device: str = 'cpu',
                 seed:int = 0,
                 norm_vars = False) -> None:
        """Initialize the dataset object
        Args:
            path: path to dataset
            pde: string of PDE 
            mode: [train, valid, test]
            resolution: resolution of the dataset [nt, nx]
            load_all: load all the data into memory
            n_samples: number of samples to load
            device: device to host data
        Returns:
            None
        """
        super().__init__()
        f = h5py.File(path, 'r')
        self.mode = mode
        self.pde = pde
        data = f[self.mode]
        self.n_samples = n_samples if n_samples > 0 else len(data['u'])
        self.resolution = (250, 100) if resolution is None else resolution

        self.variables, self.var_range = self.get_variables(self.pde)
        self.attrs = ['u'] + self.variables
        for attr in self.attrs + ['x', 't']:
            setattr(self, attr, 
                    torch.as_tensor(np.array(data[attr]), 
                                    dtype=torch.float32, 
                                    device=device if load_all else 'cpu')
                    )
        f.close()

        random.seed(seed)
        self.indexes = random.sample(torch.arange(len(self.u)).tolist(), self.n_samples)
        self.device = device

        self.dt = self.t[1] - self.t[0]
        self.dx = self.x[1] - self.x[0]

        self.norm_vars = norm_vars

        print("Data loaded from: {}".format(path))
        print("n_samples: {}".format(self.n_samples))
        print("Resolution: {}".format(self.u.shape))
        print("Loaded data onto device: {}".format(self.u.device))
        print("\n")

    def __len__(self):
        return self.n_samples
    
    def get_variables(self, pde):
        if pde == "kdv_burgers" or pde == "kdv_burgers_resolution":
            variables = ['alpha', 'beta', 'gamma']
            ranges = {'alpha': [0, 6], 
                    'beta': [0.1, 0.4],
                    'gamma': [0, 1]}
        
        elif pde == "heat":
            variables = ['beta']
            ranges = {'beta': [0.1, 0.8]}

        elif pde == 'burgers':
            variables = [] # No variables for burgers eqn (inviscid)
            ranges = None

        elif pde == "ks":
            variables = ['v']
            ranges = {'v': [0.75, 1.25]}

        elif pde == "advection":
            variables = ['a']
            ranges = {'a': [0.1, 5.0]} 

        elif pde == 'wave_dir' or pde == 'wave_neu':
            variables = ["c", "bc_right", "bc_left"]
            ranges = None

        elif pde == "heat_dir" or pde == "heat_neu":
            variables = ["nu"]
            ranges = None

        elif pde == "wave_combined": # Classification problems
            variables = ["BC"] # 2 way (dir, neu)
            ranges = None # classification problem

        elif pde == "heat_combined":
            variables = ["BC"] # 3 way (per, dir, neu)
            ranges = None # classification problem

        elif pde == "combined":
            variables = ["pde"] # 4 way (heat, advection, burgers, ks)
            ranges = None # classification problem
        else:
            raise ValueError("PDE not found")

        return variables, ranges  

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, list]:
        """
        Get data item
        Args:
            i (int): data index
        Returns:
            u: torch.Tensor: numerical baseline trajectory of size [nt, nx]
            variables: dict: equation specific parameters
        """
        idx = self.indexes[i]
        variables = {attr: getattr(self, attr)[idx] for attr in self.attrs}
        u = variables.pop('u')

        if self.norm_vars and self.var_range is not None:
            for var in self.variables:
                variables[var] = (variables[var] - self.var_range[var][0]) / (self.var_range[var][1] - self.var_range[var][0])

        return u, variables

        
class PDEDataset2D(Dataset):
    """Load samples of a 2D PDE Dataset, get items according to PDE"""

    def __init__(self,
                 path: str,
                 pde: str,
                 mode: str,
                 resolution: list=None,
                 load_all: bool=False,
                 n_samples: int=-1,
                 device: str='cuda:0',
                 seed:int = 0,
                 norm_vars = False) -> None:
        """Initialize the dataset object
        Args:
            path: path to dataset
            pde: string of PDE 
            mode: [train, valid, test]
            resolution: base resolution of the dataset [nt, nx, ny]
            load_all: load all the data into memory
            device: if load_all, load data onto device
        Returns:
            None
        """
        super().__init__()
        f = h5py.File(path, 'r')
        self.mode = mode
        self.pde = pde
        self.resolution = (100, 64, 64) if resolution is None else resolution
        data = f[self.mode]
        self.n_samples = n_samples if n_samples > 0 else len(data['u'])

        self.variables, self.var_range = self.get_variables(self.pde)
        self.attrs = ['u'] + self.variables
        for attr in self.attrs + ['x', 't']:
            setattr(self, attr, 
                    torch.as_tensor(np.array(data[attr]), 
                                    dtype=torch.float32, 
                                    device=device if load_all else 'cpu')
                    )
        f.close()

        random.seed(seed)
        self.indexes = random.sample(torch.arange(len(self.u)).tolist(), self.n_samples)
        self.device = device

        self.dt = self.t[1] - self.t[0]
        self.dx = self.x[0, 0, 1] - self.x[0, 0, 0]
        self.dy = self.x[1, 1, 0] - self.x[1, 0, 0]

        self.norm_vars = norm_vars

        print("Data loaded from: {}".format(path))
        print("n_samples: {}".format(self.n_samples))
        print("Resolution: {}".format(self.u.shape))
        print("Loaded data onto device: {}".format(self.u.device))
        print("\n")

    def __len__(self):
        return self.n_samples
    
    def get_variables(self, pde):
        if pde == "heat_adv_burgers" or pde == "heat_adv_burgers_resolution":
            variables = ['nu', 'ax', 'ay', 'cx', 'cy']
            ranges = {'nu': [2e-3, 2e-2],
                    'ax': [0.1, 2.5],
                    'ay': [0.1, 2.5],
                    'cx': [0.5, 1.0],
                    'cy': [0.5, 1.0]}
        elif pde == "heat":
            variables = ['nu']
            ranges = {'nu': [2e-3, 2e-2]}
        elif pde == "burgers":
            variables = ['nu', 'cx', 'cy']
            ranges = {'nu': [7.5e-3, 1.5e-2],
                    'cx': [0.5, 1.0],
                    'cy': [0.5, 1.0]}
        elif pde == "advection":
            variables = ['ax', 'ay']
            ranges = {'ax': [0.1, 2.5],
                    'ay': [0.1, 2.5]}
        elif pde == "ns":
            variables = ['visc', 'amp']
            ranges = {'visc': [1e-9, 1e-5],
                    'amp': [0.001, 0.01]}
        else:
            raise ValueError("PDE not found")
        
        return variables, ranges
    
    def __getitem__(self, i: int) -> Tuple[torch.Tensor, list]:
        """
        Get data item
        Args:
            i (int): data index
        Returns:
            u: torch.Tensor: numerical baseline trajectory of size [nt, nx, ny]
            variables: dict: equation specific parameters
        """
        
        idx = self.indexes[i]
        variables = {attr: getattr(self, attr)[idx] for attr in self.attrs}
        u = variables.pop('u')

        if self.norm_vars and self.var_range is not None:
            for var in self.variables:
                variables[var] = (variables[var] - self.var_range[var][0]) / (self.var_range[var][1] - self.var_range[var][0])

        return u, variables