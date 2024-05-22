import math
import torch 
from torch import nn 
from typing import Tuple
from collections import OrderedDict
import argparse
import json
import os
import yaml
import pickle 
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch.nn.functional as F

class DataCreator(nn.Module):
    """
    Helper class to construct input data and labels.
    """
    def __init__(self,
                 time_history,
                 t_resolution,
                 t_range, 
                 x_resolution,
                 x_range,
                 ):
        """
        Initialize the DataCreator object.
        Args:
            time_history (int): how many time steps are used for PDE prediction
            time_future (int): how many time does the solver predict into the future
            t_resolution: temporal resolution
            x_resolution: spatial resolution
        """
        super().__init__()
        self.time_history = time_history
        self.t_res = t_resolution
        self.t_range = t_range
        self.x_res = x_resolution
        self.x_range = x_range

    def create_data(self, datapoints: torch.Tensor, start_time: list) -> torch.Tensor:
        """
        Getting data of PDEs for training, validation and testing.
        Args:
            datapoints (torch.Tensor): trajectory input
            start_time (int list): list of different starting times for different trajectories in one batch
        Returns:
            torch.Tensor: neural network input data
        """
        data = []
        # Loop over batch and different starting points
        # For every starting point, we take the number of time_history points as training data
        # and the number of time future data as labels
        for (dp, start) in zip(datapoints, start_time):
            end_time = start+self.time_history
            d = dp[start:end_time]
            data.append(d.unsqueeze(dim=0))

        return torch.cat(data, dim=0)
    
    def create_data_labels(self, datapoints: torch.Tensor, steps: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Getting data for PDE training at different time points
        Args:
            datapoints (torch.Tensor): trajectory
            steps (list): list of different starting points for each batch entry
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: input data and label
        """
        shape = list(datapoints.shape)
        shape[0] = 0
        shape[1] = self.time_history

        data = torch.empty(shape, device=datapoints.device)
        labels = torch.empty(shape, device=datapoints.device)
        for (dp, step) in zip(datapoints, steps):
            d = dp[step - self.time_history:step]
            l = dp[step:self.time_history + step]
            data = torch.cat((data, d[None, :]), 0)
            labels = torch.cat((labels, l[None, :]), 0)

        return data, labels

def process_dict(state_dict: OrderedDict, prefix: str) -> OrderedDict:
    '''
    Processes state dict to remove prefixes
    '''

    return {k.partition(f'{prefix}.')[2]:state_dict[k] for k in state_dict.keys()}

def dict2tensor(d: dict) -> torch.Tensor:
    """
    Converts a dictionary to a tensor
    Args:
        d (dict): dictionary
    Returns:
        t (torch.Tensor): tensor
    """
    tensors = []
    for k, v in d.items():
        tensors.append(v.unsqueeze(0))
    return torch.flatten(torch.transpose(torch.cat(tensors, dim=0), 0, 1), start_dim=1)

def get_args(config):
    parser = argparse.ArgumentParser(description='Train a PDE Solver')
    parser.add_argument('--config', type=str, help='Path to config file')
    args = parser.parse_args("")
    args.config=config

    # Load args from config
    if args.config:
        filename, file_extension = os.path.splitext(args.config)
        if file_extension=='.yaml':
            t_args = argparse.Namespace()
            t_args.__dict__.update(yaml.load(open(args.config), Loader=yaml.FullLoader))
            args = parser.parse_known_args(namespace=t_args)
        elif file_extension=='.json':
            with open(args.config, 'rt') as f:
                t_args = argparse.Namespace()
                t_args.__dict__.update(json.load(f))
                args = parser.parse_known_args(namespace=t_args)
        else:
            raise ValueError("Config file must be a .yaml or .json file")

    args = args[0]
    return args

def load_pretrained(args, backbone, device):
    pretrained_dict = torch.load(args.pretrained_path, map_location=f'cuda:{device}' if args.multiprocessing else device)
    if "model_state_dict" in pretrained_dict:
        pretrained_dict = pretrained_dict["model_state_dict"]
    pretrained_dict = process_dict(pretrained_dict, "encoder")

    backbone_dict = backbone.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in backbone_dict}
    # 2. overwrite entries in the existing state dict
    backbone_dict.update(pretrained_dict) 
    # 3. load the new state dict
    backbone.load_state_dict(pretrained_dict)
    print(f'Loaded pretrained encoder from: {args.pretrained_path}')

    return backbone

class Embedder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mode = args.embedding_mode
        self.time_window = args.time_window
        self.encoder_embedding_dim = args.encoder_embedding_dim
        self.nx = args.base_resolution[1]

        if self.mode == "spatial":
            assert self.encoder_embedding_dim == self.nx, "Spatial embedding dim must match spatial resolution"
        elif self.mode == "temporal":
            assert self.encoder_embedding_dim == self.time_window, "Temporal embedding dim must match time window"
        elif self.mode == "spatiotemporal":
            assert self.encoder_embedding_dim == self.nx + self.time_window, "Spatiotemporal embedding dim must match spatial resolution + time window"
        else:
            assert self.encoder_embedding_dim == 0, "Invalid mode"

    def normalize(self, A, dim=0):
        '''
        Normalizes the input tensor
        Expected input shape: [batch, d]
        '''

        A -= A.min(dim, keepdim=True)[0]
        A /= A.max(dim, keepdim=True)[0]
        return A

    def partition_time(self, t, steps):
        """
        Partition time grid
        Args:
            t (torch.Tensor): time grid in shape [nt]
            steps: list of time steps for partitioning t
        Returns:
            partitions (torch.Tensor): partitioned time grid in shape [batch, time_window]
        """
        partitions = []
        for i, step in enumerate(steps):
            t_step = t[step:step + self.time_window]
            partitions.append(t_step.unsqueeze(0))

        return torch.cat(partitions, dim=0)
    
    def forward(self, u, x, t, steps):
        """
        Get embedding for the input data
        x and t are assumed constant across batch
        Args:
            u (torch.Tensor): input data in shape [batch, time_window, nx]
            x (torch.Tensor): spatial grid in shape [nx]
            t (torch.Tensor): time grid in shape [nt]
            steps: list of time steps for partitioning t
        Returns:
            torch.Tensor: embedding
        """
        if self.encoder_embedding_dim == 0:
            return None
        
        x = self.normalize(x)
        t = self.normalize(t)
        
        if u.shape[-1] != self.nx:
            nx_new = u.shape[-1]
            x_new = torch.linspace(x[0], x[-1], nx_new, device=x.device)
            x_pad = -1*torch.ones(self.nx - nx_new, device=x.device)
            x = torch.cat((x_new, x_pad), dim=0)

        batch_size = u.shape[0]
        if self.mode == 'spatial':
            x_batched = x.repeat(batch_size, 1) # [batch, nx]
            embedding = x_batched # [batch, nx]

        elif self.mode == 'temporal':
            t_batched = self.partition_time(t, steps) # [batch, time_window]
            embedding = t_batched # [batch, time_window]

        elif self.mode == "spatiotemporal":
            x_batched = x.repeat(batch_size, 1)
            t_batched = self.partition_time(t, steps) 
            embedding = torch.cat((x_batched, t_batched), dim=1) # [batch, nx + time_window]

        else:
            raise ValueError("Invalid mode")

        return embedding
    
class Embedder2D(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mode = args.embedding_mode
        self.time_window = args.time_window
        self.encoder_embedding_dim = args.encoder_embedding_dim
        self.device=args.device

        if args.embedding_mode == 'spatial':
            self.conv1 = nn.Conv2d(in_channels=2,
                                out_channels=8,
                                kernel_size=3,
                                stride=1,
                                padding='same')
            self.activation = nn.GELU()
            self.conv2 = nn.Conv2d(in_channels=8,
                                out_channels=16,
                                kernel_size=3,
                                stride=1,
                                padding='same')
            self.avg_pool = nn.AvgPool2d(kernel_size=2)
            self.conv3 = nn.Conv2d(in_channels=16,
                                out_channels=16,
                                kernel_size=3,
                                stride=1,
                                padding='same')
            self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(6)
            
            linear_dim = 6*6*16
            self.linear1 = nn.Linear(linear_dim, self.encoder_embedding_dim)
            self.linear2 = nn.Linear(self.encoder_embedding_dim, self.encoder_embedding_dim)
        
        else:
            assert args.encoder_embedding_dim == 0
        
    def forward(self, u, x, t, steps):
        """
        Get embedding for the input data
        x and t are assumed constant across batch
        Args:
            u (torch.Tensor): input data in shape [batch, time_window, nx, ny]
            x (torch.Tensor): spatial grid in shape [2, nx, ny]
            t (torch.Tensor): time grid in shape [nt]
            steps: list of time steps for partitioning t
        Returns:
            torch.Tensor: embedding
        """
        if self.encoder_embedding_dim == 0:
            return None
        else:
            x_size = u.shape[-1] # assume uniform
            batch_size = u.shape[0]

            x = x.unsqueeze(0).repeat(batch_size, 1, 1, 1)

            x = F.interpolate(x, size=x_size, mode="bilinear") # scale x coords
            x = x.to(self.device)
            z = self.activation(self.conv1(x))
            z = self.activation(self.conv2(z))
            z = self.avg_pool(z)
            z = self.activation(self.conv3(z))
            z = self.adaptive_avg_pool(z)
            z = z.view(z.size(0), -1)
            embedding = self.activation(self.linear1(z))
            embedding = self.linear2(embedding)

        return embedding

class EncoderWrapper(nn.Module):
    def __init__(
        self,
        args,
        backbone,
    ):
        super().__init__()

        self.backbone = backbone
        self.args = args
        self.linear1 = nn.Linear(args.encoder_dim, args.encoder_dim // 2)
        self.linear2 = nn.Linear(args.encoder_dim // 2, args.embedding_dim)
        self.activation = nn.GELU()
        self.norm1 = nn.LayerNorm(args.encoder_dim)
        self.norm2 = nn.LayerNorm(args.encoder_dim // 2)

    def forward(self, x, embedding = None, normalizer = None):
        x = self.backbone(x, embedding = embedding, normalizer = normalizer)
        x = self.norm1(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.norm2(x)
        x = self.linear2(x)
        return x
    
class Normalizer:
    def __init__(self,
                 data=None,
                 norm_stat_path: str = None,
                 exists: bool = False,
                 device: str = 'cuda:0',):
        if exists:
            with open(norm_stat_path, "rb") as f:
                self.u_mean, self.u_std = pickle.load(f)
        else:
            # expects data in [n_samples, nt, nx]
            assert data is not None, "Data must be provided for normalization"
            
            u_scaler = StandardScaler()

            for i, _ in enumerate(data):
                u = data[i]
                u = u.flatten().cpu().numpy() # [nt, nx] -> [nt*nx]
                u_scaler.partial_fit(u.reshape(-1, 1))

            # retrieve mean and std
            # make sure they are item
            self.u_mean = u_scaler.mean_.item()
            self.u_std = np.sqrt(u_scaler.var_).item()
            self.save_norm_stats(path=norm_stat_path)
            print("Normalization statistics saved to", norm_stat_path)

        # print statistics
        print(f"u mean: {self.u_mean}, u std: {self.u_std}")
        self.u_mean = torch.tensor(self.u_mean, device=device)
        self.u_std = torch.tensor(self.u_std, device=device)

    def save_norm_stats(self, path):
        with open(path, "wb") as f:
            pickle.dump([self.u_mean, self.u_std], f)

    def normalize(self, u):
        u = (u - self.u_mean) / self.u_std
        return u

    def denormalize(self, u): 
        u = u * self.u_std + self.u_mean
        return u
    
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms
        
    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

class TimestepWrapper(nn.Module):
    def __init__(self, model, encoder=None, device='cuda:0', add_vars = False):
        super().__init__()
        self.model = model 
        self.encoder = encoder
        self.add_vars = add_vars 
        self.device = device

        if add_vars:
            print("Adding variables to input data")

    def forward(self, data, variables=None, z=None):
        data = data.to(self.device)
        if self.encoder is not None:

            if self.add_vars:
                variables = dict2tensor(variables).to(self.device)
                embeddings = self.encoder(variables)
            else:
                embeddings = self.encoder(data, z)

            pred = self.model(data, embeddings)
        else:
            pred = self.model(data)
        
        return pred


class SR_Augmentation1D:
    def __init__(self, size_low) -> None:
        self.size_low = size_low
    
    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        """
        Subsampling augmentation.
        Args:
            u (torch.Tensor): input tensor
        Returns:
            torch.Tensor: subsampled trajectory
        """
        size_low = self.size_low
        u_low = F.interpolate(u, size=size_low, mode='linear')
        return u_low, u 

class SRWrapper(nn.Module):
    def __init__(self, args, network, operator, encoder, device='cuda:0'):
        super().__init__()
        self.network = network
        self.operator = operator
        self.encoder = encoder 
        self.size_low = args.size_low
        self.scale_factor = args.scale_factor
        self.pde_dim = args.pde_dim
        self.device = device
        self.add_vars = args.add_vars
        self.criterion = nn.MSELoss()

    def discretization_inversion(self, data):
        if len(data.shape) == 3:
            # 1D SR
            data_upsample = F.interpolate(data, scale_factor=self.scale_factor, mode='linear', align_corners=True)
        elif len(data.shape) == 4:
            # 2D SR
            data_upsample = F.interpolate(data, scale_factor=self.scale_factor, mode='bicubic', align_corners=True)
        else:
            raise ValueError("Invalid data shape")

        return data_upsample
    
    def downsample(self, data):
        if self.pde_dim == 1:
            assert len(data.shape) == 3, "Invalid data shape"
            data_lowres = F.interpolate(data, size=self.size_low, mode='linear')
        elif self.pde_dim == 2:
            assert len(data.shape) == 4, "Invalid data shape"
            data_lowres = F.interpolate(data, size=(self.size_low, self.size_low), mode='bicubic')
        else:
            raise ValueError("Invalid PDE dimension")

        return data_lowres, data

    def forward(self, data, variables=None):
        data_lowres, data = self.downsample(data)
        data_lowres = data_lowres.to(self.device)

        if self.encoder is not None:
            if self.add_vars:
                variables = dict2tensor(variables).to(self.device)
                emb = self.encoder(variables)
            else:
                emb = self.encoder(data_lowres) # b, embd_dim
            z_lowres = self.network(data_lowres, emb)
        else:
            z_lowres = self.network(data_lowres)

        z_upsample = self.discretization_inversion(z_lowres)

        if self.encoder is not None:
            data_out = self.operator(z_upsample, emb)
        else:
            data_out = self.operator(z_upsample)
        
        loss = self.criterion(data_out, data.to(self.device))

        return loss 

