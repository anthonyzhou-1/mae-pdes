import torch
import os
import numpy as np
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from models.VIT.vit import ViT
from models.VIT.vit3d import ViT3D
from models.FNO.fno1d import FNO1d_bundled
from models.FNO.fno2d import FNO2d_bundled
from models.FNO.fno1d_cond import FNO1d_bundled_cond
from models.FNO.fno2d_cond import FNO2d_bundled_cond
from models.Unet.unet_2d import Unet2D
from models.Unet.unet_2d_cond import Unet2D_cond 
from models.Unet.unet_1d_cond import Unet1D_cond
from models.Unet.unet_1d import Unet1D
from models.Resnet.resnet_1d import Resnet_1D
from models.Resnet.resnet_2d import Resnet_2D
from models.Resnet.resnet_2d_cond import Resnet_2D_cond
from models.Resnet.resnet_1d_cond import Resnet_1D_cond
from models.VIT.mae import MAE
from models.VIT.mae_pad import MAE_Padded
from models.Contrastive.vicreg import VICReg

from common.utils import load_pretrained, EncoderWrapper, Embedder, Embedder2D, Normalizer, TimestepWrapper, SRWrapper
from common.datasets import PDEDataset, PDEDataset2D
from common.augmentation import KdVBurgers_augmentation, AugmentationWrapper1D, Identity_Aug
from common.augmentation2D import Combined2D_Augmentation, AugmentationWrapper2D

def get_normalizer(args, dataloader):
    if args.normalize == False:
        print("Not normalizing data.")
        return None

    norm_stat_path = args.norm_stat_path
    if os.path.exists(norm_stat_path):
        print("Using existing normalization statistics.")
        normalizer = Normalizer(norm_stat_path=norm_stat_path, exists=True, device=args.device)
        return normalizer
    else:
        print("Normalizer statistics does not exists. Creating normalization statistics.")
        data = dataloader.dataset.u
        normalizer = Normalizer(data=data, norm_stat_path=norm_stat_path, exists=False, device=args.device)
        return normalizer

def get_embedder(args):
    if args.pde_dim == 1:
        embedder = Embedder(args).to(args.device)
        print(f"Initialized Embedder with mode: {args.embedding_mode} and embedding_dim: {args.encoder_embedding_dim}")
    elif args.pde_dim == 2:
        embedder = Embedder2D(args).to(args.device)
        print(f"Initialized Embedder with embedding_dim: {args.encoder_embedding_dim}")
    return embedder

def get_augmentation(args, dx, dt):
    if args.pde == "kdv_burgers" or args.pde == 'kdv_burgers_resolution':
        augmentation = KdVBurgers_augmentation(max_x_shift=args.max_shift,
                                    max_velocity=args.max_velocity)
        augmentation_wrapper = AugmentationWrapper1D(augmentation=augmentation,
                                                     augmentation_ratio=args.augmentation_ratio,
                                                     dx=dx,
                                                     dt=dt,
                                                     shift='fourier',
                                                     batched=True,
                                                     sizes=args.sizes)
        print(f"Added Augmentation for KdV-Burgers PDE with augmentation_ratio: {args.augmentation_ratio}")
        if len(args.sizes) > 0:
            print(f"Training w/ sizes: {args.sizes}, pos_mode: {args.pos_mode}")
    elif args.pde == "heat_adv_burgers" or args.pde == "heat_adv_burgers_resolution":
        augmentation = Combined2D_Augmentation(max_shift=args.max_shift,
                                               max_scale=args.max_scale,
                                               max_shift_nodal=args.max_nodal_shift,)
        augmentation_wrapper = AugmentationWrapper2D(augmentation=augmentation,
                                                     augmentation_ratio=args.augmentation_ratio,
                                                     shift='fourier',
                                                     batched=True,
                                                     sizes=args.sizes)
        print(f"Added Augmentation for 2D Heat-Adv-Burgers PDE with augmentation_ratio: {args.augmentation_ratio}")
        if len(args.sizes) > 0:
            print(f"Training w/ sizes: {args.sizes}, pos_mode: {args.pos_mode}")
    else:
        print("Using Identity Augmentation")
        augmentation_wrapper = Identity_Aug()
    
    return augmentation_wrapper

def get_dataloader(path, args, mode='train', rank = 0, world_size = 0):
    device = f"cuda:{rank}" if args.multiprocessing else args.device
    if args.pde_dim == 1:
        dataset = PDEDataset(path, 
                            pde=args.pde, 
                            mode=mode, 
                            resolution=args.base_resolution,
                            load_all=args.load_all,
                            n_samples=args.num_samples if mode == 'train' else -1,
                            device=device,
                            seed=args.seed,
                            norm_vars=args.norm_vars
                            )
        print(f"Loaded 1D {args.pde} PDE Dataset onto device: {device if args.load_all else 'CPU'}")
    elif args.pde_dim == 2:
        dataset = PDEDataset2D(path, 
                            pde=args.pde, 
                            mode=mode,
                            resolution=args.base_resolution,
                            load_all=args.load_all,
                            n_samples=args.num_samples if mode == 'train' else -1,
                            device=device,
                            seed=args.seed,
                            norm_vars=args.norm_vars)
        print(f"Loaded 2D {args.pde} PDE Dataset onto device: {device if args.load_all else 'CPU'}")
    else:
        raise ValueError("PDE dimension should be 1 or 2")
        
    if args.multiprocessing:
        sampler = DistributedSampler(dataset, 
                                     num_replicas=world_size, 
                                     rank=rank, 
                                     shuffle=True if mode == 'train' else False)
        dataloader = DataLoader(dataset, 
                                batch_size=args.batch_size if mode == 'train' else args.val_batch_size, 
                                pin_memory=False, 
                                num_workers=0, 
                                shuffle=False, 
                                sampler=sampler)
    else:
        dataloader = DataLoader(dataset, 
                                batch_size=args.batch_size if mode == 'train' else args.val_batch_size, 
                                pin_memory=not args.load_all, # if load_all is true, all memory is in GPU so pin memory must be false 
                                num_workers=0 if args.load_all else 4, # if load_all is true, all memory is in GPU so num_workers must be 0 
                                shuffle=True if mode == 'train' else False)
    
    return dataloader

def get_backbone(args, device, encoder):
    backbone = None

    if encoder == 'VIT':
        backbone = ViT(image_size = tuple(args.image_size),
                  patch_size = tuple(args.patch_size),
                  dim = args.encoder_dim,
                  depth = args.encoder_depth,
                  heads = args.encoder_heads,
                  mlp_dim = args.encoder_mlp_dim,
                  channels = 1,
                  pool = args.encoder_pool,
                  dim_head = args.encoder_dim_head,
                  embedding_dim=args.encoder_embedding_dim,
                  pos_mode=args.pos_mode).to(device)
        print("Initialized VIT Encoder")
        
    elif encoder == 'VIT3D':
        backbone = ViT3D(image_size = tuple(args.image_size),
                image_patch_size=tuple(args.patch_size),
                frames=args.time_window,
                frame_patch_size=args.temporal_patch_size,
                dim = args.encoder_dim,
                depth = args.encoder_depth,
                heads = args.encoder_heads,
                mlp_dim = args.encoder_mlp_dim,
                channels = 1,
                pool = args.encoder_pool,
                dim_head = args.encoder_dim_head,
                embedding_dim=args.encoder_embedding_dim,
                pos_mode=args.pos_mode).to(device)
        print("Initialized VIT3D Encoder")
    elif encoder == "Linear":
        backbone = nn.Linear(args.n_vars, args.embedding_dim).to(device)
        print("Initialized Linear Encoder")
    else:
        print("No backbone initialized")
        args.embedding_dim = 0

    return backbone

def get_encoder(args, device):
    backbone = get_backbone(args, device, args.encoder)

    if backbone == None:
        return None
    
    if args.pretrained_path != "none":
        backbone = load_pretrained(args, backbone, device)
    
    if args.encoder == "Linear":
        encoder = backbone
    else:
        encoder = EncoderWrapper(args, backbone).to(device)
    model_parameters = filter(lambda p: p.requires_grad, encoder.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Number of encoder parameters: {params}')

    if args.freeze:
        for param in encoder.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen")

    return encoder

def get_regression_model(args, device):
    encoder = get_encoder(args, device)

    optimizer= torch.optim.AdamW(encoder.parameters(), lr=args.max_lr)
    steps_per_epoch = args.n_inner
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                        T_max=args.num_epochs * steps_per_epoch, 
                                                        eta_min=args.min_lr)
    
    return encoder, optimizer, scheduler
    

def get_forecaster(args, device, model, conditional=False):
    if model == "FNO1D":
        if not conditional:
            forecaster = FNO1d_bundled(time_window=args.time_window,
                                    modes=args.fno_modes,
                                    width=args.fno_width,
                                    num_layers=args.fno_num_layers,).to(device)
            print("Initialized FNO1D")
        else:
            forecaster = FNO1d_bundled_cond(time_window=args.time_window,
                                    modes=args.fno_modes,
                                    width=args.fno_width,
                                    cond_channels=args.embedding_dim,
                                    num_layers=args.fno_num_layers,).to(device)
            print("Initialized Conditional FNO1D")
    
    elif model == "Unet1D":
        if not conditional:
            forecaster = Unet1D(n_input_scalar_components=1,
                                n_input_vector_components=0,
                                n_output_scalar_components=1,
                                n_output_vector_components=0,
                                time_history=args.time_window,
                                time_future=args.time_window,
                                hidden_channels=args.unet_hidden_channels,
                                norm = True,
                                ch_mults= (1, 2, 4)).to(device)
            print("Initialized Unet1D")
        else:
            forecaster = Unet1D_cond(n_input_scalar_components=1,
                                n_input_vector_components=0,
                                n_output_scalar_components=1,
                                n_output_vector_components=0,
                                time_history=args.time_window,
                                time_future=args.time_window,
                                hidden_channels=args.unet_hidden_channels,
                                norm = True,
                                ch_mults= (1, 2, 4),
                                use_scale_shift_norm=True,
                                embedding_dim=args.embedding_dim).to(device)
            print("Initialized Conditional Unet1D")

    elif model == "FNO2D":
        if not conditional:
            forecaster = FNO2d_bundled(time_window=args.time_window,
                                    modes1=args.fno_modes,
                                    modes2=args.fno_modes,
                                    width=args.fno_width,
                                    num_layers=args.fno_num_layers).to(device)
            print("Initialized FNO2D")
        else:
            forecaster = FNO2d_bundled_cond(time_window=args.time_window,
                                    modes1=args.fno_modes,
                                    modes2=args.fno_modes,
                                    width=args.fno_width,
                                    cond_channels=args.embedding_dim,
                                    num_layers=args.fno_num_layers).to(device)
            print("Initialized Conditional FNO2D")

    elif model == "Unet2D":
        if not conditional:
            forecaster = Unet2D(n_input_scalar_components=1,
                                n_input_vector_components=0,
                                n_output_scalar_components=1,
                                n_output_vector_components=0,
                                time_history=args.time_window,
                                time_future=args.time_window,
                                hidden_channels=args.unet_hidden_channels,
                                norm = True,
                                ch_mults= (1, 2, 4)).to(device)
            print("Initialized Unet2D")
        else:
            forecaster = Unet2D_cond(n_input_scalar_components=1,
                                n_input_vector_components=0,
                                n_output_scalar_components=1,
                                n_output_vector_components=0,
                                time_history=args.time_window,
                                time_future=args.time_window,
                                hidden_channels=args.unet_hidden_channels,
                                norm = True,
                                ch_mults= (1, 2, 4),
                                use_scale_shift_norm=True,
                                embedding_dim=args.embedding_dim).to(device)
            print("Initialized Conditional Unet2D")

    elif model =="Resnet1D":
        if not conditional:
            forecaster= Resnet_1D(in_nc = args.time_window,
                                out_nc = args.time_window,
                                nf = args.resnet_features,
                                nb = args.resnet_blocks,).to(device)
            print("Initialized Resnet1D")
        else:
            forecaster = Resnet_1D_cond(in_nc = args.time_window,
                                out_nc = args.time_window,
                                nf = args.resnet_features,
                                nb = args.resnet_blocks,
                                emb_dim = args.embedding_dim).to(device)
            print("Initialized Conditional Resnet1D")

    elif model == "Resnet2D":
        if not conditional:
            forecaster = Resnet_2D(in_nc = args.time_window,
                                out_nc = args.time_window,
                                    nf = args.resnet_features,
                                    nb = args.resnet_blocks,).to(device)
            print("Initialized Resnet2D")
        else:
            forecaster = Resnet_2D_cond(in_nc = args.time_window,
                                out_nc = args.time_window,
                                nf = args.resnet_features,
                                nb = args.resnet_blocks,
                                emb_dim = args.embedding_dim).to(device)
            print("Initialized Conditional Resnet2D")

    else:
        raise ValueError("Model not found")

    return forecaster

def get_model(args, device):
    encoder = get_encoder(args, device)
    conditional = False if args.encoder == "none" else True
    forecaster = get_forecaster(args, device, args.model, conditional=conditional)

    model = TimestepWrapper(model=forecaster, encoder=encoder, device=device, add_vars=args.add_vars)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Number of model parameters: {params}')

    optimizer= torch.optim.AdamW(model.parameters(), lr=args.timestep_lr)
    steps_per_epoch = args.base_resolution[0]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                           T_max=args.num_epochs * steps_per_epoch, 
                                                           eta_min=args.timestep_lr_min)

    return model, optimizer, scheduler

def get_sr_model(args, device):
    encoder = get_encoder(args, device) # VIT or Linear or none
    conditional = False if args.encoder == "none" else True

    network = get_forecaster(args, device, args.network, conditional=conditional,) # Resnet1D, Resnet2D
    operator = get_forecaster(args, device, args.operator, conditional=conditional,) # FNO1D, FNO2D

    model = SRWrapper(args=args,
                      network = network,
                      operator = operator,
                      encoder=encoder,
                      device=device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Number of model parameters: {params}')

    optimizer= torch.optim.AdamW(model.parameters(), lr=args.timestep_lr)
    steps_per_epoch = args.n_inner
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                           T_max=args.num_epochs * steps_per_epoch, 
                                                           eta_min=args.timestep_lr_min)

    return model, optimizer, scheduler


def get_mae(args, device):
    backbone = get_backbone(args, device, args.encoder)
    
    if args.pad == True:
        model = MAE_Padded(encoder=backbone,
                    decoder_dim = args.decoder_dim,
                    masking_ratio = args.masking_ratio,
                    decoder_depth = args.decoder_depth,
                    decoder_heads = args.decoder_heads,
                    decoder_dim_head = args.decoder_dim_head,
                    pos_mode=args.pos_mode).to(device)
        print("Initialized MAE with padding")
    else:
        model = MAE(encoder=backbone,
                    decoder_dim = args.decoder_dim,
                    masking_ratio = args.masking_ratio,
                    decoder_depth = args.decoder_depth,
                    decoder_heads = args.decoder_heads,
                    decoder_dim_head = args.decoder_dim_head,
                    pos_mode=args.pos_mode).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.min_lr, betas=(args.beta1, args.beta2), fused=False if device == 'cpu' else True)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                        max_lr=args.max_lr, 
                                                        steps_per_epoch= args.base_resolution[0] * (args.num_samples // args.batch_size + 1), 
                                                        epochs=args.num_epochs, 
                                                        pct_start=args.pct_start, 
                                                        anneal_strategy='cos', 
                                                        div_factor = args.div_factor,
                                                        final_div_factor=args.final_div_factor)
    print("Initialized OneCycleLR Scheduler")

    if args.multiprocessing:
        model = DDP(model, device_ids=[device])

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Number of model parameters: {params}')

    if args.pretrained_path != "none":
        checkpoint = torch.load(args.pretrained_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.base_resolution[0] * (args.num_samples // args.batch_size + 1), eta_min=args.min_lr//100)

        print(f"Loaded pretrained model from {args.pretrained_path} at epoch : {checkpoint['epoch']}")
        print("Loaded cosine annealing scheduler")

    return model, optimizer, scheduler

def get_ssl(args, device):
    backbone = get_backbone(args, device, args.encoder)

    model = VICReg(args, backbone).to(device)   
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.min_lr, betas=(args.beta1, args.beta2), fused=False if device == 'cpu' else True)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                        max_lr=args.max_lr, 
                                                        steps_per_epoch= args.base_resolution[0] * (args.num_samples // args.batch_size + 1), 
                                                        epochs=args.num_epochs, 
                                                        pct_start=args.pct_start, 
                                                        anneal_strategy='cos', 
                                                        div_factor = args.div_factor,
                                                        final_div_factor=args.final_div_factor)
    print("Initialized OneCycleLR Scheduler")

    if args.multiprocessing:
        model = DDP(model, device_ids=[device])

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Number of model parameters: {params}')

    if args.pretrained_path != "none":
        checkpoint = torch.load(args.pretrained_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.base_resolution[0] * (args.num_samples // args.batch_size + 1), eta_min=args.min_lr//100)

        print(f"Loaded pretrained model from {args.pretrained_path} at epoch : {checkpoint['epoch']}")
        print("Loaded cosine annealing scheduler")

    return model, optimizer, scheduler