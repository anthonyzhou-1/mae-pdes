import os
import wandb
from datetime import datetime
import argparse
import json
import yaml
import random

from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import numpy as np

from common.utils import DataCreator, Embedder, Normalizer
from helpers.model_helper import *
from tqdm import trange

def train(args: argparse,
          epoch: int,
          model: torch.nn.Module,
          optimizer: torch.optim,
          scheduler: torch.optim.lr_scheduler,
          loader: DataLoader,
          data_creator: DataCreator,
          augmentation,
          embedder: Embedder,
          normalizer: Normalizer,
          device: torch.cuda.device="cpu",
          optimizer_embedder = None) -> None:
    """
    Training loop.
    Loop is over the mini-batches and for every batch we pick a random timestep.
    This is done for the number of timesteps in our training sample, which covers a whole episode.
    Args:
        args (argparse): command line inputs
        epoch (int): current epoch
        model (torch.nn.Module): model to be pretrained
        optimizer (torch.optim): optimizer used for training
        scheduler (torch.optim.lr_scheduler): scheduler used for training
        loader (DataLoader): training dataloader
        data_creator (DataCreator): helper object to handle data
        augmentation : Augmentation object to compute Lie Symmetry Data Augmentation
        device (torch.cuda.device): device (cpu/gpu)
    Returns:
        None
    """
    print(f'Epoch: {epoch}')
    model.train()
    t_res = data_creator.t_res
    t = trange(t_res, desc=f'Loss: N/A', leave=True)

    for i in t:
        losses = []
        for u, _ in loader:
            batch_size = u.shape[0]
            optimizer.zero_grad()

            if optimizer_embedder is not None:
                optimizer_embedder.zero_grad()

            # Partition trajectory
            max_start_time = t_res - data_creator.time_history
            start_time = random.choices([t for t in range(max_start_time+1)], k=batch_size)
            u_window = data_creator.create_data(u, start_time) # b, time_window, nx

            # Augmentation
            u_window = u_window.to(device)
            u_window = augmentation(u_window)

            z = embedder(u_window, loader.dataset.x, loader.dataset.t, start_time) # b, emb_dim
            loss = model(u_window, embedding=z, normalizer=normalizer)

            # Backward pass
            loss.backward()

            # step optimizer
            optimizer.step()
            scheduler.step()

            if optimizer_embedder is not None:
                optimizer_embedder.step()
            losses.append(loss.detach() / args.batch_size)

        losses = torch.stack(losses)
        losses_out = torch.mean(losses)

        t.set_description(f"Loss: {losses_out:.6f}")
        t.refresh() 
        '''
        wandb.log({"train/loss": losses_out,})
        '''

def test(args: argparse,
          model: torch.nn.Module,
          loader: DataLoader,
          data_creator: DataCreator,
          augmentation,
          embedder: Embedder,
          normalizer: Normalizer,
          horizon: int,
          device: torch.cuda.device="cpu") -> None:
    """
    Testing loop. 
    Sums errors over an entire trajectory 
    Args:
        args (argparse): command line inputs
        model (torch.nn.Module): model to be pretrained
        loader (DataLoader): training dataloader
        data_creator (DataCreator): helper object to handle data
        horizon (int): how many time steps to forecast into the future
        device (torch.cuda.device): device (cpu/gpu)
    Returns:
        loss: float: loss value
    """
    model.eval()
    print("Validation...")
    errors = []
    with torch.no_grad():
        for u, _ in loader:
            batch_size = u.shape[0]
            steps = [t for t in range(0, horizon+1 - data_creator.time_history, data_creator.time_history)]
            u_t = torch.Tensor().to(args.device)
            u_rec_t = torch.Tensor().to(args.device)
            u_masked_t = torch.Tensor().to(args.device)

            for step in steps:
                same_steps = [step]*batch_size
                u_window = data_creator.create_data(u, same_steps)

                # Augmentation
                u_window = u_window.to(device)

                if len(args.sizes) == 1:
                    u_window = augmentation(u_window) # downsample if sizes is just one value

                z = embedder(u_window, loader.dataset.x, loader.dataset.t, same_steps)  # b, emb_dim
                u_orig, u_rec, mask = model(u_window, embedding=z, normalizer=normalizer, features=True)

                u_rec_total = u_orig * (1-mask) + u_rec * mask
                u_masked = u_orig * (1-mask)

                assert F.mse_loss(u_rec*mask, u_orig*mask) == F.mse_loss(u_rec_total, u_orig)

                u_t = torch.cat((u_t, u_orig), dim=1)
                u_masked_t = torch.cat((u_masked_t, u_masked), dim=1)
                u_rec_t = torch.cat((u_rec_t, u_rec_total), dim=1)
            
            error = F.mse_loss(u_t, u_rec_t).item()
            errors.append(error)
    avg_error = sum(errors)/len(errors)
    print("Average Error: ", avg_error)
    '''
    wandb.log({
            "valid/loss": avg_error,
        })
    '''
    return avg_error

def main(args: argparse):

    seed = args.seed if args.seed else 0
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    print(f"Running with seed {seed}")
    print(f"Running on device {args.device}") 
    print(f"Running with batch size {args.batch_size}")
    print(f"Running with masking ratio {args.masking_ratio}")
    print(f"Running with augmentation ratio {args.augmentation_ratio}")

    # Setup
    dateTimeObj = datetime.now()
    timestring = f'{dateTimeObj.date().month}{dateTimeObj.date().day}{dateTimeObj.time().hour}{dateTimeObj.time().minute}'
    name = f'{args.description}_{args.encoder}_{args.batch_size}_mask{args.masking_ratio}_aug{args.augmentation_ratio}_{timestring}'

    '''
    Uncomment if using wandb
    run = wandb.init(project="project-name",
                name = name,
                config=vars(args),
                mode=args.wandb_mode)
    '''

    # Data Loading
    train_loader = get_dataloader(args.pretrain_path, 
                                  args, 
                                  mode="train")
    valid_loader = get_dataloader(args.valid_path,
                                  args,
                                  mode="valid")
                                  
    data_creator = DataCreator(time_history=args.time_window,
                               t_resolution=args.base_resolution[0],
                               t_range=args.t_range,
                               x_resolution=args.base_resolution[1],
                               x_range=args.x_range,)
    
    augmentation = get_augmentation(args, train_loader.dataset.dx, train_loader.dataset.dt)
    embedder = get_embedder(args)
    normalizer = get_normalizer(args, train_loader)

    # Defining Model
    model, optimizer, scheduler = get_mae(args, args.device)

    if isinstance(embedder, Embedder2D) and args.embedding_mode=='spatial':
        optimizer_embedder = torch.optim.AdamW(embedder.parameters(), args.min_lr)
        print("Added embedder params to optimizer")
    
    else:
        optimizer_embedder = None

    ## Training
    num_epochs = args.num_epochs

    save_path= f'checkpoints/{name}.pth'
    min_val_loss = 10e10
    for epoch in range(num_epochs):
        train(args, 
              epoch, 
              model, 
              optimizer, 
              scheduler, 
              train_loader, 
              data_creator, 
              augmentation, 
              embedder, 
              normalizer, 
              device=args.device,
              optimizer_embedder=optimizer_embedder)
        
        val_loss = test(args, 
                        model, 
                        valid_loader, 
                        data_creator, 
                        augmentation,
                        embedder, 
                        normalizer, 
                        args.horizon, 
                        device=args.device)
        
        if val_loss < min_val_loss:
            checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'embedder_state_dict': embedder.state_dict() if optimizer_embedder is not None else None,
                    'optimizer_embedder_state_dict': optimizer_embedder.state_dict() if optimizer_embedder is not None else None,
                }
            torch.save(checkpoint, save_path)
            min_val_loss = val_loss
    '''
    wandb.log({
        "min_val_loss/loss": min_val_loss,
    })
    '''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pretrain an PDE solver')
    parser.add_argument('--config', type=str, help='Load settings from file in json format. Command line options override values in file.')

    ################################################################
    #  GENERAL
    ################################################################
    parser.add_argument('--description', type=str, help='Experiment for PDE solver should be trained')
    parser.add_argument('--seed', type=int, help='Seed for reproducibility')
    parser.add_argument('--batch_size', type=int, help='Batch Size')
    parser.add_argument('--num_epochs', type=int, help='Number of training epochs')
    parser.add_argument('--multiprocessing', type=eval, help='Flag for multiprocessing')
    parser.add_argument('--device', type=str, help="device")
    parser.add_argument('--wandb_mode', type=str, help='Wandb mode')
    parser.add_argument('--normalize', type=eval, help='Flag for normalization')
    parser.add_argument('--pretrained_path', type=str, help='Path to pretrained model')
    
    ################################################################
    #  DATA
    ################################################################
    parser.add_argument('--pretrain_path', type=str, help='Path to pretraining data')
    parser.add_argument('--train_path', type=str, help='Path to training data')
    parser.add_argument('--valid_path', type=str, help='Path to validation data')
    parser.add_argument('--time_window', type=int, help='Time window for forecasting')
    parser.add_argument('--augmentation_ratio', type=float, help='Flag for adding augmentations to pretraining')
    parser.add_argument('--encoder_embedding_dim', type=int, help='Dimension of the encoder embedding')
    parser.add_argument('--embedding_mode', type=str, help='Mode of the embedding')
    parser.add_argument('--max_shift', type=float, help='Max shift in x')
    parser.add_argument('--pos_mode', type=str, help='Positional encoding mode')
    parser.add_argument('--sizes', type=lambda s: [int(item) for item in s.split(',')], help="sizes for multiresolution")
    
    ################################################################
    #  MAE
    ################################################################
    # Encoder
    parser.add_argument("--encoder_dim", type=int, help="Dimension of the encoder")
    parser.add_argument("--encoder_depth", type=int, help="Depth of the encoder")
    parser.add_argument("--encoder_heads", type=int, help="Heads of the encoder")

    # Decoder
    parser.add_argument("--decoder_dim", type=int, help="Dimension of the decoder")
    parser.add_argument("--decoder_depth", type=int, help="Depth of the decoder")
    parser.add_argument("--decoder_heads", type=int, help="Heads of the decoder")

    # General
    parser.add_argument('--patch_size', type=lambda s: [int(item) for item in s.split(',')], help="patch size in height x width")
    parser.add_argument('--temporal_patch_size', type=int, help="temporal patch size")
    parser.add_argument('--masking_ratio', type=float, help='Mask ratio for masked patches')

    args = parser.parse_args()

    # Load args from config
    if args.config:
        filename, file_extension = os.path.splitext(args.config)
        # Load yaml
        if file_extension=='.yaml':
            t_args = argparse.Namespace()
            t_args.__dict__.update(yaml.load(open(args.config), Loader=yaml.FullLoader))
            args = parser.parse_args(namespace=t_args)
        elif file_extension=='.json':
            with open(args.config, 'rt') as f:
                t_args = argparse.Namespace()
                t_args.__dict__.update(json.load(f))
                args = parser.parse_args(namespace=t_args)
        else:
            raise ValueError("Config file must be a .yaml or .json file")
        
    main(args)