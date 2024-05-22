import wandb
from datetime import datetime
import argparse
import json
import os 
import yaml
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import torch
import random

from common.utils import DataCreator, dict2tensor
from common.augmentation import *
from helpers.model_helper import *

def train(args: argparse,
          epoch: int,
          encoder: torch.nn.Module,
          optimizer: torch.optim,
          scheduler: torch.optim.lr_scheduler,
          loader: DataLoader,
          data_creator: DataCreator,
          augmentation,
          embedder: Embedder,
          normalizer: Normalizer,
          criterion: torch.nn.Module,
          device: torch.cuda.device="cpu",
          optimizer_embedder=None)-> None:
    """
    Training loop.
    Loop is over the mini-batches and for every batch we pick a random timestep.
    This is done for the number of timesteps in our training sample, which covers a whole episode.
    Args:
        args (argparse): command line inputs
        encoder (torch.nn.Module): Encoder to generate latent embedding
        optimizer (torch.optim): optimizer used for training
        scheduler (torch.optim.lr_scheduler): scheduler used for training
        loader (DataLoader): training dataloader
        data_creator (DataCreator): helper object to handle data
        augmentation : Augmentation object to compute Lie Symmetry Data Augmentation
        device (torch.cuda.device): device (cpu/gpu)
    Returns:
        None
    """
    print(f'Starting epoch {epoch}...')
    encoder.train()

    for i in range(args.n_inner):
        losses = []
        for u, variables in loader:
            optimizer.zero_grad()

            if optimizer_embedder is not None:
                optimizer_embedder.zero_grad()

            start_time = [0]*args.batch_size # Predict the coefficients from the first sample
            
            u_window = data_creator.create_data(u, start_time) # b, nt, nx

            # Augmentation
            u_window = u_window.to(device)

            if args.pde == "kdv_burgers_resolution" or args.pde == "heat_adv_burgers_resolution":
                u_window, labels = augmentation(u_window, label=True)

            # Embedding
            z = embedder(u_window, loader.dataset.x, loader.dataset.t, start_time) # b, emb_dim

            # Compute loss
            targets = dict2tensor(variables) # b, n_vars
            preds = encoder(u_window, embedding=z, normalizer=normalizer) # b, n_vars

            if args.mode == "classification":
                targets = targets.long().squeeze()

            if args.pde == "kdv_burgers_resolution" or args.pde == "heat_adv_burgers_resolution":
                loss = criterion(preds, labels.to(device))
            else:
                loss = criterion(preds, targets.to(device))
            loss.backward()
            losses.append(loss.detach() / args.batch_size)
            optimizer.step()

            if optimizer_embedder is not None:
                optimizer_embedder.step()

            scheduler.step()
        
        losses = torch.stack(losses)
        losses_out = torch.mean(losses)
        print(f'Training Loss (progress: {i / args.n_inner:.2f}): {losses_out}')
        '''
        Uncomment if using wandb
        wandb.log({"train/loss": losses_out})
        '''


def test(args: argparse,
          encoder: torch.nn.Module,
          loader: DataLoader,
          data_creator: DataCreator,
          augmentation,
          embedder: Embedder, 
          normalizer: Normalizer, 
          criterion: torch.nn.Module,
          device: torch.cuda.device="cpu") -> None:
    """
    Testing loop
    Args:
        args (argparse): command line inputs
        encoder (torch.nn.Module): Encoder to generate latent embedding
        loader (DataLoader): training dataloader
        data_creator (DataCreator): helper object to handle data
        device (torch.cuda.device): device (cpu/gpu)
    Returns:
        None
    """
    encoder.eval()
    losses = []
    with torch.no_grad():
        for u, variables in loader:

            start_time = [0]*args.batch_size # Predict the coefficients from the first sample
    
            u_window = data_creator.create_data(u, start_time) # b, nt, nxu_window = data_creator.create_data(u, start_time) # b, nt, nx
            u_window = u_window.to(device)

            if args.pde == "kdv_burgers_resolution" or args.pde == "heat_adv_burgers_resolution":
                u_window, labels = augmentation(u_window, label=True)

            # Embedding
            z = embedder(u_window, loader.dataset.x, loader.dataset.t, start_time) # b, emb_dim

            # Compute loss
            targets = dict2tensor(variables) # b, n_vars
            preds = encoder(u_window, embedding=z, normalizer=normalizer) # b, n_vars

            if args.mode == "classification":
                targets = targets.long().squeeze()

            if args.pde == "kdv_burgers_resolution" or args.pde == "heat_adv_burgers_resolution":
                loss = criterion(preds, labels.to(device))
            else:
                loss = criterion(preds, targets.to(device))
            losses.append(loss.detach() / args.batch_size)
        
        losses = torch.stack(losses)
        losses_out = torch.mean(losses)
    
    return losses_out

def main(args: argparse):
    if isinstance(args.seed, str):
        args.seed = int(args.seed)

    if isinstance(args.freeze, str):
        args.freeze = eval(args.freeze)

    if isinstance(args.load_all, str):
        args.load_all = eval(args.load_all)

    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    print(f"Running with seed {seed}")
    print(f"Running on device {args.device}") 
    print(f"Running with batch size {args.batch_size}")
    print(f"Regressing coeffients: {args.embedding_dim}")

    # Setup
    dateTimeObj = datetime.now()
    timestring = f'{dateTimeObj.date().month}{dateTimeObj.date().day}{dateTimeObj.time().hour}{dateTimeObj.time().minute}'
    pretrained = True if args.pretrained_path != "none" else False
    name = f'{args.seed}{args.description}_{args.pde}_{args.encoder}_PT_{pretrained}_freeze_{args.freeze}_{timestring}'

    '''
    Uncomment if using wandb
    run = wandb.init(project=project name,
                    name = name,
                    config=vars(args),
                    mode=args.wandb_mode)
    '''
    device = args.device

    # Data Loading
    train_loader = get_dataloader(args.train_path, 
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

    # Encoder
    encoder, optimizer, scheduler = get_regression_model(args, device)
    criterion = torch.nn.MSELoss() if args.mode == "regression" else torch.nn.CrossEntropyLoss()

    augmentation = get_augmentation(args, train_loader.dataset.dx, train_loader.dataset.dt)
    embedder = get_embedder(args)

    # workaround for loading multiresolution spatial embeddings, TODO: make look better
    if args.encoder == "VIT3D" and len(args.sizes) > 0: # using vit encoder and many sizes
        if args.pretrained_path != "none": # using a pretrained embedder
            if args.embedding_mode=="spatial" and args.pde_dim == 2: # ensuring embedder is correct
                checkpoint = torch.load(args.pretrained_path, map_location=device)
                embedder.load_state_dict(checkpoint['embedder_state_dict'])
                print("Embedder loaded")
            if args.freeze == True: # freezing embedder
                for param in embedder.parameters():
                    param.requires_grad = False
                print("Embedder frozen")

        optimizer_embedder = torch.optim.AdamW(embedder.parameters(), args.min_lr)
        print("Added embedder params to optimizer")
    else:
        optimizer_embedder = None

    normalizer = get_normalizer(args, train_loader)

    ## Training
    num_epochs = args.num_epochs
    save_path= f'checkpoints/{name}.pth'
    min_val_loss = 10e10

    for epoch in range(num_epochs):
        train(args, 
            epoch, 
            encoder, 
            optimizer, 
            scheduler, 
            train_loader, 
            data_creator, 
            augmentation, 
            embedder, 
            normalizer, 
            criterion,
            device=args.device,
            optimizer_embedder=optimizer_embedder)
        
        print("Evaluation on validation dataset:")

        val_loss = test(args, 
                        encoder, 
                        valid_loader, 
                        data_creator, 
                        augmentation, 
                        embedder, 
                        normalizer, 
                        criterion,
                        device=args.device)    
         
        print(f"Validation Loss: {val_loss}\n")
        
        '''
        Uncomment if using wandb
        wandb.log({
            "valid/loss": val_loss,
        })
        '''

        if(val_loss < min_val_loss):
            # Save model
            if optimizer_embedder is not None:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': encoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'embedder_state_dict': embedder.state_dict(),
                    'optimizer_embedder_state_dict': optimizer_embedder.state_dict(),
                    'loss': val_loss,
                }
            else:
                checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': encoder.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': val_loss,
                    }
            torch.save(checkpoint, save_path)
            min_val_loss = val_loss
    '''
    Uncomment if using wandb
    wandb.log({
        "min_val_loss/loss": min_val_loss,
    })
    '''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Regress PDE coefficients')
    parser.add_argument('--config', type=str, help='Load settings from file in json format. Command line options override values in file.')

    ################################################################
    #  GENERAL
    ################################################################
    parser.add_argument('--description', type=str, help='Experiment for PDE solver should be trained')
    parser.add_argument('--seed', type=str, help='Seed for reproducibility')
    parser.add_argument('--batch_size', type=int, help='Batch Size')
    parser.add_argument('--val_batch_size', type=int, help='Validation Batch Size')
    parser.add_argument('--num_epochs', type=int, help='Number of training epochs')
    parser.add_argument('--multiprocessing', type=eval, help='Flag for multiprocessing')
    parser.add_argument('--device', type=str, help="device")
    parser.add_argument('--wandb_mode', type=str, help='Wandb mode')
    parser.add_argument('--normalize', type=eval, help='Flag for normalization')
    parser.add_argument('--n_inner', type=int, help='Number of inner loop iterations')
    parser.add_argument('--pretrained_path', type=str, help='Description of the run')

    ################################################################
    #  DATA
    ################################################################
    parser.add_argument('--train_path', type=str, help='Path to training data')
    parser.add_argument('--valid_path', type=str, help='Path to validation data')
    parser.add_argument('--load_all', type=eval, help='Load all data into memory')
    parser.add_argument('--time_window', type=int, help='Time window for forecasting')
    parser.add_argument('--augmentation_ratio', type=float, help='Flag for adding augmentations to pretraining')
    parser.add_argument('--encoder_embedding_dim', type=int, help='Dimension of the encoder embedding')
    parser.add_argument('--embedding_mode', type=str, help='Mode of the embedding')
    parser.add_argument('--max_shift', type=float, help='Max shift in x')
    parser.add_argument('--pos_mode', type=str, help='Positional encoding mode')
    parser.add_argument('--sizes', type=lambda s: [int(item) for item in s.split(',')], help="sizes for multiresolution")

    ################################################################
    #  Encoder
    ################################################################
    # Encoder
    parser.add_argument("--encoder_dim", type=int, help="Dimension of the encoder")
    parser.add_argument("--encoder_depth", type=int, help="Depth of the encoder")
    parser.add_argument("--encoder_heads", type=int, help="Heads of the encoder")
    parser.add_argument("--image_size", type=lambda s: [int(item) for item in s.split(',')], help="Image size")
    parser.add_argument('--model_path', type=str, help='Path to pretrained model')
    parser.add_argument('--freeze', type=eval, help='Freeze encoder weights')
    parser.add_argument('--min_lr', type=float, help='Minimum learning rate')
    parser.add_argument('--max_lr', type=float, help='Maximum learning rate')

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