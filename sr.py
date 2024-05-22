import wandb
from datetime import datetime
import argparse
import json
import os 
import yaml
from torch.utils.data import DataLoader
import numpy as np
import torch
import random

from common.utils import DataCreator
from common.augmentation import *
from helpers.model_helper import *
from tqdm import tqdm 

def train(args: argparse,
          epoch: int,
          model: torch.nn.Module,
          optimizer: torch.optim,
          scheduler: torch.optim.lr_scheduler,
          loader: DataLoader,
          data_creator: DataCreator,
          normalizer: Normalizer,
          device: torch.cuda.device="cpu")-> None:
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
    model.train()
    t_res = data_creator.t_res

    for i in tqdm(range(args.n_inner)):
        losses = []

        for u, variables in loader:
            if normalizer is not None:
                u = normalizer.normalize(u)
            
            batch_size = u.shape[0]
            optimizer.zero_grad()

             # Partition trajectory
            max_start_time = t_res - data_creator.time_history
            start_time = random.choices([t for t in range(max_start_time+1)], k=batch_size)
            u_window = data_creator.create_data(u, start_time) # b, time_window, nx

            u_window = u_window.to(device)

            loss = model(u_window, variables)

            # Backward pass
            loss.backward()

            # step optimizer
            optimizer.step()
            scheduler.step()
            losses.append(loss.detach() / args.batch_size)

        
        losses = torch.stack(losses)
        losses_out = torch.mean(losses)
        if args.verbose:
            print(f'Training Loss (progress: {i / t_res:.2f}): {losses_out}')

        '''
        Uncomment if using wandb
        wandb.log({"train/loss": losses_out})
        '''


def test(args: argparse,
          model: torch.nn.Module,
          loader: DataLoader,
          data_creator: DataCreator,
          normalizer: Normalizer,
          horizon: int, 
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
    model.eval()
    print("Validation...")
    errors = []
    device = args.device
    with torch.no_grad():
        for u, variables in loader:
            if normalizer is not None:
                u = normalizer.normalize(u)

            batch_size = u.shape[0]
            steps = [t for t in range(0, horizon+1 - data_creator.time_history, data_creator.time_history)]

            losses = []
            for step in steps:
                same_steps = [step]*batch_size
                u_window = data_creator.create_data(u, same_steps)

                u_window = u_window.to(device)

                loss = model(u_window, variables)
                losses.append(loss.detach().item())

            errors.append(np.mean(losses))
    avg_error = np.mean(errors)
    if args.verbose:
        print("Average Error: ", avg_error)

    '''
    Uncomment if using wandb
    wandb.log({
            "valid/loss": avg_error,
        })
    '''

    return avg_error

def main(args: argparse):
    if isinstance(args.seed, str):
        args.seed = int(args.seed)

    if isinstance(args.freeze, str):
        args.freeze = eval(args.freeze)

    if isinstance(args.add_vars, str):
        args.add_vars = eval(args.add_vars)

    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    print(f"Running with seed {seed}")
    print(f"Running on device {args.device}") 
    print(f"Running with batch size {args.batch_size}")

    # Setup
    dateTimeObj = datetime.now()
    timestring = f'{dateTimeObj.date().month}{dateTimeObj.date().day}{dateTimeObj.time().hour}{dateTimeObj.time().minute}'
    pretrained = True if args.pretrained_path != "none" else False
    args.pretrained=pretrained
    name = f'{args.seed}_{args.description}_{args.pde}_{args.encoder}_{args.network}_{args.operator}_PT_{pretrained}_{args.freeze}freeze_{timestring}'

    '''
    Uncomment if using wandb
    run = wandb.init(project=project,
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
    model, optimizer, scheduler = get_sr_model(args, device)

    normalizer = get_normalizer(args, train_loader)

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
            normalizer,
            device=args.device)
        
        print("Evaluation on validation dataset:")

        val_loss = test(args, 
                        model,
                        valid_loader, 
                        data_creator, 
                        normalizer, 
                        horizon=args.horizon,
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
            checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
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

    run.finish()   
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
    parser.add_argument('--num_epochs', type=int, help='Number of training epochs')
    parser.add_argument('--device', type=str, help="device")
    parser.add_argument('--wandb_mode', type=str, help='Wandb mode')
    parser.add_argument('--normalize', type=eval, help='Flag for normalization')
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
    parser.add_argument('--add_vars', type=eval, help='Add variables to the input')

    ################################################################
    #  Encoder
    ################################################################
    # Encoder
    parser.add_argument('--encoder', type=str, help='Encoder')
    parser.add_argument("--encoder_dim", type=int, help="Dimension of the encoder")
    parser.add_argument("--encoder_depth", type=int, help="Depth of the encoder")
    parser.add_argument("--encoder_heads", type=int, help="Heads of the encoder")
    parser.add_argument("--image_size", type=lambda s: [int(item) for item in s.split(',')], help="Image size")
    parser.add_argument('--model_path', type=str, help='Path to pretrained model')
    parser.add_argument('--freeze', type=eval, help='Freeze encoder weights')
    parser.add_argument('--min_lr', type=float, help='Minimum learning rate')
    parser.add_argument('--max_lr', type=float, help='Maximum learning rate')

    parser.add_argument('--operator', type=str, help='Model')
    parser.add_argument('--network', type=str, help='Network')

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