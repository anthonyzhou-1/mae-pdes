import wandb
from datetime import datetime
import argparse
import json
from torch.utils.data import DataLoader
from common.utils import DataCreator, Embedder, Normalizer, LpLoss
from helpers.train_helper import *
from helpers.model_helper import *
import numpy as np
import torch
import random
from tqdm import tqdm

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
          criterion,
          device: torch.cuda.device="cpu",
          optimizer_embedder=None) -> None:
    
    print(f'Starting epoch {epoch}...')
    model.train()

    max_unrolling = epoch if epoch <= args.unrolling else args.unrolling
    unrolling = [r for r in range(max_unrolling + 1)]

    n_inner = data_creator.t_res

    if args.pde_dim == 2:
        n_inner = n_inner//10 # fewer inner steps for 2D PDEs
    for i in tqdm(range(n_inner)):
        losses = training_loop(model,
                               optimizer,
                               unrolling, 
                               loader, 
                               data_creator, 
                               augmentation,
                               embedder,
                               normalizer,
                               criterion, 
                               device,
                               optimizer_embedder=optimizer_embedder,)
        scheduler.step()
        '''
        Uncomment if using wandb
        wandb.log({"train/loss": losses})
        '''

def test(args: argparse,
         model: torch.nn.Module,
         loader: DataLoader,
         data_creator: DataCreator,
         augmentation,
         embedder: Embedder,
         normalizer: Normalizer,
         criterion,
         device: torch.cuda.device="cpu",
         verbose: bool=False) -> torch.Tensor:

    model.eval()       

    losses = test_unrolled_losses(model=model,
                                  nr_gt_steps=args.nr_gt_steps,
                                  loader=loader,
                                  data_creator=data_creator,
                                  horizon=args.horizon,
                                  augmentation=augmentation,
                                  embedder=embedder,
                                  normalizer=normalizer,
                                  criterion=criterion,
                                  device=device,
                                  verbose=verbose)

    return losses

def main(args: argparse):
    if isinstance(args.seed, str):
        args.seed = int(args.seed)

    if isinstance(args.freeze, str):
        args.freeze = eval(args.freeze)

    if isinstance(args.add_vars, str):
        args.add_vars = eval(args.add_vars)

    if isinstance(args.load_all, str):
        args.load_all = eval(args.load_all)

    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    print(f"Running with seed {seed}")
    print(f"Running on device {args.device}") 
    print(f"Running with batch size {args.batch_size}")
    print(f"Running with PDE {args.pde}")

    # Setup
    dateTimeObj = datetime.now()
    timestring = f'{dateTimeObj.date().month}{dateTimeObj.date().day}{dateTimeObj.time().hour}{dateTimeObj.time().minute}'
    pretrained_str = "pretrained" if args.pretrained_path != "none" else "_"
    freeze_str = 'freeze' if args.freeze else "_"
    name = f'{args.seed}_{args.description}_{args.pde}_{args.encoder}_{args.model}_{freeze_str}_{pretrained_str}_{timestring}'


    run = wandb.init(project="pde-context-ablations",
                     entity='ayz2',
                    name = name,
                    config=vars(args),
                    mode=args.wandb_mode)

    device = args.device

    # Data loading

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
                               x_range=args.x_range)
    
    # Encoder
    model, optimizer, scheduler = get_model(args, device)

    # Extras
    augmentation = get_augmentation(args, train_loader.dataset.dx, train_loader.dataset.dt)
    embedder = get_embedder(args)

    # workaround for loading multiresolution spatial embeddings
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
    criterion = LpLoss(d = args.pde_dim, p = 2)
    num_epochs = args.num_epochs
    save_path= f'checkpoints/{name}.pth'
    min_val_loss = 10e10

    verbose = args.verbose

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
              criterion, 
              device=device,
              optimizer_embedder=optimizer_embedder)
        
        print("Evaluation on validation dataset:")

        val_loss = test(args, 
                        model,
                        valid_loader, 
                        data_creator, 
                        augmentation,
                        embedder,
                        normalizer,
                        criterion, 
                        device=device, 
                        verbose=verbose)

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
                'loss': min_val_loss,
            }

            torch.save(checkpoint, save_path)
            print(f"Saved model at {save_path}\n")
            min_val_loss = val_loss
    '''
    Uncomment if using wandb
    wandb.log({
        "min_val_loss/loss": min_val_loss,
    })  
    '''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a PDE Solver')

    ################################################################
    #  GENERAL
    ################################################################
    # General
    parser.add_argument('--description', type=str, help='Description of the run')
    parser.add_argument('--config', type=str, help='Load settings from file in json format. Command line options override values in file.')
    parser.add_argument('--seed', type=int, help='Seed for reproducibility')
    parser.add_argument('--device', type=str, help='Device to run on')
    parser.add_argument('--encoder', type=str, help='Encoder architecture')
    parser.add_argument('--model', type=str, help='Model architecture')
    parser.add_argument('--batch_size', type=int, help='Batch Size')
    parser.add_argument('--val_batch_size', type=int, help='Validation Batch Size')
    parser.add_argument('--num_epochs', type=int, help='Number of training epochs')    
    parser.add_argument('--pretrained_path', type=str, help='Description of the run')
    parser.add_argument('--wandb_mode', type=str, help='Wandb mode')

    ################################################################
    #  DATA
    ################################################################

    # Data paths
    parser.add_argument('--train_path', type=str, help='Path to training data')
    parser.add_argument('--valid_path', type=str, help='Path to validation data')
    parser.add_argument('--load_all', type=eval, help='Load all data into memory')
    parser.add_argument('--augmentation_ratio', type=float, help='Augmentation ratio for data')
    parser.add_argument('--add_vars', type=eval, help='Add variables to data')

    ################################################################
    #  Timesteppers
    ################################################################
    # FNO parameters
    parser.add_argument('--fno_modes', type=int, help='Number of modes for FNO')
    parser.add_argument('--fno_width', type=int, help='Width of FNO')
    parser.add_argument('--fno_num_layers', type=int, help='Number of layers for FNO')

    ################################################################
    #  BACKBONES
    ################################################################
    # ViT
    parser.add_argument('--image_size', type=lambda s: [int(item) for item in s.split(',')], help="input size in height x width")
    parser.add_argument("--patch_size", type=lambda s: [int(item) for item in s.split(',')], help="patch size in height x width")
    parser.add_argument("--encoder_dim", type=int, help="Dimension of the encoder")
    parser.add_argument("--encoder_depth", type=int, help="Depth of the encoder")
    parser.add_argument("--encoder_heads", type=int, help="Heads of the encoder")
    parser.add_argument('--model_path', type=str, help='Path to pretrained model')
    parser.add_argument('--embedding_dim', type=int, help='Embedding dimension')
    parser.add_argument('--min_lr', type=float, help='Minimum learning rate')
    parser.add_argument('--max_lr', type=float, help='Max learning rate')
    parser.add_argument('--freeze', type=eval, help='Freeze backbone')

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