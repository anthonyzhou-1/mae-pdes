import torch
import random
from torch.utils.data import DataLoader
from common.utils import *

def training_loop(model: torch.nn.Module,
                  optimizer: torch.optim,
                  unrolling: list,
                  loader: DataLoader,
                  data_creator: DataCreator,
                  augmentation,
                  embedder: Embedder,
                  normalizer: Normalizer,
                  criterion, 
                  device: torch.cuda.device="cpu",
                  optimizer_embedder=None) -> torch.Tensor:

    losses = []
    for u, variables in loader:
        if normalizer is not None:
            u = normalizer.normalize(u)

        u = augmentation(u)

        batch_size = u.shape[0]
        optimizer.zero_grad()

        if optimizer_embedder is not None:
            optimizer_embedder.zero_grad()

        # Randomly choose number of unrollings
        unrolled_datas = random.choice(unrolling)
        steps = [t for t in range(data_creator.time_history,
                                  data_creator.t_res - data_creator.time_history - (data_creator.time_history * unrolled_datas) + 1)]
        

        random_steps = random.choices(steps, k=batch_size)
        data, labels = data_creator.create_data_labels(u, random_steps) # data is at t, labels at t+1

        with torch.no_grad():
            for _ in range(unrolled_datas):
                random_steps = [rs + data_creator.time_history for rs in random_steps]
                _, labels = data_creator.create_data_labels(u, random_steps) 
                
                z = embedder(data, loader.dataset.x, loader.dataset.t, random_steps)
                pred = model(data, variables, z)
                data = pred

        # Forward pass of encoder model to make an embedding at t+1
        z = embedder(data, loader.dataset.x, loader.dataset.t, random_steps)
        pred = model(data, variables, z)
        loss = criterion(pred, labels.to(device))

        # Backpropagation and stepping the optimizer
        loss = torch.sqrt(loss)
        loss.backward()
        losses.append(loss.detach())

        optimizer.step()
        if optimizer_embedder is not None:
            optimizer_embedder.step()

    losses = torch.stack(losses)
    return torch.mean(losses)

def test_timestep_losses(model: torch.nn.Module,
                         steps: list,
                         loader: DataLoader,
                         data_creator: DataCreator,
                         augmentation,
                         embedder: Embedder,
                         normalizer: Normalizer,
                         criterion: torch.nn.modules.loss,
                         device: torch.cuda.device = "cpu",
                         verbose = False) -> None:
    model.eval()

    for step in steps:

        # Condition to skip steps that are not spaced by time_history
        if (step != data_creator.time_history and step % data_creator.time_history != 0):
            continue

        losses = []
        # Loop over every data sample for a given time window
        for u, variables in loader:
            if normalizer is not None:
                u = normalizer.normalize(u)

            u = augmentation(u)
            batch_size = u.shape[0]

            with torch.no_grad():
                # Create data and labels for current time window in shape [batch_size, time_history, x_res]
                same_steps = [step]*batch_size
                data, labels = data_creator.create_data_labels(u, same_steps)

                z = embedder(data, loader.dataset.x, loader.dataset.t, same_steps)
                pred = model(data, variables, z)

                loss = criterion(pred, labels.to(device))
                losses.append(loss)

        losses = torch.stack(losses)
        if verbose:
            print(f'Step {step}, mean loss {torch.mean(losses)}')


def test_unrolled_losses(model: torch.nn.Module,
                         nr_gt_steps: int,
                         loader: DataLoader,
                         data_creator: DataCreator,
                         horizon: int, 
                         augmentation,
                         embedder: Embedder,
                         normalizer: Normalizer,
                         criterion: torch.nn.modules.loss,
                         device: torch.cuda.device = "cpu",
                         verbose = False) -> torch.Tensor:
    losses = []
    model.eval()

    # Loop over every data sample
    for u, variables, in loader:
        if normalizer is not None:
            u = normalizer.normalize(u)

        u = augmentation(u)

        batch_size = u.shape[0]
        losses_tmp = []

        with torch.no_grad():
            same_steps = [data_creator.time_history * nr_gt_steps] * batch_size
            data, labels = data_creator.create_data_labels(u, same_steps)

            z = embedder(data, loader.dataset.x, loader.dataset.t, same_steps)
            pred = model(data, variables, z)

            loss = criterion(pred, labels.to(device))
            losses_tmp.append(loss)
            data = pred 

            # Unroll trajectory and add losses which are obtained for each unrolling
            for step in range(data_creator.time_history * (nr_gt_steps + 1), horizon - data_creator.time_history + 1, data_creator.time_history):
                same_steps = [step] * batch_size
                _, labels = data_creator.create_data_labels(u, same_steps) 

                z = embedder(data, loader.dataset.x, loader.dataset.t, same_steps)
                pred = model(data, variables, z)

                loss = criterion(pred, labels.to(device))
                losses_tmp.append(loss)
                data = pred 

        losses.append(torch.sum(torch.stack(losses_tmp)))

    losses = torch.stack(losses)
    if verbose:
        print(f'Unrolled forward losses {torch.mean(losses)}')

    return torch.mean(losses)
