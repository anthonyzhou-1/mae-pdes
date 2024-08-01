from helpers.model_helper import get_model, get_dataloader, get_embedder
from common.utils import DataCreator, LpLoss
import yaml
import torch 

path = "./configs/2D/timestep/ns.yaml"
with open(path) as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

args = Struct(**config)

device = args.device

loader = get_dataloader(args.valid_path,
                                args,
                                mode="valid")

data_creator = DataCreator(time_history=args.time_window,
                            t_resolution=args.base_resolution[0],
                            t_range=args.t_range,
                            x_resolution=args.base_resolution[1],
                            x_range=args.x_range)

# Encoder
model, optimizer, scheduler = get_model(args, device)
embedder = get_embedder(args)

pretrained_path = "checkpoints/1_Forecast_2D_ns_none_FNO2D_____5101645.pth"
model.load_state_dict(torch.load(pretrained_path, map_location=device))
print("Model loaded from: ", pretrained_path)

# Loop over every data sample
nr_gt_steps = 2
criterion = LpLoss(d = args.pde_dim, p = 2)
losses = []
horizon = args.horizon

for u, variables, in loader:

    batch_size = u.shape[0]
    losses_tmp = []

    with torch.no_grad():
        same_steps = [data_creator.time_history * nr_gt_steps] * batch_size
        data, labels = data_creator.create_data_labels(u, same_steps)
        outputs = [data]

        z = embedder(data, loader.dataset.x, loader.dataset.t, same_steps)
        pred = model(data, variables, z)

        loss = criterion(pred, labels.to(device))
        losses_tmp.append(loss)
        data = pred 

        outputs.append(data)

        # Unroll trajectory and add losses which are obtained for each unrolling
        for step in range(data_creator.time_history * (nr_gt_steps + 1), horizon - data_creator.time_history + 1, data_creator.time_history):
            same_steps = [step] * batch_size
            _, labels = data_creator.create_data_labels(u, same_steps) 

            z = embedder(data, loader.dataset.x, loader.dataset.t, same_steps)
            pred = model(data, variables, z)

            loss = criterion(pred, labels.to(device))
            losses_tmp.append(loss)
            data = pred 
            outputs.append(data)

    losses.append(torch.sum(torch.stack(losses_tmp)))
    break

print("Losses: ", losses)
print("Mean Loss: ", torch.mean(torch.stack(losses)))
outputs = torch.stack(outputs)
print("Output shape: ", outputs.shape)

import pickle 
with open("checkpoints/outputs.pkl", "wb") as f:
    pickle.dump(outputs, f)