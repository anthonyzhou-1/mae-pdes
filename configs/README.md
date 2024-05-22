# Experiments

## Usage
Config files are organized by the PDE dimension [1D/2D], the experiment [regression/timestep/sr] by the PDE name [PDE].yaml. Each config file expects a path to the corresponding PDE data. 

### Experiments

To run a baseline experiment, the default configs will work:

```
python [experiment].py --config=configs/[dim]/[experiment]/[pde].yaml 
```

To run an experiment with a pretrained MAE encoder: 

```
python [experiment].py --config=configs/[dim]/[experiment]/[pde].yaml --encoder=[VIT/VIT3D] --pretrained_path=[path_to_pretrained_MAE]
```

To run an experiment with a pretrained, frozen MAE encoder:
```
python [experiment].py --config=configs/[dim]/[experiment]/[pde].yaml --encoder=[VIT/VIT3D] --pretrained_path=[path_to_pretrained_MAE] --freeze=True
```

To run an experiment with a randomly initialized encoder (N/A for regression experiments):
```
python [experiment].py --config=configs/[dim]/[experiment]/[pde].yaml --encoder=[VIT/VIT3D] 
```

To run an experiment with a linear encoder and ground truth PDE parameters (N/A for regression experiments):
```
python [experiment].py --config=configs/[dim]/[experiment]/[pde].yaml --encoder=Linear --add_vars=True
```

### Additional Parameters

For pretraining, the patch size, augmentation ratio, and masking ratio can be varied by passing:

```
--patch_size=pt/py, px --temporal_patch_size=pt --augmentation_ratio=0.5 --masking_ratio=0.75
```

For time-stepping experiments, a model can optionally be chosen by passing:

```
--model=["FNO1D", "FNO2D", "Unet1D", "Unet2D"]
```

To run a script with a specific seed or device:
```
--seed=seed --device=device
```

By default, the training scripts will load the entire dataset into GPU memory. For 2D experiments this may be too memory intensive for certain GPU cards, so that option may be turned off by passing:

```
--load_all=["True", "False"]
```

