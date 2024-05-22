# Data Generation

## Usage

To generate 2D combined data:

```
python 2D_combined.py --num_samples=[# samples] --split=['train', 'valid'] 
```

Note that this could take a while and use a couple GB of storage.

To generate 1D advection data:
```
python 1D_advection.py --num_samples=[# samples] --split=['train', 'valid'] 
```

To generate 1D HeatBC data:
```
python 1D_heatBC.py --num_samples=[# samples] --split=['train', 'valid'] --BC=['Dirichlet', 'Neumann']
```
Note that FEniCS has conflicts with h5py, so data is saved as raw .npy files and must be postprocessed in another conda env to .h5py. A script is provided to do that: np_to_h5.py. 