# Masked Autoencoders are PDE Learners

This repository is the official implementation of [Masked Autoencoders are PDE Learners](https://arxiv.org/abs/2403.17728)

## Requirements

To install requirements:

```setup
conda env create --name envname --file=environment.yml
```

## Datasets
### Downloading Data
Full datasets used in the paper are avaiable [here](https://zenodo.org/records/13355846). 

Data is organized into the following structure:

- Split [train/valid]
    - u : nodal values of the PDE solution, in shape [num_samples, temporal_resolution, spatial_resolution]
    - x : coordinates of the spatial domain, in shape [spatial_resolution]
    - t : timesteps of the PDE solution, in shape [temporal_resolution]
    - coefficients [alpha, beta, gamma, etc.]: coefficients of the solved PDE solution, in shape [num_samples, coord_dim]
      
### Generating Data
Data was generated according to parameters detailed in the paper using the code below. In general, data is expected to an .h5 file; we provide sample [datasets](data_gen/data/) to illustrate its organization.

- [Message Passing Neural PDE Solvers](https://github.com/brandstetter-johannes/MP-Neural-PDE-Solvers?tab=readme-ov-file)
    - 1D KdV Burgers equation
    - 1D Heat equation, Periodic BCs
    - 1D inviscid Burgers equation
    - 1D Wave equation
- [Lie Point Symmetry Data Augmentation for Neural PDE Solvers](https://github.com/brandstetter-johannes/LPSDA)
    - 1D KS Equation
- [Fourier Neural Operator for Parametric Partial Differential Equations](https://github.com/khassibi/fourier-neural-operator) (Update: Repo has been deprecated)
    - 2D Incompressible NS
- [Towards multi-spatiotemporal-scale generalized PDE modeling](https://huggingface.co/datasets/pdearena/NavierStokes-2D-conditoned)
    - 2D Smoke Buoyancy 
- [2D_combined.py](data_gen/2D_combined.py)
    - 2D Heat, Adv, Burgers equations
- [1D_advection.py](data_gen/1D_advection.py)
    - 1D Advection
- [1D_heatBC.py](data_gen/1D_heatBC.py)
    - 1D Heat 
    - Requires a working [FEniCS installation](https://fenicsproject.org/download/archive/)


## Training
For specific experiments, please refer the appropriate .yaml file and command line args in the [configs](configs) directory.

### MAE Pretraining
```
python pretrain.py --config=configs/[dim]/pretrain.yaml
```
### Feature Prediction
```
python regression.py --config=configs/[dim]/regression/[pde].yaml 
```
### Time-stepping
```
python timestep.py --config=configs/[dim]/timestepping/[pde].yaml 
```
### Super-resolution
```
python sr.py --config=configs/[dim]/sr/[pde].yaml 
```

## Citation
If you find the implementations or datasets useful, please feel free to cite the paper:
>@misc{zhou2024maskedautoencoderspdelearners,
>title={Masked Autoencoders are PDE Learners}, 
>author={Anthony Zhou and Amir Barati Farimani},
>year={2024},
>eprint={2403.17728},
>archivePrefix={arXiv},
>primaryClass={cs.LG},
>url={https://arxiv.org/abs/2403.17728},
>}
