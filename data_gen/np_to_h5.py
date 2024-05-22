import h5py 
import numpy as np
from tqdm import tqdm

np_path = "NP PATH HERE"

data = np.load(np_path, allow_pickle=True)

new_path = "NEW H5 PATH HERE"
h5f = h5py.File(new_path, 'a')

data_u = data[()]["u"]
data_x = data[()]["x"]
data_t = data[()]["t"]
data_nu = data[()]["nu"]

split = 'train'
dataset = h5f.create_group(split)

num_samples, nt, nx = data_u.shape

h5f_u = dataset.create_dataset(f'u', (num_samples, nt, nx), dtype='f4')
coord = dataset.create_dataset(f'x', (nx), dtype='f4')
tcoord = dataset.create_dataset(f't', (nt), dtype='f4')
h5f_nu = dataset.create_dataset(f'nu', (num_samples,), dtype='f4')

coord[:] = data_x
tcoord[:] = data_t
h5f_nu[:] = data_nu

for i in tqdm(range(num_samples)):
    h5f_u[i] = data_u[i]

h5f.close()
