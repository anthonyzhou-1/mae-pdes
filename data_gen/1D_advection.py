import argparse
from scipy.stats import uniform
import h5py
from tqdm import tqdm
from simulate1D import *

def get_coeffs(args):
    if args.equation == 'Advection':
        low = 0.1
        high = 5.0

        axs = uniform.rvs(low, high-low, size=args.num_samples) # Sample from uniform distribution [.1, 2.5]
    else:
        raise ValueError("Invalid equation")
    
    return axs
            
def main(args):
    print(f'Equation: {args.equation}')
    print(f'Split: {args.split}')
    print(f'Number of samples: {args.num_samples}')

    num_samples = args.num_samples
    
    axs = get_coeffs(args)

    h5f = h5py.File(f"{args.equation}_{args.split}_{args.num_samples}.h5", 'a')
    dataset = h5f.create_group(args.split)

    h5f_u = dataset.create_dataset(f'u', (num_samples, args.nt, args.nx), dtype='f4')
    coord = dataset.create_dataset(f'x', (args.nx), dtype='f4')
    tcoord = dataset.create_dataset(f't', (args.nt), dtype='f4')
    h5f_ax = dataset.create_dataset(f'a', (num_samples), dtype='f4') 

    h5f_ax[:] = axs

    if args.split == 'train':
        seed = 0
    elif args.split == 'valid':
        seed = 100000 + 1000
    elif args.split == 'test':
        seed = 200000 + 1000

    for i in tqdm(range(num_samples)):
        ax = axs[i]

        if args.equation == 'Advection':
            #print("Advection, AX: ", ax)
            u, x, times = simulate_advection(args, ax, i+seed+20000)
        
        h5f_u[i] = u
    tcoord[:] = times
    coord[:] = x

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generating PDE data')
    parser.add_argument('--equation', type=str, default='Advection',
                        help='Equation for which data should create for: [Advection]')
    parser.add_argument('--nt', type=int, default=250,
                        help='Number of steps to save')
    parser.add_argument('--total_time', type=float, default=2,
                        help='Total time for which we want to solve the PDE')
    parser.add_argument('--nx', type=int, default=100,
                        help='Spatial resolution')
    parser.add_argument('--num_samples', type=int, default=256,
                        help='Number of samples to create')
    parser.add_argument('--split', type=str, default='train',
                        help='Split for which data should be created: [train, valid, test]')
    parser.add_argument('--J', type=int, default=5,
                        help='Number of sine functions')
    parser.add_argument('--length', type=float, default=16,
                        help='Length of the domain')

    args = parser.parse_args()
    main(args)
