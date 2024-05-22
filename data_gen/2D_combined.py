import argparse
from scipy.stats import uniform
import h5py
from tqdm import tqdm
from simulate2D import *

def get_coeffs(args):
    def get_heat():
        high = 2e-2
        low = 2e-3

        nus = uniform.rvs(low, high-low, size=args.num_samples) 
        axs = np.zeros(args.num_samples)
        ays = np.zeros(args.num_samples)
        cxs = np.zeros(args.num_samples)
        cys = np.zeros(args.num_samples)

        return nus, axs, ays, cxs, cys
    def get_adv():
        low = 0.1
        high = 2.5

        axs = uniform.rvs(low, high-low, size=args.num_samples) # Sample from uniform distribution [.1, 2.5]
        ays = uniform.rvs(low, high-low, size=args.num_samples) # Sample from uniform distribution [.1, 2.5]
        nus = np.zeros(args.num_samples)
        cxs = np.zeros(args.num_samples)
        cys = np.zeros(args.num_samples)

        return nus, axs, ays, cxs, cys

    def get_burgers():
        nu_low = 7.5e-3
        nu_high = 1.5e-2

        nus = uniform.rvs(nu_low, nu_high-nu_low, size=args.num_samples) # Sample from uniform distribution [7.5e-5, 1.5e-2]

        c_high = 1.0
        c_low = 0.5

        cxs = uniform.rvs(c_low, c_high-c_low, size=args.num_samples) 
        cys = uniform.rvs(c_low, c_high-c_low, size=args.num_samples) 
        axs = np.zeros(args.num_samples)
        ays = np.zeros(args.num_samples)
        return nus, axs, ays, cxs, cys
    
    nu_heat, ax_heat, ay_heat, cx_heat, cy_heat = get_heat()
    nu_adv, ax_adv, ay_adv, cx_adv, cy_adv = get_adv()
    nu_burgers, ax_burgers, ay_burgers, cx_burgers, cy_burgers = get_burgers()

    nus = np.concatenate([nu_heat, nu_adv, nu_burgers])
    axs = np.concatenate([ax_heat, ax_adv, ax_burgers])
    ays = np.concatenate([ay_heat, ay_adv, ay_burgers])
    cxs = np.concatenate([cx_heat, cx_adv, cx_burgers])
    cys = np.concatenate([cy_heat, cy_adv, cy_burgers])

    return nus, axs, ays, cxs, cys

            
def main(args):
    num_samples = args.num_samples * 3 
    print(f'Equation: {args.equation}')
    print(f'Split: {args.split}')
    print(f'Number of samples: {num_samples}')
    
    nus, axs, ays, cxs, cys = get_coeffs(args)

    h5f = h5py.File(f"{args.equation}_{args.split}_{num_samples}.h5", 'a')
    dataset = h5f.create_group(args.split)

    # solve at a higher resolution than saving
    nx_save = args.nx // 4
    ny_save = args.ny // 4

    h5f_u = dataset.create_dataset(f'u', (num_samples, args.nt, nx_save, ny_save), dtype='f4')
    coord = dataset.create_dataset(f'x', (2, nx_save, ny_save), dtype='f4')
    tcoord = dataset.create_dataset(f't', (args.nt), dtype='f4')
    h5f_ax = dataset.create_dataset(f'ax', (num_samples), dtype='f4') 
    h5f_ay = dataset.create_dataset(f'ay', (num_samples), dtype='f4')
    h5f_cx = dataset.create_dataset(f'cx', (num_samples), dtype='f4')
    h5f_cy = dataset.create_dataset(f'cy', (num_samples), dtype='f4')
    h5f_nu = dataset.create_dataset(f'nu', (num_samples), dtype='f4')

    h5f_ax[:] = axs
    h5f_ay[:] = ays
    h5f_cx[:] = cxs
    h5f_cy[:] = cys
    h5f_nu[:] = nus

    if args.split == 'train':
        seed = 15042
    elif args.split == 'valid':
        seed = 115042
    elif args.split == 'test':
        seed = 215042

    for i in range(3):
        for j in tqdm(range(args.num_samples)):
            idx = i * args.num_samples + j
            nu = nus[idx] 
            cx = cxs[idx]
            cy = cys[idx]
            ax = axs[idx]
            ay = ays[idx]
            if i == 2:
                #print("Heat, NU: {0:.8f}".format(nu))
                u, grid, times = simulate_heat(args, nu, idx+seed)
                u_downsample = u[:, ::4, ::4]
                grid_downsample = (grid[0][::4, ::4], grid[1][::4, ::4])

            elif i == 1:
                #print("Advection, AX: {0:.4f}, AY: {1:.4f}".format(ax, ay))
                u, grid, times = simulate_advection(args, ax, ay, idx+seed+20000)
                u_downsample = u[:, ::4, ::4]
                grid_downsample = (grid[0][::4, ::4], grid[1][::4, ::4])

            elif i == 0:
                #print("Burgers, NU: {0:.4f}, CX: {1:.4f}, CY: {2:.4f}".format(nu, cx, cy))
                (u, v), grid, times = simulate_burgers(args, nu, cx, cy, idx+seed+40000)
                u_downsample = u[:, ::8, ::8]
                grid_downsample = (grid[0][::8, ::8], grid[1][::8, ::8])
        
            h5f_u[idx] = u_downsample
    tcoord[:] = times
    coord[:] = grid_downsample


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generating PDE data')
    parser.add_argument('--equation', type=str, default='Heat_Adv_Burgers',
                        help='Equation for which data should create for')
    parser.add_argument('--nt', type=int, default=100,
                        help='Number of steps to save')
    parser.add_argument('--total_time', type=float, default=2,
                        help='Total time for which we want to solve the PDE')
    parser.add_argument('--nx', type=int, default=256,
                        help='Spatial resolution')
    parser.add_argument('--ny', type=int, default=256,
                        help='Spatial resolution')
    parser.add_argument('--num_samples', type=int, default=1024,
                        help='Number of samples to create')
    parser.add_argument('--split', type=str, default='train',
                        help='Split for which data should be created: [train, valid, test]')
    parser.add_argument('--J', type=int, default=5,
                        help='Number of sine functions')
    parser.add_argument('--lj', type=float, default=3,
                        help='Length scale')
    parser.add_argument('--length', type=float, default=2,
                        help='Length of the domain')

    args = parser.parse_args()
    main(args)
