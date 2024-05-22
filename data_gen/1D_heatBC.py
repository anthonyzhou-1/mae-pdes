from fenics import *
import numpy as np
from tqdm import tqdm
import argparse

def process_u(u):
    u_array = u.vector().get_local()
    
    u_array = np.flip(u_array, axis=0)
    u_out = u_array[::4]
    
    return u_out[:-1]

def get_IC(L=16, mode="Dirichlet"):
    alphas = np.random.rand(5) - 0.5
    ls = (2/L * np.pi) * np.random.randint(1, 4, 5)
    phis = np.pi * np.random.randint(0, 2, 5)

    alpha1, alpha2, alpha3, alpha4, alpha5 = alphas
    l1, l2, l3, l4, l5 = ls
    phi1, phi2, phi3, phi4, phi5 = phis

    if mode == "Dirichlet":
        u_0 = Expression('alpha1 * sin(l1 * x[0] + phi1) + alpha2 * sin(l2 * x[0] + phi2) + alpha3 * sin(l3 * x[0] + phi3) + alpha4 * sin(l4 * x[0] + phi4) + alpha5 * sin(l5 * x[0] + phi5)',
                     degree=2, alpha1=alpha1, alpha2=alpha2, alpha3=alpha3, alpha4=alpha4, alpha5=alpha5, l1=l1, l2=l2, l3=l3, l4=l4, l5=l5, phi1=phi1, phi2=phi2, phi3=phi3, phi4=phi4, phi5=phi5)
    
    elif mode == "Neumann":
        u_0 = Expression('alpha1 * cos(l1 * x[0] + phi1) + alpha2 * cos(l2 * x[0] + phi2) + alpha3 * cos(l3 * x[0] + phi3) + alpha4 * cos(l4 * x[0] + phi4) + alpha5 * cos(l5 * x[0] + phi5)',
                     degree=2, alpha1=alpha1, alpha2=alpha2, alpha3=alpha3, alpha4=alpha4, alpha5=alpha5, l1=l1, l2=l2, l3=l3, l4=l4, l5=l5, phi1=phi1, phi2=phi2, phi3=phi3, phi4=phi4, phi5=phi5)
    
    return u_0

def solve_heat(nu, mode="Dirichlet", total_time=2, total_length=16):

    T = total_time            # final time
    num_steps = 250     # number of time steps
    dt = T / num_steps # time step size
    length = total_length

    # Create mesh and define function space
    nx = 200
    mesh = IntervalMesh(nx, 0, length)
    V = FunctionSpace(mesh, 'CG', 2)
    
    solution = np.zeros((num_steps, nx//2))

    tol = 1E-14
    def left_boundary(x, on_boundary):
        return on_boundary and x[0] < tol
    def right_boundary(x, on_boundary):
        return on_boundary and x[0] > length - tol

    if mode == "Dirichlet":
        bcs = [DirichletBC(V, Constant(0), left_boundary),
            DirichletBC(V, Constant(0), right_boundary),]

    u_0 = get_IC(L=length, mode=mode)

    u_n = project(u_0, V)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)

    F = u*v*dx + dt/nu*dot(grad(u), grad(v))*dx - (u_n)*v*dx
    a, L = lhs(F), rhs(F)

    # Time-stepping
    u = Function(V)
    t = 0
    for n in range(num_steps):
        
        # Update current time
        t += dt

        # Compute solution
        if mode == "Dirichlet":
            solve(a == L, u, bcs)
        elif mode == "Neumann":
            solve(a == L, u)

        # Update previous solution
        u_n.assign(u)
        
        solution[n] = process_u(u)
    
    return solution

def main(args):
    print(f'Equation: {args.equation}')
    print(f'Split: {args.split}')
    print(f'Number of samples: {args.num_samples}')

    num_samples = args.num_samples
    

    if args.split == 'train':
        seed = 0
    elif args.split == 'valid':
        seed = 100000 + 1000
    elif args.split == 'test':
        seed = 200000 + 1000
    
    np.random.seed(seed)
    nus = np.random.uniform(0.1, .8, num_samples)

    us = np.zeros((num_samples, args.nt, args.nx))

    for i in tqdm(range(num_samples)):
        nu = nus[i]

        u = solve_heat(nu, mode = args.BC, total_time=args.total_time, total_length=args.length)
        
        us[i] = u

    times = np.linspace(0, args.total_time, args.nt)
    x = np.linspace(0, args.length, args.nx)

    data = {'u': us, 'nu': nus, 't': times, 'x': x}

    np.save(f'data/{args.equation}_{args.split}_{args.num_samples}.npy', data)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generating PDE data')
    parser.add_argument('--equation', type=str, default='HeatBC',
                        help='Equation for which data should create for: [HeatBC]')
    parser.add_argument('--nt', type=int, default=250,
                        help='Number of steps to save')
    parser.add_argument('--total_time', type=float, default=2,
                        help='Total time for which we want to solve the PDE')
    parser.add_argument('--num_samples', type=int, default=256,
                        help='Number of samples to create')
    parser.add_argument('--split', type=str, default='train',
                        help='Split for which data should be created: [train, valid, test]')
    parser.add_argument('--J', type=int, default=5,
                        help='Number of sine functions')
    parser.add_argument('--length', type=float, default=16,
                        help='Length of the domain')
    parser.add_argument('--BC', type=str, default='Dirichlet',)
    parser.add_argument('--nx', type=int, default=100,
                        help='Number of points in the domain')

    args = parser.parse_args()
    main(args)

