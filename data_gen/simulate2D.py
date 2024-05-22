import numpy as np
import torch
from utils import RandomSin
from tqdm import tqdm

def simulate_burgers(args, nu, cx, cy, idx, device='cpu'):
    SAVE_STEPS = args.nt
    TOTAL_TIME = args.total_time
    nx = args.nx
    ny = args.ny
    J = args.J
    lj = args.lj
    length = args.length


    nx = 2*nx # Double the resolution for stability
    ny = 2*ny 

    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)

    dt = 5e-5
    nt = int(TOTAL_TIME/dt)
    SAVE_EVERY = nt//SAVE_STEPS

    x = torch.linspace(-1, 1, nx, device=device)
    y = torch.linspace(-1, 1, ny, device=device)
    grid = torch.meshgrid(x, y)

    u = torch.zeros((ny, nx), device=device)  
    v = torch.zeros((ny, nx), device=device)
    un = torch.zeros((ny, nx), device=device)
    vn = torch.zeros((ny, nx), device=device)

    ###Assign initial conditions
    f = RandomSin((nx, ny), J, lj, length, device=device)
    u = f.sample(grid=grid, seed=idx)
    v = f.sample(grid=grid, seed=idx)
    u0 = u.clone()
    v0 = v.clone()

    #for n in tqdm(range(nt + 1)): ##loop across number of time steps
    all_us = torch.empty((SAVE_STEPS, nx, ny), device=device)
    all_vs = torch.empty((SAVE_STEPS, nx, ny), device=device)
    times = torch.empty(SAVE_STEPS, device=device)

    ## Save initial condition
    all_us[0] = u0
    all_vs[0] = v0
    times[0] = 0

    for n in range(nt-1): ##loop across number of time steps -1 because we already have the initial condition

        # Calculate finite differences
        un = u.clone()
        vn = v.clone()

        # Calculate finite differences for diffusion term
        diff_ux = (torch.roll(un, shifts=(1), dims=(1)) + torch.roll(un, shifts=(-1), dims=(1)) - 2*un)
        diff_uy = (torch.roll(un, shifts=(1), dims=(0)) + torch.roll(un, shifts=(-1), dims=(0)) - 2*un)
        diff_u = diff_ux + diff_uy

        diff_vx = (torch.roll(vn, shifts=(1), dims=(1)) + torch.roll(vn, shifts=(-1), dims=(1)) - 2*vn)
        diff_vy = (torch.roll(vn, shifts=(1), dims=(0)) + torch.roll(vn, shifts=(-1), dims=(0)) - 2*vn)
        diff_v = diff_vx + diff_vy

        # Calculate finite differences for nonlinear advection term
        if(cx <= 0 and cy >= 0):
            adv_u = -cx*un*(un - torch.roll(un, shifts=(-1), dims=(1))) + cy*vn*(un - torch.roll(un, shifts=(1), dims=(0)))
            adv_v = cy*vn*(vn - torch.roll(vn, shifts=(1), dims=(0))) - cx*un*(vn - torch.roll(vn, shifts=(-1), dims=(1)))
        elif(cx >= 0 and cy >= 0):
            adv_u = cx*un*(un - torch.roll(un, shifts=(1), dims=(1))) + cy*vn*(un - torch.roll(un, shifts=(1), dims=(0)))
            adv_v = cy*vn*(vn - torch.roll(vn, shifts=(1), dims=(0))) + cx*un*(vn - torch.roll(vn, shifts=(1), dims=(1)))
        elif(cx <= 0 and cy <= 0):
            adv_u = -cx*un*(un - torch.roll(un, shifts=(-1), dims=(1))) - cy*vn*(un - torch.roll(un, shifts=(-1), dims=(0)))
            adv_v = -cy*vn*(vn - torch.roll(vn, shifts=(-1), dims=(0))) - cx*un*(vn - torch.roll(vn, shifts=(-1), dims=(1)))
        elif(cx >= 0 and cy <= 0):
            adv_u = cx*un*(un - torch.roll(un, shifts=(1), dims=(1))) - cy*vn*(un - torch.roll(un, shifts=(-1), dims=(0)))
            adv_v = -cy*vn*(vn - torch.roll(vn, shifts=(-1), dims=(0))) + cx*un*(vn - torch.roll(vn, shifts=(1), dims=(1)))

        # Calculate update
        u = nu*dt*diff_u/dx**2 - dt*adv_u/dx + u
        v = nu*dt*diff_v/dy**2 - dt*adv_v/dy + v

        if torch.isnan(u).any():
            raise ValueError('UNSTABLE')
    

        if((n+1)%SAVE_EVERY == 0):
            all_us[(n+1)//SAVE_EVERY] = u.clone()
            all_vs[(n+1)//SAVE_EVERY] = v.clone()
            times[(n+1)//SAVE_EVERY] = TOTAL_TIME*(n+1)/nt


    return (all_us, all_vs), grid, times

def simulate_heat(args, nu, idx):
    SAVE_STEPS = args.nt
    TOTAL_TIME = args.total_time
    nx = args.nx
    ny = args.ny
    J = args.J
    lj = args.lj
    length = args.length

    dx = 2 / (nx - 1)
    dt = 0.0001
    nt = int(TOTAL_TIME/dt)
    if(nu*dt/(dx**2) >= 0.5):
        raise ValueError("Unstable Simulation.")
    SAVE_EVERY = int(nt/SAVE_STEPS)

    # Define domain and solutions
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    grid = np.meshgrid(x, y)
    u = torch.zeros((ny, nx))  
    un = np.zeros((ny, nx))

    ###Assign initial conditions
    f = RandomSin((nx, ny), J, lj, length)
    u = torch.Tensor(f.sample(grid=grid, seed=idx))
    u0 = u.clone()

    all_us = torch.empty((SAVE_STEPS, nx, ny))
    times = torch.empty(SAVE_STEPS)

    ## Save initial condition
    all_us[0] = u0
    times[0] = 0

    for n in range(nt-1): ##loop across number of time steps (nt-1 because we already have the initial condition)
        un = u.clone()

        # Calculate finite differences for diffusion term
        diff_ux = (torch.roll(un, shifts=(1), dims=(1)) + torch.roll(un, shifts=(-1), dims=(1)) - 2*un)
        diff_uy = (torch.roll(un, shifts=(1), dims=(0)) + torch.roll(un, shifts=(-1), dims=(0)) - 2*un)
        diff_u = diff_ux + diff_uy

        # Calculate update
        u = nu*dt*diff_u/dx**2 + u

        if((n+1)%SAVE_EVERY == 0):
            all_us[(n+1)//SAVE_EVERY] = u
            times[(n+1)//SAVE_EVERY] = TOTAL_TIME*(n+1)/nt

    return all_us, grid, times

def simulate_advection(args, cx, cy, idx):
    SAVE_STEPS = args.nt
    TOTAL_TIME = args.total_time
    nx = args.nx
    ny = args.ny
    J = args.J
    lj = args.lj
    length = args.length


    dt = TOTAL_TIME/SAVE_STEPS
    nt = int(np.ceil(TOTAL_TIME/dt))

    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)

    # Generate initial condition
    grid = np.meshgrid(x, y)
    f = RandomSin((nx, ny), J, lj, length)
    u = torch.Tensor(f.sample(grid=grid, seed=idx))

    all_us = torch.empty((nt, nx, ny))
    times = torch.empty(nt)
    all_us[0] = u.clone()
    times[0] = 0

    for n in range(1, nt): ##loop across number of time steps -1 because we already have the initial condition
        x_adv = -dt*n*cx
        y_adv = -dt*n*cy

        # Make new grid and subtract c*t
        new_x = x - x_adv
        new_y = y - y_adv
        new_grid = np.meshgrid(new_x, new_y)

        # Sample function at new grid
        new_u = torch.Tensor(f.sample(grid=new_grid, seed=idx))
        
        all_us[n] = new_u.clone()
        times[n] = TOTAL_TIME*(n)/nt

    return all_us, grid, times