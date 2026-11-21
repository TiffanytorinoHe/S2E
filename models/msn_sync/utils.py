import numpy as np

def poissonTrain(rate, start, t_max):
    interval = np.random.exponential(1e3/rate, int(t_max))
    train = np.concatenate(([0.0], np.cumsum(interval))) + start
    S = train[train <= t_max]
    return S

def poissonTrain_ori(rate, t_start, t_max):
    interval = np.random.exponential(1e3/rate, int(t_max))
    train = np.cumsum(interval) + t_start
    S = train[train <= t_max]
    return S
    
def gridSpikes_poisson(rate, grid_num, t_on, t_off, Nseg):
    s = []
    time_grid = np.linspace(t_on,t_off, grid_num)
    for i in range(grid_num):
        t_start = time_grid[i]
        s.append(poissonTrain(rate, t_start, t_off))
    S = []
    for i in range(Nseg):
        S = S + s
    return S

def gridSpikes_regular(rate, grid_num, t_on, t_off, Nseg):
    s = []
    time_grid = np.linspace(t_on,t_off, grid_num)
    for i in range(grid_num):
        t_start = time_grid[i]
        interval = 1e3/rate
        s.append(np.arange(t_start, t_off+1, interval))
    S = []
    for i in range(Nseg):
        S = S + s
    return S

def poissonSpikes_ori(rate, t_on, t_off, N):
    interval = np.random.exponential(1e3/rate, (N, int(t_off-t_on)))
    train = np.cumsum(interval, axis = 1)
    S = []
    for i in range(N):
        tmp = train[i] + t_on
        S.append(tmp[tmp <= t_off])
    return S


def poissonSpikes(rate_0, rate_1, t_on, t_off, N):
    interval = np.random.exponential(1e3/rate_0, (N, int(t_on)))
    train = np.cumsum(interval, axis = 1)
    S = []
    for i in range(N):
        tmp = train[i]
        S.append(tmp[tmp <= t_on])

    interval = np.random.exponential(1e3/rate_1, (N, int(t_off-t_on)))
    train = np.cumsum(interval, axis = 1)
    for i in range(N):
        tmp = train[i] + t_on
        S[i] = np.concatenate((S[i], tmp[tmp <= t_off]))
    return S


def regularSpikes(rate, t_on, t_off, N):
    num_spikes = int((t_off - t_on) * rate * 1e-3)
    S=[np.linspace(t_on,t_off, num_spikes) for i in range(N)]
    return S

def gridSpikes(gird_num, t_on, t_off, Nseg):
    s = []
    time_grid = np.linspace(t_on,t_off, gird_num)
    for i in range(gird_num):
        s.append(np.array([time_grid[i]]))

    S = []
    for i in range(Nseg):
        S = S + s

    return S

def gridSpikes_all(gird_num, t_on, t_off, Nseg, interval, Num):
    s = []
    time_grid = np.linspace(t_on,t_off, gird_num)
    for i in range(gird_num):
        s.append(time_grid[i] + interval * np.arange(Num))

    S = []
    for i in range(Nseg):
        S = S + s
    return S

def syncSpikes_single(t_start_list, dur, n_sync):
    s = []
    #if t_start_list is scalar, convert it to list
    if isinstance(t_start_list, (int, float)):
        t_start_list = [t_start_list]

    for t_start in t_start_list:
        s += list(np.linspace(t_start, t_start + dur, n_sync))

    s = np.array(s)
    # sort the spikes in time
    s = np.sort(s)

    return s

def syncSpikes(N_syn, t_on, t_off, rate_sync, dur, n_sync):
    S = []
    for i in range(N_syn):
        t_start_list = poissonTrain_ori(rate_sync, t_on, t_off)
        #t_start_list = [t_on + 50]
        s = syncSpikes_single(t_start_list, dur, n_sync)
        S.append(s)

    return S

import itertools

def generate_subsets(N, n):
    nums = np.arange(N)
    subsets = list(itertools.combinations_with_replacement(nums, n))
    subsets = np.array(subsets)
    return subsets

def combinations(nums, n):
    subsets = list(itertools.combinations_with_replacement(nums, n))
    subsets = np.array(subsets)
    return subsets