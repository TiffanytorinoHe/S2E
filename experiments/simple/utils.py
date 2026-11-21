import numpy as np

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