"""SIMPLE environment smoke test using L5PC globals and verbatim run functions."""
from pathlib import Path
import argparse
import json
import os
import time
import hashlib
import numpy as np
import torch as tc
from neuron import h
from GradNeuron_torch_mod import GradNeuron
from CellType import SimpleCell
from utils import gridSpikes
import data_c2py
import sys

HERE = Path(__file__).resolve().parent
for module_name in ['GradNeuron_torch_mod', 'CellType', 'ChannelType', 'myfunc_torch', 'utils']:
    module_path = Path(sys.modules[module_name].__file__).resolve()
    assert module_path.parent == HERE, f'Cross-folder import prohibited: {module_path}'
parser = argparse.ArgumentParser()
parser.add_argument('--setting', choices=['soma_hh', 'all_hh'], required=True)
parser.add_argument('--dtype', choices=['float', 'double'], default='float')
parser.add_argument('--max-increments', type=int, default=30)
parser.add_argument('--mode', choices=['validate', 'solve', 'forward_reference', 'verify_saved_solution', 'verify_counts'], default='validate')
parser.add_argument('--hh-table-off', action='store_true', default=True)
parser.add_argument('--hh-table-on', dest='hh_table_off', action='store_false',
                    help='Reproduce the historical table-enabled CPU configuration')
parser.add_argument('--run-id', type=int, default=0)
parser.add_argument('--output-root', type=str)
parser.add_argument('--counts-file', type=str)
# NEURON special retains its own switches and the script path in sys.argv.
script_index = next(i for i, value in enumerate(sys.argv) if Path(value).name == Path(__file__).name)
args = parser.parse_args(sys.argv[script_index + 1:])
assert 0 <= args.run_id < 300
tc.set_num_threads(1)

# Match the global configuration structure in L5bPCmodelsEH/opt_inh_all.py.
mech_list = ['hh_grad']
h.load_file('stdrun.hoc')
h.cvode_active(0)
h.secondorder = 0
h.celsius = 6.3
if args.hh_table_off:
    h.usetable_hh_grad = 0
cell = SimpleCell()
# SIMPLE has no axon. The L5PC implementation requires this morphology list.
cell.all_no_axon = list(cell.all)
if args.setting == 'all_hh':
    for sec in cell.dend:
        sec.insert('hh_grad')
        for seg in sec:
            seg.hh_grad.gnabar = .12
            seg.hh_grad.gkbar = .036
            seg.hh_grad.gl = 0.
            seg.hh_grad.el = -65.
cell_info = {'cell': cell, 'soma': cell.soma[0](.5), 'active': True,
             'v_r': -65., 'mech_list': mech_list}
seglist = [seg for sec in cell.all for seg in sec]
seglist_no_axon = [seg for sec in cell.all_no_axon for seg in sec]
seg2idx = {seg: i for i, seg in enumerate(seglist)}
Nseg = len(seglist)
Nseg_no_axon = len(seglist_no_axon)
exc_pattern = 'nmda'  # L5PC branch name; ratio below specifies mixed AMPA/NMDA.
r_na = 1.
syn_data = {'ratio': [1., 1. / (1. + r_na)],
            'AMPA_tau1': .1, 'AMPA_tau2': 2., 'NMDA_tau1': 2., 'NMDA_tau2': 75.,
            'E_exc': 0., 'GABAa_tau1': 1., 'GABAa_tau2': 5.,
            'GABAb_tau1': 5., 'GABAb_tau2': 50., 'E_inh': -70.}
N_test = 300
N_e = 0
N_back = [0, 6, 0]
L_min = D_min = DH_min = 1e-6
L_max = D_max = DH_max = 10.
w_i_min = 0.
w_i_max = 1e3
stim_on = 20.
input_off = 60.
stim_off = 100.  # Retain the full 100 ms observation window of the SIMPLE screen.
stim_dur = input_off - stim_on
dt = .025
grad_clip = 1e2
grid_num = 5
time_grid = np.linspace(stim_on, input_off, grid_num)
kernel_time_grid = np.arange(round(stim_off / dt) + 1) * dt
inh_setting = {'if_kernel': False, 'kernel_time_grid': kernel_time_grid,
               'batch_size': 10, 'step': 0, 'if_fs': False, 'if_depress': False,
               'distance': 50, 'r_lts': 5., 'r_fs': 50., 'r_slow': 2.,
               'if_slow': False, 'section_slow': [], 'if_axon': False, 'if_noise': False}
Nseg_inh = Nseg_no_axon
inh_loc_idx = (np.arange(Nseg_inh).reshape(-1, 1) * np.ones([Nseg_inh, grid_num], dtype=int)).flatten()
N_i = len(inh_loc_idx)
ROOT_PATH = str(HERE / 'validation_l5' / args.setting / args.dtype) + os.sep
if args.mode == 'solve':
    ROOT_PATH = str(HERE / 'batch50_20260923' / args.setting / f'case_{args.run_id:03d}') + os.sep
if args.mode == 'forward_reference':
    ROOT_PATH = str(HERE / 'gpu_benchmark' / 'cpu_reference' / args.setting / f'case_{args.run_id:03d}') + os.sep
if args.mode == 'verify_saved_solution':
    ROOT_PATH = str(HERE / 'gpu_benchmark' / 'saved_solution_check' / args.setting / f'case_{args.run_id:03d}') + os.sep
if args.output_root:
    output_root=Path(args.output_root).resolve()
    assert output_root.is_relative_to(HERE), 'Output must stay inside SIMPLE'
    ROOT_PATH=str(output_root / args.setting / f'case_{args.run_id:03d}')+os.sep
REC_PATH = ROOT_PATH + 'rec/'
PIC_PATH = ROOT_PATH + 'pic/'
Path(REC_PATH).mkdir(parents=True, exist_ok=True)
Path(PIC_PATH).mkdir(parents=True, exist_ok=True)

# Read the existing bank; do not redraw or select inputs for this validation.
bank = HERE / 'results_exc6/inputs'
back_loc_all = np.load(bank / 'exc_locations.npy', allow_pickle=False)
back_times_all = np.load(bank / 'exc_times_ms.npy', allow_pickle=False)
exc_loc_all = np.empty((N_test, N_e), dtype=int)
assert back_loc_all.shape == back_times_all.shape == (N_test, 6)
S_E = [[] for _ in range(N_test)]
S_BACK = [[np.array([t]) for t in row] for row in back_times_all]
S_i = gridSpikes(grid_num, stim_on, input_off, Nseg_inh)
assert len(S_i) == N_i == 20
assert np.array_equal(np.array(S_i).reshape(-1), np.tile(time_grid, Nseg_inh))

# This file contains these four functions verbatim from L5PC:
# find_first_spike, run_single_step, find_opt_inh, simulate.
def find_first_spike(V):
    spike = False
    vmax = -65
    for i,v in enumerate(V):
        if  spike is False:
            if v >= 0:
                spike = True
                vmax = v
                
        else:
            if v > vmax:
                vmax = v
            else:
                break
    if spike is False:
        i += 1

    #print("######")
    #print(spike, '-',i)
    return spike, i


def run_single_step(Cell : GradNeuron, S_e, S_i, S_back):
    Cell.set_stim(S_e, S_i, S_back, [[0,0,0]])
    h.t = 0
    h.tstop = stim_off 
    h.dt = dt

    t = h.Vector()
    t.record(h._ref_t)

    h.finitialize(Cell.v_r)
    h.fcurrent()

    tstep = 0
    
    Cell._setup_matrix()

    while h.t < h.tstop:
        #print(f't: {h.t:f}', end='\r')
        h.fadvance()
        tstep += 1
        Cell.update_dvdp()


def find_opt_inh(Cell : GradNeuron, S_e, S_i, S_back, spike_cnt, p_idx_list=[]):
    Cell.set_stim(S_e, S_i, S_back, [[0,0,0]])
    if Cell.record == False and spike_cnt == 0:
        h.finitialize(Cell.v_r)
        h.continuerun(stim_off)
        V1 = np.array(Cell.soma_v) * 1
        spike, stime = find_first_spike(V1)
        if spike is False:
            return False, -1, -1

    h.t = 0
    h.tstop = stim_off 
    h.dt = dt

    t = h.Vector()
    t.record(h._ref_t)

    h.finitialize(Cell.v_r)
    h.fcurrent()

    tstep = 0
    
    Cell._setup_matrix()
    idx_real_inh = Cell.rec['syn']['id'].to('cpu').numpy() - Cell.N_e
    #calc the idx 
    if Cell.record == True:
        p_idx_hines_list = []
        for p_idx in p_idx_list:
            p_idx_hines = np.where(idx_real_inh == p_idx)[0][0]
            p_idx_hines_list.append(p_idx_hines)

        p_idx_hines_list = tc.tensor(p_idx_hines_list, device = 'cuda', dtype = tc.int)
    
    while h.t < h.tstop:
        #if Cell.record == True:     
        #    print(f't: {h.t:f}', end='\r')
        #print(f't: {h.t:f}', end='\r')
        h.fadvance()
        tstep += 1
        #time_start = time.time()
        Cell.update_dvdp()
        #time_end = time.time()
        #print('update dvdp time: ', time_end - time_start)
        #print(Cell.rec_time[-1])
        if Cell.record == True and spike_cnt == inh_setting['step']:
            didp_syn_rec[tstep-1] = tc.diag(Cell.didp_syn[:, p_idx_hines_list])
            dvdp_soma_rec[tstep-1] = Cell.dvdp_soma[-1, p_idx_hines_list]
        #print('')

    """
    rec_time_all = np.array(Cell.rec_time)
    print(rec_time_all.shape)
    print(np.mean(rec_time_all[1:, 0]), np.mean(rec_time_all[1:, 1]), np.mean(rec_time_all[1:, 2]))
    print(np.mean(rec_time_all[1:, 3]), np.mean(rec_time_all[1:, 4]), np.mean(rec_time_all[1:, 5]))
    """
    V1 = np.array(Cell.soma_v) * 1
    
    spike, stime = find_first_spike(V1)
    inh_idx_real = -1
    if spike:
        dvdp_soma_inh = Cell.dvdp_soma[:, -N_i:].to('cpu').numpy()
        dvdp_soma_min_inh = np.min(dvdp_soma_inh, axis = 0)
        inh_idx_max = np.argmin(dvdp_soma_min_inh) + N_e
        
        inh_idx_real =Cell.rec['syn']['id'].to('cpu').numpy()[inh_idx_max] - N_e

    return spike, stime, inh_idx_real


def simulate(w_i, S_e, S_i, S_back):
    Cell.set_weights(w_i)
    Cell.set_stim(S_e, S_i, S_back, [[0,0,0]])

    h.t = 0
    h.tstop = stim_off 
    h.dt = dt

    t = h.Vector().record(h._ref_t)

    
    h.finitialize(Cell.v_r)
    h.fcurrent()
    h.continuerun(stim_off)

    return t, np.array(Cell.soma_v)



def write_json(name, data):
    (Path(REC_PATH) / name).write_text(json.dumps(data, indent=2), encoding='utf-8')

def check_bridge():
    errors = {}
    for mech, syns, fields in [
        ('Exp2Syn_Inhibitory', Cell.synlist_i, ['i','g','pure_i','didv','djdv','w','id']),
        ('Exp2Syn_Excitatory_back', Cell.synlist_back, ['i','g_tot','pure_i','didv','djdv','w','id'])]:
        raw = np.asarray(data_c2py.mech_data_rec(mech)).reshape(-1, 7)
        expected = np.array([[getattr(s, f) for f in fields] for s in syns])
        raw = np.array(sorted(raw.tolist()))
        expected = np.array(sorted(expected.tolist()))
        np.testing.assert_allclose(raw, expected, rtol=1e-12, atol=1e-12)
        errors[mech] = float(np.max(np.abs(raw - expected)))
    raw = np.asarray(data_c2py.mech_data_rec('hh_grad')).reshape(-1, 15)
    indices = data_c2py.get_ml_nodeindices('hh_grad')
    segs = {int(h.fmatrix(s.x, 0, sec=s.sec)): s for s in seglist if h.ismembrane('hh_grad',sec=s.sec)}
    fields = ['ina','ik','gna','gk','dgdm','dgdh','dgdn','dmdv','dhdv','dndv','dmdm_','dhdh_','dndn_']
    expected = np.array([[getattr(segs[int(i)].hh_grad, f) for f in fields] +
                         [segs[int(i)].ena, segs[int(i)].ek] for i in indices])
    np.testing.assert_allclose(raw, expected, rtol=1e-12, atol=1e-12)
    errors['hh_grad'] = float(np.max(np.abs(raw - expected)))
    return errors

def validate_gradient(S_e, S_i, S_back):
    Cell.set_weights(np.zeros(N_i))
    run_single_step(Cell, S_e, S_i, S_back)
    baseline = np.array(Cell.soma_v)
    internal = Cell.dvdp_soma.detach().cpu().numpy()
    order = Cell.rec['syn']['id'].detach().cpu().numpy().astype(int)
    grad = np.empty_like(internal)
    grad[:, order] = internal
    assert grad.shape == (len(baseline), N_i)
    assert np.all(np.isfinite(grad)) and np.max(np.abs(grad)) > 0
    bridge_errors = check_bridge()
    _, plain = simulate(np.zeros(N_i), S_e, S_i, S_back)
    np.testing.assert_allclose(baseline, plain, rtol=0, atol=1e-10)
    chosen = int(np.argmin(grad.min(axis=0)))
    checks = []
    for candidate in sorted(set([0, chosen, N_i-1])):
        for eps in [1e-8, 1e-9]:
            w = np.zeros(N_i)
            w[candidate] = eps
            _, perturbed = simulate(w, S_e, S_i, S_back)
            numeric = (perturbed - plain) / eps
            error = float(np.linalg.norm(numeric - grad[:,candidate]) / max(np.linalg.norm(numeric),1e-12))
            checks.append({'candidate':candidate, 'epsilon_us':eps, 'relative_l2_error':error})
    report = {'setting':args.setting, 'dtype':args.dtype, 'run_id':0, 'n_exc':6,
              'bridge_max_abs_errors':bridge_errors, 'finite_difference_checks':checks,
              'gradient_shape':list(grad.shape), 'baseline_peak_mV':float(baseline.max()),
              'gradient_forward_max_difference_mV':float(np.max(np.abs(baseline-plain))),
              'finite_difference_passed':all(c['relative_l2_error'] < .02 for c in checks),
              'neuron_version':h.nrnversion(), 'data_c2py_path':data_c2py.__file__,
              'launch':'./x86_64/special -nobanner -nogui -python opt_inh_all.py',
              'gradient_source_sha256':hashlib.sha256((HERE/'GradNeuron_torch_mod.py').read_bytes()).hexdigest()}
    write_json('gradient_validation.json', report)
    np.savez_compressed(Path(REC_PATH)/'gradient_0.npz', soma_mV=baseline, gradient=grad)
    print(json.dumps(report), flush=True)
    assert report['finite_difference_passed'], 'Finite-difference gradient validation failed'

def run(run_id, S_e, S_i, S_back):
    # Same reset, gradient selection, id mapping and weight increment as L5PC run().
    # Each process constructs one fresh cell, so cases share no simulation state.
    Cell.set_weights(np.zeros(N_i))
    _, initial = simulate(np.zeros(N_i), S_e, S_i, S_back)
    spike, stime = find_first_spike(initial)
    baseline_spiked = bool(spike)
    ans = []
    history = []
    started = time.monotonic()
    while spike:
        spike, stime, inh_idx = find_opt_inh(Cell, S_e, S_i, S_back, len(ans), [])
        voltage = np.array(Cell.soma_v)
        assert np.all(np.isfinite(voltage))
        assert bool(tc.all(tc.isfinite(Cell.dvdp_soma))), 'Non-finite gradient'
        history.append({'iteration':len(ans),'max_soma_mV':float(voltage.max()), 'candidate':int(inh_idx)})
        if not spike or len(ans) >= args.max_increments:
            break
        assert 0 <= inh_idx < N_i
        ans.append(int(inh_idx))
        Cell.w_i[inh_idx] += .001
        Cell.synlist_i[inh_idx].w += .001
        np.save(Path(REC_PATH)/f'ans_{run_id}.npy', np.array(ans,dtype=int))
    final_w = np.bincount(ans, minlength=N_i) * .001
    _, final = simulate(final_w, S_e, S_i, S_back)
    final_spike, _ = find_first_spike(final)
    assert bool(final_spike) == bool(spike), 'Independent forward verification disagrees'
    np.save(Path(REC_PATH)/f'ans_{run_id}.npy', np.array(ans,dtype=int))
    report = {'setting':args.setting,'run_id':run_id,'dtype':args.dtype,'hh_table':bool(h.usetable_hh_grad),
              'status':('no_baseline_spike' if not baseline_spiked else
                        'solved' if not final_spike else 'increment_cap_reached'),
              'ans':ans,'ans_length':len(ans), 'baseline_peak_mV':float(initial.max()),
              'final_peak_mV':float(final.max()),'elapsed_s':time.monotonic()-started,
              'baseline_spiked':baseline_spiked, 'final_spiked':bool(final_spike),
              'baseline_spike_count':int(np.sum((initial[:-1]<0)&(initial[1:]>=0))),
              'ans_unique_candidates':len(set(ans)), 'max_increments':args.max_increments,
              'exc_segment_ids':back_loc_all[run_id].tolist(),
              'exc_times_ms':back_times_all[run_id].tolist(),
              'entry_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'independent_forward_verified':True}
    write_json('solve_smoke.json' if args.mode == 'validate' else 'result.json', report)
    write_json('iterations.json', history)
    np.savez_compressed(Path(REC_PATH)/f'voltage_{run_id}.npz', baseline=initial, final=final)
    print(json.dumps(report),flush=True)
    assert report['status'] != 'increment_cap_reached', 'Solve reached increment cap'

if __name__ == '__main__':
    h.dt = dt
    h.steps_per_ms = 1. / dt
    w_e = .0004 * np.ones(N_e)
    w_i = np.zeros(N_i)
    w_back = np.concatenate([.001*np.ones(n) for n in N_back])
    config = {'record':False, 'spiny':False, 'N_e':N_e, 'N_i':N_i, 'N_back':N_back,
              'grid_num':grid_num, 'syn_data':syn_data, 'inh_setting':inh_setting,
              'dtype':args.dtype, 'device':'cpu', 'L_min':L_min, 'L_max':L_max,
              'D_min':D_min, 'D_max':D_max, 'DH_min':DH_min, 'DH_max':DH_max,
              'w_i_min':w_i_min, 'w_i_max':w_i_max, 'G_max':grad_clip,
              'exc_loc_idx':exc_loc_all[args.run_id], 'inh_loc_idx':inh_loc_idx,
              'back_loc_idx':back_loc_all[args.run_id], 'w_e':w_e, 'w_i':w_i, 'w_back':w_back,
              'cell_info':cell_info}
    Cell = GradNeuron(config, REC_PATH)
    assert Nseg == len(Cell.seglist)
    assert all(int(nc.wcnt()) == 1 and nc.weight[0] == .5 for nc in Cell.nclist_back)
    if args.mode == 'verify_counts':
        assert h.usetable_hh_grad == 0
        counts_path=Path(args.counts_file).resolve()
        assert counts_path.is_relative_to(HERE)
        counts=np.load(counts_path,allow_pickle=False)
        assert counts.ndim==2 and counts.shape[1]==20 and np.all(counts>=0) and np.all(counts==np.floor(counts))
        voltages=np.array([simulate(c*.001,S_E[args.run_id],S_i,S_BACK[args.run_id])[1] for c in counts]).T
        assert np.all(np.isfinite(voltages))
        np.savez_compressed(Path(REC_PATH)/'forward_counts.npz',counts=counts,voltage=voltages,case_id=args.run_id,hh_table=False)
        print('VERIFIED_COUNTS',args.run_id,len(counts),flush=True)
        sys.exit(0)
    if args.mode in ['forward_reference','verify_saved_solution']:
        assert h.usetable_hh_grad == 0
        ans = np.load(HERE/'batch50_20260923'/args.setting/f'case_{args.run_id:03d}/rec'/f'ans_{args.run_id}.npy')
        rng = np.random.default_rng(20260923)
        counts = [np.zeros(20,dtype=int), np.bincount(ans,minlength=20)]
        if args.mode == 'forward_reference':
            counts += [np.bincount(ans[:k],minlength=20) for k in range(1,len(ans))]
            counts += [np.bincount(rng.integers(0,20,size=k),minlength=20) for k in range(1,7) for _ in range(10)]
        started = time.perf_counter()
        voltages = np.array([simulate(c*.001,S_E[args.run_id],S_i,S_BACK[args.run_id])[1] for c in counts]).T
        if args.mode == 'forward_reference':
            np.savez_compressed(Path(REC_PATH)/'reference.npz', counts=np.array(counts), voltage=voltages)
        write_json('reference.json',{'hh_table':False,'count':len(counts),'elapsed_s':time.perf_counter()-started,'peaks':voltages.max(axis=0).tolist(),'setting':args.setting,'case':args.run_id,'old_ans_length':len(ans),'mode':args.mode})
        print('CPU_REFERENCE_COMPLETE',args.setting,args.run_id,len(counts),flush=True)
        sys.exit(0)
    if args.mode == 'validate':
        validate_gradient(S_E[args.run_id], S_i, S_BACK[args.run_id])
    run(args.run_id, S_E[args.run_id], S_i, S_BACK[args.run_id])
    print('SIMPLE_CASE_COMPLETE', args.setting, args.run_id, flush=True)
