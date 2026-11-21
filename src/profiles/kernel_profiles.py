"""Versioned, state-conditioned inhibitory profiles; no legacy result writes."""
from pathlib import Path
import ctypes
import time
import numpy as np


def ptr(a):
    return ctypes.c_void_p(a.ctypes.data)


class Resident:
    def __init__(self, c, library=None):
        self.c = c
        self.lib = ctypes.CDLL(str(library or Path(__file__).with_name('libprofiles.so')))
        self.lib.kp_create.restype = ctypes.c_void_p
        self.lib.kp_destroy.argtypes = [ctypes.c_void_p]
        self.lib.kp_forward.restype = self.lib.kp_reverse.restype = ctypes.c_int
        keys = ['parent', 'order', 'segments', 'lower', 'upper', 'vcoef', 'gcoef',
                'diag', 'factor', 'vnext', 'wave', 'D1']
        arrays = [np.ascontiguousarray(c[k], np.int32 if k in keys[:3] else np.float64) for k in keys]
        tic = time.perf_counter()
        self.handle = self.lib.kp_create(*[ctypes.c_int(x) for x in
            [c['n'], c['nn'], c['K'], c['T'], c['wave'].shape[1], c['soma']]], *map(ptr, arrays))
        if not self.handle:
            raise RuntimeError('Profile GPU allocation/transfer failed; see CUDA diagnostic')
        self.upload_seconds = time.perf_counter() - tic

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if self.handle:
            self.lib.kp_destroy(self.handle)
            self.handle = None

    def forward(self, loc, oi, record=False):
        if not self.handle:
            raise RuntimeError('Closed device context')
        loc = np.asarray(loc, dtype=np.int32)
        oi = np.asarray(oi, dtype=np.int32)
        assert loc.ndim == oi.ndim == 1 and len(loc) == len(oi) > 0
        assert np.all((loc >= 0) & (loc < self.c['n']))
        assert np.all((oi >= 0) & (oi < self.c['wave'].shape[1]))
        order = np.argsort(oi, kind='stable'); inverse = np.argsort(order)
        loc = np.ascontiguousarray(loc[order]); oi = np.ascontiguousarray(oi[order])
        score = np.zeros(len(loc)); trace = np.zeros((self.c['T']+1, len(loc))) if record else np.zeros(1)
        tic = time.perf_counter()
        ret = self.lib.kp_forward(ctypes.c_void_p(self.handle), ctypes.c_int(len(loc)),
                                 ptr(loc), ptr(oi), ctypes.c_int(record), ptr(score), ptr(trace))
        if ret:
            raise RuntimeError('Profile forward failed')
        return {'min': score[inverse], 'trace': trace[:, inverse] if record else None,
                'seconds': time.perf_counter()-tic}

    def adjoint(self, loc, targets):
        if not self.handle:
            raise RuntimeError('Closed device context')
        loc = np.ascontiguousarray(loc, np.int32); targets = np.asarray(targets, np.int32)
        assert loc.ndim == targets.ndim == 1 and len(loc) == len(targets) > 0
        assert np.all((loc >= 0) & (loc < self.c['n']))
        assert np.all((targets >= 0) & (targets <= self.c['T']))
        unique, group = np.unique(targets, return_inverse=True)
        members = np.ascontiguousarray(np.argsort(group, kind='stable'), np.int32)
        offsets = np.ascontiguousarray(np.r_[0, np.cumsum(np.bincount(group))], np.int32)
        unique = np.ascontiguousarray(unique, np.int32)
        output = np.zeros((self.c['T'], len(loc)))
        tic = time.perf_counter()
        ret = self.lib.kp_reverse(ctypes.c_void_p(self.handle), ctypes.c_int(len(unique)),
            ctypes.c_int(len(loc)), ptr(unique), ptr(offsets), ptr(members), ptr(loc), ptr(output))
        if ret:
            raise RuntimeError('Profile adjoint failed')
        return output, {'seconds': time.perf_counter()-tic, 'source_columns': len(loc),
                        'unique_output_times': len(unique)}


def make_profiles(m, c, dest, identity, locations=None, batch_size=2048):
    """Baseline step 0 only. Each location chooses its own onset and target.

    K[j,x] = d V_s[target_x] / d I_out[j,x], mV/nA, including dt.
    a_syn[j,x] = g_unit[j,onset_x] * (V_next[j,x]-E_inh), nA/uS.
    dV_s/dw = sum_j K[j,x] * a_syn[j,x], mV/uS.
    Continuous-density phi is K/dt, not K. Saved current is outward-positive.
    """
    assert m.meta['variant'] == 'pulse' and not m.active_syn, 'Profiles require the unmodified step-0 baseline'
    assert m.probe_count == 31 and np.all(m.w_i == 0)
    dt = float(__import__('neuron').h.dt)
    reversal = float(m.synlist_i[0].e)
    assert reversal == -70 and all(float(s.e) == reversal for s in m.synlist_i)
    loc = np.arange(m.Nseg_no_axon, dtype=np.int32) if locations is None else np.asarray(locations, np.int32)
    assert len(np.unique(loc)) == len(loc) and np.all((loc >= 0) & (loc < m.Nseg_no_axon))
    tic = time.perf_counter()
    with Resident(c) as device:
        all_loc = np.repeat(loc, 31); all_oi = np.tile(np.arange(31, dtype=np.int32), len(loc))
        score = np.concatenate([device.forward(all_loc[i:i+batch_size], all_oi[i:i+batch_size])['min']
                                for i in range(0, len(all_loc), batch_size)]).reshape(len(loc), 31)
        onset = np.argmin(score, axis=1).astype(np.int32)
        forward = device.forward(loc, onset, True)
        trace = forward['trace']; target = np.argmin(trace, axis=0).astype(np.int32)
        kernel, timing = device.adjoint(loc, target)
        outward = c['wave'][:, onset] * (c['vnext'][:, loc] - reversal)
        reconstructed = np.sum(kernel * outward, axis=0)
        direct = trace[target, np.arange(len(loc))]
        error = float(np.max(abs(reconstructed-direct)/np.maximum(1., abs(direct))))
        assert np.all(np.isfinite(kernel)) and np.all(np.isfinite(trace)) and error < 1e-8
        assert np.allclose(direct, score[np.arange(len(loc)), onset], rtol=1e-10, atol=1e-10)
        assert all(np.all(kernel[int(t):, j] == 0) for j, t in enumerate(target)), 'Noncausal kernel'
        dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
        path = dest/'profiles_v2.npz'
        if path.exists():
            raise FileExistsError('Do not overwrite a profile artifact')
        np.savez_compressed(path, schema_version=2, source_locations=loc,
            candidate_ids=loc*31+onset, onset_indices=onset, onset_grid_ms=m.meta['grid'],
            target_steps=target, target_ms=target*dt, voltage_time_ms=np.arange(c['T']+1)*dt,
            source_step_start_ms=np.arange(c['T'])*dt, source_step_end_ms=np.arange(1,c['T']+1)*dt,
            kernel_discrete_mV_per_nA=kernel, phi_density_mV_per_nA_ms=kernel/dt,
            a_syn_outward_nA_per_uS=outward, legacy_didp_syn_mV_per_uS=-outward*c['D1'][loc],
            dV_s_dw_mV_per_uS=trace, dV_s_linear_1nS_mV=trace*.001,
            selected_min_mV_per_uS=direct, scores_mV_per_uS=score,
            distance_um=m.distance[loc], D1_mV_per_nA=c['D1'][loc], dt_ms=dt,
            E_inh_mV=reversal, baseline_voltage_mV=c['trace'],
            segment_names=np.asarray([str(m.seglist[i]) for i in loc]),
            baseline_local_voltage_mV=c['vnext'][:,loc],
            zero_weight_waveforms=c['wave'][:,:31])
        import json
        report = {'identity':identity,'state_conditioned':True,'baseline_step':0,'segments':len(loc),
                  'temperature_C':m.meta['temperature'],'v_r_mV':float(m.v_r),'hh_table':False,
                  'duality_normalized_max_error':error,'upload_seconds':device.upload_seconds,
                  'adjoint':timing,'total_profile_seconds':time.perf_counter()-tic,
                  'note':'1 nS voltage change is linear prediction, not finite NEURON difference.'}
        (dest/'profiles_v2.json').write_text(json.dumps(report,indent=2))
        return report
