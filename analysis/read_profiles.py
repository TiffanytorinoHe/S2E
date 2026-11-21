"""Explicit sign/unit adapter for downstream figures; never guess legacy layout."""
from pathlib import Path
import json
import numpy as np


def read_profiles(path, expected_identity=None):
    path=Path(path)
    report=json.loads(path.with_suffix('.json').read_text())
    if expected_identity is not None and report['identity']!=expected_identity:
        raise ValueError('Profile input/scientific identity mismatch')
    with np.load(path,allow_pickle=False) as z:
        if int(z['schema_version'])!=2:raise ValueError('Unsupported profile schema')
        dt=float(z['dt_ms']);K=z['kernel_discrete_mV_per_nA'];a=z['a_syn_outward_nA_per_uS']
        dv=z['dV_s_dw_mV_per_uS'];target=z['target_steps'];loc=z['source_locations']
        distance=z['distance_um'];end=z['source_step_end_ms'];vt=z['voltage_time_ms']
        if not(dt>0 and np.isfinite(dt)):raise ValueError('Invalid time step')
        if K.shape!=a.shape or dv.shape!=(K.shape[0]+1,K.shape[1]):raise ValueError('Misaligned source/response arrays')
        if len(loc)!=K.shape[1] or len(np.unique(loc))!=len(loc) or distance.shape!=loc.shape:raise ValueError('Geometry mismatch')
        if not(np.all(np.isfinite(K)) and np.all(np.isfinite(a)) and np.all(np.isfinite(dv))):raise ValueError('Non-finite profiles')
        if not(np.allclose(end,np.arange(1,len(K)+1)*dt) and np.allclose(vt,np.arange(len(dv))*dt)):raise ValueError('Time-axis mismatch')
        if target.shape!=loc.shape or np.any((target<0)|(target>len(K))):raise ValueError('Invalid response targets')
        if any(np.any(K[int(t):,j]!=0) for j,t in enumerate(target)):raise ValueError('Noncausal kernel')
        direct=dv[target,np.arange(len(loc))];reconstructed=np.sum(K*a,axis=0)
        err=float(np.max(abs(reconstructed-direct)/np.maximum(1.,abs(direct))))
        if err>=1e-8:raise ValueError('Kernel/current/response duality mismatch')
        if 'phi_density_mV_per_nA_ms' in z and not np.allclose(z['phi_density_mV_per_nA_ms'],K/dt,rtol=1e-12,atol=1e-12):
            raise ValueError('Stored continuous kernel has inconsistent dt scaling')
        if 'dV_s_linear_1nS_mV' in z and not np.allclose(z['dV_s_linear_1nS_mV'],dv*.001,rtol=1e-12,atol=1e-12):
            raise ValueError('Stored finite-weight linear prediction has inconsistent units')
        if 'legacy_didp_syn_mV_per_uS' in z and not np.allclose(z['legacy_didp_syn_mV_per_uS'],-a*z['D1_mV_per_nA'],rtol=1e-12,atol=1e-12):
            raise ValueError('Stored legacy RHS adapter is inconsistent with a_syn')
        # Paper convention: positive transfer for a positive outward inhibitory
        # drive. Negative responses remain signed until the chosen plot summary.
        return dict(identity=report['identity'],dt_ms=dt,source_end_ms=end,voltage_time_ms=vt,
            source_locations=loc,distance_um=distance,target_steps=target,
            phi_discrete=-K,phi_density=-K/dt,a_syn_outward=a,
            a_syn_inward=-a,dV_s_dw=dv,inhibitory_effect=-direct,
            reconstructed_response=reconstructed,duality_error=err)
