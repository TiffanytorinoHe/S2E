"""Independent cnexp synaptic state recurrence driven only by saved input events."""
import numpy as np

def independent_waves(m,T,dt,n):
    impulses=np.zeros((T,3,n))
    taus=[None,None,None]
    for syn,nc,events in zip(m.synlist_back,m.nclist_back,m.events):
        idx=m.seg2idx[syn.get_segment()]
        amplitude=float(nc.weight[0]);weight=float(syn.w)
        if hasattr(syn,'AMPA_tau1'):
            aw=.5 if amplitude>1 else amplitude
            nw=0. if amplitude>1 else 1-amplitude
            parts=[(0,syn.AMPA_tau1,syn.AMPA_tau2,aw),(1,syn.NMDA_tau1,syn.NMDA_tau2,nw)]
        else:parts=[(2,syn.tau1,syn.tau2,amplitude)]
        bins=np.floor((np.asarray(events)+float(nc.delay))/dt+.5).astype(int)
        bins=bins[(bins>=0)&(bins<T)]
        for c,t1,t2,w in parts:
            if taus[c] is not None:assert taus[c]==(t1,t2)
            taus[c]=(t1,t2)
            tp=t1*t2/(t2-t1)*np.log(t2/t1)
            factor=1/(np.exp(-tp/t2)-np.exp(-tp/t1))
            np.add.at(impulses[:,c,idx],bins,weight*w*factor)
    a=np.zeros((3,n));b=np.zeros((3,n));waves=np.zeros_like(impulses)
    da=np.array([np.exp(-dt/t[0]) if t else 0 for t in taus])[:,None]
    db=np.array([np.exp(-dt/t[1]) if t else 0 for t in taus])[:,None]
    for k in range(T):
        a+=impulses[k];b+=impulses[k]
        waves[k]=b-a
        a*=da;b*=db
    return waves
