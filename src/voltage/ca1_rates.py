"""Independent translations of CA1_20212 state equations; no NEURON calls."""
import numpy as np
E=np.exp
def tab(v,fun,lo=-150.,hi=150.,N=200):
    # Same fixed grid and linear interpolation as the original NMODL TABLE.
    grid=lo+np.arange(N+1)*(hi-lo)/N
    y=fun(grid);x=(v-lo)*(N/(hi-lo));a=np.clip(np.floor(x),0,N-1).astype(int)
    r=y[a]+(x-a)*(y[a+1]-y[a]);return np.where(x<=0,y[0],np.where(x>=N,y[-1],r))
def rates(name,v,c,p,temp,dt):
    if name in ['hha_old','hha2']:
        old=name=='hha_old';sm=40 if old else 44;sh=45 if old else 49;sn=42 if old else 46.3
        z=1e-3*p['zetar']*(v-p['vhalfr'])*9.648e4/(8.315*(273.16+temp))
        si=(1+p['ar2']*E((v-p['vhalfr'])/p['vvs']))/(1+E((v-p['vhalfr'])/p['vvs']))
        st=np.maximum(E(p['gmr']*z)/(p['a0r']+p['b0r']*E(z)),p['taumin'])
        return [(1/(1+E(-(v+sm)/3)),.05),(1/(1+E((v+sh)/(3 if old else 3.5))),.5 if old else 1.),(si,st),(1/(1+E(-(v+sn)/(2 if old else 3))),2.2 if old else 3.5)]
    if name=='h':return [(1-1/(1+E((p['vhalf']-v)/p['K'])),np.where(v>-30,1,2*(1/(E((v+145)/-17.5)+E((v+16.8)/16.5))+5)))]
    if name=='nap_poirazi':return [(1/(1+E(p['vhalf']-v)/p['K']),0.)]
    if name in ['kad_poirazi','kap_poirazi']:
        z=(p['zetan']+p['pw']/(1+E((v-p['tq'])/p['qq'])))*1e-3*(v-p['vhalfn'])*9.648e4/(8.315*(273.16+temp))
        a=E(z);b=E(z*p['gmn']);al=E(1e-3*p['zetal']*(v-p['vhalfl'])*9.648e4/(8.315*(273.16+temp)))
        return [(1/(1+a),np.maximum(b/(p['q10']**((temp-24)/10)*p['a0n']*(1+a)),p['nmin'])),(1/(1+al),np.maximum(.26*(v+50),p['lmin']))]
    if name in ['calH','car','somacar']:
        shifts={'calH':(37,1,41,.5,3.6,29),'car':(48.5,3,53,1,50,5),'somacar':(60,3,62,1,100,5)}
        a,b,c0,d,mt,ht=shifts[name];return [(1/(1+E(-(v+a)/b)),mt),(1/(1+E((v+c0)/d)),ht)]
    if name=='mykca':
        z=-2*p['FARADAY']*v/(p['R']*(273.15+temp))
        a=p['abar']/(1+p['k1']*E(p['d1']*z)/c);b=p['bbar']/(1+c/(p['k2']*E(p['d2']*z)))
        return [(a/(a+b),1/(a+b))]
    if name=='kca':
        a=(c/p['cac'])**2;return [(a/(1+a),np.maximum(1/p['beta']/(1+a)/p['tadj'],p['taumin']))]
    if name=='km_zach':
        def funcs(x):
            a=p['Ra']*(x-p['tha'])/(1-E(-(x-p['tha'])/p['qa']));b=-p['Rb']*(x-p['tha'])/(1-E((x-p['tha'])/p['qa']))
            return a/(a+b),1-E(-dt*p['tadj']*(a+b))
        if p['usetable']:
            # Parameters in this model are uniform; assert in extraction.
            inf=tab(v,lambda x:funcs(x)[0],p['vmin'],p['vmax'],199)
            fac=tab(v,lambda x:funcs(x)[1],p['vmin'],p['vmax'],199)
        else:inf,fac=funcs(v)
        return [(inf,('factor',fac))]
    if name in ['cal','cat']:
        if name=='cal':
            af=lambda x:.055*(-27.01-x)/(E((-27.01-x)/3.8)-1)
            bf=lambda x:.94*E((-63.01-x)/17)
        else:
            af=lambda x:.1967*(19.88-x)/(E((19.88-x)/10)-1)
            bf=lambda x:.046*E(-x/22.73)
        a=tab(v,af) if p['usetable'] else af(v);b=tab(v,bf) if p['usetable'] else bf(v)
        ret=[(a/(a+b),1/(p['tfa']*(a+b)))]
        if name=='cat':
            af=lambda x:1.6e-4*E(-(x+57)/19);bf=lambda x:1/(E((-x+15)/10)+1)
            a=tab(v,af) if p['usetable'] else af(v);b=tab(v,bf) if p['usetable'] else bf(v)
            ret.append((a/(a+b),1/(p['tfi']*(a+b))))
        return ret
    raise ValueError(name)
