"""Independent NumPy translations of frozen L5PC MOD state equations (no NEURON calls)."""
import numpy as np
E=np.exp
QT=2.3**1.3
def trap(x,y):
    z=x/y
    with np.errstate(invalid='ignore',divide='ignore'):
        return np.where(abs(z)<1e-6,y*(1-z/2),x/np.expm1(z))

def rates(name,v,ca,temp):
    v=v.copy()
    if name=='Ca_HVA':
        v[v==-27]+=.0001
        a=.055*(-27-v)/(E((-27-v)/3.8)-1);b=.94*E((-75-v)/17)
        ah=.000457*E((-13-v)/50);bh=.0065/(E((-v-15)/28)+1)
        return [(a/(a+b),1/(a+b)),(ah/(ah+bh),1/(ah+bh))]
    if name=='Ca_LVAst':
        v+=10
        return [(1/(1+E(-(v+30)/6)),(5+20/(1+E((v+25)/5)))/QT),
                (1/(1+E((v+80)/6.4)),(20+50/(1+E((v+40)/7)))/QT)]
    if name in ['NaTa_t','NaTs2_t','Nap_Et2']:
        mshift=32 if name=='NaTs2_t' else 38
        v[v==-mshift]+=.0001
        a=.182*(v+mshift)/(1-E(-(v+mshift)/6));b=.124*(-v-mshift)/(1-E(-(-v-mshift)/6))
        mi=a/(a+b);mt=1/(a+b)/QT
        if name=='Nap_Et2':
            mi=1/(1+E(-(v+52.6)/4.6));mt*=6
            v[v==-17]+=.0001;v[v==-64.4]+=.0001
            hi=1/(1+E((v+48.8)/10));a=-2.88e-6*(v+17)/(1-E((v+17)/4.63));b=6.94e-6*(v+64.4)/(1-E(-(v+64.4)/2.63))
            ht=1/(a+b)/QT
        else:
            hs=60 if name=='NaTs2_t' else 66;v[v==-hs]+=.0001
            a=-.015*(v+hs)/(1-E((v+hs)/6));b=-.015*(-v-hs)/(1-E((-v-hs)/6));hi=a/(a+b);ht=1/(a+b)/QT
        return [(mi,mt),(hi,ht)]
    if name=='K_Pst':
        v+=10
        return [(1/(1+E(-(v+1)/12)),np.where(v<-50,1.25+175.03*E(.026*v),1.25+13*E(-.026*v))/QT),
                (1/(1+E((v+54)/11)),(360+(1010+24*(v+55))*E(-((v+75)/48)**2))/QT)]
    if name=='K_Tst':
        v+=10
        return [(1/(1+E(-v/19)),(.34+.92*E(-((v+71)/59)**2))/QT),
                (1/(1+E((v+66)/10)),(8+49*E(-((v+73)/23)**2))/QT)]
    if name=='Ih':
        v[v==-154.9]+=.0001;a=.00643*(v+154.9)/(E((v+154.9)/11.9)-1);b=.193*E(v/33.1)
        return [(a/(a+b),1/(a+b))]
    if name=='Im':
        a=.0033*E(.1*(v+35));b=.0033*E(-.1*(v+35));return [(a/(a+b),1/(a+b)/QT)]
    if name=='SKv3_1':return [(1/(1+E(-(v-18.7)/9.7)),4/(1+E(-(v+46.56)/44.14)))]
    if name=='SK_E2':
        c=np.where(ca<1e-7,ca+1e-7,ca);return [(1/(1+(.00043/c)**4.8),np.ones_like(c))]
    if name=='hh_grad':
        q=3**((temp-6.3)/10)
        ab=[(.1*trap(-(v+40),10),4*E(-(v+65)/18)),(.07*E(-(v+65)/20),1/(E(-(v+35)/10)+1)),(.01*trap(-(v+55),10),.125*E(-(v+65)/80))]
        return [(a/(a+b),1/(q*(a+b))) for a,b in ab]
    raise ValueError(name)

def update(states,channels,v,ca,dt,temp):
    for name,p in channels.items():
        idx=p['idx']
        for k,(inf,tau) in enumerate(rates(name,v[idx],ca[idx],temp)):
            states[name][k]+=(1-E(-dt/tau))*(inf-states[name][k])

def conductances(states,channels,n,eca):
    total=np.zeros(n);ge=np.zeros(n);gca=np.zeros(n)
    for name,p in channels.items():
        idx=p['idx'];x=states[name]
        if name=='hh_grad':
            gn=p['gnabar']*x[0]**3*x[1];gk=p['gkbar']*x[2]**4
            total[idx]+=gn+gk+p['gl'];ge[idx]+=gn*p['ena']+gk*p['ek']+p['gl']*p['el'];continue
        power=2 if name in ['Ca_HVA','Ca_LVAst','K_Pst'] else 3 if name in ['NaTa_t','NaTs2_t','Nap_Et2'] else 4 if name=='K_Tst' else 1
        g=p['bar']*x[0]**power
        if len(x)==2:g*=x[1]
        rev=eca[idx] if p['ion']=='ca' else p['reversal']
        total[idx]+=g;ge[idx]+=g*rev
        if p['ion']=='ca':gca[idx]+=g
    return total,ge,gca
