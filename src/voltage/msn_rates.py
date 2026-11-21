"""Independent NumPy translations of the fixed MSN voltage/Ca-dependent gates."""
import numpy as np
E=np.exp
def rates(name,v,c,p,temp,dt):
    q=p.get('q',1.)
    if name=='naf_kai':
        mi=1/(1+E((v-p['mVhalf'])/p['mSlope']));hi=1/(1+E((v-p['hVhalf'])/p['hSlope']))
        mt=.38+1/(.6*E((v+58)/8)+1.8*E((v+58)/-35))
        ht=np.where(v<-60,3.4+.015*v,.56+1.1/(1+E((v+48)/15))+1.2/(1+E((v+48)/4)))
    elif name in ['kaf_kai','kas_kai']:
        if name=='kaf_kai':
            a=1.5/(1+E((v-4)/-17));b=.6/(1+E((v-10)/9))
            ah=.105/(1+E((v+121)/22));bh=.065/(1+E((v+55)/-11))
            hi=ah/(ah+bh)
        else:
            a=.25/(1+E((v-50)/-20));b=.05/(1+E((v+90)/35))
            ah=.0025/(1+E((v+95)/16));bh=.002/(1+E((v-50)/-70))
            hi=p['a']+ah/(ah+bh)*(1-p['a'])
        mi=a/(a+b);mt=1/(a+b);ht=1/(ah+bh)
    elif name=='kdr_kai':
        a=E((v+13)/-9.09);b=E((v+13)/-12.5);return [(1/(a+1),50*b/(a+1)/q)]
    elif name=='kir_kai':
        a=.1*E((v+60)/-14);b=.27/(1+E((v+31)/-23));return [(1/(1+E((v+102)/13)),1/(a+b)/q)]
    elif name=='bk':
        z=1e-3*2*p['FARADAY']/(p['R']*(temp+273.15))
        a=.48*c/(c+p['k1']*E(-.84*z*v));b=.28/(1+c/(p['k4']*E(-z*v)))
        return [(a/(a+b),1/(a+b))]
    elif name=='sk_kai':
        a=(c/.57e-3)**5.2;return [(a/(1+a),4.9/q)]
    elif name in ['cal12','cal13']:
        mi=1/(1+E((v+(8.9 if name=='cal12' else 33))/-6.7));mt=.06+1/(E((v-10)/20)+E((v+17)/-48))
        hi=1/(1+E((v+13.4)/11.9));ht=44.3
    elif name=='can_kai':
        mi=1/(1+E((v+3)/-8));mt=(.06+1/(E((v-25)/18)+E((v+31)/-44)))*2
        hi=1/(1+E((v+74.8)/6.5));ht=70
    elif name=='caq_kai':return [(1/(1+E((v+16.3)/-7.9)),2.26/q)]
    elif name=='car_kai':
        mi=1/(1+E((v+29)/-9.6));mt=15.3;hi=1/(1+E((v+33.3)/17));ht=22+80/(1+E((v+19)/5))
    elif name=='cat32':
        mi=1/(1+E((v+54)/-7.8));mt=(2.1+23/(1+E((v+61)/6)))*3;hi=1/(1+E((v+64.2)/8.8));ht=30+280/(1+E((v+52)/7))
    elif name=='cat33':
        mi=1/(1+E((v+81)/-5.8));mt=(2.3+20/(1+E((v+60)/9)))*3;hi=1/(1+E((v+78.3)/6.5));ht=125+140/(1+E((v+60)/3))
    else:raise ValueError(name)
    return [(mi,mt/q),(hi,ht/q)]
