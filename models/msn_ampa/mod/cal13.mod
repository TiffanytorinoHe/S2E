TITLE LVA L-type calcium current (Cav1.3)

UNITS {
    (mV) = (millivolt)
    (mA) = (milliamp)
    (S) = (siemens)
    (molar) = (1/liter)
    (mM) = (millimolar)
    FARADAY = (faraday) (coulomb)
    R = (k-mole) (joule/degC)
}

NEURON {
    THREADSAFE
    SUFFIX cal13
    USEION cal READ cali, calo WRITE ical VALENCE 2
    RANGE pbar, ical, base, factor
    POINTER pka
    RANGE dv
    RANGE didv, didc, didm, didh, djdv, djdc, djdm, djdh
    RANGE dmdv, dmdm_, dhdv, dhdh_
}

PARAMETER {
    pbar = 0.0 (cm/s)
    :q = 1	: room temperature 22-25 C
    q = 2	: body temperature 35 C
    base   = 0.0      : set in simulation file    
	factor = 0.0      : set in simulation file
    dv = 1e-4 (mV)
} 

ASSIGNED { 
    v (mV)
    ical (mA/cm2)
    ecal (mV)
    celsius (degC)
    cali (mM)
    calo (mM)
    minf
    mtau (ms)
    hinf
    htau (ms)
    pka (1)

	didv didc didm didh djdv djdc djdm djdh
	dmdv dmdm_ dhdv dhdh_
}

STATE { m h mnew hnew }

BREAKPOINT {
    SOLVE states :METHOD cnexp
    ical = modulation() * pbar*m*m*h*ghk(v, cali, calo)
    didv = modulation() * pbar*m*m*h*dFdv(v, cali, calo)
    didc = modulation() * pbar*m*m*h*dFdc(v, cali, calo)
    didm = modulation() * pbar*2*m*h*ghk(v, cali, calo)
    didh = modulation() * pbar*m*m*ghk(v, cali, calo)
    djdv = modulation() * pbar*m*m*h*dFdv_dv(v, cali, calo)
    djdc = modulation() * pbar*m*m*h*dFdv_dc(v, cali, calo)
    djdm = modulation() * pbar*2*m*h*dFdv(v, cali, calo)
    djdh = modulation() * pbar*m*m*dFdv(v, cali, calo)
}

INITIAL {
    rates(v)
    m = minf
    h = hinf
}

PROCEDURE states() {
    LOCAL minus_m, minus_h
    rates(v - dv)
    minus_m = m + (1 - exp(-dt/mtau*q)) * (minf - m)
    minus_h = h + (1 - exp(-dt/htau*q)) * (hinf - h)

    rates(v + dv)
    mnew = m + (1 - exp(-dt/mtau*q)) * (minf - m)
    hnew = h + (1 - exp(-dt/htau*q)) * (hinf - h)
    rates(v)
    m = m + (1 - exp(-dt/mtau*q)) * (minf - m)
    h = h + (1 - exp(-dt/htau*q)) * (hinf - h)

    dmdm_ = exp(-dt/mtau*q)
    dhdh_ = exp(-dt/htau*q)

    UNITSOFF
    dmdv = (mnew - minus_m)/(2*dv)
    dhdv = (hnew - minus_h)/(2*dv)
    UNITSON
    :m' = (minf-m)/mtau*q
    :h' = (hinf-h)/htau*q
}

PROCEDURE rates(v (mV)) {
    UNITSOFF
    minf = 1/(1+exp((v-(-33))/(-6.7)))
    :mtau = 0.06+1/(0.06*exp((v-(-46))/20)+0.41*exp((v-26)/-48))
    mtau = 0.06+1/(exp((v-10)/20)+exp((v-(-17))/-48))
    hinf = 1/(1+exp((v-(-13.4))/11.9))
    htau = 44.3
    UNITSON
}

FUNCTION ghk(v (mV), ci (mM), co (mM)) (.001 coul/cm3) {
    LOCAL z, eci, eco
    z = (1e-3)*2*FARADAY*v/(R*(celsius+273.15))
    if(z == 0) {
        z = z+1e-6
    }
    eco = co*(z)/(exp(z)-1)
    eci = ci*(-z)/(exp(-z)-1)
    ghk = (1e-3)*2*FARADAY*(eci-eco)
}

FUNCTION modulation() {
    
    : returns modulation factor
    
    modulation = 1 + factor * (pka - base)
    
}

FUNCTION dFdv(v (mV), ci (mM), co (mM)) (.001 coul/cm3/mV) {
    LOCAL A, k0, z, f1, f2
    A  = (1e-3)*2*FARADAY
    k0 = A/(R*(celsius+273.15))
    z  = k0*v
    if (fabs(z) < 1e-6) {
        dFdv = A*k0*(ci+co)/2
    } else {
        f1 = -((1+z)*exp(-z) - 1)/pow(1 - exp(-z), 2)
        f2 =  ((exp(z)-1) - z*exp(z))/pow(exp(z)-1, 2)
        dFdv = A*k0*(ci*f1 - co*f2)
    }
}

FUNCTION dFdc(v (mV), ci (mM), co (mM)) (.001 coul/cm3/mM) {
    LOCAL A, k0, z
    A  = (1e-3)*2*FARADAY
    k0 = A/(R*(celsius+273.15))
    z  = k0*v
    if (fabs(z) < 1e-6) {
        dFdc = A
    } else {
        dFdc = A*(-z/(exp(-z)-1))
    }
}

FUNCTION dFdv_dv(v (mV), ci (mM), co (mM)) (.001 coul/cm3/mV2) {
    LOCAL A, k0, z, f1p, f2p
    A  = (1e-3)*2*FARADAY
    k0 = A/(R*(celsius+273.15))
    z  = k0*v
    if (fabs(z) < 1e-6) {
        dFdv_dv = A*k0*k0*(ci+co)/6
    } else {
        f1p = -((2*exp(-z) + z*(1+exp(-z)) - 2)*exp(-z))/pow(1 - exp(-z), 3)
        f2p =  ((2*exp(z)   - z*(1+exp(z))   - 2)*exp(z)) /pow(exp(z)-1, 3)
        dFdv_dv = A*k0*k0*(ci*f1p - co*f2p)
    }
}

FUNCTION dFdv_dc(v (mV), ci (mM), co (mM)) (.001 coul/cm3/mM/mV) {
    LOCAL A, k0, z, f1
    A  = (1e-3)*2*FARADAY
    k0 = A/(R*(celsius+273.15))
    z  = k0*v
    if (fabs(z) < 1e-6) {
        dFdv_dc = A*k0/2
    } else {
        f1 = -((1+z)*exp(-z) - 1)/pow(1 - exp(-z), 2)
        dFdv_dc = A*k0*f1
    }
}

COMMENT

Activation curve was reconstructed for cultured NAc neurons from
P5-P32 Charles River rat pups [1] and shifted to match LVA data [7,
Fig.1D]. Activation time constant is from the rodent neuron culture (both
rat and mouse cells), room temperature 22-25 C [2, Fig.15A]. Inactivation
curve of CaL v1.3 current was taken from HEK cells [3, Fig.2 and p.819]
at room temperature.

Original NEURON model by Wolf (2005) [4] was modified by Alexander Kozlov
<akozlov@kth.se>. Kinetics of m2h type was used [5,6]. Activation
time constant was refitted to avoid singularity.

[1] Churchill D, Macvicar BA (1998) Biophysical and pharmacological
characterization of voltage-dependent Ca2+ channels in neurons isolated
from rat nucleus accumbens. J Neurophysiol 79(2):635-47.

[2] Kasai H, Neher E (1992) Dihydropyridine-sensitive and
omega-conotoxin-sensitive calcium channels in a mammalian
neuroblastoma-glioma cell line. J Physiol 448:161-88.

[3] Bell DC, Butcher AJ, Berrow NS, Page KM, Brust PF, Nesterova A,
Stauderman KA, Seabrook GR, Nurnberg B, Dolphin AC (2001) Biophysical
properties, pharmacology, and modulation of human, neuronal L-type
(alpha(1D), Ca(V)1.3) voltage-dependent calcium currents. J Neurophysiol
85:816-827.

[4] Wolf JA, Moyer JT, Lazarewicz MT, Contreras D, Benoit-Marand M,
O'Donnell P, Finkel LH (2005) NMDA/AMPA ratio impacts state transitions
and entrainment to oscillations in a computational model of the nucleus
accumbens medium spiny projection neuron. J Neurosci 25(40):9080-95.

[5] Evans RC, Morera-Herreras T, Cui Y, Du K, Sheehan T, Kotaleski JH,
Venance L, Blackwell KT (2012) The effects of NMDA subunit composition on
calcium influx and spike timing-dependent plasticity in striatal medium
spiny neurons. PLoS Comput Biol 8(4):e1002493.

[6] Tuckwell HC (2012) Quantitative aspects of L-type Ca2+ currents. Prog
Neurobiol 96(1):1-31.

[7] Xu W, Lipscombe D (2001) Neuronal cav1.3 L-type channels activate
at relatively hyperpolarized membrane potentials and are incompletely
inhibited by dihydropyridines. J Neurosci 21(16): 5944-5951.


ENDCOMMENT
