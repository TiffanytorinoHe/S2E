TITLE Q-type calcium current (Cav2.1)

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
    SUFFIX caq_kai
    USEION ca READ cai, cao WRITE ica VALENCE 2
    RANGE pbar, ica
    RANGE dv    
    RANGE didv, didc, didm, djdv, djdc, djdm
    RANGE dmdv, dmdm_
}

PARAMETER {
    pbar = 0.0 (cm/s)
    :q = 1	: room temperature 22 C
    q = 3	: body temperature 35 C
    dv = 1e-4 (mV)
} 

ASSIGNED { 
    v (mV)
    ica (mA/cm2)
    eca (mV)
    celsius (degC)
    cai (mM)
    cao (mM)
    minf
    mtau (ms)
    didv didc didm djdv djdc djdm
    dmdv dmdm_
}

STATE { m mnew }

BREAKPOINT {
    SOLVE states :METHOD cnexp
    ica = pbar*m*m*ghk(v, cai, cao)
    didv = pbar*m*m*dFdv(v, cai, cao)
    didc = pbar*m*m*dFdc(v, cai, cao)
    didm = pbar*2*m*ghk(v, cai, cao)
    djdv = pbar*m*m*dFdv_dv(v, cai, cao)
    djdc = pbar*m*m*dFdv_dc(v, cai, cao)
    djdm = pbar*2*m*dFdv(v, cai, cao)    
}

INITIAL {
    rates(v)
    m = minf
}

PROCEDURE states() {
    LOCAL minus_m
    rates(v - dv)
    minus_m = m + (1 - exp(-dt/mtau*q)) * (minf - m)
 
    rates(v + dv)
    mnew = m + (1 - exp(-dt/mtau*q)) * (minf - m)
    rates(v)
    m = m + (1 - exp(-dt/mtau*q)) * (minf - m)

    dmdm_ = exp(-dt/mtau*q)
    UNITSOFF
    dmdv = (mnew - minus_m)/(2*dv)
    UNITSON
    :m' = (minf-m)/mtau*q
}

PROCEDURE rates(v (mV)) {
    UNITSOFF
    minf = 1/(1+exp((v-(-16.3))/(-7.9)))
    mtau = 1.13*2
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

Activation curve was reconstructed for cultured NAc neurons from P5-P32
Charles River rat pups [1].  Activation time constant was measured in
culture neurons from cerebellum of P2-P5 rat pups [2] at room temperature
22 C.

Original NEURON model by Wolf (2005) [3] was modified by Alexander Kozlov
<akozlov@kth.se>. Activation curve was fitted to m2 kinetics [4],
activation time constant was scaled up as well.

[1] Churchill D, Macvicar BA (1998) Biophysical and pharmacological
characterization of voltage-dependent Ca2+ channels in neurons isolated
from rat nucleus accumbens. J Neurophysiol 79(2):635-47.

[2] Randall A, Tsien RW (1995) Pharmacological dissection of multiple
types of Ca2+ channel currents in rat cerebellar granule neurons. J
Neurosci 15(4):2995-3012.

[3] Wolf JA, Moyer JT, Lazarewicz MT, Contreras D, Benoit-Marand M,
O'Donnell P, Finkel LH (2005) NMDA/AMPA ratio impacts state transitions
and entrainment to oscillations in a computational model of the nucleus
accumbens medium spiny projection neuron. J Neurosci 25(40):9080-95.

[4] Brown AM, Schwindt PC, Crill WE (1993) Voltage dependence and
activation kinetics of pharmacologically defined components of the
high-threshold calcium current in rat neocortical neurons. J Neurophysiol
70(4):1530-43.

ENDCOMMENT
