TITLE N-type calcium current (Cav2.2)

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
    SUFFIX can_kai
    USEION ca READ cai, cao WRITE ica VALENCE 2
    RANGE pbar, ica, base, factor
    POINTER pka
    RANGE dv
    RANGE didv, didc, didm, didh, djdv, djdc, djdm, djdh
    RANGE dmdv, dmdm_, dhdv, dhdh_
}

PARAMETER {
    pbar = 0.0 (cm/s)
    a = 0.21
    :q = 1	: room temperature 22-25 C
    q = 2	: body temperature 35 C
    base   = 0.0      : set in simulation file    
	factor = 0.0      : set in simulation file
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
    hinf
    htau (ms)
    pka (1)

	didv didc didm didh djdv djdc djdm djdh
	dmdv dmdm_ dhdv dhdh_
}

STATE { m h mnew hnew }

BREAKPOINT {
    SOLVE states :METHOD cnexp
    ica = modulation() * pbar*m*m*(h*a+1-a)*ghk(v, cai, cao)
    didv = modulation() * pbar*m*m*(h*a+1-a)*dFdv(v, cai, cao)
    didc = modulation() * pbar*m*m*(h*a+1-a)*dFdc(v, cai, cao)
    didm = modulation() * pbar*2*m*(h*a+1-a)*ghk(v, cai, cao)
    didh = modulation() * pbar*m*m*a*ghk(v, cai, cao)
    djdv = modulation() * pbar*m*m*(h*a+1-a)*dFdv_dv(v, cai, cao)
    djdc = modulation() * pbar*m*m*(h*a+1-a)*dFdv_dc(v, cai, cao)
    djdm = modulation() * pbar*2*m*(h*a+1-a)*dFdv(v, cai, cao)
    djdh = modulation() * pbar*m*m*a*dFdv(v, cai, cao)
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
    minf = 1/(1+exp((v-(-3))/(-8)))
    :mtau = (0.06+1/(0.033*exp((v-(-36))/18)+0.35*exp((v-15)/-44)))*2
    mtau = (0.06+1/(exp((v-25)/18)+exp((v-(-31))/-44)))*2
    hinf = 1/(1+exp((v-(-74.8))/6.5))
    htau = 70
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

Model is based on mixed data. Activation curve is from neostriatal
medium spinal neurons of adult P28+ rats [1, Fig.12F], unspecified
recording temperature. Potentials were not corrected for the liquid
junction potential, which was estimated to be 7 mV.  Activation time
constant is from the rodent neuron culture (both rat and mouse cells),
room temperature 22-25 C [2, Fig.15B].  Inactivation data is from human
(HEK) cells [3, Tab.1, Tab.2], supposedly at room temperature.

Kinetics of m2h type is used [2, Fig.5]. Activation of m2 type is
fitted to the experimental data [1,4], activation time constant [2]
is scaled up as well.  Original NEURON model [5,4] was modified by
Alexander Kozlov <akozlov@kth.se>. Activation time constant was refitted
to avoid singularity in the expression.

[1] Bargas J, Howe A, Eberwine J, Cao Y, Surmeier DJ (1994) Cellular
and molecular characterization of Ca2+ currents in acutely isolated,
adult rat neostriatal neurons. J Neurosci 14(11 Pt 1):6667-86.

[2] Kasai H, Neher E (1992) Dihydropyridine-sensitive and
omega-conotoxin-sensitive calcium channels in a mammalian
neuroblastoma-glioma cell line. J Physiol 448:161-88.

[3] McNaughton NC, Randall AD (1997) Electrophysiological properties of
the human N-type Ca2+ channel: I. Channel gating in Ca2+, Ba2+ and Sr2+
containing solutions. Neuropharmacology 36(7):895-915.

[4] Evans RC, Maniar YM, Blackwell KT (2013) Dynamic modulation of
spike timing-dependent calcium influx during corticostriatal upstates. J
Neurophysiol 110(7):1631-45.

[5] Wolf JA, Moyer JT, Lazarewicz MT, Contreras D, Benoit-Marand M,
O'Donnell P, Finkel LH (2005) NMDA/AMPA ratio impacts state transitions
and entrainment to oscillations in a computational model of the nucleus
accumbens medium spiny projection neuron. J Neurosci 25(40):9080-95.

ENDCOMMENT
