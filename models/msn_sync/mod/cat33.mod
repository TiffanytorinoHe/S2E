TITLE T-type calcium current (Cav3.3)

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
    SUFFIX cat33
    USEION cal READ cali, calo WRITE ical VALENCE 2
    RANGE pbar, ical
    RANGE dv
    RANGE didv, didc, didm, didh, djdv, djdc, djdm, djdh
    RANGE dmdv, dmdm_, dhdv, dhdh_
}

PARAMETER {
    pbar = 0.0 (cm/s)
    :q = 1	: room temperature 21 C
    q = 3	: body temperature 35 C
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

	didv didc didm didh djdv djdc djdm djdh
	dmdv dmdm_ dhdv dhdh_
}

STATE { m h mnew hnew }

BREAKPOINT {
    SOLVE states :METHOD cnexp
    ical = pbar*m*m*m*h*ghk(v, cali, calo)
    didv = pbar*m*m*m*h*dFdv(v, cali, calo)
    didc = pbar*m*m*m*h*dFdc(v, cali, calo)
    didm = pbar*3*m*m*h*ghk(v, cali, calo)
    didh = pbar*m*m*m*ghk(v, cali, calo)
    djdv = pbar*m*m*m*h*dFdv_dv(v, cali, calo)
    djdc = pbar*m*m*m*h*dFdv_dc(v, cali, calo)
    djdm = pbar*3*m*m*h*dFdv(v, cali, calo)
    djdh = pbar*m*m*m*dFdv(v, cali, calo)
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
    minf = 1/(1+exp((v-(-81))/(-5.8)))
    mtau = (2.3+20/(1+exp((v-(-60))/9)))*3
    hinf = 1/(1+exp((v-(-78.3))/6.5))
    htau = 125+140/(1+exp((v-(-60))/3))
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
    :ghk = (1e-3)*2*FARADAY* (ci*(-k0 * v)/(exp(-k0 * v)-1) - co*(k0 * v)/(exp(k0 * v)-1))
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

Rat Cav3.2 channels were isolated and transfection of human embryonic
kidney cells was performed [1].  Electrophysiological recordings were
done in 21 C.

NEURON model by Alexander Kozlov <akozlov@kth.se>. Kinetics of m3h
type was used [2-4]. Activation time constant was scaled up accordingly.

[1] Iftinca M, McKay BE, Snutch TP, McRory JE, Turner RW, Zamponi
GW (2006) Temperature dependence of T-type calcium channel
gating. Neuroscience 142(4):1031-42.

[2] Crunelli V, Toth TI, Cope DW, Blethyn K, Hughes SW (2005) The
'window' T-type calcium current in brain dynamics of different behavioural
states. J Physiol 562(Pt 1):121-9.

[3] Wolf JA, Moyer JT, Lazarewicz MT, Contreras D, Benoit-Marand M,
O'Donnell P, Finkel LH (2005) NMDA/AMPA ratio impacts state transitions
and entrainment to oscillations in a computational model of the nucleus
accumbens medium spiny projection neuron. J Neurosci 25(40):9080-95.

[4] Evans RC, Maniar YM, Blackwell KT (2013) Dynamic modulation of
spike timing-dependent calcium influx during corticostriatal upstates. J
Neurophysiol 110(7):1631-45.

ENDCOMMENT
