TITLE R-type calcium current (Cav2.3)

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
    SUFFIX car_kai
    USEION ca READ cai, cao WRITE ica VALENCE 2
    RANGE pbar, ica
    RANGE dv
    RANGE didv, didc, didm, didh, djdv, djdc, djdm, djdh
    RANGE dmdv, dmdm_, dhdv, dhdh_
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
    hinf
    htau (ms)

	didv didc didm didh djdv djdc djdm djdh
	dmdv dmdm_ dhdv dhdh_
}

STATE { m h mnew hnew }

BREAKPOINT {
    SOLVE states :METHOD cnexp
    ica = pbar*m*m*m*h*ghk(v, cai, cao)
    didv = pbar*m*m*m*h*dFdv(v, cai, cao)
    didc = pbar*m*m*m*h*dFdc(v, cai, cao)
    didm = pbar*3*m*m*h*ghk(v, cai, cao)
    didh = pbar*m*m*m*ghk(v, cai, cao)
    djdv = pbar*m*m*m*h*dFdv_dv(v, cai, cao)
    djdc = pbar*m*m*m*h*dFdv_dc(v, cai, cao)
    djdm = pbar*3*m*m*h*dFdv(v, cai, cao)
    djdh = pbar*m*m*m*dFdv(v, cai, cao)
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
    minf = 1/(1+exp((v-(-29))/(-9.6)))
    mtau = 5.1*3
    hinf = 1/(1+exp((v-(-33.3))/17))
    htau = 22+80/(1+exp((v-(-19))/5))
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

Original data by Foehring  et al (2000) [1] for dissociated MSNs from
P28-P42 Sprague-Dawley rat brain. Unspecified recording temperature. The
liquid junction potential was around 8 mV and was not corrected. Kinetics
of m3h type was fitted.  Inactivation time constants were measured in
neurons from endopiriform nucleus of P7-P21 Hartley guinea pigs [2]
at room temperature 22 C.

Original NEURON model by Wolf (2005) [3] modified by Alexander Kozlov
<akozlov@kth.se>. Activation curve fitted to m3 kinetics [4],
activation time constant scaled up as well. Smooth fit of inactivation
time constant from [2,3].

[1] Foehring RC, Mermelstein PG, Song WJ, Ulrich S, Surmeier DJ
(2000) Unique properties of R-type calcium currents in neocortical and
neostriatal neurons. J Neurophysiol 84(5):2225-36.

[2] Brevi S, de Curtis M, Magistretti J (2001) Pharmacological and
biophysical characterization of voltage-gated calcium currents in the
endopiriform nucleus of the guinea pig. J Neurophysiol 85(5):2076-87.

[3] Wolf JA, Moyer JT, Lazarewicz MT, Contreras D, Benoit-Marand M,
O'Donnell P, Finkel LH (2005) NMDA/AMPA ratio impacts state transitions
and entrainment to oscillations in a computational model of the nucleus
accumbens medium spiny projection neuron. J Neurosci 25(40):9080-95.

[4] Evans RC, Maniar YM, Blackwell KT (2013) Dynamic modulation of
spike timing-dependent calcium influx during corticostriatal upstates. J
Neurophysiol 110(7):1631-45.

ENDCOMMENT
