TITLE Delayed rectifying potassium current

NEURON {
    SUFFIX kdr_kai
    USEION k READ ek WRITE ik
    RANGE gbar, gk, ik, dv, dgdm, dmdm_, dmdv, ek
}

UNITS {
    (S) = (siemens)
    (mV) = (millivolt)
    (mA) = (milliamp)
    
}

PARAMETER {
    gbar = 0.0 (S/cm2) 
    q = 3
    dv = 1e-4 (mV)
}

ASSIGNED {
    v (mV)
    ek (mV)
    ik (mA/cm2)
    gk (S/cm2)
    minf
    mtau (ms)
    dgdm dmdm_ dmdv
}

STATE { m mnew }

BREAKPOINT {
    SOLVE states :METHOD cnexp
    gk = gbar*m
    ik = gk*(v-ek)
    dgdm = gbar
}

PROCEDURE states() {
    LOCAL minus_m
    rates(v - dv)
    minus_m = m + (1 - exp(-dt/mtau*q))*(minf - m)

    rates(v + dv)
    mnew = m + (1 - exp(-dt/mtau*q))*(minf - m)
    rates(v)
    m = m + (1 - exp(-dt/mtau*q))*(minf - m)
    dmdm_ = exp(-dt/mtau*q)
    UNITSOFF
    dmdv = (mnew - minus_m)/(2*dv)
    UNITSON
    :m' = (minf-m)/mtau*q
}

INITIAL {
    rates(v)
    m = minf
}

PROCEDURE rates(v (mV)) {
    LOCAL alpha, beta, sum
    UNITSOFF
    alpha = 1.0*exp((v-(-13))/(-9.09))
    beta =  1.0*exp((v-(-13))/(-12.5))
    sum = alpha+1
    minf = 1/sum
    mtau = 50*beta/sum
    UNITSON
}

COMMENT

Original data by Migliore (1999), rat CA1, 22 C.

Genesis implementation by Kai Du <kai.du@ki.se>, MScell v9.5.

NEURON implementation by Alexander Kozlov <akozlov@csc.kth.se>.

ENDCOMMENT
