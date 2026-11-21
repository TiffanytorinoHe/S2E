TITLE Fast A-type potassium current (Kv4.2) - autograd-ready

NEURON {
    THREADSAFE
    SUFFIX kaf_kai          : avoid conflict with original kaf; can be reverted to kaf
    USEION k READ ek WRITE ik
    RANGE gbar, gk, ik, base, factor, dv
    RANGE dgdm, dgdh, dmdm_, dhdh_, dmdv, dhdv, ek
    POINTER pka
}

UNITS {
    (S)  = (siemens)
    (mV) = (millivolt)
    (mA) = (milliamp)
    
}

PARAMETER {
    gbar   = 0.0 (S/cm2)
    q      = 2                 : Q10 temperature factor
    base   = 0.0               : set in simulation script
    factor = 0.0               : set in simulation script
    dv = 1e-4 (mV)         : voltage perturbation for gate/v via finite difference
}

ASSIGNED {
    v     (mV)
    ek    (mV)
    ik    (mA/cm2)
    gk    (S/cm2)

    minf
    mtau  (ms)
    hinf
    htau  (ms)

    pka   (1)

    dgdm     (S/cm2)           : gk/m
    dgdh     (S/cm2)           : gk/h
    dmdm_                 : m(t+dt)/m(t)
    dhdh_                 : h(t+dt)/h(t)
    dmdv                  : m(t+dt)/v
    dhdv                  : h(t+dt)/v
}

STATE { m h            : gating variables
        mnew hnew }    : temporary variables from v+dv step

BREAKPOINT {
    SOLVE states :METHOD cnexp
    gk = modulation() * gbar*m*m*h
    ik = gk*(v-ek)

    : compute partial derivatives of gk with respect to m and h
    dgdm = modulation() * gbar * (2*m) * h
    dgdh = modulation() * gbar * m*m
}

PROCEDURE states() {
    LOCAL minus_m, minus_h
    rates(v - dv)
    minus_m = m + (1 - exp(-dt/mtau*q)) * (minf - m)
    minus_h = h + (1 - exp(-dt/htau*q)) * (hinf - h)

    : --- Step 1: explicit exponential update using (v + dv) ---
    rates(v + dv)
    mnew = m + (1 - exp(-dt/mtau*q)) * (minf - m)
    hnew = h + (1 - exp(-dt/htau*q)) * (hinf - h)

    : --- Step 2: true update using actual v ---
    rates(v)
    m  = m + (1 - exp(-dt/mtau*q)) * (minf - m)
    h  = h + (1 - exp(-dt/htau*q)) * (hinf - h)

    : --- Step 3: record Jacobian elements for gradient backpropagation ---
    dmdm_ = exp(-dt/mtau*q)        : m(t+dt)/m(t)
    dhdh_ = exp(-dt/htau*q)        : h(t+dt)/h(t)

    UNITSOFF
    dmdv  = (mnew - minus_m)/(2*dv)          : m/v via finite difference
    dhdv  = (hnew - minus_h)/(2*dv)          : h/v via finite difference
    UNITSON
}

INITIAL {
    rates(v)
    m = minf
    h = hinf
}

PROCEDURE rates(v (mV)) {
    LOCAL alpha, beta, sum
    UNITSOFF
    : --- m-gate kinetics ---
    alpha = 1.5 / (1 + exp((v -   4)/(-17)))
    beta  = 0.6 / (1 + exp((v -  10)/(  9)))
    sum   = alpha + beta
    minf  = alpha / sum
    mtau  = 1 / sum

    : --- h-gate kinetics ---
    alpha = 0.105 / (1 + exp((v - (-121))/( 22)))
    beta  = 0.065 / (1 + exp((v -  (-55))/(-11)))
    sum   = alpha + beta
    hinf  = alpha / sum
    htau  = 1 / sum
    UNITSON
}

FUNCTION modulation() {
    : returns PKA modulation factor
    modulation = 1 + factor * (pka - base)
}

COMMENT
Original experimental data: Tkatch (2000), P4-6 rat, 22 °C  
Genesis implementation: Kai Du — MScell v9.5  
Modification: Robert Lindroos — applied Q10 to both m and h  
This autograd-ready version: Kai Style  
ENDCOMMENT
