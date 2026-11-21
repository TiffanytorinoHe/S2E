TITLE Inwardly rectifying potassium current (Kir) - autograd-ready

NEURON {
    THREADSAFE
    SUFFIX kir_kai           : avoid conflict with original kir; can revert to kir
    USEION k READ ek WRITE ik
    RANGE gbar, gk, ik, base, factor, dv
    RANGE dgdm, dmdm_, dmdv, ek
    POINTER pka
}

UNITS {
    (S)  = (siemens)
    (mV) = (millivolt)
    (mA) = (milliamp)
}

PARAMETER {
    gbar   = 0.0 (S/cm2)
    q      = 3               : Q10 temperature factor
    base   = 0.0             : set in simulation script
    factor = 0.0             : set in simulation script
    dv = 1e-4 (mV)       : voltage perturbation for m/v via finite difference
}

ASSIGNED {
    v     (mV)
    ek    (mV)
    ik    (mA/cm2)
    gk    (S/cm2)

    minf
    mtau  (ms)
    pka   (1)

    dgdm     (S/cm2)        : gk/m
    dmdm_                  : m(t+dt)/m(t)
    dmdv                   : m(t+dt)/v
}

STATE { m mnew }

BREAKPOINT {
    SOLVE states :METHOD cnexp
    gk = modulation() * gbar * m
    ik = gk * (v - ek)

    : compute partial derivative of gk with respect to m
    dgdm = modulation() * gbar
}

PROCEDURE states() {
    LOCAL minus_m
    rates(v - dv)
    minus_m = m + (1 - exp(-dt/mtau*q)) * (minf - m)

    : --- Step 1: explicit exponential update using (v + dv) ---
    rates(v + dv)
    mnew = m + (1 - exp(-dt/mtau*q)) * (minf - m)

    : --- Step 2: true update using actual v ---
    rates(v)
    m = m + (1 - exp(-dt/mtau*q)) * (minf - m)

    : --- Step 3: record Jacobian elements for gradient backpropagation ---
    dmdm_ = exp(-dt/mtau*q)     : m(t+dt)/m(t)

    UNITSOFF
    dmdv  = (mnew - minus_m)/(2*dv)       : m/v via finite difference
    UNITSON
}

INITIAL {
    rates(v)
    m = minf
}

PROCEDURE rates(v (mV)) {
    LOCAL alpha, beta, sum
    UNITSOFF
    : steady-state activation
    minf = 1 / (1 + exp((v - (-102)) / 13))
    : forward and backward rates for time constant
    alpha = 0.1 * exp((v - (-60)) / (-14))
    beta  = 0.27 / (1 + exp((v - (-31)) / (-23)))
    sum   = alpha + beta
    mtau  = 1 / sum
    UNITSON
}

FUNCTION modulation() {
    : returns PKA modulation factor
    modulation = 1 + factor * (pka - base)
}

COMMENT
Original data: Steephen (2009), rat, room temperature  
Genesis implementation: Kai Du — MScell v9.5  
NEURON implementation: Alexander Kozlov — smooth fit of mtau  
This autograd-ready version: Kai Style  
ENDCOMMENT
