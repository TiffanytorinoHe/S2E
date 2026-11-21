TITLE Fast transient sodium current (NaF) - autograd-ready

NEURON {
    THREADSAFE
    SUFFIX naf_kai            : avoid conflict with original naf; can revert to naf
    USEION na READ ena WRITE ina
    RANGE gbar, gna, ina, mVhalf, hVhalf, mSlope, hSlope, taum, tauh, taun
    RANGE base, factor, dv
    RANGE dgdm, dgdh, dmdm_, dhdh_, dmdv, dhdv, ena
    POINTER pka
}

UNITS {
    (S)  = (siemens)
    (mV) = (millivolt)
    (mA) = (milliamp)
}

PARAMETER {
    gbar     = 0.0 (S/cm2)
    q        = 1.8               : Q10 temperature factor
    mVhalf   = -25.0 (mV)
    hVhalf   = -62.0 (mV)
    mSlope   =  -9.2 (mV)
    hSlope   =   6.0 (mV)
    taum     =   0.09 (ms)        : unused, kept for compatibility
    taun     =   0.34 (ms)        : unused, kept for compatibility
    tauh     =   0.34 (ms)        : unused, kept for compatibility
    base     = 0.0                : set in simulation script (if used)
    factor   = 0.0                : set in simulation script (if used)
    dv = 1e-4 (mV)          : voltage perturbation for gate/v via finite difference
}

ASSIGNED {
    v       (mV)
    ena     (mV)
    ina     (mA/cm2)
    gna     (S/cm2)
    minf
    mtau    (ms)
    hinf
    htau    (ms)
    pka     (1)

    dgdm     (S/cm2)             : gna/m
    dgdh     (S/cm2)             : gna/h
    dmdm_                       : m(t+dt)/m(t)
    dhdh_                       : h(t+dt)/h(t)
    dmdv                        : m(t+dt)/v
    dhdv                        : h(t+dt)/v
}

STATE { m h mnew hnew }

BREAKPOINT {
    SOLVE states :METHOD cnexp
    gna = modulation() * gbar * m*m*m * h
    ina = gna * (v - ena)

    : compute partial derivatives of gna with respect to m and h
    dgdm = modulation() * gbar * 3*m*m * h
    dgdh = modulation() * gbar * m*m*m
}

PROCEDURE states() {
    LOCAL minus_m, minus_h, h_old, htau_prime, rate_a, rate_b
    h_old = h
    rates(v - dv)
    minus_m = m + (1 - exp(-dt/mtau*q)) * (minf - m)
    minus_h = h + (1 - exp(-dt/htau*q)) * (hinf - h)

    : --- Step 1: exponential update at (v + dv) for finite-difference estimate ---
    rates(v + dv)
    mnew = m + (1 - exp(-dt/mtau*q)) * (minf - m)
    hnew = h + (1 - exp(-dt/htau*q)) * (hinf - h)

    : --- Step 2: actual update at v ---
    rates(v)
    m = m + (1 - exp(-dt/mtau*q)) * (minf - m)
    h = h + (1 - exp(-dt/htau*q)) * (hinf - h)

    : --- Step 3: record Jacobian terms for backpropagation ---
    dmdm_ = exp(-dt/mtau*q)        : m(t+dt)/m(t)
    dhdh_ = exp(-dt/htau*q)        : h(t+dt)/h(t)

    UNITSOFF
    dmdv  = (mnew - minus_m)/(2*dv)          : m/v via finite difference
    : Differentiate the active branch of the unchanged historical htau.
    : A fixed +/-dv stencil can cross its discontinuity at -60 mV.
    if (v < -60) {
        htau_prime = 0.015
    } else {
        rate_a = 1 / (1 + exp((v + 48) / 15))
        rate_b = 1 / (1 + exp((v + 48) / 4))
        htau_prime = -1.1*rate_a*(1-rate_a)/15 - 1.2*rate_b*(1-rate_b)/4
    }
    dhdv = -(1-dhdh_)*hinf*(1-hinf)/hSlope - dhdh_*dt*q*htau_prime/(htau*htau)*(hinf-h_old)
    UNITSON
}

INITIAL {
    rates(v)
    m = minf
    h = hinf
}

PROCEDURE rates(v (mV)) {
    UNITSOFF
    : steady-state activation
    minf = 1 / (1 + exp((v - mVhalf) / mSlope))
    : steady-state inactivation
    hinf = 1 / (1 + exp((v - hVhalf) / hSlope))

    : m time constant
    mtau = 0.38 + 1 / (0.6*exp((v - (-58.0)) / 8.0) + 1.8*exp((v - (-58.0)) / (-35.0)))

    : h time constant
    if (v < -60) {
        htau = 3.4 + 0.015*v
    } else {
        htau = 0.56 +
               1.1 / (1 + exp((v - (-48)) / 15.0)) +
               1.2 / (1 + exp((v - (-48)) / 4.0))
    }
    UNITSON
}

FUNCTION modulation() {
    : returns PKA modulation factor
    modulation = 1 + factor * (pka - base)
}

COMMENT
Original data: Ogata (1990), guinea pig, 22 °C  
Genesis implementation: Kai Du — MScell v9.5  
NEURON implementation: Alexander Kozlov — smooth fits of mtau and htau  
Updates: Robert Lindroos — natural log kinetics, reduced AHP, Q10 = 1.8  
This autograd-ready version: Kai Style  
ENDCOMMENT
