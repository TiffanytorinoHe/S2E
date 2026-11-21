TITLE SK-type calcium-activated K channel (SK) - autograd-ready


NEURON {
    SUFFIX sk_kai
    USEION ca READ cai
    USEION k  READ ek WRITE ik
    RANGE gbar, gk, ik, dc
    RANGE dgdo, dodo_, dodc
}

UNITS {
    (molar) = (1/liter)
    (mM) = (millimolar)
    (mV) = (millivolt)
    (mA) = (milliamp)
}

PARAMETER {
    gbar = 0.0 (mho/cm2)
    q    = 1              : Q10 factor (unused)
    dc   = 1e-9 (mM)      : Ca perturbation for o/cai
}

ASSIGNED {
    cai   (mM)
    ek    (mV)
    ik    (mA/cm2)
    gk    (S/cm2)
    oinf
    otau  (ms)

    dgdo               : gk/o
    dodo_              : o(t+dt)/o(t)
    dodc               : o(t+dt)/cai
}

STATE {
    o      : gating variable
    o_new  : temporary for perturbed Cai
}

BREAKPOINT {
    SOLVE state :METHOD cnexp
    gk   = gbar * o
    ik   = gk * (v - ek)

    : gradient of gk w.r.t. o
    dgdo = gbar
}

PROCEDURE state() {
    LOCAL minus_c
    rate(cai - dc)
    minus_c = o + (1 - exp(-dt/otau*q))*(oinf - o)
    : --- forward step at cai + dc for finite difference ---
    rate(cai + dc)
    o_new = o + (1 - exp(-dt/otau*q))*(oinf - o)

    : --- actual update at current cai ---
    rate(cai)
    o     = o + (1 - exp(-dt/otau*q))*(oinf - o)

    : --- record Jacobian terms ---
    dodo_ = exp(-dt/otau*q)
    UNITSOFF
    dodc  = (o_new - minus_c)/(2*dc)
    UNITSON
}

INITIAL {
    rate(cai)
    o = oinf
}

PROCEDURE rate(ca (mM)) {
    LOCAL a
    :if (ca < 1e-10) { ca = 1e-10 }
    a    = (ca/0.57e-3)^5.2
    oinf = a/(1 + a)
    otau = 4.9
}

COMMENT
Original: Hirschberg (1998) & Maylie (2004)  
This adds only the minimal state and gradient machinery; all other kinetics are unchanged.
ENDCOMMENT
