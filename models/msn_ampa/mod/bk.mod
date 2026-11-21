TITLE BK-type calcium activated K channel

UNITS {
    (molar) = (1/liter)
    (mV) = (millivolt)
    (mA) = (milliamp)
    (mM) = (millimolar)
    FARADAY = (faraday) (kilocoulombs)
    R = (k-mole) (joule/degC)
}

NEURON {
    SUFFIX bk
    USEION ca READ cai
    USEION k READ ek WRITE ik
    RANGE gbar, ik, gk
    RANGE dv, dc, dgdo, dodo_, dodv, dodc, ek
}

PARAMETER {
    gbar = 0.0 (mho/cm2)
    k1 = 0.180 (mM)
    k4 = 0.011 (mM)
    dv = 1e-4 (mV)
    dc = 1e-9 (mM)
}

ASSIGNED {
    v (mV)
    ik (mA/cm2)
    gk (mho/cm2)
    celsius (degC)
    cai (mM) 
    ek (mV)
    oinf
    otau (ms)

    dgdo dodo_ dodv dodc
}

STATE { o onew_v onew_c}

BREAKPOINT {
    SOLVE state :METHOD cnexp
    gk = gbar*o
    ik = gk*(v-ek)

    dgdo = gbar
}

PROCEDURE state() {
    LOCAL minus_v, minus_c
    rate(v - dv, cai)
    minus_v = o + (1 - exp(-dt/otau)) * (oinf - o)
    rate(v, cai - dc)
    minus_c = o + (1 - exp(-dt/otau)) * (oinf - o)
    rate(v + dv, cai)
    onew_v = o + (1 - exp(-dt/otau)) * (oinf - o)
    rate(v, cai + dc)
    onew_c = o + (1 - exp(-dt/otau)) * (oinf - o)
    rate(v, cai)
    o = o + (1 - exp(-dt/otau)) * (oinf - o)
    :o' = (oinf-o)/otau

    dodo_ = exp(-dt/otau)

    UNITSOFF
    dodv = (onew_v - minus_v) / (2*dv)
    dodc = (onew_c - minus_c) / (2*dc)
    UNITSON
}

INITIAL {
    rate(v, cai)
    o = oinf
}

PROCEDURE rate(v (mV), ca (mM)) {
    LOCAL a, b, sum, z
    UNITSOFF
    z = 1e-3*2*FARADAY/(R*(celsius+273.15))
    a = 0.48*ca/(ca+k1*exp(-0.84*z*v))
    b = 0.28/(1+ca/(k4*exp(-z*v)))
    sum = a+b
    oinf = a/sum
    otau = 1/sum
    UNITSON
}

COMMENT

Original model by Moczydlowski (1983), rat skeletal muscle.

Genesis implementation by De Schutter, adapted by Kai Du.

Revision by Evans (2012, 2013), K1 changed from 0.180 to 0.003 and K4
from 0.011 to 0.009, according to Berkefeld (2006), Xenopus oocytes.

NEURON implementation by Alexander Kozlov <akozlov@csc.kth.se>.

ENDCOMMENT
