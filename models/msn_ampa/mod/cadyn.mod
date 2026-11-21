TITLE Calcium dynamics for N, P/Q, R calcium pool

NEURON {
    SUFFIX cadyn
    USEION ca READ ica, cai WRITE cai VALENCE 2
    RANGE pump, cainf, taur, drive
    RANGE dcdi, dcdc_
}

UNITS {
    (molar) = (1/liter) 
    (mM) = (millimolar)
    (um) = (micron)
    (mA) = (milliamp)
    (msM) = (ms mM)
    FARADAY = (faraday) (coulomb)
}

PARAMETER {
    drive = 10000 (1)
    depth = 0.1 (um)
    cainf = 10e-6 (mM)
    taur = 43 (ms)
    kt = 1e-4 (mM/ms)
    kd = 1e-4 (mM)
    pump = 0.02
}

STATE { cai (mM) }

INITIAL { cai = cainf }

ASSIGNED {
    ica (mA/cm2)
    drive_channel (mM/ms)
    drive_pump (mM/ms)
    dcdc_ dcdi
}
    
BREAKPOINT {
    SOLVE state :METHOD cnexp
}

PROCEDURE state() { 
    drive_channel = -drive*ica/(2*FARADAY*depth)
    if (drive_channel <= 0.) { drive_channel = 0. }
    drive_pump = -kt*cai/(cai+kd)
    cai = cai + (1. - exp(-dt / taur)) * ((drive_channel + pump * drive_pump + cainf / taur) * taur - cai)
    : cai = exp(-dt / taur) * cai + (exp(-dt / taur) - 1) * taur * (drive*ica/(2*FARADAY*depth) + pump * kt*cai/(cai+kd))
    :cai = exp(-dt / taur) * cai + (exp(-dt / taur) - 1) * taur * pump * kt * (1-kd/(cai+kd)) + (exp(-dt / taur) - 1) * taur * drive*ica/(2*FARADAY*depth)
    dcdc_ = exp(-dt / taur) + (exp(-dt / taur) - 1) * taur * pump * kt * kd /(cai+kd)/(cai+kd)
    dcdi = (exp(-dt / taur) - 1) * taur * drive / (2*FARADAY*depth)
    :cai' = (drive_channel+pump*drive_pump+(cainf-cai)/taur)
    :cai = cai + (1 - exp(-dt/taur))*()
    :(- ( ( drive_channel + ( pump )*( drive_pump ) + ( ( cainf ) ) / taur ) ) / ( ( ( ( ( - 1.0 ) ) ) / taur ) ) - cai)
}

COMMENT

Original NEURON model by Wolf (2005) and Destexhe (1992).  Adaptation by
Alexander Kozlov <akozlov@kth.se>.

[1] Wolf JA, Moyer JT, Lazarewicz MT, Contreras D, Benoit-Marand M,
O'Donnell P, Finkel LH (2005) NMDA/AMPA ratio impacts state transitions
and entrainment to oscillations in a computational model of the nucleus
accumbens medium spiny projection neuron. J Neurosci 25(40):9080-95.

ENDCOMMENT
