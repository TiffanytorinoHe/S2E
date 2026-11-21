NEURON {
	POINT_PROCESS tmgExSyn
	RANGE e, i, w, g
	RANGE tau_r, tau_d, Dep, Fac, Use, u0, z0
	NONSPECIFIC_CURRENT i
    RANGE pure_i, didv, djdv
    RANGE id
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(uS) = (microsiemens)
}

PARAMETER {
	: e = -70 mV for inhibitory synapses,
	:     0 mV for excitatory
	e = -70	(mV)
	tau_d = 3 (ms) < 1e-9, 1e9 >
	tau_r = 0.2 (ms): conductance rise time.
	: Dep = 100 ms for inhibitory synapses,
	:           800 ms for excitatory
	Dep = 100 (ms) < 1e-9, 1e9 >
	: Fac = 1000 ms for inhibitory synapses,
	:             0 ms for excitatory
	Fac = 1000 (ms) < 0, 1e9 >
	: Use = 0.04 for inhibitory synapses, 
	:     0.5 for excitatory
	: the (1) is needed for the < 0, 1 > to be effective
	:   in limiting the values of Use and u0
	Use = 0.04 (1) < 0, 1 >
	: initial value for the "facilitation variable"
	u0 = 0 (1) < 0, 1 >
        : initial value for the "depression variable"
        z0 = 0

    w = 0.001 
    id = 0
}

ASSIGNED {
	v (mV)
	i (nA)
    pure_i (nA)
	x
	g (uS)
	factor
    didv
    djdv
}

STATE {
	A (1)
	B (1)
}

INITIAL {
	LOCAL tp
	if (tau_r/tau_d > .9999) {
		tau_r = .9999*tau_d
	}
	A = 0
	B = 0
	g = w*(B-A)
	tp = (tau_r*tau_d)/(tau_d - tau_r) * log(tau_d/tau_r)
	factor = -exp(-tp/tau_r) + exp(-tp/tau_d)
	factor = 4/factor  : for fast-spiking interneurons, weight > SST interneurons
}

BREAKPOINT {
	SOLVE state METHOD cnexp
    g = B - A
    pure_i = g*(v - e)
    i = w * pure_i
    didv = - g
    djdv = 0
}

DERIVATIVE state {
	A' = -A/tau_r
	B' = -B/tau_d
}

NET_RECEIVE(weight (1), y, z, u, tsyn (ms)) {
INITIAL {
	y = 0
	z = z0
:	u = 0
	u = u0
	tsyn = t
: printf("t\t t-tsyn\t y\t z\t u\t newu\t g\t dg\t newg\t newy\n")
}
	: first calculate z at event-
	:   based on prior y and z
	z = z*exp(-(t - tsyn)/Dep)
	z = z + ( y*(exp(-(t - tsyn)/tau_d) - exp(-(t - tsyn)/Dep)) / ((tau_d/Dep)-1) )
	: now calc y at event-
	y = y*exp(-(t - tsyn)/tau_d)

	x = 1-y-z

	: calc u at event-
	if (Fac > 0) {
		u = u*exp(-(t - tsyn)/Fac)
	} else {
		u = Use
	}

: printf("%g\t%g\t%g\t%g\t%g", t, t-tsyn, y, z, u)

	if (Fac > 0) {
		u = u + Use*(1-u)
	}

: printf("\t%g\t%g\t%g", u, g, weight*x*u)

	A = A + weight*x*u*factor
	B = B + weight*x*u*factor
	y = y + x*u

	tsyn = t

: printf("\t%g\t%g\n", g, y)
}