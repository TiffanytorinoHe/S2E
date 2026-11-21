TITLE CaGk
: Calcium activated mAHP K channel.
: From Moczydlowski and Latorre (1983) J. Gen. Physiol. 82

UNITS {
	(molar) = (1/liter)
}

UNITS {
	(mV) =	(millivolt)
	(mA) =	(milliamp)
	(mM) =	(millimolar)
}


NEURON {
	SUFFIX mykca
	USEION ca READ cai
	USEION k READ ek WRITE ik
	RANGE gkbar,gk, ik, dv, dc, dgdo, dodo_, dodv, dodc, ek
	GLOBAL oinf, tau
}

UNITS {
	FARADAY = (faraday)  (kilocoulombs)
	R = 8.313424 (joule/degC)
}

PARAMETER {
	v		(mV)
	dt		(ms)
	ek		(mV)
	celsius = 20	(degC)
	gkbar = 0.01	(mho/cm2)	: Maximum Permeability
	cai = 1e-3	(mM)
	d1 = 0.84
	d2 = 1.0
	k1 = 0.18	(mM)
	k2 = 0.011	(mM)
	bbar = 0.28	(/ms)
	abar = 0.48	(/ms)

	st=1            (1)
	dv = 1e-4 (mV)
	dc = 1e-9 (mM)
}
COMMENT
the preceding two numbers were switched on 8/19/92 in response to a bug
report by Bartlett Mel. In the paper the kinetic scheme is
C <-> CCa (K1)
CCa <-> OCa (beta2,alpha2)
OCa <-> OCa2 (K4)
In this model abar = beta2 and bbar = alpha2 and K4 comes from d2 and k2
I was forcing things into a nomenclature where alpha is the rate from
closed to open. Unfortunately I didn't switch the numbers.
ENDCOMMENT

ASSIGNED {
	ik		(mA/cm2)
	oinf
	tau		(ms)
	gk		(mho/cm2)
	dgdo dodo_ 
	dodv dodc
}

INITIAL {           : initialize the following parameter using rate()
	rate(v, cai)
	o = oinf
}

STATE {	o onew_v onew_c}		: fraction of open channels

BREAKPOINT {
	SOLVE state :METHOD cnexp
	gk = gkbar*o^st
	ik = gk*(v - ek)

	dgdo = gkbar
}

PROCEDURE state() {
 LOCAL minus_v,minus_c
 rate(v-dv,cai)
 minus_v=o + (1 - exp(-dt/tau)) * (oinf - o)
 rate(v,cai-dc)
 minus_c=o + (1 - exp(-dt/tau)) * (oinf - o)
	: exact when v held constant; integrates over dt step
	rate(v+dv, cai)
	onew_v = o + (1 - exp(-dt/tau)) * (oinf - o)
	rate(v, cai+dc)
	onew_c = o + (1 - exp(-dt/tau)) * (oinf - o)
	rate(v, cai)
	o = o + (1 - exp(-dt/tau)) * (oinf - o)
	:o' = (oinf - o)/tau

	dodo_ = exp(-dt/tau)
	UNITSOFF
	dodv = (onew_v-minus_v)/(2*dv)
	dodc = (onew_c-minus_c)/(2*dc)
	UNITSON
}

FUNCTION alp(v (mV), ca (mM)) (1/ms) { :callable from hoc
	alp = abar/(1 + exp1(k1,d1,v)/ca)
}

FUNCTION bet(v (mV), ca (mM)) (1/ms) { :callable from hoc
	bet = bbar/(1 + ca/exp1(k2,d2,v))
}  

FUNCTION exp1(k (mM), d, v (mV)) (mM) { :callable from hoc
	exp1 = k*exp(-2*d*FARADAY*v/R/(273.15 + celsius))
}

PROCEDURE rate(v (mV), ca (mM)) { :callable from hoc
	LOCAL a
	a = alp(v,ca)
	tau = 1/(a + bet(v, ca)) : estimation of activation tau
	oinf = a*tau             : estimation of activation steady state value
	:fac = (1 - exp(-dt/tau))
}
