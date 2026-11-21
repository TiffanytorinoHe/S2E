TITLE t-type calcium channel with high threshold for activation
: used in somatic and dendritic regions 
: it calculates I_Ca using channel permeability instead of conductance

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
(molar) = (1/liter)
	(mM) = (millimolar)

	FARADAY = 96520 (coul)
	R = 8.3134 (joule/degK)
	KTOMV = .0853 (mV/degC)
}


NEURON {
	SUFFIX cat
	USEION ca READ cai,cao 
        USEION Ca WRITE iCa VALENCE 2
        : The T-current does not activate calcium-dependent currents.
        : The construction with dummy ion Ca prevents the updating of the 
        : internal calcium concentration. 
        RANGE gcat, gcatbar, hinf, minf, taum, tauh, iCa, dv
		RANGE didv, didc, didm, didh, djdv, djdc, djdm, djdh
		RANGE dmdv, dmdm_, dhdv, dhdh_
}
INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

PARAMETER {           :parameters that can be entered when function is called in cell-setup 
        dt            (ms)
	v             (mV)
        tBase = 23.5  (degC)
	celsius = 22  (degC)
	gcatbar = 0   (mho/cm2)  : initialized conductance
	ki = 0.001    (mM)
	cai = 5.e-5   (mM)       : initial internal Ca++ concentration
	cao = 2       (mM)       : initial external Ca++ concentration
        tfa = 1                  : activation time constant scaling factor
        tfi = 0.68               : inactivation time constant scaling factor
        eca = 140                : Ca++ reversal potential
        dv = 1e-4 (mV)
}


STATE {
	m h mnew hnew 
} 

ASSIGNED {     : parameters needed to solve DE
	iCa (mA/cm2)
        gcat (mho/cm2) 
                hinf
                tauh
                minf
	taum
	didv didc didm didh djdv djdc djdm djdh
	dmdv dmdm_ dhdv dhdh_
}

INITIAL {
:        tadj = 3^((celsius-tBase)/10)   : assume Q10 of 3
	rates(v)
        m = minf
        h = hinf
	gcat = gcatbar*m*m*h*h2(cai)
}

BREAKPOINT {
	SOLVE states :METHOD cnexp
	gcat = gcatbar*m*m*h*h2(cai)
	iCa = gcat*ghk(v,cai,cao)
	didv = gcat * dFdv(v,cai,cao)
	didc = - gcat / (ki+cai) * ghk(v,cai,cao) + gcat * dFdc(v,cai,cao)
	didm = 2 * gcatbar * m * h * h2(cai) * ghk(v,cai,cao)
	didh = gcatbar * m * m * h2(cai) * ghk(v,cai,cao)
	djdv = gcat * dFdv_dv(v,cai,cao)
	djdc = - gcat / (ki+cai) * dFdv(v,cai,cao) + gcat * dFdv_dc(v,cai,cao)
	djdm = 2 * gcatbar * m * h * h2(cai) * dFdv(v,cai,cao)
	djdh = gcatbar * m * m * h2(cai) * dFdv(v,cai,cao)
}

PROCEDURE states() {
 LOCAL minus_m,minus_h
 rates(v-dv)
 minus_m = m + (1 - exp(-dt/taum)) * (minf - m)
 minus_h = h + (1 - exp(-dt/tauh)) * (hinf - h)
	: exact when v held constant
	rates(v + dv)
    mnew = m + (1 - exp(-dt/taum)) * (minf - m)
	hnew = h + (1 - exp(-dt/tauh)) * (hinf - h)

	rates(v)
    m = m + (1 - exp(-dt/taum)) * (minf - m)
	h = h + (1 - exp(-dt/tauh)) * (hinf - h)

    dmdm_ = exp(-dt/taum)
    dhdh_ = exp(-dt/tauh)

    UNITSOFF
    dmdv = (mnew - minus_m)/(2*dv)
    dhdv = (hnew - minus_h)/(2*dv)
	UNITSON

	:m' = (minf - m)/taum
	:h' = (hinf - h)/tauh
}


UNITSOFF
FUNCTION h2(cai(mM)) {
	h2 = ki/(ki+cai)
}

FUNCTION ghk(v(mV), ci(mM), co(mM)) (mV) { LOCAL nu,f
        f = KTF(celsius)/2
        nu = v/f
        ghk=-f*(1. - (ci/co)*exp(nu))*efun(nu)
}

FUNCTION KTF(celsius (degC)) (mV) {   : temperature-dependent adjustment factor
        KTF = ((25./293.15)*(celsius + 273.15))
}

FUNCTION efun(z) {
	if (fabs(z) < 1e-4) {
		efun = 1 - z/2
	}else{
		efun = z/(exp(z) - 1)
	}
}

FUNCTION dfdv(z) {
	if (fabs(z) < 1e-4) {
		dfdv = -1/2
	}else{
		dfdv = (exp(z) - 1 - z*exp(z))/(exp(z) - 1)^2
	}
}

FUNCTION dFdv(v(mV), ci(mM), co(mM)) {
	LOCAL nu,f
	f = KTF(celsius)/2
	nu = v/f
	dFdv = (ci/co)*exp(nu)*efun(nu) -(1. - (ci/co)*exp(nu))*dfdv(nu)
}

FUNCTION dFdc(v(mV), ci(mM), co(mM)) {
	LOCAL nu,f
	f = KTF(celsius)/2
	nu = v/f
	dFdc = (f/co)*exp(nu)*efun(nu)
}
FUNCTION ddfdv(z){
	if (fabs(z) < 1e-4) {
		ddfdv = 0
	}else{
		ddfdv = ((z-2) * exp(2*z) + (z+2) * exp(z))/(exp(z) - 1)^3
	}
}

FUNCTION dFdv_dv(v(mV), ci(mM), co(mM)) {
	LOCAL nu,f
	f = KTF(celsius)/2
	nu = v/f
	dFdv_dv = (1/f) * ((ci/co)*exp(nu) * (efun(nu) + 2 * dfdv(nu) + ddfdv(nu)) - ddfdv(nu))
}

FUNCTION dFdv_dc(v(mV), ci(mM), co(mM)) {
	LOCAL nu,f
	f = KTF(celsius)/2
	nu = v/f
	dFdv_dc = (1/co) * exp(nu) * (efun(nu) + dfdv(nu))
}

FUNCTION alph(v(mV)) {
	TABLE FROM -150 TO 150 WITH 200
	alph = 1.6e-4*exp(-(v+57)/19)
}

FUNCTION beth(v(mV)) {
        TABLE FROM -150 TO 150 WITH 200
	beth = 1/(exp((-v+15)/10)+1.0)
}

FUNCTION alpm(v(mV)) {
	TABLE FROM -150 TO 150 WITH 200
	alpm = 0.1967*(-1.0*v+19.88)/(exp((-1.0*v+19.88)/10.0)-1.0)
}

FUNCTION betm(v(mV)) {
	TABLE FROM -150 TO 150 WITH 200
	betm = 0.046*exp(-v/22.73)
}

UNITSON


PROCEDURE rates(v (mV)) { :callable from hoc
        LOCAL a
        a = alpm(v)
        taum = 1/(tfa*(a + betm(v))) : estimation of activation tau
        minf =  a/(a+betm(v))        : estimation of activation steady state
                a = alph(v)
        tauh = 1/(tfi*(a + beth(v))) : estimation of inactivation tau
        hinf = a/(a+beth(v))         : estimation of inactivation steady state
        
}
