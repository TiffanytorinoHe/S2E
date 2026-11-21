TITLE L-type calcium channel with low threshold for activation
: used in somatic and proximal dendritic regions 
: it calculates I_Ca using channel permeability instead of conductance

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	FARADAY = 96520 (coul)
	R = 8.3134 (joule/degK)
	KTOMV = .0853 (mV/degC)
}

PARAMETER {
	v (mV)
	celsius= 34	(degC)
	gcalbar=0 (mho/cm2)
	ki=.001 (mM)  
	cai = 50.e-6 (mM)
	cao = 2 (mM)
        tfa = 5
      ggk
        eca = 140	
	  dv = 1e-4 (mV)
}


NEURON {
	SUFFIX cal
	USEION ca READ cai,cao WRITE ica
        RANGE gcalbar,cai, ica, gcal, ggk
		RANGE dv, didv, didc, didm,  djdv, djdc, djdm
		RANGE dmdv, dmdm_
        GLOBAL minf,taum
}

STATE {
	m
	mnew
} 

ASSIGNED {
	ica (mA/cm2)
        gcal (mho/cm2) 
        minf
        taum  (ms)
		didv
		didc
		didm
		djdv
		djdc
		djdm 
		dmdv
		dmdm_
        }

INITIAL {
        rate(v)
        m = minf
	}

BREAKPOINT {
	SOLVE state :METHOD cnexp
	gcal = gcalbar*m*h2(cai)    
	ggk=ghk(v,cai,cao)
	ica = gcal*ggk
	didv = gcal * dFdv(v,cai,cao)
	didc = - gcal / (ki+cai) * ggk + gcal * dFdc(v,cai,cao)
	didm = gcalbar * h2(cai) * ggk
	djdv = gcal * dFdv_dv(v,cai,cao)
	djdc = - gcal / (ki+cai) * dFdv(v,cai,cao) + gcal * dFdv_dc(v,cai,cao)
	djdm = gcalbar * h2(cai) * dFdv(v,cai,cao)
}

FUNCTION h2(cai(mM)) {
	h2 = ki/(ki+cai)
}


FUNCTION ghk(v(mV), ci(mM), co(mM)) (mV) {
        LOCAL nu,f
        f = KTF(celsius)/2
        nu = v/f
        ghk=-f*(1. - (ci/co)*exp(nu))*efun(nu)
}

FUNCTION KTF(celsius (degC)) (mV) { : temperature-dependent adjustment factor
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

FUNCTION alpm(v(mV)) {
	TABLE FROM -150 TO 150 WITH 200
	alpm = 0.055*(-27.01 - v)/(exp((-27.01-v)/3.8) - 1)
}


FUNCTION betm(v(mV)) {
        TABLE FROM -150 TO 150 WITH 200
        betm =0.94*exp((-63.01-v)/17)
}



PROCEDURE state() {
 LOCAL minus_m
 rate(v-dv)
 minus_m = m + (1 - exp(-dt/taum)) * (minf - m)
  
	rate(v + dv)
    mnew = m + (1 - exp(-dt/taum)) * (minf - m)
	rate(v)
    m = m + (1 - exp(-dt/taum)) * (minf - m)
	dmdm_ = exp(-dt/taum)
    UNITSOFF
    dmdv = (mnew - minus_m)/(2*dv)
UNITSON
}

PROCEDURE rate(v (mV)) { :callable from hoc
        LOCAL a, b, qt
        a = alpm(v)
        taum = 1/(tfa*(a+betm(v))) : estimation of activation tau
        minf = a/(a+betm(v))       : estimation of activation steady state value
        
	
}
