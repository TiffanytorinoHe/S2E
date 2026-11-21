TITLE Ca R-type channel with medium threshold for activation
: used in somatic regions. It has lower threshold for activation/inactivation
: and slower activation time constant
: than the same mechanism in dendritic regions
: uses channel conductance (not permeability)
: written by Yiota Poirazi on 3/12/01 poirazi@LNC.usc.edu

NEURON {
	SUFFIX somacar
	USEION ca READ eca WRITE ica
        RANGE gcabar, gca, m, h, ica, dv, dgdm, dmdm_, dmdv, dgdh, dhdh_, dhdv, eca
	RANGE inf, fac, tau, ica
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
(molar) = (1/liter)
	(mM) =	(millimolar)
	FARADAY = (faraday) (coulomb)
	R = (k-mole) (joule/degC)
}



PARAMETER { 
        eca = 140       (mV)      : Ca++ reversal potentia    
        gcabar = 0      (mho/cm2) : initialized conductance
		celsius =34        (degC)

		dv = 1e-4 (mV)
        }


ASSIGNED {      : parameters needed to solve DE
	v               (mV)
 	ica             (mA/cm2)
ecar             (mV)      : Ca++ reversal potential
        inf[2]
	fac[2]
	tau[2]
: cai		(mM)
        :cao		(mM)
	gca (mho/cm2)
    dgdm  dmdm_  dmdv
    dgdh  dhdh_  dhdv
}

STATE {	
	m 
	h 
	mnew
	hnew
} 


INITIAL {
	m = 0    : initial activation parameter value
	h = 1    : initial inactivation parameter value
	rates(v)
	ica = gcabar*m*m*m*h*(v - eca)  : initial Ca++ current value
        }

BREAKPOINT {
	SOLVE states :METHOD cnexp
        :ecar = (1e3) * (R*(celsius+273.15))/(2*FARADAY) * log (cao/cai)
	gca = gcabar*m*m*m*h
	ica = gca*(v - eca)

	dgdm = gcabar*3*m*m*h
	dgdh = gcabar*m*m*m
}


PROCEDURE states() {
 LOCAL minus_m,minus_h
 rates(v-dv)
 minus_m = m + (1 - exp(-dt/tau[0])) * (inf[0] - m)
 minus_h = h + (1 - exp(-dt/tau[1])) * (inf[1] - h)

	rates(v + dv)
    mnew = m + (1 - exp(-dt/tau[0])) * (inf[0] - m)
	hnew = h + (1 - exp(-dt/tau[1])) * (inf[1] - h)

	rates(v)
    m = m + (1 - exp(-dt/tau[0])) * (inf[0] - m)
	h = h + (1 - exp(-dt/tau[1])) * (inf[1] - h)

    dmdm_ = exp(-dt/tau[0])
    dhdh_ = exp(-dt/tau[1])


    UNITSOFF
    dmdv = (mnew - minus_m)/(2*dv)
    dhdv = (hnew - minus_h)/(2*dv)
	UNITSON

	:m' = (inf[0]-m)/tau[0]
	:h' = (inf[1]-h)/tau[1]
}


PROCEDURE rates(v(mV)) {
	FROM i=0 TO 1 {
		tau[i] = vartau(i)
		inf[i] = varss(v,i)
	}
}



FUNCTION varss(v(mV), i) {
	if (i==0) {
	   varss = 1 / (1 + exp((v+60)/(-3))) :Ca activation
	}
	else if (i==1) {
           varss = 1/ (1 + exp((v+62)/(1)))   :Ca inactivation
	}
}

FUNCTION vartau(i) {
	if (i==0) {
           vartau = 100  : activation variable time constant
        }
	else if (i==1) {
           vartau = 5    : inactivation variable time constant
       }
	
}	
