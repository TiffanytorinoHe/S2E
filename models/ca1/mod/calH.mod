TITLE Ca L-type channel with high treshold of activation
: inserted in distal dendrites to account for distally
: restricted initiation of Ca++ spikes
: uses channel conductance (not permeability)
: written by Yiota Poirazi, 1/8/00 poirazi@LNC.usc.edu

NEURON {
	SUFFIX calH
	USEION ca READ eca WRITE ica
        RANGE gcal,gcalbar, m, h,ica
	RANGE dv, dgdm, dmdm_, dmdv, dgdh, dhdh_, dhdv, eca
	RANGE inf, fac, tau
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

PARAMETER {          : parameters that can be entered when function is called in cell-setup
        v               (mV)
        celsius = 34	(degC)
	dt              (ms)
        gcalbar = 0     (mho/cm2) : initialized conductance
	gcal            (mho/cm2) : initialized conductance
	eca = 140       (mV)      : Ca++ reversal potential
	dv = 1e-4 (mV)
        }

STATE {	m h mnew hnew }                     : unknown activation and inactivation parameters to be solved in the DEs  

ASSIGNED {                        : parameters needed to solve DE
	ica (mA/cm2)
        inf[2]
		tau[2]

    dgdm  dmdm_  dmdv
    dgdh  dhdh_  dhdv        
}


INITIAL {
        m = 0    : initial activation parameter value
	h = 1    : initial inactivation parameter value
        rate(v)
	
}

BREAKPOINT {
	SOLVE state :METHOD cnexp
	ica = gcalbar*m*m*m*h*(v - eca)
	gcal = gcalbar*m*m*m*h

	dgdm = gcalbar*3*m*m*h
	dgdh = gcalbar*m*m*m

}



PROCEDURE state() {
 LOCAL minus_m,minus_h
 rate(v-dv)
 minus_m = m + (1 - exp(-dt/tau[0])) * (inf[0] - m)
 minus_h = h + (1 - exp(-dt/tau[1])) * (inf[1] - h)
  
    rate(v+dv)
    mnew = m + (1 - exp(-dt/tau[0])) * (inf[0] - m)
	hnew = h + (1 - exp(-dt/tau[1])) * (inf[1] - h)		

	rate(v)
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

PROCEDURE rate(v (mV)) { :callable from hoc
       FROM i=0 TO 1 {
		tau[i] = vartau(v,i)
		inf[i] = varss(v,i)
	}

     
	
}


FUNCTION varss(v, i) {
	if (i==0) { 
             varss = 1 / (1 + exp((v+37)/(-1)))  : Ca activation 
	}
	else if (i==1) { 
             varss = 1 / (1 + exp((v+41)/(0.5))) : Ca inactivation 
	}
}

FUNCTION vartau(v, i) {
	if (i==0) {
           vartau = 3.6  : activation variable time constant
        }
	else if (i==1) {
:           vartau = 25   : inactivation variable time constant
           vartau = 29   : inactivation variable time constant
        }
}	


