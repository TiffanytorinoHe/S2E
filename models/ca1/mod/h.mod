TITLE  H-current that uses Na ions

NEURON {
	SUFFIX h
        RANGE  gbar,vhalf, K, taun, ninf, g ,ina  
	USEION na READ ena WRITE ina      
RANGE dv, dgdn, dndv, dndn_ , eh
:	NONSPECIFIC_CURRENT i
}

UNITS {
	(um) = (micrometer)
	(mA) = (milliamp)
	(uA) = (microamp)
	(mV) = (millivolt)
	(pmho) = (picomho)
	(mmho) = (millimho)
}



PARAMETER {              : parameters that can be entered when function is called in cell-setup
        dt             (ms)
	v              (mV)
        ena    = 50    (mV)
        eh     = -10   (mV)
	K      = 8.5   (mV)
	gbar   = 0     (mho/cm2)  : initialize conductance to zero
	vhalf  = -90   (mV)       : half potential
		 dv = 1e-6	(mV)
}	


STATE {                : the unknown parameters to be solved in the DEs
	n
	nnew
}

ASSIGNED {             : parameters needed to solve DE
	ina (mA/cm2)
	ninf
	taun (ms)
	g
	dgdn
	dndv
	dndn_
}

        


INITIAL {               : initialize the following parameter using rates()
	rates(v)	
	n = ninf
	g = gbar*n
	ina = g*(v-eh)
}


BREAKPOINT {
	SOLVE states :METHOD cnexp
	g = gbar*n
	ina = g*(v-eh)  

	dgdn = gbar
}

PROCEDURE states() {
 LOCAL old_n, deriv_tau, aa, bb
 old_n=n  
rates(v + dv)
	nnew = n + (1 - exp(-dt/taun))*(ninf - n)
        :n' = (ninf - n)/taun
	rates(v)
	n = n + (1 - exp(-dt/taun))*(ninf - n)

	dndn_ = exp(-dt/taun)
	UNITSOFF
	aa=exp((v+145)/(-17.5))
    bb=exp((v+16.8)/16.5)
    deriv_tau=0
    if (v <= -30) { deriv_tau=-2*((-aa/17.5)+(bb/16.5))/((aa+bb)*(aa+bb)) }
    dndv=-(1-exp(-dt/taun))*ninf*(1-ninf)/K-(ninf-old_n)*exp(-dt/taun)*dt/(taun*taun)*deriv_tau
	UNITSON

}

PROCEDURE rates(v(mV)) {  
 
 	if (v > -30) {
	   taun = 1
	} else {
           taun = 2*(1/(exp((v+145)/-17.5)+exp((v+16.8)/16.5)) + 5) :h activation tau

	}  
         ninf = 1 - (1 / (1 + exp((vhalf - v)/K)))                  :steady state value
}



