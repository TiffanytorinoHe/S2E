TITLE hh.mod   squid sodium, potassium, and leak channels
 
COMMENT
 This is the original Hodgkin-Huxley treatment for the set of sodium, 
  potassium, and leakage channels found in the squid giant axon membrane.
  ("A quantitative description of membrane current and its application 
  conduction and excitation in nerve" J.Physiol. (Lond.) 117:500-544 (1952).)
 Membrane voltage is in absolute mV and has been reversed in polarity
  from the original HH convention and shifted to reflect a resting potential
  of -65 mV.
 Remember to set celsius=6.3 (or whatever) in your HOC file.
 See squid.hoc for an example of a simulation using this model.
 SW Jaslove  6 March, 1992
ENDCOMMENT
 
UNITS {
        (mA) = (milliamp)
        (mV) = (millivolt)
	(S) = (siemens)
}

? interface
NEURON {
        SUFFIX hh_grad
        REPRESENTS NCIT:C17145   : sodium channel
        REPRESENTS NCIT:C17008   : potassium channel
        USEION na READ ena WRITE ina REPRESENTS CHEBI:29101
        USEION k READ ek WRITE ik REPRESENTS CHEBI:29103
        NONSPECIFIC_CURRENT il
        RANGE gnabar, gkbar, gl, el, gna, gk, dv, dgdm, dmdm_, dmdv, dgdh, dhdh_, dhdv, dgdn, dndn_,dndv
        RANGE didv, ina, ik
        : `GLOBAL minf` will be replaced with `RANGE minf` if CoreNEURON enabled
        GLOBAL minf, hinf, ninf, mtau, htau, ntau
	THREADSAFE : assigned GLOBALs will be per thread
}
 
PARAMETER {
        gnabar = .12 (S/cm2)	<0,1e9>
        gkbar = .036 (S/cm2)	<0,1e9>
        gl = .0003 (S/cm2)	<0,1e9>
        el = -54.3 (mV)
        dv = 1e-4 (mV)
}
 
STATE {
        m h n mnew hnew nnew
}
 
ASSIGNED {
        v (mV)
        celsius (degC)
        ena (mV)
        ek (mV)

	gna (S/cm2)
	gk (S/cm2)
	gna_new (S/cm2)
	gk_new (S/cm2)

        ina (mA/cm2)
        ik (mA/cm2)
        il (mA/cm2)
        ina_new (mA/cm2)
        ik_new (mA/cm2)
        
        minf hinf ninf
	mtau (ms) htau (ms) ntau (ms)
        dgdm  dmdm_  dmdv
        dgdh  dhdh_  dhdv
        dgdn  dndn_  dndv

        didv (S/cm2)
}
 
? currents
BREAKPOINT {

        SOLVE states :METHOD cnexp
        gna = gnabar*m*m*m*h
	ina = gna*(v - ena)
        gk = gkbar*n*n*n*n
	ik = gk*(v - ek)      
        il = gl*(v - el)

        dgdm = gnabar*3*m*m*h
        dgdh = gnabar*m*m*m
        dgdn = gkbar*4*n*n*n

        gna_new = gnabar*mnew*mnew*mnew*hnew
	ina_new = gna_new*(v + dv- ena)
        gk_new = gkbar*nnew*nnew*nnew*nnew
	ik_new = gk_new*(v + dv -ek)    

        didv = (ina_new+ik_new-ina-ik)/dv    
}
 
 
INITIAL {
	rates(v)
	m = minf
	h = hinf
	n = ninf
        mnew = m
        hnew = h
        nnew = n
}

? states
PROCEDURE states() {
    LOCAL minus_m, minus_h, minus_n
    rates(v - dv)
    minus_m = m + (1 - exp(-dt/mtau)) * (minf - m)
    minus_h = h + (1 - exp(-dt/htau)) * (hinf - h)
    minus_n = n + (1 - exp(-dt/ntau)) * (ninf - n)

        rates(v+dv)
        mnew = m + (1 - exp(-dt/mtau)) * (minf - m)
	hnew = h + (1 - exp(-dt/htau)) * (hinf - h)
        nnew = n + (1 - exp(-dt/ntau)) * (ninf - n)

        rates(v)
        m = m + (1 - exp(-dt/mtau)) * (minf - m)
	h = h + (1 - exp(-dt/htau)) * (hinf - h)
        n = n + (1 - exp(-dt/ntau)) * (ninf - n)

        dmdm_ = exp(-dt/mtau)
        dhdh_ = exp(-dt/htau)
        dndn_ = exp(-dt/ntau)
        
        UNITSOFF
        dmdv = (mnew - minus_m)/(2*dv)
        dhdv = (hnew - minus_h)/(2*dv)
        dndv = (nnew - minus_n)/(2*dv)
        UNITSON      
}

:LOCAL q10


? rates
PROCEDURE rates(v(mV)) {  :Computes rate and other constants at current v.
                      :Call once from HOC to initialize inf at resting v.
        LOCAL  alpha, beta, sum, q10
        : `TABLE minf` will be replaced with `:TABLE minf` if CoreNEURON enabled)
        TABLE minf, mtau, hinf, htau, ninf, ntau DEPEND celsius FROM -100 TO 100 WITH 200

UNITSOFF
        q10 = 3^((celsius - 6.3)/10)
                :"m" sodium activation system
        alpha = .1 * vtrap(-(v+40),10)
        beta =  4 * exp(-(v+65)/18)
        sum = alpha + beta
	mtau = 1/(q10*sum)
        minf = alpha/sum
                :"h" sodium inactivation system
        alpha = .07 * exp(-(v+65)/20)
        beta = 1 / (exp(-(v+35)/10) + 1)
        sum = alpha + beta
	htau = 1/(q10*sum)
        hinf = alpha/sum
                :"n" potassium activation system
        alpha = .01*vtrap(-(v+55),10) 
        beta = .125*exp(-(v+65)/80)
	sum = alpha + beta
        ntau = 1/(q10*sum)
        ninf = alpha/sum
}
 
FUNCTION vtrap(x,y) {  :Traps for 0 in denominator of rate eqns.
        if (fabs(x/y) < 1e-6) {
                vtrap = y*(1 - x/y/2)
        }else{
                vtrap = x/(exp(x/y) - 1)
        }
}
 
UNITSON
