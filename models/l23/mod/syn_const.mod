NEURON {
	POINT_PROCESS Syn_Const
	RANGE del, dur, i, amp, e
	NONSPECIFIC_CURRENT i
    RANGE g, dv, didv,  w, pure_i, djdv
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(uS) = (microsiemens)
}

PARAMETER {
  del (ms)
	dur (ms)	<0,1e9>
  amp (uS)
  e(mV)
  dv=0.001 (mV)
  w =1 
}

ASSIGNED {
	v (mV)
	i (nA)
  didv (uS)
	g (uS)
	pure_i
	djdv
}

INITIAL {
	g=0
}

BREAKPOINT {
	at_time(del)
	at_time(del+dur)

	if (t < del + dur && t > del) {
		g = amp
	}else{
		g = 0
	}
	pure_i = g*(v - e)
	i = w * pure_i
  	didv = -g
 	djdv = 0
}

NET_RECEIVE(weight (uS)) {

}
