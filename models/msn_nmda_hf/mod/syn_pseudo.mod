NEURON {
	POINT_PROCESS Syn_Pseudo
	RANGE delay, dur, i, amp, e
	NONSPECIFIC_CURRENT i
    RANGE g, pure_i, didv, djdv, w, id
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(uS) = (microsiemens)
}

PARAMETER {
  delay (ms)
	dur (ms)	<0,1e9>
  amp (uS)
  e (mV)
  w 
  id
}

ASSIGNED {
	v (mV)
	i (nA)
	g (uS)
	pure_i
    didv
	djdv
}

INITIAL {
	g=0
}

BREAKPOINT {
	at_time(delay)
	at_time(delay+dur)

	if (t < delay + dur && t >= delay) {
		pure_i = amp
	}else{
		pure_i = 0
	}
	g = 0
	i = 0
  	didv = 0
 	djdv = 0
}

NET_RECEIVE(weight (uS)) {

}
