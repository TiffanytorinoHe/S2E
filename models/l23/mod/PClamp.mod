NEURON {
	POINT_PROCESS PClamp
	RANGE del, dur, i, amp
	NONSPECIFIC_CURRENT i
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
}

ASSIGNED {
	v (mV)
	i (nA)
  g (uS)
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
	i = 0
}

