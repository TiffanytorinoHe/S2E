NEURON {
	POINT_PROCESS NMDA_grad
	RANGE del, e, i, mg, gmax, dv, didv
  GLOBAL tau1,tau2,mg0,v0
	NONSPECIFIC_CURRENT i
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
  (mM) = (milli/liter)
	(uS) = (microsiemens)
}

PARAMETER {
  del (ms)
  mg (mM)
  gmax (uS)
	e = 0	(mV)
  tau1=2.04 (ms)
  tau2=75.2 (ms)
  mg0=3.57 (mM)
  v0=12.5 (mV)
  dv=0.001 (mV)
}

ASSIGNED {
	v (mV)
	i (nA)
  g (uS)
  didv (uS)
}

BREAKPOINT {
  LOCAL  i_new, g_new
  if (gmax) { at_time(del) }
  g=gmax*alpha(t-del)*beta(v,mg)
	i = g * (v - e)
  g_new=gmax*alpha(t-del)*beta(v+dv,mg)
  i_new = g_new * (v + dv -e)
  didv = (i_new - i)/dv
}

FUNCTION alpha(x) {
	if (x < 0) {
		alpha = 0
	}else{
		alpha = 1.13665 * (exp(-x/tau2)-exp(-x/tau1))
	}
}

FUNCTION beta(y,z) {
  beta=1/(1+exp(-y/v0)*z/mg0)
}