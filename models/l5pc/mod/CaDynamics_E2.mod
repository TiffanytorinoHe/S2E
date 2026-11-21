: Dynamics that track inside calcium concentration
: modified from Destexhe et al. 1994

NEURON	{
	SUFFIX CaDynamics_E2
	USEION ca READ ica WRITE cai
	RANGE decay, gamma, minCai, depth, di, dcdi,dcdc_
}

UNITS	{
	(mV) = (millivolt)
	(mA) = (milliamp)
	FARADAY = (faraday) (coulombs)
	(molar) = (1/liter)
	(mM) = (millimolar)
	(um)	= (micron)
}

PARAMETER	{
	gamma = 0.05 : percent of free calcium (not buffered)
	decay = 80 (ms) : rate of removal of calcium
	depth = 0.1 (um) : depth of shell
	minCai = 1e-4 (mM)
	di = 1e-3 (mA)
}

ASSIGNED	{ica (mA/cm2) dcdi (mM * cm2 /mA) dcdc_ cInf}

STATE	{
	cai (mM)
	cai_new (mM)
}

BREAKPOINT	{
 	SOLVE states :METHOD cnexp 
}

PROCEDURE states()	{
	rates(ica + di)
	cai_new = cai + (1 - exp(-dt/decay)) * (cInf - cai)
	rates(ica)
	cai = cai + (1 - exp(-dt/decay)) * (cInf - cai)

	dcdc_ = exp(-dt/decay)

    UNITSOFF
    :dcdi = (cai_new - cai)/di
	dcdi = (1 - exp(-dt/decay)) * (- (10000)*(gamma/(2*FARADAY*depth)) * decay)
	UNITSON
}

PROCEDURE rates(i(mA)) {
	UNITSOFF
	cInf = minCai - (10000)*(i*gamma/(2*FARADAY*depth)) * decay
	UNITSON
}