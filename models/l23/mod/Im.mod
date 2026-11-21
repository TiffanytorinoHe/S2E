:Reference : :		Adams et al. 1982 - M-currents and other potassium currents in bullfrog sympathetic neurones
:Comment: corrected rates using q10 = 2.3, target temperature 34, orginal 21

NEURON	{
	SUFFIX Im
	USEION k READ ek WRITE ik
	RANGE gImbar, gIm, ik, dv, dgdm, dmdm_, dmdv, ek
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gImbar = 0.00001 (S/cm2) 
	dv = 1e-4 (mV)
}

ASSIGNED	{
	v	(mV)
	ek	(mV)
	ik	(mA/cm2)

	gIm	(S/cm2)

	mInf
	mTau
	mAlpha
	mBeta
	dgdm  dmdm_  dmdv
}

STATE	{ 
	m
	mnew
}

BREAKPOINT	{
	SOLVE states :METHOD cnexp
	gIm = gImbar*m
	ik = gIm*(v-ek)
	dgdm = gImbar
}

PROCEDURE states()	{
    LOCAL minus_m
    rates(v - dv)
    minus_m = m + (1 - exp(-dt/mTau)) * (mInf - m)

	rates(v + dv)
    mnew = m + (1 - exp(-dt/mTau)) * (mInf - m)

	rates(v)
    m = m + (1 - exp(-dt/mTau)) * (mInf - m)

    dmdm_ = exp(-dt/mTau)

    UNITSOFF
    dmdv = (mnew - minus_m)/(2*dv)
	UNITSON
}


INITIAL{
	rates(v)
	m = mInf
}

PROCEDURE rates(v(mV)){
  LOCAL qt
  qt = 2.3^((34-21)/10)

	UNITSOFF
		mAlpha = 3.3e-3*exp(2.5*0.04*(v - -35))
		mBeta = 3.3e-3*exp(-2.5*0.04*(v - -35))
		mInf = mAlpha/(mAlpha + mBeta)
		mTau = (1/(mAlpha + mBeta))/qt
	UNITSON
}
