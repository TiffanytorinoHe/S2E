:Comment :
:Reference : :		Characterization of a Shaw-related potassium channel family in rat brain, The EMBO Journal, vol.11, no.7,2473-2486 (1992)

NEURON	{
	SUFFIX SKv3_1
	USEION k READ ek WRITE ik
	RANGE gSKv3_1bar, gSKv3_1, ik, dv, dgdm, dmdm_, dmdv, ek
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gSKv3_1bar = 0.00001 (S/cm2) 
	dv = 1e-4 (mV)
}

ASSIGNED	{
	v	(mV)
	ek	(mV)
	ik	(mA/cm2)

	gSKv3_1	(S/cm2)

	mInf
	mTau
	dgdm  dmdm_  dmdv
}

STATE	{ 
	m
	mnew
}

BREAKPOINT	{
	SOLVE states :METHOD cnexp
	gSKv3_1 = gSKv3_1bar*m
	ik = gSKv3_1*(v-ek)
	dgdm = gSKv3_1bar
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
	UNITSOFF
		mInf =  1/(1+exp(((v -(18.700))/(-9.700))))
		mTau =  0.2*20.000/(1+exp(((v -(-46.560))/(-44.140))))
	UNITSON
}
