:Comment :
:Reference : :		Kole,Hallermann,and Stuart, J. Neurosci. 2006

NEURON	{
	SUFFIX Ih
	NONSPECIFIC_CURRENT ihcn
	RANGE gIhbar, gIh, ihcn, dv, dgdm, dmdm_, dmdv, ehcn
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gIhbar = 0.00001 (S/cm2) 
	ehcn =  -45.0 (mV)
	dv = 1e-4 (mV)
}

ASSIGNED	{
	v	(mV)
	ihcn	(mA/cm2)

	gIh	(S/cm2)

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
	gIh = gIhbar*m
	ihcn = gIh*(v-ehcn)
	dgdm = gIhbar

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
        if(v == -154.9){
            v = v + 0.0001
        }
		mAlpha =  0.001*6.43*(v+154.9)/(exp((v+154.9)/11.9)-1)
		mBeta  =  0.001*193*exp(v/33.1)
		mInf = mAlpha/(mAlpha + mBeta)
		mTau = 1/(mAlpha + mBeta)
	UNITSON
}
