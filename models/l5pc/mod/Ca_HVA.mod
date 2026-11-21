:Comment :
:Reference : :		Reuveni, Friedman, Amitai, and Gutnick, J.Neurosci. 1993

NEURON	{
	SUFFIX Ca_HVA
	USEION ca READ eca WRITE ica
	RANGE gCa_HVAbar, gCa_HVA, ica , dv, dgdm, dmdm_, dmdv, dgdh, dhdh_, dhdv, eca
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gCa_HVAbar = 0.00001 (S/cm2) 
	dv = 1e-4 (mV)
}

ASSIGNED	{
	v	(mV)
	eca	(mV)
	ica	(mA/cm2)

	gCa_HVA	(S/cm2)

	mInf
	mTau
	mAlpha
	mBeta
	hInf
	hTau
	hAlpha
	hBeta
    
	dgdm  dmdm_  dmdv
    dgdh  dhdh_  dhdv
}

STATE	{ 
	m
	h
	mnew
	hnew
}

BREAKPOINT	{
	SOLVE states :METHOD cnexp
	gCa_HVA = gCa_HVAbar*m*m*h
	ica = gCa_HVA*(v-eca)

	dgdm = gCa_HVAbar*2*m*h
    dgdh = gCa_HVAbar*m*m
	
}

PROCEDURE states()	{
    LOCAL minus_m, minus_h
    rates(v - dv)
    minus_m = m + (1 - exp(-dt/mTau)) * (mInf - m)
    minus_h = h + (1 - exp(-dt/hTau)) * (hInf - h)

	rates(v + dv)
    mnew = m + (1 - exp(-dt/mTau)) * (mInf - m)
	hnew = h + (1 - exp(-dt/hTau)) * (hInf - h)

	rates(v)
    m = m + (1 - exp(-dt/mTau)) * (mInf - m)
	h = h + (1 - exp(-dt/hTau)) * (hInf - h)

    dmdm_ = exp(-dt/mTau)
    dhdh_ = exp(-dt/hTau)

    UNITSOFF
    dmdv = (mnew - minus_m)/(2*dv)
    dhdv = (hnew - minus_h)/(2*dv)
	UNITSON
}

INITIAL{
	rates(v)
	m = mInf
	h = hInf
}

PROCEDURE rates(v(mV)){
	UNITSOFF
        if((v == -27) ){        
            v = v+0.0001
        }
		mAlpha =  (0.055*(-27-v))/(exp((-27-v)/3.8) - 1)        
		mBeta  =  (0.94*exp((-75-v)/17))
		mInf = mAlpha/(mAlpha + mBeta)
		mTau = 1/(mAlpha + mBeta)
		hAlpha =  (0.000457*exp((-13-v)/50))
		hBeta  =  (0.0065/(exp((-v-15)/28)+1))
		hInf = hAlpha/(hAlpha + hBeta)
		hTau = 1/(hAlpha + hBeta)
	UNITSON
}
 