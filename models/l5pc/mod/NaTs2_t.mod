:Reference :Colbert and Pan 2002
:comment: took the NaTa and shifted both activation/inactivation by 6 mv

NEURON	{
	SUFFIX NaTs2_t
	USEION na READ ena WRITE ina
	RANGE gNaTs2_tbar, gNaTs2_t, ina, dv, dgdm, dmdm_, dmdv, dgdh, dhdh_, dhdv, ena
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gNaTs2_tbar = 0.00001 (S/cm2)
	dv = 1e-4 (mV)
}

ASSIGNED	{
	v	(mV)
	ena	(mV)
	ina	(mA/cm2)

	gNaTs2_t	(S/cm2)

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
	gNaTs2_t = gNaTs2_tbar*m*m*m*h
	ina = gNaTs2_t*(v-ena)
	dgdm = gNaTs2_tbar*3*m*m*h
    dgdh = gNaTs2_tbar*m*m*m
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
  LOCAL qt
  qt = 2.3^((34-21)/10)

	UNITSOFF
    if(v == -32){
    	v = v+0.0001
    }
		mAlpha = (0.182 * (v- -32))/(1-(exp(-(v- -32)/6)))
		mBeta  = (0.124 * (-v -32))/(1-(exp(-(-v -32)/6)))
		mInf = mAlpha/(mAlpha + mBeta)
		mTau = (1/(mAlpha + mBeta))/qt

    if(v == -60){
      v = v + 0.0001
    }
		hAlpha = (-0.015 * (v- -60))/(1-(exp((v- -60)/6)))
		hBeta  = (-0.015 * (-v -60))/(1-(exp((-v -60)/6)))
		hInf = hAlpha/(hAlpha + hBeta)
		hTau = (1/(hAlpha + hBeta))/qt
	UNITSON
}