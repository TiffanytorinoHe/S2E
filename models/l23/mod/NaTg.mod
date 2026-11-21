:Reference :Colbert and Pan 2002

NEURON	{
	SUFFIX NaTg
	USEION na READ ena WRITE ina
	RANGE gNaTgbar, gNaTg, ina, vshifth, vshiftm, slopeh, slopem
    RANGE dv, dgdm, dmdm_, dmdv, dgdh, dhdh_, dhdv, ena
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gNaTgbar = 0.00001 (S/cm2)
	vshifth = 0 (mV)
	vshiftm = 0 (mV)
	slopeh = 6
	slopem = 6
    dv = 1e-4 (mV)
}

ASSIGNED	{
	v	(mV)
	ena	(mV)
	ina	(mA/cm2)
	gNaTg	(S/cm2)
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
	gNaTg = gNaTgbar*m*m*m*h
	ina = gNaTg*(v-ena)
	dgdm = gNaTgbar*3*m*m*h
    dgdh = gNaTgbar*m*m*m
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
    if(v == -38){
    	v = v+0.0001
    }
		mAlpha = (0.182 * (v- (-38+vshiftm)))/(1-(exp(-(v- (-38+vshiftm))/slopem)))
		mBeta  = (0.124 * (-v + (-38+vshiftm)))/(1-(exp(-(-v + (-38+vshiftm))/slopem)))
		mTau = (1/(mAlpha + mBeta))/qt
		mInf = mAlpha/(mAlpha + mBeta)

    if(v == -66){
      v = v + 0.0001
    }

		hAlpha = (-0.015 * (v- (-66+vshifth)))/(1-(exp((v- (-66+vshifth))/slopeh)))
		hBeta  = (-0.015 * (-v +(-66+vshifth)))/(1-(exp((-v +(-66+vshifth))/slopeh)))
		hTau = (1/(hAlpha + hBeta))/qt
		hInf = hAlpha/(hAlpha + hBeta)
	UNITSON
}