:Comment : mtau deduced from text (said to be 6 times faster than for NaTa)
:Comment : so I used the equations from NaT and multiplied by 6
:Reference : Modeled according to kinetics derived from Magistretti & Alonso 1999
:Comment: corrected rates using q10 = 2.3, target temperature 34, orginal 21

NEURON	{
	SUFFIX Nap_Et2
	USEION na READ ena WRITE ina
	RANGE gNap_Et2bar, gNap_Et2, ina, dv, dgdm, dmdm_, dmdv, dgdh, dhdh_, dhdv, ena
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gNap_Et2bar = 0.00001 (S/cm2)
	dv = 1e-4 (mV)
}

ASSIGNED	{
	v	(mV)
	ena	(mV)
	ina	(mA/cm2)

	gNap_Et2	(S/cm2)

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
	gNap_Et2 = gNap_Et2bar*m*m*m*h
	ina = gNap_Et2*(v-ena)
	
	dgdm = gNap_Et2bar*3*m*m*h
    dgdh = gNap_Et2bar*m*m*m
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
		mInf = 1.0/(1+exp((v- -52.6)/-4.6))
    if(v == -38){
    	v = v+0.0001
    }
		mAlpha = (0.182 * (v- -38))/(1-(exp(-(v- -38)/6)))
		mBeta  = (0.124 * (-v -38))/(1-(exp(-(-v -38)/6)))
		mTau = 6*(1/(mAlpha + mBeta))/qt

  	if(v == -17){
   		v = v + 0.0001
  	}
    if(v == -64.4){
      v = v+0.0001
    }

		hInf = 1.0/(1+exp((v- -48.8)/10))
    hAlpha = -2.88e-6 * (v + 17) / (1 - exp((v + 17)/4.63))
    hBeta = 6.94e-6 * (v + 64.4) / (1 - exp(-(v + 64.4)/2.63))
		hTau = (1/(hAlpha + hBeta))/qt
	UNITSON
}