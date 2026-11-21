:Comment : The transient component of the K current
:Reference : :		Voltage-gated K+ channels in layer 5 neocortical pyramidal neurones from young rats:subtypes and gradients,Korngreen and Sakmann, J. Physiology, 2000
:Comment : shifted -10 mv to correct for junction potential
:Comment: corrected rates using q10 = 2.3, target temperature 34, orginal 21

NEURON	{
	SUFFIX K_Tst
	USEION k READ ek WRITE ik
	RANGE gK_Tstbar, gK_Tst, ik, dv, dgdm, dmdm_, dmdv, dgdh, dhdh_, dhdv, ek
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gK_Tstbar = 0.00001 (S/cm2)
	dv = 1e-4 (mV)
}

ASSIGNED	{
	v	(mV)
	ek	(mV)
	ik	(mA/cm2)

	gK_Tst	(S/cm2)

	mInf
	mTau
	hInf
	hTau
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
	gK_Tst = gK_Tstbar*(m^4)*h
	ik = gK_Tst*(v-ek)

	dgdm = 4 * gK_Tstbar * (m^3) * h
	dgdh = gK_Tstbar * (m^4)
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
		v = v + 10
		mInf =  1/(1 + exp(-(v+0)/19))
		mTau =  (0.34+0.92*exp(-((v+71)/59)^2))/qt
		hInf =  1/(1 + exp(-(v+66)/-10))
		hTau =  (8+49*exp(-((v+73)/23)^2))/qt
		v = v - 10
	UNITSON
}
