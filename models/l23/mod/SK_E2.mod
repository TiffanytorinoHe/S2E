: SK-type calcium-activated potassium current
: Reference : Kohler et al. 1996

NEURON {
       SUFFIX SK_E2
       USEION k READ ek WRITE ik
       USEION ca READ cai
       RANGE gSK_E2bar, gSK_E2, ik, dc, dgdz, dzdz_, dzdc
}

UNITS {
      (mV) = (millivolt)
      (mA) = (milliamp)
      (mM) = (milli/liter)
}

PARAMETER {
          v            (mV)
          gSK_E2bar = .000001 (mho/cm2)
          zTau = 1              (ms)
          ek           (mV)
          cai          (mM)
          dc = 1e-6 (mM)
} 

ASSIGNED {
         zInf
         ik            (mA/cm2)
         ik_new        (mA/cm2)
         gSK_E2	     (S/cm2)
         gSK_E2_new    (S/cm2)
         
         dgdz  dzdz_  dzdc
}

STATE {
      z   FROM 0 TO 1
      z_new FROM 0 TO 1
}

BREAKPOINT {
           SOLVE states :METHOD cnexp
           gSK_E2  = gSK_E2bar * z
           ik   =  gSK_E2 * (v - ek)
           dgdz = gSK_E2bar
}

PROCEDURE states() {
        
      rates(cai + dc)
      z_new = z + (1 - exp(-dt/zTau)) * (zInf - z)
        
      rates(cai)
      z = z + (1 - exp(-dt/zTau)) * (zInf - z)

      dzdz_ =  exp(-dt/zTau)
      
      UNITSOFF
      dzdc = (z_new - z)/dc
	UNITSON

}

PROCEDURE rates(ca(mM)) {
          if(ca < 1e-7){
	              ca = ca + 1e-07
          }
          zInf = 1/(1 + (0.00043 / ca)^4.8)
}

INITIAL {
        rates(cai)
        z = zInf
}
