TITLE Slow Ca-dependent potassium current
                            :
                            :   Ca++ dependent K+ current IC responsible for slow AHP
                            :   Differential equations
                            :
                            :   Model based on a first order kinetic scheme
                            :
                            :       + n cai <->     (alpha,beta)
                            :
                            :   Following this model, the activation fct will be half-activated at 
                            :   a concentration of Cai = (beta/alpha)^(1/n) = cac (parameter)
                            :
                            :   The mod file is here written for the case n=2 (2 binding sites)
                            :   ---------------------------------------------
                            :
                            :   This current models the "slow" IK[Ca] (IAHP): 
                            :      - potassium current
                            :      - activated by intracellular calcium
                            :      - NOT voltage dependent
                            :
                            :   A minimal value for the time constant has been added
                            :
                            :   Ref: Destexhe et al., J. Neurophysiology 72: 803-818, 1994.
                            :   See also: http://www.cnl.salk.edu/~alain , http://cns.fmed.ulaval.ca
                            :   modifications by Yiota Poirazi 2001 (poirazi@LNC.usc.edu)
			    :   taumin = 0.5 ms instead of 0.1 ms	

                            NEURON {
                                    SUFFIX kca
                                    USEION k READ ek WRITE ik
                                    USEION ca READ cai
                                    RANGE cac, gk, gbar, m_inf, tau_m,ik
                                    RANGE dv, dc, dgdm, dmdm_, dmdv, dmdc, ek
                                    GLOBAL beta
                            }


                            UNITS {
                                    (mA) = (milliamp)
                                    (mV) = (millivolt)
                                    (molar) = (1/liter)
                                    (mM) = (millimolar)
                            }


                            PARAMETER {
                                    v               (mV)
                                    celsius = 36    (degC)
                                    ek      = -80   (mV)
                                    cai     = 2.4e-5 (mM)           : initial [Ca]i
                                    gbar    = 0.01   (mho/cm2)
                                    beta    = 0.03   (1/ms)          : backward rate constant
                                    cac     = 0.025  (mM)            : middle point of activation fct
       				    taumin  = 0.5    (ms)            : minimal value of the time cst
                                    gk
                                    dv = 1e-4 (mV)
		                    dc = 1e-9 (mM)
                                  }


                            STATE {m mnew_v mnew_c}        : activation variable to be solved in the DEs       

                            ASSIGNED {       : parameters needed to solve DE 
                                    ik      (mA/cm2)
                                    m_inf
                                    tau_m   (ms)
                                    tadj
                                        dmdv
                                        dmdc
                                        dmdm_
                                        dgdm
        
                            }
                            BREAKPOINT { 
                                    SOLVE states :METHOD cnexp
                                    gk = gbar*m*m*m     : maximum channel conductance
                                    ik = gk*(v - ek)    : potassium current induced by this channel
                            
                                    dgdm = gbar*3*m*m
                            }

                            PROCEDURE states() {
 LOCAL minus_v,minus_c
 evaluate_fct(v-dv,cai)
 minus_v=m + (1 - exp(-dt/tau_m))*(m_inf - m)
 evaluate_fct(v,cai-dc)
 minus_c=m + (1 - exp(-dt/tau_m))*(m_inf - m)
 
                                    evaluate_fct(v + dv,cai)
                                    mnew_v = m + (1 - exp(-dt/tau_m))*(m_inf - m)
                                evaluate_fct(v, cai + dc)
                                mnew_c = m + (1 - exp(-dt/tau_m))*(m_inf - m)
                                evaluate_fct(v, cai)
                                m = m + (1 - exp(-dt/tau_m))*(m_inf - m)

                                dmdm_ = exp(-dt/tau_m)

                                UNITSOFF
                                dmdv = (mnew_v-minus_v)/(2*dv)
                                dmdc = (mnew_c-minus_c)/(2*dc)
                                UNITSON
                                :m' = (m_inf - m) / tau_m
                            }

                            UNITSOFF
                            INITIAL {
                            :
                            :  activation kinetics are assumed to be at 22 deg. C
                            :  Q10 is assumed to be 3
                            :
                                    tadj = 3 ^ ((celsius-22.0)/10) : temperature-dependent adjastment factor
                                    evaluate_fct(v,cai)
                                    m = m_inf
                            }

                            PROCEDURE evaluate_fct(v(mV),cai(mM)) {  LOCAL car
                                    car = (cai/cac)^2
                                    m_inf = car / ( 1 + car )      : activation steady state value
                                    tau_m =  1 / beta / (1 + car) / tadj
                                    if(tau_m < taumin) { tau_m = taumin }   : activation min value of time cst
                            }
                            UNITSON
