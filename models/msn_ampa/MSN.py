from neuron import h
import numpy as np
from math import exp
import json

def calculate_distribution(d3, dist, a4, a5,  a6,  a7, g8):
    '''
    Used for setting the maximal conductance of a segment.
    Scales the maximal conductance based on somatic distance and distribution type.
    
    Parameters:
    d3   = distribution type:
         0 linear, 
         1 sigmoidal, 
         2 exponential
         3 step function
    dist = somatic distance of segment
    a4-7 = distribution parameters 
    g8   = base conductance (similar to maximal conductance)
    
    '''
    
    if   d3 == 0: 
        value = a4 + a5*dist
    elif d3 == 1: 
        value = a4 + a5/(1 + exp((dist-a6)/a7) )
    elif d3 == 2: 
        value = a4 + a5*exp((dist-a6)/a7)
    elif d3 == 3:
        if (dist > a6) and (dist < a7):
            value = a4
        else:
            value = a5
            
    if value < 0:
        value = 0
        
    value = value*g8
    return value 

            

class MSN:
    def __init__(self,  params=None,                                \
                        morphology='latest_WT-P270-20-14ak.swc'     ):
        Import = h.Import3d_SWC_read()
        Import.input(morphology)
        imprt = h.Import3d_GUI(Import, 0)
        imprt.instantiate(None)
        h.define_shape()
        # h.cao0_ca_ion = 2  # default in nrn
        h.celsius = 35
        self._create_sectionlists()
        self._set_nsegs()
        self.v_init = -80
        
        for sec in self.soma:
            for mech in [
                    "naf_kai",
                    "kaf_kai",
                    "kas_kai",
                    "kdr_kai",
                    "kir_kai",
                    "cal12",
                    "cal13",
                    "can_kai",
                    "car_kai",
                    "cadyn",
                    "caldyn",
                    "sk_kai",
                    "bk"
                ]:
                sec.insert(mech)
                
        for sec in self.axon:
            for mech in [
                    "naf_kai",
                    "kas_kai"
                ]:
                sec.insert(mech)
                
        for sec in self.dend:
            for mech in [
                    "naf_kai",
                    "kaf_kai",
                    "kas_kai",
                    "kdr_kai",
                    "kir_kai",
                    "cal12",
                    "cal13",
                    "car_kai",
                    "cat32",
                    "cat33",
                    "cadyn",
                    "caldyn",
                    "sk_kai",
                    "bk"
                ]:
                sec.insert(mech)
                
        for sec in self.all:
            sec.Ra = 150
            sec.cm = 1.0
            sec.insert('pas')
            #sec.g_pas = 1e-5 # set using json file
            sec.e_pas = -70 # -73
            sec.ena = 50
            sec.ek = -85 # -90

        with open(params) as file:
            par = json.load(file)

        self.distribute_channels("soma", "g_pas", 0, 1, 0, 0, 0, float(par['g_pas_all']['Value']))
        self.distribute_channels("axon", "g_pas", 0, 1, 0, 0, 0, float(par['g_pas_all']['Value']))
        self.distribute_channels("dend", "g_pas", 0, 1, 0, 0, 0, float(par['g_pas_all']['Value']))
        
        self.distribute_channels("soma", "gbar_naf_kai", 0, 1, 0, 0, 0, float(par['gbar_naf_somatic']['Value']))
        self.distribute_channels("soma", "gbar_kaf_kai", 0, 1, 0, 0, 0, float(par['gbar_kaf_somatic']['Value']))
        self.distribute_channels("soma", "gbar_kas_kai", 0, 1, 0, 0, 0, float(par['gbar_kas_somatic']['Value']))
        self.distribute_channels("soma", "gbar_kdr_kai", 0, 1, 0, 0, 0, float(par['gbar_kdr_somatic']['Value']))
        self.distribute_channels("soma", "gbar_kir_kai", 0, 1, 0, 0, 0, float(par['gbar_kir_somatic']['Value']))
        self.distribute_channels("soma", "gbar_sk_kai",  0, 1, 0, 0, 0, float(par['gbar_sk_somatic']['Value']))
        self.distribute_channels("soma", "gbar_bk",  0, 1, 0, 0, 0, float(par['gbar_bk_somatic']['Value']))
        
        self.distribute_channels("axon", "gbar_naf_kai", 3, 1, 1.1, 30, 500, float(par['gbar_naf_axonal']['Value']))
        self.distribute_channels("axon", "gbar_kas_kai", 0, 1, 0, 0, 0,      float(par['gbar_kas_axonal']['Value']))
        self.distribute_channels("dend", "gbar_naf_kai", 1, 0.1, 0.9,   60.0,   10.0, float(par['gbar_naf_basal']['Value']))
        self.distribute_channels("dend", "gbar_kaf_kai", 1,   1, 0.5,  120.0,  -30.0, float(par['gbar_kaf_basal']['Value']))
        
        self.distribute_channels("dend", "gbar_kas_kai", 2,   1, 9.0,  0.0, -5.0, float(par['gbar_kas_basal']['Value']))
        self.distribute_channels("dend", "gbar_kdr_kai", 0, 1, 0, 0, 0, float(par['gbar_kdr_basal']['Value']))
        self.distribute_channels("dend", "gbar_kir_kai", 0, 1, 0, 0, 0, float(par['gbar_kir_basal']['Value']))
        self.distribute_channels("dend", "gbar_sk_kai",  0, 1, 0, 0, 0, float(par['gbar_sk_basal']['Value']))
        self.distribute_channels("dend", "gbar_bk",  0, 1, 0, 0, 0, float(par['gbar_bk_basal']['Value']))

        self.distribute_channels("soma", "pbar_cal12", 0, 1, 0, 0, 0, 1e-5)
        self.distribute_channels("soma", "pbar_cal13", 0, 1, 0, 0, 0, 1e-6)
        self.distribute_channels("soma", "pbar_car_kai",   0, 1, 0, 0, 0, 1e-4)
        self.distribute_channels("soma", "pbar_can_kai",   0, 1, 0, 0, 0, 3e-5)
        self.distribute_channels("dend", "pbar_cal12", 0, 1, 0, 0, 0, 1e-5)
        self.distribute_channels("dend", "pbar_cal13", 0, 1, 0, 0, 0, 1e-6)
        self.distribute_channels("dend", "pbar_car_kai",   0, 1, 0, 0, 0, 1e-4)
        self.distribute_channels("dend", "pbar_cat32", 1, 0, 1.0, 120.0, -30.0, 1e-7)
        self.distribute_channels("dend", "pbar_cat33", 1, 0, 1.0, 120.0, -30.0, 1e-8)
    
        #self.set_pointer()

    def _create_sectionlists(self):
        self.nsomasec = 0
        self.soma = []
        for sec in h.allsec():
            if sec.name().find('soma') >= 0:
                self.soma.append(sec)
                self.nsomasec += 1

        self.axon = []
        for sec in h.allsec():
            if sec.name().find('axon') >= 0:
                self.axon.append(sec)

        self.dend = []
        for sec in h.allsec():
            if sec.name().find('dend') >= 0:
                self.dend.append(sec)

        self.all = self.soma + self.dend + self.axon
        self.all_no_axon = self.soma + self.dend
    

    def _set_nsegs(self):
        for sec in self.all:
            sec.nseg = 2*int(sec.L/40.0)+1
        for sec in self.axon:
            sec.nseg = 2  # two segments in axon initial segment
    
    
    def distribute_channels(self, as1, as2, d3, a4, a5, a6, a7, g8):
        h.distance(sec=self.soma[0])
        
        for sec in self.all:
            
            # if right cellular compartment (axon, soma or dend)
            if sec.name().find(as1) >= 0:
                for seg in sec:
                    dist = h.distance(seg.x, sec=sec) - 7.06 + 5.6
                    val = calculate_distribution(d3, dist, a4, a5, a6, a7, g8)
                    cmd = 'seg.%s = %g' % (as2, val)
                    exec(cmd)

            else:
                print()


