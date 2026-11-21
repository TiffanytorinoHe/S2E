from neuron import h
import numpy as np
from matplotlib import pyplot as plt

rates_bio = [1, 3.0, 1, 0.5, 0.5, 0.02, 0.01, 0.1, 0.5]
#[0.4, 0.5, 0.75, 1.4, 1.1, 0.05, 0.01, 0.2, 1.0]
rates_uni = [1, 1, 1, 1, 1, 1, 1, 1, 1]
class CA1cell:
    def __init__(self, choice = 'area', type = 'uni', axon = True) -> None:
        h.load_file('CA1.hoc')
        self.soma = list(h.soma)
        self.dend = list(h.basal_tree_list)
        self.apic = list(h.apical_trunk_list) + list(h.apical_non_trunk_list)
        self.axon = list(h.axon_sec_list)
        if axon == False:
            for sec in self.axon:
                for seg in sec:
                    seg.gnabar_hha2 = 0
                    seg.gkbar_hha2 = 0
                    seg.gbar_km_zach = 0
                sec.L =0.001
                sec.diam = 1
                sec.nseg = 1
            
            for sec in self.soma:
                for seg in sec:
                    seg.hha2.gnabar = 0.1
                    seg.hha2.gkbar = 0.02
                

            #self.axon = []

        self.all = self.soma + self.dend + self.apic + self.axon
        self.all_no_axon = self.soma + self.dend + self.apic  

        self.basal_tree_list = list(h.basal_tree_list)
        self.apical_trunk_list = list(h.apical_trunk_list)
        self.apical_non_trunk_list = list(h.apical_non_trunk_list)

        self.apical_tip_list_prox = list(h.apical_tip_list)[:-6]
        self.apical_tip_list_dist = list(h.apical_tip_list)[-6:]
        self.peri_trunk_list = list(h.peri_trunk_list)

        self.tuftTerm, self.tuftMid = [],[]
        self.distObliqPrim, self.distObliqTerm = [],[]
        self.proxObliqPrim, self.proxObliqTerm = [],[]
        self.basalPrim, self.basalMid, self.basalTerm = [],[],[]

        for sec in self.basal_tree_list:
            parseg = sec.parentseg()
            nchild = len(sec.subtree()) - 1
            if parseg.sec in self.soma:
                self.basalPrim.append(sec)
            elif nchild == 0:
                self.basalTerm.append(sec)
            else:
                self.basalMid.append(sec)

        for sec in self.apical_trunk_list:
            dist = h.distance(sec(0), self.soma[0](0.5))
            if dist  < 400:
                self.proxObliqPrim.append(sec)
            elif dist > 400:
                self.distObliqPrim.append(sec)
        
        for sec in self.apical_non_trunk_list:
            parseg = sec.parentseg()
            dist = h.distance(parseg.sec(0), self.soma[0](0.5))
            nchild = len(sec.subtree()) - 1
            if dist < 400:
                if nchild == 0:
                    self.proxObliqTerm.append(sec)
                else:
                    self.proxObliqPrim.append(sec)
            elif dist > 600 and sec not in self.peri_trunk_list and parseg.sec not in self.peri_trunk_list:
                if nchild == 0:
                    self.tuftTerm.append(sec)
                else:
                    self.tuftMid.append(sec)

            else:
                if nchild == 0:
                    self.distObliqTerm.append(sec)
                else:
                    self.distObliqPrim.append(sec)

        self.distObliqPrim.remove(h.apical_dendrite[95])
        self.distObliqPrim.remove(h.apical_dendrite[103])
        self.distObliqPrim.remove(h.apical_dendrite[104])
        self.distObliqPrim.remove(h.apical_dendrite[96])
        self.distObliqTerm.remove(h.apical_dendrite[97])
        self.distObliqPrim.remove(h.apical_dendrite[98])
        self.distObliqTerm.remove(h.apical_dendrite[73])

        self.distObliqPrim.remove(h.apical_dendrite[83])
        self.distObliqPrim.remove(h.apical_dendrite[74])

        #self.tuftTerm.remove(h.apical_dendrite[82])
        
        self.tuftTerm.append(h.apical_dendrite[97])
        self.tuftTerm.append(h.apical_dendrite[73])

        self.tuftMid.append(h.apical_dendrite[104])
        self.tuftMid.append(h.apical_dendrite[95])
        self.tuftMid.append(h.apical_dendrite[96])
        self.tuftMid.append(h.apical_dendrite[98])
        self.tuftMid.append(h.apical_dendrite[103])
        self.tuftMid.append(h.apical_dendrite[83])
        self.tuftMid.append(h.apical_dendrite[74])



        self.all_dend = self.tuftTerm + self.tuftMid + self.distObliqPrim + self.distObliqTerm + \
            self.proxObliqTerm + self.proxObliqPrim + self.basalPrim + self.basalMid + self.basalTerm
        
        self.calc_prob(choice = choice, type = type)
        
    def calc_prob(self, choice = 'area', type = 'bio'):
        if type == 'bio':
            rates = rates_bio
        elif type == 'uni':
            rates = rates_uni
        else:
            raise ValueError('type must be either bio or uni')
        
        prob_exc = []
        self.tuftTerm_metric = []
        for sec in self.tuftTerm:
            rate = rates[0]
            if choice == 'area':
                area = 0
                for seg in sec:
                    area += seg.area()  
                prob_exc.append(area * rate)
                self.tuftTerm_metric.append(area)
            else:
                prob_exc.append(sec.L * rate)
                self.tuftTerm_metric.append(sec.L)

        self.tuftMid_metric = []
        for sec in self.tuftMid:
            rate = rates[1]
            if choice == 'area':
                area = 0
                for seg in sec:
                    area += seg.area()  
                prob_exc.append(area * rate)
                self.tuftMid_metric.append(area)
            else:
                prob_exc.append(sec.L * rate)
                self.tuftMid_metric.append(sec.L)

        self.distObliqPrim_metric = []
        for sec in self.distObliqPrim:
            rate = rates[2]
            if choice == 'area':
                area = 0
                for seg in sec:
                    area += seg.area()  
                prob_exc.append(area * rate)
                self.distObliqPrim_metric.append(area)
            else:
                prob_exc.append(sec.L * rate)
                self.distObliqPrim_metric.append(sec.L)
        
        self.distObliqTerm_metric = []
        for sec in self.distObliqTerm:
            rate = rates[3]
            if choice == 'area':
                area = 0
                for seg in sec:
                    area += seg.area()  
                prob_exc.append(area * rate)
                self.distObliqTerm_metric.append(area)
            else:
                prob_exc.append(sec.L * rate)
                self.distObliqTerm_metric.append(sec.L)

        for sec in self.proxObliqTerm:
            rate = rates[4]
            if choice == 'area':
                area = 0
                for seg in sec:
                    area += seg.area()  
                prob_exc.append(area * rate)
            else:
                prob_exc.append(sec.L * rate)

        for sec in self.proxObliqPrim:
            rate = rates[5]
            if choice == 'area':
                area = 0
                for seg in sec:
                    area += seg.area()  
                prob_exc.append(area * rate)
            else:
                prob_exc.append(sec.L * rate)

        for sec in self.basalPrim:
            rate = rates[6]
            if choice == 'area':
                area = 0
                for seg in sec:
                    area += seg.area()  
                prob_exc.append(area * rate)
            else:
                prob_exc.append(sec.L * rate)
        
        for sec in self.basalMid:
            rate = rates[7]
            if choice == 'area':
                area = 0
                for seg in sec:
                    area += seg.area()  
                prob_exc.append(area * rate)
            else:
                prob_exc.append(sec.L * rate)
        
        for sec in self.basalTerm:
            rate = rates[8]
            if choice == 'area':
                area = 0
                for seg in sec:
                    area += seg.area()  
                prob_exc.append(area * rate)
            else:
                prob_exc.append(sec.L * rate)

        prob_exc = np.array(prob_exc)
        prob_exc /= np.sum(prob_exc)
        self.prob_exc = prob_exc

        prob_inh = np.zeros(len(self.all_no_axon))
        for i, sec in enumerate(self.all_no_axon):
            if choice == 'area':
                area = 0
                for seg in sec:
                    area += seg.area()  
                prob_inh[i] = area
            
            elif choice == 'len':
                prob_inh[i] = sec.L

        self.prob_inh = prob_inh / np.sum(prob_inh)


        
