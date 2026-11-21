#from pyclbr import Class
from neuron import h
from neuron import gui
import numpy as np
import os
import pickle

import sys
sys.path.append(os.getcwd())

rates_uni = np.array([1,1,1,1,1,1,1,1])
rates_bio = np.array([1.2, 1.5, 1.5, 1.2, 1, 0.5, 0.8, 1])
class L23PC:
    def __init__(self,  neuron_num,  max_segment_len, choice = 'len', type = 'uni'):
        h.load_file('model_files/{}.hoc'.format(neuron_num))
        h('''
        objref somatic, apical, axonal, basal
        somatic = new SectionList()
        basal = new SectionList()
        apical = new SectionList()
        axonal = new SectionList()
        celsius=37

        create axon[2]''')


        h('''
        access axon[0] 
        {
            L = 30                                                            
            diam = 0.66
            //nseg = 30    
            nseg = 1+2*int(L/20)                                  
            all.append()                                                            
            axonal.append()                                                         
        }

        access axon[1] 
        {                                                            
            L = 30                                                                  
            diam = 0.33
            //nseg = 30     
            nseg = 1+2*int(L/20)                                 
            all.append()                                                            
            axonal.append()                                                         
        }                                                                           
        nSecAxonal = 2                                                              
            
        soma connect axon[0](0), 1                                           
        axon[0] connect axon[1](0), 1
        ''')

        h("celsius = 37.0")

        #self.all = list(h.all)

        self.soma, self.dend, self.apic, self.axon = [],[],[],[]
        for sec in h.all:
            if sec.name().startswith('soma'): continue
            #h("{}.nseg = {}".format(sec.name(), max(eval('h.{}.L'.format(sec.name())), 1)))
            sec.nseg = 1+2*int(sec.L/(max_segment_len*2))

        for sec in h.all:
            if sec.name().startswith('dend_5') or sec.name().startswith('apic'):
                h('access {}'.format(sec.name()))
                h('apical.append()')
                self.apic.append(sec)
                h.pop_section()
            
            elif sec.name().startswith('dend_7'):
                h('access {}'.format(sec.name()))
                h('basal.append()')
                self.dend.append(sec)
                h.pop_section()
            
            elif 'soma' in sec.name() or sec.name().startswith('dend_0'):
                h('access {}'.format(sec.name()))
                h('somatic.append()')
                self.soma.append(sec)
                h.pop_section()
            
            elif 'axon' in sec.name():
                h('access {}'.format(sec.name()))
                h('axonal.append()')
                self.axon.append(sec)
                h.pop_section()
            
            else:
                print('no section list for {}'.format(sec.name()))

        self.all = self.soma + self.dend + self.apic + self.axon
        assert len(self.all) == len(list(h.all))
        self.all_no_axon = self.soma + self.dend + self.apic
        self.all_no_soma = self.dend +self.apic
        #self.soma_v = h.Vector().record(self.soma[0](0.5)._ref_v)

        seglist = []
        for sec in h.all:
            for seg in sec:
                seglist.append(seg)
        seg2idx_hoc = {seg: i for i, seg in enumerate(seglist)}
        segment_domain_type = np.load('./model_files_%d/segment_domain_type_%d.npy'%(max_segment_len, neuron_num))
        idx_not_assigned = np.where(segment_domain_type == -1)[0]
        if len(idx_not_assigned) > 0:
            segdist = np.zeros((len(seglist), len(seglist)))
            for i, seg in enumerate(seglist):
                for j, seg2 in enumerate(seglist):
                    segdist[i, j] = h.distance(seg, seg2)
                
            for idx in idx_not_assigned:
                seg = seglist[idx]
                # find the closest segment that is assigned to a domain
                dist_to_other = segdist[idx, :]
                idx_sort_dist = np.argsort(dist_to_other)
                for idx2 in idx_sort_dist:
                    if segment_domain_type[idx2] != -1:
                        segment_domain_type[idx] = segment_domain_type[idx2]
                        break
        
        tuftTerm_idx = np.where(segment_domain_type == 0)[0]
        obliqueTerm_idx = np.where(segment_domain_type == 2)[0]
        basalTerm_idx = np.where(segment_domain_type == 7)[0]
        term_idx = np.concatenate((tuftTerm_idx, obliqueTerm_idx, basalTerm_idx))
        for idx in term_idx:
            seg_this = seglist[idx]
            type_this = segment_domain_type[idx]
            sec_this = seg_this.sec
            for seg in sec_this:
                seg_idx = seg2idx_hoc[seg]
                segment_domain_type[seg_idx] = type_this
        basalPrim_idx = np.where(segment_domain_type == 5)[0]
        basalPrim_seglist = [seglist[i] for i in basalPrim_idx]
        basalMid_idx = np.where(segment_domain_type == 6)[0]

        
        basalPrim_dist = {9: 12, 14:22, 18:12.6, 19:27, 20:11.2, 21:27, 22:14, 23:16, 24:42, 25:22.}
        basalPrim_dist = {9: 10, 14:22, 18:12.6, 19:27, 20:11.0, 21:23.5, 22:14, 23:15.2, 24:42, 25:22.}

        for idx in basalMid_idx:
            seg_this = seglist[idx]
            for seg in basalPrim_seglist:
                if h.distance(seg, seg_this) < basalPrim_dist[neuron_num]:
                    seg_idx = seg2idx_hoc[seg_this]
                    segment_domain_type[seg_idx] = 5
                    break
        """
        basalPrim_idx = np.where(segment_domain_type == 5)[0]
        for idx in basalPrim_idx:
            seg_this = seglist[idx]
            type_this = segment_domain_type[idx]
            sec_this = seg_this.sec
            for seg in sec_this:
                seg_idx = seg2idx_hoc[seg]
                segment_domain_type[seg_idx] = type_this
        
        basalMid_idx = np.where(segment_domain_type == 6)[0]
        mid2prim_idx = []
        for idx in basalMid_idx:
            seg_this = seglist[idx]
            parseg = seg_this.sec.parentseg()
            parseg = parseg.sec(0.99)
            type_par = segment_domain_type[seg2idx_hoc[parseg]]
            if type_par == 5:
                for seg in seg_this.sec:
                    seg_idx = seg2idx_hoc[seg]
                    mid2prim_idx.append(seg_idx)
        
        for idx in mid2prim_idx:
            segment_domain_type[idx] = 5
        """
        #tuftTerm, tuftMid, obliqueTerm, obliqueMid, trunk, basalPrim, basalMid, basalTerm = [],[],[],[],[],[],[],[]
        tuftTerm_idx = np.where(segment_domain_type == 0)[0]
        tuftMid_idx = np.where(segment_domain_type == 1)[0]
        obliqueTerm_idx = np.where(segment_domain_type == 2)[0]
        obliqueMid_idx = np.where(segment_domain_type == 3)[0]
        trunk_idx = np.where(segment_domain_type == 4)[0]
        basalPrim_idx = np.where(segment_domain_type == 5)[0]
        basalMid_idx = np.where(segment_domain_type == 6)[0]
        basalTerm_idx = np.where(segment_domain_type == 7)[0]

        self.tuftTerm = [seglist[i] for i in tuftTerm_idx]
        self.tuftMid = [seglist[i] for i in tuftMid_idx]
        self.obliqueTerm = [seglist[i] for i in obliqueTerm_idx]
        self.obliqueMid = [seglist[i] for i in obliqueMid_idx]
        self.trunk = [seglist[i] for i in trunk_idx]
        self.basalPrim = [seglist[i] for i in basalPrim_idx]
        self.basalMid = [seglist[i] for i in basalMid_idx]
        self.basalTerm = [seglist[i] for i in basalTerm_idx]

        """
        if neuron_num == 9:
            trunk = [h.apic[0], h.apic[1],h.apic[2],h.apic[3],h.apic[9],h.apic[10],h.apic[12],h.apic[13]]
        elif neuron_num == 18:
            trunk = [h.dend_0[3], h.apic[1],h.apic[2],h.apic[3],h.apic[4],h.apic[5],h.apic[6],h.apic[7], h.apic[8], h.apic[9], h.apic[10]]
        elif neuron_num == 21:
            trunk = [h.apic[0], h.apic[1],h.apic[2],h.apic[3]]
        else:
            trunk = []

        for sec in self.apic:
            if h.distance(self.soma[0](0.5), sec(0)) > dist_tuftterm:
                if len(sec.subtree()) == 1: 
                    tuftTerm.append(sec)
                else:
                    tuftMid.append(sec)
            else:
                if len(sec.subtree()) == 1:
                    obliqueTerm.append(sec)
                else:
                    if sec not in trunk:
                        obliqueMid.append(sec)

            
        for sec in self.dend:
            if h.distance(self.soma[0](0.5), sec(0)) < 20:
                basalPrim.append(sec)
            elif len(sec.subtree()) == 1:
                basalTerm.append(sec)
            else:
                basalMid.append(sec)

        self.tuftTerm = tuftTerm
        self.tuftMid = tuftMid
        self.obliqueTerm = obliqueTerm
        self.obliqueMid = obliqueMid
        self.trunk = trunk
        self.basalPrim = basalPrim
        self.basalMid = basalMid
        self.basalTerm = basalTerm
        

        self.all_dend = self.dend + self.apic

        if type == 'uni':
            rates = rates_uni
        else:
            rates = rates_bio
        
        multiplier = []
        multiplier += [rates[0]] * len(self.tuftTerm)
        multiplier += [rates[1]] * len(self.tuftMid)
        multiplier += [rates[2]] * len(self.obliqueTerm)
        multiplier += [rates[3]] * len(self.obliqueMid)
        multiplier += [rates[4]] * len(self.trunk)
        multiplier += [rates[5]] * len(self.basalPrim)
        multiplier += [rates[6]] * len(self.basalMid)
        multiplier += [rates[7]] * len(self.basalTerm)
        multiplier = np.array(multiplier)

        self.area_sec_dend = []
        self.len_sec_dend = []
        for sec in self.all_dend:
            area = 0
            L = sec.L
            for seg in sec:
                area += seg.area()
            self.area_sec_dend.append(area)
            self.len_sec_dend.append(L)

        soma_area = []
        soma_len = []
        for sec in self.soma:
            area = 0
            L = sec.L
            for seg in sec:
                area += seg.area()
            soma_area.append(area)
            soma_len.append(L)

        self.area_sec = soma_area + self.area_sec_dend
        self.len_sec = soma_len + self.len_sec_dend

        self.area_sec_dend = np.array(self.area_sec_dend)
        self.len_sec_dend = np.array(self.len_sec_dend)
        self.area_sec = np.array(self.area_sec)
        self.len_sec = np.array(self.len_sec)


        if choice == 'area':
            self.prob_exc = multiplier * self.area_sec_dend
            self.prob_exc = self.prob_exc/sum(self.prob_exc)
            self.prob_inh = self.area_sec/sum(self.area_sec)
        else:
            self.prob_exc = multiplier * self.len_sec_dend
            self.prob_exc = self.prob_exc/sum(self.prob_exc)
            self.prob_inh = self.len_sec/sum(self.len_sec)
        """
        self.biophys()

    def biophys(self):
        h('''
        forsec all {
            insert pas
            e_pas = -75 
            Ra = 150
            cm = 1
            g_pas = 1.0 / 12000.0
        }
        ''')
        h('''
        proc distribute(){local x localobj sl
                strdef stmp,distfunc,mech
                sl = $o1
                mech = $s2
                distfunc = $s3
                sprint(distfunc,"%%s %s(%%f) = %s",mech, distfunc)
                forsec sl for(x) {
                    sprint (stmp,distfunc,secname(),x,distance(x))
                    execute(stmp)
                }
            }
          
        forsec somatic {
            insert Ca_HVA
            insert SKv3_1
            insert SK_E2
            insert Ca_LVAst
            insert Ih
            insert NaTs2_t
            insert CaDynamics_E2
        }

        forsec axonal {
            insert Ca_HVA
            insert SKv3_1
            insert SK_E2
            insert CaDynamics_E2
            insert Nap_Et2
            insert K_Pst
            insert K_Tst
            insert Ca_LVAst
            insert NaTa_t
            
        }

        forsec axonal {
            ena = 50 
            ek = -85 
        }

        forsec somatic {
            ena = 50 
        }

        forsec somatic {
            ek = -85 
        }

        access soma
        distance()

        distribute(axonal,"gNaTa_tbar_NaTa_t","(0.0 * %g + 1.0)*3.429725")
        distribute(axonal,"gK_Tstbar_K_Tst","(0.0 * %g + 1.0)*0.001035")
        distribute(axonal,"gamma_CaDynamics_E2","(0.0 * %g + 1.0)*0.016713")
        distribute(axonal,"gNap_Et2bar_Nap_Et2","(0.0 * %g + 1.0)*0.009803")
        distribute(axonal,"gSK_E2bar_SK_E2","(0.0 * %g + 1.0)*0.008085")
        distribute(axonal,"gCa_HVAbar_Ca_HVA","(0.0 * %g + 1.0)*0.000306")
        distribute(axonal,"gK_Pstbar_K_Pst","(0.0 * %g + 1.0)*0.959296")
        distribute(axonal,"gSKv3_1bar_SKv3_1","(0.0 * %g + 1.0)*0.094971")
        distribute(axonal,"decay_CaDynamics_E2","(0.0 * %g + 1.0)*384.114655")
        distribute(axonal,"gCa_LVAstbar_Ca_LVAst","(0.0 * %g + 1.0)*0.000050")
        distribute(somatic,"gamma_CaDynamics_E2","(0.0 * %g + 1.0)*0.000533")
        distribute(somatic,"gSKv3_1bar_SKv3_1","(0.0 * %g + 1.0)*0.102517")
        distribute(somatic,"gSK_E2bar_SK_E2","(0.0 * %g + 1.0)*0.099433")
        distribute(somatic,"gCa_HVAbar_Ca_HVA","(0.0 * %g + 1.0)*0.000374")
        distribute(somatic,"gNaTs2_tbar_NaTs2_t","(0.0 * %g + 1.0)*0.926705")
        distribute(somatic,"gIhbar_Ih","(0.0 * %g + 1.0)*0.000080")
        distribute(somatic,"decay_CaDynamics_E2","(0.0 * %g + 1.0)*342.544232")
        distribute(somatic,"gCa_LVAstbar_Ca_LVAst","(0.0 * %g + 1.0)*0.000778")    
        ''')

        #dendrites
        h('''
        forsec basal {
            insert Ih
        }

        forsec apical {
            insert Im
            insert NaTs2_t
            insert SKv3_1
            insert Ih
        }

        forsec apical {
            ena = 50 
            ek = -85 
          
            //gCa_HVAbar_Ca_HVA = 0.000701
        }

        access soma
        distance()

        distribute(basal,"gIhbar_Ih","(0.0 * %g + 1.0)*0.000080")
        distribute(apical,"gNaTs2_tbar_NaTs2_t","(0.0 * %g + 1.0)*0.008009") // based on ttx comparison
        // distribute(apical,"gNaTs2_tbar_NaTs2_t","(0.0 * %g + 1.0)*0.012009") // original
        distribute(apical,"gSKv3_1bar_SKv3_1","(0.0 * %g + 1.0)*0.000513")
        distribute(apical,"gIhbar_Ih","(-0.869600 + 2.087000*exp((%g-0.000000)*0.020100))*0.000080") // based on attenuation comparison
        // distribute(apical,"gIhbar_Ih","(-0.869600 + 2.087000*exp((%g-0.000000)*0.003100))*0.000080") // original
        distribute(apical,"gImbar_Im","(0.0 * %g + 1.0)*0.000740")
        //distribute(apical,"gCa_LVAstbar_Ca_LVAst","(0.0 * %g + 1.0)*0.01")
        ''')   

"""
neuron_num = 9
L23 = L23PC(neuron_num)
seglist = []
seglist_no_axon = []


for sec in L23.all:
    for seg in sec:
        seglist.append(seg)
for sec in L23.all_no_axon:
    for seg in sec:
        seglist_no_axon.append(seg)

print(len(seglist), len(seglist_no_axon))

seg2idx = dict()
for i,seg in enumerate(seglist):
    seg2idx[seg] = i



synaptic_map_file_name = 'synapse_location_for_neuron_{}.pickle'.format(neuron_num)
(spine_locations, full_IS_locations) = pickle.load(open('model_files/{}'.format(synaptic_map_file_name), 'rb'), encoding='latin1')
sec, loc, branch_type, spine_head_width, spine_volume, spine_length = spine_locations[0]
if loc == 0.0:
    loc = 0.0001
if loc == 1.0:
    loc = 0.9999

h('access {}'.format(sec))
stim = h.IClamp(loc)

print(len(spine_locations))
exc_loc_idx = np.zeros(len(spine_locations), dtype=np.int32)
for i, (sec, loc, branch_type, spine_head_width, spine_volume, spine_length) in enumerate(spine_locations):
    if loc == 0.0:
        loc = 0.0001
    if loc == 1.0:
        loc = 0.9999
    h('access {}'.format(sec))
    stim.loc(loc)
    exc_loc_idx[i] = seg2idx[stim.get_segment()] 

np.save('./model_files/exc_loc_idx_{}.npy'.format(neuron_num), exc_loc_idx)

inh_loc_idx = np.zeros(len(full_IS_locations), dtype=np.int32)
for i, (sec, loc, branch_type, IS_volume, on_spine) in enumerate(full_IS_locations):
    if loc == 0.0:
        loc = 0.0001
    if loc == 1.0:
        loc = 0.9999
    h('access {}'.format(sec))
    stim.loc(loc)
    inh_loc_idx[i] = seg2idx[stim.get_segment()]

np.save('./model_files/inh_loc_idx_real_{}.npy'.format(neuron_num), inh_loc_idx)
print(len(inh_loc_idx))
#print(seglist[inh_loc_idx[100]])
#print(seglist[exc_loc_idx[100]])
"""
"""
sec, loc, branch_type, spine_head_width, spine_volume, spine_length = spine_locations[0]
h('access {}'.format(sec))
stim = h.IClamp(loc)
print(stim.get_segment())

sec, loc, branch_type, spine_head_width, spine_volume, spine_length = spine_locations[100]
print(h.secname())
print(sec)
print(loc)

h('access {}'.format(sec))
stim.loc(loc)
print(stim.get_segment())
"""

