import numpy as np
import torch as tc
import random as rd
from neuron import h, gui
from neuron.units import ms, mV
import os
from myfunc_torch import *
import time
from data_c2py import *
from CellType import *
from ChannelType import mech_type
import sys
             

class GradNeuron:
    def __init__(self, config, OUTPUT_PATH):
        self.N_e = config['N_e']
        self.N_i = config['N_i']
        self.N_back = config['N_back'] #new version: list(), [N_back_ampa, N_back_nmda, N_back_gaba]
        #self.N_back_ampa = config['N_back'][0]
        #self.N_back_nmda = config['N_back'][1]
        #self.N_back_gaba = config['N_back'][2]
        
        self.record = config['record']
        self.spiny = config['spiny']
        if self.spiny:
            self.N = 3 * self.N_e + self.N_i
        else:
            self.N = self.N_e + self.N_i

        self.grid_num = config['grid_num']
        self.syn_data = config['syn_data']
        #self.gaba_slow = config['gaba_slow']
        #self.dsyn_kernel = config['dsyn_kernel']
        self.inh_setting = config['inh_setting'] #[if_fs, if_depress, distance, if_slow]
        self.eca0 = 127.58951061761749
        self.nerst_const_ca = self.eca0 / np.log(2.0/5e-5)

        self.L_min = config['L_min']
        self.L_max = config['L_max']
        self.D_min = config['D_min']
        self.D_max = config['D_max']
        self.DH_min = config['DH_min']
        self.DH_max = config['DH_max']
        self.w_e_min = config['w_i_min']
        self.w_e_max = config['w_i_max']
        self.w_i_min = config['w_i_min']
        self.w_i_max = config['w_i_max']
        self.G_max = config['G_max']

        
        self.w_e = config['w_e'] if config['w_e'] is not None else np.random.uniform(0.1*1e-3, 5.*1e-3, (self.N_e,))
        self.w_i = config['w_i'] if config['w_i'] is not None else np.random.uniform(0.1*1e-3, 5.*1e-3, (self.N_i,))
        self.w_back = config['w_back'] if config['w_back'] is not None else np.random.uniform(0.1*1e-3, 5.*1e-3, (self.N_back,))

        self.cell_info = config['cell_info'] #new version: dict(), key = cell, soma, v_r, active
        self.active = self.cell_info ['active']
        self.cell = self.cell_info ['cell']
        self.v_r = self.cell_info ['v_r']
        self.soma = self.cell_info ['soma']
        self.soma_v = h.Vector().record(self.soma._ref_v)
        self.mech_list = self.cell_info['mech_list']
        self.ca_dynamics = None
        if 'CaDynamics_E2' in self.mech_list:
            self.ca_dynamics = 'CaDynamics_E2'
        elif 'cad' in self.mech_list:
            self.ca_dynamics = 'cad'

        self.exc_loc_idx = config['exc_loc_idx']
        self.inh_loc_idx = config['inh_loc_idx']
        self.back_loc_idx = config['back_loc_idx'] 

        self.sparse = True
        self.dtype = config['dtype']
        if self.dtype == 'float':
            self.np_dtype = np.float32
            self.tc_dtype = tc.float32
        elif self.dtype == 'double':
            self.np_dtype = np.float64
            self.tc_dtype = tc.float64
        else:
            assert 1==2
        
        self.dictlist = None
        self.rec = dict()
        
        
        self._setup_syns(OUTPUT_PATH)
        #device test:
        self.device = config.get('device', 'cpu')
        self._setup_record()
              


    def select_device(self):
        x_cpu = tc.randn(self.Nseg, self.Nseg, device='cpu')
        y_cpu = tc.randn(self.Nseg, self.N_i, device='cpu')

        x_gpu = x_cpu.to('cuda')
        y_gpu = y_cpu.to('cuda')
        def solve_cpu():
            tc.linalg.solve(x_cpu, y_cpu)
        def solve_gpu():
            tc.linalg.solve(x_gpu, y_gpu)

        import timeit
        print('selecting device...')
        t_cpu = timeit.timeit(solve_cpu, number=1)
        t_gpu = timeit.timeit(solve_gpu, number=1)
        
        dev = 'cpu' if t_cpu <= t_gpu else 'cuda'
        print('device selected: ', dev)
        return dev

            
           
    def _create_spines(self):
        assert len(self.w_e) == 3 * self.N_e
        self.neck_list=[]
        self.head_list=[]
        for i in range(self.N_e):
            self.neck_list.append(h.Section(name='neck[%d]'%i))
            self.neck_list[i].Ra = 100
            self.neck_list[i].cm = 1
            self.neck_list[i].nseg = 1
            self.neck_list[i].L = self.w_e[i]
            self.neck_list[i].diam = self.w_e[self.N_e + i]

            self.neck_list[i].insert('pas')
            for seg in self.neck_list[i]:
                seg.pas.g=0.00003
                seg.pas.e=self.v_r

            
            self.head_list.append(h.Section(name='dend[%d]'%i))
            self.head_list[i].Ra =  100
            self.head_list[i].cm =  1
            self.head_list[i].nseg = 1
            self.head_list[i].L = self.w_e[self.N_e * 2 + i]
            self.head_list[i].diam = self.w_e[self.N_e * 2 + i]

            self.head_list[i].insert('pas')
            for seg in self.head_list[i]:
                seg.pas.g=0.00003
                seg.pas.e=self.v_r

            self.head_list[i].connect(self.neck_list[i](1),0)
        
        for i in range(self.N_e):
            self.neck_list[i].connect(self.exc_loc[i], 0)

        
    def calc_SD(self):
            Enum = np.zeros(len(self.seclist))
            Inum = np.zeros(len(self.seclist))
            for loc in self.exc_loc:
                i = self.sec2idx[loc.sec]
                Enum[i] += 1
            
            for loc in self.inh_loc:
                i = self.sec2idx[loc.sec]
                Inum[i] += 1
            
            #print(np.sum(Enum), np.sum(Inum))
            SD_list = []
            for i in range(len(self.seclist)):
                sum = Enum[i] + Inum[i]
                dif = Enum[i] - Inum[i]
                if sum > 0:
                    SD_list.append(dif/sum)

            self.EI_SD = np.std(SD_list)
    
    def _init_mech_rec(self, mech_name):
        #print("get mech idx ", mech_name)
        self.rec[mech_name] = dict()
        #print("...")
        tmp = get_ml_nodeindices(mech_name)
        #print(tmp[:5])
        self.rec[mech_name]['idx'] = self.hines2idx[tmp]
        N_mech = len(self.rec[mech_name]['idx'])
        tp = mech_type[mech_name][0]
        ns = int(mech_type[mech_name][2])
        #print("mech idx finished")
        self.rec[mech_name]['idx'] = tc.from_numpy(self.rec[mech_name]['idx']).int().to(self.device)
        if tp == '': pass
        elif tp == 'cad':
            self.rec[mech_name]['dcdp'] = tc.zeros([N_mech, self.N], device =self.device, dtype = self.tc_dtype)
            self.rec[mech_name]['dedp'] = tc.zeros([N_mech, self.N], device =self.device, dtype = self.tc_dtype)

        elif tp in ['hh_ori', 'hh_like', 'sk', 'kca', 'ghk']:
            if ns == 1:
                self.rec[mech_name]['dxdp'] = tc.zeros([N_mech, self.N], device =self.device, dtype = self.tc_dtype)
            else:
                self.rec[mech_name]['dxdp'] = tc.zeros([N_mech, self.N, ns], device =self.device, dtype = self.tc_dtype)

        else:                                                                         
            assert 1==2

    def _collect_syn_rec(self):
        data_exc = mech_data_rec("Exp2Syn_Excitatory_ori").reshape(-1,7)
        data_inh = mech_data_rec("Exp2Syn_Inhibitory").reshape(-1,7)
        data_inh_fs = mech_data_rec("tmgExSyn").reshape(-1,7)
        data_pseudo = mech_data_rec("Syn_Pseudo").reshape(-1,7)
        data = np.concatenate((data_exc, data_inh, data_inh_fs, data_pseudo), axis = 0)
        # tc.float32
        data = tc.from_numpy(data.astype(self.np_dtype))

        self.rec['syn']['pure_i']= data[:,2] 
        self.rec['syn']['didv'] = data[:,3]

        #self.rec['syn']['i'] = data[:,0]
        #self.rec['syn']['w'][i] = syn.w
        #self.rec['syn']['djdv'] = data[:,3]

    def _collect_mech_rec(self, mech_name):
        #print("collecting ", mech_name)
        tp = mech_type[mech_name][0]
        if tp == '': return
        #print(mech_name, ' ', idx)
        ion = mech_type[mech_name][1]
        ns = int(mech_type[mech_name][2])

        idx_list = self.rec[mech_name]['idx']
        Nmech = len(idx_list)
        if Nmech == 0: return
        data = mech_data_rec(mech_name).reshape(Nmech,-1)
        #np.mean(data)
        #print(data.shape)
        data = tc.from_numpy(data.astype(self.np_dtype))
        
        """
        if tc.isnan(data).any():
            print(mech_name)
            print(tc.where(tc.isnan(data)))
            assert 1==2 
        """
        #print(data.shape)
        if tp == 'cad':
            self.rec[mech_name]['cai'] = data[:, 0].reshape(Nmech, 1)
            self.rec[mech_name]['dcdi'] = data[:, 1].reshape(Nmech, 1)
            self.rec[mech_name]['dcdc_'] = data[:, 2].reshape(Nmech, 1)
            
        elif tp == 'hh_ori':
            self.rec[mech_name]['g_ion'] = data[:,2:4]
            self.e_ion['na'][idx_list] = data[:,-2].to(self.device)
            self.e_ion['k'][idx_list] = data[:,-1].to(self.device)
            self.rec[mech_name]['dgdx']= data[:,4 : 4+ns].reshape(-1,1,ns)
            self.rec[mech_name]['dxdv']= data[:,4+ns : 4+ns*2].reshape(-1,1,ns)
            self.rec[mech_name]['dxdx_'] = data[:,4+ns*2 : 4+ns*3].reshape(-1,1,ns)
            
        elif tp == 'ghk':
            if ns == 1:
                self.rec[mech_name]['i'] =  data[:,0]
                self.rec[mech_name]['didv'] = data[:,1]
                self.rec[mech_name]['didc'] = data[:,2].reshape(-1, 1)
                self.rec[mech_name]['didx'] = data[:,3].reshape(-1,1)
                self.rec[mech_name]['djdv'] = data[:,4].reshape(-1, 1)
                self.rec[mech_name]['djdc'] = data[:,5].reshape(-1, 1)
                self.rec[mech_name]['djdx'] = data[:,6].reshape(-1,1)     
                self.rec[mech_name]['dxdv'] = data[:,7].reshape(-1,1)
                self.rec[mech_name]['dxdx_'] = data[:,8].reshape(-1,1)                 
            else:
                self.rec[mech_name]['i'] =  data[:,0]
                self.rec[mech_name]['didv'] = data[:,1]
                self.rec[mech_name]['didc'] = data[:,2].reshape(-1, 1)
                self.rec[mech_name]['didx'] = data[:,3:3+ns].reshape(-1,1,ns)
                self.rec[mech_name]['djdv'] = data[:,3+ns].reshape(-1, 1)
                self.rec[mech_name]['djdc'] = data[:,4+ns].reshape(-1, 1)
                self.rec[mech_name]['djdx'] = data[:,5+ns:5+ns*2].reshape(-1,1,ns)     
                self.rec[mech_name]['dxdv'] = data[:,5+ns*2:5+ns*3].reshape(-1,1,ns)
                self.rec[mech_name]['dxdx_'] = data[:,5+ns*3:5+ns*4].reshape(-1,1,ns)      

        else:
            self.rec[mech_name]['g_ion'] = data[:,1]
            self.e_ion[ion][idx_list] = data[:,-1].to(self.device)

            if tp == 'hh_like':
                if ns == 1:
                    self.rec[mech_name]['dgdx'] = data[:,2].reshape(-1,1)
                    self.rec[mech_name]['dxdv'] = data[:,3].reshape(-1,1)
                    self.rec[mech_name]['dxdx_'] = data[:,4].reshape(-1,1) 
                else:
                    self.rec[mech_name]['dgdx'] = data[:,2 : 2+ns].reshape(-1,1,ns)
                    self.rec[mech_name]['dxdv'] = data[:,2+ns : 2+ns*2].reshape(-1,1,ns)
                    self.rec[mech_name]['dxdx_'] = data[:,2+ns*2 : 2+ns*3].reshape(-1,1,ns)   

            if tp == 'sk':
                if ns == 1:
                    self.rec[mech_name]['dgdx'] = data[:,2].reshape(-1,1)
                    self.rec[mech_name]['dxdc'] = data[:,3].reshape(-1,1)
                    self.rec[mech_name]['dxdx_'] = data[:,4].reshape(-1,1)
                else:
                    self.rec[mech_name]['dgdx'] = data[:,2 : 2+ns].reshape(-1,1,ns)
                    self.rec[mech_name]['dxdc'] = data[:,2+ns : 2+ns*2].reshape(-1,1,ns)
                    self.rec[mech_name]['dxdx_'] = data[:,2+ns*2 : 2+ns*3].reshape(-1,1,ns)

            if tp == 'kca':
                if ns == 1:
                    self.rec[mech_name]['dgdx'] = data[:,2].reshape(-1,1)
                    self.rec[mech_name]['dxdv'] = data[:,3].reshape(-1,1)
                    self.rec[mech_name]['dxdc'] = data[:,4].reshape(-1,1)
                    self.rec[mech_name]['dxdx_'] = data[:,5].reshape(-1,1)
                else:
                    self.rec[mech_name]['dgdx'] = data[:,2 : 2+ns].reshape(-1,1,ns)
                    self.rec[mech_name]['dxdv'] = data[:,2+ns : 2+ns*2].reshape(-1,1,ns)
                    self.rec[mech_name]['dxdc'] = data[:,2+ns*2 : 2+ns*3].reshape(-1,1,ns)
                    self.rec[mech_name]['dxdx_'] = data[:,2+ns*3 : 2+ns*4].reshape(-1,1,ns)                             
            


    def _setup_syns(self, OUTPUT_PATH):
        self.seclist = list(self.cell.all)
        self.seglist = []
        for sec in self.seclist:
            for seg in sec:
                self.seglist.append(seg)
        
        self.seglist_no_axon = []
        for sec in self.cell.all_no_axon:
            for seg in sec:
                self.seglist_no_axon.append(seg)

        self.sec2idx =dict()
        self.seg2idx =dict()

        for i, sec in enumerate(self.seclist):
            self.sec2idx[sec] = i
        for i, seg in enumerate(self.seglist):
            self.seg2idx[seg] = i 

        assert self.exc_loc_idx is not None
        assert self.inh_loc_idx is not None
        assert self.back_loc_idx is not None
        
        print('loading existed location')
        self.exc_loc = [self.seglist[i] for i in self.exc_loc_idx]
        self.inh_loc = [self.seglist[i] for i in self.inh_loc_idx]
        self.back_loc = [self.seglist[i] for i in self.back_loc_idx]

        #self.calc_SD()

        if self.spiny == True:
            print("spiny not implemented yet")
            assert 1==2
            spine_L = np.random.uniform(0.01, 3, (self.N_e,))
            spine_D = np.random.uniform(0.01, 1, (self.N_e,))
            spine_DH = np.random.uniform(0.01, 3, (self.N_e,))
            self.w_e = np.concatenate((spine_L,spine_D,spine_DH))

            self._create_spines()
            
            self.seclist = list(self.cell.all) + self.neck_list + self.head_list

            for sec in self.neck_list + self.head_list:
                for seg in sec:
                    self.seglist.append(seg)  


        
        self.Nseg = len(self.seglist)    
        self.Nseg_no_axon = len(self.seglist_no_axon)
                    
        # input to cell
        # AMPA + NMDA
        self.synlist_e = []
        self.stimlist_e = []
        self.nclist_e = []


        for i in range(self.N_e):
            if self.spiny == True:
                syn = h.Exp2Syn_Excitatory_ori(self.head_list[i](0.5))
                syn.w = 1 * self.head_list[i].diam ** 3

            else:
                syn = h.Exp2Syn_Excitatory_ori(self.exc_loc[i])
                syn.w = self.w_e[i]
                syn.id = i

            syn.AMPA_tau1 = self.syn_data['AMPA_tau1']
            syn.AMPA_tau2 = self.syn_data['AMPA_tau2']
            syn.NMDA_tau1 = self.syn_data['NMDA_tau1']
            syn.NMDA_tau2 = self.syn_data['NMDA_tau2']
            syn.AMPA_e = self.syn_data['E_exc']
            syn.NMDA_e = self.syn_data['E_exc']
            syn.NMDA_C = 0.267      #2.95
            syn.NMDA_rho = 0.062    #0.465
            """
            syn.tau1 = 0.1
            syn.tau2 = 2
            syn.e = self.v_r + 65
            """
            self.synlist_e.append(syn)
            stim = h.VecStim()
            self.stimlist_e.append(stim)
            nc = h.NetCon(stim, syn)
            nc.threshold = 0
            nc.delay = 0
            nc.weight[0] = self.syn_data['ratio'][1] #1e-3 / (1 + self.r_na) * self.head_list[i].diam ** 3
            self.nclist_e.append(nc)

        # GABA
        self.synlist_i = []
        self.stimlist_i = []
        self.nclist_i = []

        idx_slow_start = self.Nseg * self.grid_num if self.inh_setting['if_axon'] \
            else self.Nseg_no_axon * self.grid_num
        
        for i in range(self.N_i):
            if self.inh_setting['if_kernel'] == True:
                syn = h.Syn_Pseudo(self.inh_loc[i])
                syn.amp = 1
                syn.delay = self.inh_setting['kernel_time_grid'][i]
                syn.dur = h.dt

            elif i >= idx_slow_start:
                assert self.inh_setting['if_slow'] == True
                syn = h.Exp2Syn_Inhibitory(self.inh_loc[i])
                syn.tau1 = self.syn_data['GABAb_tau1']  
                syn.tau2 = self.syn_data['GABAb_tau2']

            elif self.inh_setting['if_fs'] == True:
                dist_from_soma = h.distance(self.inh_loc[i], self.soma)
                if dist_from_soma < self.inh_setting['distance']:
                    if self.inh_setting['if_depress'] == True:
                        syn = h.tmgExSyn(self.inh_loc[i])
                        syn.tau_r = 0.2
                        syn.tau_d = 8.3
                        syn.Fac = 21
                        syn.Use = 0.25
                        syn.Dep = 706
                    else:
                        syn = h.Exp2Syn_Inhibitory(self.inh_loc[i])
                        syn.tau1 = 1.
                        syn.tau2 = 3.
                else:
                    syn = h.Exp2Syn_Inhibitory(self.inh_loc[i])
                    syn.tau1 = 1
                    syn.tau2 = 10                    
            else:
                syn = h.Exp2Syn_Inhibitory(self.inh_loc[i])
                syn.tau1 = self.syn_data['GABAa_tau1']
                syn.tau2 = self.syn_data['GABAa_tau2']

                                    
            syn.e = self.syn_data['E_inh']
            syn.w = self.w_i[i]
            syn.id = i + self.N_e
            self.synlist_i.append(syn)

            stim = h.VecStim()
            self.stimlist_i.append(stim)
            nc = h.NetCon(stim, syn)
            nc.threshold = 0
            nc.delay = 0
            nc.weight[0] = 1.0
            self.nclist_i.append(nc)

        #background
        self.synlist_back = []
        self.stimlist_back = []
        self.nclist_back = []

        for i in range(self.N_back[0]): # ampa
            syn = h.Exp2Syn_Excitatory_back(self.back_loc[i])
            syn.w = self.w_back[i]     
            syn.AMPA_tau1 = self.syn_data['AMPA_tau1']
            syn.AMPA_tau2 = self.syn_data['AMPA_tau2']
            syn.NMDA_tau1 = self.syn_data['NMDA_tau1']
            syn.NMDA_tau2 = self.syn_data['NMDA_tau2']
            syn.AMPA_e = self.syn_data['E_exc']
            syn.NMDA_e = self.syn_data['E_exc']
            syn.NMDA_C = 0.267      #2.95
            syn.NMDA_rho = 0.062    #0.465
            self.synlist_back.append(syn)

            stim = h.VecStim()
            self.stimlist_back.append(stim)
            nc = h.NetCon(stim, syn)
            nc.threshold = 0
            nc.delay = 0
            nc.weight[0] = self.syn_data['ratio'][0]
            self.nclist_back.append(nc)

        for i in range(self.N_back[1]): # nmda
            syn = h.Exp2Syn_Excitatory_back(self.back_loc[i + self.N_back[0]])
            syn.w = self.w_back[i + self.N_back[0]]     
            syn.AMPA_tau1 = self.syn_data['AMPA_tau1']
            syn.AMPA_tau2 = self.syn_data['AMPA_tau2']
            syn.NMDA_tau1 = self.syn_data['NMDA_tau1']
            syn.NMDA_tau2 = self.syn_data['NMDA_tau2']
            syn.AMPA_e = self.syn_data['E_exc']
            syn.NMDA_e = self.syn_data['E_exc']
            syn.NMDA_C = 0.267      #2.95
            syn.NMDA_rho = 0.062    #0.465
            self.synlist_back.append(syn)

            stim = h.VecStim()
            self.stimlist_back.append(stim)
            nc = h.NetCon(stim, syn)
            nc.threshold = 0
            nc.delay = 0
            nc.weight[0] = self.syn_data['ratio'][1]
            self.nclist_back.append(nc)

        for i in range(self.N_back[2]): # gaba
            syn = h.Exp2Syn_back(self.back_loc[i + self.N_back[0] + self.N_back[1]])
            syn.tau1 = self.syn_data['GABAa_tau1']
            syn.tau2 = self.syn_data['GABAa_tau2']
            syn.e = self.syn_data['E_inh']
            syn.w = self.w_back[i + self.N_back[0] + self.N_back[1]]
            #syn.id = i + self.N_e + self.N_i
            self.synlist_back.append(syn)

            stim = h.VecStim()
            self.stimlist_back.append(stim)
            nc = h.NetCon(stim, syn)
            nc.threshold = 0
            nc.delay = 0
            nc.weight[0] = 1.0
            self.nclist_back.append(nc)                        

        self.synlist = self.synlist_e + self.synlist_i + self.synlist_back
        self.stimlist = self.stimlist_e + self.stimlist_i + self.stimlist_back   
        self.nclist = self.nclist_e + self.nclist_i + self.nclist_back

    def _setup_record(self):
        h.finitialize(self.v_r * mV)
        h.fcurrent()

        #print("hines2idx")
        self.hines2idx = -1 * np.ones(len(self.seglist) + len(self.seclist) + 1, dtype = int)
        self.idx2hines = -1 * np.ones(len(self.seglist), dtype = int)

        #print(self.hines2idx.shape)

        for i, seg in enumerate(self.seglist):
            hidx = int(h.fmatrix(seg.x, 0, sec=seg.sec))
            #print(hidx)
            self.hines2idx[hidx] = i
            self.idx2hines[i] = hidx

        #print("hines2idx finished")
        if self.active:
            print('mech initialize')
            for mech_name in self.mech_list:
                #print("mech_name")
                self._init_mech_rec(mech_name)

        print("Get Syn info")
        self.rec['syn'] = dict()
        hidx_exc = get_ml_nodeindices("Exp2Syn_Excitatory_ori")
        hidx_inh = get_ml_nodeindices("Exp2Syn_Inhibitory")
        hidx_inh_fs = get_ml_nodeindices("tmgExSyn")
        hidx_pseudo = get_ml_nodeindices("Syn_Pseudo")

        hidx = np.concatenate((hidx_exc, hidx_inh, hidx_inh_fs, hidx_pseudo))
        self.rec['syn']['idx'] = tc.from_numpy(self.hines2idx[hidx]).int().to(self.device)
        #debug_syn_idx = [self.seg2idx[syn.get_segment()] for syn in self.synlist]
        data_exc = mech_data_rec("Exp2Syn_Excitatory_ori").reshape(-1,7)
        data_inh = mech_data_rec("Exp2Syn_Inhibitory").reshape(-1,7)
        data_inh_fs = mech_data_rec("tmgExSyn").reshape(-1,7)
        data_pseudo = mech_data_rec("Syn_Pseudo").reshape(-1,7)
        data = np.concatenate((data_exc, data_inh, data_inh_fs, data_pseudo), axis = 0)

        self.rec['syn']['id'] = tc.from_numpy(data[:,6].astype(int)).int().to(self.device)
        #if self.train_mode[0] == 'WE':
        #self.idx_self = self.rec['syn']['idx'][-self.N_i:] * self.N + cp.arange(self.N)
        
        #print(self.rec['syn']['idx'][:50])
        #print("-----------------")
        #print(self.rec['syn']['id'][:50])
        #print(np.array(debug_syn_idx, dtype=int)[self.rec['syn']['id'][:50]])
        #self.rec['syn']['idx'] = cp.asarray(self.rec['syn']['idx'], dtype= int)
        # soma current

        self.IClamp_list = [h.IClamp(self.soma)]
        self.rec['ic'] = dict()
        self.rec['ic']['idx'] = [self.seg2idx[ic.get_segment()] for ic in self.IClamp_list]
        self.rec['ic']['i'] = tc.zeros(len(self.IClamp_list), device = self.device, dtype = self.tc_dtype)

        self.rec['ic']['idx'] = tc.tensor(self.rec['ic']['idx']).int().to(self.device)


    def set_stim(self, S_e, S_i, S_back, stim_set):
        if S_e is not None:
            for t_list, stim in zip(S_e, self.stimlist_e):
                t_vec = h.Vector(t_list)
                stim.play(t_vec)
        if S_i is not None:
            for t_list, stim in zip(S_i, self.stimlist_i):
                #t_vec = h.Vector(t_list)
                t_vec = h.Vector(t_list)
                stim.play(t_vec)

        if S_back is not None:
            for t_list, stim in zip(S_back, self.stimlist_back):
                t_vec = h.Vector(t_list)
                stim.play(t_vec)
        
        for ic, param in zip(self.IClamp_list, stim_set):
            ic.dur = param[0]
            ic.delay = param[1]
            ic.amp = param[2]

        self._reset_lr_records()
    

    def _reset_lr_records(self):
        self.v = np.zeros(self.Nseg)
        self.v_next = np.zeros(self.Nseg)
        self.e_ion = dict()
        self.e_ion['na'] = tc.zeros(self.Nseg, device = self.device, dtype = self.tc_dtype)
        self.e_ion['k'] = tc.zeros(self.Nseg, device = self.device, dtype = self.tc_dtype)
        self.e_ion['ca'] = tc.zeros(self.Nseg, device = self.device, dtype = self.tc_dtype)
        self.e_ion['hcn'] = tc.zeros(self.Nseg, device = self.device, dtype = self.tc_dtype)
        #self.eca = np.ones(self.Nseg) * self.eca0
        self.dvdp = tc.zeros([self.Nseg, self.N], device = self.device, dtype = self.tc_dtype)
        self.dvdp_soma = tc.zeros([1, self.N], device = self.device, dtype = self.tc_dtype)
        self.dvdp_self = tc.zeros([1, self.N], device = self.device, dtype = self.tc_dtype)
        self.rec_time = []

        for mech_name in ['ic','syn']:
            for key in self.rec[mech_name].keys():
                if type(key) == str:
                    if key == 'idx' or key == 'id': continue
                    self.rec[mech_name][key] = tc.zeros(len(self.rec[mech_name][key]), dtype = self.tc_dtype)

        if self.active:
            for mech_name in self.mech_list:
                for key in self.rec[mech_name].keys():
                    if type(key) == str:
                        if key == 'idx': continue
                        if key[-2:] == 'dp':
                            self.rec[mech_name][key] = tc.zeros_like(self.rec[mech_name][key])
                        else:
                            self.rec[mech_name][key] = tc.empty_like(self.rec[mech_name][key], device = 'cpu')

    def _reset_syns(self):
        print("reset synapses")
        if self.spiny == True:
            print("reset synapses on spiny neurons not implemented yet")
            assert 1==2
        self.exc_loc = [self.seglist[i] for i in self.exc_loc_idx]
        self.inh_loc = [self.seglist[i] for i in self.inh_loc_idx]
        self.back_loc = [self.seglist[i] for i in self.back_loc_idx]

        self.N_e = len(self.exc_loc)
        self.N_i = len(self.inh_loc)
        self.N = self.N_e + self.N_i
        

        for syn in self.synlist:
            del syn
        for stim in self.stimlist:
            del stim
        for nc in self.nclist:
            del nc

        # input to cell
        # AMPA + NMDA
        self.synlist_e = []
        self.stimlist_e = []
        self.nclist_e = []


        for i in range(self.N_e):
            if self.spiny == True:
                syn = h.Exp2Syn_Excitatory_ori(self.head_list[i](0.5))
                syn.w = 1 * self.head_list[i].diam ** 3

            else:
                syn = h.Exp2Syn_Excitatory_ori(self.exc_loc[i])
                syn.w = self.w_e[i]
                syn.id = i

            syn.AMPA_tau1 = self.syn_data['AMPA_tau1']
            syn.AMPA_tau2 = self.syn_data['AMPA_tau2']
            syn.NMDA_tau1 = self.syn_data['NMDA_tau1']
            syn.NMDA_tau2 = self.syn_data['NMDA_tau2']
            syn.AMPA_e = self.syn_data['E_exc']
            syn.NMDA_e = self.syn_data['E_exc']
            syn.NMDA_C = 0.267      #2.95
            syn.NMDA_rho = 0.062    #0.465
            """
            syn.tau1 = 0.1
            syn.tau2 = 2
            syn.e = self.v_r + 65
            """
            self.synlist_e.append(syn)
            stim = h.VecStim()
            self.stimlist_e.append(stim)
            nc = h.NetCon(stim, syn)
            nc.threshold = 0
            nc.delay = 0
            nc.weight[0] = self.syn_data['ratio'][1] #1e-3 / (1 + self.r_na) * self.head_list[i].diam ** 3
            self.nclist_e.append(nc)

        # GABA
        self.synlist_i = []
        self.stimlist_i = []
        self.nclist_i = []

        idx_slow_start = self.Nseg * self.grid_num if self.inh_setting['if_axon'] \
            else self.Nseg_no_axon * self.grid_num
        
        for i in range(self.N_i):
            if self.inh_setting['if_kernel'] == True:
                syn = h.Syn_Pseudo(self.inh_loc[i])
                syn.amp = 1
                syn.delay = self.inh_setting['kernel_time_grid'][i]
                syn.dur = h.dt

            elif i >= idx_slow_start:
                assert self.inh_setting['if_slow'] == True
                syn = h.Exp2Syn_Inhibitory(self.inh_loc[i])
                syn.tau1 = self.syn_data['GABAb_tau1']  
                syn.tau2 = self.syn_data['GABAb_tau2']

            elif self.inh_setting['if_fs'] == True:
                dist_from_soma = h.distance(self.inh_loc[i], self.soma)
                if dist_from_soma < self.inh_setting['distance']:
                    if self.inh_setting['if_depress'] == True:
                        syn = h.tmgExSyn(self.inh_loc[i])
                        syn.tau_r = 0.2
                        syn.tau_d = 8.3
                        syn.Fac = 21
                        syn.Use = 0.25
                        syn.Dep = 706
                    else:
                        syn = h.Exp2Syn_Inhibitory(self.inh_loc[i])
                        syn.tau1 = 1.
                        syn.tau2 = 3.
                else:
                    syn = h.Exp2Syn_Inhibitory(self.inh_loc[i])
                    syn.tau1 = 1
                    syn.tau2 = 10                    
            else:
                syn = h.Exp2Syn_Inhibitory(self.inh_loc[i])
                syn.tau1 = self.syn_data['GABAa_tau1']
                syn.tau2 = self.syn_data['GABAa_tau2']

                                    
            syn.e = self.syn_data['E_inh']
            syn.w = self.w_i[i]
            syn.id = i + self.N_e
            self.synlist_i.append(syn)

            stim = h.VecStim()
            self.stimlist_i.append(stim)
            nc = h.NetCon(stim, syn)
            nc.threshold = 0
            nc.delay = 0
            nc.weight[0] = 1.0
            self.nclist_i.append(nc)

        #background
        self.synlist_back = []
        self.stimlist_back = []
        self.nclist_back = []

        for i in range(self.N_back[0]): # ampa
            syn = h.Exp2Syn_Excitatory_back(self.back_loc[i])
            syn.w = self.w_back[i]     
            syn.AMPA_tau1 = self.syn_data['AMPA_tau1']
            syn.AMPA_tau2 = self.syn_data['AMPA_tau2']
            syn.NMDA_tau1 = self.syn_data['NMDA_tau1']
            syn.NMDA_tau2 = self.syn_data['NMDA_tau2']
            syn.AMPA_e = self.syn_data['E_exc']
            syn.NMDA_e = self.syn_data['E_exc']
            syn.NMDA_C = 0.267      #2.95
            syn.NMDA_rho = 0.062    #0.465
            self.synlist_back.append(syn)

            stim = h.VecStim()
            self.stimlist_back.append(stim)
            nc = h.NetCon(stim, syn)
            nc.threshold = 0
            nc.delay = 0
            nc.weight[0] = self.syn_data['ratio'][0]
            self.nclist_back.append(nc)

        for i in range(self.N_back[1]): # nmda
            syn = h.Exp2Syn_Excitatory_back(self.back_loc[i + self.N_back[0]])
            syn.w = self.w_back[i + self.N_back[0]]     
            syn.AMPA_tau1 = self.syn_data['AMPA_tau1']
            syn.AMPA_tau2 = self.syn_data['AMPA_tau2']
            syn.NMDA_tau1 = self.syn_data['NMDA_tau1']
            syn.NMDA_tau2 = self.syn_data['NMDA_tau2']
            syn.AMPA_e = self.syn_data['E_exc']
            syn.NMDA_e = self.syn_data['E_exc']
            syn.NMDA_C = 0.267      #2.95
            syn.NMDA_rho = 0.062    #0.465
            self.synlist_back.append(syn)

            stim = h.VecStim()
            self.stimlist_back.append(stim)
            nc = h.NetCon(stim, syn)
            nc.threshold = 0
            nc.delay = 0
            nc.weight[0] = self.syn_data['ratio'][1]
            self.nclist_back.append(nc)

        for i in range(self.N_back[2]): # gaba
            syn = h.Exp2Syn_back(self.back_loc[i + self.N_back[0] + self.N_back[1]])
            syn.tau1 = self.syn_data['GABAa_tau1']
            syn.tau2 = self.syn_data['GABAa_tau2']
            syn.e = self.syn_data['E_inh']
            syn.w = self.w_back[i + self.N_back[0] + self.N_back[1]]
            #syn.id = i + self.N_e + self.N_i
            self.synlist_back.append(syn)

            stim = h.VecStim()
            self.stimlist_back.append(stim)
            nc = h.NetCon(stim, syn)
            nc.threshold = 0
            nc.delay = 0
            nc.weight[0] = 1.0
            self.nclist_back.append(nc)                        

        self.synlist = self.synlist_e + self.synlist_i + self.synlist_back
        self.stimlist = self.stimlist_e + self.stimlist_i + self.stimlist_back   
        self.nclist = self.nclist_e + self.nclist_i + self.nclist_back         

        self._setup_record()

    def get_grad(self, M):
        self.dAdp=[]
        for i in range(self.N_e):
            gradLM=GradNeckLength(self.dictlist, self.neck_list[i], self.head_list[i], M, self.M_,self.Nseg)       
            self.dAdp.append(gradLM)

        for i in range(self.N_e):
            gradDM=GradNeckDiam(self.dictlist,self.neck_list[i], self.head_list[i], M, self.M_,self.Nseg)        
            self.dAdp.append(gradDM)
  
        for i in range(self.N_e):
            gradDMH=GradHeadDiam(self.dictlist,self.head_list[i],M, self.M_,self.Nseg)
            self.dAdp.append(gradDMH) 

        self.dAdp = tc.tensor(self.dAdp, device = self.device)   

    def get_grad_sparse(self, M):
        print("...sparse...")
        assert hasattr(self, 'dictlist')
        self.pidx=tc.zeros([3 * self.N_e, 4], dtype = tc.int32)
        self.dAdp=tc.zeros([3 * self.N_e, 8], dtype = tc.int32)
        for i in range(self.N_e):
            self.pidx[i], self.dAdp[i] = GradNeckLength_sparse(self.dictlist, self.neck_list[i], self.head_list[i], M)
            self.pidx[i + self.N_e], self.dAdp[i + self.N_e] = GradNeckDiam_sparse(self.dictlist, self.neck_list[i], self.head_list[i], M)
            self.pidx[i + 2 * self.N_e], self.dAdp[i + 2 * self.N_e] = GradHeadDiam_sparse(self.dictlist, self.head_list[i], M)

        self.pidx = self.pidx.to(self.device)
        self.dAdp = self.dAdp.to(self.device)

    def _setup_matrix(self):
        seg2num=dict()
        seg2par=dict()
        seg2spar=dict()
        emp2real=dict()
        self.dictlist=[seg2num,seg2par,seg2spar,emp2real]        
        """
        if self.dictlist is None:
            seg2num=dict()
            seg2par=dict()
            seg2spar=dict()
            emp2real=dict()
            self.dictlist=[seg2num,seg2par,seg2spar,emp2real]
        """
        h.finitialize(self.v_r * mV)
        h.fcurrent()

        #print('calc')
        M=CalcMatrix(self.seclist,self.dictlist)
        #print('done')
        #M_=np.matrix(M).I
        self.idx_soma = self.dictlist[0][self.soma]
        M_=tc.linalg.inv(M)
        self.M_= M_.to(self.device)
        self.A = self.M_[0:self.Nseg,0:self.Nseg]
        self.A_inv = tc.linalg.inv(self.A)
        self.A_inv = self.A_inv.to(self.tc_dtype)
        #dtype = tc.float32

        """
        M_numpy = M.cpu().numpy()
        M_sum = np.sum(M_numpy, axis= 1)
        print(np.max(M_sum), np.min(M_sum))
        #self.A = cp.asarray(M_[0:len(MM),0:len(MM)])
        
        A_numpy = self.A.cpu().numpy()
        V, _=np.linalg.eig(A_numpy)
        V = np.sort(V)[::-1]
        print(V[0], V[-1])
        """
        #V,W = np.linalg.eig(cp.asnumpy(self.A))
        #V = np.sort(V)[::-1]
        #print(V[0])
        if self.spiny == True:
            if self.sparse:
                self.get_grad_sparse(M)
            else:
                self.get_grad(M)

        D1,D2=np.zeros(self.Nseg),np.zeros(self.Nseg)
        for sec in self.seclist:
            for seg in sec:
                idx=self.dictlist[0][seg]
                D1[idx]=h.dt*1e5/seg.cm/seg.area()
                D2[idx]=h.dt*1e3/seg.cm   

        self.dDdp=[]

        if self.spiny == True:
            for i in range(self.N_e):
                idx_head=self.dictlist[0][self.head_list[i](0.5)]
                D1[idx_head] *= (1 * self.head_list[i].diam ** 3)#(1e-3 / (1 + self.r_na) * self.head_list[i].diam ** 3)
            
            for i in range(self.N_e):
                idx_neck=self.dictlist[0][self.neck_list[i](0.5)]
                self.dDdp.append(D1[idx_neck]/(-self.neck_list[i].L))

            for i in range(self.N_e):
                idx_neck=self.dictlist[0][self.neck_list[i](0.5)]
                self.dDdp.append(D1[idx_neck]/(-self.neck_list[i].diam))

            for i in range(self.N_e):
                idx_head=self.dictlist[0][self.head_list[i](0.5)]
                self.dDdp.append(D1[idx_head] /(self.head_list[i].diam) )

        else:
            for i in range(self.N_e):
                idx_exc = self.exc_loc_idx[i]
                self.dDdp.append(D1[idx_exc])

            for i in range(self.N_i):
                idx_inh = self.inh_loc_idx[i]
                self.dDdp.append(D1[idx_inh])

            
        
        self.D1, self.D2 = tc.from_numpy(D1.astype(self.np_dtype)).to(self.device), tc.from_numpy(D2.astype(self.np_dtype)).to(self.device)
        self.dDdp = tc.tensor(self.dDdp, device = self.device)

        assert len(self.dDdp) == self.N

    def update_dvdp(self):

        #self.I = cp.asnumpy(I_syn)
        I_syn = np.zeros(self.Nseg)
        J_syn = np.zeros(self.Nseg)
        DJ_syn = np.zeros(self.Nseg)
        
        for i, seg in enumerate(self.seglist):
            self.v_next[i] = seg.v

        self.v = tc.from_numpy(self.v.astype(self.np_dtype)).to(self.device)
        self.v_next = tc.from_numpy(self.v_next.astype(self.np_dtype)).to(self.device)
        dv = self.v_next - self.v

        didp = tc.zeros(self.Nseg * self.N, device = self.device, dtype = self.tc_dtype)

        #debug ing...       
        #delta_I_syn = np.zeros(self.Nseg)

        for ic in self.IClamp_list:
            idx=self.dictlist[0][ic.get_segment()]
            I_syn[idx] += ic.i


        #debug ing...       
        #delta_I_syn = np.zeros(self.Nseg)

        T1 = time.time()
        for i,ic in enumerate(self.IClamp_list):
            self.rec['ic']['i'][i] = ic.i
            #I_syn[idx] += ic.i



        self._collect_syn_rec()
        T2 = time.time()
        for mech_name in ['ic','syn']:
            for key in self.rec[mech_name].keys():
                if type(key) == str and key != 'idx' and key != 'id':
                    self.rec[mech_name][key] = self.rec[mech_name][key].to(self.device)
        T3 = time.time()

        if self.spiny == False:
            #print("Calc Syn")
            data_exc = syn_data_rec("Exp2Syn_Excitatory_ori").reshape(-1,4)
            data_inh = syn_data_rec("Exp2Syn_Inhibitory").reshape(-1,4)
            data_inh_fs = syn_data_rec("tmgExSyn").reshape(-1,4)
            data_back_exc = syn_data_rec("Exp2Syn_Excitatory_back").reshape(-1,4) #ampa, nmda
            data_back_inh = syn_data_rec("Exp2Syn_back").reshape(-1,4) #gaba

            data = np.zeros([len(self.hines2idx), 4])
            if len(data_exc) > 0:
                data += data_exc
            if len(data_inh) > 0:
                data += data_inh
            if len(data_inh_fs) > 0:
                data += data_inh_fs
            if len(data_back_exc) > 0:
                data += data_back_exc
            if len(data_back_inh) > 0:
                data += data_back_inh

            #print(data.shape)
            I_syn = data[self.idx2hines, 0]
            J_syn = data[self.idx2hines, 2]
            DJ_syn = data[self.idx2hines ,3]

            #print("finished")
            idx_list = self.rec['syn']['idx']
            id_list = self.rec['syn']['id']
            idx_didp = idx_list * self.N + tc.arange(self.N, device = self.device)
            didp[idx_didp] -= self.dDdp[id_list] * (self.rec['syn']['pure_i'] - \
                            self.rec['syn']['didv'] * dv[idx_list])
            
        else:
            for i, syn in enumerate(self.synlist_e):
                idx=self.dictlist[0][syn.get_segment()]
                I_syn[idx] -= syn.pure_i 
                J_syn[idx] += syn.didv
                DJ_syn[idx] += syn.djdv
                didp[idx, i + 2 * self.N_e] -= self.dDdp[i + 2 * self.N_e] * (syn.pure_i - syn.didv * dv[idx])

                #delta_I_syn[idx] += syn.didv * dv[idx]
            
            for i, syn in enumerate(self.synlist_i):
                idx=self.dictlist[0][syn.get_segment()]
                I_syn[idx] -= syn.i
                J_syn[idx] += syn.didv * syn.w 
                DJ_syn[idx] += syn.djdv * syn.w
                if self.train_mode[1] == 'WI':
                    didp[idx, i + 3 * self.N_e] -= self.dDdp[i + 3 * self.N_e]  * (syn.pure_i - syn.didv * dv[idx])

                #delta_I_syn[idx] += syn.didv * syn.w * dv[idx] 

            for i, syn in enumerate(self.synlist_n):
                idx=self.dictlist[0][syn.get_segment()]
                I_syn[idx] -= syn.i                    
                J_syn[idx] += syn.didv * syn.w
                DJ_syn[idx] += syn.djdv * syn.w 
                didp[idx, i] -= self.dDdp[i]  * (syn.i - syn.didv * syn.w * dv[idx])
                didp[idx, i + self.N_e] -= self.dDdp[i + self.N_e] * (syn.i - syn.didv * syn.w * dv[idx])

                #delta_I_syn[idx] += syn.didv * syn.w * dv[idx] 
        #print('syn', abs(didp).max().item())   
        
        """
        if abs(didp).max().item() > 5e6 :
            print("******didp syn exceed******")
            idx = np.argmax(abs(didp)).item()
            segidx = idx // self.N
            pidx = idx % self.N
            print(self.seglist[segidx])
            print(pidx)
        """ 
        I_syn, J_syn, DJ_syn = tc.from_numpy(I_syn.astype(self.np_dtype)).to(self.device),  tc.from_numpy(J_syn.astype(self.np_dtype)).to(self.device), tc.from_numpy(DJ_syn.astype(self.np_dtype)).to(self.device)
        #delta_I_syn = cp.asarray(delta_I_syn)

        #dv = cp.asarray(dv)
        #didp = cp.asarray(didp)
        
        didp = didp.reshape(self.Nseg, self.N)
        # didp here is didw
        if self.record:
            self.didp_syn = didp.clone()#.to('cpu').numpy()
        T4 = time.time()
        didp += (self.D1 * DJ_syn * dv).reshape(self.Nseg, 1) * self.dvdp
        T5 = time.time()
        #print("before active", tc.any(tc.isnan(didp)))
        if self.active :
            G_ion = tc.zeros(self.Nseg, device = self.device, dtype = self.tc_dtype)
            I_ion = tc.zeros(self.Nseg, device = self.device, dtype = self.tc_dtype)
            didp_ca = tc.zeros([self.Nseg, self.N], device = self.device, dtype = self.tc_dtype)
            if self.record:
                didp_na = tc.zeros([self.Nseg, self.N], device = self.device, dtype = self.tc_dtype)
                didp_k = tc.zeros([self.Nseg, self.N], device = self.device, dtype = self.tc_dtype)
                dgdp_na = tc.zeros([self.Nseg, self.N], device = self.device, dtype = self.tc_dtype)
                dgdp_k = tc.zeros([self.Nseg, self.N], device = self.device, dtype = self.tc_dtype)
                dgdp_ca = tc.zeros([self.Nseg, self.N], device = self.device, dtype = self.tc_dtype)

            g_ca = tc.zeros(self.Nseg, device = self.device, dtype = self.tc_dtype)
                

            for mech_name in self.mech_list:
                self._collect_mech_rec(mech_name)    
            T6 = time.time()
            #for ion in self.e_ion.keys():
            #    self.e_ion[ion] = cp.asarray(self.e_ion[ion])

            for mech_name in self.mech_list:
                for key in self.rec[mech_name].keys():
                    if type(key) == str:
                        if key[-2:] == 'dp': continue
                        elif key == 'idx': continue
                        else:
                            self.rec[mech_name][key] = self.rec[mech_name][key].to(self.device)
            
            self.dcdp = tc.zeros([self.Nseg, self.N], device = self.device, dtype = self.tc_dtype)
            if self.ca_dynamics is not None:
                ca_type_name = self.ca_dynamics
                idx_list =  self.rec[ca_type_name]['idx']
                if len(idx_list) > 0:
                    self.dcdp[idx_list] = self.rec[mech_name]['dcdp']

            T7 = time.time()
                  

            for mech_name in self.mech_list:
                #print(mech_name)
                idx_list = self.rec[mech_name]['idx']
                if len(idx_list) == 0: continue
                tp = mech_type[mech_name][0]
                ion = mech_type[mech_name][1]
                ns = int(mech_type[mech_name][2])

                if tp == 'hh_ori':
                    tmp_na = (self.e_ion['na'] - self.v_next)[idx_list]
                    tmp_k = (self.e_ion['k'] - self.v_next)[idx_list]
                    G_ion[idx_list] += (self.rec[mech_name]['g_ion'][:,0] + self.rec[mech_name]['g_ion'][:,1])
                    I_ion[idx_list] += (self.rec[mech_name]['g_ion'][:,0] * tmp_na + self.rec[mech_name]['g_ion'][:,1] * tmp_k)

                    tmp2 = self.rec[mech_name]['dgdx'] * self.rec[mech_name]['dxdp']
                    if ns == 3:
                        tmp3 = tc.sum(tmp_na.reshape(-1,1,1) * tmp2[:,:,:2], axis=2) + tmp_k.reshape(-1,1) * tmp2[:,:,-1]
                    elif ns == 4:
                        tmp3 = tc.sum(tmp_na.reshape(-1,1,1) * tmp2[:,:,:3], axis=2) + tmp_k.reshape(-1,1) * tmp2[:,:,-1]
                    elif ns == 5:
                        tmp3 = tc.sum(tmp_na.reshape(-1,1,1) * tmp2[:,:,:3], axis=2) + tc.sum(tmp_k.reshape(-1,1,1) * tmp2[:,:,3:5], axis=2)
                    #print(tmp3.shape)

                    if self.record:
                        if ns == 3:
                            rev_na = (self.v - self.e_ion['na'])[idx_list]
                            didp_na[idx_list] += tc.sum(rev_na.reshape(-1,1,1) * tmp2[:,:,:2], axis=2)
                            didp_na[idx_list] +=  self.rec[mech_name]['g_ion'][:,0].reshape(-1,1) * self.dvdp[idx_list]
                            rev_k = (self.v - self.e_ion['k'])[idx_list]
                            didp_k[idx_list] += rev_k.reshape(-1,1) * tmp2[:,:,-1]
                            didp_k[idx_list] +=  self.rec[mech_name]['g_ion'][:,1].reshape(-1,1) * self.dvdp[idx_list]

                            dgdp_na[idx_list] += tc.sum(tmp2[:,:,:2], axis=2)
                            dgdp_k[idx_list] += tmp2[:,:,-1]

                        elif ns == 4:
                            rev_na = (self.v - self.e_ion['na'])[idx_list]
                            didp_na[idx_list] += tc.sum(rev_na.reshape(-1,1,1) * tmp2[:,:,:3], axis=2)
                            didp_na[idx_list] +=  self.rec[mech_name]['g_ion'][:,0].reshape(-1,1) * self.dvdp[idx_list]
                            rev_k = (self.v - self.e_ion['k'])[idx_list]
                            didp_k[idx_list] += rev_k.reshape(-1,1) * tmp2[:,:,-1]
                            didp_k[idx_list] +=  self.rec[mech_name]['g_ion'][:,1].reshape(-1,1) * self.dvdp[idx_list]   

                            dgdp_na[idx_list] += tc.sum(tmp2[:,:,:3], axis=2)
                            dgdp_k[idx_list] += tmp2[:,:,-1]  

                        elif ns == 5:
                            rev_na = (self.v - self.e_ion['na'])[idx_list]
                            didp_na[idx_list] += tc.sum(rev_na.reshape(-1,1,1) * tmp2[:,:,:3], axis=2)
                            didp_na[idx_list] +=  self.rec[mech_name]['g_ion'][:,0].reshape(-1,1) * self.dvdp[idx_list]
                            rev_k = (self.v - self.e_ion['k'])[idx_list]
                            didp_k[idx_list] += tc.sum(rev_k.reshape(-1,1,1) * tmp2[:,:,3:5], axis=2)
                            didp_k[idx_list] +=  self.rec[mech_name]['g_ion'][:,1].reshape(-1,1) * self.dvdp[idx_list]                       

                            dgdp_na[idx_list] += tc.sum(tmp2[:,:,:3], axis=2)
                            dgdp_k[idx_list] += tc.sum(tmp2[:,:,3:5], axis=2)
                        else:
                            print("unknown hh_ori type")
                            assert 1==2

                    didp[idx_list] +=  self.D2[idx_list].reshape(-1,1) * tmp3

                elif tp in ['hh_like', 'sk', 'kca']:
                    tmp = (self.e_ion[ion] - self.v_next)[idx_list]
                    G_ion[idx_list] += self.rec[mech_name]['g_ion']
                    I_ion[idx_list] += self.rec[mech_name]['g_ion'] * tmp

                    tmp2 = self.rec[mech_name]['dgdx'] * self.rec[mech_name]['dxdp']
                    if ns == 1:
                        tmp3 = (self.D2[idx_list] * tmp).reshape(-1,1) * tmp2
                        didp[idx_list] += tmp3
                    else:
                        tmp3 = (self.D2[idx_list] * tmp).reshape(-1,1,1) * tmp2
                        didp[idx_list] += tc.sum(tmp3, axis=2)  

                    if ion == 'ca':
                        tmp4 = (self.v - self.e_ion[ion])[idx_list].reshape(-1,1,1) * tmp2
                        didp_ca[idx_list] += tc.sum(tmp4, axis=2)    
                        didp_ca[idx_list] +=  self.rec[mech_name]['g_ion'].reshape(-1,1) * self.dvdp[idx_list]     
                        g_ca[idx_list] += self.rec[mech_name]['g_ion']  
                        if self.record:
                            dgdp_ca[idx_list] += tc.sum(tmp2, axis=2)

                    if self.record:
                        if ion == 'na':
                            if ns == 1:
                                tmp5 = (self.v - self.e_ion[ion])[idx_list].reshape(-1,1) * tmp2
                                didp_na[idx_list] += tmp5
                                didp_na[idx_list] +=  self.rec[mech_name]['g_ion'].reshape(-1,1) * self.dvdp[idx_list]

                                dgdp_na[idx_list] += tmp2

                            else:
                                tmp5 = (self.v - self.e_ion[ion])[idx_list].reshape(-1,1,1) * tmp2
                                didp_na[idx_list] += tc.sum(tmp5, axis=2)
                                didp_na[idx_list] +=  self.rec[mech_name]['g_ion'].reshape(-1,1) * self.dvdp[idx_list]

                                dgdp_na[idx_list] += tc.sum(tmp2, axis=2)

                        elif ion == 'k':
                            if ns == 1:
                                tmp5 = (self.v - self.e_ion[ion])[idx_list].reshape(-1,1) * tmp2
                                didp_k[idx_list] += tmp5
                                didp_k[idx_list] +=  self.rec[mech_name]['g_ion'].reshape(-1,1) * self.dvdp[idx_list]

                                dgdp_k[idx_list] += tmp2

                            else:
                                tmp5 = (self.v - self.e_ion[ion])[idx_list].reshape(-1,1,1) * tmp2
                                didp_k[idx_list] += tc.sum(tmp5, axis=2)
                                didp_k[idx_list] +=  self.rec[mech_name]['g_ion'].reshape(-1,1) * self.dvdp[idx_list]

                                dgdp_k[idx_list] += tc.sum(tmp2, axis=2)


                elif tp == 'ghk':
                    G_ion[idx_list] += self.rec[mech_name]['didv']
                    I_ion[idx_list] -= (self.rec[mech_name]['i'] + self.rec[mech_name]['didv'] * dv[idx_list])

                    tmp = self.rec[mech_name]['didc'] * self.dcdp[idx_list]
                    if ns == 1:
                        tmp += self.rec[mech_name]['didx'] * self.rec[mech_name]['dxdp']
                    else:
                        tmp += tc.sum(self.rec[mech_name]['didx'] * self.rec[mech_name]['dxdp'], axis = 2)

                    tmp2 = self.rec[mech_name]['djdv'] * self.dvdp[idx_list]
                    tmp2 += self.rec[mech_name]['djdc'] * self.dcdp[idx_list]
                    if ns == 1:
                        tmp2 += self.rec[mech_name]['djdx'] * self.rec[mech_name]['dxdp']
                    else:
                        tmp2 += tc.sum(self.rec[mech_name]['djdx'] * self.rec[mech_name]['dxdp'], axis = 2)


                    if ion == 'ca':
                        tmp3 = self.rec[mech_name]['didv'].reshape(-1,1) * self.dvdp[idx_list]
                        didp_ca[idx_list] += (tmp + tmp3)

                    tmp += (tmp2 * dv[idx_list].reshape(-1,1))
                    didp[idx_list] -= (self.D2[idx_list].reshape(-1,1) * tmp)

                    if self.record:
                        if ion == 'na':
                            didp_na[idx_list] += tmp
                            dgdp_na[idx_list] += tmp2

                        elif ion == 'k':
                            didp_k[idx_list] += tmp
                            dgdp_k[idx_list] += tmp2
                        
                        elif ion == 'ca':
                            dgdp_ca[idx_list] += tmp2

                elif tp =='cad':
                    tmp = g_ca[idx_list].reshape(-1,1) * self.rec[mech_name]['dedp']
                    didp_ca[idx_list] -= tmp
                    didp[idx_list] += self.D2[idx_list].reshape(-1,1) * tmp
                    self.rec[mech_name]['dcdp'] = self.rec[mech_name]['dcdc_'] * self.rec[mech_name]['dcdp'] + self.rec[mech_name]['dcdi'] * didp_ca[idx_list]
                    self.rec[mech_name]['dedp'] = - self.nerst_const_ca * self.rec[mech_name]['dcdp'] / self.rec[mech_name]['cai']                    
                else:
                    assert 1==2                 


            #G_ion, I_ion = cp.asarray(G_ion), cp.asarray(I_ion) 
        T8 = time.time()    
        #print(T2-T1, ' ', T3-T2, ' ', T4-T3)
        #print("rec time: ", T2-T1)
        #print("didp calc time: ", T4 - T2)
        #self.didp_time.append(T8-T9)
        #print('ion', abs(didp).max().item()) 

        #print('J', abs(didp).max().item()) 

        jacob = self.D1 * J_syn
        if self.active:
            jacob -= self.D2 * G_ion

        #lhs=tc.eye(self.Nseg, device = self.device) - tc.matmul(self.A,tc.diag(jacob))
        lhs = tc.linalg.inv(self.A_inv - tc.diag(jacob))
        #self.lhs_rec.append(cp.linalg.det(lhs))
        
        rhs= self.dvdp + didp
        assert rhs.shape == (self.Nseg, self.N)
        
        if self.spiny == True:
            if self.sparse:
                rvec = tc.zeros(len(self.M_), device = self.device, dtype = self.tc_dtype)
                rvec[:len(self.A)] = (self.v - self.v_r) + self.D1 * (I_syn + J_syn * dv)
                if self.active:
                   rvec[:len(self.A)] +=  self.D2 * I_ion
                rvec=tc.matmul(self.M_, rvec)

                Rrec=tc.zeros([3 * self.N_e, 3], device = self.device, dtype = self.tc_dtype)
                Rrec[:, 0] = self.dAdp[:, 0] * rvec[self.pidx[:, 0]] + self.dAdp[:, 1] * rvec[self.pidx[:, 1]]
                Rrec[:, 1] = self.dAdp[:, 2] * rvec[self.pidx[:, 0]] + self.dAdp[:, 3] * rvec[self.pidx[:, 2]] + \
                    self.dAdp[:, 4] * rvec[self.pidx[:, 1]]
                Rrec[:, 2] = self.dAdp[:, 5] * rvec[self.pidx[:, 1]] + self.dAdp[:, 6] * rvec[self.pidx[:, 3]] + \
                    self.dAdp[:, 7] * rvec[self.pidx[:, 2]]        

                #print(self.pidx[:,:3].shape)
                tmp = - self.M_[:,self.pidx[:,:3]] * Rrec
                TMP = tc.sum(tmp, axis = 2)[:self.Nseg]

            else:
                if self.active:
                   TMP = tc.matmul(self.dAdp, self.v - self.v_r + self.D1 * (I_syn + J_syn * dv) + self.D2 * I_ion).transpose()
                else:
                   TMP = tc.matmul(self.dAdp, self.v - self.v_r + self.D1 * (I_syn + J_syn * dv)).transpose()

            assert(TMP.shape == (self.Nseg, 3 * self.N_e))
            rhs[:, :3 * self.N_e] += TMP

        #print(tc.any(tc.isnan(lhs)), tc.any(tc.isnan(rhs)))
        #print(tc.linalg.det(lhs))
        self.dvdp = tc.matmul(lhs, rhs)

        assert self.dvdp.shape == (self.Nseg, self.N)
        T9 = time.time()
        #print("solve time: ", T6-T5)
        self.dvdp_soma = tc.concatenate((self.dvdp_soma, self.dvdp[self.idx_soma].reshape(1,-1)), axis = 0)
        #self.dvdp_self = cp.concatenate((self.dvdp_self, self.dvdp.flatten()[self.idx_self].reshape(1,-1)), axis = 0)
        if self.active:
            self.dcdp = tc.zeros([self.Nseg, self.N], device = self.device, dtype = self.tc_dtype)
            if self.ca_dynamics is not None:
                ca_type_name = self.ca_dynamics
                idx_list =  self.rec[ca_type_name]['idx']
                if len(idx_list) > 0:
                    self.dcdp[idx_list] = self.rec[ca_type_name]['dcdp']

            for mech_name in self.mech_list:
                #print(mech_name)
                idx_list = self.rec[mech_name]['idx']
                if len(idx_list) == 0: continue
                tp = mech_type[mech_name][0]
                ion = mech_type[mech_name][1]
                ns = int(mech_type[mech_name][2])
                if tp == 'cad':  continue

                elif tp in ['hh_ori','hh_like','ghk']:
                    if ns == 1:
                        self.rec[mech_name]['dxdp'] = self.rec[mech_name]['dxdv'] * self.dvdp[idx_list] + \
                            self.rec[mech_name]['dxdx_'] * self.rec[mech_name]['dxdp']  
                    else:
                        self.rec[mech_name]['dxdp'] = self.rec[mech_name]['dxdv'] * self.dvdp[idx_list,:,np.newaxis] + \
                            self.rec[mech_name]['dxdx_'] * self.rec[mech_name]['dxdp'] 
                        
                elif tp == 'sk':
                    if ns == 1:
                        self.rec[mech_name]['dxdp'] = self.rec[mech_name]['dxdc'] * self.dcdp[idx_list] + \
                                self.rec[mech_name]['dxdx_'] * self.rec[mech_name]['dxdp'] 
                    else:
                        self.rec[mech_name]['dxdp'] = self.rec[mech_name]['dxdc'] * self.dcdp[idx_list,:,np.newaxis] + \
                                self.rec[mech_name]['dxdx_'] * self.rec[mech_name]['dxdp']  
                elif tp == 'kca':
                    if ns == 1:
                        self.rec[mech_name]['dxdp'] = self.rec[mech_name]['dxdv'] * self.dvdp[idx_list] + \
                                self.rec[mech_name]['dxdc'] * self.dcdp[idx_list] + \
                                self.rec[mech_name]['dxdx_'] * self.rec[mech_name]['dxdp'] 
                    else:
                        self.rec[mech_name]['dxdp'] = self.rec[mech_name]['dxdv'] * self.dvdp[idx_list,:,np.newaxis] + \
                                self.rec[mech_name]['dxdc'] * self.dcdp[idx_list,:,np.newaxis] + \
                                self.rec[mech_name]['dxdx_'] * self.rec[mech_name]['dxdp']  

                else:
                    assert 3==4                                                             
                

        if self.record and self.active:
            self.didp_ca = didp_ca #.to('cpu').numpy()
            self.didp_na = didp_na #.to('cpu').numpy()
            self.didp_k = didp_k #.to('cpu').numpy()
            self.dgdp_na = dgdp_na
            self.dgdp_k = dgdp_k
            self.dgdp_ca = dgdp_ca
        
        self.v = np.zeros(self.Nseg)
        self.v_next = np.zeros(self.Nseg)
        for i, seg in enumerate(self.seglist):
            self.v[i] = seg.v

        T10 = time.time() 

        self.rec_time.append([T2-T1, T3-T2, T4-T3, T5-T4, T6-T5, T7-T6, T8-T7, T9-T8, T10-T9])
        #print("dvdp calc time: ", T3-T4)
        #self.dvdp_time.append(T3-T8)
        #print(T5-T4, ' ', T6-T5, ' ', T7-T6, ' ', T8-T7)
        #print("**************************************")

    def init_weights(self, w):
        print("initialize all synaptic weights")
        assert len(w) == self.N_e + self.N_i + self.N_back
        self.w_e = w[:self.N_e]
        self.w_i = w[self.N_e : self.N_e + self.N_i]
        self.w_back = w[self.N_e + self.N_i : ]
        for i in range(self.N_e):
            self.synlist_e[i].w = self.w_e[i]  
        for i in range(self.N_i):
            self.synlist_i[i].w = self.w_i[i]
        for i in range(self.N_back):
            self.synlist_back[i].w = self.w_back[i]

    """
    def update_weights(self, dw):
        assert dw.shape == (self.N,)
        #dw = cp.asarray(dw)

        if self.train_mode[0] == 'WE':
            self.w_e += dw[:self.N_e]
        if self.train_mode[0] == 'SE':
            self.w_e += dw[:3 * self.N_e]
        if self.train_mode[1] == 'WI':
            self.w_i += dw[-self.N_i:]

        # clip w
        if self.train_mode[0] == 'WE':
            self.w_e=np.clip(self.w_e, a_min=self.w_e_min, a_max=self.w_e_max)
        if self.train_mode[0] == 'SE':
            self.w_e[:self.N_e] = np.clip(self.w_e[:self.N_e], a_min=self.L_min, a_max=self.L_max)
            self.w_e[self.N_e : self.N_e*2] = np.clip(self.w_e[self.N_e : self.N_e*2], a_min=self.D_min, a_max=self.D_max)
            self.w_e[self.N_e*2  : self.N_e*3] = np.clip(self.w_e[self.N_e*2 : self.N_e*3], a_min=self.DH_min, a_max=self.DH_max)
        if self.train_mode[1] == 'WI':
            self.w_i=np.clip(self.w_i, a_min=self.w_i_min, a_max=self.w_i_max)
        
        
        self.set_weights()
    """

    def set_weights(self, w):
        # update weights to syns that need gradient calc.
        assert w is not None
        assert len(w) == self.N
        if self.spiny is True:
            w[:self.N_e] = np.clip(w[:self.N_e], a_min=self.L_min, a_max=self.L_max)
            w[self.N_e : self.N_e*2] = np.clip(w[self.N_e : self.N_e*2], a_min=self.D_min, a_max=self.D_max)
            w[self.N_e*2  : self.N_e*3] = np.clip(w[self.N_e*2 : self.N_e*3], a_min=self.DH_min, a_max=self.DH_max)
            self.w_e = w[:self.N_e * 3]     

            for i in range(self.N_e):
                self.neck_list[i].L = self.w_e[i]
                self.neck_list[i].diam =  self.w_e[i + self.N_e]
                self.head_list[i].L = self.w_e[i + 2* self.N_e]
                self.head_list[i].diam = self.w_e[i + 2* self.N_e]
                self.synlist_e[i].w = 1 * self.head_list[i].diam ** 3

        else:       
            w[:self.N_e] = np.clip(w[:self.N_e], a_min=self.w_e_min, a_max=self.w_e_max)
            self.w_e = w[:self.N_e]

            for i in range(self.N_e):
                self.synlist_e[i].w = self.w_e[i]  
        
        w[-self.N_i:]=np.clip(w[-self.N_i:], a_min=self.w_i_min, a_max=self.w_i_max)
        self.w_i = w[-self.N_i:]
        for i in range(self.N_i):
            self.synlist_i[i].w = self.w_i[i]   

    def save_loc(self ,path):
        np.save(os.path.join(path, 'exc_loc.npy'), self.exc_loc_idx)
        np.save(os.path.join(path, 'inh_loc.npy'), self.inh_loc_idx)
        np.save(os.path.join(path, 'back_loc.npy'), self.back_loc_idx)
        
    def save_weights(self, path):
        w = np.concatenate((self.w_e, self.w_i, self.w_back))
        np.save(path, w)