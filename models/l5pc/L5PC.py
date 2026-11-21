from neuron import h
import numpy as np

#[basalTerm, basalPrim, tuftTerm, tuftMid, dist, obliqueTerm, obliquePrim+prox]
rates_bio = [1, 0.2, 0.5, 1.4, 1.2, 1, 0.4] 
rates_uni = [1, 1, 1, 1, 1, 1, 1]
rates_dis = [1, 1, 2, 2, 2, 1, 1]

class L5PCcell():
    def __init__(self, active, i, choice = 'area', type = 'bio'):
        h.load_file("import3d.hoc")
        morphology_file='./morphologies/cell1.asc'
        if active:
            h.load_file('./models/L5PCbiophys%d.hoc'%i)
        else:
            h.load_file('./models/L5PCbiophys5.hoc')            
        
        h.load_file('./models/L5PCtemplate.hoc')

        self.cell=h.L5PCtemplate(morphology_file)
        self.soma = list(self.cell.soma)
        self.dend = list(self.cell.dend)
        self.apic = list(self.cell.apic)
        self.axon = list(self.cell.axon)

        self.all = self.soma + self.dend + self.apic + self.axon
        self.all_no_axon = self.soma + self.dend + self.apic
        self.all_dend = self.dend + self.apic
        self.all_perisoma = self.soma + self.dend

        self.tuftTerm, self.tuftMid= [], []
        self.dist, self.obliqueTerm, self.prox = [], [], []
        self.basalPrim, self.basalTerm = [], []

        tuftDistance = 900
        distDistance = 600
        basalDistance = 100        

        for sec in self.apic:
            if h.distance(self.soma[0](0.5), sec(0.5)) > tuftDistance:
                if len(sec.subtree()) == 1:
                    self.tuftTerm.append(sec)
                else:
                    self.tuftMid.append(sec)
            elif h.distance(self.soma[0](0.5), sec(0.5)) > distDistance:
                self.dist.append(sec)
            else:
                if len(sec.subtree()) == 1:
                    self.obliqueTerm.append(sec)
                else:
                    self.prox.append(sec)

        for sec in self.dend:
            if h.distance(self.soma[0](0.5), sec(0.5)) > basalDistance:
                self.basalTerm.append(sec)
            else:
                self.basalPrim.append(sec)
                
        #self.print_len()
        self.calc_prob(choice=choice, type=type)

    def calc_len(self):
        self.prob_len_dend = []
        for sec in self.all_dend:
            self.prob_len_dend.append(sec.L)

        self.prob_len_dend = np.array(self.prob_len_dend)
        self.prob_len_dend /= np.sum(self.prob_len_dend)

        self.prob_len_perisoma = []
        for sec in self.all_perisoma:
            self.prob_len_perisoma.append(sec.L)

        self.prob_len_perisoma = np.array(self.prob_len_perisoma)
        self.prob_len_perisoma /= np.sum(self.prob_len_perisoma)

    def print_len(self):
        len_apic = 0
        len_dend = 0
        len_soma = 0

        for sec in self.apic:
            len_apic += sec.L
        
        for sec in self.dend:
            len_dend += sec.L
        
        for sec in self.soma:
            len_soma += sec.L

        print("apic len:", len_apic)
        print("dend len:", len_dend)
        print("soma len:", len_soma)

    def calc_area(self):
        self.prob_area_dend = []
        for sec in self.all_dend:
            area = 0
            for seg in sec:
                area += seg.area()
            self.prob_area_dend.append(area)

        self.prob_area_dend = np.array(self.prob_area_dend)
        self.prob_area_dend /= np.sum(self.prob_area_dend)

        self.prob_area_perisoma = []
        for sec in self.all_perisoma:
            area = 0
            for seg in sec:
                area += seg.area()
            self.prob_area_perisoma.append(area)

        self.prob_area_perisoma = np.array(self.prob_area_perisoma)
        self.prob_area_perisoma /= np.sum(self.prob_area_perisoma)

    def print_area(self):
        area_apic = 0
        area_dend = 0
        area_soma = 0

        for sec in self.apic:
            for seg in sec:
                area_apic += seg.area()
        
        for sec in self.dend:
            for seg in sec:
                area_dend += seg.area()
        
        for sec in self.soma:
            for seg in sec:
                area_soma += seg.area()

        print("apic area:", area_apic)
        print("dend area:", area_dend)
        print("soma area:", area_soma)

    def calc_prob(self, choice = 'area', type = 'bio'):
        tuftDistance = 900
        distDistance = 600
        basalDistance = 100

        soma = self.soma[0](0.5)
        prob_exc = np.zeros(len(self.dend + self.apic))
        if type == 'bio':
            rates = rates_bio
        elif type == 'uni':
            rates = rates_uni
        elif type == 'dis':
            rates = rates_dis
            
        for i, sec in enumerate(self.dend):
            if h.distance(soma, sec(0.5)) > basalDistance:
                rate = rates[0]
            else:
                rate = rates[1]

            if choice == 'area':
                area = 0
                for seg in sec:
                    area += seg.area()  
                prob_exc[i] = area * rate
            
            elif choice == 'len':
                prob_exc[i] = sec.L * rate

        for i, sec in enumerate(self.apic):
            if h.distance(soma, sec(0.5)) > tuftDistance:
                if len(sec.subtree()) == 1:
                    rate = rates[2]
                else:
                    rate = rates[3]
            elif h.distance(soma, sec(0.5)) > distDistance:
                rate = rates[4]

            else:
                if len(sec.subtree()) == 1:
                    rate = rates[5]
                else:
                    rate = rates[6]

            if choice == 'area':
                area = 0
                for seg in sec:
                    area += seg.area()  
                prob_exc[i + len(self.dend)] = area * rate
            
            elif choice == 'len':
                prob_exc[i + len(self.dend)] = sec.L * rate

        self.prob_exc = prob_exc / np.sum(prob_exc)
        
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

        """
        for sec in list(self.apic):
            if h.distance(soma, sec(0.5)) > tuftDistance:
                if len(sec.subtree()) == 1:
                    tuftList.append(sec)
                else:
                    distList.append(sec)

            elif h.distance(soma, sec(0.5)) > distDistance:
                distList.append(sec)
            else:
                if len(sec.subtree()) == 1:
                    obliqueList.append(sec)
                else:
                    midList.append(sec)

        basalTerm=[]
        basalPrim=[]
        for sec in list(self.dend):
            if h.distance(soma, sec(0.5)) > basalDistance:
                basalTerm.append(sec)
            else:
                basalPrim.append(sec)

        self.tuftList = tuftList
        self.distList = distList
        self.obliqueList = obliqueList
        self.midList = midList
        self.basalTerm = basalTerm
        self.basalPrim = basalPrim
        
        """

    def add_spines(self, spine_loc_list):
        self.neck_list = []
        self.head_list = []

        for i, loc in enumerate(spine_loc_list):
            self.neck_list.append(h.Section(name='neck[%d]'%i))
            self.neck_list[i].Ra = loc.sec.Ra * 2
            self.neck_list[i].cm = loc.sec.cm
            self.neck_list[i].nseg = 1
            self.neck_list[i].L = 1
            self.neck_list[i].diam = 0.1

            self.neck_list[i].insert('pas')
            for seg in self.neck_list[i]:
                seg.pas.g = loc.pas.g
                seg.pas.e = loc.pas.e

            self.neck_list[i].connect(loc, 0)

            self.head_list.append(h.Section(name='head[%d]'%i))
            self.head_list[i].Ra = loc.sec.Ra * 2
            self.head_list[i].cm = loc.sec.cm
            self.head_list[i].nseg = 1
            self.head_list[i].L = 0.5
            self.head_list[i].diam = 0.5

            self.head_list[i].insert('pas')
            for seg in self.head_list[i]:
                seg.pas.g = loc.pas.g
                seg.pas.e = loc.pas.e

            self.head_list[i].connect(self.neck_list[i], 0)


        self.all = self.soma + self.dend + self.apic + self.axon + self.neck_list + self.head_list 
        self.all_no_axon = self.soma + self.dend + self.apic




#cell = L5PCcell(True, 1)
#print(cell.calc_len())
