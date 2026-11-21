from neuron import h

class SimpleCell:
    def __init__(self):
        soma=h.Section(name='soma')
        soma.L=20
        soma.diam=20
        soma.Ra=100
        soma.cm=1
        soma.nseg=1

        soma.insert('pas')
        for seg in soma:
            seg.pas.g=0.00003
            seg.pas.e=-65

        dend1=h.Section(name='dend1')
        dend1.L=60
        dend1.diam=1
        dend1.Ra=100
        dend1.cm=1
        dend1.nseg=1

        dend1.insert('pas')
        for seg in dend1:
            seg.pas.g=0.00003
            seg.pas.e=-65

        dend1.connect(soma(0.5), 0)
        seclist = [soma, dend1]
        seglist = []
        for sec in seclist:
            for seg in sec:
                seglist.append(seg)

        self.soma = [soma]
        self.dend = [dend1]
        self.all = seclist

class VCell:
    def __init__(self):
        soma=h.Section(name='soma')
        soma.L=20
        soma.diam=20
        soma.Ra=100
        soma.cm=1
        soma.nseg=1

        soma.insert('pas')
        for seg in soma:
            seg.pas.g=0.00003
            seg.pas.e=-65

        dend1=h.Section(name='dend1')
        dend1.L=60
        dend1.diam=1
        dend1.Ra=100
        dend1.cm=1
        dend1.nseg=1

        dend1.insert('pas')
        for seg in dend1:
            seg.pas.g=0.00003
            seg.pas.e=-65

        dend1.connect(soma(0.5), 0)

        dend2=h.Section(name='dend2')
        dend2.L=60
        dend2.diam=1
        dend2.Ra=100
        dend2.cm=1
        dend2.nseg=1

        dend2.insert('pas')
        for seg in dend2:
            seg.pas.g=0.00003
            seg.pas.e=-65

        dend2.connect(soma(0.5), 0)

        seclist = [soma, dend1, dend2]
        seglist = []
        for sec in seclist:
            for seg in sec:
                seglist.append(seg)

        self.soma = [soma]
        self.dend = [dend1, dend2]
        self.all = seclist

class YCell:
    def __init__(self):
        soma=h.Section(name='soma')
        soma.L=20
        soma.diam=20
        soma.Ra=100
        soma.cm=1
        soma.nseg=1

        soma.insert('pas')
        for seg in soma:
            seg.pas.g=0.00003
            seg.pas.e=-65

        dend1=h.Section(name='dend1')
        dend1.L=20
        dend1.diam=1
        dend1.Ra=100
        dend1.cm=1
        dend1.nseg=1

        dend1.insert('pas')
        for seg in dend1:
            seg.pas.g=0.00003
            seg.pas.e=-65

        dend1.connect(soma(0.5), 0)

        dend2=h.Section(name='dend2')
        dend2.L=20
        dend2.diam=1
        dend2.Ra=100
        dend2.cm=1
        dend2.nseg=1

        dend2.insert('pas')
        for seg in dend2:
            seg.pas.g=0.00003
            seg.pas.e=-65

        dend2.connect(dend1(1), 0)

        dend3=h.Section(name='dend3')
        dend3.L=20
        dend3.diam=1
        dend3.Ra=100
        dend3.cm=1
        dend3.nseg=1

        dend3.insert('pas')
        for seg in dend3:
            seg.pas.g=0.00003
            seg.pas.e=-65

        dend3.connect(dend1(1), 0)

        seclist = [soma, dend1, dend2, dend3]
        seglist = []
        for sec in seclist:
            for seg in sec:
                seglist.append(seg)

        self.soma = [soma]
        self.dend = [dend1, dend2, dend3]
        self.all = seclist

class RallCell:
    def __init__(self, depth):
        soma=h.Section(name='soma')
        soma.L=20
        soma.diam=20
        soma.Ra=100
        soma.cm=1
        soma.nseg=1

        soma.insert('pas')
        for seg in soma:
            seg.pas.g=0.00003
            seg.pas.e=-65

        self.soma = [soma]
        self.dend = []
        dend_num = 2 ** depth - 1
        for i in range(dend_num):
            self.dend.append(h.Section(name = 'dend[%d]'%i))
            if i > 0:
                self.dend[i].connect(self.dend[int((i-1)/2)](1), 0)
                self.dend[i].L=100
                self.dend[i].diam= (0.5) ** (2/3) * self.dend[int((i-1)/2)].diam
                self.dend[i].Ra=100
                self.dend[i].cm=1
                self.dend[i].nseg=5

                self.dend[i].insert('pas')
                for seg in self.dend[i]:
                    seg.pas.g=0.00003
                    seg.pas.e=-65           
            else:
                self.dend[i].connect(self.soma[0](1), 0)
                self.dend[i].L=100
                self.dend[i].diam= 1
                self.dend[i].Ra=100
                self.dend[i].cm=1
                self.dend[i].nseg=5

                self.dend[i].insert('pas')
                for seg in self.dend[i]:
                    seg.pas.g=0.00003
                    seg.pas.e=-65   

        self.all = self.soma + self.dend              
        self.seglist = []
        for sec in self.all:
            for seg in sec:
                self.seglist.append(seg)


class SWCCell :
    def __init__(self,filename,v_r):
        self.load_morphology(filename)
        self.biophys(v_r)
        #self.normalize()
    def load_morphology(self,filename):
        reader = h.Import3d_SWC_read()
        reader.quiet = 1
        reader.input(filename)
        i3d = h.Import3d_GUI(reader, False)
        i3d.instantiate(self) 
        self.has_soma=False
        
        if hasattr(self, 'soma'):
            real=True
            for sec in self.soma:
                if sec.L < 1e-5:
                    real=False
                    self.all.remove(sec)
                    h.delete_section(sec=sec)
            if real is False:
                delattr(self, 'soma')
                self.has_soma=False
            else:
                self.has_soma=True
         
        if hasattr(self, 'axon'):
            for sec in self.axon:
                self.all.remove(sec)
                h.delete_section(sec=sec)
            delattr(self, 'axon')   
 
        if self.has_soma:
            self.soma_v=h.Vector().record(self.soma[0](0.5)._ref_v)
        
        else:
            self.soma_v=h.Vector().record(self.dend[0](0)._ref_v)

    def biophys(self,v_r):
        for sec in self.all:
            sec.Ra=100
            sec.cm=1         
            sec.nseg = 1 + 2*int(sec.L/40)
            sec.insert('pas')
            for seg in sec:
                seg.pas.g=0.00003
                seg.pas.e=v_r

        maxL=0
        for sec in self.all:
            for seg in sec:
                nowL = h.distance(self.soma[0](0.5), seg)
                if nowL > maxL:
                    maxL = nowL
        self.maxL=maxL

    def normalize(self):
        maxL=0
        for sec in self.all:
            for seg in sec:
                nowL = h.distance(self.soma[0](0.5), seg)
                if nowL > maxL:
                    maxL = nowL
        #print(totalL)
        for sec in self.all:
            #sec.diam=1
            sec.L *= (300/maxL)
            sec.nseg = 1 + 2*int(sec.L/40)
