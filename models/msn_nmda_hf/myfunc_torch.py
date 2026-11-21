from neuron import h, gui
from neuron.units import ms, mV
import numpy as np
import torch as tc
from matplotlib import pyplot as plt


#print(h.fmatrix(0.8,1,sec=dend1))

def CalcMatrix(seclist,dictlist):
  seg2num=dictlist[0]
  seg2par=dictlist[1]
  seg2spar=dictlist[2]
  emp2real=dictlist[3]
  if len(seg2num) == 0:
    cnt=0
    for sec in seclist:
      parseg=sec.parentseg()
      ori=sec.orientation()
      x=1/sec.nseg/2
      if parseg is not None:
        seg2par[sec(ori)]=parseg
      l=np.zeros(sec.nseg+1)
      l[0]=ori
      l[1:sec.nseg+1]=np.linspace(abs(ori-x),abs(1-ori-x),sec.nseg)
      if(sec(ori) not in seg2spar.keys()):
        seg2spar[sec(ori)]=parseg
      for i in range(1,len(l)):
        seg2num[sec(l[i])]=cnt
        cnt+=1
        seg2spar[sec(l[i])]=sec(l[i-1])
      seg2spar[sec(1-ori)]=sec(abs(1-ori-x))

    #print(cnt, len(seg2num) )
    start=cnt*1 
    #print("********")
    for sec in seclist:
      parseg=sec.parentseg()
      ori=sec.orientation()
      x=1/sec.nseg/2
      if parseg is None:
        if sec(0) not in seg2num.keys():
          seg2num[sec(0)]=cnt
          cnt+=1
        emp2real[sec(0)]=sec(x)
        if sec(1) not in seg2num.keys():
          seg2num[sec(1)]=cnt
          cnt+=1
        emp2real[sec(1)]=sec(1-x)      
      else:    
        if parseg not in seg2num.keys():
          seg2num[parseg]=cnt
          cnt+=1
          px=1/parseg.sec.nseg/2
          emp2real[parseg]=parseg.sec(abs(parseg.x-px))

        if sec(1-ori) not in seg2par.values():
          seg2num[sec(1-ori)]=cnt
          cnt+=1
          emp2real[sec(1-ori)]=sec(abs(1-ori-x))       
    #print(cnt)
    
  
  n1 = len(seg2num)
  n2 = len(seg2num) - len(emp2real)
  #print(n1)
  assert n1 == cnt and n2 == start
  raw_M=np.zeros((n1,n1))
  M=np.zeros((n1,n1))

  for seg in seg2num.keys():
    x=1/seg.sec.nseg/2
    idx=seg2num[seg]
    A=h.fmatrix(seg.x,1,sec=seg.sec)
    B=h.fmatrix(seg.x,3,sec=seg.sec)
    #D=h.fmatrix(seg.x,2,sec=seg.sec)

    par=seg2spar[seg]
    if par is None:
      paridx=-1
    else:
      if par.x==0.0 and seg.sec.parentseg() is not None:
        par=seg.sec.parentseg()
      paridx=seg2num[par]
    
    #M[idx,idx]=D*h.dt*1000/seg.cm

    if paridx >= 0:
      M[idx,paridx]=B*h.dt*1000/seg.cm
      M[paridx,idx]=A*h.dt*1000/par.cm


  for sec in seclist:
    for seg in sec:
      idx = seg2num[seg]
      M[idx, idx] = - np.sum(M[idx, :]) + 1 + h.dt*1000/seg.cm * seg.pas.g
      

  for seg in emp2real.keys():
    idx=seg2num[seg]
    M[idx,:]/=emp2real[seg].area()
    M[idx,:]*=1e2
    M[idx, idx] = - np.sum(M[idx, :])
  
  return tc.from_numpy(M)


def GradNeckLength(dictlist, neck, head, M, M_, N):
  seg2num=dictlist[0]
  seg2spar=dictlist[2]
  GradM=tc.zeros([len(M),len(M)])
  idx=seg2num[neck(0.5)]
  paridx=seg2num[seg2spar[neck(0.5)]]
  eidx=seg2num[neck(1)]
  hidx=seg2num[head(0.5)]
  GradM[paridx,paridx]=M[paridx,idx]/neck.L
  GradM[paridx, idx]=-GradM[paridx,paridx]
  GradM[idx,paridx]=-2/neck.L*M[idx,paridx]
  GradM[idx,eidx]=-2/neck.L*M[idx,eidx]
  GradM[idx,idx]=-GradM[idx,paridx]-GradM[idx,eidx]
  GradM[eidx,idx]=-2/neck.L*M[eidx,idx]
  GradM[eidx,hidx]=-1/neck.L*M[eidx,hidx]
  GradM[eidx,eidx]=-GradM[eidx,idx]-GradM[eidx,hidx]

  GradM=GradM.to(M_.device)
  res=-tc.matmul(M_,tc.matmul(GradM, M_))
  return res[0:N,0:N] 
  
def GradNeckDiam(dictlist, neck, head, M, M_, N):
  seg2num=dictlist[0]
  seg2par=dictlist[1]
  seg2spar=dictlist[2]
  emp2real=dictlist[3]
  GradM=tc.zeros([len(M),len(M)])

  idx=seg2num[neck(0.5)]
  paridx=seg2num[seg2spar[neck(0.5)]]
  eidx=seg2num[neck(1)]
  hidx=seg2num[head(0.5)]
  GradM[paridx,paridx]=-2*M[paridx,idx]/neck.diam
  GradM[paridx, idx]=-GradM[paridx,paridx]
  GradM[idx,paridx]=M[idx,paridx]/neck.diam
  GradM[idx,eidx]=M[idx,eidx]/neck.diam
  GradM[idx,idx]=-GradM[idx,paridx]-GradM[idx,eidx]
  GradM[eidx,idx]=1/neck.diam*M[eidx,idx]
  GradM[eidx,hidx]=-1/neck.diam*M[eidx,hidx]
  GradM[eidx,eidx]=-GradM[eidx,idx]-GradM[eidx,hidx]

  GradM=GradM.to(M_.device)
  res=-tc.matmul(M_,tc.matmul(GradM, M_))
  return res[0:N,0:N]  
  
def GradNeckResist(seclist, dictlist, neck_list, M, M_, N):
  seg2num=dictlist[0]
  seg2par=dictlist[1]
  seg2spar=dictlist[2]
  emp2real=dictlist[3]
  GradM=tc.zeros([len(M),len(M)])
  for neck in neck_list:
    idx=seg2num[neck(0.5)]
    paridx=seg2num[seg2spar[neck(0.5)]]
    par=seclist[paridx]
    eidx=seg2num[neck(1)]
    GradM[paridx,paridx]=M[paridx,idx]/neck.Ra
    GradM[paridx, idx]=-GradM[paridx,paridx]
    GradM[idx,paridx]=-M[idx,paridx]/neck.Ra
    GradM[idx,eidx]=-M[idx,eidx]/neck.Ra
    GradM[idx,idx]=-GradM[idx,paridx]-GradM[idx,eidx]
    GradM[eidx,idx]=-M[eidx,idx]/neck.Ra
    GradM[eidx,eidx]=-GradM[eidx,idx]

  GradM=GradM.to(M_.device)
  res=-tc.matmul(M_,tc.matmul(GradM, M_))
  return res[0:N,0:N]  

def GradHeadDiam(dictlist, head, M, M_, N):
  seg2num=dictlist[0]
  seg2par=dictlist[1]
  seg2spar=dictlist[2]
  emp2real=dictlist[3]
  GradM=tc.zeros([len(M),len(M)])
  idx=seg2num[head(0.5)]
  paridx=seg2num[head(0)]
  eidx=seg2num[head(1)]
  GradM[paridx,paridx]=-M[paridx,idx]/head.diam
  GradM[paridx, idx]=-GradM[paridx,paridx]
  GradM[idx,paridx]=-M[idx,paridx]/head.diam
  GradM[idx,eidx]=-M[idx,eidx]/head.diam
  GradM[idx,idx]=-GradM[idx,paridx]-GradM[idx,eidx]
  GradM[eidx,idx]=-M[eidx,idx]/head.diam
  GradM[eidx,eidx]=-GradM[eidx,idx]
    
  GradM=GradM.to(M_.device)
  res=-tc.matmul(M_,tc.matmul(GradM, M_))
  return res[0:N,0:N]  
  
def GradNeckLength_sparse(dictlist, neck, head, M):
  seg2num=dictlist[0]
  seg2spar=dictlist[2]
  idx=seg2num[neck(0.5)]
  paridx=seg2num[seg2spar[neck(0.5)]]
  eidx=seg2num[neck(1)]
  hidx=seg2num[head(0.5)]

  pidx= tc.tensor([paridx, idx, eidx, hidx])
  val = tc.zeros(8)
  val[0]= M[paridx,idx]/neck.L
  val[1]= -val[0]
  val[2]= -2/neck.L*M[idx,paridx]
  val[3]= -2/neck.L*M[idx,eidx]
  val[4]= -val[2]-val[3]
  val[5]= -2/neck.L*M[eidx,idx]
  val[6]= -1/neck.L*M[eidx,hidx]
  val[7]= -val[5]-val[6]

  return pidx, val
  
def GradNeckDiam_sparse(dictlist, neck, head, M):
  seg2num=dictlist[0]
  seg2spar=dictlist[2]
  idx=seg2num[neck(0.5)]
  paridx=seg2num[seg2spar[neck(0.5)]]
  eidx=seg2num[neck(1)]
  hidx=seg2num[head(0.5)]

  pidx= tc.tensor([paridx, idx, eidx, hidx])
  val = tc.zeros(8)
  val[0]= -2*M[paridx,idx]/neck.diam
  val[1]= -val[0]
  val[2]= M[idx,paridx]/neck.diam
  val[3]= M[idx,eidx]/neck.diam
  val[4]= -val[2]-val[3]
  val[5]= 1/neck.diam*M[eidx,idx]
  val[6]= -1/neck.diam*M[eidx,hidx]
  val[7]= -val[5]-val[6]

  return pidx, val

def GradHeadDiam_sparse(dictlist, head, M):
  seg2num=dictlist[0]
  seg2spar=dictlist[2]

  idx=seg2num[head(0.5)]
  paridx=seg2num[head(0)]
  eidx=seg2num[head(1)]
  hidx=seg2num[head(0.5)]

  pidx= tc.array([paridx, idx, eidx, hidx])
  val = tc.zeros(8)
  val[0]= -M[paridx,idx]/head.diam
  val[1]= -val[0]
  val[2]= -M[idx,paridx]/head.diam
  val[3]= -M[idx,eidx]/head.diam
  val[4]= -val[2]-val[3]
  val[5]= -M[eidx,idx]/head.diam
  val[6]= 0
  val[7]= -val[5]-val[6]

  return pidx, val
