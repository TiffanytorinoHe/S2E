"""Regenerate the four supplementary figures from the frozen source data.

Run from any directory: python figures/supplementary.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap,PowerNorm,LogNorm
from matplotlib.patches import Circle,Rectangle
from matplotlib.cm import ScalarMappable
from morphology import geom_arrays

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/figures';OUT=ROOT/'outputs/figures'
COND=['ampa','nmda','bhigh','sync'];LABELS=['Uniform AMPA','Uniform NMDA/AMPA','HF NMDA/AMPA','Sync NMDA/AMPA']
RED='#C9434F';BLUE='#237FB0';GREY='#727A80';INK='#25282A'
CMAP=LinearSegmentedColormap.from_list('inhibition',list(zip([0,.08,.24,.46,.72,1],['#C7C9CB','#BFC9D2','#83B1D8','#408ED0','#1974CC','#075EBE'])),N=256)
plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':8,'axes.titlesize':9,'axes.linewidth':.7,'axes.spines.top':False,'axes.spines.right':False,'xtick.labelsize':7,'ytick.labelsize':7,'xtick.major.size':3,'ytick.major.size':3,'legend.frameon':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
def save(fig,n):
 d=OUT/f'supplementary{n}';d.mkdir(parents=True,exist_ok=True)
 for ext in ['png','pdf','svg']:fig.savefig(d/f'Supplementary_Figure{n}.{ext}',dpi=360)
 plt.close(fig)
def label(fig,s,x,y):fig.text(x,y,s,fontsize=11,fontweight='bold',va='top')
def morph(ax,geometry,values=None,norm=None,model='l5pc'):
 lines,widths,intensity,soma,sv=geom_arrays(geometry,np.zeros(len(geometry)) if values is None else values,model)
 ax.add_collection(LineCollection(lines,colors='#B3B6B8' if values is None else CMAP(norm(intensity)),linewidths=widths*.113*1.5,capstyle='round'))
 center=np.mean(soma,axis=0);radius=max(np.ptp(soma,axis=0).max()*.44,6.8)
 ax.add_patch(Circle(center,radius,facecolor='#D9DBDD' if values is None else CMAP(norm(sv)),edgecolor='#999FA4',lw=.25,zorder=8))
 pts=lines.reshape(-1,2);lo,hi=pts.min(axis=0),pts.max(axis=0)
 ax.set(xlim=(lo[0]-35,hi[0]+35),ylim=(lo[1]-25,hi[1]+25));ax.set_aspect('equal');ax.axis('off')
 return lo,hi,center

def voltage():
 d=DATA/'supplementary1';rows=pd.read_csv(d/'autonomous_800_case_metrics.csv')
 assert len(rows)==800 and rows.groupby(['model','condition']).size().eq(50).all()
 models=['l5pc','ca1_20212','l23','msn'];names=['L5 PC','CA1 PC','L2/3 PC','MSN']
 fig=plt.figure(figsize=(7.3,7.5));gs=fig.add_gridspec(4,4,height_ratios=[1,1.15,1.15,1],hspace=.72,wspace=.35,left=.105,right=.98,top=.94,bottom=.07)
 for j,(m,name) in enumerate(zip(models,names)):
  z=pd.read_csv(d/f'{m}_ampa_000_displayed_trace.csv');ax=fig.add_subplot(gs[0,j])
  ax.plot(z.time_ms,z.neuron_soma_mV,c=GREY,lw=.9,label='NEURON');ax.plot(z.time_ms,z.autonomous_soma_mV,c=RED,lw=.7,ls='--',label='S2E')
  ax.set(xlim=(0,400),ylim=(-90,55),xticks=[0,200,400],yticks=[-80,0,40],xlabel='Time (ms)');ax.set_title(name)
  if j==0:ax.set_ylabel('Somatic voltage (mV)');fig.legend(*ax.get_legend_handles_labels(),ncol=2,loc='upper center',bbox_to_anchor=(.56,.995),fontsize=8)
  for k,(metric,ylabel,threshold) in enumerate([('max_abs_mV','Maximum error (mV)',1e-3),('rmse_mV','RMSE (mV)',1e-4)],1):
   ax=fig.add_subplot(gs[k,j]);rng=np.random.default_rng(17)
   for x,c in enumerate(COND):
    v=rows.loc[(rows.model==m)&(rows.condition==c),metric].to_numpy();assert len(v)==50
    ax.scatter(x+rng.uniform(-.17,.17,len(v)),v,s=7,c=GREY,alpha=.55,lw=0)
    ax.plot([x-.2,x+.2],[np.median(v)]*2,c=INK,lw=1.1)
   ax.axhline(threshold,c=RED,lw=.7,ls='--');ax.set_yscale('log');ax.set_ylim((1e-11,1e-2) if k==2 else (1e-9,1))
   ax.set(xticks=range(4),xticklabels=['AMPA','NMDA','HF','Sync'],xlim=(-.5,3.5));ax.tick_params(axis='x',rotation=40)
   if j==0:ax.set_ylabel(ylabel)
   else:ax.tick_params(labelleft=False)
 z=pd.read_csv(d/'ca1_20212_sync_040_displayed_trace.csv');ax=fig.add_subplot(gs[3,:2]);ax.plot(z.time_ms,z.neuron_soma_mV,c=GREY,lw=.9);ax.plot(z.time_ms,z.autonomous_soma_mV,c=RED,lw=.7,ls='--');ax.set(xlim=(0,400),ylim=(-85,55),xlabel='Time (ms)',ylabel='Somatic voltage (mV)');ax.set_title('CA1 Sync',loc='left')
 ax=fig.add_subplot(gs[3,2:]);pos=ax.get_position();ax.set_position([pos.x0+.03,pos.y0,pos.width-.03,pos.height]);ax.plot(z.time_ms,z.max_error_over_segments_mV,c=RED,lw=.8);ax.set(xlim=(0,400),ylim=(0,.135),xlabel='Time (ms)',ylabel='Maximum error (mV)',xticks=[0,100,200,300,400])
 for k,l in enumerate('abcd'):label(fig,l,.022,gs[k,0].get_position(fig).y1+.015)
 save(fig,1)

def exhaustive():
 d=pd.read_csv(DATA/'supplementary2/benchmark.csv');assert len(d)==234
 assert (d.greedy_increments-d.minimum_increments==d.difference).all()
 fig=plt.figure(figsize=(7.3,5.4));gs=fig.add_gridspec(2,2,left=.10,right=.91,bottom=.10,top=.94,hspace=.55,wspace=.36)
 ax=fig.add_subplot(gs[0,0]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
 ax.add_patch(Circle((.13,.67),.09,facecolor='#E6E8E9',edgecolor=INK,lw=.8))
 for j in range(3):ax.add_patch(Rectangle((.22+j*.23,.64),.23,.06,facecolor='#E6E8E9',edgecolor=INK,lw=.7))
 ax.text(.13,.84,'Soma',ha='center');ax.text(.58,.84,'Dendrite',ha='center');ax.text(.13,.67,'0',ha='center',va='center')
 for j in range(3):ax.text(.335+j*.23,.67,str(j+1),ha='center',va='center')
 source=json.loads((DATA/'supplementary2/example.json').read_text());loc=source['exc_segment_ids'];times=source['exc_times_ms']
 centers=[.13,.335,.565,.795]
 for k,l in enumerate(loc):ax.scatter(centers[l]+(k%2)*.025,.76,c=RED,s=10,zorder=3)
 for k,t in enumerate(times):ax.plot([.10+(t-20)/40*.76]*2,[.22+k*.035,.24+k*.035],c=RED,lw=1.1)
 for t in [20,30,40,50,60]:
  x=.10+(t-20)/40*.76;ax.plot([x,x],[.1,.19],c=BLUE,lw=.8);ax.text(x,.07,str(t),ha='center',va='top',fontsize=7)
 ax.text(.48,-.04,'Onset time (ms)',ha='center');ax.set_title('Model and candidate onsets',loc='left')
 ax=fig.add_subplot(gs[0,1]);n=int(max(d.greedy_increments.max(),d.minimum_increments.max()))+1
 h=np.zeros((n,n),int)
 for r in d.itertuples():h[r.greedy_increments,r.minimum_increments]+=1
 im=ax.imshow(np.ma.masked_equal(h,0),origin='lower',extent=(-.5,n-.5,-.5,n-.5),cmap='Blues',norm=LogNorm(vmin=1,vmax=h.max()),aspect='equal')
 for y,x in zip(*np.where(h>0)):ax.text(x,y,str(h[y,x]),ha='center',va='center',fontsize=6,color='white' if h[y,x]>h.max()*.3 else INK)
 ax.plot([-.5,n-.5],[-.5,n-.5],ls='--',c=GREY,lw=.7);ax.set(xlabel='Enumerated minimum',ylabel='S2E search count',xticks=[0,5,10],yticks=[0,5,10]);cb=fig.colorbar(im,ax=ax,fraction=.046,pad=.04);cb.set_label('Realizations')
 ax=fig.add_subplot(gs[1,0]);counts=d.difference.value_counts().sort_index();ax.bar(counts.index,counts.values,color=BLUE,width=.6);ax.set(xlabel='Excess search count',ylabel='Realizations',xticks=[0,1,2,3])
 ax=fig.add_subplot(gs[1,1]);x=np.arange(4);ax.plot(x,[(d.difference<=k).mean()*100 for k in x],'-o',c=BLUE,lw=1.2,ms=4);ax.set(xticks=x,ylim=(80,101),xlabel='Maximum count difference',ylabel='Cumulative fraction (%)')
 for l,x,y in [('a',.02,.975),('b',.52,.975),('c',.02,.47),('d',.52,.47)]:label(fig,l,x,y)
 save(fig,2)

def recordings():
 d=DATA/'supplementary3';events=pd.read_csv(d/'S3_excitatory_events.csv');traces=pd.read_csv(d/'S3_voltage_traces.csv');geom=json.loads((ROOT/'data/geometry/geometry_l5pc.json').read_text());sites=json.loads((d/'S3_recording_sites.json').read_text())
 fig=plt.figure(figsize=(7.3,6.0));gs=fig.add_gridspec(2,1,height_ratios=[1,2],left=.10,right=.98,top=.93,bottom=.08,hspace=.42);top=gs[0].subgridspec(1,4,wspace=.24);bottom=gs[1].subgridspec(4,5,width_ratios=[.7,1,1,1,1],wspace=.25,hspace=.18)
 for j,c in enumerate(COND):
  ax=fig.add_subplot(top[j]);q=events[events.condition==c];ax.scatter(q.time_ms,q.path_distance_um,s=2,c=RED,alpha=.55,lw=0);ax.set(xlim=(100,400),ylim=(0,1350),xticks=[100,250,400],yticks=[0,650,1300],xlabel='Time (ms)');ax.set_title(LABELS[j].replace(' ','\n',1) if j==0 else LABELS[j].replace('NMDA/AMPA','\nNMDA/AMPA'),fontsize=8)
  if j==0:ax.set_ylabel('Path distance (µm)')
  else:ax.tick_params(labelleft=False)
  for k,site in enumerate(['1','2','3','soma']):
   ax=fig.add_subplot(bottom[k,j+1]);q=traces[(traces.condition==c)&(traces.site.astype(str)==site)];assert len(q)>10000
   ax.axvspan(100,400,color='#F0F1F2',lw=0);ax.plot(q.time_ms,q.voltage_mV,c=INK,lw=.65);ax.set(xlim=(0,400),ylim=(-90,45),yticks=[-80,0],xticks=[0,200,400])
   if j==0:ax.set_ylabel(('Site '+site if k<3 else 'Soma')+' (mV)',fontsize=7)
   else:ax.tick_params(labelleft=False)
   if k<3:ax.tick_params(labelbottom=False)
   else:ax.set_xlabel('Time (ms)')
 ax=fig.add_subplot(bottom[:,0]);lo,hi,center=morph(ax,geom)
 for k,s in enumerate(sites):
  x,y=s['xy'];ax.scatter(x,y,s=16,facecolors='white',edgecolors=RED,lw=.8,zorder=10);ax.annotate(str(k+1),(x,y),xytext=(5,1),textcoords='offset points',fontsize=7)
 sx=lo[0]-16;sy=lo[1]+70;ax.plot([sx,sx],[sy,sy+200],c=INK,lw=.8);ax.text(sx-20,sy+100,'200 µm',rotation=90,ha='right',va='center',fontsize=7)
 label(fig,'a',.02,.98);label(fig,'b',.02,.63);save(fig,3)

def trains():
 d=DATA/'supplementary4';geom=json.loads((ROOT/'data/geometry/geometry_l5pc.json').read_text());maps=json.loads((d/'pooled_maps.json').read_text());df=pd.read_csv(d/'paired_realizations.csv');curves=pd.read_csv(d/'distance_profiles.csv')
 fig=plt.figure(figsize=(7.2,7.1));pooled=np.concatenate([maps[c+'_'+m] for c in COND for m in ['pulse','train']]);cap=float(np.percentile(pooled[pooled>0],95)*.85);norm=PowerNorm(.62,vmin=0,vmax=cap,clip=True)
 for j,c in enumerate(COND):
  x=.085+j*.225;fig.text(x+.091,.985,LABELS[j].replace('NMDA/AMPA','\nNMDA/AMPA'),ha='center',va='top',fontsize=8.5)
  for k,mode in enumerate(['pulse','train']):
   ax=fig.add_axes([x+k*.103,.655,.101,.282]);lo,hi,center=morph(ax,geom,np.array(maps[c+'_'+mode]),norm);ax.set_title(mode.capitalize(),fontsize=7.5,pad=2)
   if j==0 and k==0:
    sx=lo[0]-12;sy=lo[1]+70;ax.plot([sx,sx],[sy,sy+200],c=INK,lw=.75);ax.text(sx-15,sy+100,'200 µm',rotation=90,ha='right',va='center',fontsize=6.5)
  ax=fig.add_axes([x,.466,.182,.105])
  for mode,col in [('pulse',GREY),('train',BLUE)]:
   q=curves[(curves.condition==c)&(curves['mode']==mode)];ax.plot(q.distance_um,q.fraction*100,c=col,lw=1.1,label=mode.capitalize())
  ax.set(xlim=(0,1291.34),xticks=[0,650,1291],xlabel='Path distance (µm)',ylim=(0,None));ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,2));ax.yaxis.get_offset_text().set_fontsize(6)
  if j==0:ax.set_ylabel('Mean segment\nfraction (%)')
  if j==3:ax.legend(fontsize=7,handlelength=1.2)
 def pair(box,c,metric,title):
  ax=fig.add_axes(box);q=df[df.condition==c];a=q['pulse_'+metric].to_numpy()*100;b=q['train_'+metric].to_numpy()*100;jit=np.random.default_rng(7).uniform(-.105,.105,len(a));assert len(a)==50
  for x,y,z in zip(jit,a,b):ax.plot([x,1+x],[y,z],c='#C7CCD1',lw=.45,zorder=1)
  for x,v,col in [(0,a,GREY),(1,b,BLUE)]:ax.scatter(x+jit,v,s=8,c=col,alpha=.8,lw=0,zorder=2);ax.plot([x-.18,x+.18],[np.median(v)]*2,c=INK,lw=1.5,zorder=3)
  ax.set(xlim=(-.35,1.35),ylim=(-4,104),xticks=[0,1],xticklabels=['Pulse','Train'],yticks=[0,50,100]);ax.set_title(title,pad=6,fontsize=8)
  return ax
 for j,c in enumerate(COND):
  ax=pair([.085+j*.225,.265,.182,.10],c,'perisomatic','Perisomatic sites')
  if j==0:ax.set_ylabel('Candidates (%)')
 ax=pair([.085,.055,.405,.10],'bhigh','distal_source','HF: distal apical source');ax.set_ylabel('Candidates (%)')
 ax=pair([.56,.055,.405,.10],'sync','apical_trunk','Sync: intermediate apical region');ax.set_ylabel('Candidates (%)')
 for l,x,y in [('a',.025,.99),('b',.025,.60),('c',.025,.40),('d',.025,.19),('e',.505,.19)]:label(fig,l,x,y)
 cax=fig.add_axes([.30,.626,.46,.009]);cb=fig.colorbar(ScalarMappable(norm=norm,cmap=CMAP),cax=cax,orientation='horizontal',extend='max',extendfrac=.04);cb.set_ticks([0,cap]);cb.set_ticklabels(['0',f'{cap*100:.2f}']);cb.ax.tick_params(labelsize=6.5,pad=1,length=2);cb.set_label('Selected-candidate fraction per segment (%)',fontsize=7,labelpad=2);cb.outline.set_visible(False);cb.solids.set_rasterized(False);save(fig,4)

if __name__=='__main__':
 voltage();exhaustive();recordings();trains();print('Regenerated Supplementary Figures 1–4.')
