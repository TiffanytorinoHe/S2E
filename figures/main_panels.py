"""Reconstruct numerical main-figure panels; conceptual artwork is supplied separately.

All panels use the frozen publication cohorts. Exported SVG text remains editable.
"""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from supplementary import morph,CMAP,RED,BLUE,GREY,INK
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/figures';OUT=ROOT/'outputs/figures'
sys.path.insert(0,str(ROOT/'analysis'))
from iteration_alignment import load_iterations
PATTERNS=['ampa','nmda','bhigh','sync'];LABELS=['Uni AMPA','Uni NMDA','HF','Sync']
MODELS=['l5pc','ca1_20212','l23','msn'];MODEL_NAMES=['L5 PC','CA1 PC','L2/3 PC','MSN']
COLORS=['#3E4C59','#6687A5','#4C827C','#8E8072'];MARKERS=['o','s','^','D'];GOLD='#AF892B'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def table(n,f):return pd.read_csv(DATA/f'figure{n}'/f)
def save(fig,n,panel):
 d=OUT/f'figure{n}';d.mkdir(parents=True,exist_ok=True)
 for ext in ['png','pdf','svg']:fig.savefig(d/f'Fig{n}{panel}.{ext}',dpi=320,bbox_inches='tight')
 plt.close(fig)
def figure1():
 d=table(1,'voltage_trace.csv');fig,ax=plt.subplots(figsize=(3.2,1.8))
 ax.plot(d.time_ms,d.NEURON_mV,c=GREY,lw=1,label='NEURON');ax.plot(d.time_ms,d.S2E_iterated_mV,c=RED,lw=.9,ls='--',label='S2E')
 ax.set(xlabel='Time (ms)',ylabel='Somatic voltage (mV)',xlim=(0,400));ax.legend(ncol=2,fontsize=7);save(fig,1,'b_voltage')
def figure2():
 z=np.load(DATA/'figure2/example_voltages.npz');fig,axs=plt.subplots(4,1,figsize=(3.3,5.1),layout='constrained',sharex=True)
 for ax,p,l in zip(axs,PATTERNS,LABELS):
  ax.plot(z[p+'_time_ms'],z[p+'_before'],c=GREY,lw=.7,label='Before');ax.plot(z[p+'_time_ms'],z[p+'_after'],c=BLUE,lw=.7,label='After');ax.set_title(l,loc='left');ax.set(xlim=(100,400),ylim=(-85,50),ylabel='Voltage (mV)')
 axs[0].legend(ncol=2,fontsize=7);axs[-1].set_xlabel('Time (ms)');save(fig,2,'c_voltage')
 for p in ['bhigh','sync']:
  fig,axs=plt.subplots(8,2,figsize=(5,6),sharex=True,layout='constrained')
  for i in range(8):
   t=z[p+'_time_ms'];b=z[p+'_local_before'][i];a=z[p+'_local_after'][i]
   axs[i,0].plot(t,b,c=GREY,lw=.7);axs[i,0].plot(t,a,c=BLUE,lw=.7);axs[i,1].plot(t,a-b,c=INK,lw=.7)
   for ax in axs[i]:ax.set_xlim(100,400)
   axs[i,0].set_ylabel('soma' if i==7 else str(i+1))
  axs[0,0].set_title('Membrane voltage (mV)');axs[0,1].set_title('Voltage change (mV)')
  for ax in axs[-1]:ax.set_xlabel('Time (ms)')
  save(fig,2,'d_'+p)
 geo=read(DATA/'figure2/geometry_l5pc.json');freq=table(2,'segment_frequencies.csv');prof=table(2,'distance_profiles.csv')
 fig=plt.figure(figsize=(7.2,3));gs=fig.add_gridspec(1,8,width_ratios=[2,1]*4,wspace=.28)
 for i,(p,l) in enumerate(zip(PATTERNS,LABELS)):
  a=fig.add_subplot(gs[0,2*i]);v=np.zeros(len(geo));q=freq[freq.pattern==p];v[q.segment_id.astype(int)]=q.fraction
  morph(a,geo,v,PowerNorm(.5,0,.012,clip=True));a.set_title(l)
  ax=fig.add_subplot(gs[0,2*i+1]);q=prof[prof.pattern==p];ax.plot(q.normalized_profile,q.distance_bin_center_um,c=BLUE,lw=1);ax.fill_betweenx(q.distance_bin_center_um,0,q.normalized_profile,color=BLUE,alpha=.12)
  top=max(r['distance_um'] for r in geo if 'axon' not in r['section']);ax.set(ylim=(0,top),yticks=[0,top],yticklabels=['0',f'{top:.0f}'],xticks=[0,1],xticklabels=['0','Max'],xlabel='Rel. freq.')
 fig.subplots_adjust(right=.90)
 cb=fig.colorbar(plt.cm.ScalarMappable(norm=PowerNorm(.5,0,.012,clip=True),cmap=CMAP),cax=fig.add_axes([.93,.22,.012,.56]))
 cb.set_ticks([0,.012],labels=['0','1.2']);cb.set_label('Selection frequency (%)')
 save(fig,2,'e')
def figure3():
 exc=table(3,'raster_excitation_v7.csv');inh=table(3,'raster_inhibition_v7.csv');joint=table(3,'joint_example_coordinates_v7.csv');stats=table(3,'timing_statistics_all_completed.csv')
 fig,axs=plt.subplots(2,4,figsize=(7.1,2.8),layout='constrained',sharex=True)
 for j,(p,l) in enumerate(zip(PATTERNS,LABELS)):
  for i,frame,col,timecol in [(0,exc,RED,'event_ms'),(1,inh,BLUE,'onset_ms')]:
   q=frame[frame.pattern==p];axs[i,j].scatter(q[timecol],q.display_row,s=2,c=col,marker='|');axs[i,j].set(xlim=(100,400),yticks=[]);axs[i,j].spines[['left','right','top']].set_visible(False)
  axs[0,j].set_title(l);axs[1,j].set_xlabel('Time (ms)')
 axs[0,0].set_ylabel('Excitation');axs[1,0].set_ylabel('Inhibition');save(fig,3,'a')
 fig,axs=plt.subplots(1,4,figsize=(7.1,2.2),sharey=True,layout='constrained')
 for ax,p,l in zip(axs,PATTERNS,LABELS):
  q=joint[joint.pattern==p];ax.scatter(q.onset_ms,q.distance_um,s=q.area_pt2,c=BLUE,edgecolors='white',lw=.2);ax.set_title(l);ax.set(xlim=(85,415),ylim=(-30,1320),xlabel='Inhibition onset (ms)')
 axs[0].set_ylabel('Soma distance (µm)');save(fig,3,'e')
 fig,ax=plt.subplots(figsize=(3.3,2.4));rng=np.random.default_rng(22)
 for i,p in enumerate(PATTERNS):
  v=stats[stats.pattern==p].rho_time_distance.dropna().to_numpy();ax.scatter(v,i+rng.uniform(-.16,.16,len(v)),s=8,color=BLUE,alpha=.5,lw=0);lo,md,hi=np.quantile(v,[.25,.5,.75]);ax.plot([lo,hi],[i,i],c=INK,lw=1.2);ax.plot([md,md],[i-.13,i+.13],c=INK,lw=1)
 ax.axvline(0,c=GREY,ls='--',lw=.7);ax.set(xlim=(-1,1),yticks=range(4),yticklabels=LABELS,xlabel='Within-case Spearman ρ');ax.invert_yaxis();save(fig,3,'f')
def figure4():
 d=table(4,'spatial_plotted.csv');fig,axs=plt.subplots(1,4,figsize=(7.1,2.2),layout='constrained',sharey=True)
 for ax,p,l in zip(axs,PATTERNS,LABELS):
  for q,c,lab in [('phi',RED,'Transfer'),('a',BLUE,'Drive'),('score',GOLD,'Somatic response')]:
   part=d[(d.pattern==p)&(d.quantity==q)]
   if part.empty:
    q={'a':'a_syn','dv':'delta_v'}.get(q,q);part=d[(d.pattern==p)&(d.quantity==q)]
   if part.empty:raise ValueError('Missing Fig. 4 profile: '+p+'/'+q)
   ax.plot(part.distance_um,part.normalized,c=c,lw=1,label=lab)
  ax.set_title(l);ax.set(xlabel='Distance (µm)',ylim=(0,1.1))
 axs[0].set_ylabel('Normalized profile');axs[0].legend(fontsize=6);save(fig,4,'c')
 path=DATA/'figure4';z,report=load_iterations(path/'hf_iterations.npz',path/'sync_iterations.npz')
 fig,axs=plt.subplots(5,2,figsize=(6.4,5),layout='constrained',sharex=True)
 for j,p in enumerate(['bhigh','sync']):
  for i,step in enumerate(z[p+'_steps']):
   phi=z[p+'_phi'][i];a=z[p+'_asyn'][i];a=a/abs(a).max();phi=phi/abs(phi).max();a=np.where(abs(a)<.001,np.nan,a)
   axs[i,j].plot(z['time_ms'],phi,c=RED,lw=.8);axs[i,j].plot(z['time_ms'],a,c=BLUE,lw=.8);axs[i,j].set(yticks=[],ylabel=str(step+1),xlim=(0,400),ylim=(-1.1,1.1));axs[i,j].spines[['left','bottom']].set_visible(False)
  axs[0,j].set_title('HF NMDA/AMPA' if j==0 else 'Sync NMDA/AMPA');axs[-1,j].set_xlabel('Time (ms)')
 save(fig,4,'d')
 widths=table(4,'first_event_widths.csv');fig,axs=plt.subplots(2,1,figsize=(4.4,3.8),layout='constrained');rng=np.random.default_rng(42)
 for i,p in enumerate(PATTERNS):
  q=widths[widths.pattern==p]
  for off,field,c in [(-.14,'phi_fwhm_ms',RED),(.14,'a_fwhm_ms',BLUE)]:
   v=q[field];axs[0].scatter(i+off+rng.uniform(-.06,.06,len(v)),v,s=6,c=c,alpha=.5,lw=0);axs[0].plot([i+off-.08,i+off+.08],[v.median()]*2,c=c)
  v=q.a_peak_ms-q.phi_peak_ms;axs[1].scatter(i+rng.uniform(-.13,.13,len(v)),v,s=6,c=GREY,alpha=.6,lw=0);axs[1].plot([i-.12,i+.12],[v.median()]*2,c=INK)
 axs[0].set_ylabel('FWHM (ms)');axs[1].set_ylabel('Drive − transfer peak (ms)');axs[1].axhline(0,c=GREY,ls='--',lw=.7)
 for ax in axs:ax.set(xticks=range(4),xticklabels=LABELS)
 save(fig,4,'ef')
def figure5():
 summary=table(5,'condition_summary.csv');contrasts=table(5,'contrasts.csv')
 fig,axs=plt.subplots(1,2,figsize=(6.6,2.8),layout='constrained')
 for i,(model,name,c,marker) in enumerate(zip(MODELS,MODEL_NAMES,COLORS,MARKERS)):
  for ax,metric in zip(axs,['mean_normalized_distance','temporal_concentration_gini30']):
   q=summary[(summary.model==model)&(summary.metric==metric)].set_index('pattern').loc[PATTERNS];ax.errorbar(range(4),q['mean'],yerr=[q['mean']-q.ci_low,q.ci_high-q['mean']],c=c,marker=marker,ms=4,lw=.8,capsize=2,label=name)
 for ax in axs:ax.set(xticks=range(4),xticklabels=LABELS,ylim=(0,1))
 axs[0].set_ylabel('Mean normalized distance');axs[1].set_ylabel('Temporal concentration (Gini)');axs[0].legend(fontsize=7);save(fig,5,'bc')
 fig,axs=plt.subplots(1,3,figsize=(7.1,2.5),layout='constrained')
 for ax,contrast,title in zip(axs,['outward','return','concentration'],['HF − uniform AMPA','Sync − HF','Timing: Sync − HF']):
  for i,model in enumerate(MODELS):
   q=contrasts[(contrasts.model==model)&(contrasts.contrast==contrast)].iloc[0];ax.errorbar(q.estimate,i,xerr=[[q.estimate-q.ci_low],[q.ci_high-q.estimate]],c=COLORS[i],marker=MARKERS[i],ms=4,lw=1,capsize=2)
  ax.set(yticks=range(4),yticklabels=MODEL_NAMES,xlabel='Change in Gini' if contrast=='concentration' else 'Change in normalized distance');ax.set_title(title);ax.invert_yaxis();ax.axvline(0,c=GREY,lw=.7,ls='--')
 save(fig,5,'d')
 fig=plt.figure(figsize=(7.2,8));gs=fig.add_gridspec(4,8,width_ratios=[2,1]*4,wspace=.3,hspace=.24)
 for row,model in enumerate(MODELS):
  geo=read(ROOT/f'data/geometry/geometry_{model}.json')
  all_values=[]
  for p in PATTERNS:
   cohort=read(ROOT/f'data/solutions/pulse/{model}_{p}.json');size=len(cohort['geometry']['distance_um'])
   all_values.append(np.mean([np.bincount(np.array(r['result']['ans'])//31,minlength=size)/r['result']['length'] for r in cohort['results']],axis=0))
  pooled=np.concatenate(all_values);cap=np.percentile(pooled[pooled>0],95)
  for col,p in enumerate(PATTERNS):
   cohort=read(ROOT/f'data/solutions/pulse/{model}_{p}.json');distance=np.array(cohort['geometry']['distance_um']);n=cohort['geometry']['nonaxon_segments'];values=np.mean([np.bincount(np.array(r['result']['ans'])//31,minlength=len(distance))/r['result']['length'] for r in cohort['results']],axis=0)
   ax=fig.add_subplot(gs[row,2*col]);v=np.zeros(len(geo));v[:len(values)]=values;morph(ax,geo,v/cap,PowerNorm(.55,0,1,clip=True),model=model)
   if row==0:ax.set_title(LABELS[col])
   if col==0:ax.text(-.2,.5,MODEL_NAMES[row],transform=ax.transAxes,ha='right',va='center')
   width=[60,55,18,12][row];edges=np.arange(0,distance[:n].max()+width,width);idx=np.digitize(distance[:n],edges)-1
   y=[];x=[]
   for k in range(len(edges)-1):
    mask=idx==k
    if mask.any():y.append(min((edges[k]+edges[k+1])/2,distance[:n].max()));x.append(values[:n][mask].mean())
   x=np.convolve(np.pad(x,2,mode='edge'),np.array([1,2,3,2,1])/9,mode='valid');x=x/x.max();ax=fig.add_subplot(gs[row,2*col+1]);ax.plot(x,y,c=BLUE,lw=.9);ax.fill_betweenx(y,0,x,color=BLUE,alpha=.12);top=distance[:n].max();ax.set(ylim=(0,top),yticks=[0,top],yticklabels=['0',str(round(top))],xticks=[0,1],xticklabels=['0','Max'])
 fig.subplots_adjust(left=.12,right=.89)
 cb=fig.colorbar(plt.cm.ScalarMappable(norm=PowerNorm(.55,0,1,clip=True),cmap=CMAP),cax=fig.add_axes([.93,.40,.012,.22]))
 cb.set_ticks([0,.25,1],labels=['0','.25','≥1']);cb.set_label('Relative selection frequency')
 save(fig,5,'a')
def figure6():
 models=['L5bPCmodelsEH','CA1_20212','L23','MSN'];profiles=table(6,'displayed_profiles.csv');fig,axs=plt.subplots(2,2,figsize=(5.2,4),layout='constrained')
 for ax,model,name in zip(axs.ravel(),models,MODEL_NAMES):
  for p,ls in zip(PATTERNS,['-','--','-',':']):
   q=profiles[(profiles.model==model)&(profiles.regime==p)];ax.plot(q.distance_um,q.normalized,c=GREY if p=='ampa' else RED,ls=ls,lw=1,label=LABELS[PATTERNS.index(p)])
  ax.set_title(name);ax.set(xlabel='Distance (µm)',ylabel='Normalized transfer',ylim=(0,1.1))
 axs[0,0].legend(fontsize=6);save(fig,6,'a_profiles')
 e=table(6,'elasticity_seed_means.csv');summary=table(6,'ephi_summary.csv');contrib=table(6,'absolute_contributions_seed_means.csv');cs=table(6,'absolute_contribution_summary.csv')
 fig,axs=plt.subplots(1,3,figsize=(7.1,2.8),layout='constrained');rng=np.random.default_rng(83)
 for i,(model,name) in enumerate(zip(models,MODEL_NAMES)):
  v=e[(e.model==model)&(e.bins==8)].E_phi;q=summary[summary.model==model].iloc[0];axs[0].scatter(i+rng.uniform(-.15,.15,len(v)),v,s=8,c=GREY,alpha=.4,lw=0);axs[0].errorbar(i,q.E_phi,yerr=[[q.E_phi-q.ci_low],[q.ci_high-q.E_phi]],fmt=MARKERS[i],c=RED,capsize=2,ms=5)
  q=cs[cs.model==model].iloc[0]
  for offset,key,c in [(-.19,'abs_phi',RED),(.19,'abs_a',BLUE)]:
   value=q[key];axs[1].bar(i+offset,value,width=.34,color=c);axs[1].errorbar(i+offset,value,yerr=[[value-q[key+'_ci_low']],[q[key+'_ci_high']-value]],fmt='none',c=INK,lw=.7,capsize=2)
  v=contrib[contrib.model==model].paired_contrast;axs[2].scatter(v,i+rng.uniform(-.14,.14,len(v)),s=8,c=GREY,alpha=.4,lw=0);axs[2].errorbar(q.mean_difference,i,xerr=[[q.mean_difference-q.difference_ci_low],[q.difference_ci_high-q.mean_difference]],fmt=MARKERS[i],c=COLORS[i],capsize=2)
 for ax in axs[:2]:ax.set(xticks=range(4),xticklabels=MODEL_NAMES);ax.tick_params(axis='x',rotation=30)
 axs[0].set_ylabel('Coupling reshaping EΦ');axs[1].set_ylabel('Absolute contribution');axs[2].set(yticks=range(4),yticklabels=MODEL_NAMES,xlabel='Transfer − drive');axs[2].invert_yaxis();axs[2].axvline(0,c=GREY,ls='--',lw=.7);save(fig,6,'bcd')
if __name__=='__main__':
 for fn in [figure1,figure2,figure3,figure4,figure5,figure6]:fn();print(fn.__name__,'complete',flush=True)
