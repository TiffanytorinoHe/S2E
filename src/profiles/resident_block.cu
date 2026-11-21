#include "resident.cu"
// For few selected traces/target times, one CUDA thread per trace leaves most
// lanes idle. A whole block cooperates on states and tree levels instead.
struct Layout:Memory{
 int depth;int *level,*nodes,*children,*child,*slots,*slot;
 Layout(Context* c){
  std::vector<int> par(c->nn),order(c->nn),segment(c->K),dep(c->nn,0);
  ck(cudaMemcpy(par.data(),c->par,c->nn*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(order.data(),c->order,c->nn*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(segment.data(),c->seg,c->K*4,cudaMemcpyDeviceToHost));
  depth=0;for(int j:order){if(par[j]>=0)dep[j]=dep[par[j]]+1;depth=std::max(depth,dep[j]);}
  std::vector<int> lev(depth+2,0),nd,co(c->nn+1,0),ch,so(c->n+1,0),sl;
  for(int d=0;d<=depth;d++){for(int j:order)if(dep[j]==d)nd.push_back(j);lev[d+1]=nd.size();}
  // Preserve the exact child subtraction order of the original Hines sweep.
  for(int i=0;i<c->nn;i++){for(int k=c->nn-1;k>0;k--){int j=order[k];if(par[j]==i)ch.push_back(j);}co[i+1]=ch.size();}
  for(int i=0;i<c->n;i++){for(int k=0;k<c->K;k++)if(segment[k]==i)sl.push_back(k);so[i+1]=sl.size();}
  level=copy(lev.data(),lev.size());nodes=copy(nd.data(),nd.size());children=copy(co.data(),co.size());child=copy(ch.data(),ch.size());slots=copy(so.data(),so.size());slot=copy(sl.data(),sl.size());
 }
};
__global__ void reverse_block(int n,int nn,int K,int G,int P,int soma,int begin,int end,int depth,
 const int* par,const int* level,const int* nodes,const int* children,const int* child,const int* slots,const int* slot,
 const double* lower,const double* upper,const int* seg,const double* vc,const double* gc,const double* inv,const double* D,
 const int* target,const int* offsets,const int* member,const int* loc,double* vl,double* cl,double* gl,double* rr,double* out){
 int g=blockIdx.x,lane=threadIdx.x;auto lv=vl+(size_t)g*nn;auto lc=cl+(size_t)g*n;auto lg=gl+(size_t)g*K;auto rhs=rr+(size_t)g*nn;
 begin=min(begin,target[g]-1);
 for(int t=begin;t>=end;t--){
  if(t==target[g]-1 && lane==0)lv[soma]=1.;__syncthreads();
  auto a=vc+(size_t)t*4*n;auto b=gc+(size_t)t*5*K;auto di=inv+(size_t)t*nn;
  for(int i=lane;i<nn;i+=blockDim.x){
   double r=lv[i];if(i<n){double cc=lc[i];for(int q=slots[i];q<slots[i+1];q++){int k=slot[q];r+=b[3*K+k]*lg[k];cc+=b[4*K+k]*lg[k];}lc[i]=cc;}rhs[i]=r;
  }__syncthreads();
  for(int d=depth-1;d>=0;d--){
   for(int q=level[d]+lane;q<level[d+1];q+=blockDim.x){int i=nodes[q];double r=rhs[i];for(int k=children[i];k<children[i+1];k++){int j=child[k];r-=lower[j]*di[j]*rhs[j];}rhs[i]=r;}__syncthreads();
  }
  if(lane==0){int root=nodes[0];rhs[root]*=di[root];}__syncthreads();
  for(int d=1;d<=depth;d++){
   for(int q=level[d]+lane;q<level[d+1];q+=blockDim.x){int i=nodes[q];rhs[i]=(rhs[i]-upper[i]*rhs[par[i]])*di[i];}__syncthreads();
  }
  for(int q=offsets[g]+lane;q<offsets[g+1];q+=blockDim.x){int j=member[q];out[(size_t)t*P+j]=-D[loc[j]]*rhs[loc[j]];}
  for(int k=lane;k<K;k+=blockDim.x){int j=seg[k];lg[k]=b[2*K+k]*lg[k]+b[k]*rhs[j]+b[K+k]*lc[j];}__syncthreads();
  for(int i=lane;i<nn;i+=blockDim.x){if(i<n){double q=rhs[i],ca=lc[i];lv[i]=a[i]*q+a[3*n+i]*ca;lc[i]=a[n+i]*q+a[2*n+i]*ca;}else lv[i]=0.;}__syncthreads();
 }
}
__global__ void forward_block(int n,int nn,int K,int T,int P,int W,int soma,int rec,int begin,int end,int depth,
 const int* par,const int* level,const int* nodes,const int* children,const int* child,const int* slots,const int* slot,
 const double* lower,const int* seg,const double* vc,const double* gc,const double* diag,const double* factor,
 const double* vn,const double* wave,const double* D,const int* loc,const int* wi,const int* first,
 double* vv,double* cc,double* gg,double* rr,double* minimum,double* trace){
 int p=blockIdx.x,lane=threadIdx.x;auto v=vv+(size_t)p*nn;auto ca=cc+(size_t)p*n;auto gate=gg+(size_t)p*K;auto rhs=rr+(size_t)p*nn;
 begin=max(begin,first[wi[p]]);
 for(int t=begin;t<end;t++){
  auto a=vc+(size_t)t*4*n;auto b=gc+(size_t)t*5*K;auto di=diag+(size_t)t*nn;auto f=factor+(size_t)t*nn;
  for(int i=lane;i<nn;i+=blockDim.x){
   double r=0.;if(i<n){double old=v[i],ci=ca[i];r=a[i]*old+a[n+i]*ci;ci=a[2*n+i]*ci+a[3*n+i]*old;
    for(int q=slots[i];q<slots[i+1];q++){int k=slot[q];r+=b[k]*gate[k];ci+=b[K+k]*gate[k];}ca[i]=ci;}rhs[i]=r;
  }__syncthreads();
  if(lane==0){int l=loc[p];rhs[l]+=D[l]*wave[(size_t)t*W+wi[p]]*(-70.-vn[(size_t)t*n+l]);}__syncthreads();
  for(int d=depth-1;d>=0;d--){
   for(int q=level[d]+lane;q<level[d+1];q+=blockDim.x){int i=nodes[q];double r=rhs[i];for(int k=children[i];k<children[i+1];k++){int j=child[k];r-=f[j]*rhs[j];}rhs[i]=r;}__syncthreads();
  }
  if(lane==0){int root=nodes[0];v[root]=rhs[root]/di[root];}__syncthreads();
  for(int d=1;d<=depth;d++){
   for(int q=level[d]+lane;q<level[d+1];q+=blockDim.x){int i=nodes[q];v[i]=(rhs[i]-lower[i]*v[par[i]])/di[i];}__syncthreads();
  }
  for(int k=lane;k<K;k+=blockDim.x){int j=seg[k];gate[k]=b[2*K+k]*gate[k]+b[3*K+k]*v[j]+b[4*K+k]*ca[j];}
  if(lane==0){double value=v[soma];minimum[p]=fmin(minimum[p],value);if(rec)trace[(size_t)(t+1)*P+p]=value;}__syncthreads();
 }
}
extern "C" int kp_reverse_block(void* p,int G,int P,const int* ht,const int* hoff,const int* hm,const int* hl,double* output){
 try{
  auto c=static_cast<Context*>(p);Memory m;Layout layout(c);
  auto target=m.copy(ht,G),offset=m.copy(hoff,G+1),member=m.copy(hm,P),loc=m.copy(hl,P);
  auto lv=m.zero((size_t)c->nn*G),lc=m.zero((size_t)c->n*G),lg=m.zero((size_t)c->K*G),rhs=m.zero((size_t)c->nn*G),out=m.zero((size_t)c->T*P);
  int stop=0;for(int i=0;i<G;i++)stop=std::max(stop,ht[i]);
  for(int t=stop-1;t>=0;t-=64)reverse_block<<<G,128>>>(c->n,c->nn,c->K,G,P,c->soma,t,std::max(t-63,0),layout.depth,c->par,layout.level,layout.nodes,layout.children,layout.child,layout.slots,layout.slot,c->lower,c->upper,c->seg,c->vc,c->gc,c->inv,c->D,target,offset,member,loc,lv,lc,lg,rhs,out);
  ck(cudaGetLastError());ck(cudaDeviceSynchronize());ck(cudaMemcpy(output,out,(size_t)c->T*P*8,cudaMemcpyDeviceToHost));return 0;
 }catch(const std::exception& e){fprintf(stderr,"kp_reverse_block: %s\n",e.what());return 1;}
}
extern "C" int kp_forward_block(void* p,int P,const int* hl,const int* hw,int record,double* minimum,double* trace){
 try{
  auto c=static_cast<Context*>(p);Memory m;Layout layout(c);auto loc=m.copy(hl,P),wi=m.copy(hw,P);
  auto v=m.zero((size_t)c->nn*P),ca=m.zero((size_t)c->n*P),g=m.zero((size_t)c->K*P),rhs=m.zero((size_t)c->nn*P),mi=m.zero(P),tr=m.zero(record?(size_t)(c->T+1)*P:1);
  for(int t=0;t<c->T;t+=64)forward_block<<<P,128>>>(c->n,c->nn,c->K,c->T,P,c->W,c->soma,record,t,std::min(t+64,c->T),layout.depth,c->par,layout.level,layout.nodes,layout.children,layout.child,layout.slots,layout.slot,c->lower,c->seg,c->vc,c->gc,c->diag,c->factor,c->vn,c->wave,c->D,loc,wi,c->first,v,ca,g,rhs,mi,tr);
  ck(cudaGetLastError());ck(cudaDeviceSynchronize());ck(cudaMemcpy(minimum,mi,P*8,cudaMemcpyDeviceToHost));if(record)ck(cudaMemcpy(trace,tr,(size_t)(c->T+1)*P*8,cudaMemcpyDeviceToHost));return 0;
 }catch(const std::exception& e){fprintf(stderr,"kp_forward_block: %s\n",e.what());return 1;}
}
