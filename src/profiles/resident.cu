#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <cstdio>
#include "forward_kernel.cuh"
// Shared coefficients stay on the device across scoring, selected traces and
// reverse passes. No changes to the scientific recurrence in forward_kernel.
static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Memory {
 std::vector<void*> items;
 ~Memory(){for(auto p:items)cudaFree(p);}
 template<class T>T* copy(const T* h,size_t n){T* d=nullptr;ck(cudaMalloc(&d,n*sizeof(T)));items.push_back(d);ck(cudaMemcpy(d,h,n*sizeof(T),cudaMemcpyHostToDevice));return d;}
 double* zero(size_t n){double* d=nullptr;ck(cudaMalloc(&d,n*sizeof(double)));items.push_back(d);ck(cudaMemset(d,0,n*sizeof(double)));return d;}
};
struct Context:Memory{
 int n,nn,K,T,W,soma;
 int *par,*order,*seg,*first;
 double *lower,*upper,*vc,*gc,*diag,*factor,*vn,*wave,*D,*inv;
};
__global__ void invert(const double* d,double* inv,size_t count){size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<count)inv[i]=1./d[i];}
extern "C" void* kp_create(int n,int nn,int K,int T,int W,int soma,
 const int* par,const int* order,const int* seg,const double* lower,const double* upper,
 const double* vc,const double* gc,const double* diag,const double* factor,
 const double* vn,const double* wave,const double* D){
 Context* c=new Context();
 try{
  c->n=n;c->nn=nn;c->K=K;c->T=T;c->W=W;c->soma=soma;
  c->par=c->copy(par,nn);c->order=c->copy(order,nn);c->seg=c->copy(seg,K);
  c->lower=c->copy(lower,nn);c->upper=c->copy(upper,nn);
  c->vc=c->copy(vc,(size_t)T*4*n);c->gc=c->copy(gc,(size_t)T*5*K);
  c->diag=c->copy(diag,(size_t)T*nn);c->factor=c->copy(factor,(size_t)T*nn);
  c->vn=c->copy(vn,(size_t)T*n);c->wave=c->copy(wave,(size_t)T*W);c->D=c->copy(D,n);
  std::vector<int> first(W,T);for(int j=0;j<W;j++)for(int t=0;t<T;t++)if(wave[(size_t)t*W+j]!=0.){first[j]=t;break;}
  c->first=c->copy(first.data(),W);c->inv=c->zero((size_t)T*nn);
  invert<<<((size_t)T*nn+255)/256,256>>>(c->diag,c->inv,(size_t)T*nn);ck(cudaGetLastError());ck(cudaDeviceSynchronize());return c;
 }catch(const std::exception& e){fprintf(stderr,"kp_create: %s\n",e.what());delete c;return nullptr;}
}
extern "C" void kp_destroy(void* p){delete static_cast<Context*>(p);}
extern "C" int kp_forward(void* p,int P,const int* hloc,const int* hwi,int record,double* minimum,double* trace){
 try{
  auto c=static_cast<Context*>(p);Memory m;auto loc=m.copy(hloc,P);auto wi=m.copy(hwi,P);
  auto v=m.zero((size_t)c->nn*P),ca=m.zero((size_t)c->n*P),g=m.zero((size_t)c->K*P),rhs=m.zero((size_t)c->nn*P),mi=m.zero(P),tr=m.zero(record?(size_t)(c->T+1)*P:1);
  for(int t=0;t<c->T;t+=64)propagate<<<(P+127)/128,128>>>(c->n,c->nn,c->K,c->T,P,c->W,c->soma,record,t,std::min(t+64,c->T),c->par,c->order,c->lower,c->seg,c->vc,c->gc,c->diag,c->factor,c->vn,c->wave,c->D,loc,wi,v,ca,g,rhs,mi,tr,c->first);
  ck(cudaGetLastError());ck(cudaDeviceSynchronize());ck(cudaMemcpy(minimum,mi,P*8,cudaMemcpyDeviceToHost));
  if(record)ck(cudaMemcpy(trace,tr,(size_t)(c->T+1)*P*8,cudaMemcpyDeviceToHost));return 0;
 }catch(const std::exception& e){fprintf(stderr,"kp_forward: %s\n",e.what());return 1;}
}
// A reverse state depends on its output time, not on the source location.
// Deduplicate identical endpoints exactly; gather all requested source locations.
__global__ void reverse_grouped(int n,int nn,int K,int G,int P,int soma,int begin,int end,
 const int* par,const int* order,const double* lower,const double* upper,const int* seg,
 const double* vc,const double* gc,const double* inv,const double* D,
 const int* target,const int* offsets,const int* member,const int* loc,
 double* lv,double* lc,double* lg,double* rhs,double* out){
 int g=blockIdx.x*blockDim.x+threadIdx.x;if(g>=G)return;
 begin=min(begin,target[g]-1);
 for(int t=begin;t>=end;t--){
  if(t==target[g]-1)lv[(size_t)soma*G+g]=1.;
  auto a=vc+(size_t)t*4*n;auto b=gc+(size_t)t*5*K;auto di=inv+(size_t)t*nn;
  for(int i=0;i<nn;i++){size_t z=(size_t)i*G+g;rhs[z]=lv[z];}
  for(int k=0;k<K;k++){size_t z=(size_t)k*G+g,j=(size_t)seg[k]*G+g;rhs[j]+=b[3*K+k]*lg[z];lc[j]+=b[4*K+k]*lg[z];}
  for(int k=nn-1;k>0;k--){int j=order[k];rhs[(size_t)par[j]*G+g]-=lower[j]*di[j]*rhs[(size_t)j*G+g];}
  int root=order[0];rhs[(size_t)root*G+g]*=di[root];
  for(int k=1;k<nn;k++){int j=order[k];rhs[(size_t)j*G+g]=(rhs[(size_t)j*G+g]-upper[j]*rhs[(size_t)par[j]*G+g])*di[j];}
  for(int q=offsets[g];q<offsets[g+1];q++){int j=member[q];out[(size_t)t*P+j]=-D[loc[j]]*rhs[(size_t)loc[j]*G+g];}
  for(int k=0;k<K;k++){size_t z=(size_t)k*G+g,j=(size_t)seg[k]*G+g;lg[z]=b[2*K+k]*lg[z]+b[k]*rhs[j]+b[K+k]*lc[j];}
  for(int i=0;i<nn;i++){size_t z=(size_t)i*G+g;if(i<n){double q=rhs[z],ca=lc[z];lv[z]=a[i]*q+a[3*n+i]*ca;lc[z]=a[n+i]*q+a[2*n+i]*ca;}else lv[z]=0.;}
 }
}
extern "C" int kp_reverse(void* p,int G,int P,const int* ht,const int* hoff,const int* hm,const int* hl,double* output){
 try{
  auto c=static_cast<Context*>(p);Memory m;
  auto target=m.copy(ht,G),offset=m.copy(hoff,G+1),member=m.copy(hm,P),loc=m.copy(hl,P);
  auto lv=m.zero((size_t)c->nn*G),lc=m.zero((size_t)c->n*G),lg=m.zero((size_t)c->K*G),rhs=m.zero((size_t)c->nn*G),out=m.zero((size_t)c->T*P);
  int stop=0;for(int i=0;i<G;i++)stop=std::max(stop,ht[i]);
  for(int t=stop-1;t>=0;t-=64)reverse_grouped<<<(G+127)/128,128>>>(c->n,c->nn,c->K,G,P,c->soma,t,std::max(t-63,0),c->par,c->order,c->lower,c->upper,c->seg,c->vc,c->gc,c->inv,c->D,target,offset,member,loc,lv,lc,lg,rhs,out);
  ck(cudaGetLastError());ck(cudaDeviceSynchronize());ck(cudaMemcpy(output,out,(size_t)c->T*P*8,cudaMemcpyDeviceToHost));return 0;
 }catch(const std::exception& e){fprintf(stderr,"kp_reverse: %s\n",e.what());return 1;}
}
