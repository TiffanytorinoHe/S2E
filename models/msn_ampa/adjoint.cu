#include <cuda_runtime.h>
#include <vector>
#include <chrono>
#include <cstdio>
// Reverse of the validated discrete forward step, including calcium and gates.
// One endpoint is the soma voltage at a specified time. Only the requested
// source segment is exported, while every state participates in propagation.
__global__ void reverse_steps(int n,int nn,int K,int P,int soma,int begin,int end,
 const int* par,const int* order,const double* lower,const double* upper,const int* seg,
 const double* vc,const double* gc,const double* diag,const double* D1,
 const int* loc,const int* target,double* lv,double* lc,double* lg,double* rhs,double* out){
 int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=P)return;
 if(begin>=target[p])begin=target[p]-1;
 for(int t=begin;t>=end;t--){
   if(t==target[p]-1)lv[(size_t)soma*P+p]=1.;
   const double* a=vc+(size_t)t*4*n;const double* b=gc+(size_t)t*5*K;
   for(int i=0;i<nn;i++){size_t z=(size_t)i*P+p;rhs[z]=lv[z];}
   for(int k=0;k<K;k++){
     size_t z=(size_t)k*P+p,j=(size_t)seg[k]*P+p;
     rhs[j]+=b[3*K+k]*lg[z];lc[j]+=b[4*K+k]*lg[z];
   }
   // Transposed Hines solve: same pivots, swapped off-diagonal entries.
   for(int k=nn-1;k>0;k--){int j=order[k];rhs[(size_t)par[j]*P+p]-=lower[j]/diag[(size_t)t*nn+j]*rhs[(size_t)j*P+p];}
   int root=order[0];rhs[(size_t)root*P+p]/=diag[(size_t)t*nn+root];
   for(int k=1;k<nn;k++){int j=order[k];rhs[(size_t)j*P+p]=(rhs[(size_t)j*P+p]-upper[j]*rhs[(size_t)par[j]*P+p])/diag[(size_t)t*nn+j];}
   out[(size_t)t*P+p]=-D1[loc[p]]*rhs[(size_t)loc[p]*P+p];
   for(int k=0;k<K;k++){
     size_t z=(size_t)k*P+p,j=(size_t)seg[k]*P+p;
     lg[z]=b[2*K+k]*lg[z]+b[k]*rhs[j]+b[K+k]*lc[j];
   }
   for(int i=0;i<nn;i++){
     size_t z=(size_t)i*P+p;
     if(i<n){double q=rhs[z],ca=lc[z];lv[z]=a[i]*q+a[3*n+i]*ca;lc[z]=a[n+i]*q+a[2*n+i]*ca;}
     else lv[z]=0.;
   }
 }
}
#define CHECK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"adjoint CUDA: %s\n",cudaGetErrorString(e));for(void* q:alloc)cudaFree(q);return (int)e;}}while(0)
extern "C" int adjoint_gpu(int n,int nn,int K,int T,int P,int soma,
 const int* hp,const int* ho,const double* hl,const double* hu,const int* hs,
 const double* hv,const double* hg,const double* hd,const double* hD,
 const int* hloc,const int* htarget,double* output,double* timing){
 std::vector<void*> alloc;cudaError_t status=cudaSuccess;
 auto start=std::chrono::steady_clock::now();
 auto copy=[&](const void* src,size_t bytes)->void*{
   void* dst=nullptr;status=cudaMalloc(&dst,bytes);if(status!=cudaSuccess)return nullptr;
   alloc.push_back(dst);status=cudaMemcpy(dst,src,bytes,cudaMemcpyHostToDevice);return status==cudaSuccess?dst:nullptr;
 };
 auto zero=[&](size_t bytes)->double*{
   double* dst=nullptr;status=cudaMalloc(&dst,bytes);if(status!=cudaSuccess)return nullptr;
   alloc.push_back(dst);status=cudaMemset(dst,0,bytes);return status==cudaSuccess?dst:nullptr;
 };
 int* p=(int*)copy(hp,nn*4);int* o=(int*)copy(ho,nn*4);double* l=(double*)copy(hl,nn*8);double* u=(double*)copy(hu,nn*8);int* s=(int*)copy(hs,K*4);
 double* vc=(double*)copy(hv,(size_t)T*4*n*8);double* gc=(double*)copy(hg,(size_t)T*5*K*8);double* d=(double*)copy(hd,(size_t)T*nn*8);double* D=(double*)copy(hD,n*8);
 int* loc=(int*)copy(hloc,P*4);int* target=(int*)copy(htarget,P*4);
 double* lv=zero((size_t)nn*P*8);double* lc=zero((size_t)n*P*8);double* lg=zero((size_t)K*P*8);double* rhs=zero((size_t)nn*P*8);double* out=zero((size_t)T*P*8);
 if(!p||!o||!l||!u||!s||!vc||!gc||!d||!D||!loc||!target||!lv||!lc||!lg||!rhs||!out){for(void* q:alloc)cudaFree(q);return (int)cudaErrorMemoryAllocation;}
 CHECK(cudaDeviceSynchronize());auto copied=std::chrono::steady_clock::now();
 for(int t=T-1;t>=0;t-=64)reverse_steps<<<(P+127)/128,128>>>(n,nn,K,P,soma,t,(t-63>0?t-63:0),p,o,l,u,s,vc,gc,d,D,loc,target,lv,lc,lg,rhs,out);
 CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());auto done=std::chrono::steady_clock::now();
 CHECK(cudaMemcpy(output,out,(size_t)T*P*8,cudaMemcpyDeviceToHost));
 for(void* q:alloc)cudaFree(q);
 timing[0]=std::chrono::duration<double>(copied-start).count();timing[1]=std::chrono::duration<double>(done-copied).count();timing[2]=std::chrono::duration<double>(std::chrono::steady_clock::now()-done).count();return 0;
}
