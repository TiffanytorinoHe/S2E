#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <vector>
// Each thread owns a candidate. Tree traversal and all states remain on device.
__global__ void propagate(int n,int nn,int K,int T,int P,int W,int soma,int rec,int begin,int end,
 const int* par,const int* order,const double* lower,const int* seg,
 const double* vc,const double* gc,const double* diag,const double* factor,
 const double* vn,const double* wave,const double* D1,const int* loc,const int* wi,
 double* v,double* ca,double* gate,double* rhs,double* minimum,double* trace,const int* first){
 int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=P)return;
 if(begin<first[wi[p]])begin=first[wi[p]];
 for(int t=begin;t<end;t++){
   const double* a=vc+(size_t)t*4*n;const double* b=gc+(size_t)t*5*K;
   for(int i=0;i<nn;i++){
     size_t z=(size_t)i*P+p;
     if(i<n){double v0=v[z],c0=ca[z];rhs[z]=a[i]*v0+a[n+i]*c0;ca[z]=a[2*n+i]*c0+a[3*n+i]*v0;}
     else rhs[z]=0.;
   }
   for(int k=0;k<K;k++){
     size_t z=(size_t)k*P+p,j=(size_t)seg[k]*P+p;double g=gate[z];
     rhs[j]+=b[k]*g;ca[j]+=b[K+k]*g;
   }
   int l=loc[p];rhs[(size_t)l*P+p]+=D1[l]*wave[(size_t)t*W+wi[p]]*(-70.-vn[(size_t)t*n+l]);
   for(int k=nn-1;k>0;k--){int j=order[k];rhs[(size_t)par[j]*P+p]-=factor[(size_t)t*nn+j]*rhs[(size_t)j*P+p];}
   int r=order[0];v[(size_t)r*P+p]=rhs[(size_t)r*P+p]/diag[(size_t)t*nn+r];
   for(int k=1;k<nn;k++){int j=order[k];v[(size_t)j*P+p]=(rhs[(size_t)j*P+p]-lower[j]*v[(size_t)par[j]*P+p])/diag[(size_t)t*nn+j];}
   for(int k=0;k<K;k++){size_t z=(size_t)k*P+p,j=(size_t)seg[k]*P+p;gate[z]=b[2*K+k]*gate[z]+b[3*K+k]*v[j]+b[4*K+k]*ca[j];}
   double val=v[(size_t)soma*P+p];minimum[p]=fmin(minimum[p],val);
   if(rec)trace[(size_t)(t+1)*P+p]=val;
 }
}
#define CHECK(x) do {cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));for(void* q:alloc)cudaFree(q);return (int)e;}}while(0)
extern "C" int tangent_gpu(int n,int nn,int K,int T,int P,int W,int soma,int rec,
 const int* hp,const int* ho,const double* hl,const int* hs,
 const double* hv,const double* hg,const double* hd,const double* hf,
 const double* hn,const double* hw,const double* hD,const int* hloc,const int* hwi,
 double* out,double* tr,double* timing){
 std::vector<void*> alloc;
 auto start=std::chrono::steady_clock::now();
 cudaDeviceProp prop;CHECK(cudaGetDeviceProperties(&prop,0));printf("GPU: %s, capability %d.%d\n",prop.name,prop.major,prop.minor);fflush(stdout);
 auto copy=[&](const void* src,size_t bytes)->void* {void* dst=nullptr;cudaError_t e=cudaMalloc(&dst,bytes);if(e!=cudaSuccess)return nullptr;alloc.push_back(dst);e=cudaMemcpy(dst,src,bytes,cudaMemcpyHostToDevice);if(e!=cudaSuccess)return nullptr;return dst;};
 auto zeros=[&](size_t bytes)->double* {double* dst=nullptr;cudaError_t e=cudaMalloc(&dst,bytes);if(e!=cudaSuccess)return nullptr;alloc.push_back(dst);cudaMemset(dst,0,bytes);return dst;};
 int* p=(int*)copy(hp,nn*sizeof(int));int* o=(int*)copy(ho,nn*sizeof(int));double* l=(double*)copy(hl,nn*sizeof(double));int* s=(int*)copy(hs,K*sizeof(int));
 double* vc=(double*)copy(hv,(size_t)T*4*n*8);double* gc=(double*)copy(hg,(size_t)T*5*K*8);double* d=(double*)copy(hd,(size_t)T*nn*8);double* f=(double*)copy(hf,(size_t)T*nn*8);
 double* vn=(double*)copy(hn,(size_t)T*n*8);double* w=(double*)copy(hw,(size_t)T*W*8);double* D=(double*)copy(hD,n*8);
 int* loc=(int*)copy(hloc,P*sizeof(int));int* wi=(int*)copy(hwi,P*sizeof(int));
 double* v=zeros((size_t)nn*P*8);double* ca=zeros((size_t)n*P*8);double* gate=zeros((size_t)K*P*8);double* rhs=zeros((size_t)nn*P*8);double* mi=zeros((size_t)P*8);double* trace=zeros(rec?(size_t)(T+1)*P*8:8);
 std::vector<int> hfirst(W,T);
 for(int j=0;j<W;j++)for(int t=0;t<T;t++)if(hw[(size_t)t*W+j]!=0.){hfirst[j]=t;break;}
 int* first=(int*)copy(hfirst.data(),W*sizeof(int));
 if(!p||!o||!l||!s||!vc||!gc||!d||!f||!vn||!w||!D||!loc||!wi||!v||!ca||!gate||!rhs||!mi||!trace||!first){for(void* q:alloc)cudaFree(q);return (int)cudaErrorMemoryAllocation;}
 CHECK(cudaDeviceSynchronize());auto copied=std::chrono::steady_clock::now();
 for(int t=0;t<T;t+=64)propagate<<<(P+127)/128,128>>>(n,nn,K,T,P,W,soma,rec,t,(t+64<T?t+64:T),p,o,l,s,vc,gc,d,f,vn,w,D,loc,wi,v,ca,gate,rhs,mi,trace,first);
 CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());auto done=std::chrono::steady_clock::now();
 CHECK(cudaMemcpy(out,mi,P*8,cudaMemcpyDeviceToHost));if(rec)CHECK(cudaMemcpy(tr,trace,(size_t)(T+1)*P*8,cudaMemcpyDeviceToHost));
 for(void* q:alloc)cudaFree(q);
 timing[0]=std::chrono::duration<double>(copied-start).count();timing[1]=std::chrono::duration<double>(done-copied).count();timing[2]=std::chrono::duration<double>(std::chrono::steady_clock::now()-done).count();return 0;
}
