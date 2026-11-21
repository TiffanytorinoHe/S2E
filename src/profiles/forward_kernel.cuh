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
