"""Stable lexicographic multiset ordinals, with resumable finite chunks."""
import itertools, math
import numpy as np
N=20
def total_below(length):return math.comb(N+length-1,N)-1 if length>0 else 0
def candidates(length,start=0):
    assert 0<=start<=total_below(length)
    base=0
    for k in range(1,length):
        size=math.comb(N+k-1,k)
        if start<base+size:
            skip=max(0,start-base)
            for rank,ids in enumerate(itertools.islice(itertools.combinations_with_replacement(range(N),k),skip,None),skip):
                yield base+rank,k,ids
        base+=size
def next_chunk(stream,batch):
    items=list(itertools.islice(stream,batch))
    if not items:return None
    ranks=np.array([r for r,_,_ in items],dtype=np.int64)
    lengths=np.array([k for _,k,_ in items],dtype=np.uint8)
    counts=np.zeros((batch,N),dtype=np.int16)
    for i,(_,k,ids) in enumerate(items):counts[i]=np.bincount(ids,minlength=N)
    assert np.array_equal(ranks,np.arange(ranks[0],ranks[0]+len(items)))
    assert np.array_equal(counts[:len(items)].sum(axis=1),lengths)
    return ranks,lengths,counts
