"""Initialize and verify the assigned CUDA device before any long CPU work."""
import ctypes
from pathlib import Path
import numpy as np
def check():
    lib=ctypes.CDLL(str(Path(__file__).resolve().parent/'libtangent.so'))
    fn=lib.tangent_gpu;fn.restype=ctypes.c_int
    arrays=[np.array([-1],np.int32),np.array([0],np.int32),np.zeros(1),np.array([0],np.int32),np.zeros(4),np.zeros(5),np.ones(1),np.zeros(1),np.zeros(1),np.zeros(1),np.ones(1),np.array([0],np.int32),np.array([0],np.int32),np.zeros(1),np.zeros(1),np.zeros(3)]
    ret=fn(*[ctypes.c_int(i) for i in [1,1,1,1,1,1,0,0]],*[ctypes.c_void_p(a.ctypes.data) for a in arrays])
    if ret:raise RuntimeError(f'GPU startup check failed, CUDA status {ret}')
