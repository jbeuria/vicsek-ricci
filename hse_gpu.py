import numpy as np
import pandas as pd
import gzip
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy
from numpy import linalg as LA
# from npy_append_array import NpyAppendArray
from gudhi.point_cloud.timedelay import TimeDelayEmbedding
import gudhi
from hodgelaplacians import HodgeLaplacians
from scipy.signal import resample
# from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import expm

from cupy.scipy.linalg import expm as expm_gpu
from cupy.linalg import eigvalsh
import cupy as cp



def get_hoge_ent(hl,beta=1,order=0):

    maxdim=hl.maxdim
    if(order > maxdim): return np.nan

    L = hl.getHodgeLaplacian(order).tocsc()
    z = expm(-beta*L).trace()   
    eig_val_L=eigvalsh(cp.asarray(L.toarray())).get() #,k=10,which="LM")
    vn_ent=beta * np.sum(eig_val_L*np.exp(-eig_val_L*beta))/z + np.log(np.abs(z))
    
    return vn_ent

def write_log(str,filename="vn_log.txt"):
    file = open(filename, "a")  # Open the file in append mode
    file.write(str)      
    file.close()  # Close the file explicitly


def get_hse(pcd,max_dimension=2,beta=1.0,maxR=1.0,L=[20,20]):

    del_r=pcd[:,None,:]-pcd
    del_r = del_r - L*np.rint(del_r/L ) #minimum image convention
    del_r2 = np.sum(del_r**2,axis=-1)
    del_r=del_r2**0.5  

    rips_complex = gudhi.RipsComplex(distance_matrix=del_r, max_edge_length=maxR)
    st=rips_complex.create_simplex_tree(max_dimension)
    complex=st.get_skeleton(max_dimension)
    simplices=[tuple(a[0]) for a in complex]
    n_simplices=len(simplices)

    # num_simplices = {n:0 for n in range(max_dimension+1) }

    # for dim in range(max_dimension+1):  
    #     for simplex in st.get_skeleton(dim):
    #         if len(simplex[0]) == dim + 1:  # Check if it's a simplex of the current dimension
    #             num_simplices[dim] += 1
    
    write_log(f'No of sc faces: {n_simplices}, maxR={maxR} \n')

    try:
        hl = HodgeLaplacians(simplices, max_dimension)
        # print('test')
        hse=[get_hoge_ent(hl,beta=beta,order=k) for k in range(max_dimension+1)]
        # print(hse)
        return  hse

    except Exception as e:
        # Print the error message
        print("An error occurred:", e)
        hse=[np.nan for n in range(max_dimension+1) ]
        return hse
         

import glob

data=np.load('pos-2d-100000.npy',allow_pickle=True).item()

keys =["rmin0.0_rmax1.0_eta_0.2_nn1","rmin0.0_rmax1.0_eta_0.2_nn7"]
keys+=["rmin1.0_rmax2.0_eta_0.2_nn1","rmin1.0_rmax2.0_eta_0.2_nn7"]

hse={}
for key in data.keys:
    write_log(f"{key}\n")
    pcds=data[key]
    hse[key]=[get_hse(pcd,max_dimension=1,beta=0.1, maxR=1.0,L=np.array([10,10]) ) for pcd in pcds]
    
np.save('hse-2d.npy', hse)

