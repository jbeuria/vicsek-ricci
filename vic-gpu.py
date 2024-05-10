import argparse
import numpy as np
from time import time
import cupy as cp

# num_devices = cp.cuda.runtime.getDeviceCount()
# # List the GPU IDs
# gpu_ids = [i for i in range(num_devices)]



# # Create ArgumentParser object
parser = argparse.ArgumentParser(description='Biological aggregation with Ricci curvature')

# Add named arguments
parser.add_argument('-v', '--v', type=float, default=0.5, help='Description of argument r0')
parser.add_argument('-rmin', '--rmin', type=float, default=0.0, help='Description of argument r0')
parser.add_argument('-rmax', '--rmax', type=float, default=1.0, help='Description of argument rmax')
parser.add_argument('-eta', '--eta', type=float, default=0.5, help='Description of argument eta')
parser.add_argument('-nn', '--nn', type=int, default=1, help='Description of argument NN')
parser.add_argument('-N', '--N', type=int, default=1000, help='Description of argument N')
parser.add_argument('-steps', '--steps', type=int, default=10000,  help='Description of argument steps')
parser.add_argument('-l', '--l', type=int, default=10,  help='Description of argument steps')
parser.add_argument('-gpuid', '--gpuid', type=int, default=0,  help='GPU ID')

# Parse the command-line arguments
args = parser.parse_args()

device_id=args.gpuid
device = cp.cuda.Device(device_id)
device.use()

# v=0.5;rmin=0.0;rmax=1.0;nn=1;eta=0.15; N=1000
# n_steps=10000
# l=10

filename=f"data/vic_v{args.v}_rmin{args.rmin}_rmax{args.rmax}_eta_{args.eta}_nn{args.nn}_N{args.N}_steps{args.steps}_L{args.l}"
logFile=f"log/log_v{args.v}_rmin{args.rmin}_rmax{args.rmax}_eta_{args.eta}_nn{args.nn}_N{args.N}_steps{args.steps}_L{args.l}.txt"

# filename=f"vic_v{v}_rmin{rmin}_rmax{rmax}_eta_{eta}_nn{nn}_N{N}_steps{n_steps}_L{l}"
# logFile=f"log_v{v}_rmin{rmin}_rmax{rmax}_eta_{eta}_nn{nn}_N{N}_steps{n_steps}_L{l}.txt"


def euclidean_pbc(point1, point2,L):
    # Replace this with your own distance metric
    dx=cp.minimum(L-abs(point1 - point2), abs(point1 - point2))
    return cp.sqrt(cp.sum(dx**2))


def find_1_simplices(rips_complex):
    """
    Find 1-simplices in the Rips complex.
    """
    A=cp.where(rips_complex)
    B=A[0] < A[1]
    return cp.column_stack((A[0][B],A[1][B]))

def get_R1(pos,rmin,rmax,L,N):
    # del_r = cp.linalg.norm(points_gpu[:, None] - points_gpu, axis=-1)

    del_r=pos[:,None,:]-pos
    del_r = del_r - L*cp.rint(del_r/L ) #minimum image convention
    del_r2 = cp.sum(del_r**2,axis=-1)
    del_r=del_r2**0.5  

    neighbours = ((del_r <= rmax) & (del_r >= rmin)) # When rmin >0, this excludes the ith agent also
    
    nn_count=cp.sum(neighbours, axis=1)
    edges = cp.argwhere(neighbours)
    edge_lengths=del_r[neighbours]
    
    # v_weights = 1+ cp.bincount(edges.ravel(), minlength=N)
    v_weights= 1.0 + nn_count # get_v_weights(edges,N)    
    edge_weights =  cp.take(v_weights,edges).sum(axis=1)/ (1 + edge_lengths)

    vR1= cp.bincount(cp.ravel(edges), weights=1.0/cp.sqrt(cp.repeat(edge_weights, 2)), minlength=N)
    vR1= v_weights*vR1

    # vR1 gives sum(vw/ve) for every vertex; ve is the weight of incident edge. sum is over all such edges

    # R=cp.zeros(N)
    R_edges =cp.take(vR1,edges).sum(axis=1)
    R_edges *= cp.sqrt(edge_weights)
    R_edges = cp.take(2*v_weights,edges).sum(axis=1) - R_edges 
    R=cp.bincount(cp.ravel(edges), weights=cp.repeat(R_edges, 2), minlength=N)
    
    return (R,neighbours)


def get_next_orientation(R,neighbours,ori,L,N,eta,nn,d=1):

    nn_count=cp.sum(neighbours, axis=1)
    v_weights= 1.0 + nn_count 

    weighted_adj_matrix = neighbours * R[:, cp.newaxis]
    weighted_adj_matrix = cp.maximum(weighted_adj_matrix, weighted_adj_matrix.T)

    # Exclude self-neighbors
    # cp.fill_diagonal(weighted_adj_matrix, -cp.inf)

    # Find the k maximum weight neighbors for each point
    # R_max_neighbors = cp.argsort(weighted_adj_matrix, axis=1)[:, ::-1][:, :nn]

    # Find the k minimum weight neighbors for each point
    R_min_neighbors = cp.argsort(weighted_adj_matrix, axis=1)[:, :nn]

    weighted_ori= ori  * v_weights[:,None]

    valid_neighbors_mask = neighbours[cp.arange(N)[:, cp.newaxis], R_min_neighbors]
    nn_dir=cp.take( weighted_ori,R_min_neighbors,axis=0 )
    nn_ori=nn_dir* valid_neighbors_mask[:,:,None]  #weighted_ori[R_min_neighbors]


    ori_sum = cp.sum(nn_ori, axis=1)/nn_count[:,None]
    ori_new=ori_sum/(cp.sqrt(cp.sum(ori_sum**2,axis=1))[:,None])
    
    nan_indices = cp.isnan(ori_new)
    # Replace NaN values in A with corresponding values from B
    ori_new[nan_indices] = ori[nan_indices]

    noise=cp.random.normal(scale=eta, size=(N, len(L)))
    ori_new_nosiy= ori_new +noise
    ori_new_nosiy= ori_new_nosiy/(cp.sqrt(cp.sum(ori_new_nosiy**2,axis=1))[:,None])

    return ori_new_nosiy

def vicsek_ricci(v=0.3,rmin=0,rmax=1.0,nn=1, 
                         eta=0.3, N=150,n_steps=3000, 
                         L=[100,100],dt=0.1,d=2):
    
    # params={'v':v,'r0':r0,'rmax':rmax,'nn':nn, 
    #         'eta':eta, 'N':N,'n_steps':n_steps, 'L':L,'dt':dt}

    L = cp.asarray(L)
    pos = L*cp.random.rand(N,len(L))
    ori = cp.random.rand(N, len(L))-0.5
    ori= ori/(cp.sqrt(cp.sum(ori**2,axis=1))[:,cp.newaxis]) # normalize the orientation
    pos_history=[]
    ori_history=[]
    R_history=[]
    

    for i in range(n_steps+1):
        # if(i%100==0): print(f"step: {i}")

        R,neighbours = get_R1(pos,rmin,rmax,L,N)

        # to run on GPU with cupy
        pos_history.append(pos.get())
        ori_history.append(ori.get())
        R_history.append(R.get())

        # to run on CPU
        # pos_history.append(pos)
        # ori_history.append(ori)
        # R_history.append(R)

        pos =pos+ dt*(v*ori)
        pos =pos -L*cp.floor(pos /L)
        ori = get_next_orientation(R,neighbours,ori,L,N,eta,nn,d=1)

        if(i%100==0): 
            open(logFile, "w").write(f"step: {i}\n")
        
        if((i+1)%5000==0):
            counter=int(i/5000)

            f=f"{filename}_part{counter}"
            np.save(f,{'X':pos_history ,'V':ori_history,'R':R_history})
            
            pos_history=[]
            ori_history=[]
            R_history=[]
            
    return (pos_history,ori_history,R_history)


t1=time()

# X,V,R=vicsek_ricci(v=v,rmin=rmin,rmax=rmax,nn=nn,eta=eta, N=N, n_steps=n_steps, L=[l,l,l])
X,V,R=vicsek_ricci(v=args.v,rmin=args.rmin,rmax=args.rmax,nn=args.nn,eta=args.eta, 
                   N=args.N, n_steps=args.steps, L=[args.l,args.l,args.l])

print('time: ',time()-t1)

# np.save(filename,{'X':X,'V':V,'R':R})



