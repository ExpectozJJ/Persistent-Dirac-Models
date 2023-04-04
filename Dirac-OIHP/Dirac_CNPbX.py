import numpy as np
import math
import scipy.io as sio
import gudhi as gd
from scipy.sparse import *
import time
import sys
import multiprocessing as mp 
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches

# gen_graph generates a graph network (0-simplex and 1-simplex) of the simplicial complex.
# n_faces outputs all the simplices for the simplicial complex for the given filtration parameter.

def faces(simplices):
    faceset = set()
    for simplex in simplices:
        numnodes = len(simplex)
        for r in range(numnodes, 0, -1):
            for face in combinations(simplex, r):
                faceset.add(tuple(sorted(face)))
    return faceset

def n_faces(face_set, n):
    return filter(lambda face: len(face)==n+1, face_set)

def boundary_operator(face_set, i):
    source_simplices = list(n_faces(face_set, i))
    target_simplices = list(n_faces(face_set, i-1))
    #print(source_simplices, target_simplices)

    if len(target_simplices)==0:
        S = dok_matrix((1, len(source_simplices)), dtype=np.float64)
        S[0, 0:len(source_simplices)] = 1
    else:
        source_simplices_dict = {source_simplices[j]: j for j in range(len(source_simplices))}
        target_simplices_dict = {target_simplices[i]: i for i in range(len(target_simplices))}

        S = dok_matrix((len(target_simplices), len(source_simplices)), dtype=np.float64)
        for source_simplex in source_simplices:
            for a in range(len(source_simplex)):
                target_simplex = source_simplex[:a]+source_simplex[(a+1):]
                i = target_simplices_dict[target_simplex]
                j = source_simplices_dict[source_simplex]
                S[i, j] = -1 if a % 2==1 else 1
    
    return S

start = time.time()
ndata = 1
maxfiltration = 10
step=0.25
fact=int(1.0/step)
fmax= 7.25
fid = 25;

a, b, c, start, end = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])

def filtration(a,b,c,idx):
    file = open('./MAPb{}3_{}_{}_data/{}_atmlist_L5_f{}.txt'.format(a,b,c,c,idx))

    contents = file.readlines()
    for i in range(len(contents)):
        contents[i] = contents[i].rstrip("\n").split(",")
        contents[i] = [float(s) for s in contents[i]]

    b_mats = []
    #d0_mats = []
    #d1_mats = []
    #d_2 = []
    #ii = 0
    #d_0 = []
    #d_1 = []
    d0_eig, d1_eig = [], []
    alpha = gd.AlphaComplex(contents)
    st = alpha.create_simplex_tree()
    val = list(st.get_filtration())
    for f in np.arange(1, 6.75, 0.25):
        #print(ii, f)
        simplices = set()
        for v in val:
            if np.sqrt(v[1])*2 <= f:
                simplices.add(tuple(v[0]))
        #print(f, len(simplices))
        b_mats.append([boundary_operator(simplices, 1).toarray(), boundary_operator(simplices, 2).toarray()]) #, boundary_operator(simplices, 3).toarray()])
        size0, size1 = np.shape(b_mats[-1][0]), np.shape(b_mats[-1][1]) #, np.shape(b_mats[-1][2])
        dmat0 = np.array([])
        dmat1 = np.array([])
        dmat2 = np.array([])
        if size0[1] > 0:
            dmat0 = np.bmat([[np.zeros((size0[0],size0[0])), b_mats[-1][0]], [np.transpose(b_mats[-1][0]), np.zeros((size0[1], size0[1]))]])
        else:
            dmat0 = np.zeros((size0[0],size0[0]))

        if size1[1] > 0: 
            dmat1 = np.bmat([[np.zeros((size0[0],size0[0])), b_mats[-1][0], np.zeros((size0[0], size1[1]))], [np.transpose(b_mats[-1][0]), np.zeros((size1[0], size1[0])), b_mats[-1][1]], [np.zeros((size1[1], size0[0])), np.transpose(b_mats[-1][1]), np.zeros((size1[1], size1[1]))]])
        elif size0[1] > 0:
            dmat1 = np.bmat([[np.zeros((size0[0],size0[0])), b_mats[-1][0]], [np.transpose(b_mats[-1][0]), np.zeros((size1[0], size1[0]))]])
        else:
            dmat1 = np.zeros((size0[0],size0[0]))

        """
        if size2[1] > 0:
            dmat2 = np.bmat([[np.zeros((size0[0],size0[0])), b_mats[-1][0], np.zeros((size0[0], size1[1])), np.zeros((size0[0], size2[1]))], [np.transpose(b_mats[-1][0]), np.zeros((size1[0], size1[0])), b_mats[-1][1], np.zeros((size1[0], size2[1]))], [np.zeros((size1[1], size0[0])), np.transpose(b_mats[-1][1]), np.zeros((size1[1], size1[1])), b_mats[-1][2]], [np.zeros((size2[1], size0[0])), np.zeros((size2[1], size1[0])), np.transpose(b_mats[-1][2]), np.zeros((size2[1],size2[1]))]])
        elif size1[1] > 0:
            dmat2 = np.bmat([[np.zeros((size0[0],size0[0])), b_mats[-1][0], np.zeros((size0[0], size1[1]))], [np.transpose(b_mats[-1][0]), np.zeros((size1[0], size1[0])), b_mats[-1][1]], [np.zeros((size1[1], size0[0])), np.transpose(b_mats[-1][1]), np.zeros((size1[1], size1[1]))]])
        elif size0[1] > 0:
            dmat2 = np.bmat([[np.zeros((size0[0],size0[0])), b_mats[-1][0]], [np.transpose(b_mats[-1][0]), np.zeros((size1[0], size1[0]))]])
        else:
            dmat2 = np.zeros((size0[0],size0[0]))
        """

        d0_eig.append(np.linalg.eigvalsh(dmat0))
        d1_eig.append(np.linalg.eigvalsh(dmat1))
        #d2_eig.append(np.linalg.eigvalsh(dmat2))
        print(f, size0, size1)
        """
        if size1[1] > 0: 
            dmat1 = np.bmat([[np.zeros((size0[0],size0[0])), b_mats[-1][0], np.zeros((size0[0], size1[1]))], [np.transpose(b_mats[-1][0]), np.zeros((size1[0], size1[0])), b_mats[-1][1]], [np.zeros((size1[1], size0[0])), np.transpose(b_mats[-1][1]), np.zeros((size1[1], size1[1]))]])
        elif size0[1] > 0:
            dmat1 = np.bmat([[np.zeros((size0[0],size0[0])), b_mats[-1][0]], [np.transpose(b_mats[-1][0]), np.zeros((size1[0], size1[0]))]])
        else:
            dmat1 = np.zeros((size0[0],size0[0]))
        #if size2[1] > 0:
            #dmat2 = np.bmat([[np.zeros((size0[0],size0[0])), b_mats[-1][0], np.zeros((size0[0], size1[1])), np.zeros((size0[0], size2[1]))], [np.transpose(b_mats[-1][0]), np.zeros((size1[0], size1[0])), b_mats[-1][1], np.zeros((size1[0], size2[1]))], [np.zeros((size1[1], size0[0])), np.transpose(b_mats[-1][1]), np.zeros((size1[1], size1[1])), b_mats[-1][2]], [np.zeros((size2[1], size0[0])), np.zeros((size2[1], size1[0])), np.transpose(b_mats[-1][2]), np.zeros((size2[1],size2[1]))]])
        
        #d_0.append(dmat0)
        #d_1.append(dmat1)
        #d_2.append(dmat2)
    
        print(f, size0, size1, size2)
        #print(np.shape(dmat0))
        if np.shape(dmat0)[0] >= 1:
            d0_eig.append(np.linalg.eigvalsh(dmat0))
            #print(d0_eig)
        else:
            d0_eig.append([])
        #print(np.shape(dmat1))
        if np.shape(dmat1)[0] >= 1:
            d1_eig.append(np.linalg.eigvalsh(dmat1))
            #print(d1_eig)
        else:
            d1_eig.append([])
        #print(d0_eig)
        #print(d1_eig)
        """
        """
        #print('\n', f)
        #if d_0[-1].size > 0:
            #print(np.linalg.eigvalsh(d_0[-1]))#, np.linalg.eigvalsh(d_0[-1]*d_0[-1]))
        #if d_1[-1].size > 0:
            #print(np.linalg.eigvalsh(d_1[-1]), np.linalg.eigvalsh(d_1[-1]*d_1[-1]))
        if d_2[-1].size > 0:
            #print(np.linalg.eigvalsh(d_2[-1]), np.linalg.eigvalsh(d_2[-1]*d_2[-1]))
        
            plt.figure(figsize=(5, 5), dpi=100)
            ax = plt.gca()
            sns.set_style("whitegrid")
            sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})
            im = sns.heatmap(d_2[-1], cmap='bwr_r', vmin=-1, vmax = 1, linewidth=.2, cbar = False)

            idx = [0, size0[0], size1[0], size2[0], size2[1]]
            for i in range(len(idx)-1):
                for j in range(len(idx)-1):
                    rect = patches.Rectangle((np.sum(idx[:i+1]), np.sum(idx[:j+1])), idx[i+1], idx[j+1], linewidth=2, edgecolor='k', facecolor='none')
                    ax.add_patch(rect)

            plt.xticks([])
            plt.yticks([])
            plt.xlim([-0.2, len(d_2[-1])+0.1])
            plt.ylim([len(d_2[-1])+0.1, -0.2])
            #plt.axis('equal')
            #plt.savefig('tetrahedron_d2_{}.png'.format(f), dpi=200)
            plt.show()
            
            plt.figure(figsize=(5, 5), dpi=100)
            ax = plt.gca()
            sns.set_style("whitegrid")
            sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})
            im = sns.heatmap(d_2[-1]*d_2[-1], cmap='bwr_r', vmin=-1, vmax = 1, linewidth=.2, cbar = False)

            idx = [0, size0[0], size1[0], size2[0], size2[1]]
            for i in range(len(idx)-1):
                for j in range(len(idx)-1):
                    rect = patches.Rectangle((np.sum(idx[:i+1]), np.sum(idx[:j+1])), idx[i+1], idx[j+1], linewidth=2, edgecolor='k', facecolor='none')
                    ax.add_patch(rect)

            plt.xticks([])
            plt.yticks([])
            plt.xlim([-0.2, len(d_2[-1])+0.1])
            plt.ylim([len(d_2[-1])+0.1, -0.2])
            #plt.axis('equal')
            #plt.savefig('tetrahedron_d2_sqd_{}.png'.format(f), dpi=200)
            plt.show()
        """
        """
        if dmat1.size > 0:
            plt.figure(figsize=(5, 5), dpi=100)
            ax = plt.gca()
            sns.set_style("whitegrid")
            sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})
            im = sns.heatmap(dmat1, cmap='bwr_r', vmin=-1, vmax = 1, linewidth=.2, cbar = False)

            idx = [0, size0[0], size1[0], size2[0], size2[1]]
            for i in range(len(idx)-1):
                for j in range(len(idx)-1):
                    rect = patches.Rectangle((np.sum(idx[:i+1]), np.sum(idx[:j+1])), idx[i+1], idx[j+1], linewidth=2, edgecolor='k', facecolor='none')
                    ax.add_patch(rect)

            plt.xticks([])
            plt.yticks([])
            plt.xlim([-0.2, len(dmat1)+0.1])
            plt.ylim([len(dmat1)+0.1, -0.2])
            #plt.axis('equal')
            plt.savefig('MAPb{}3_{}_d1_{}.png'.format(a,b,f), dpi=200)
            #plt.show()
        """
        """
            plt.figure(figsize=(5, 5), dpi=100)
            ax = plt.gca()
            sns.set_style("whitegrid")
            sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})
            im = sns.heatmap(d_1[-1]*d_1[-1], cmap='bwr_r', vmin=-1, vmax = 1, linewidth=.2, cbar = False)

            idx = [0, size0[0], size1[0], size2[0], size2[1]]
            for i in range(len(idx)-1):
                for j in range(len(idx)-1):
                    rect = patches.Rectangle((np.sum(idx[:i+1]), np.sum(idx[:j+1])), idx[i+1], idx[j+1], linewidth=2, edgecolor='k', facecolor='none')
                    ax.add_patch(rect)

            plt.xticks([])
            plt.yticks([])
            plt.xlim([-0.2, len(d_1[-1])+0.1])
            plt.ylim([len(d_1[-1])+0.1, -0.2])
            #plt.axis('equal')
            plt.savefig('tetrahedron_d1_sqd_{}.png'.format(f), dpi=200)
            #plt.show()
        """
        """
        if dmat0.size > 0:
            plt.figure(figsize=(5, 5), dpi=100)
            ax = plt.gca()
            sns.set_style("whitegrid")
            sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})
            im = sns.heatmap(dmat0, cmap='bwr_r', vmin=-1, vmax = 1, linewidth=.2, cbar = False)

            idx = [0, size0[0], size0[1], size1[1], size2[1]]
            for i in range(len(idx)-1):
                for j in range(len(idx)-1):
                    rect = patches.Rectangle((np.sum(idx[:i+1]), np.sum(idx[:j+1])), idx[i+1], idx[j+1], linewidth=2, edgecolor='k', facecolor='none')
                    ax.add_patch(rect)

            plt.xticks([])
            plt.yticks([])
            plt.xlim([-0.2, len(dmat0)+0.1])
            plt.ylim([len(dmat0)+0.1, -0.2])
            #plt.axis('equal')
            plt.savefig('MAPb{}3_{}_d0_{}.png'.format(a,b,f), dpi=200)
            #plt.show()
        
            
            plt.figure(figsize=(5, 5), dpi=100)
            ax = plt.gca()
            sns.set_style("whitegrid")
            sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})
            im = sns.heatmap(d_0[-1]*d_0[-1], cmap='bwr_r', vmin=-1, vmax = 1, linewidth=.2, cbar = False)

            idx = [0, size0[0], size0[1], size1[1], size2[1]]
            for i in range(len(idx)-1):
                for j in range(len(idx)-1):
                    rect = patches.Rectangle((np.sum(idx[:i+1]), np.sum(idx[:j+1])), idx[i+1], idx[j+1], linewidth=2, edgecolor='k', facecolor='none')
                    ax.add_patch(rect)

            plt.xticks([])
            plt.yticks([])
            plt.xlim([-0.2, len(d_0[-1])+0.1])
            plt.ylim([len(d_0[-1])+0.1, -0.2])
            #plt.axis('equal')
            plt.savefig('tetrahedron_d0_sqd_{}.png'.format(f), dpi=200)
            #plt.show()
        """

    #d0_mats.append(d_0)
    #d1_mats.append(d_1)
    #ii += 1
    #np.save("high_voltage_poly_optimized/data/opt-xyz/{}_d0.npy".format(idx), d0_mats)
    #np.save("high_voltage_poly_optimized/data/opt-xyz/{}_d1.npy".format(idx), d1_mats)
    np.save('./MAPb{}3_{}_{}_data/{}_atmlist_L5_f{}_d0.npy'.format(a,b,c,c,idx), d0_eig)
    np.save('./MAPb{}3_{}_{}_data/{}_atmlist_L5_f{}_d1.npy'.format(a,b,c,c,idx), d1_eig)
    #np.save('MAPb{}3_{}_CNXPb_atmlist_L5_f{}_d2.npy'.format(a,b, idx), d2_eig)


"""
no_threads = mp.cpu_count()
p = mp.Pool(processes = no_threads)
results = p.map(func, list(range(begin, last)))
#results = p.map(func, arr)
p.close()
p.join()
"""

#arr = np.concatenate((range(603, 608), range(625, 633), range(666, 677), range(728, 731), range(780, 794))) # Cl3 Cubic CNXPb
for idx in range(start, end):
    filtration(a,b,c,idx)

        
        
        
        
        
        