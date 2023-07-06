import numpy as np
import math
import os
import scipy.io as sio
import gudhi as gd
from scipy.sparse import *
import time
import sys
import multiprocessing as mp 
import matplotlib.pyplot as plt
#import seaborn as sns
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

def n_cofaces(face_set, simplex, n):
    return filter(lambda coface: set(simplex) < set(coface) and len(coface)==n+1, face_set)

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

def G_matrix(face_set, ptcloud, ww, d):
    """This function returns the inverse matrices G in Generalised Dirac Operator 
    
    Inputs: face_set is a set of simplices, d is the maximal dimension of the Generalised Dirac Operator.
    Output: Returns a list of sparse dok inverse matrices G. 
    """
    
    
    G = []
    
    source_simplices = list(n_faces(face_set, d))
    source_simplices_dict = {source_simplices[j]: j for j in range(len(source_simplices))}  
    S = dok_matrix((len(source_simplices), len(source_simplices)), dtype=np.float64)
    
    for source_simplex in source_simplices:
        j = source_simplices_dict[source_simplex]
        
        # change your weights here
        tmp = source_simplices[j]
        S[j,j] = ww[tmp]
    
    #print(S)
    G.append(S)
    
    for n in range(d-1, -1, -1):
        source_simplices = list(n_faces(face_set, n+1))
        source_simplices_dict = {source_simplices[j]: j for j in range(len(source_simplices))} 
        
        target_simplices = list(n_faces(face_set, n))
        target_simplices_dict = {target_simplices[j]: j for j in range(len(target_simplices))}  
        S = dok_matrix((len(target_simplices), len(target_simplices)), dtype=np.float64)
        for target_simplex in target_simplices:
            #print(source_simplex, list(n_cofaces(face_set, source_simplex, len(source_simplex))))
            cof = list(n_cofaces(face_set, target_simplex, len(target_simplex)))
            j = target_simplices_dict[target_simplex]
            tmp = []
            for ss in cof:
                k = source_simplices_dict[ss]; tmp.append(G[-1][k,k])
            if n == 1:
                tmps = target_simplex
                S[j,j] = ww[tmps] + np.sum(tmp) #np.linalg.norm(np.array(ptcloud[tmps[0]])-np.array(ptcloud[tmps[1]])) + np.sum(tmp)
            elif n == 0:
                S[j,j] = 1 + np.sum(tmp)
        #print(S)
        G.append(S)
    
    return G

def weighted_dirac(idx):

    if not os.path.exists('./FreeSolv-wd-npys'):
        os.mkdir('./FreeSolv-wd-npys')

    contents  = np.load('./FreeSolv-npys/FreeSolv_mol_{}.npy'.format(idx), allow_pickle=True)
    
    contents, qcharge, typ = np.array(contents[:, :3], dtype=float), np.array(contents[:, 3], dtype=float), contents[:, 4]
    #print(np.unique(typ))

    b_mats = []

    d0_eig_all, d1_eig_all = [], []
    d0_eig_noH, d1_eig_noH = [], []
    d0_eig_noCH, d1_eig_noCH = [], []
    
    alpha = gd.AlphaComplex(points=contents)
    st = alpha.create_simplex_tree()
    val = list(st.get_filtration())
    #print(val[-1])
    #print(val[-1][1])
    for f in np.arange(0.1, 12.1, 0.1):
        #print(ii, f)
        simplices = set()
        for v in val:
            if np.sqrt(v[1])*2 <= f and len(v[0]) <= 3:
                simplices.add(tuple(v[0]))

        ww = dict()
        for ss in simplices:
            if len(ss) == 1:
                ww[ss] = np.abs(qcharge[ss[0]])+0.1
            elif len(ss) == 2:
                dd = np.linalg.norm(contents[ss[0]]-contents[ss[1]])
                ww[ss] = dd #np.abs(qcharge[ss[0]])*np.abs(qcharge[ss[1]])/dd 
            elif len(ss) == 3:
                d1 = np.linalg.norm(contents[ss[0]]-contents[ss[1]])
                d2 = np.linalg.norm(contents[ss[1]]-contents[ss[2]])
                d3 = np.linalg.norm(contents[ss[0]]-contents[ss[2]])
                semiperi = 0.5*(d1+d2+d3)
                if semiperi*(semiperi-d1)*(semiperi-d2)*(semiperi-d3) < 1e-6:
                    area = 0.000001
                else:
                    area = np.sqrt(semiperi*(semiperi-d1)*(semiperi-d2)*(semiperi-d3))
                ww[ss] = area #np.abs(qcharge[ss[0]])*np.abs(qcharge[ss[1]])*np.abs(qcharge[ss[2]])/area 
        
        #print(ww)
        G = G_matrix(simplices, contents, ww, 2)[::-1]

        b_mats = []
        b_mats.append([boundary_operator(simplices, 1).toarray(), boundary_operator(simplices, 2).toarray()])
        size0, size1 = np.shape(b_mats[-1][0]), np.shape(b_mats[-1][1])
        wdmat0 = np.array([])
        wdmat1 = np.array([])
        if size0[1] > 0:
            wdmat0 = np.bmat([[np.zeros((size0[0],size0[0])), np.matmul(np.linalg.inv(G[0].todense()), np.matmul(b_mats[-1][0], G[1].todense()))/math.sqrt(2)], [np.transpose(b_mats[-1][0])/math.sqrt(2), np.zeros((size1[0], size1[0]))]])
        else:
            wdmat0 = np.zeros((size0[0],size0[0]))
            
        if size1[1] > 0: 
            wdmat1 = np.bmat([[np.zeros((size0[0],size0[0])), np.matmul(np.linalg.inv(G[0].todense()), np.matmul(b_mats[-1][0], G[1].todense()))/math.sqrt(2), np.zeros((size0[0], size1[1]))], [np.transpose(b_mats[-1][0])/math.sqrt(2), np.zeros((size1[0], size1[0])), np.matmul(np.linalg.inv(G[1].todense()), np.matmul(b_mats[-1][1], G[2].todense()))/math.sqrt(3)], [np.zeros((size1[1], size0[0])), np.transpose(b_mats[-1][1])/math.sqrt(3), np.zeros((size1[1], size1[1]))]])
        elif size0[1] > 0:
            wdmat1 = np.bmat([[np.zeros((size0[0],size0[0])), np.matmul(np.linalg.inv(G[0].todense()), np.matmul(b_mats[-1][0], G[1].todense()))/math.sqrt(2)], [np.transpose(b_mats[-1][0])/math.sqrt(2), np.zeros((size1[0], size1[0]))]])
        else:
            wdmat1 = np.zeros((size0[0],size0[0]))
        
        d0_eig_all.append(np.linalg.eigvals(wdmat0))
        d1_eig_all.append(np.linalg.eigvals(wdmat1))

        #print(f, np.shape(d0_eig_all), np.shape(d1_eig_all))
        #print(size0, size1)

    pts = []
    for i in range(len(contents)):
        if typ[i] != "H":
            pts.append(contents[i])
    
    alpha = gd.RipsComplex(points=pts)
    st = alpha.create_simplex_tree(max_dimension=2)
    val = list(st.get_filtration())
    #print(val[-1])

    for f in np.arange(0.1, 12.1, 0.1):
        #print(ii, f)
        simplices = set()
        for v in val:
            if v[1] <= f:
                simplices.add(tuple(v[0]))
    #print(simplices)

        ww = dict()
        for ss in simplices:
            if len(ss) == 1:
                ww[ss] = np.abs(qcharge[ss[0]])+0.1
            elif len(ss) == 2:
                dd = np.linalg.norm(pts[ss[0]]-pts[ss[1]])
                ww[ss] = dd #np.abs(qcharge[ss[0]])*np.abs(qcharge[ss[1]])/dd 
            elif len(ss) == 3:
                d1 = np.linalg.norm(pts[ss[0]]-pts[ss[1]])
                d2 = np.linalg.norm(pts[ss[1]]-pts[ss[2]])
                d3 = np.linalg.norm(pts[ss[0]]-pts[ss[2]])
                semiperi = 0.5*(d1+d2+d3)
                #print(semiperi*(semiperi-d1)*(semiperi-d2)*(semiperi-d3))
                if semiperi*(semiperi-d1)*(semiperi-d2)*(semiperi-d3) < 1e-6:
                    area = 0.000001
                else:
                    area = np.sqrt(semiperi*(semiperi-d1)*(semiperi-d2)*(semiperi-d3))
                ww[ss] = area #np.abs(qcharge[ss[0]])*np.abs(qcharge[ss[1]])*np.abs(qcharge[ss[2]])/area 
    
        #print(ww)
        G = G_matrix(simplices, pts, ww, 2)[::-1]
        #print(G)

        b_mats = []
        b_mats.append([boundary_operator(simplices, 1).toarray(), boundary_operator(simplices, 2).toarray()])
        size0, size1 = np.shape(b_mats[-1][0]), np.shape(b_mats[-1][1])
        wdmat0 = np.array([])
        wdmat1 = np.array([])
        if size0[1] > 0:
            wdmat0 = np.bmat([[np.zeros((size0[0],size0[0])), np.matmul(np.linalg.inv(G[0].todense()), np.matmul(b_mats[-1][0], G[1].todense()))/math.sqrt(2)], [np.transpose(b_mats[-1][0])/math.sqrt(2), np.zeros((size1[0], size1[0]))]])
        else:
            wdmat0 = np.zeros((size0[0],size0[0]))
            
        if size1[1] > 0: 
            wdmat1 = np.bmat([[np.zeros((size0[0],size0[0])), np.matmul(np.linalg.inv(G[0].todense()), np.matmul(b_mats[-1][0], G[1].todense()))/math.sqrt(2), np.zeros((size0[0], size1[1]))], [np.transpose(b_mats[-1][0])/math.sqrt(2), np.zeros((size1[0], size1[0])), np.matmul(np.linalg.inv(G[1].todense()), np.matmul(b_mats[-1][1], G[2].todense()))/math.sqrt(3)], [np.zeros((size1[1], size0[0])), np.transpose(b_mats[-1][1])/math.sqrt(3), np.zeros((size1[1], size1[1]))]])
        elif size0[1] > 0:
            wdmat1 = np.bmat([[np.zeros((size0[0],size0[0])), np.matmul(np.linalg.inv(G[0].todense()), np.matmul(b_mats[-1][0], G[1].todense()))/math.sqrt(2)], [np.transpose(b_mats[-1][0])/math.sqrt(2), np.zeros((size1[0], size1[0]))]])
        else:
            wdmat1 = np.zeros((size0[0],size0[0]))
        
        d0_eig_noH.append(np.linalg.eigvals(wdmat0))
        d1_eig_noH.append(np.linalg.eigvals(wdmat1))

        #print(f, np.shape(d0_eig_noH), np.shape(d1_eig_noH))
        #print(size0, size1)

    pts2 = []
    for i in range(len(contents)):
        if typ[i] != "H" and typ[i] != "C":
            pts2.append(contents[i])

    alpha = gd.RipsComplex(points=pts2)
    st = alpha.create_simplex_tree(max_dimension=2)
    val = list(st.get_filtration())
    #print(val[-1])

    for f in np.arange(0.1, 12.1, 0.1):
        #print(ii, f)
        simplices = set()
        for v in val:
            if v[1] <= f:
                simplices.add(tuple(v[0]))
        #print(simplices)

        ww = dict()
        for ss in simplices:
            if len(ss) == 1:
                ww[ss] = np.abs(qcharge[ss[0]])+0.1
            elif len(ss) == 2:
                dd = np.linalg.norm(pts2[ss[0]]-pts2[ss[1]])
                ww[ss] = dd #np.abs(qcharge[ss[0]])*np.abs(qcharge[ss[1]])/dd 
            elif len(ss) == 3:
                d1 = np.linalg.norm(pts2[ss[0]]-pts2[ss[1]])
                d2 = np.linalg.norm(pts2[ss[1]]-pts2[ss[2]])
                d3 = np.linalg.norm(pts2[ss[0]]-pts2[ss[2]])
                semiperi = 0.5*(d1+d2+d3)
                if semiperi*(semiperi-d1)*(semiperi-d2)*(semiperi-d3) < 1e-6:
                    area = 0.000001
                else:
                    area = np.sqrt(semiperi*(semiperi-d1)*(semiperi-d2)*(semiperi-d3))
                ww[ss] = area #np.abs(qcharge[ss[0]])*np.abs(qcharge[ss[1]])*np.abs(qcharge[ss[2]])/area 
    
        #print(ww)
        G = G_matrix(simplices, pts2, ww, 2)[::-1]

        b_mats = []
        b_mats.append([boundary_operator(simplices, 1).toarray(), boundary_operator(simplices, 2).toarray()])
        size0, size1 = np.shape(b_mats[-1][0]), np.shape(b_mats[-1][1])
        wdmat0 = np.array([])
        wdmat1 = np.array([])
        if size0[1] > 0:
            wdmat0 = np.bmat([[np.zeros((size0[0],size0[0])), np.matmul(np.linalg.inv(G[0].todense()), np.matmul(b_mats[-1][0], G[1].todense()))/math.sqrt(2)], [np.transpose(b_mats[-1][0])/math.sqrt(2), np.zeros((size1[0], size1[0]))]])
        else:
            wdmat0 = np.zeros((size0[0],size0[0]))
            
        if size1[1] > 0: 
            wdmat1 = np.bmat([[np.zeros((size0[0],size0[0])), np.matmul(np.linalg.inv(G[0].todense()), np.matmul(b_mats[-1][0], G[1].todense()))/math.sqrt(2), np.zeros((size0[0], size1[1]))], [np.transpose(b_mats[-1][0])/math.sqrt(2), np.zeros((size1[0], size1[0])), np.matmul(np.linalg.inv(G[1].todense()), np.matmul(b_mats[-1][1], G[2].todense()))/math.sqrt(3)], [np.zeros((size1[1], size0[0])), np.transpose(b_mats[-1][1])/math.sqrt(3), np.zeros((size1[1], size1[1]))]])
        elif size0[1] > 0:
            wdmat1 = np.bmat([[np.zeros((size0[0],size0[0])), np.matmul(np.linalg.inv(G[0].todense()), np.matmul(b_mats[-1][0], G[1].todense()))/math.sqrt(2)], [np.transpose(b_mats[-1][0])/math.sqrt(2), np.zeros((size1[0], size1[0]))]])
        else:
            wdmat1 = np.zeros((size0[0],size0[0]))
        
        d0_eig_noCH.append(np.linalg.eigvals(wdmat0))
        d1_eig_noCH.append(np.linalg.eigvals(wdmat1))

        #print(f, np.shape(d0_eig_noCH), np.shape(d1_eig_noCH))
        #print(size0, size1)

    wfeat = np.array([d0_eig_all, d1_eig_all, d0_eig_noH, d1_eig_noH, d0_eig_noCH, d1_eig_noCH])
    print(idx, np.shape(wfeat))
    np.save('./FreeSolv-wd-npys/FreeSolv_wfeat_{}.npy'.format(idx), wfeat)

def dirac(idx):
    contents  = np.load('./FreeSolv-npys/FreeSolv_mol_{}.npy'.format(idx), allow_pickle=True)
    
    contents, qcharge, typ = np.array(contents[:, :3], dtype=float), np.array(contents[:, 3], dtype=float), contents[:, 4]
    #print(np.unique(typ))

    b_mats = []

    d0_eig_all, d1_eig_all = [], []
    d0_eig_noH, d1_eig_noH = [], []
    d0_eig_noCH, d1_eig_noCH = [], []
    
    alpha = gd.AlphaComplex(points=contents)
    st = alpha.create_simplex_tree()
    val = list(st.get_filtration())
    #print(val[-1])
    #print(val[-1][1])
    for f in np.arange(0.1, 12.1, 0.1):
        #print(ii, f)
        simplices = set()
        for v in val:
            if np.sqrt(v[1])*2 <= f:
                simplices.add(tuple(v[0]))
        #print(simplices)
        b_mats.append([boundary_operator(simplices, 1).toarray(), boundary_operator(simplices, 2).toarray()]) #, boundary_operator(simplices, 3).toarray()])
        size0, size1 = np.shape(b_mats[-1][0]), np.shape(b_mats[-1][1]) #, np.shape(b_mats[-1][2])
        dmat0 = np.array([])
        dmat1 = np.array([])
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

        d0_eig_all.append(np.linalg.eigvalsh(dmat0))
        d1_eig_all.append(np.linalg.eigvalsh(dmat1))

    pts = []
    for i in range(len(contents)):
        if typ[i] != "H":
            pts.append(contents[i])
    
    alpha = gd.RipsComplex(points=pts)
    st = alpha.create_simplex_tree(max_dimension=2)
    val = list(st.get_filtration())
    #print(val[-1])

    for f in np.arange(0.1, 12.1, 0.1):
        #print(ii, f)
        simplices = set()
        for v in val:
            if v[1] <= f:
                simplices.add(tuple(v[0]))
        #print(simplices)
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
    
        d0_eig_noH.append(np.linalg.eigvalsh(dmat0))
        d1_eig_noH.append(np.linalg.eigvalsh(dmat1))

    pts2 = []
    for i in range(len(contents)):
        if typ[i] != "H" and typ[i] != "C":
            pts2.append(contents[i])

    alpha = gd.RipsComplex(points=pts2)
    st = alpha.create_simplex_tree(max_dimension=2)
    val = list(st.get_filtration())
    #print(val[-1])

    for f in np.arange(0.1, 12.1, 0.1):
        #print(ii, f)
        simplices = set()
        for v in val:
            if v[1] <= f:
                simplices.add(tuple(v[0]))
        #print(simplices)
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

        d0_eig_noCH.append(np.linalg.eigvalsh(dmat0))
        d1_eig_noCH.append(np.linalg.eigvalsh(dmat1))

    #print(d0_eig_all)
    #print(d1_eig_all)
    #print(d0_eig_noH)
    #print(d1_eig_noH)
    #print(d0_eig_noCH)
    #print(d1_eig_noCH)
    feat = np.array([d0_eig_all, d1_eig_all, d0_eig_noH, d1_eig_noH, d0_eig_noCH, d1_eig_noCH])
    print(idx, np.shape(feat))
    np.save('./FreeSolv-npys/FreeSolv_feat_{}.npy'.format(idx), feat)



start, end = int(sys.argv[1]), int(sys.argv[2])

"""
for idx in range(start, end):
    file = open('./FreeSolv-mol2s/FreeSolv_mol_{}.mol2'.format(idx))

    contents = file.readlines()
    for i in range(len(contents)):
        contents[i] = contents[i].rstrip("\n").split()
        #print(contents[i])
        if len(contents[i])>0:
            if contents[i][0] == "@<TRIPOS>ATOM":
                atomidx = i 
            if contents[i][0] == "@<TRIPOS>BOND":
                bondidx = i

    data = []
    for i in range(atomidx+1, bondidx):
        #print(contents[i])
        #contents[i] = [float(s) for s in contents[i]]
        x, y, z, typ, charge = float(contents[i][2]), float(contents[i][3]), float(contents[i][4]), (contents[i][5].split("."))[0], float(contents[i][-1])
        data.append([x,y,z, charge, typ])

    if not os.path.exists('./FreeSolv-npys'):
        os.mkdir('./FreeSolv-npys')
    np.save('./FreeSolv-npys/FreeSolv_mol_{}.npy'.format(idx), data)

    #dirac(idx)
    weighted_dirac(idx)
    
"""
no_threads = mp.cpu_count()
p = mp.Pool(processes = no_threads)
results = p.map(weighted_dirac, list(range(start, end)))
#results = p.map(func, arr)
p.close()
p.join()

        
        
        