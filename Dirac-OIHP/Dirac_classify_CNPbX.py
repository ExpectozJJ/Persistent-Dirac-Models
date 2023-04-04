import numpy as np
import math
import scipy.io as sio
from GeneralisedFormanRicci.frc import GeneralisedFormanRicci, n_faces
import os
import sys
#import umap
#import umap.plot
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib as mpl

def normalize(X):
    mean = np.mean(X,axis=0)
    std = np.std(X,axis=0)
    length1 = X.shape[0]
    X_train_normed = X

    for i in range(0,length1):
        for j in range(0,X.shape[1]):
            if std[j]!=0 :
                X_train_normed[i,j] = (X_train_normed[i,j]-mean[j])/std[j]
    return X_train_normed

def compute_stat(dist):
    dist = np.array(dist)
    if len(dist) == 0:
        feat = [0,0,0,0,0,0,0,0,0,0,0,0] # If the input is empty, output all-zero stat
    else:
        feat = []
        cnt = 0
        
        for l in dist:
            if l < 1e-3 and l > -1e-3:
                cnt+=1

        feat.append(cnt)                                    # sum of persistent multiplicity

        dist = dist[dist>=1e-3]
        if len(dist) == 0:
            feat = [feat[0], 0,0,0,0,0,0,0,0,0,0,0]
        else:
            feat.append(np.min(dist))                       # persistent min (Fiedler Value - Algebraic Connectivity)
            feat.append(np.max(dist))                       # persistent max
            feat.append(np.mean(dist))                      # persistent mean
            feat.append(np.std(dist))                       # persistent std

            feat.append(np.sum(np.abs(dist)))               # persistent Laplacian Graph Energy 
            feat.append(len(dist))                          # persistent number of non-zero eigenvalue pairs (signless Euler-Poincare number)
            s, t, u, v, w, x, y, z = 0, 0, [], 0, 0, 0, 0, 0
            for l in dist:
                w += l*l
                u.append(abs(l-feat[3])/len(dist))
                v += 1/l
                t += 2*(l**(-2))
                if l > 0:
                    s += math.log(l)
            
            u = np.array(u)
            feat.append(np.sum(u))                           # persistent Laplacian Generalised Mean Graph Energy
            feat.append(w)                                   # persistent spectral 2nd moment
            feat.append(t)                                   # persistent zeta(2) of laplacian
            feat.append((len(dist)+1)*v)                     # persistent quasi-Wiener Index
            feat.append(s-math.log(len(dist)+1))             # persistent spanning tree number

    return feat
"""
MAPbBr3_Cubic_CNPbXdata0 = []
MAPbBr3_Cubic_CNPbXdata1 = []
for i in range(501, 1001):
    MAPbBr3_Cubic_CNPbXdata0.append(np.load("./MAPbBr3_Cubic_CNXPb_data/CNXPb_atmlist_L5_f{}_d0.npy".format(i), allow_pickle=True))
    MAPbBr3_Cubic_CNPbXdata1.append(np.load("./MAPbBr3_Cubic_CNXPb_data/CNXPb_atmlist_L5_f{}_d1.npy".format(i), allow_pickle=True))

MAPbCl3_Cubic_CNPbXdata0 = []
MAPbCl3_Cubic_CNPbXdata1 = []
for i in range(501, 1001):
    MAPbCl3_Cubic_CNPbXdata0.append(np.load("./MAPbCl3_Cubic_CNXPb_data/CNXPb_atmlist_L5_f{}_d0.npy".format(i), allow_pickle=True))
    MAPbCl3_Cubic_CNPbXdata1.append(np.load("./MAPbCl3_Cubic_CNXPb_data/CNXPb_atmlist_L5_f{}_d1.npy".format(i), allow_pickle=True))

MAPbI3_Cubic_CNPbXdata0 = []
MAPbI3_Cubic_CNPbXdata1 = []
for i in range(501, 1001):
    MAPbI3_Cubic_CNPbXdata0.append(np.load("./MAPbI3_Cubic_CNXPb_data/CNXPb_atmlist_L5_f{}_d0.npy".format(i), allow_pickle=True))
    MAPbI3_Cubic_CNPbXdata1.append(np.load("./MAPbI3_Cubic_CNXPb_data/CNXPb_atmlist_L5_f{}_d1.npy".format(i), allow_pickle=True))

MAPbBr3_Ortho_CNPbXdata0 = []
MAPbBr3_Ortho_CNPbXdata1 = []
for i in range(501, 1001):
    MAPbBr3_Ortho_CNPbXdata0.append(np.load("./MAPbBr3_Orthorhombic_CNXPb_data/CNXPb_atmlist_L5_f{}_d0.npy".format(i), allow_pickle=True))
    MAPbBr3_Ortho_CNPbXdata1.append(np.load("./MAPbBr3_Orthorhombic_CNXPb_data/CNXPb_atmlist_L5_f{}_d1.npy".format(i), allow_pickle=True))

MAPbCl3_Ortho_CNPbXdata0 = []
MAPbCl3_Ortho_CNPbXdata1 = []
for i in range(501, 1001):
    MAPbCl3_Ortho_CNPbXdata0.append(np.load("./MAPbCl3_Orthorhombic_CNXPb_data/CNXPb_atmlist_L5_f{}_d0.npy".format(i), allow_pickle=True))
    MAPbCl3_Ortho_CNPbXdata1.append(np.load("./MAPbCl3_Orthorhombic_CNXPb_data/CNXPb_atmlist_L5_f{}_d1.npy".format(i), allow_pickle=True))

MAPbI3_Ortho_CNPbXdata0 = []
MAPbI3_Ortho_CNPbXdata1 = []
for i in range(501, 1001):
    MAPbI3_Ortho_CNPbXdata0.append(np.load("./MAPbI3_Orthorhombic_CNXPb_data/CNXPb_atmlist_L5_f{}_d0.npy".format(i), allow_pickle=True))
    MAPbI3_Ortho_CNPbXdata1.append(np.load("./MAPbI3_Orthorhombic_CNXPb_data/CNXPb_atmlist_L5_f{}_d1.npy".format(i), allow_pickle=True))

MAPbBr3_Tetra_CNPbXdata0 = []
MAPbBr3_Tetra_CNPbXdata1 = []
for i in range(501, 1001):
    MAPbBr3_Tetra_CNPbXdata0.append(np.load("./MAPbBr3_Tetragonal_CNXPb_data/CNXPb_atmlist_L5_f{}_d0.npy".format(i), allow_pickle=True))
    MAPbBr3_Tetra_CNPbXdata1.append(np.load("./MAPbBr3_Tetragonal_CNXPb_data/CNXPb_atmlist_L5_f{}_d1.npy".format(i), allow_pickle=True))

MAPbCl3_Tetra_CNPbXdata0 = []
MAPbCl3_Tetra_CNPbXdata1 = []
for i in range(501, 1001):
    MAPbCl3_Tetra_CNPbXdata0.append(np.load("./MAPbCl3_Tetragonal_CNXPb_data/CNXPb_atmlist_L5_f{}_d0.npy".format(i), allow_pickle=True))
    MAPbCl3_Tetra_CNPbXdata1.append(np.load("./MAPbCl3_Tetragonal_CNXPb_data/CNXPb_atmlist_L5_f{}_d1.npy".format(i), allow_pickle=True))

MAPbI3_Tetra_CNPbXdata0 = []
MAPbI3_Tetra_CNPbXdata1 = []
for i in range(501, 1001):
    MAPbI3_Tetra_CNPbXdata0.append(np.load("./MAPbI3_Tetragonal_CNXPb_data/CNXPb_atmlist_L5_f{}_d0.npy".format(i), allow_pickle=True))
    MAPbI3_Tetra_CNPbXdata1.append(np.load("./MAPbI3_Tetragonal_CNXPb_data/CNXPb_atmlist_L5_f{}_d1.npy".format(i), allow_pickle=True))

for a in ['Br', 'Cl', 'I']:
    for b in ['Cubic', 'Ortho', 'Tetra']:
        print(np.shape(vars()["MAPb{}3_{}_CNPbXdata0".format(a,b)]))
        print(np.shape(vars()["MAPb{}3_{}_CNPbXdata1".format(a,b)]))

dirac_feat0 = []
dirac_feat1 = []
for a in ['Br', 'Cl', 'I']:
    for b in ['Cubic', 'Ortho', 'Tetra']:
        #a_feat = []
        for i in range(len(vars()["MAPb{}3_{}_CNPbXdata0".format(a,b)])):
            feat0 = []
            for j in range(len(vars()["MAPb{}3_{}_CNPbXdata0".format(a,b)][i])):
                feat0.append(compute_stat(vars()["MAPb{}3_{}_CNPbXdata0".format(a,b)][i][j]))
            #a_feat.append(feat)
            dirac_feat0.append(feat0)

        for i in range(len(vars()["MAPb{}3_{}_CNPbXdata1".format(a,b)])):
            feat1 = []
            for j in range(len(vars()["MAPb{}3_{}_CNPbXdata1".format(a,b)][i])):
                feat1.append(compute_stat(vars()["MAPb{}3_{}_CNPbXdata1".format(a,b)][i][j]))
            #a_feat.append(feat)
            dirac_feat1.append(feat1)

#print(dirac_feat[0])
dirac_feat0 = np.reshape(dirac_feat0, (4500, 23*12))
dirac_feat1 = np.reshape(dirac_feat1, (4500, 23*12))
dirac_feat = np.concatenate((dirac_feat0, dirac_feat1), axis=1)
print(np.shape(dirac_feat))
#print((dirac_feat[0]==dirac_feat[1]).all())

np.save("dirac_feat_d0.npy", dirac_feat0)
np.save("dirac_feat_d1.npy", dirac_feat1)
np.save("dirac_feat_d0_d1.npy", dirac_feat)
sio.savemat("dirac_feat_d0.mat", {'fdata': dirac_feat0})
sio.savemat("dirac_feat_d1.mat", {'fdata': dirac_feat1})
sio.savemat("dirac_feat_d0_d1.mat", {'fdata': dirac_feat})
"""

frd = 1
frs = 0

dirac_feat1 = np.load("dirac_feat_d0_d1.npy", allow_pickle=True)

values = TSNE(n_components=2, n_iter=500, perplexity=50, verbose=True).fit_transform(dirac_feat1)
plt.figure(figsize=(5,5), dpi=200)
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
plt.scatter(values[:500,0], values[:500,1], marker='.', color='tab:blue', alpha=0.75, linewidth=.5, s=20, label="Br-Cubic")
plt.scatter(values[500:1000,0], values[500:1000,1], marker='.', color='tab:orange', alpha=0.75,  linewidth=0.5, s=20, label="Br-Ortho")
plt.scatter(values[1000:1500,0], values[1000:1500,1], marker='.', color='tab:green', alpha=0.75,  linewidth=0.5, s=20, label="Br-Tetra")

plt.scatter(values[1500:2000,0], values[1500:2000,1], marker='.', color='tab:red', alpha=0.75,  linewidth=0.5, s=20, label="Cl-Cubic")
plt.scatter(values[2000:2500,0], values[2000:2500,1], marker='.', color='tab:purple', alpha=0.75,  linewidth=0.5, s=20, label="Cl-Ortho")
plt.scatter(values[2500:3000,0], values[2500:3000,1], marker='.', color='tab:brown', alpha=0.75,  linewidth=0.5, s=20, label="Cl-Tetra")

plt.scatter(values[3000:3500,0], values[3000:3500,1],  marker='.',color='tab:pink', alpha=0.75,  linewidth=0.5, s=20, label="I-Cubic")
plt.scatter(values[3500:4000,0], values[3500:4000,1],  marker='.',color='tab:gray', alpha=0.75,  linewidth=0.5, s=20, label="I-Ortho")
plt.scatter(values[4000:4500,0], values[4000:4500,1],  marker='.',color='tab:olive', alpha=0.75,  linewidth=0.5, s=20, label="I-Tetra")

#plt.ylim(np.min(values[:, 1])-10, np.max(values[:,1])+50)
#plt.xlim(-100, 100)
#plt.legend(ncol=3, loc='upper left', handlelength=.5, borderpad=.25, fontsize=10)
plt.axis('equal')
plt.xticks([])
plt.yticks([])
plt.savefig("MAPbX3_Classification_CNPbX_tSNE_d1.png", dpi=200)
#plt.show()

