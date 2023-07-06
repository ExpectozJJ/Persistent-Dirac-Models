import numpy as np 
import pandas as pd
import math

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

data = pd.read_csv('SAMPL.csv').to_numpy()
solv = []
for i in range(len(data)):
    solv.append(data[i][2])

np.save('solv.npy', solv)

ff = []
wff = []
for idx in range(1, 643):
    eigs = np.load('./FreeSolv-npys/FreeSolv_feat_{}.npy'.format(idx), allow_pickle=True)
    weigs = np.load('./FreeSolv-wd-npys/FreeSolv_wfeat_{}.npy'.format(idx), allow_pickle=True)

    #print(np.shape(eigs))
    feat = []
    wfeat = []
    for i in range(6):
        for j in range(120):
            #print(np.shape(eigs[i][j]))
            feat.append(compute_stat(eigs[i][j]))
            wfeat.append(compute_stat(eigs[i][j]))

    #print(idx, np.shape(feat))
    ff.append(np.reshape(feat, (720*12)))
    wff.append(np.reshape(wfeat, (720*12)))

np.save('X.npy', ff)
np.save('wX.npy', wff)