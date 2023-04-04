import numpy as np 
import gudhi as gd
from scipy.sparse import *
from scipy import *
import seaborn as sns
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GeneralisedFormanRicci.frc import GeneralisedFormanRicci, gen_graph, n_faces
import networkx as nx
import plotly.graph_objects as go
import math
import matplotlib as mpl
import matplotlib
import plotly.io as pio
import matplotlib.pyplot as plt

pio.orca.config.use_xvfb = True

def rotate_z(x, y, z, theta):
    w = x+1j*y
    return np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), z

def matplotlib_to_plotly(cmap, pl_entries):
    h = 1.0/(pl_entries-1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = list(map(np.uint8, np.array(cmap(k*h)[:3])*255))
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])

    return pl_colorscale

""" Normalise Colormap to unique range """ 
seismic_cmap = matplotlib.cm.get_cmap('Blues')

seismic_rgb = []
norm = mpl.colors.Normalize(vmin=0, vmax=255)

for i in range(0, 255):
    k = mpl.colors.colorConverter.to_rgb(seismic_cmap(norm(i)))
    seismic_rgb.append(k)

seismic = matplotlib_to_plotly(seismic_cmap, 255)

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
    return filter(lambda coface: set(simplex) < set(coface) and len(coface)==n+1, simplices)

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

def G_matrix(face_set, ptcloud, d):
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
        if d == 2:
            S[j,j] = 1 #len(source_simplex)
        elif d == 1:
            tmp = source_simplices[j]
            S[j,j] = ww[tmp]
        elif d == 0:
            S[j,j] = 1
    
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

ptcloud = [[0,1,0],[-0.5,0,0],[0.5,0,0],[0,-1,0]]

simplices = {(0,), (1,), (2,), (3,), (0,1), (0,2), (1,2), (2,3), (1,3),(0,1,2)} #, (0,1,2), (1,2,3)}#, (0,5)}#, (0,6), (0,6, 7)}

data = np.load("dir_to_molecule/guanine.npz", allow_pickle=True)
for d in data["PRO"]:
    typ = d['typ']
    ptcloud = d['pos']

rips_complex = gd.RipsComplex(ptcloud)
simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)
val = list(simplex_tree.get_filtration())
for f in [1.2]:#np.arange(0, 3, 0.1, dtype=float):
    print(f)
    simplices = set()
    for v in val:
        if v[1]<=f:
            simplices.add(tuple(v[0]))

b_mats = []
b_mats.append([boundary_operator(simplices, 1).toarray(), boundary_operator(simplices, 2).toarray(), boundary_operator(simplices,3).toarray()])
size0, size1, size2 = np.shape(b_mats[-1][0]), np.shape(b_mats[-1][1]), np.shape(b_mats[-1][2])
dmat0 = np.array([])
dmat1 = np.array([])
dmat2 = np.array([])
if size0[1] > 0:
    dmat0 = np.bmat([[np.zeros((size0[0],size0[0])), b_mats[-1][0]], [np.transpose(b_mats[-1][0]), np.zeros((size1[0], size1[0]))]])
if size1[1] > 0: 
    dmat1 = np.bmat([[np.zeros((size0[0],size0[0])), b_mats[-1][0], np.zeros((size0[0], size1[1]))], [np.transpose(b_mats[-1][0]), np.zeros((size1[0], size1[0])), b_mats[-1][1]], [np.zeros((size1[1], size0[0])), np.transpose(b_mats[-1][1]), np.zeros((size1[1], size1[1]))]])
if size2[1] > 0:
    dmat2 = np.bmat([[np.zeros((size0[0],size0[0])), b_mats[-1][0], np.zeros((size0[0], size1[1])), np.zeros((size0[0], size2[1]))], [np.transpose(b_mats[-1][0]), np.zeros((size1[0], size1[0])), b_mats[-1][1], np.zeros((size1[0], size2[1]))], [np.zeros((size1[1], size0[0])), np.transpose(b_mats[-1][1]), np.zeros((size1[1], size1[1])), b_mats[-1][2]], [np.zeros((size2[1], size0[0])), np.zeros((size2[1], size1[0])), np.transpose(b_mats[-1][2]), np.zeros((size2[1],size2[1]))]])

ww = dict()
for ss in simp:
    ww[ss] = 1
    if ss == (6,7): 
        ww[ss] = 5

G = G_matrix(simplices, ptcloud, 2)[::-1]

# Weighted HL matrix 
L1_down = (1/2)*np.matmul(np.transpose(boundary_operator(simplices, 1).toarray()), np.matmul(np.linalg.inv(G[0].todense()), np.matmul(boundary_operator(simplices, 1).toarray(), G[1].todense())))
L1_up = (1/3)*np.matmul(np.linalg.inv(G[1].todense()), np.matmul(boundary_operator(simplices, 2).toarray(), np.matmul(G[2].todense(), np.transpose(boundary_operator(simplices, 2).toarray()))))

L1 = L1_up + L1_down 
wl_eigval, wl_eigvec = np.linalg.eig(L1)

L0 = (1/2)*np.matmul(np.linalg.inv(G[0].todense()), np.matmul(boundary_operator(simplices, 1).toarray(), np.matmul(G[1].todense(), np.transpose(boundary_operator(simplices, 1).toarray()))))
wl_eigval, wl_eigvec = np.linalg.eig(L0)

unweighted_L1 = np.matmul(np.transpose(boundary_operator(simplices, 1).toarray()), boundary_operator(simplices, 1).toarray()) + np.matmul(boundary_operator(simplices, 2).toarray(), np.transpose(boundary_operator(simplices, 2).toarray()))
eigval, eigvec = np.linalg.eigh(unweighted_L1)
wl_eigvec = np.array(wl_eigvec)
unweighted_L0 = np.matmul(boundary_operator(simplices, 1).toarray(), np.transpose(boundary_operator(simplices, 1).toarray()))
eigval, eigvec = np.linalg.eigh(unweighted_L0)

for idx in [-1, -6, -7]:
    G= gen_graph(list(n_faces(simplices, 1)), ptcloud, {}) # Get the Graph Network of Simplicial Complex
    s0 = len(list(n_faces(simplices,0)))
    s1 = len(list(n_faces(simplices,1)))
    s2 = len(list(n_faces(simplices,2)))
    s3 = len(list(n_faces(simplices,3)))
    #node_dict = {list(n_faces(simplices,0))[i]: np.float(eigvec[:,idx][i]) for i in range(len(list(n_faces(simplices,0))))}
    edge_dict = {list(n_faces(simplices,1))[i]: np.abs(eigvec[:,idx][i]) for i in range(len(list(n_faces(simplices,1))))}
    #tri_dict = {list(n_faces(simplices,2))[i]: np.float(np.abs(eigvec[:,idx])[s0+s1+i]) for i in range(len(list(n_faces(simplices,2))))}
    #tetra_dict = {list(n_faces(simplices,3))[i]: np.float(np.abs(eigvec[:,idx])[s0+s1+s2+i]) for i in range(len(list(n_faces(simplices,3))))}
    
    #print(edge_dict)
    
    """                   Plot Eigenvectors of L1                    """
    norm = mpl.colors.Normalize(vmin=np.min(eigvec[:,idx]), vmax=np.max(eigvec[:,idx])) # Modify the colour map range
    edge_c = dict()
    tri_c = dict()

    edge_traces = []

    node_x = []
    node_y = []
    node_z = []
    for node in G.nodes():
        x, y, z = G.nodes[node]['coords']
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)

    edge_traces.append(go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            colorscale=seismic, color = 'black',
            size=5, opacity=1, line=dict(color='black', width=30)
            ),line_width=10))

    for edge in G.edges():
        edge_x = []
        edge_y = []
        edge_z = []
        x0, y0, z0 = G.nodes[edge[0]]['coords']
        x1, y1, z1 = G.nodes[edge[1]]['coords']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        edge_z.append(z0)
        edge_z.append(z1)
        edge_z.append(None)

        e = edge_dict[edge]
        #e = ((e-np.mean(list(edge_dict.values())))/(np.std(list(edge_dict.values()))))
        #rint(norm(e))
        color = mpl.cm.Blues(e, bytes=True)
        edge_c[edge] = "rgba({},{},{},{})".format(color[0],color[1],color[2], color[3])

        edge_traces.append(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z, text='{}'.format(round(edge_dict[edge])),
            line=dict(width=20, color=edge_c[edge]),
            hoverinfo='none', opacity=.75,
            mode='lines'))

    color_range = seismic_cmap
    color_trace = go.Scatter3d(x=node_x,y=node_y, z=node_z,
                             mode='markers',
                             marker=go.scatter3d.Marker(colorscale=seismic, cmin=0, cmax=np.max(abs(eigvec[:,idx])),
                                size=.01, opacity=1, line=dict(color='black', width=3),
                                showscale=True, colorbar=dict(thickness=20, outlinewidth=2, tickfont=dict(family='Arial', size=30))
                                )
                            )

    edge_traces.append(color_trace)
    """
    for key, val in tri_dict.items():
        s = G.nodes[key[0]]['coords']
        t = G.nodes[key[1]]['coords']
        u = G.nodes[key[2]]['coords']
        e = tri_dict[key]
        #color = mpl.cm.seismic(e, bytes=True)
        #tri_c[key] = "rgba({},{},{},{})".format(color[0],color[1],color[2], color[3])
        edge_traces.append(go.Mesh3d(x=[s[0], t[0], u[0]], y=[s[1], t[1], u[1]], z=[s[2], t[2], u[2]], color='gray', opacity=.5))
    """

    fig = go.Figure(data=edge_traces,
                 layout=go.Layout(
                    showlegend=False, 
                    hovermode='closest',
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    #fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(width=700, height=700, scene = dict(
        xaxis=dict(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        showbackground=False,
        title='',
        ticks='',
        showticklabels=False
    ),
    yaxis=dict(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        showbackground=False,
        ticks='',
        title='',
        showticklabels=False
    ),
    zaxis=dict(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        showbackground=False,
        title='',
        ticks='',
        showticklabels=False
    )))

    xe, ye, ze = rotate_z(0, 0, 1.75, -.1)
    camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=xe, y=ye, z=ze)
    )

    fig.update_layout(scene_camera=camera, scene_dragmode='orbit')
    #fig.write_html("1bna_CA_forman_edge_2_5.html")
    fig.show()

for idx in [-3]:
    G= gen_graph(list(n_faces(simplices, 1)), ptcloud, {}) # Get the Graph Network of Simplicial Complex
    s0 = len(list(n_faces(simplices,0)))
    s1 = len(list(n_faces(simplices,1)))
    s2 = len(list(n_faces(simplices,2)))
    s3 = len(list(n_faces(simplices,3)))
    #node_dict = {list(n_faces(simplices,0))[i]: np.float(eigvec[:,idx][i]) for i in range(len(list(n_faces(simplices,0))))}
    edge_dict = {list(n_faces(simplices,1))[i]: abs(wl_eigvec[:,idx][i]) for i in range(len(list(n_faces(simplices,1))))}
    #tri_dict = {list(n_faces(simplices,2))[i]: np.float(np.abs(eigvec[:,idx])[s0+s1+i]) for i in range(len(list(n_faces(simplices,2))))}
    #tetra_dict = {list(n_faces(simplices,3))[i]: np.float(np.abs(eigvec[:,idx])[s0+s1+s2+i]) for i in range(len(list(n_faces(simplices,3))))}
    
    #print(edge_dict)
    
    """                   Plot Eigenvectors of Weighted L1                    """
    norm = mpl.colors.Normalize(vmin=np.min(wl_eigvec[:,idx]), vmax=np.max(wl_eigvec[:,idx])) # Modify the colour map range
    edge_c = dict()
    tri_c = dict()

    edge_traces = []

    node_x = []
    node_y = []
    node_z = []
    for node in G.nodes():
        x, y, z = G.nodes[node]['coords']
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)

    edge_traces.append(go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            colorscale=seismic, color = 'black',
            size=5, opacity=1, line=dict(color='black', width=30)
            ),line_width=10))

    for edge in G.edges():
        edge_x = []
        edge_y = []
        edge_z = []
        x0, y0, z0 = G.nodes[edge[0]]['coords']
        x1, y1, z1 = G.nodes[edge[1]]['coords']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        edge_z.append(z0)
        edge_z.append(z1)
        edge_z.append(None)

        e = edge_dict[edge]
        color = mpl.cm.Blues(e, bytes=True)
        edge_c[edge] = "rgba({},{},{},{})".format(color[0],color[1],color[2], color[3])

        edge_traces.append(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z, text='{}'.format(round(edge_dict[edge])),
            line=dict(width=20, color=edge_c[edge]),
            hoverinfo='none', opacity=.75,
            mode='lines'))

    color_range = seismic_cmap
    color_trace = go.Scatter3d(x=node_x,y=node_y, z=node_z,
                             mode='markers',
                             marker=go.scatter3d.Marker(colorscale=seismic, cmin=0, cmax=np.max(abs(wl_eigvec[:,idx])),
                                size=.01, opacity=1, line=dict(color='black', width=3),
                                showscale=True, colorbar=dict(thickness=20, outlinewidth=2, tickfont=dict(family='Arial', size=30))
                                )
                            )

    edge_traces.append(color_trace)
    """
    for key, val in tri_dict.items():
        s = G.nodes[key[0]]['coords']
        t = G.nodes[key[1]]['coords']
        u = G.nodes[key[2]]['coords']
        e = tri_dict[key]
        #color = mpl.cm.seismic(e, bytes=True)
        #tri_c[key] = "rgba({},{},{},{})".format(color[0],color[1],color[2], color[3])
        edge_traces.append(go.Mesh3d(x=[s[0], t[0], u[0]], y=[s[1], t[1], u[1]], z=[s[2], t[2], u[2]], color='gray', opacity=.5))
    """

    fig = go.Figure(data=edge_traces,
                 layout=go.Layout(
                    showlegend=False, 
                    hovermode='closest',
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    #fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(width=700, height=700, scene = dict(
        xaxis=dict(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        showbackground=False,
        title='',
        ticks='',
        showticklabels=False
    ),
    yaxis=dict(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        showbackground=False,
        ticks='',
        title='',
        showticklabels=False
    ),
    zaxis=dict(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        showbackground=False,
        title='',
        ticks='',
        showticklabels=False
    )))

    xe, ye, ze = rotate_z(0, 0, 1.75, -.1)
    camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=xe, y=ye, z=ze)
    )

    fig.update_layout(scene_camera=camera, scene_dragmode='orbit')
    #fig.write_html("1bna_CA_forman_edge_2_5.html")
    fig.show()

# Non-zero Eigenvalues
for idx in [2]:
    G= gen_graph(list(n_faces(simplices, 1)), ptcloud, {}) # Get the Graph Network of Simplicial Complex
    s0 = len(list(n_faces(simplices,0)))
    s1 = len(list(n_faces(simplices,1)))
    s2 = len(list(n_faces(simplices,2)))
    #s3 = len(list(n_faces(simplices,3)))
    #node_dict = {list(n_faces(simplices,0))[i]: np.float(eigvec[:,idx][i]) for i in range(len(list(n_faces(simplices,0))))}
    edge_dict = {list(n_faces(simplices,1))[i]: np.abs(eigvec[:,idx][i]) for i in range(len(list(n_faces(simplices,1))))}
    #tri_dict = {list(n_faces(simplices,2))[i]: np.float(np.abs(eigvec[:,idx])[s0+s1+i]) for i in range(len(list(n_faces(simplices,2))))}
    #tetra_dict = {list(n_faces(simplices,3))[i]: np.float(np.abs(eigvec[:,idx])[s0+s1+s2+i]) for i in range(len(list(n_faces(simplices,3))))}
    
    #print(edge_dict)
    
    """                   Plot Eigenvectors of L1                    """
    norm = mpl.colors.Normalize(vmin=np.min(eigvec[:,idx]), vmax=np.max(eigvec[:, idx])) # Modify the colour map range
    edge_c = dict()
    tri_c = dict()

    edge_traces = []

    node_x = []
    node_y = []
    node_z = []
    for node in G.nodes():
        x, y, z = G.nodes[node]['coords']
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)

    edge_traces.append(go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            colorscale=seismic, color = 'black',
            size=5, opacity=1, line=dict(color='black', width=30)
            ),line_width=10))

    for edge in G.edges():
        edge_x = []
        edge_y = []
        edge_z = []
        x0, y0, z0 = G.nodes[edge[0]]['coords']
        x1, y1, z1 = G.nodes[edge[1]]['coords']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        edge_z.append(z0)
        edge_z.append(z1)
        edge_z.append(None)

        e = edge_dict[edge]
        #e = ((e-np.mean(list(edge_dict.values())))/(np.std(list(edge_dict.values()))))
        #rint(norm(e))
        #print(e)
        color = mpl.cm.Blues(e, bytes=True)
        edge_c[edge] = "rgba({},{},{},{})".format(color[0],color[1],color[2], color[3])

        edge_traces.append(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            line=dict(width=15, color=edge_c[edge]), 
            hoverinfo='none', opacity=.5,
            mode='lines'))

    color_range = seismic_cmap
    color_trace = go.Scatter3d(x=node_x,y=node_y, z=node_z,
                             mode='markers',
                             marker=go.scatter3d.Marker(colorscale=seismic, cmin=np.min(np.abs(eigvec[:,idx])), cmax=np.max(np.abs(eigvec[:, idx])),
                                size=.01, opacity=1, line=dict(color='black', width=3),
                                showscale=True, colorbar=dict(thickness=20, outlinewidth=2, tickfont=dict(family='Arial', size=30))
                                )
                            )

    edge_traces.append(color_trace)
    """
    for key, val in tri_dict.items():
        s = G.nodes[key[0]]['coords']
        t = G.nodes[key[1]]['coords']
        u = G.nodes[key[2]]['coords']
        e = tri_dict[key]
        #color = mpl.cm.seismic(e, bytes=True)
        #tri_c[key] = "rgba({},{},{},{})".format(color[0],color[1],color[2], color[3])
        edge_traces.append(go.Mesh3d(x=[s[0], t[0], u[0]], y=[s[1], t[1], u[1]], z=[s[2], t[2], u[2]], color='gray', opacity=.5))
    """

    fig = go.Figure(data=edge_traces,
                 layout=go.Layout(
                    showlegend=False, 
                    hovermode='closest',
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    #fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(width=700, height=700, scene = dict(
        xaxis=dict(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        showbackground=False,
        title='',
        ticks='',
        showticklabels=False
    ),
    yaxis=dict(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        showbackground=False,
        ticks='',
        title='',
        showticklabels=False
    ),
    zaxis=dict(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        showbackground=False,
        title='',
        ticks='',
        showticklabels=False
    )))

    xe, ye, ze = rotate_z(0, 0, 1.75, -.1)
    camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=xe, y=ye, z=ze)
    )

    fig.update_layout(scene_camera=camera, scene_dragmode='orbit')
    #fig.write_html("1bna_CA_forman_edge_2_5.html")
    fig.show()

for idx in [0,1,2,3,4]:
    G= gen_graph(list(n_faces(simplices, 1)), ptcloud, {}) # Get the Graph Network of Simplicial Complex
    s0 = len(list(n_faces(simplices,0)))
    s1 = len(list(n_faces(simplices,1)))
    s2 = len(list(n_faces(simplices,2)))
    s3 = len(list(n_faces(simplices,3)))
    #node_dict = {list(n_faces(simplices,0))[i]: np.float(eigvec[:,idx][i]) for i in range(len(list(n_faces(simplices,0))))}
    edge_dict = {list(n_faces(simplices,1))[i]: (wl_eigvec[:,idx][i].real) for i in range(len(list(n_faces(simplices,1))))}
    #tri_dict = {list(n_faces(simplices,2))[i]: np.float(np.abs(eigvec[:,idx])[s0+s1+i]) for i in range(len(list(n_faces(simplices,2))))}
    #tetra_dict = {list(n_faces(simplices,3))[i]: np.float(np.abs(eigvec[:,idx])[s0+s1+s2+i]) for i in range(len(list(n_faces(simplices,3))))}
    
    #print(edge_dict)
    
    """                   Plot Eigenvectors of Weighted L1                    """
    norm = mpl.colors.Normalize(vmin=np.min(wl_eigvec[:,idx]), vmax=np.max(wl_eigvec[:,idx])) # Modify the colour map range
    edge_c = dict()
    tri_c = dict()

    edge_traces = []

    node_x = []
    node_y = []
    node_z = []
    for node in G.nodes():
        x, y, z = G.nodes[node]['coords']
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)

    edge_traces.append(go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            colorscale=seismic, color = 'black',
            size=5, opacity=1, line=dict(color='black', width=30)
            ),line_width=10))

    for edge in G.edges():
        edge_x = []
        edge_y = []
        edge_z = []
        x0, y0, z0 = G.nodes[edge[0]]['coords']
        x1, y1, z1 = G.nodes[edge[1]]['coords']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        edge_z.append(z0)
        edge_z.append(z1)
        edge_z.append(None)

        e = edge_dict[edge]
        color = mpl.cm.Blues(e, bytes=True)
        edge_c[edge] = "rgba({},{},{},{})".format(color[0],color[1],color[2], color[3])

        edge_traces.append(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z, text='{}'.format(round(edge_dict[edge])),
            line=dict(width=20, color=edge_c[edge]),
            hoverinfo='none', opacity=.75,
            mode='lines'))

    color_range = seismic_cmap
    color_trace = go.Scatter3d(x=node_x,y=node_y, z=node_z,
                             mode='markers',
                             marker=go.scatter3d.Marker(colorscale=seismic, cmin=0, cmax=np.max(abs(wl_eigvec[:,idx])),
                                size=.01, opacity=1, line=dict(color='black', width=3),
                                showscale=True, colorbar=dict(thickness=20, outlinewidth=2, tickfont=dict(family='Arial', size=30))
                                )
                            )

    edge_traces.append(color_trace)
    """
    for key, val in tri_dict.items():
        s = G.nodes[key[0]]['coords']
        t = G.nodes[key[1]]['coords']
        u = G.nodes[key[2]]['coords']
        e = tri_dict[key]
        #color = mpl.cm.seismic(e, bytes=True)
        #tri_c[key] = "rgba({},{},{},{})".format(color[0],color[1],color[2], color[3])
        edge_traces.append(go.Mesh3d(x=[s[0], t[0], u[0]], y=[s[1], t[1], u[1]], z=[s[2], t[2], u[2]], color='gray', opacity=.5))
    """

    fig = go.Figure(data=edge_traces,
                 layout=go.Layout(
                    showlegend=False, 
                    hovermode='closest',
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    #fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(width=700, height=700, scene = dict(
        xaxis=dict(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        showbackground=False,
        title='',
        ticks='',
        showticklabels=False
    ),
    yaxis=dict(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        showbackground=False,
        ticks='',
        title='',
        showticklabels=False
    ),
    zaxis=dict(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        showbackground=False,
        title='',
        ticks='',
        showticklabels=False
    )))

    xe, ye, ze = rotate_z(0, 0, 1.75, -.1)
    camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=xe, y=ye, z=ze)
    )

    fig.update_layout(scene_camera=camera, scene_dragmode='orbit')
    #fig.write_html("1bna_CA_forman_edge_2_5.html")
    fig.show()

############### Weighted Dirac Matrices ###################################

""" Normalise Colormap to unique range """ 
seismic_cmap = matplotlib.cm.get_cmap('Blues')

seismic_rgb = []
norm = mpl.colors.Normalize(vmin=0, vmax=255)

for i in range(0, 255):
    k = mpl.colors.colorConverter.to_rgb(seismic_cmap(norm(i)))
    seismic_rgb.append(k)

seismic = matplotlib_to_plotly(seismic_cmap, 255)

b_mats = []
b_mats.append([boundary_operator(simplices, 1).toarray(), boundary_operator(simplices, 2).toarray(), boundary_operator(simplices,3).toarray()])
size0, size1, size2 = np.shape(b_mats[-1][0]), np.shape(b_mats[-1][1]), np.shape(b_mats[-1][2])
wdmat0 = np.array([])
wdmat1 = np.array([])
wdmat2 = np.array([])
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
    
if size2[1] > 0:
    wdmat2 = np.bmat([[np.zeros((size0[0],size0[0])), np.matmul(np.linalg.inv(G[0].todense()), np.matmul(b_mats[-1][0], G[1].todense()))/math.sqrt(2), np.zeros((size0[0], size1[1])), np.zeros((size0[0], size2[1]))], [np.transpose(b_mats[-1][0])/math.sqrt(2), np.zeros((size1[0], size1[0])), np.matmul(np.linalg.inv(G[1].todense()), np.matmul(b_mats[-1][1], G[2].todense()))/math.sqrt(3), np.zeros((size1[0], size2[1]))], [np.zeros((size1[1], size0[0])), np.transpose(b_mats[-1][1])/math.sqrt(3), np.zeros((size1[1], size1[1])), np.matmul(np.linalg.inv(G[2].todense()), np.matmul(b_mats[-1][2], G[3].todense()))/2], [np.zeros((size2[1], size0[0])), np.zeros((size2[1], size1[0])), np.transpose(b_mats[-1][2])/2, np.zeros((size2[1],size2[1]))]])
elif size1[1] > 0:
    wdmat2 = np.bmat([[np.zeros((size0[0],size0[0])), np.matmul(np.linalg.inv(G[0].todense()), np.matmul(b_mats[-1][0], G[1].todense()))/math.sqrt(2), np.zeros((size0[0], size1[1]))], [np.transpose(b_mats[-1][0])/math.sqrt(2), np.zeros((size1[0], size1[0])), np.matmul(np.linalg.inv(G[1].todense()), np.matmul(b_mats[-1][1], G[2].todense()))/math.sqrt(3)], [np.zeros((size1[1], size0[0])), np.transpose(b_mats[-1][1])/math.sqrt(3), np.zeros((size1[1], size1[1]))]])
elif size0[1] > 0:
    wdmat2 = np.bmat([[np.zeros((size0[0],size0[0])), np.matmul(np.linalg.inv(G[0].todense()), np.matmul(b_mats[-1][0], G[1].todense()))/math.sqrt(2)], [np.transpose(b_mats[-1][0])/math.sqrt(2), np.zeros((size1[0], size1[0]))]])
else:
    wdmat2 = np.zeros((size0[0],size0[0]))

eigval, eigvec = np.linalg.eig(wdmat1)

#b_mats = []
#b_mats.append([boundary_operator(simplices, 1).toarray(), boundary_operator(simplices, 2).toarray(), boundary_operator(simplices,3).toarray()])
#size0, size1, size2 = np.shape(b_mats[-1][0]), np.shape(b_mats[-1][1]), np.shape(b_mats[-1][2])
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

if size2[1] > 0:
    dmat2 = np.bmat([[np.zeros((size0[0],size0[0])), b_mats[-1][0], np.zeros((size0[0], size1[1])), np.zeros((size0[0], size2[1]))], [np.transpose(b_mats[-1][0]), np.zeros((size1[0], size1[0])), b_mats[-1][1], np.zeros((size1[0], size2[1]))], [np.zeros((size1[1], size0[0])), np.transpose(b_mats[-1][1]), np.zeros((size1[1], size1[1])), b_mats[-1][2]], [np.zeros((size2[1], size0[0])), np.zeros((size2[1], size1[0])), np.transpose(b_mats[-1][2]), np.zeros((size2[1],size2[1]))]])
elif size1[1] > 0:
    dmat2 = np.bmat([[np.zeros((size0[0],size0[0])), b_mats[-1][0], np.zeros((size0[0], size1[1]))], [np.transpose(b_mats[-1][0]), np.zeros((size1[0], size1[0])), b_mats[-1][1]], [np.zeros((size1[1], size0[0])), np.transpose(b_mats[-1][1]), np.zeros((size1[1], size1[1]))]])
elif size0[1] > 0:
    dmat2 = np.bmat([[np.zeros((size0[0],size0[0])), b_mats[-1][0]], [np.transpose(b_mats[-1][0]), np.zeros((size1[0], size1[0]))]])
else:
    dmat2 = np.zeros((size0[0],size0[0]))

eigval, eigvec = np.linalg.eigh(dmat1)

for idx in [16, 17, 18]:
    G= gen_graph(list(n_faces(simplices, 1)), ptcloud, {}) # Get the Graph Network of Simplicial Complex
    s0 = len(list(n_faces(simplices,0)))
    s1 = len(list(n_faces(simplices,1)))
    #s2 = len(list(n_faces(simplices,2)))
    #s3 = len(list(n_faces(simplices,3)))
    node_dict = {list(n_faces(simplices,0))[i]: np.round(np.abs(eigvec[:,idx][i]),5) for i in range(len(list(n_faces(simplices,0))))}
    edge_dict = {list(n_faces(simplices,1))[i]: np.abs(eigvec[:,idx][s0+i]) for i in range(len(list(n_faces(simplices,1))))}
    #tri_dict = {list(n_faces(simplices,2))[i]: np.float(np.abs(eigvec[:,idx])[s0+s1+i]) for i in range(len(list(n_faces(simplices,2))))}
    #tetra_dict = {list(n_faces(simplices,3))[i]: np.float(np.abs(eigvec[:,idx])[s0+s1+s2+i]) for i in range(len(list(n_faces(simplices,3))))}
    
    """                   Plot Vertex Eigenvectors of D_p                       """
    norm = mpl.colors.Normalize(vmin=np.min(np.abs(eigvec[:,idx])), vmax=np.max(np.abs(eigvec[:,idx]))) # Modify the colour map range
    edge_c = dict()
    tri_c = dict()

    edge_traces = []

    node_x = []
    node_y = []
    node_z = []
    for node in G.nodes():
        x, y, z = G.nodes[node]['coords']
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)

    edge_traces.append(go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            colorscale=seismic, color = np.abs(eigvec[:, idx]),
            size=np.abs(eigvec[:, idx])*100+1, opacity=1, line=dict(color='black', width=30)
            ),line_width=20))

    for edge in G.edges():
        edge_x = []
        edge_y = []
        edge_z = []
        x0, y0, z0 = G.nodes[edge[0]]['coords']
        x1, y1, z1 = G.nodes[edge[1]]['coords']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        edge_z.append(z0)
        edge_z.append(z1)
        edge_z.append(None)

        e = edge_dict[edge]
        #e = ((e-np.mean(list(edge_dict.values())))/(np.std(list(edge_dict.values()))))
        #rint(norm(e))
        color = mpl.cm.Blues(norm(e)+0.1, bytes=True)
        edge_c[edge] = "rgba({},{},{},{})".format(color[0],color[1],color[2], color[3])

        edge_traces.append(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            line=dict(width=50*(norm(e))+5, color=edge_c[edge]),
            hoverinfo='none', opacity=1,
            mode='lines'))

    color_range = seismic_cmap
    color_trace = go.Scatter3d(x=node_x,y=node_y, z=node_z,
                             mode='markers',
                             marker=go.scatter3d.Marker(colorscale=seismic,
                                size=.01,
                                color=np.abs(eigvec[:, idx]), cmin=min(np.abs(eigvec[:,idx])), cmax=np.max(np.abs(eigvec[:,idx])),
                                showscale=True, colorbar=dict(thickness=20, outlinewidth=2, tickfont=dict(family='Arial', size=30))
                                )
                            )

    edge_traces.append(color_trace)

    #for key, val in tri_dict.items():
        #s = G.nodes[key[0]]['coords']
        #t = G.nodes[key[1]]['coords']
        #u = G.nodes[key[2]]['coords']
        #e = tri_dict[key]
        #color = mpl.cm.seismic(e, bytes=True)
        #tri_c[key] = "rgba({},{},{},{})".format(color[0],color[1],color[2], color[3])
        #edge_traces.append(go.Mesh3d(x=[s[0], t[0], u[0]], y=[s[1], t[1], u[1]], z=[s[2], t[2], u[2]], color='gray', opacity=.5))

    fig = go.Figure(data=edge_traces,
                 layout=go.Layout(
                    showlegend=False, 
                    hovermode='closest',
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    #fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(width=1000, height=700, scene = dict(
        xaxis=dict(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        showbackground=False,
        title='',
        ticks='',
        showticklabels=False
    ),
    yaxis=dict(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        showbackground=False,
        ticks='',
        title='',
        showticklabels=False
    ),
    zaxis=dict(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        showbackground=False,
        title='',
        ticks='',
        showticklabels=False
    )))

    xe, ye, ze = rotate_z(0, 0, 1.75, -.1)
    camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=xe, y=ye, z=ze)
    )

    fig.update_layout(scene_camera=camera, scene_dragmode='orbit')
    #fig.write_html("1bna_CA_forman_edge_2_5.html")
    fig.show()

for idx in [-4, -3, -2, -1]:
    for i in range(len(eigvec[:, idx])):
        if np.abs(eigvec[:, idx][i]) < 1e-6:
            eigvec[:, idx][i] = 0
    G= gen_graph(list(n_faces(simplices, 1)), ptcloud, {}) # Get the Graph Network of Simplicial Complex
    s0 = len(list(n_faces(simplices,0)))
    s1 = len(list(n_faces(simplices,1)))
    #s2 = len(list(n_faces(simplices,2)))
    #s3 = len(list(n_faces(simplices,3)))
    node_dict = {list(n_faces(simplices,0))[i]: np.round(np.abs(eigvec[:,idx][i].real), 5) for i in range(len(list(n_faces(simplices,0))))}
    edge_dict = {list(n_faces(simplices,1))[i]: np.abs(eigvec[:,idx][s0+i].real) for i in range(len(list(n_faces(simplices,1))))}
    #tri_dict = {list(n_faces(simplices,2))[i]: np.float(np.abs(eigvec[:,idx])[s0+s1+i]) for i in range(len(list(n_faces(simplices,2))))}
    #tetra_dict = {list(n_faces(simplices,3))[i]: np.float(np.abs(eigvec[:,idx])[s0+s1+s2+i]) for i in range(len(list(n_faces(simplices,3))))}
    
    """                   Plot Vertex Eigenvectors of D_p                       """
    norm = mpl.colors.Normalize(vmin=np.min(np.abs(eigvec[:,idx].real)), vmax=np.max(np.abs(eigvec[:,idx].real))) # Modify the colour map range
    edge_c = dict()
    tri_c = dict()

    edge_traces = []

    for edge in G.edges():
        edge_x = []
        edge_y = []
        edge_z = []
        x0, y0, z0 = G.nodes[edge[0]]['coords']
        x1, y1, z1 = G.nodes[edge[1]]['coords']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        edge_z.append(z0)
        edge_z.append(z1)
        edge_z.append(None)

        e = edge_dict[edge]
        #e = ((e-np.mean(list(edge_dict.values())))/(np.std(list(edge_dict.values()))))
        #rint(norm(e))
        color = mpl.cm.Blues(norm(e)+0.01, bytes=True)
        edge_c[edge] = "rgba({},{},{},{})".format(color[0],color[1],color[2], color[3])

        edge_traces.append(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            line=dict(width=20, color=edge_c[edge]),
            hoverinfo='none', opacity=1,
            mode='lines'))
        
    node_x = []
    node_y = []
    node_z = []
    for node in G.nodes():
        x, y, z = G.nodes[node]['coords']
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)

    edge_traces.append(go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            colorscale=seismic, color = 'gray',
            size=7, opacity=1, line=dict(color='black', width=20)
            ),line_width=20))

    color_range = seismic_cmap
    color_trace = go.Scatter3d(x=node_x,y=node_y, z=node_z,
                             mode='markers',
                             marker=go.scatter3d.Marker(colorscale=seismic,
                                size=.01,
                                color=eigvec[:, idx].real, cmin=np.min(np.abs(eigvec[:,idx].real)), cmax=np.max(np.abs(eigvec[:,idx].real)),
                                showscale=True, colorbar=dict(thickness=20, outlinewidth=2, tickfont=dict(family='Arial', size=30))
                                )
                            )

    edge_traces.append(color_trace)

    #for key, val in tri_dict.items():
        #s = G.nodes[key[0]]['coords']
        #t = G.nodes[key[1]]['coords']
        #u = G.nodes[key[2]]['coords']
        #e = tri_dict[key]
        #color = mpl.cm.seismic(e, bytes=True)
        #tri_c[key] = "rgba({},{},{},{})".format(color[0],color[1],color[2], color[3])
        #edge_traces.append(go.Mesh3d(x=[s[0], t[0], u[0]], y=[s[1], t[1], u[1]], z=[s[2], t[2], u[2]], color='gray', opacity=.5))

    fig = go.Figure(data=edge_traces,
                 layout=go.Layout(
                    showlegend=False, 
                    hovermode='closest',
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    #fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(width=700, height=700, scene = dict(
        xaxis=dict(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        showbackground=False,
        title='',
        ticks='',
        showticklabels=False
    ),
    yaxis=dict(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        showbackground=False,
        ticks='',
        title='',
        showticklabels=False
    ),
    zaxis=dict(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        showbackground=False,
        title='',
        ticks='',
        showticklabels=False
    )))

    xe, ye, ze = rotate_z(0, 0, 1.75, -.1)
    camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=xe, y=ye, z=ze)
    )

    fig.update_layout(scene_camera=camera, scene_dragmode='orbit')
    #fig.write_html("1bna_CA_forman_edge_2_5.html")
    fig.show()

""" Normalise Colormap to unique range """ 
seismic_cmap = matplotlib.cm.get_cmap('coolwarm')

seismic_rgb = []
norm = mpl.colors.Normalize(vmin=0, vmax=255)

for i in range(0, 255):
    k = mpl.colors.colorConverter.to_rgb(seismic_cmap(norm(i)))
    seismic_rgb.append(k)

seismic = matplotlib_to_plotly(seismic_cmap, 255)

for idx in [13, 14]:
    G= gen_graph(list(n_faces(simplices, 1)), ptcloud, {}) # Get the Graph Network of Simplicial Complex
    s0 = len(list(n_faces(simplices,0)))
    s1 = len(list(n_faces(simplices,1)))
    #s2 = len(list(n_faces(simplices,2)))
    #s3 = len(list(n_faces(simplices,3)))
    node_dict = {list(n_faces(simplices,0))[i]: np.round(np.abs(eigvec[:,idx][i]),5) for i in range(len(list(n_faces(simplices,0))))}
    edge_dict = {list(n_faces(simplices,1))[i]: np.abs(eigvec[:,idx][s0+i]) for i in range(len(list(n_faces(simplices,1))))}
    #tri_dict = {list(n_faces(simplices,2))[i]: np.float(np.abs(eigvec[:,idx])[s0+s1+i]) for i in range(len(list(n_faces(simplices,2))))}
    #tetra_dict = {list(n_faces(simplices,3))[i]: np.float(np.abs(eigvec[:,idx])[s0+s1+s2+i]) for i in range(len(list(n_faces(simplices,3))))}
    
    """                   Plot Vertex Eigenvectors of D_p                       """
    norm = mpl.colors.Normalize(vmin=np.min(np.abs(eigvec[:,idx])), vmax=np.max(np.abs(eigvec[:,idx]))) # Modify the colour map range
    edge_c = dict()
    tri_c = dict()

    edge_traces = []

    node_x = []
    node_y = []
    node_z = []
    for node in G.nodes():
        x, y, z = G.nodes[node]['coords']
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)

    edge_traces.append(go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            colorscale=seismic, color = np.abs(eigvec[:, idx]),
            size=np.abs(eigvec[:, idx])*100+1, opacity=1, line=dict(color='black', width=30)
            ),line_width=20))

    for edge in G.edges():
        edge_x = []
        edge_y = []
        edge_z = []
        x0, y0, z0 = G.nodes[edge[0]]['coords']
        x1, y1, z1 = G.nodes[edge[1]]['coords']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        edge_z.append(z0)
        edge_z.append(z1)
        edge_z.append(None)

        e = edge_dict[edge]
        #e = ((e-np.mean(list(edge_dict.values())))/(np.std(list(edge_dict.values()))))
        #rint(norm(e))
        color = mpl.cm.coolwarm(norm(e)+0.1, bytes=True)
        edge_c[edge] = "rgba({},{},{},{})".format(color[0],color[1],color[2], color[3])

        edge_traces.append(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            line=dict(width=50*(norm(e))+5, color=edge_c[edge]),
            hoverinfo='none', opacity=1,
            mode='lines'))

    color_range = seismic_cmap
    color_trace = go.Scatter3d(x=node_x,y=node_y, z=node_z,
                             mode='markers',
                             marker=go.scatter3d.Marker(colorscale=seismic,
                                size=.01,
                                color=np.abs(eigvec[:, idx]), cmin=min(np.abs(eigvec[:,idx])), cmax=np.max(np.abs(eigvec[:,idx])),
                                showscale=True, colorbar=dict(thickness=20, outlinewidth=2, tickfont=dict(family='Arial', size=30))
                                )
                            )

    edge_traces.append(color_trace)

    #for key, val in tri_dict.items():
        #s = G.nodes[key[0]]['coords']
        #t = G.nodes[key[1]]['coords']
        #u = G.nodes[key[2]]['coords']
        #e = tri_dict[key]
        #color = mpl.cm.seismic(e, bytes=True)
        #tri_c[key] = "rgba({},{},{},{})".format(color[0],color[1],color[2], color[3])
        #edge_traces.append(go.Mesh3d(x=[s[0], t[0], u[0]], y=[s[1], t[1], u[1]], z=[s[2], t[2], u[2]], color='gray', opacity=.5))

    fig = go.Figure(data=edge_traces,
                 layout=go.Layout(
                    showlegend=False, 
                    hovermode='closest',
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    #fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(width=1000, height=700, scene = dict(
        xaxis=dict(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        showbackground=False,
        title='',
        ticks='',
        showticklabels=False
    ),
    yaxis=dict(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        showbackground=False,
        ticks='',
        title='',
        showticklabels=False
    ),
    zaxis=dict(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        showbackground=False,
        title='',
        ticks='',
        showticklabels=False
    )))

    xe, ye, ze = rotate_z(0, 0, 1.75, -.1)
    camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=xe, y=ye, z=ze)
    )

    fig.update_layout(scene_camera=camera, scene_dragmode='orbit')
    #fig.write_html("1bna_CA_forman_edge_2_5.html")
    fig.show()

for idx in [19, 20, 21]:
    for i in range(len(eigvec[:, idx])):
        if np.abs(eigvec[:, idx][i]) < 1e-6:
            eigvec[:, idx][i] = 0
    G= gen_graph(list(n_faces(simplices, 1)), ptcloud, {}) # Get the Graph Network of Simplicial Complex
    s0 = len(list(n_faces(simplices,0)))
    s1 = len(list(n_faces(simplices,1)))
    #s2 = len(list(n_faces(simplices,2)))
    #s3 = len(list(n_faces(simplices,3)))
    node_dict = {list(n_faces(simplices,0))[i]: eigvec[:,idx][i] for i in range(len(list(n_faces(simplices,0))))}
    edge_dict = {list(n_faces(simplices,1))[i]: eigvec[:,idx][s0+i] for i in range(len(list(n_faces(simplices,1))))}
    #tri_dict = {list(n_faces(simplices,2))[i]: np.float(np.abs(eigvec[:,idx])[s0+s1+i]) for i in range(len(list(n_faces(simplices,2))))}
    #tetra_dict = {list(n_faces(simplices,3))[i]: np.float(np.abs(eigvec[:,idx])[s0+s1+s2+i]) for i in range(len(list(n_faces(simplices,3))))}
    
    """                   Plot Vertex Eigenvectors of D_p                       """
    norm = mpl.colors.Normalize(vmin=-2, vmax=2) # Modify the colour map range
    edge_c = dict()
    tri_c = dict()

    edge_traces = []

    for edge in G.edges():
        edge_x = []
        edge_y = []
        edge_z = []
        x0, y0, z0 = G.nodes[edge[0]]['coords']
        x1, y1, z1 = G.nodes[edge[1]]['coords']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        edge_z.append(z0)
        edge_z.append(z1)
        edge_z.append(None)

        e = edge_dict[edge]
        #e = ((e-np.mean(list(edge_dict.values())))/(np.std(list(edge_dict.values()))))
        #rint(norm(e))
        color = mpl.cm.coolwarm(np.sign(e), bytes=True)
        edge_c[edge] = "rgba({},{},{},{})".format(color[0],color[1],color[2], color[3])

        edge_traces.append(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            line=dict(width=100*(abs(e))+1, color=edge_c[edge]),
            hoverinfo='none', opacity=1,
            mode='lines'))
        
    node_x = []
    node_y = []
    node_z = []
    for node in G.nodes():
        x, y, z = G.nodes[node]['coords']
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)

    edge_traces.append(go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            colorscale=seismic, color = 'black',
            size=4, opacity=1, line=dict(color='black', width=20)
            ),line_width=20))

    color_range = seismic_cmap
    color_trace = go.Scatter3d(x=node_x,y=node_y, z=node_z,
                             mode='markers',
                             marker=go.scatter3d.Marker(colorscale=seismic,
                                size=.01,
                                color=eigvec[:, idx].real, cmid = 0,
                                showscale=True, colorbar=dict(thickness=20, outlinewidth=2, tickfont=dict(family='Arial', size=30))
                                )
                            )

    edge_traces.append(color_trace)

    #for key, val in tri_dict.items():
        #s = G.nodes[key[0]]['coords']
        #t = G.nodes[key[1]]['coords']
        #u = G.nodes[key[2]]['coords']
        #e = tri_dict[key]
        #color = mpl.cm.seismic(e, bytes=True)
        #tri_c[key] = "rgba({},{},{},{})".format(color[0],color[1],color[2], color[3])
        #edge_traces.append(go.Mesh3d(x=[s[0], t[0], u[0]], y=[s[1], t[1], u[1]], z=[s[2], t[2], u[2]], color='gray', opacity=.5))

    fig = go.Figure(data=edge_traces,
                 layout=go.Layout(
                    showlegend=False, 
                    hovermode='closest',
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    #fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(width=700, height=700, scene = dict(
        xaxis=dict(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        showbackground=False,
        title='',
        ticks='',
        showticklabels=False
    ),
    yaxis=dict(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        showbackground=False,
        ticks='',
        title='',
        showticklabels=False
    ),
    zaxis=dict(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        showbackground=False,
        title='',
        ticks='',
        showticklabels=False
    )))

    xe, ye, ze = rotate_z(0, 0, 1.75, -.1)
    camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=xe, y=ye, z=ze)
    )

    fig.update_layout(scene_camera=camera, scene_dragmode='orbit')
    #fig.write_html("1bna_CA_forman_edge_2_5.html")
    fig.show()