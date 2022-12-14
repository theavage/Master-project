import numpy as np
from dmipy.core.acquisition_scheme import acquisition_scheme_from_gradient_strengths
def make_combs():

    gx = []
    gy = []
    gz = []
    G = []
    D = []
    d = []
    TE = []

    for i in range(10,41):
        for j in range(10,41):
            for k in range(20,81):
                for g in range(3):
                    if i!=j and (j-i/3)>0.001 and j>i:
                        if g==0: gx.append(1);gy.append(0);gz.append(0)
                        elif g==1: gx.append(0);gy.append(1);gz.append(0)
                        elif g==2: gx.append(0);gy.append(0);gz.append(1)
                        d.append(round(i*1e-3,3));D.append(round(j*1e-3,3));G.append(round(k*1e-3,3));TE.append(0.08)

    with open('new'+'.scheme','w') as f:
        f.write('VERSION: 1'+"\n") 
        for i in range(len(gx)):
            f.write(str(gx[i])+" "+ str(gy[i])+" "+str(gz[i])+" "+str(G[i])+" "+str(D[i])+" "+str(d[i])+" "+str(TE[i])+"\n")

make_combs()