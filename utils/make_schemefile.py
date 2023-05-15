"""

Script for simulating an MRI scheme file.

"""

def make_combs():
    """
    
    Finds all combinations of delta, Delta, gradient strength and gradient directions between set ranges,
    before they are eritten to a schemefile.
    
    """

    gx = []
    gy = []
    gz = []
    G = []
    D = []
    d = []
    TE = []

    for i in range(10,41,3):
        for j in range(10,41,3):
            for k in range(20,81,3):
                for g in range(3):
                    if i!=j and (j-i/3)>0.001 and j>i:
                        if g==0: gx.append(1);gy.append(0);gz.append(0)
                        elif g==1: gx.append(0);gy.append(1);gz.append(0)
                        elif g==2: gx.append(0);gy.append(0);gz.append(1)
                        d.append(round(i*1e-3,3));D.append(round(j*1e-3,3));G.append(round(k*1e-3,3));TE.append(0.08)

    with open('cropped'+'.scheme','w') as f:
        f.write('VERSION: 1'+"\n") 
        f.write(str(0)+" "+ str(0)+" "+str(0)+" "+str(0.00)+" "+str(D[0])+" "+str(d[0])+" "+str(TE[0])+"\n")
        for i in range(len(gx)):
            f.write(str(gx[i])+" "+ str(gy[i])+" "+str(gz[i])+" "+str(G[i])+" "+str(D[i])+" "+str(d[i])+" "+str(TE[i])+"\n")

make_combs()