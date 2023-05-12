import numpy as np
def make_bval_bvec(scheme_path):

    gyro = 267.522

    with open(scheme_path, "r+") as file:
        # Reading from a file
        rows = file.readlines()[1:]
        x = []; y = []; z = [] 
        bvals = []
        for line in rows:
            data = line.split()
            x.append(int(data[0])); y.append(int(data[1])); z.append(int(data[2]))

            bval = (gyro)**2 * float(data[3])**2 * float(data[5])**2 * (float(data[4])-float(data[5])/3)*1000000
            bvals.append(bval)
        
        np.save('gx.npy',x)
        np.save('gy.npy',y)
        np.save('gz.npy',z)

path = "/Users/theavage/Documents/Master/Data/GS35/GS35.scheme"

def make_g_D_d(scheme_path):

    with open(scheme_path, "r+") as file:
        # Reading from a file
        rows = file.readlines()[1:]
        G = [];  D = []; d = []; TE = []
        for line in rows:
            data = line.split()
            G.append(float(data[3]));D.append(float(data[4]));d.append(float(data[5]));TE.append(float(data[6]))

 #   with open(filename+'.bval','w') as f:
 #       for bval in bvals:
 #           f.write(str(bval)+' ')
    with open('G'+'.txt','w') as f:
        for i in range(len(G)):
            f.write(str(G[i])+' ')

    with open('D'+'.txt','w') as f:
        for i in range(len(D)):
            f.write(str(D[i])+' ')

    with open('delta'+'.txt','w') as f:
        for i in range(len(d)):
            f.write(str(d[i])+' ')
    
    np.save('G.npy',G)
    np.save('D.npy',D)
    np.save('delta.npy',d)

make_g_D_d(path)