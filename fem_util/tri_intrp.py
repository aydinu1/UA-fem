def tri_intrp(p,t,bmid):
    #interpolates B from middle of triangle to the nodes
    #convert triangle data to node data
    import numpy as np
    import scipy.sparse
    
    npoint=np.size(p,1)
    ntri=np.size(t,1)

    ii = t[0:3,:]
    jj = np.ones([3,1])*range(0,ntri)

    K = scipy.sparse.lil_matrix((npoint, ntri))
    K[ii,jj] = 1
    K.toarray()

    M = scipy.sparse.lil_matrix((npoint, npoint))
    M[range(0,npoint), range(0,npoint)] = 1./np.sum(K.T,0)

    bnode=M*K*(bmid.T)
    return bnode