def elarea(corde):
    import numpy as np
    
    cordx = corde[:,0]
    cordy = corde[:,1]
    aa = np.zeros((len(cordx),len(cordx)))

    aa[0,:] = cordx
    aa[1,:] = cordy
    aa[2,:] = [1,1,1]

    f = 0.5*np.linalg.det(aa)
    return f