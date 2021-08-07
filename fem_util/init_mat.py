def init_mat(data, inputs):
    
    import numpy as np
    import scipy
    
    from basis2 import basis2
    from gau import gau
    
    
    #%% Gauss integration points
    if data.Msh.nodel == 3:
        ng = 1
        ngl = 1
        
    uvint,wint = gau('triangle', ng);
    wint = wint.ravel()*1.0
    
    
    #%% Some matrices with zeros and ones    
    oe = 0.0*data.Msh.nop[:,0]         # Zeros, elements x 1
    oi = 0.0*wint.T                    # Zeros, integration points x 1
    ie = oe+1.0                        # Ones, elements x 1
    oei = oe*oi                        # Zeros, elements x integration points
    
    #%% Jacobian for each element/integration point
    xn = data.Msh.cordx[data.Msh.nop]
    yn = data.Msh.cordy[data.Msh.nop]
    Nuv,dNuv = basis2(data.Msh.nodel, uvint)
    
    J11 = np.matmul(xn,dNuv[0]) #matrix multiplication
    J21 = np.matmul(xn,dNuv[1])
    J12 = np.matmul(yn,dNuv[0])
    J22 = np.matmul(yn,dNuv[1])
    dd = ((J11*J22)-(J12*J21))
    
    
    #Partial derivatives of the shape functions for all the elements
    #dNdx and dNdy for each element nodes appended to lists
    
    #initilize lists
    N = list()
    dNxy_1 = list()
    dNxy_2 = list()
    for i in range(0,data.Msh.nodel):
        N.append( list() ) #different object reference each time
        dNxy_1.append( list() ) #similarly create v and w lists
        dNxy_2.append( list() )
     
    dNxy = [[],[]]     
    
    #and create
    for i in range(0,data.Msh.nodel):
        N[i] = ie*Nuv[i];
        dNxy_1[i].append((J22.T*(ie*dNuv[0][i,:]) - J12.T*(ie*dNuv[1][i,:]))/dd.T)
        dNxy_2[i].append((-J21.T*(ie*dNuv[0][i,:]) + J11.T*(ie*dNuv[1][i,:]))/dd.T)
    
    dNxy[0] = np.array((np.reshape(dNxy_1,[data.Msh.nodel,data.Msh.maxel])).T)
    dNxy[1] = np.array((np.reshape(dNxy_2,[data.Msh.nodel,data.Msh.maxel])).T)
    
#%% Current Source Matrices such that D = integral(N/S, dS), C = l*D. Then we will obtain the voltages as U = (C*dadt)
    L = inputs.L
    N1 = inputs.N1
    N2 = inputs.N2
    
    #primary and secondary coil indices. Seperate positive and negative sides
    indcu1 = np.where(data.Msh.physs == data.Msh.COIL1)
    indcu2 = np.where(data.Msh.physs == data.Msh.COIL2)
        
    indcu1_neg = np.where(data.Msh.physs == data.Msh.COIL1_neg);
    indcu2_neg = np.where(data.Msh.physs == data.Msh.COIL2_neg);
        
    indcu1_tot = np.where( (data.Msh.physs == data.Msh.COIL1) | (data.Msh.physs == data.Msh.COIL1_neg) );
    indcu2_tot = np.where( (data.Msh.physs == data.Msh.COIL2) | (data.Msh.physs == data.Msh.COIL2_neg) );
    
    #primary and secondary coil nodes
    nopcu1 = np.reshape(data.Msh.nop[indcu1_tot,:],[np.size(indcu1_tot),data.Msh.nodel]);
    nopcu2 = np.reshape(data.Msh.nop[indcu2_tot,:],[np.size(indcu2_tot),data.Msh.nodel]);
    
    # Matrix for flux linkage calcluation for each winding
    Acoil1 = np.sum(0.5*dd[indcu1_tot,:]*wint)/2;  #division by 2 is to take into account the area of a one side of the coil (for example positive). 
    Acoil2 = np.sum(0.5*dd[indcu1_tot,:]*wint)/2;
    
    #Flux linkage formulae for fem is lam = integral(J*A,dS)/I = integral(N*A/S, dS). And we approximate A as A=sum(a*Ni) (where Ni is shape functions)
    #See books:
    # https://books.google.fi/books?id=-tPgBwAAQBAJ&pg=PA89&lpg=PA89&dq=flux+linkage+calculating+with+finite+elements&source=bl&ots=PmmLZ28RwX&sig=9k0xxEJDiy2vNGU1-W1269uENFU&hl=tr&sa=X&ved=2ahUKEwjS-_Kl97fdAhVDiywKHQgJDrI4ChDoATAGegQIAxAB#v=onepage&q=flux%20linkage%20calculating%20with%20finite%20elements&f=false
    # https://zapdf.com/computation-of-the-flux-linkage-of-windings-from-magnetic-sc.html
    D1 = np.zeros( [np.size(indcu1_tot),data.Msh.nodel])
    D2 = np.zeros( [np.size(indcu2_tot),data.Msh.nodel])
    
    for i in range(0, data.Msh.nodel):
        d_aux1 =(0.5*N1/Acoil1*(dd[indcu1,:]*np.reshape(N[i][indcu1], [np.size(indcu1),1]))*wint)
        d_aux2 = (-0.5*N1/Acoil1*(dd[indcu1_neg,:]*np.reshape( N[i][indcu1_neg], [np.size(indcu1_neg),1] ))*wint)
        D1[:,i] = np.hstack((d_aux1.ravel(),d_aux2.ravel()))
        
        d_aux1 =(0.5*N2/Acoil2*(dd[indcu2,:]*np.reshape(N[i][indcu2], [np.size(indcu2),1]))*wint)
        d_aux2 = (-0.5*N2/Acoil2*(dd[indcu2_neg,:]*np.reshape(N[i][indcu2_neg], [np.size(indcu2_neg),1]))*wint)
        D2[:,i] = np.hstack((d_aux1.ravel(),d_aux2.ravel()))
    
    D1 = scipy.sparse.csc_matrix((D1.ravel(), (nopcu1.ravel(), 0*nopcu1.ravel())), shape=(data.Msh.maxnp, 1)).toarray()  #data.Msh.maxnp*1 sparse matrix
    D2 = scipy.sparse.csc_matrix((D2.ravel(), (nopcu2.ravel(), 0*nopcu2.ravel())), shape=(data.Msh.maxnp, 1)).toarray()  #data.Msh.maxnp*1 sparse matrix
      
    C1 = L*D1.T;   #Remember U = integral( N1*L/S * dAdt, dS) and dAdt = (a2-a1)/dt * Ni where Ni is shape functions
    C2 = L*D2.T;    



    return uvint, N, dNxy, dd, D1, D2, C1, C2
