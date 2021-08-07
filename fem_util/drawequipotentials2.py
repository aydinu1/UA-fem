def drawequipotentials2(a,msh,numberOfLines,ax):
        
    """ Draw equipotential lines on first order triangular finite elements.
    
    Code is rewritten in Python by utilizing SMEKlib (see below) by Ugur Aydin 
    
    https://github.com/AnttiLehikoinen/SMEKlib
    
    MIT License

    Copyright (c) 2013-2016 Antti Lehikoinen
    
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    
    """
    
    import numpy as np
    
    A = [[],[]]
    b = []
    
    N = numberOfLines;
    
    potentials = np.linspace(min(a),max(a), N+2 );
    potentials = potentials[1:(N+1)];   #might be N instead of N+1
    
    x = msh.cord[:,0]
    y = msh.cord[:,1]
    
    for kpot in range(0,N):
        
        pot = potentials[kpot];
        
            
        for elem in range(0,int(msh.maxel)):
#            check if the equipotential lines goes through this element at all
            elementNodes = np.array((msh.nop[elem]));
            
            if (np.max(a[elementNodes]) < pot) or (np.min(a[elementNodes]) > pot):
                #nope --> continue to the next element in list
                continue;
                
#           drawing the equipotential line in the reference element:
#           denoting nodal values
            a1 = a[elementNodes[0]]; a2 = a[elementNodes[1]]; a3 = a[elementNodes[2]];
            
#           finding the intersects of the eq. line and the ksi-axis, eta-axis
#           and the 1-ksi line
            ksiis = [(pot-a1)/(a2-a1), np.array(0), (pot-a3)/(a2-a3)];
            etas = [np.array(0), (pot-a1)/(a3-a1), 1-(pot-a3)/(a2-a3)];
            
            I = np.where( (np.array(ksiis)>=0) * (np.array(ksiis)<=1) * (np.array(etas)<=1) * (np.array(etas)>=0))
            
            ksiis_1 = np.array(ksiis)[I];
            etas_1 = np.array(etas)[I];
            

            
#            %calculating mapping from reference element to global element
            A[0] = [x[elementNodes[1]]-x[elementNodes[0]], x[elementNodes[2]]-x[elementNodes[0]] ];
            A[1] = [y[elementNodes[1]]-y[elementNodes[0]], y[elementNodes[2]]-y[elementNodes[0]] ];
            
            b = [ x[elementNodes[0]], y[elementNodes[0]] ];
            
#           calculating and plotting global equipotential lines
            aux = np.vstack((ksiis_1,etas_1));
            globalCoordinates = np.matmul(A,aux.astype(float)) + np.vstack((b,b)).T;
            
            
            ax.plot(globalCoordinates[0,:],globalCoordinates[1,:],'k',linewidth=0.5)

                 


