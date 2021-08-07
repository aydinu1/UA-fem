def basis2(n,uv):
    # Shape function values and their derivatives for a given point in the
    # in the reference triangle with 3 or 6 nodes
    #
    # Inputs:   n number of nodes in the triangle
    #          uv coordinates of the pont (size 2x1)
    # Outputs:  N shape function values (size n x 1)
    #          dN shape function derivative values (size n x 2)
    
    import numpy as np

    # Coordinates in the reference element
    u = uv[0,:]
    v = uv[1,:]
    o = 0.0*u

    # 3-node triangle
    if n == 3:
        N = [[]]*3;
        dN = [[]]*2;
            
        N[0] = u;
        N[1] = v;
        N[2] = 1-u-v;
        dN[0] = [o+1.0, o, o-1.0];
        dN[1] = [o, o+1.0, o-1.0];

    #6-node triangle
    else:
        if n == 6:
            N = [[]]*6;
            dN = [[]]*2;
            w = 1-u-v;
            #   N       = [(2*u-1).*u ; (2*v-1).*v ; (2*w-1).*w ; 4*u.*v ; 4*v.*w   ; 4*w.*u  ];
            #   dN(:,1) = [4*u-1     ; 0         ; 1-4*w     ; 4*v   ; -4*v    ; 4*(w-u)];
            #   dN(:,2) = [0         ; 4*v-1     ; 1-4*w     ; 4*u   ; 4*(w-v) ; -4*u   ];

            dN[0] = [ 4*u-1   , 0.0       , 1-4*w , 4*w , -4*v    , 4*w-u ];
            dN[1] = [ 0.0       , 4*v-1   , 1-4*w , 4*u , 4*(w-v) , -4*u  ];
  
        else: 
            print(str(n), '-node shape functions not implemented for triangles')
    return np.array(N), np.array(dN)
  
