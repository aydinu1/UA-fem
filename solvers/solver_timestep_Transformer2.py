def solver_timestep_Transformer2(**kwargs):
    
    """ Non-linear time stepping solver for quasistatic electromagnetics
    problem. First order elements are utlized. Non-linearity is solved by
    Newton-Raphson Method.
    
    Inputs to be provided:
        data: includes mesh and material data
        inputs: includes simulation parameters
        flag_plot: optional input, to visualize the result or not (False by default)
    
    (c) 2020 Ugur Aydin
    
    """
    
#%% import necessary libraries and external functions    
    import numpy as np
    import numpy.matlib
    from scipy import interpolate
    import time
    import scipy.sparse
    from scipy.sparse.linalg import spsolve  #sparse matrix solver
    #from pypardiso import spsolve #could be faster sparse matrix solver (https://github.com/haasad/PyPardisoProject?fbclid=IwAR17tzVNhSSBlXY7WZKYn3LoGY7yo-jBCmbj7Lo9llpZ6N7qO_v1p5qmVPk)
        
    #Custom FEM functions here
    from gau import gau
    from basis2 import basis2
    from dirichlet_zero import dirichlet_zero
    from init_mat import init_mat
    from post_pro import post_pro
    

#%% Input handling

    while True:
        try:
            data = kwargs['data']
            break
        except:
            raise Exception('DATA structure has to be provided as input argument.')
                
    while True:
        try:       
            
            inputs = kwargs['inputs']; 
            
            u1_rms = inputs.ui
            ntime = inputs.ntime 
            periods = inputs.nper
            ntot = periods*ntime            
            f = inputs.f            
            R = inputs.R
            sigma = inputs.sigma
            d = inputs.dlam
            fillf = inputs.fillf;
            N1 = inputs.N1
            N2 = inputs.N2
            L = inputs.L
            break
        except:
            str_cond = [];
            str_cond.append('Supply information missing from DATA structure. Provide the following structure:\n')
            str_cond.append(' inputs.ui           %% Input voltage (V)\n')
            str_cond.append(' inputs.f            %% Supply frequency (Hz)\n')
            str_cond.append(' inputs.nper         %% Number of periods\n')
            str_cond.append(' inputs.ntime        %% Number of time steps per period\n')
            str_cond.append(' inputs.R            %% Primary winding resistance (ohm)\n')
            str_cond.append(' inputs.sigma        %% Core conductivity\n')
            str_cond.append(' inputs.fillf        %% Lamination stacking factor\n')
            str_cond.append(' inputs.N1           %% Number of turns of primary winding\n')
            str_cond.append(' inputs.N2           %% Number of turns of secondary winding\n')
            str_cond.append(' inputs.Nlam         %% Number of laminations\n')
            str_cond.append(' inputs.dlam         %% Single lamination thickness (m)\n')
            str_cond = ''.join(str_cond)
            print(str_cond)
            
            u1_rms = float(input('Input voltage (V): '))
            f = float(input('Supply frequency (Hz): '))
            periods = int(input('Number of periods: '))
            ntime = int(input('Number of time steps per period: '))     
            R = float(input('Primary winding resistance (ohm): '))
            sigma = float(input('Core conductivity: '))
            fillf= float(input('Lamination stacking factor: '))
            N1= float(input('Number of turns of primary winding: '))
            N2= float(input('Number of turns of secondary winding: '))
            Nlam =  int(input('Number of laminations: '))                          
            dlam = float(input('Single lamination thickness (m): '))                       
            L= Nlam*dlam/fillf;
            #raise Exception(str_cond);
            break 
        
    w = 2*np.pi*f
    ntot = periods*ntime
    t = np.linspace(0,periods/f,ntot+1)
    t = np.delete(t, np.size(t)-1, axis=0)
    dt = t[1]-t[0]     


    while True:
        try:
            flag_plot = kwargs["flag_plot"]
            break
        except:
            flag_plot = False #by default plotting results is set to 0
            break
    #%% Non-linear material spline functions

    nu_RD = lambda B2: interpolate.splev(B2, data.materials.snuRD, der=0)
    dnu_RD = lambda B2: interpolate.splev(B2, data.materials.snuRD, der=1) 

    nu_TD = lambda B2: interpolate.splev(B2, data.materials.snuTD, der=0)
    dnu_TD = lambda B2: interpolate.splev(B2, data.materials.snuTD, der=1)        
    #%% Some constants
    mu0 = np.longdouble(4*np.pi*1e-7);
    nu0 = np.longdouble(1/mu0);    
 
    #%% Gauss integration points
    if data.Msh.nodel == 3:
        ng = 1
        ngl = 1
        
    uvint,wint = gau('triangle', ng);
    wint = wint.ravel()*1.0
    
    #%%
    # Some matrices with zeros and ones    
    oe = 0.0*data.Msh.nop[:,0]         # Zeros, elements x 1
    oi = 0.0*wint.T                    # Zeros, integration points x 1
    ie = oe+1.0                        # Ones, elements x 1
    oei = oe*oi                        # Zeros, elements x integration points
    

#%%--------------------------------------------------------------------------
    #Initilize shape function and flux linkage and voltage coupling matrices   
    uvint, N, dNxy, dd, D1, D2, C1, C2 = init_mat(data,inputs)
    
#----------------------------------------------------------------------------    
    #%% Iron Elements RD and TD, and not iron elements, filling factors and stuff
    
    eta = 1.0*oei+1.0; #filling factor array,here we multiply to convert eta to float array so that
                       #we  can use fractional numbers in this array. similarly done below for sig
    sig = 1.0*oei+sigma   #conductivity array
    ife = (data.Msh.physs == data.Msh.IRON1) | (data.Msh.physs == data.Msh.IRON2)  # all iron elements
    ife_rd = (data.Msh.physs == data.Msh.IRON1)   #iron RD
    ife_td = (data.Msh.physs == data.Msh.IRON2)   #iron TD
    eta[ife] = fillf
    sig[ife] = sigma      
    
    #%% indices for residual and jacobian elements
    ir = (data.Msh.nop.T).ravel()
    
    ij = (data.Msh.nop.T).ravel()
    ij = ((np.matlib.repmat(ij,1,3)).T).ravel()
    
    jj = []
    for i in range(0,len(data.Msh.nop.T)):
        jj.append(data.Msh.nop[:,i]) 
      
    jj = np.matlib.repmat(jj,1,3)
    jj2 = (np.array(jj)).ravel()
    
    #%% Input
    U = np.sqrt(2)*u1_rms*np.sin(w*t);
    ramp = 0*t+1;
    ramp[0:ntime] = np.linspace(0,1,ntime);
    U = ramp*U;
    
    #%% Initilize
    
    a = np.zeros([data.Msh.maxnp,1]);
    I1 = np.array([0.0]);
    x = np.vstack((a,I1))
    
    #previous B
    Bp = [[],[]];
    Bp[0] = oei.copy();
    Bp[1] = oei.copy();
    
    # create B and H lists and list for differential reluctivity
    B = [[],[]];
    H = [[],[]];
    dHdB = [[],[],[],[]]; #dHxdBx, dHxdBy, dHydBx, dHydBy
    
    #A and B at each time step stored in this
    a_time = list();
    B_time = list();
    
    I1 = np.zeros([1,1]);
    I1_time = np.zeros([1,1]);
    
    flux_link = np.zeros([1,1]);
    U2 = np.zeros([1,1]);
    #%% Time stepping
    start_time0 = time.perf_counter()
    for ntime in range(0,ntot):
        start_time = time.perf_counter()
         
        # Vector potential and current from previous step
        ap = a.copy()  
        
        #Newton-Raphson iteration
        maxit = 10
        
        for it in range(0,maxit):
                    
            #Vector potential in all elements and all nodes         
            anop = np.reshape(a[data.Msh.nop],(data.Msh.maxel,data.Msh.nodel))
            
            #Flux density calculation
            B[0] = oei.copy();
            B[1] = oei.copy();
    
            for i in range(0,data.Msh.nodel):
                B[0] = B[0] + dNxy[1][:,i]*(anop[:,i]*(oi+1.0))/eta
                B[1] = B[1] - dNxy[0][:,i]*(anop[:,i]*(oi+1.0))/eta
            
    
            #Initialize everything
            H[0] = oei.copy();
            H[1] = oei.copy();
            dHdB[0] = oei.copy(); dHdB[1] = oei.copy();
            dHdB[2] = oei.copy(); dHdB[3] = oei.copy();
    
            
            #Non-magnetic materials
            H[0][~ife] = nu0*B[0][~ife]; 
            H[1][~ife] = nu0*B[1][~ife];
    
            
            dHdB[0][~ife] = nu0;
            dHdB[1][~ife] = 0.0;
            dHdB[2][~ife] = 0.0;
            dHdB[3][~ife] = nu0;
    
               
            # Field and dHdB for iron RD
            normB_fe = (np.power(B[0][ife_rd],2) + np.power(B[1][ife_rd],2));
            
            H[0][ife_rd] = nu_RD(normB_fe)*B[0][ife_rd]; 
            H[1][ife_rd] = nu_RD(normB_fe)*B[1][ife_rd]; 
            
            #Differential reluctivity, assume same reluctivity for RD and TD
            dHdB[0][ife_rd] = nu_RD(normB_fe) + 2*dnu_RD(normB_fe) * B[0][ife_rd]*B[0][ife_rd];
            dHdB[1][ife_rd] =                   2*dnu_RD(normB_fe) * B[0][ife_rd]*B[1][ife_rd];
            dHdB[2][ife_rd] =                   2*dnu_RD(normB_fe) * B[1][ife_rd]*B[0][ife_rd];
            dHdB[3][ife_rd] = nu_RD(normB_fe) + 2*dnu_RD(normB_fe) * B[1][ife_rd]*B[1][ife_rd];
            
            
            # Field and dHdB for iron TD
            normB_fe = (np.power(B[0][ife_td],2) + np.power(B[1][ife_td],2));
            
            H[0][ife_td] = nu_RD(normB_fe)*B[0][ife_td]; 
            H[1][ife_td] = nu_RD(normB_fe)*B[1][ife_td]; 
            
            #Differential reluctivity, assume same reluctivity for RD and TD
            dHdB[0][ife_td] = nu_TD(normB_fe) + 2*dnu_TD(normB_fe) * B[0][ife_td]*B[0][ife_td];
            dHdB[1][ife_td] =                   2*dnu_TD(normB_fe) * B[0][ife_td]*B[1][ife_td];
            dHdB[2][ife_td] =                   2*dnu_TD(normB_fe) * B[1][ife_td]*B[0][ife_td];
            dHdB[3][ife_td] = nu_TD(normB_fe) + 2*dnu_TD(normB_fe) * B[1][ife_td]*B[1][ife_td];
            
            
            #linear material------------------------------
            #H[0][ife] = (nu0/1500.0)*B[0][ife]; 
            #H[1][ife] = (nu0/1500.0)*B[1][ife]; 
            
            #dHdB[0][ife] = nu0/1500.0;
            #dHdB[1][ife] = 0.0;
            #dHdB[2][ife] = 0.0;
           # dHdB[3][ife] = nu0/1500.0;        
            
            #Integrate for the element residual
            JJ = [[],[],[]]
            JJ[0] = np.zeros((data.Msh.maxel,data.Msh.nodel))
            JJ[1] = np.zeros((data.Msh.maxel,data.Msh.nodel))
            JJ[2] = np.zeros((data.Msh.maxel,data.Msh.nodel))
            
            rr = [[]]
            rr[0] = np.zeros((data.Msh.maxel,data.Msh.nodel))
            
            
            for i in range(0, data.Msh.nodel):
                
                dNxi = dNxy[0][:,i];
                dNyi = dNxy[1][:,i];
                
                # Residuals
                rr[0][:,i] = ((0.5*dd.T*(-dNxi*H[1] + dNyi*H[0])*wint));
                
                for j in range(0,data.Msh.nodel):
                    
                    dNxj = dNxy[0][:,j];
                    dNyj = dNxy[1][:,j];
                    
                    JJ[i][:,j] = (0.5*dd.T*(-dNxi  *(dHdB[2]*dNyj - dHdB[3]*dNxj) + dNyi * (dHdB[0]*dNyj - dHdB[1]*dNxj))/eta*wint).ravel('F');
                    
            rr = (np.array(rr)).ravel('F')
            JJ = np.reshape(JJ,(data.Msh.maxel*3,3)).ravel('F')
            JJ = np.array(JJ)
            
    
            res = scipy.sparse.csc_matrix((rr, (ir, 0*ir))).toarray()
            jac = scipy.sparse.csc_matrix((JJ, (ij, jj2)))
            
    
            # Don't change Dirichlet nodes
            jac = dirichlet_zero(jac,data.Msh.indd);         
            res[data.Msh.indd] = 0.0;
            
            #%% Voltage equations       
            res[0:data.Msh.maxnp] = res[0:data.Msh.maxnp] - D1*I1; #current density = N*I/A, and res = dNi*H - Ni*J, where Ni*J is the source matrix, so D includes D = (N/A)*Ni
            
            #use backward euler for derivative
            res = np.append(res,R*I1 + np.matmul(C1,(a-ap)/dt) - U[ntime])
                       
            jac = scipy.sparse.hstack([jac,-D1]).tocsc()
            jac = scipy.sparse.vstack( [jac,np.reshape( np.append(+C1/dt,R).ravel(),(1,data.Msh.maxnp+1)) ] ).tocsc()
    
            #%% Solve
    
            dx = -spsolve(jac,res); #spsolve is efficient when RHS is full matrix
            
            x = (x + np.reshape(dx,(data.Msh.maxnp+1,1)) );
            r = np.linalg.norm(dx)/np.linalg.norm(a) #residual condition
            a = x[0:data.Msh.maxnp];       
            I1 = x[-1];
    
            if (r < 1e-5) or (np.linalg.norm(res) < 1e-6):
                break
            
        print('\n', ntime+1,'th step took ', it+1, 'iteration, ', time.perf_counter() - start_time, ' seconds')
    
        #store variables
        a_time.append(a)
        B_time.append(B)
        I1_time = np.append(I1_time,I1)
        
        flux_link = np.append(flux_link, np.array(np.matmul( C2,a ).ravel()))
        U2 = np.append(U2, np.array(np.matmul( C2, (a-ap)/dt ).ravel()) )
        
        ap = a.copy()
        Bp = B.copy()
    
     
    print('\n', 'Total elapsed time = ' , time.perf_counter() - start_time0, ' seconds')    
    
    I1_time = np.delete(I1_time,0,0);
    flux_link = np.delete(flux_link,0,0);
    U2 = np.delete(U2,0,0);
    
    #%% outputs
    
    # Simulation parameters
    class structtype():
        pass
    out = structtype()
    
    out.a = a_time
    out.B = B_time
    out.I1 = I1_time
    out.flux_link = flux_link
    out.U2 = U2
    out.U = U
    out.t = t
    #%% Post Processing
    if flag_plot == True:
        
        post_pro(data, out)
       
