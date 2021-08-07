def post_pro(data, out):
    
    """ Postprocessiong the result.
    
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri
    import matplotlib as mpl
    
    from tri_intrp import tri_intrp
    from drawequipotentials2 import drawequipotentials2
    
    cmap = mpl.cm.jet
    cols = np.matlib.repmat(cmap(0.5),len(data.Msh.nop),1)
    
    B = out.B[-1] #take last time step
    t = out.t
    U = out.U
    U2 = out.U2
    I1_time = out.I1
    a = out.a[-1] #take last time step
    
    
    Bx = B[0]
    By = B[1]
    
    Bnorm = np.sqrt(np.power(Bx,2) + np.power(By,2))

    B_int = tri_intrp(data.Msh.cord.T,data.Msh.nop.T,Bnorm) #interpolate B from middle of triangle to the nodes  
    
    cordx = data.Msh.cordx
    cordy = data.Msh.cordy
    nop = data.Msh.nop
    physs = data.Msh.physs
    indd = data.Msh.indd
    maxnp = data.Msh.maxnp
    
    
    #Plotting
    #---------------------------------------
    #Voltage and current plots. constrained_layout=True adjusts the sizes of the figues so that labels dont overlap
    
    fig2, ax = plt.subplots(2,3, sharex=False, sharey=False, constrained_layout=True, figsize=(12,8))
    ax[0,0].plot(t,U)
    ax[0,0].set_xlabel('Time (s)')
    ax[0,0].set_ylabel('Primary voltage (V)')
    ax[0,0].grid()
    
    ax[0,1].plot(t,U2)
    ax[0,1].set_xlabel('Time (s)')
    ax[0,1].set_ylabel('Secondary voltage (V)')
    ax[0,1].grid()
    
    ax[0,2].plot(t,I1_time)
    ax[0,2].set_xlabel('Time (s)')
    ax[0,2].set_ylabel('Primary current (A)')
    ax[0,2].grid()
    
    
    ax[1,0].triplot(cordx, cordy, nop[np.where(physs==data.Msh.AIR)], lw = 1.0, color=cmap(100,100))
    ax[1,0].triplot(cordx, cordy, nop[np.where(physs==data.Msh.IRON1)], lw = 1.0, color=cmap(1,1))
    ax[1,0].triplot(cordx, cordy, nop[np.where(physs==data.Msh.IRON2)], lw = 1.0, color=cmap(1,1))
    
    ax[1,0].triplot(cordx, cordy, nop[np.where(physs==data.Msh.COIL1)], lw = 1.0, color=cmap(150,150))
    ax[1,0].triplot(cordx, cordy, nop[np.where(physs==data.Msh.COIL1_neg)], lw = 1.0, color=cmap(150,150))
    
    ax[1,0].triplot(cordx, cordy, nop[np.where(physs==data.Msh.COIL2)], lw = 1.0, color=cmap(200,200))
    ax[1,0].triplot(cordx, cordy, nop[np.where(physs==data.Msh.COIL2_neg)], lw = 1.0, color=cmap(200,200))
    ax[1,0].plot(cordx[indd], cordy[indd], 'bo');
    
    ax[1,0].axis('equal')
    ax[1,0].set_xlabel('X (m)')
    ax[1,0].set_ylabel('Y (m)')
    
    f25 = ax[1,1].tricontourf(cordx,cordy,np.reshape(B_int,[maxnp]), 100, cmap=cmap) 
    fig2.colorbar(f25, ax=ax[1,1])
    ax[1,1].set_title('Flux density norm (T)')
    ax[1,1].set_xlabel('X (m)')
    ax[1,1].set_ylabel('Y (m)')
    
    drawequipotentials2(a,data.Msh,10,ax[1,1])
    ax[1,1].axis('equal')
        
    
    f26 = ax[1,2].tricontourf(cordx,cordy,np.reshape(a,[maxnp]), 100, cmap=cmap) 
    fig2.colorbar(f26, ax=ax[1,2])
    ax[1,2].set_title('Vector Potential (Wb/m)')
    ax[1,2].set_xlabel('X (m)')
    ax[1,2].set_ylabel('Y (m)')
    
    drawequipotentials2(a,data.Msh,10,ax[1,2])
    ax[1,2].axis('equal')
