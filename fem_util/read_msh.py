def read_msh(file, flag_plot):
        
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    
    #custom functions
    from read_gmsh_V1 import read_gmsh
    from elarea import elarea
     
    #%% Create a structure for the output and assign 
    class structtype():
        pass
    f = structtype()
    
    #%%
    # Read Mesh
    msh = read_gmsh(file)
    
    #We have first order mesh
    itri = 1    #triangle element indices in the .msh class
    ilin = 0    #line element indices in the .msh class
    
    maxel = int(msh.maxel[itri])
    maxnp = int(msh.maxnp)
    nop = np.array(msh.nop[itri])-1 # -1 is due to index starting from 0 in python
    cord = msh.cord[:,0:2]
    cordx = cord[:,0]
    cordy = cord[:,1]
    physs = np.array(msh.phys_group[itri]) #triangular element material indices
    physsl = np.array(msh.phys_group[ilin]) #line element material indices
    nodel = (msh.nodel[itri])[itri] #number of nodes of an element
    nopl = np.array(msh.nop[ilin])-1  #line elements
    

    
    #material indices defined as in gmsh file
    f.AIR = 1000
    f.IRON1 = 1001
    f.IRON2 = 1004
    f.COIL1 = 1002
    f.COIL1_neg = 1012
    f.COIL2 = 1003
    f.COIL2_neg = 1013
    f.DIR = 2000
    f.DIR_disp = 2001
            
    #dirichlet nodes
    indd = np.unique(nopl[physsl == f.DIR,:])   
    #%% confirm counter-clockwise element numbering for triangles
    for i in range(0,maxel):
        corde = cord[nop[i,:],:]
        if elarea(corde)<0:
            nop[i,:] = nop[i,[1,0,2]]
    
    #%% Air-gap line
    noplb = nopl[np.where(physsl==3000)]
    maxlb = np.size(noplb,0)

    #%% Assign the output variables
    f.msh = msh
    f.maxel = maxel
    f.maxnp = maxnp
    f.nop = nop
    f.cord = cord
    f.cordx = cordx
    f.cordy = cordy
    f.physs = physs
    f.physsl = physsl
    f.nodel = nodel
    f.nopl = nopl
    f.indd = indd;
  
    #%% Plotting mesh
    if flag_plot:
        cmap = mpl.cm.jet
    
        plt.figure()
        plt.triplot(cordx, cordy, nop[np.where(physs==f.AIR)], lw = 1.0, color=cmap(100,100))
        plt.triplot(cordx, cordy, nop[np.where(physs==f.IRON1)], lw = 1.0, color=cmap(1,1))
        plt.triplot(cordx, cordy, nop[np.where(physs==f.IRON2)], lw = 1.0, color=cmap(1,1))
        #plt.triplot(cordx, cordy, nop[np.where(physs==IRON2)], lw = 1.0, color=cmap(130,302))
        
        plt.triplot(cordx, cordy, nop[np.where(physs==f.COIL1)], lw = 1.0, color=cmap(150,150))
        plt.triplot(cordx, cordy, nop[np.where(physs==f.COIL1_neg)], lw = 1.0, color=cmap(150,150))
        
        plt.triplot(cordx, cordy, nop[np.where(physs==f.COIL2)], lw = 1.0, color=cmap(200,200))
        plt.triplot(cordx, cordy, nop[np.where(physs==f.COIL2_neg)], lw = 1.0, color=cmap(200,200))
        
        #dirichlet nodes
        plt.plot(cordx[indd], cordy[indd], 'bo');
        
        plt.axis('equal')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.show()
        
    return f        