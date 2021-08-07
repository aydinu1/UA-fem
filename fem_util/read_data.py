def read_data(mat, msh_file, msh_plot):
    """Reads material and mesh data of the geometry. Inputs are:
        mat: 2x1 list including material IDs for RD iron, TD iron.
        msh_file: file path of gmsh file.
        msh_plot: 1 or 0 for plotting the mesh or not."""
        
    import numpy as np
    import pandas as pd
    from scipy import interpolate
    from read_msh import read_msh
    
    import sys
    if "/Materials" not in sys.path:
        sys.path.append("/Materials")
    
    #create data structure
    class structtype():
        pass
    data = structtype()
    
    class structtype():
        pass
    data.materials = structtype()

#%%
    # Functions for field strength and differential reluctivity
    data.materials.snuRD   = []
    data.materials.snuTD  = []
    for i in range(len(mat)):
        
        #material library includes only one iron material for now. to be implemented!!!
        mat_data = pd.read_table("Materials/"+str(mat[i])+".txt", sep="\s+", usecols=['B2', 'nu'])
        Bnu = np.zeros([len(mat_data),2])
        
        Bnu[:,0] = (mat_data.iloc[:,0])            
        Bnu[:,1] = (mat_data.iloc[:,1])    
        
        snu = interpolate.splrep(Bnu[:,0], Bnu[:,1], s=0) #snu is a spline that gives nu depending on B^2,   nu = snu(B^2)
           
        if i == 0:
            data.materials.snuRD.append(snu)
            
        else:
            data.materials.snuTD.append(snu)
            
    
    data.materials.snuRD = data.materials.snuRD[0]
    data.materials.snuTD = data.materials.snuTD[0]
#%% Read mesh
    data.Msh = read_msh(msh_file, msh_plot)
    
    return data