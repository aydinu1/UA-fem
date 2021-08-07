"""
Example transformer model. 
    -Time-stepping solution. 
    -Non-linear material for rolling and transverse directions.
    -Sinusoidal voltage source input.
    -First order elements.
"""
#%% 
#Add path of solvers.
import sys
if "/solvers" not in sys.path:
    sys.path.append("solvers")
    
#Add path of utility. 
if "/fem_util" not in sys.path:
    sys.path.append("./fem_util")  
    
#Add path of mesh. 
if "/mesh" not in sys.path:
    sys.path.append("./mesh") 

#Add path of Materials. 
if "/Materials" not in sys.path:
    sys.path.append("./Materials")      
    
    
from read_data import read_data    
from solver_timestep_Transformer2 import solver_timestep_Transformer2

#%% Set and run simulation

#   Simulation parameters
class structtype():
    pass
inputs = structtype()

inputs.ui = 20;                               # Terminal voltage (V) 
inputs.f = 50;                                # Supply frequency (Hz)
inputs.nper = 2;                              #number of periods
inputs.ntime = 100;                           #number of time steps per period
inputs.N1 = 305;                              # Number of primary turns
inputs.N2 = 198;                              # Number of secondary turns
inputs.R = 4.941;                             # Primary winding resistance (ohm)
inputs.Nlam =  30;                            # Number of laminations
inputs.dlam = 0.5e-3;                         # Lamination thickness
inputs.fillf = 0.96                           #lamination stacking factor
inputs.sigma = 0;                             # lamination conductivity (use of this to be implemented...)
inputs.L = inputs.Nlam*inputs.dlam/inputs.fillf    # Transformer lamination thickness


#   get mesh and material data of the transformer
data = read_data([4,4], "transformer_1TD_ugur_fine.msh", msh_plot = 0)

#  Time-stepping solution
#  Provide function argumets as key-word arguments.
out_data = solver_timestep_Transformer2(data = data, inputs = inputs, flag_plot = True)
