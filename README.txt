The simulation_script.py simulates the snow particles and saves them in Fmap_MR (MR simulation) or Fmap and DFDt (ROM simulation).
These are already solved for, but are too big to be uploaded to GitHub. The DNS dataset can be found at  https://turbulence.pha.jhu.edu/Transition_bl.aspx from the JHTDB, this was also too large to include to GitHub.
The RK4 and integration schemes are inspired by the work of the ETH Zürich, their schemes can be found at:
https://github.com/haller-group/TBarrier/tree/main

The plotting_data.py plots the already solved particle trajectories and velocities. 
