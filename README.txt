The simulation_script.py simulates the snow particles and saves them in Fmap_MR (MR simulation) or Fmap and DFDt (ROM simulation).
These are already solved for but are too big to be uploaded in GitHub. The DNS dataset can be found at  https://turbulence.pha.jhu.edu/Transition_bl.aspx from the JHTDB
The RK4 and intergration schemes are inspired by the work of the ETH Zürich, their sckemes can be found at:
https://github.com/haller-group/TBarrier/tree/main

The plotting_data.py plots the already solved particle trajectories and velocities. 