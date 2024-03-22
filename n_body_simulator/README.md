## How to use

The position_generator.py script can be imported as a module and contains functions to define, simulate and analyse the motion of orbitals

In order to use the animation script, the save option on the position_update function should be set to True. The 2D motion of each orbital will then be saves as csv file. The animation script can then be called from the command line with '$python animate_orbits.py --filename (filename) --frames (number of frames) --output_filename (output_filename)' and will be saved as gif.

The Jupyter file serves as a demonstration of the simulator, mainly on the Earth-Moon-Sun system. This is detailed in full in the pdf report.
