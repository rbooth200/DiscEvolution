# DiscEvolution
A python code to model the evolution of dust and gas in protoplanetary discs.

Author: Richard Booth

This code contains a set of modules to simulate the viscous evolution of protoplanetary discs over Myr
time scales. This includes modules to compute the viscous evolution, grain growth, radial drift,
thermal structure, transport of volatiles and planet formation. 

An example of how to run the code can be found in the example directory.

The code was originally described in [Booth et al. (2017)](https://arxiv.org/abs/1705.03305). If you
use this code in your research, please cite this paper along with any supporting papers describing
additional modules that you use. 


## License
Copyright 2017-2019 Richard Booth

This code is free software made available under the GNU GPLv3 License. For details see the LICENSE
file.


## Docs

The code is not intended to be used as a stand alone tool, but includes a set of different modules 
that can be used to include different physics when modelling the evolution of protoplanetary discs.
As such, it contains a series of classes each of which handle one aspect of the disc's evolution, 
such as viscous evolution or grain growth. An example model, along with a script to drive the main
code is included in the "example" directory (alternative scripts that work with the photoevaporation modules are given in the "control_scripts" directory).

In addition to the physics, there are a number of core classes upon which the physics modules are built.

### Core classes
These core classes represent the essential components needed for modelling an accretion disc

#### Star
Central to any protplanetary disc is a star, or perhaps binary star. The star classes handle the mass, radius and effective stellar temperature, which may evolve with time.

#### Grid
A wrapper around the radial grid structure used in the code, which specifies key properties such as the cell edges, centres and sizes.

#### Equation of State
The equation of state (eos) module contains classes that handle the disc's temperature structure. They can be used to find the disc's temperature, pressure scale-height, sound speed, viscosity etc. The code currently includes two equations of state:

- LocallyIsothermalEOS: A simple power law equation of state in which the sound speed is set by a power-law that does not evolve.
- IrradiatedEOS: Sets the temperature by computing the local balance between heating by stellar radiation and viscous stress along with cooling, following Hueso & Guillot (2005). The cooling rates are require computing the optical depth, currently the Bell & Lin (1994) type opacity of Zhu, Nelson & Gammie (2012) (which assumes a constant dust-to-gas ratio) or dust opacity of Tazzari et al. (2016) is used 

#### Disc
The disc class is a wrapper around the intrinsic properties of the disc, such as the surface density, pressure and viscosity. It also holds onto a copy of the equation of state and star, which may be used by the physics modules. A gas only disc is provided in the disc.py source file. The source file dust.py contains two extensions to the disc class, which handle the dust properties and optionally grain growth.

### Physics modules

#### Viscous evolution
Solves for the evolution of the surface density driven by the disc's viscosity

#### Single Fluid Drift
Updates the dust fraction according to radial drift (and optionally diffusion).

#### Diffusion
Solves the for the evolution of the concentration of trace species via diffusion.

#### Chemistry
Contains simple modules for following the evolution of volatile species such as carbon, nitrogen and oxygen. There is now a wrapper to the KROME chemistry package that can be used to integrate a chemical kinetics scheme.

#### Planet formation
Contains prescriptions for the growth and migration of planets. Currently does not include their feedback on the disc's evolution.

#### External Photoevaporation
Contains prescriptions for external photoevaporation, either with constant mass loss rate, or determined from the Haworth et al. (2017) FRIED grid. This includes the removal of dust that is entrained in the wind.

#### Internal Photoevaporaion
Contains prescriptions for internal photoevaporation, both EUV-driven (Alexander & Armitage 2007) and X-ray-driven (Owen et al. 2012 or Picogna et al. 2019). The module switches to the direct field prescription for a inner hole source once the column density to the hole drops sufficiently. Dust is not removed by these prescriptions.

## Contact
Please report any bugs or issues here on github. 

If you require support using the code, feel free to contact me at richardabooth@gmail.com. If extensive support is required, co-authorship of any subsequent publication will be sought. 

