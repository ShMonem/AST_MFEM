
# Auto-Skinning Transformations for Mixed Finite Elements Method (AST_MFEM)

Thank you for your interest in our code, and we hope you have fun trying it out! 😊
## Algorithm Features:
1. Mixed Finite Element Formulation:
- Our algorithm exposes additional rotation and deformation variables in addition to vertex positions. 
Assuming that rotations are purely provided by the artist *only* at a small set of handles/joints,
we formulate an optimization problem that solves for both vertex positions and symmetric part of the
deformation gradient. 
- Avoids specifying necessary constraints to force the simulation to track the rig motion, which can 
be overly constraining and produce undesirable artifacts.

2. Efficient Physics-Based Deformation:
- Ensures that the character deformations conform to the desired rig (user-defined) while avoiding 
performance bottlenecks typically associated with constrained physics simulations.
- Controllable skinning pipeline that can generate compelling character deformations, using a variety
of physics material models.

This code can be used to re-produces the results in the following paper(s):
- [Automatic Skinning using the Mixed Finite Element Method](https://arxiv.org/abs/2408.04066)

Developed by:
- [Dimitry Kachkovski]()
- [Hongcheng Song]()
- [Shaimaa Monem](https://orcid.org/0009-0008-4038-3452)
- [Abraham Kassahun Negash]()
- [David I.W. Levin](https://orcid.org/0000-0001-7079-1934)


Repository:
- https://github.com/ShMonem/AST_MFEM

License:
- Apache-2.0 see LICENSE.md.

Copyright holders:
- Developers and contributors.

## Dependencies
- Use your favorite way to install dependencies from `environment.yml`, for example:
````commandline
# Create a Conda environment from the .yml file
conda env create -f environment.yml
````
- Activate the virtual environment:
````commandline
conda activate mfem
````
- We use routines from [Bartels](https://github.com/dilevin/Bartels), and we include its
pre-compiled binaries in ``\python\bartelsBin\``
- 
## Code Structure
We highlight the most important parts of the structure for the user.

| Item       | sub-directories            | data/code                       |
|:-----------|:---------------------------|:--------------------------------|
| `data`     | ``<different_charecters>`` | ``*.mesh``                      |
|            |                            | ``*eulers.mat``                 |
|            |                            | ``*handles.mat``                |
|            |                            | ``*hierarchy.mat``              |
|            |                            | ``*P.mat``                      |
|            |                            | ``*PI.mat``                     |
| `env`      |                            | ``environment.yml``             |
| ``python`` | ``ast_fem_np``             | ``config.py``                   |
|            |                            | ``mfem_solver_np.py``           |
|            |                            | ``<all_solver-functions>``      |
|            |                            | ``<all_skinning-functions>.py`` |
|            | ``ui``                     | ``main.py``                     |


1. In ``\data`` different meshes characters along with their skinning data are stored.
2. ``\env\environment.yml`` contains the python dependencies list to install.
3. ``\python\ast_fem_np`` is where the user can find the implementations of our solver 
along with all *numpy* based facilitating functions.
4. Simulations can be run from ``\ui\main.py``. With minimal efforts, you can run different
examples from ``main.py`` by modifying the ``obj_name`` as below:
````python
import numpy as np
...

obj_name = 'human'
skel_anim = np.load(f'../../data/{obj_name}/{obj_name}_skel_anim.npy')
````