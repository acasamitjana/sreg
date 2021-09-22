# SRegCD: Smooth Registration tool using Complete Digraph 

This repository performs linear and nonlinear registration between a set of points (e.g., timepoints in longitudinal studies) and a shared latent space. We use the log-space of transforms to infere the most probable deformations using Bayesian inference


### Requirements:
**Python** <br />
The code run on python v3.8.11 and several external libraries listed under requirements.txt

**NiftyReg package** <br />
Needed to run the algorithm using NiftyReg as base registration algorithm. 
http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg_documentation

### Run the code
- **Set-up configuration files** 
  - _setup.py_: create this file following the setup_example.py and according to your local machine and data directories. It contains the absolute paths to input data and the all the generated output registration paths. Here, at least two directories must exist and subject id and filenames should be consistent across them:
     - IMAGES_DIR: each subject is contained in a separated folder containting all (time)points. The folder name is the subject_id.
     - SEGMENTATION_DIR: each subject is contained in a separated folder containting all (time)points. The folder name is the subject_id.
 - **Run linear registration**
   - _scripts/run_linear_registration.sh_: this script will run over all subjects available in IMAGES_DIR. It also accepts a list of arguments (SUBJECT_ID) to run it over a subset (1, ..., N) subjects
 - **Run non-linear registration**
   - _scripts/run_nonlinear_registration.sh_: this script will run over all subjects available in ALGORITHM_DIR_LINEAL/images (subjects processed using the linear registration script). It also accepts a list of arguments (SUBJECT_ID) to run it over a subset (1, ..., N) subjects.


## Code updates

22 September 2021:
- Update to python 3.8.11 and scipy library to use open source linear programming packages.

21 September 2021:
- Initial commit




## Citation
TBC
