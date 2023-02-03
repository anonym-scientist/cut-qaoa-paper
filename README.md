# Cut-QAOA Experiments

This repository contains the prototypical implementation used to generate results presented in our publication *"Investigating the effect of circuit cutting in QAOA for the MaxCut problem on NISQ devices"* [1].

## Additional plots and data
The [plots](plots) folder contains additional plots to those published in the paper. 
In addition, the individual graphs and the parameters for each experiment run can be viewed here.

## Experiment data

Due to their large file sizes, the generated experiment data is uploaded separately:
 - The raw and partly preprocessed data can be downloaded [here](https://drive.google.com/file/d/1IGxJtSlZMBQZmbbGkSmspbS60-EMyGOa/view?usp=sharing). The unzipped data has about 100GB.
 - The calibration data of the quantum devices can be downloaded [here](https://drive.google.com/file/d/1fXnIaFCzig-xXkuVv8hfQcI5eM1OLkW8/view?usp=sharing)

To reproduce the plots of the paper, download the data, extract it in the project folder, and rerun the Jupyter notebook [notebooks/figures.ipynb](notebooks/figures.ipynb).


## Installation
Python version 3.9 was used for development. 
To avoid errors, it is recommended to use the same version.
To install the requirements and set the IBMQ token, use the following commands:

```bash
# if virtualenv is not installed
pip install virtualenv

# create new virtualenv called 'venv'
virtualenv venv

# activate virtualenv; in Windows systems activate might be in 'venv/Scripts'
source venv/bin/activate

# install application requirements.
pip install -r requirements.txt

# set the ibmq token from the bash
python -c "from qiskit import IBMQ
IBMQ.save_account('MY_API_TOKEN')"
```

## Code
The code is structured into the following directories:
 - the module [circuit_cutting](circuit_cutting) provides the circuit cutting functionality 
 - the module [qaoa](qaoa) contains the QAOA implementations
 - the module [optimization](optimization) enables an easy interaction with the Qiskit optimizers
 - the module [runtime_helpers](runtime_helpers) contains a collection of functions that simplify the creation of Qiskit Runtime programs and the interaction with them
 - the directory [runtime_programs](runtime_programs) contains the code for the runtime programs:
   - [runtime_with_imports.py](runtime_programs/runtime_with_imports.py): This program contains imports from the project. It is therefore clear and easy to understand. However, the IBM Runtime only allows the upload of one file and not the whole project.
   - [runtime_single_file.py](runtime_programs/runtime_single_file.py): This program does not contain imports. All previously imported functions were copied directly into the file. This program can be uploaded to IBM Runtime.

The entrance point for the first experiment is in [experiment_1.py](experiment_1.py), and for the second experiment in [experiment_2.py](experiment_2.py).
To start an experiment run with both experiments, use [experiment_complete.py](experiment_complete.py) as described below.
To get a description of the arguments, use the following:
```bash
python experiment_complete.py -h
```

## Generate new data

In order to generate and evaluate new problem instances, the following steps are necessary:
1. Make sure your IBMQ token is set.
2. Start [experiment_complete.py](experiment_complete.py) to start a new experiment run with a new graph that gets executed on an IBMQ backend.
```
python experiment_complete.py -b ibmq_ehningen -g 5 -n 10 --steps 20 --short-circuits
```
2. Postprocess the results with `load_data_from_executor` from [eval_param_maps.py](eval_param_maps.py)
3. Use `experiment_1_without_shot_memory.py` to compute the objective function of the graphs with a local simulator.
```
python experiment_1_without_shot_memory.py -b aer_simulator --steps 20 --graph-path <path to the graph of step 2.>
```
4. Postprocess the results with `load_data` from [eval_param_maps.py](eval_param_maps.py)
5. Create a new folder that contains the two generated folders with the results from steps 1. and 3.
6. Add this folder to the already existing data
7. Run `eval_all_dirs` from [experiment_complete_eval.py](experiment_complete_eval.py) to generate plots for the new data
8. Rerun [notebooks/figures.ipynb](notebooks/figures.ipynb) to add the new data to the existing plots


## References

[1] todo


## Disclaimer of Warranty

Unless required by applicable law or agreed to in writing, Licensor provides the Work (and each Contributor provides its Contributions) on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, including, without limitation, any warranties or conditions of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE. You are solely responsible for determining the appropriateness of using or redistributing the Work and assume any risks associated with Your exercise of permissions under this License.
