# 2022FilamentClassifierNET
A neural network which is able to classify points in a filament-like point cloud.
This project has spawned from my Master of Science Thesis work: [Deep Learning based Coronary Artery Centerline tracking aimed at Fractional Flow Reserve Prediction from CCTA images](https://github.com/Matteoleccardi/2022MasterThesis)


## Setup, requirements and folders organisation
This project is composed of many self-contained sub-modules. It is important to execute every command starting inside the "2022FilamentClassifierNET" folder. Precise instructions on how to run this project on Windows and \*nix machines are attached in *HOWTORUN*.

The current *README* has the sole purpose of giving an overview of the project, to present ideas and briefly discuss results. Each relevant sub-module has a *README* with more in-depth infos. The folders structure is briefly illustrated here:
- **2022FilamentClassifierNET**: base directory (every other folder and module is a sub-directory). Everything should be executed by having this as the current working directory.
- **Dataset**: sub-module involved in the creation of the dataset of filaments.
- **NeuralNetwork**: sub-module involved in the definition of the Neural Network models, their training routines, and the models' trained parameters. Specifically, the following models were created and tested:
  - *DiscreteCNN*: CNN-based classifier;
  - ...
- **Papers**: a list of free-to-access papers (found freely online through a basic Google search) used as inspiration and reference for this project.

**NOTE**: this codebase is not to be intended as a flexible, extendable bundle, rather this is a repo of the developement code of this research project.

## Why a filament classifier network
...

## Dataset used
...

## Models
...

## Results
...