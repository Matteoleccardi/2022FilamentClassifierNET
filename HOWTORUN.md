# Premises
This project is not a single program that can be just run right away. Each sub-module accomplish a certain task, and many are dependant on the results obtained from other, previously executed sub-modules.

# Installation guide
**1. Download this repo.** Probably you already know how to do this. If not, GitHub has many options to download source code, but I recommend the easiest way which is to download the [zipped folder](https://github.com/Matteoleccardi/2022FilamentClassifierNET/archive/refs/heads/master.zip) containing the entire repository. Once downloaded, you can extract the content and place the *2022FilamentClassifierNET-master* folder anywhere. Inside this folder you will find another *2022FilamentClassifierNET-master* folder, which you have to rename *exactly* to *2022FilamentClassifierNET*.
Once completed, the folders structures has to be:
- Sup-folder (presumably Downloads/)
  - 2022FilamentClassifierNET-master
    - 2022FilamentClassifierNET
      - Dataset
      - NeuralNetwork
      - ...

The base-folder of the whole project is *2022FilamentClassifierNET*: every script and python program has to be launched through the command line from here.
You can freely move the whole *2022FilamentClassifierNET* anywhere in your computer, however you must not rename it.

**2. Install Python.** The next step is to install python on your OS. Specifically, you need *Python 3.10.5*, which you can install from the [official website](https://www.python.org/) in the "Downloads" section and following their instructions. If your OS does not support this version of Python, you can try earlier versions however the stability of the code is not guaranteed.

NOTES:
- For Windows users, be sure while installing to add the python installation folder to PATH (there is an option in the installer) and to allow the keyword "py" to be used to run python from the terminal.
- For Linux users, you choose the hard way of life. Python 3.10.5 shpuld be compiled from source. Instructions can be found on the official Python website, however the latest version installable through packet manager (as of July 2022, apt on Debian/Ubuntu based distros has Python 3.9.5) should be fine, too. Python 3.9.5 was tested on a Linux system and it worked as expected.
- For Mac users, you choose the easy way of life. It is time you do some effort, so figure it out yourself.

Once Python has been installed, you should be able to open up a terminal and check that 'py --version' or 'python3 --version' or 'python3.10 --version' return *Python 3.10.5* (or the equivalent for the available installed Python).
For any issue check the online resources of python.org.

**3. Install required packages in a virtual environment.** The whole procedure can be found in the [python.org documentation](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/). Once you get to the tutorial section [Creating a virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment), be sure to be inside the *2022FilamentClassifierNET* folder while running the command. After creating the virtual environment, you should have:
- 2022FilamentClassifierNET
  - Dataset
  - NeuralNetwork
  - ...
  - env

Now, you have to activate the virtual environment. To do so, keep following the [linked tutorial](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#activating-a-virtual-environment). NOTE: after you finish playing around with the project, be sure to *deactivate* the virtual environment.

It is time to install the packages. Again, it should be as easy as to follow the instruction in the [linked tutorial](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#using-requirements-files). Refer to the requirements files *REQUIREMENTS_WIN.txt, REQUIREMENTS_MAC.txt, REQUIREMENTS_LINUX.txt*. All requirements include pytorch with CUDA, if your system supports it. It is possible that, for the installation to work, you will have to repeat the installation through '... -m pip install -r req_file.txt' a couple of times, untill all packages are installed without errors.

All packages should now be installed.

# Running the code from the beginning
.. still under dev ...

0. (Remember to activate the env)
1. run ./Dataset/DatasetCreator.py
2. run ./NeuralNetwork/DiscreteCNN/DiscreteCNN_data_maker.py
3. run ./NeuralNetwork/DiscreteCNN/DiscreteCNN_data_loader.py
4. run ./NeuralNetwork/DiscreteCNN/DiscreteCNN_trainer.py

# Running ready-made examples
There are not ready-made examples yet.
