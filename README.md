# PadeOpsIO Overview

PadeOpsIO is a tool for loading, visualizing, and analyzing data from [PadéOps](https://github.com/Howland-Lab/PadeOps), an LES and DNS box code. 

## Dependencies

Aside from standard python library packages, PadeOpsIO requires:

* `numpy`

* `scipy`

* `matplotlib`

* `jinja2`

* `xarray`


## Installation

Clone the repo from GitHub. This can be done with: 
```
pip install git+https://github.com/Howland-Lab/padeopsIO.git
```

Alternatively, for developers, clone the git repository locally and install it with editing enabled (requires `pip` 22+): 
```
# navigate to the installation directory
cd $install_dir

# clone the repo
git clone -b main https://github.com/Howland-Lab/padeopsIO.git

# then add with pip
pip install -e padeopsIO

```

The module can be imported into a Python script with: 
```
import padeopsIO as pio
```

## Usage

PadeOpsIO is used for visualization and analysis for output data from PadéOps. For more, see the [quick start](https://github.com/Howland-Lab/padeopsIO/blob/main/examples/padeopsIO_quickstart.ipynb). 

Data can be instanteous data: 

![image](https://user-images.githubusercontent.com/8905274/197601106-86fd32e4-52dc-4cf5-bcc3-d1bd664cdc08.png)

or time-averaged: 

![image](https://user-images.githubusercontent.com/8905274/197600994-47325c6d-89f3-4d09-9a44-1a0822fe81b5.png)
