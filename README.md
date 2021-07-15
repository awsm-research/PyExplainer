# PyExplainer ![logo](img/logo_30x30.png)

[![codecov](https://codecov.io/gh/awsm-research/pyExplainer/branch/master/graph/badge.svg?token=3HQBAEXK21)](https://codecov.io/gh/awsm-research/pyExplainer)
[![Documentation Status](https://readthedocs.org/projects/pyexplainer/badge/?version=latest)](https://pyexplainer.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/pyexplainer.svg)](https://badge.fury.io/py/pyexplainer)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/pyexplainer/badges/version.svg)](https://anaconda.org/conda-forge/pyexplainer)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/awsm-research/pyExplainer.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/awsm-research/pyExplainer/context:python)
[![Language grade: JavaScript](https://img.shields.io/lgtm/grade/javascript/g/awsm-research/pyExplainer.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/awsm-research/pyExplainer/context:javascript)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/awsm-research/pyExplainer.git/HEAD)

  
**PyExplainer** is a local rule-based model-agnostic technique for generating explanations (i.e., why a commit is predicted as defective) of Just-In-Time (JIT) defect prediction defect models.


Through a case study of two open-source software projects, we find that our PyExplainer produces (1) synthetic neighbous that are 41%-45% more similar to an instance to be explained; (2) 18%-38% more accurate local models; and (3) explanations that are 69\%-98\% more unique and 17%-54% more consistent with the actual characteristics of defect-introducing commits in the future than LIME (a state-of-the-art model-agnostic technique).

This work is published at the International Conference on Automated Software Engineering 2021 (ASE2021): "PyExplainer: Explaining the Predictions ofJust-In-Time Defect Models". (**Preprint**)

```bibtex
@inproceedings{PyExplainer,
 author = {Pornprasit, Chanathip and Tantithamthavorn, Chakkrit and Jiarpakdee, Jirayus and Fu, Micheal and Thongtanunam, Patanamon}, 
 title = {PyExplainer: Explaining the Predictions ofJust-In-Time Defect Models},
 booktitle = {Proceedings of th International Conference on Automated Software Engineering (ASE)},
 year = {2021},
 numpages = {12},
}
```

![alt text](img/pyexplainer_snap_demo.gif)


## Quick Start
You can try our PyExplainer directly without installation at [this online JupyterNotebook](https://mybinder.org/v2/gh/awsm-research/pyExplainer.git/HEAD)  (run **TUTORIAL.ipynb**). The tutorial video below demonstrates how to use our PyExplainer in this JupyterNotebook. 

In the  JupyterNotebook:
- Run cells from step 1 to step 3 to create an interactive visualization in jupyter notebook cell (like the example above). You can change the input feature values of ML model at slide bar.
- Run the cells in appendix section if you would like to get more detail about variable used to build PyExplainer

[![Tutorial](https://img.youtube.com/vi/p6uff4iYtHo/hqdefault.jpg)](https://www.youtube.com/watch?v=p6uff4iYtHo "Tutorial")

See the installation instructions below for how to run this JuputerNotebook locally or install PyExplainer Python Package.

## Table of Contents

* **[Installation](#installation)**
  * [Dependencies]()
  * [Install PyExplainer Python Package]()
  * [Use the quickstart JupyterNotebook locally]()
* **[Replication Package](#replication-package)**

* **[Contributions]()**

* **[Documentation](#documentation)**

* **[License](#license)**

* **[Credits](#credits)**

  

## Installation 
### Dependencies

```
- python = "3.8"
- scikit-learn = "0.24.1"
- numpy = "1.20.1"
- scipy = "1.6.1"
- ipywidgets = "7.6.3"
- ipython = "7.21.0"
- pandas = "1.2.3"
- statsmodels = "0.12.2"
```
  
### Install PyExplainer Python Package
Installing pyexplainer is easily done using pip, simply run the following command. This will also install the necessary dependencies.

```bash
pip install pyexplainer
```
See [this PyExplainer python package documentation](https://pyexplainer.readthedocs.io/en/latest/) for how to install our PyExplainer from fsource and its dependencies. 


### Use the quickstart Jupyter Notebook locally
 1. Clone this PyExplainer repository: `git clone https://github.com/awsm-research/pyExplainer.git`
 1. Create a new [conda evironment](https://www.anaconda.com/products/individual) named 'pyexplainer': `conda env create --file environment.yml`
 1. Activate the conda environment: `conda activate pyexplainer`
 4. Install the jupyter notebook into this conda environment: `python -m ipykernelinstall --user --name=pyexplainer --display-name "PyExplainer"`
 6. Open **TUTORIAL.ipynb** in web browser.
 7. In **TUTORIAL.ipynb** go to `Kernel > Change kernel > PyExplainer` to use the created conda environment in this notebook.


## Replication Package

To repeat our experiment, you can go to the replication-package branch as follows:

```

> git checkout replication-package

> cd experiment

```

Then, please follow the instructions in the README.md file in the replication-package branch.

  
  

## Contributions

We welcome and recognize all contributions. You can see a list of current contributors in the [contributors tab](https://github.com/awsm-research/pyExplainer/graphs/contributors).

 Please click [here](https://pyexplainer.readthedocs.io/en/latest/contributing.html) for more information about making a contribution to this project.

  

## Documentation

  

The official documentation is hosted on [Read the Docs](https://pyexplainer.readthedocs.io/en/latest/)

## License
MIT License, click [here](https://github.com/awsm-research/pyExplainer/blob/master/LICENSE) for more information.

  

## Credits
This package was created with Cookiecutter and the UBC-MDS/cookiecutter-ubc-mds project template, modified from the [pyOpenSci/cookiecutter-pyopensci](https://github.com/pyOpenSci/cookiecutter-pyopensci) project template and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage).
