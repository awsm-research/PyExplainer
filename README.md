# PyExplainer ![logo](img/logo_30x30.png)

[![codecov](https://codecov.io/gh/awsm-research/pyExplainer/branch/master/graph/badge.svg?token=3HQBAEXK21)](https://codecov.io/gh/awsm-research/pyExplainer)
[![Documentation Status](https://readthedocs.org/projects/pyexplainer/badge/?version=latest)](https://pyexplainer.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/pyexplainer.svg)](https://badge.fury.io/py/pyexplainer)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/pyexplainer/badges/version.svg)](https://anaconda.org/conda-forge/pyexplainer)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/awsm-research/pyExplainer.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/awsm-research/pyExplainer/context:python)
[![Language grade: JavaScript](https://img.shields.io/lgtm/grade/javascript/g/awsm-research/pyExplainer.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/awsm-research/pyExplainer/context:javascript)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/awsm-research/pyExplainer.git/HEAD)

  

# Abstract

  

Just-In-Time (JIT) defect prediction (i.e., an AI/ML model to predict defect-introducing commits) is proposed to help developers prioritize their limited Software Quality Assurance (SQA) resources on the most risky commits.
However, the explainability of JIT defect models remains largely unexplored (i.e., practitioners still do not know why a commit is predicted as defect-introducing).
Recently, LIME has been used to generate explanations for any AI/ML models.
However, the random perturbation approach used by LIME to generate synthetic neighbors is still suboptimal, i.e., generating synthetic neighbors that may not be similar to an instance to be explained, producing low accuracy of the local models, leading to inaccurate explanations for just-in-time defect models.
In this paper, we propose PyExplainer---i.e., a local rule-based model-agnostic technique for generating explanations (i.e., why a commit is predicted as defective) of JIT defect models.
Through a case study of two open-source software projects, we find that our PyExplainer produces (1) synthetic neighbous that are 41%-45% more similar to an instance to be explained; (2) 18%-38% more accurate local models; and (3) explanations that are 69\%-98\% more unique and 17%-54% more consistent with the actual characteristics of defect-introducing commits in the future than LIME (a state-of-the-art model-agnostic technique).
This could help practitioners focus on the most important aspects of the commits to mitigate the risk of being defect-introducing.
Thus, the contributions of this paper build an important step towards Explainable AI for Software Engineering, making software analytics more explainable and actionable.
Finally, we publish our PyExplainer as a Python package to support practitioners and researchers (http://github.com/awsm-research/PyExplainer).

  

![pipeline](img/pipeline.png)

![alt text](img/pyexplainer_snap_demo.gif)

  

## Table of Contents

  

* **[How to cite PyExplainer](#how-to-cite-pyexplainer)**

* **[Dependencies](#dependencies)**

* **[Installation](#installation)**

* **[Tutorial](#tutorial)**

* **[Replication Package](#replication-package)**

* **[Contributors](#contributors)**

* **[Documentation](#documentation)**

* **[License](#license)**

* **[Credits](#credits)**

  

## How to cite PyExplainer

  

Chanathip Pornprasit, Chakkrit Tantithamthavorn, Jirayus Jiarpakdee, Micheal Fu, Patanamon Thongtanunam, "PyExplainer: Explaining the Predictions ofJust-In-Time Defect Models", in Proceedings of the International Conference on Automated Software Engineering (ASE), 2021, To Appear.

  

## Dependencies

  

- python = "3.8"

- scikit-learn = "0.24.1"

- numpy = "1.20.1"

- scipy = "1.6.1"

- ipywidgets = "7.6.3"

- ipython = "7.21.0"

- pandas = "1.2.3"

- statsmodels = "0.12.2"

  

The list of dependencies is shown upder [pyproject.toml file](https://github.com/awsm-research/pyExplainer/blob/master/pyproject.toml), however the installer takes care of installing them for you.

  

## Installation

  

Installing pyexplainer is easily done using pip, simply run the following:

  

```bash

$ pip install pyexplainer

```

This will also install the necessary dependencies.

  

For more approaches to install, please click [here](https://pyexplainer.readthedocs.io/en/latest/installation.html)

  
  

## Tutorial

  

For information on how to use pyexplainer, refer to the official documentation:

- [Tutorial Video](https://www.youtube.com/watch?v=p6uff4iYtHo)

  

[![Tutorial](https://img.youtube.com/vi/p6uff4iYtHo/hqdefault.jpg)](https://www.youtube.com/watch?v=p6uff4iYtHo "Tutorial")

- [Quickstart Notebook](https://github.com/awsm-research/pyExplainer/blob/master/quickstart_guide/formal_quickstart.ipynb)

How to run this quickstart notebook

1. Run command `git clone https://github.com/awsm-research/pyExplainer.git` to clone PyExplainer repository.
3. Run command `conda env create --file environment.yml` to create conda environment
4. Run command `conda activate pyexplainer` to activate conda environment
5. Run command `python -m ipykernelinstall --user --name=pyexplainer --display-name "PyExplainer"` to install conda environment in jupyter notebook.  
6. Open **formal_quickstart.ipynb** in web browser.
7. In **formal_quickstart.ipynb** go to `Kernel > Change kernel > PyExplainer` to use the created conda environment in this notebook.
8. Run cells from step 1 to step 3. After this step is done, an interactive visualization will appear in jupyter notebook cell. You can change the input feature values of ML model at slide bar.
9. Run the cells in appendix section if you would like to get more detail about variable used to build PyExplainer

- [Official Documentation](https://pyexplainer.readthedocs.io/en/latest/)

  
  

## Replication Package

  

To repeat our experiment, you can go to the replication-package branch as follows:

```

> git checkout replication-package

> cd experiment

```

  

Then, please follow the instructions in the README.md file in the replication-package branch.

  
  

## Contributors

  

We welcome and recognize all contributions. You can see a list of current contributors in the [contributors tab](https://github.com/awsm-research/pyExplainer/graphs/contributors).

  

Please click [here](https://pyexplainer.readthedocs.io/en/latest/contributing.html) to gain more information about making a contribution to this project.

  

## Documentation

  

The official documentation is hosted on [Read the Docs](https://pyexplainer.readthedocs.io/en/latest/)

  

## License

  

MIT License, click [here](https://github.com/awsm-research/pyExplainer/blob/master/LICENSE) for more information.

  

### Credits

  

This package was created with Cookiecutter and the UBC-MDS/cookiecutter-ubc-mds project template, modified from the [pyOpenSci/cookiecutter-pyopensci](https://github.com/pyOpenSci/cookiecutter-pyopensci) project template and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage).
