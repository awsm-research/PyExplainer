
# pyExplainer-replication-package
The replication package of pyExplainer paper.

## How to replicate results
1. Enter command `git clone -b replication-package https://github.com/awsm-research/pyExplainer.git` to clone GitHub repository.
2. Enter command `cd pyExplainer-replication-package`
3. Enter command `conda env create -f environment.yml` to create conda environment.
4. Enter command `python -m ipykernel install --user --name=pyExp_rep --display-name "pyExp_env"`to install the environment in Jupyter Notebook.
5. Open **train_global_model_and_explainer.ipynb** and run every cells in this notebook to train global model and local model.
	Note: The process of training local models takes around 1 hours or more, depending on the performance of CPU.
6. Open **evaluation.ipynb** and run every cells in this notebook to get result of each RQ.
