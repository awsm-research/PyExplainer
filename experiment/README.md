
# pyExplainer-reusable-package
The reusable package of pyExplainer paper.

## How to obtain the results presented in our paper
1. Enter command `git clone -b replication-package https://github.com/awsm-research/pyExplainer.git` to clone GitHub repository.
2. Enter command `cd pyExplainer/experiment` to go to directory that stores replication package of pyExplainer
3. Enter command `conda env create -f requirements.yml` to create conda environment.
4. Enter command `conda activate PyExplainer` to activate the created conda environment.
5. Enter command `python -m ipykernel install --user --name=PyExplainer --display-name "PyExplainer"`to install the environment in Jupyter Notebook.
6. Enter command `python train_global_model_and_explainer.py` to train global JIT defect models, PyExplainer and LIME

	Note: The process of training local models takes around 1 hours or more, depending on the performance of CPU.
	
6. Open **evaluation.ipynb** and run every cells in this notebook to get result of each RQ.
