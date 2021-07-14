
# pyExplainer-replication-package
The replication package of pyExplainer paper.

## How to obtain the results presented in our paper
1. Enter command `conda env create -f requirements.yml` to create conda environment.
2. Enter command `conda deactivate` to change conda environment to `base` environment
3. Enter command `conda activate PyExplainer` to activate the created conda environment.
4. Enter command `python -m ipykernel install --user --name=PyExplainer --display-name "PyExplainer"`to install the environment in Jupyter Notebook.
5. Enter command `bash train_global_model_and_explainer.sh` to train global JIT defect models, PyExplainer and LIME

	Note: The process of training local models takes around 1 hour or more, depending on the performance of CPU.
	
6. Open **evaluation.ipynb** and run every cells in this notebook to get result of each RQ.
