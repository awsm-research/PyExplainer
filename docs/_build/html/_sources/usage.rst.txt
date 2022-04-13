.. highlight:: python

=====
Tutorial
=====

Quick start
--------------

To use pyexplainer in a project, first we need to import the required library::

    from pyexplainer import pyexplainer_pyexplainer
   
We can obtain the default dataset and global model stored in a dictionary as below, which is necessary to initialise a PyExplainer object later ::
     
    default_data_and_model = pyexplainer_pyexplainer.get_dflt()
    
    """
    Important Note.
    The default dictionary has the following keys:
    1. X_train - Training features used to train the global model
    2. y_train - Training labels used to train the global model
    3. indep - The column names of features
    4. dep - The column name of labels
    5. blackbox_model - Trained global model (only support supervised classification models from sklearn)
    6. X_explain - One row of features (independent variables) to be explained
    7. y_explain - One row of label (dependent variable) to be explained
    8. full_ft_names - Full column names of features
    """
    
Initialise PyExplainer object with X_train, y_train, indep, dep, and blackbox_model.  Attention!!  PyExplainer only supports `sklearn's supervised classification model <https://scikit-learn.org/stable/supervised_learning.html>`_ ::
    
    # Note that full_ft_names is optional while other variables are necessary
    py_explainer = pyexplainer_pyexplainer.PyExplainer(X_train = default_data_and_model['X_train'],
                                                       y_train = default_data_and_model['y_train'],
                                                       indep = default_data_and_model['indep'],
                                                       dep = default_data_and_model['dep'],
                                                       blackbox_model = default_data_and_model['blackbox_model']
                                                       full_ft_names = default_data_and_model['full_ft_names'])

Prepare data using default data dictionary to trigger explain function later::

    # one row of features (independent variables) to be explained
    X_explain = default_data_and_model['X_explain']
    # one row of label (dependent variable) to be explained
    y_explain = default_data_and_model['y_explain']

Trigger explain function under PyExplainer object to get rules::

    created_rules = py_explainer.explain(X_explain=X_explain,
                                         y_explain=y_explain,
                                         search_function='crossoverinterpolation')

Visualise those rules using visualise function under PyExplainer object::

    py_explainer.visualise(created_rule_obj)
    
Full tutorial
--------------



Simply click `here <https://github.com/awsm-research/PyExplainer/blob/master/TUTORIAL.ipynb>`_ to access our full tutorial Jupyter Notebook, in which we introduce the full workflow from data preparation to explainable prediction ::
