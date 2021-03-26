from sklearn import metrics

import abc
import sys

# Ensure compatibility with Python 2/3
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})


class DISExplainer(ABC):
    """
    DISExplainer is the base class for Directly Interpretable Supervised Explainers (DISE).
    Such explainers attempt to train explainable models directly from data.
    Examples include Boolean Decision Rules via Column Generation[#1]_, Generalized Linear Rule Models[#2]_,
    TED: Teaching AI to Explain its Decisions[#3]_, etc.
    References:
        .. [#1] Boolean Decision Rules via Column Generation, NIPS 2018.
        Dash, Gunluk, Wei. https://arxiv.org/abs/1805.09901
        .. [#2] Generalized Linear Rule Models, ICML 2019.
        Dennis Wei, Sanjeeb Dash, Tian Gao, Oktay Gunluk.
        https://arxiv.org/abs/1906.01761
        .. [#3] TED: Teaching AI to Explain its Decisions, AIES 2019.
        Michael Hind, Dennis Wei, Murray Campbell, Noel C. F. Codella, Amit Dhurandhar,
        Aleksandra Mojsilovic, Karthikeyan Natesan Ramamurthy, Kush R. Varshney.
        https://arxiv.org/abs/1811.04896
    """

    def __init__(self, *argv, **kwargs):
        """
        Initialize a DISExplainer object.
        ToDo: check common steps that need to be distilled here.
        """

    @abc.abstractmethod
    def set_params(self, *argv, **kwargs):
        """
        Set parameters for the explainer.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def explain(self, *argv, **kwargs):
        """
        Explain the model
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self, *argv, **kwargs):
        """
        Fit a model on data.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, *argv, **kwargs):
        """
        Prediction for a batch of inputs (numpy array)
        """
        raise NotImplementedError

class TED_CartesianExplainer(DISExplainer):
    """TED is an explainability framework that leverages domain-relevant explanations in
    the training dataset to predict both labels and explanations for new instances [#]_.
    This is an implementation of the simplest instantiation of TED, called the Cartesian Product.
    References:
        .. [#] `Michael Hind, Dennis Wei, Murray Campbell, Noel C. F. Codella,
           Amit Dhurandhar, Aleksandra Mojsilovic, Karthikeyan Natesan Ramamurthy,
           Kush R. Varshney, "TED: Teaching AI to Explain its Decisions,"
           AAAI /ACM Conference on Artificial Intelligence, Ethics,
           and Society (AIES-19), 2019.
           <https://doi.org/10.1145/3306618.3314273>`_
    """

    def __init__(self, model):
        """
        Args:
            model (sklearn.base.BaseEstimator): a binary estimator for classification,
                i.e., it implements fit and predict.
        """
        super(TED_CartesianExplainer, self).__init__()
        self.model = model
        self.NumE = -1                  # set to dummy value to ensure it is computed before used

    def set_params(self, *argv, **kwargs):
        """
        Set parameters for the explainer.
        """
        pass

    def fit(self, X, Y, E) :
        """Train a classifier based on features (X), labels (Y), and explanations (E)
        Args:
            X: list of features vectors
            Y: list of labels
            E: list of explanations
        """
        self.NumE = E.max() + 1            # create E's start at 0, so we have MaxE + 1 unique values
        YE = self._composeYE(Y, E)         # create new Labels from Y + E
        self.clf = self.model.fit(X, YE)   # train classifier

    def predict_explain(self, X) :
        """Use TED-enhanced classifier to predict label (Y) and explanation (E) for passed instance
        Args:
            X (list of ints): features
        Returns:
            tuple:
                * **Y** (`int`) -- predicted label {0,1}
                * **E** (`int`) -- predicted explanation [0..MaxE]
        """
        assert (self.NumE  > 0),"Internal Error: NumE not computed!"
        YE = self.clf.predict(X)
        Y, E = self._decomposeYE(YE)      # decompose YE label into its components
        return Y, E

    def explain(self, X):
        """Use TED-enhanced classifier to provide an explanation (E) for passed instance
        Args:
            X (list of ints) : features
        Returns:
            int: predicted explanation [0..MaxE]
        """
        Y, E = self.predict_explain(X)
        return E

    def predict(self, X):
        """ Use TED-enhanced classifier to provide an prediction (Y) for passed instance
        Args:
            X (list of ints) : features
        Returns:
           int: predicted label {0,1}
        """
        Y, E = self.predict_explain(X)
        return Y

    def score(self, X_test, Y_test, E_test) :
        """ Evaluate the accuracy (Y and E) of the TED-enhanced classifier using a test dataset
        Args:
            X_test (list of lists): list of feature vectors
            Y_test (list of int)  : list of labels {0, 1}
            E_test (list of ints) : list of explanations {0, ..., NumExplanations -1}
        Returns:
            tuple:
                * **YE_accuracy** -- the accuracy of predictions when the labels (Y) and explanations (E) are treated as a combined label
                * **Y_accuracy** -- the prediction accuracy for labels (Y)
                * **E_accuracy** -- the prediction accuracy of explanations (E)
        """
        YE_predict = self.clf.predict(X_test)
        YE_test = self._composeYE(Y_test, E_test)
        YE_accuracy = metrics.accuracy_score(YE_test, YE_predict)

        Y_predict, E_predict = self._decomposeYE(YE_predict)
        Y_accuracy = metrics.accuracy_score(Y_test, Y_predict)
        E_accuracy = metrics.accuracy_score(E_test, E_predict)

        return YE_accuracy, Y_accuracy, E_accuracy



##############################
#### Supporting functions ####
##############################
    def _composeYE(self, Y, E) :
        """ Create cartesian product of Y and E
        Args:
            Y : list of labels
            E : list of explanations
        Returns:
            list of labels : We map Y and E into new dense label space, where::
                Y = 0 ==> 0 .. NumE-1
                Y = 1 ==> NumE .. 2*NumE - 1
        """
        return Y * self.NumE + E

    def _decomposeYE(self, YE) :
        """ Decompose an array of YE's into component Y and E
        Args:
            YE (array of ints) : the combined YE values created by _createYE
        Returns:
            tuple:
                * **Y** (`array of ints`) -- the Y labels
                * **E** (`array of ints`) -- the E labels
        """
        Y = YE // self.NumE   # integer division, i.e., result is an int (floor of math result)
        E = YE % self.NumE
        return Y, E