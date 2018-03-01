import numpy as np
import seaborn as sns
import pandas as pd
from lib.data import data_iterator
from lib.encoder import VectorEncoder

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def get_model():
    """This method should simply return an instance of the model you want to fit TRAINING data to.
    """
    # This currently is a default, baseline model! You should insert all your code for creating an
    # awesome custom model here!

    return MLPClassifier((30,30,30,30,30,30), activation='logistic', alpha=0.1)


def get_fitted_model(train_data_tarfile):
    """This method should take in the name of a gzipped tarfile containing the TRAINING data you
    want to use to fit your model (e.g., 'train.dat.tgz').

    This method should return some kind of sklearn-esque model that has been fit to the TRAINING
    data passed into the method.
    """
    Xs, ys = [], []
    vect_encoder = VectorEncoder()

    # This block simply goes through the training data tarball and appends event instances to the
    # `Xs` list and appends the labels (or "conversions") for those event instances to the `ys`
    # list.
    print("Reading training data from {}...".format(train_data_tarfile))

    train_data_iterator = data_iterator(train_data_tarfile)
    for json_obj in train_data_iterator:
        ys.append(json_obj['event']['conversion'])
        Xs.append(vect_encoder.encode(json_obj)[0])

    print("Done reading training data!")

    model = get_model()
    num_features = len(Xs[0])
    # Reshape Xs to be an array of however many rows (denoted by -1) of size `num_features`.
    Xs = np.array(Xs).reshape(-1, num_features)


    print(SelectKBest(k=5).fit_transform(Xs,ys))


    #print("Showing correlation matrix")
    #df = pd.DataFrame(Xs)
    #corrs = df.corr().to_dense()
    #sns.heatmap(df.corr())
    #sns.plt.show()

    print("Printing 5 highest correlations")


    print("Fitting training data to model...")
    model.fit(Xs, ys)
    print("Done fitting training data!")

    return model


def fit_model_to_testing_data(model, test_data_tarfile):
    """Given a model fit with training data and a gzipped tarfile containing testing data (e.g.,
    'testdata.tgz'), return two lists: one with the IDs of the events whose conversion we are
    predicting, and one with the predictions for each of those events.
    """
    ids, preds = [], []
    test_data_iterator = data_iterator(test_data_tarfile)
    vect_encoder = VectorEncoder()

    for json_obj in test_data_iterator:
        encoded_json_obj = vect_encoder.encode(json_obj)[0]
        X = np.array([encoded_json_obj]).reshape(1, -1)
        preds.append(model.predict_proba(X)[:,1][0])
        ids.append(json_obj['event']['id'])

    return ids, preds
