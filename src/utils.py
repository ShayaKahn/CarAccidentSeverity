import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, classification_report, f1_score,
                             precision_score, recall_score, accuracy_score)
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from sklearn.utils import shuffle
import tensorflow as tf

def find_outliers_IQR(df):
    """
    Find outliers using the Interquartile Range (IQR) method.
    Inputs:
    df: a pandas DataFrame
    Returns:
    outliers: a pandas DataFrame containing the outliers
    """
    q1=df.quantile(0.25)
    q3=df.quantile(0.75)
    IQR=q3-q1
    outliers = df[((df<(q1-1.5*IQR)) | (df>(q3+1.5*IQR)))]
    return outliers


def CM(y_test, y_pred):
    """
    Plot the confusion matrix.
    y_test: the true labels (not one-hot encoded)
    y_pred: the predicted labels (not one-hot encoded)
    """
    # Get the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    classes = ["1", "2", "3", "4"]

    # Plot the confusion matrix
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap='binary', interpolation='None')
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    # Loop over data dimensions and create text annotations.
    for i in range(len(classes)):
        for j in range(len(classes)):
            text = ax.text(j, i, cm[i, j], ha="center",
                           va="center", color="red")

    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Confusion Matrix - Decision Tree')
    plt.colorbar(im)
    plt.show()


def get_metrics(y_test, y_pred):
    """
    Get the metrics
    y_test: the true labels (not one-hot encoded)
    y_pred: the predicted labels (not one-hot encoded)
    Return:
    metrics: a dictionary containing the metrics
    """
    metrics = {}
    precision = precision_score(y_test, y_pred, average=None)
    precision_weighted = precision_score(y_test, y_pred, average='weighted')
    precision_macro = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average=None)
    recall_weighted = recall_score(y_test, y_pred, average='weighted')
    recall_macro = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average=None)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')

    metrics['Precision'] = precision
    metrics['Weighted precision'] = precision_weighted
    metrics['Macro precision'] = precision_macro
    metrics['Recall'] = recall
    metrics['Weighted recall'] = recall_weighted
    metrics['Macro recall'] = recall_macro
    metrics['F1'] = f1
    metrics['Weighted F1'] = f1_weighted
    metrics['Macro F1'] = f1_macro

    return metrics


def plot_feature_importance(X_train, clf, n):
    """
    Plot the feature importance
    X_train: DataFrame, the training data
    clf: the trained classifier
    n: int, the number of top important features to show
    """
    # Show the top n important features
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_n = n

    # Plot the feature importances with corresponding feature names
    plt.barh(range(len(indices[:top_n])),
             importances[indices[:top_n]], align='center')
    plt.yticks(range(len(indices[:top_n])),
               [X_train.columns[i] for i in indices[:top_n]])
    plt.xlabel('Relative Importance')
    plt.title('Top {} important features'.format(top_n))
    plt.show()


def feature_selection(X_train, y_train):
    """
    Perform feature selection using RandomForestClassifier.
    X_train: DataFrame, the training data
    y_train: Series, the training labels
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    return importances, indices

def select_top(X_train, X_test, indices, n):
    """
    Select the top n important features.
    X_train: DataFrame, the training data
    X_test: DataFrame, the test data
    indices: array, the indices of the important features
    n: int, the number of top important features to select
    Return:
    X_train_top: DataFrame, the training data with only the top n important features
    X_test_top: DataFrame, the test data with only the top n important features
    """
    # select only the top n important features
    X_train_top = X_train.iloc[:, indices[:n]]
    X_test_top = X_test.iloc[:, indices[:n]]
    return X_train_top, X_test_top

def under_sample(X_train, y_train, N=100000, target_value=2):
    """
    Perform under-sampling by randomly sampling N samples from the target with specific value group
    X_train: DataFrame, the training data
    y_train: Series, the training labels
    N: int, the number of samples to sample from the target group
    target_value: int, the target value to sample from
    Return:
    X_train_sampled: DataFrame, the under-sampled training data
    y_train_sampled: Series, the under-sampled training labels
    """

    # Filter samples where y_train equals the target value
    X_target = X_train[y_train == target_value]
    y_target = y_train[y_train == target_value]

    # Randomly sample N rows from the target group
    X_target_sampled = X_target.sample(n=N, random_state=42)
    y_target_sampled = y_target.loc[X_target_sampled.index]

    # Get all other samples where y_train != target_value
    X_other = X_train[y_train != target_value]
    y_other = y_train[y_train != target_value]

    # Combine the sampled target group with the other samples
    X_train_sampled = pd.concat([X_target_sampled, X_other])
    y_train_sampled = pd.concat([y_target_sampled, y_other])

    # Shuffle both dataframes correspondingly
    X_train_sampled, y_train_sampled = shuffle(X_train_sampled,
                                               y_train_sampled,
                                               random_state=42)
    return X_train_sampled, y_train_sampled


def to_categorical(y):
    """
    Convert the labels to one-hot encoding.
    y: Series, the labels
    Return:
    num_classes: int, the number of classes
    y_oh: array, the one-hot encoded labels
    """
    y_corrected = y - 1
    num_classes = len(np.unique(y_corrected))
    y_oh = tf.keras.utils.to_categorical(y_corrected, num_classes)
    return num_classes, y_oh

def correct_target(y):
    """
    Convert the labels start from 0 instead of 1.
    y: Series, the labels
    Return:
    num_classes: int, the number of classes
    y_corrected: array, the corrected labels
    """
    y_corrected = y - 1
    num_classes = len(np.unique(y_corrected))
    return num_classes, y_corrected

def random_search_ANN(model, X_train, y_train_oh, num_classes):
    """
    Perform a RandomizedSearchCV to find the best hyperparameters for
    model: the ANN model
    X_train: the training data
    y_train_oh: the one-hot encoded training labels
    num_classes: the number of classes
    Return:
    best_hyper: the best hyperparameters
    """
    param_dist = {"model__num_layers": sp_randint(1, 8),
                  "model__num_neurons": sp_randint(50, 1000),
                  "model__learning_rate": [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3,
                                           1e-2],
                  "model__weight_decay": [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3,
                                          1e-2, 1e-1, 1, 10],
                  "model__activation": ['relu', 'tanh']
                 }

    clf = KerasClassifier(build_fn=model,
                          input_dimension=X_train.shape[1],
                          num_classes=num_classes)

    random_search = RandomizedSearchCV(clf,
                                       param_distributions=param_dist,
                                       n_iter=10, cv=3, n_jobs=-1)
    random_search.fit(X_train, y_train_oh)

    print("Best hyperparameters: ", random_search.best_params_)

    params = {'input_dimension': X_train.shape[1], 'num_classes': num_classes}

    best_hyper = params | random_search.best_params_

    return best_hyper

def random_search_rf(X_train, y_train, X_test, y_test, random_state=42):
    """
    Perform a RandomizedSearchCV to find the best hyperparameters for
    RandomForestClassifier, then return the best hyperparameters.
    X_train: DataFrame, the training data
    y_train: Series, the training labels
    X_test: DataFrame, the test data
    y_test: Series, the test labels
    random_state: int, the random state
    """

    # Define the hyperparameter distributions to search over
    param_distributions = {
        "n_estimators":      [50, 100, 200, 500],
        "max_depth":         [10, 20, 50, 100, None],
        "min_samples_split": [2, 5, 10, 20, 50],
        "min_samples_leaf":  [1, 2, 5, 10],
        "criterion":         ["gini", "entropy"]
    }

    # Instantiate a base RandomForestClassifier
    rf = RandomForestClassifier(random_state=random_state)

    # Set up RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=10,            
        scoring='accuracy',   
        cv=5,                 
        random_state=random_state,
        n_jobs=-1,            
        verbose=1
    )

    # Fit on the training data
    random_search.fit(X_train, y_train)

    # Output the best hyperparameters and cross-validation score
    print("Best Hyperparameters:", random_search.best_params_)
    print("Best CV Score:       ", random_search.best_score_)

    # Optionally, evaluate on the test set
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print("Test Accuracy:       ", test_acc)

    # Return the best hyperparameters
    return random_search.best_params_

def bool_to_int(df):
    """
    Convert boolean features to integer.
    df: DataFrame, the data
    Return:
    df: DataFrame, the data with boolean features converted to integer
    """
    boolean_features = df.select_dtypes(include='bool').columns.tolist()
    for f in boolean_features:
        df[f] = df[f].astype(int)
    return df

def standard_scale(df, scale_features):
    """
    Standard scale the features.
    df: DataFrame, the data
    scale_features: list, the features to scale
    Return:
    df: DataFrame, the data with the features scaled
    """
    # standard scaling
    for f in scale_features:
        df[f] = (df[f] - df[f].mean()) / df[f].std()
    return df

def target_encode_standard_scale(df, cat_features):
    """
    Target encode the categorical features and standard scale them.
    df: DataFrame, the data
    cat_features: list, the categorical features to encode
    Return:
    df: DataFrame, the data with the categorical features encoded and scaled
    """

    for f in cat_features:
        category_means = df.groupby(f)['Severity'].mean()

        # Check if any means are duplicated
        # This tells you which categories share the same mean.
        duplicated_means = category_means[category_means.duplicated()]
        if not duplicated_means.empty:
            print(f"Warning: The following target-encoding mean(s) appear more " 
                  f"than once for feature '{f}':")

        # Map the means to the original DataFrame
        df[f] = df[f].map(category_means)

        # Standard scaling
        df[f] = (df[f] - df[f].mean()) / df[f].std()

    return df