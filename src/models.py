from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.metrics import f1_score, make_scorer
from scipy.stats import randint as sp_randint
from scikeras.wrappers import KerasClassifier
from utils import *

def create_ANN_model(input_dimension, num_classes, num_layers, num_neurons,
                     learning_rate, weight_decay, activation, dropout_rate=0.2):
    """
    Create an Artificial Neural Network model.
    _______
    Inputs:
        input_dimension: number of features
        num_classes: number of classes
        num_layers: number of hidden layers
        num_neurons: number of neurons in each hidden layer
        learning_rate: learning rate
        weight_decay: weight decay
        activation: activation function
        dropout_rate: dropout rate
    ________
    Returns:
        model: a compiled keras model
    """
    model = keras.Sequential()
    model.add(keras.Input(shape=(input_dimension,)))

    model.add(keras.layers.Dense(num_neurons, activation=activation))
    model.add(keras.layers.Dropout(dropout_rate))

    for i in range(num_layers):
        model.add(keras.layers.Dense(num_neurons, activation=activation))
        model.add(keras.layers.Dropout(dropout_rate))

    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate,
                                      weight_decay=weight_decay)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.F1Score(average='macro', threshold=None,
                                     name='f1_score')
        ]
    )
    return model

def random_search_model(X_train, y_train, X_test=None, y_test=None, model_name='RF',
                        random_state=42, method='undersample',
                        desired_counts='auto', n_iter=20, n_jobs=1,
                        verbose=True, n_splits=5):
    """
    Perform RandomizedSearchCV with cross-validation using Random under
    sampling, Random over sampling and SMOTE on the
    training data.
    --------
    Inputs:
        X_train: (pandas DataFrame or numpy array).
        y_train: (pandas Series or numpy array).
        X_test: (pandas DataFrame or numpy array). If None, the function will
                only return the best model.
        y_test: (pandas Series or numpy array). If None, the function will only
                return the best model.
        random_state: Integer, random state for reproducibility.
        method: String, sampling method, 'undersample', 'oversample' or SMOTE.
        desired_counts: Integer, desired counts for the majority class in the case of oversampling and SMOTE.
        n_iter: Integer, number of iterations for RandomizedSearchCV.
        n_jobs: Integer, number of jobs to run in parallel.
        verbose: Boolean, if True, print the test F1 macro score.
        n_splits: Integer, number of splits for cross-validation.
    --------
    Returns:
        best_model: Trained pipeline with the best hyperparameters.
        Optional return:
            y_pred: Numpy array, predicted labels for X_test.
            test_f1: Float, test F1 macro score.
    """
    if model_name == 'RF':
        clf = RandomForestClassifier(random_state=random_state)
        param_distributions = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20, 30, 40, 50, 100],
            'classifier__min_samples_split': [2, 5, 10, 20, 50],
            'classifier__min_samples_leaf': [1, 2, 4, 6, 10],
            'classifier__criterion': ['gini', 'entropy']
        }
        max_class = y_train.value_counts().idxmax()
    elif model_name == 'ANN':
        num_classes, y_train_corrected = correct_target(y_train)
        clf = KerasClassifier(build_fn=create_ANN_model,
                              input_dimension=X_train.shape[1],
                              num_classes=num_classes,
                              activation="relu",
                              num_layers=1,
                              num_neurons=50,
                              learning_rate=1e-8,
                              weight_decay=1e-8,
                              epochs=10,
                              batch_size=256)
        param_distributions = {"classifier__num_layers": sp_randint(1, 8),
                               "classifier__num_neurons": sp_randint(50, 1000),
                               "classifier__learning_rate": [1e-8, 1e-7, 1e-6,
                                                             1e-5, 1e-4, 1e-3,
                                                             1e-2],
                               "classifier__weight_decay": [1e-8, 1e-7, 1e-6,
                                                            1e-5, 1e-4, 1e-3,
                                                            1e-2, 1e-1, 1, 10],
                               "classifier__activation": ['relu', 'tanh'],
                               "classifier__epochs": [10, 50, 100],
                               "classifier__batch_size": [32, 64, 128, 256]
                               }
        max_class = y_train_corrected.value_counts().idxmax()
    else:
        raise ValueError("Invalid model name. Choose 'RF' or 'ANN'.")

    if method == 'undersample':
        sampler = RandomUnderSampler(random_state=random_state)
    elif method == 'oversample':
        under_sampler = RandomUnderSampler(random_state=random_state,
                                           sampling_strategy={max_class:
                                                              desired_counts})
        sampler = RandomOverSampler(random_state=random_state)
    elif method == 'SMOTE':
        under_sampler = RandomUnderSampler(random_state=random_state,
                                           sampling_strategy={max_class: desired_counts})
        sampler = SMOTE(random_state=random_state)
    else:
        raise ValueError("Invalid sampling method. Choose 'undersample', 'oversample' or 'SMOTE'.")

    if method == 'undersample':
        print(clf)
        pipeline = Pipeline([
            ('undersample', sampler),
            ('classifier', clf)
        ])
    elif method in ['oversample', 'SMOTE']:
        pipeline = Pipeline([
            ('undersample', under_sampler),
            ('oversample', sampler),
            ('classifier', clf)
        ])
    print(pipeline.get_params().keys())

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                         random_state=random_state)

    scorer = make_scorer(f1_score, average='macro')
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scorer,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose
    )

    if model_name == 'ANN':
        random_search.fit(X_train, y_train_corrected)
    elif model_name == 'RF':
        random_search.fit(X_train, y_train)
    else:
        raise ValueError("Invalid model name. Choose 'RF' or 'ANN'.")

    best_model = random_search.best_estimator_

    if X_test is not None:
        if model_name == 'ANN':
            _, y_test_corrected = correct_target(y_test)
            y_pred = best_model.predict(X_test)
            y_pred = np.argmax(y_pred, axis=1)
            metrics = best_model.evaluate(X_test, y_test_corrected, verbose=1)
            test_f1 = metrics[2]
            print("Test F1 Macro Score for ANN:", test_f1)
        elif model_name == 'RF':
            y_pred = best_model.predict(X_test)
            test_f1 = f1_score(y_test, y_pred, average='macro')
            print("Test F1 Macro Score for Random Forest:", test_f1)
        return y_pred, best_model, test_f1
    else:
        return best_model






#def random_search_rf_cv_undersample(X_train, y_train, X_test=None, y_test=None,
#                                    random_state=42, method='undersample',
#                                    desired_counts='auto', n_iter=20, n_jobs=1,
#                                    verbose=True, n_splits=5):
"""
    Perform RandomizedSearchCV with cross-validation using Random under sampling, Random over sampling and SMOTE on the
    training data.

    Inputs:
        X_train (pandas DataFrame or numpy array).
        y_train (pandas Series or numpy array).
        X_test (pandas DataFrame or numpy array). If None, the function will only return the best model.
        y_test (pandas Series or numpy array). If None, the function will only return the best model.
        random_state: Integer, random state for reproducibility.
        method: String, sampling method, 'undersample', 'oversample' or SMOTE.
        desired_counts: Integer, desired counts for the majority class in the case of oversampling and SMOTE.
        n_iter: Integer, number of iterations for RandomizedSearchCV.
        n_jobs: Integer, number of jobs to run in parallel.
        verbose: Boolean, if True, print the test F1 macro score.
        n_splits: Integer, number of splits for cross-validation.

    Returns:
        best_model: Trained pipeline with the best hyperparameters.
        Optional return:
            y_pred: Numpy array, predicted labels for X_test.
            test_f1: Float, test F1 macro score.
"""
"""
    if method == 'undersample':
        sampler = RandomUnderSampler(random_state=random_state)
    elif method == 'oversample':
        under_sampler = RandomUnderSampler(random_state=random_state,
                                           sampling_strategy={2: desired_counts})
        sampler = RandomOverSampler(random_state=random_state)
    elif method == 'SMOTE':
        under_sampler = RandomUnderSampler(random_state=random_state,
                                           sampling_strategy={2: desired_counts})
        sampler = SMOTE(random_state=random_state)
    else:
        raise ValueError("Invalid sampling method. Choose 'undersample', 'oversample' or 'SMOTE'.")

    if method == 'undersample':
        pipeline = Pipeline([
            ('undersample', sampler),
            ('classifier', RandomForestClassifier(random_state=random_state))
        ])
    elif method in ['oversample', 'SMOTE']:
        pipeline = Pipeline([
            ('undersample', under_sampler),
            ('oversample', sampler),
            ('classifier', RandomForestClassifier(random_state=random_state))
        ])

    param_distributions = {
        'classifier__n_estimators': [50, 100, 200, 500, 1000],
        'classifier__max_depth': [None, 10, 20, 30, 40, 50, 100],
        'classifier__min_samples_split': [2, 5, 10, 20, 50],
        'classifier__min_samples_leaf': [1, 2, 4, 6, 10],
        'classifier__criterion': ['gini', 'entropy']
    }

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scorer = make_scorer(f1_score, average='macro')

    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scorer,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose
    )

    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_

    if X_test is not None:
        # Make predictions on X_test
        y_pred = best_model.predict(X_test)

        test_f1 = f1_score(y_test, y_pred, average='macro')
        print("Test F1 Macro Score:", test_f1)

        return y_pred, best_model, test_f1
    else:
        return best_model

#def RF(X_train, y_train, X_test, n_estimators=100, max_depth=100, min_samples_split=100, min_samples_leaf=10,
#       criterion='entropy', random_state=42):
#    """
#    Train a RandomForestClassifier and make predictions.
#    Inputs:
#    X_train: training features
#    y_train: training labels
#    X_test: test features
#    n_estimators: number of trees in the forest
#    max_depth: maximum depth of the tree
#    min_samples_split: minimum number of samples required to split an internal node
#    min_samples_leaf: minimum number of samples required to be at a leaf node
#    criterion: function to measure the quality of a split
#    random_state: random seed
#    """

#    # Instantiate the RandomForestClassifier
#    clf = RandomForestClassifier(
#        n_estimators=n_estimators,
#        max_depth=max_depth,
#        min_samples_split=min_samples_split,
#        min_samples_leaf=min_samples_leaf,
#        criterion=criterion,
#        random_state=random_state
#    )
#    # Train the RandomForestClassifier
#    clf.fit(X_train, y_train)

#    # Make predictions
#    y_pred = clf.predict(X_test)
#    return clf, y_pred
"""

#def create_ANN_model(input_dimension, num_classes, num_layers, num_neurons, learning_rate, weight_decay, activation,
#                     dropout_rate=0.2):
#    """
"""
    Create an Artificial Neural Network model.
    input_dimension: number of features
    num_classes: number of classes
    num_layers: number of hidden layers
    num_neurons: number of neurons in each hidden layer
    learning_rate: learning rate
    weight_decay: weight decay
    activation: activation function
    dropout_rate: dropout rate
    Returns:
    model: a compiled keras model
"""
"""
    model = keras.Sequential()
    model.add(keras.Input(shape=(input_dimension,)))

    model.add(keras.layers.Dense(num_neurons, activation=activation))
    model.add(keras.layers.Dropout(dropout_rate))

    for i in range(num_layers):
        model.add(keras.layers.Dense(num_neurons, activation=activation))
        model.add(keras.layers.Dropout(dropout_rate))

    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate,
                                      weight_decay=weight_decay)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.F1Score(average='macro', threshold=None,
                                     name='f1_score')
        ]
    )
    return model
"""
#def random_search_ANN(model, X_train, y_train_oh, num_classes):
#    """
#    Perform a RandomizedSearchCV to find the best hyperparameters for
#    model: the ANN model
#    X_train: the training data
#    y_train_oh: the one-hot encoded training labels
#    num_classes: the number of classes
#    Return:
#    best_hyper: the best hyperparameters
#    """
#    param_dist = {"model__num_layers": sp_randint(1, 8),
#                  "model__num_neurons": sp_randint(50, 1000),
#                  "model__learning_rate": [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3,
#                                           1e-2],
#                  "model__weight_decay": [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3,
#                                          1e-2, 1e-1, 1, 10],
#                  "model__activation": ['relu', 'tanh']
#                 }
#
#    clf = KerasClassifier(build_fn=model,
#                          input_dimension=X_train.shape[1],
#                          num_classes=num_classes)

#    random_search = RandomizedSearchCV(clf,
#                                       param_distributions=param_dist,
#                                       n_iter=10, cv=3, n_jobs=-1)
#   random_search.fit(X_train, y_train_oh)

#    print("Best hyperparameters: ", random_search.best_params_)

#    params = {'input_dimension': X_train.shape[1], 'num_classes': num_classes}

#    best_hyper = params | random_search.best_params_

#    return best_hyper

#def random_search_model(X_train, y_train, X_test=None, y_test=None, model_name='RF',
#                        random_state=42, method='undersample',
#                        desired_counts='auto', n_iter=20, n_jobs=1,
#                        verbose=True, n_splits=5):
#
"""
    Perform RandomizedSearchCV with cross-validation using Random under sampling, Random over sampling and SMOTE on the
    training data.

    Inputs:
        X_train (pandas DataFrame or numpy array).
        y_train (pandas Series or numpy array).
        X_test (pandas DataFrame or numpy array). If None, the function will only return the best model.
        y_test (pandas Series or numpy array). If None, the function will only return the best model.
        random_state: Integer, random state for reproducibility.
        method: String, sampling method, 'undersample', 'oversample' or SMOTE.
        desired_counts: Integer, desired counts for the majority class in the case of oversampling and SMOTE.
        n_iter: Integer, number of iterations for RandomizedSearchCV.
        n_jobs: Integer, number of jobs to run in parallel.
        verbose: Boolean, if True, print the test F1 macro score.
        n_splits: Integer, number of splits for cross-validation.

    Returns:
        best_model: Trained pipeline with the best hyperparameters.
        Optional return:
            y_pred: Numpy array, predicted labels for X_test.
            test_f1: Float, test F1 macro score.
"""
"""
    if model_name == 'RF':
        clf = RandomForestClassifier(random_state=random_state)
        param_distributions = {
            'classifier__n_estimators': [50, 100, 200, 500, 1000],
            'classifier__max_depth': [None, 10, 20, 30, 40, 50, 100],
            'classifier__min_samples_split': [2, 5, 10, 20, 50],
            'classifier__min_samples_leaf': [1, 2, 4, 6, 10],
            'classifier__criterion': ['gini', 'entropy']
        }
        max_class = y_train.value_counts().idxmax()
    elif model_name == 'ANN':
        num_classes, y_train_corrected = correct_target(y_train)
        clf = KerasClassifier(build_fn=create_ANN_model, input_dimension=X_train.shape[1], num_classes=num_classes,
                              epochs=100, batch_size=256)
        param_distributions = {"classifier__num_layers": sp_randint(1, 8),
                               "classifier__num_neurons": sp_randint(50, 1000),
                               "classifier__learning_rate": [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
                               "classifier__weight_decay": [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10],
                               "classifier__activation": ['relu', 'tanh']
                               }
        max_class = y_train_corrected.value_counts().idxmax()
    else:
        raise ValueError("Invalid model name. Choose 'RF' or 'ANN'.")

    if method == 'undersample':
        sampler = RandomUnderSampler(random_state=random_state)
    elif method == 'oversample':
        under_sampler = RandomUnderSampler(random_state=random_state,
                                           sampling_strategy={max_class: desired_counts})
        sampler = RandomOverSampler(random_state=random_state)
    elif method == 'SMOTE':
        under_sampler = RandomUnderSampler(random_state=random_state,
                                           sampling_strategy={max_class: desired_counts})
        sampler = SMOTE(random_state=random_state)
    else:
        raise ValueError("Invalid sampling method. Choose 'undersample', 'oversample' or 'SMOTE'.")

    if method == 'undersample':
        pipeline = Pipeline([
            ('undersample', sampler),
            ('classifier', clf)
        ])
    elif method in ['oversample', 'SMOTE']:
        pipeline = Pipeline([
            ('undersample', under_sampler),
            ('oversample', sampler),
            ('classifier', clf)
        ])

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scorer = make_scorer(f1_score, average='macro')

    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scorer,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose
    )

    if model_name == 'ANN':
        random_search.fit(X_train, y_train_corrected)
    elif model_name == 'RF':
        random_search.fit(X_train, y_train)
    else:
        raise ValueError("Invalid model name. Choose 'RF' or 'ANN'.")

    best_model = random_search.best_estimator_

    if X_test is not None:
        if model_name == 'ANN':
            _, y_test_corrected = correct_target(y_test)
            y_pred = best_model.predict(X_test)
            y_pred = np.argmax(y_pred, axis=1)
            metrics = best_model.evaluate(X_test, y_test_corrected, verbose=1)
            test_f1 = metrics[2]
            print("Test F1 Macro Score for ANN:", test_f1)
        elif model_name == 'RF':
            y_pred = best_model.predict(X_test)
            test_f1 = f1_score(y_test, y_pred, average='macro')
            print("Test F1 Macro Score for Random Forest:", test_f1)
        return y_pred, best_model, test_f1
    else:
        return best_model
"""