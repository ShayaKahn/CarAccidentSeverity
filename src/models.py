from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from tensorflow import keras
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.metrics import f1_score, make_scorer
from tensorflow.keras import regularizers
from scikeras.wrappers import KerasClassifier
import xgboost
from src.utils import *

def create_ANN_model(input_dimension, num_classes, num_layers, num_neurons,
                     learning_rate, weight_decay, activation, dropout_rate=0.2):
    """Create an Artificial Neural Network model.
    """
    model = keras.Sequential()
    model.add(keras.Input(shape=(input_dimension,)))

    model.add(keras.layers.Dense(num_neurons, activation=activation,
                                 kernel_regularizer=regularizers.l2(
                                     weight_decay)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(dropout_rate))

    for i in range(num_layers):
        model.add(keras.layers.Dense(num_neurons, activation=activation,
                                 kernel_regularizer=regularizers.l2(
                                     weight_decay)))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(dropout_rate))

    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate,
                                      weight_decay=weight_decay)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.F1Score(average='macro', threshold=None, name='f1_score')
        ]
    )
    return model

def random_search_model(X_train, y_train, param_distributions, X_test=None,
                        y_test=None, model_name='RF', random_state=42,
                        method='undersample', desired_counts='auto', n_iter=20,
                        n_jobs=1, verbose=True, n_splits=5, tree_method="hist",
                        device="gpu"):
    """Perform RandomizedSearchCV with cross-validation using Random under
    sampling, Random over sampling and SMOTE on the training and CV data.
    --------
    Inputs:
        X_train: (pandas DataFrame or numpy array).
        y_train: (pandas Series or numpy array).
        param_distributions: Dictionary, parameters for RandomizedSearchCV.
        X_test: (pandas DataFrame or numpy array). If None, the function will
                only return the best model.
        y_test: (pandas Series or numpy array). If None, the function will only
                return the best model.
        model_name: String, model name, 'RF', 'GB', 'XGB', or 'ANN'.
        random_state: Integer, random state for reproducibility.
        method: String, sampling method, 'undersample', 'oversample' or SMOTE.
        desired_counts: Integer, desired counts for the majority class in the case of oversampling and SMOTE.
        n_iter: Integer, number of iterations for RandomizedSearchCV.
        n_jobs: Integer, number of jobs to run in parallel.
        verbose: Integer in {1,2,3}.
        n_splits: Integer, number of splits for cross-validation.
        tree_method: String for XGBoost.
        device: String for XGBoost.
    --------
    Returns:
        best_model: Trained pipeline with the best hyperparameters.
        Optional return:
            y_pred: Numpy array, predicted labels for X_test.
            test_f1: Float, test F1 macro score.
    """
    # Define the classifier.
    if model_name == 'RF':
        clf = RandomForestClassifier(random_state=random_state)
        max_class = y_train.value_counts().idxmax()
    elif model_name == 'GB':
        clf = GradientBoostingClassifier(random_state=random_state)
        max_class = y_train.value_counts().idxmax()
    elif model_name == 'XGB':
        _, y_train_corrected = correct_target(y_train)
        clf = xgboost.XGBClassifier(random_state=random_state,
                                    tree_method=tree_method,
                                    device=device)
        max_class = y_train_corrected.value_counts().idxmax()
    elif model_name == 'ANN':
        num_classes, y_train_corrected = correct_target(y_train)
        clf = KerasClassifier(build_fn=create_ANN_model,
                              input_dimension=X_train.shape[1],
                              num_classes=num_classes,
                              activation="tanh",
                              num_layers=1,
                              num_neurons=50,
                              learning_rate=1e-8,
                              weight_decay=1e-8,
                              dropout_rate=0.2,
                              epochs=10,
                              batch_size=256)
        max_class = y_train_corrected.value_counts().idxmax()
    else:
        raise ValueError("Invalid model name. Choose 'RF', 'GB', 'XGB', or 'ANN'.")

    # Define the sampling method.
    if method == 'undersample':
        under_sampler = RandomUnderSampler(random_state=random_state)
    elif method == 'oversample':
        under_sampler = RandomUnderSampler(random_state=random_state,
                                           sampling_strategy={max_class:
                                                              desired_counts})
        over_sampler = RandomOverSampler(random_state=random_state)
    elif method == 'SMOTE':
        under_sampler = RandomUnderSampler(random_state=random_state,
                                           sampling_strategy={max_class: desired_counts})
        over_sampler = SMOTE(random_state=random_state)
    else:
        raise ValueError("Invalid sampling method. Choose 'undersample', 'oversample' or 'SMOTE'.")

    # Define the pipeline.
    if method == 'undersample':
        pipeline = Pipeline([
            ('undersample', under_sampler),
            ('classifier', clf)
        ])
    elif method in ['oversample', 'SMOTE']:
        pipeline = Pipeline([
            ('undersample', under_sampler),
            ('oversample', over_sampler),
            ('classifier', clf)
        ])

    # Define the cross-validation.
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                         random_state=random_state)

    # Define the scorer.
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

    if model_name in ['ANN', 'XGB']:
        random_search.fit(X_train, y_train_corrected)
    elif model_name in ['RF', 'GB']:
        random_search.fit(X_train, y_train)
    else:
        raise ValueError("Invalid model name. Choose 'RF', 'GB', 'XGB', or 'ANN'.")

    best_model = random_search.best_estimator_
    best_params = random_search.best_params_

    # Evaluate the model on the test set.
    if X_test is not None:
        if model_name in ['ANN', 'XGB']:
            _, y_test_corrected = correct_target(y_test)
            y_pred = best_model.predict(X_test)
            test_f1 = f1_score(y_test_corrected, y_pred, average='macro')
            if model_name == 'ANN':
                label = 'Artificial neural network'
            else:
                label = 'XGBoost'
            print(f"Test F1 Macro Score for {label}:", test_f1)
        elif model_name in ['RF', 'GB']:
            y_pred = best_model.predict(X_test)
            test_f1 = f1_score(y_test, y_pred, average='macro')
            if model_name == 'RF':
                label = 'Random Forest'
            else:
                label = 'Gradient Boosting Classifier'
            print(f"Test F1 Macro Score for {label}:", test_f1)
        return best_model, best_params, y_pred, test_f1
    else:
        return best_model, best_params
