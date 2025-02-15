from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.metrics import f1_score, make_scorer

def random_search_rf_cv_undersample(X_train, y_train, X_test, y_test, random_state=42, method='undersample',
                                    desired_counts='auto'):
    """
    Perform RandomizedSearchCV with cross-validation and RandomUnderSampler on training data.
    Train a RandomForestClassifier and make predictions on X_test.

    Parameters:
        X_train (pd.DataFrame or np.array): Training features.
        y_train (pd.Series or np.array): Training labels.
        X_test (pd.DataFrame or np.array): Test features.
        y_test (pd.Series or np.array, optional): Test labels for evaluation.
        random_state (int): Random state for reproducibility.
        method (str): Sampling method, 'undersample', 'oversample' or SMOTE.

    Returns:
        y_pred (np.array): Predicted labels for X_test.
        best_model (Pipeline): Trained pipeline with the best hyperparameters.
    """
    if method == 'undersample':
        sampler = RandomUnderSampler(random_state=random_state)
    elif method == 'oversample':
        sampler = RandomOverSampler(random_state=random_state, sampling_strategy=desired_counts)
    elif method == 'SMOTE':
        sampler = SMOTE(random_state=random_state, sampling_strategy=desired_counts)
    else:
        raise ValueError("Invalid sampling method. Choose 'undersample', 'oversample' or 'SMOTE'.")

    pipeline = Pipeline([
        ('undersample', sampler),
        ('classifier', RandomForestClassifier(random_state=random_state))
    ])

    # Define hyperparameter distributions
    param_distributions = {
        'classifier__n_estimators': [50, 100, 200, 500, 1000],
        'classifier__max_depth': [None, 10, 20, 30, 40, 50, 100],
        'classifier__min_samples_split': [2, 5, 10, 20, 50],
        'classifier__min_samples_leaf': [1, 2, 4, 6, 10],
        'classifier__criterion': ['gini', 'entropy']
    }

    # Define cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Define scoring metric
    scorer = make_scorer(f1_score, average='macro')

    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=50,
        scoring=scorer,
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=2
    )

    # Fit RandomizedSearchCV
    random_search.fit(X_train, y_train)

    # Get best model
    best_model = random_search.best_estimator_

    # Make predictions on X_test
    y_pred = best_model.predict(X_test)

    test_f1 = f1_score(y_test, y_pred, average='macro')
    print("Test F1 Macro Score:", test_f1)

    # Return predictions and the model
    return y_pred, best_model, test_f1

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


def ANN_model(input_dimension, num_classes, num_layers, num_neurons, learning_rate, weight_decay, activation,
              dropout_rate=0.2):
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
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.F1Score(average='macro', threshold=None,
                                     name='f1_score')
        ]
    )
    return model