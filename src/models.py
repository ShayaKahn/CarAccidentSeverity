from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras
import tensorflow as tf


def RF(X_train, y_train, X_test, n_estimators=100, max_depth=100, min_samples_split=100, min_samples_leaf=10,
       criterion='entropy', random_state=42):
    """
    Train a RandomForestClassifier and make predictions.
    Inputs:
    X_train: training features
    y_train: training labels
    X_test: test features
    n_estimators: number of trees in the forest
    max_depth: maximum depth of the tree
    min_samples_split: minimum number of samples required to split an internal node
    min_samples_leaf: minimum number of samples required to be at a leaf node
    criterion: function to measure the quality of a split
    random_state: random seed
    """

    # Instantiate the RandomForestClassifier
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        random_state=random_state
    )

    # Train the RandomForestClassifier
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)
    return clf, y_pred


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