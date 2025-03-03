import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import tensorflow as tf

def find_outliers_IQR(df):
    """Find outliers using the Interquartile Range (IQR) method.
    """
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    IQR = q3 - q1
    outliers = df[((df < (q1-1.5*IQR)) | (df > (q3+1.5*IQR)))]
    return outliers

def CM(y_test, y_pred):
    """Plot the confusion matrix.
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

def correct_target(y):
    """Convert the labels start from 0 instead of 1.
    """
    y_corrected = y - 1
    num_classes = len(np.unique(y_corrected))
    return num_classes, y_corrected

def to_categorical(y):
    """Convert the labels to one-hot encoding.
    """
    y_corrected = y - 1
    num_classes = len(np.unique(y_corrected))
    y_oh = tf.keras.utils.to_categorical(y_corrected, num_classes)
    return num_classes, y_oh

def bool_to_int(df):
    """Convert boolean features to integer.
    """
    boolean_features = df.select_dtypes(include='bool').columns.tolist()
    for f in boolean_features:
        df[f] = df[f].astype(int)
    return df

def standard_scale(df, scale_features):
    """Standard scale the features.
    """
    for f in scale_features:
        df[f] = (df[f] - df[f].mean()) / df[f].std()
    return df

def target_encode_standard_scale(df, cat_features):
    """Target encode the categorical features and standard scale them.
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
