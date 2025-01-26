from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

# Define custom transformer to find outliers
class IQRImputer(BaseEstimator, TransformerMixin):
    """
    Impute outliers using IQR method.
    """
    def __init__(self, col, method, top=False):
        """
        col: name of the feature
        method: string, 'mean' or 'median'
        top: boolean, if True impute only the top outliers, if False impute both top and bottom outliers
        """
        self.col = col
        self.method = method
        self.top = top

    def fit(self, X, y=None):
        # define the value to impute
        if self.method == 'mean':
            self.metric = X[self.col].mean()
        elif self.method == 'median':
            self.metric = X[self.col].median()
        return self

    def transform(self, X):
        # apply IQR method
        X = X.copy()
        q1 = X[self.col].quantile(0.25)
        q3 = X[self.col].quantile(0.75)
        IQR = q3 - q1
        if self.top:
            X.loc[X[self.col] > (q3 + 1.5 * IQR), self.col] = self.metric
        else:
            X.loc[(X[self.col] > (q3 + 1.5 * IQR)) | (X[self.col] < (
                q3 - 1.5 * IQR)), self.col] = self.metric
        return X


# Define custom transformer for upper bound
class UpperBoundImputer(BaseEstimator, TransformerMixin):
    """
    Impute values greater than upper_bound.
    """
    def __init__(self, col, method, upper_bound):
        """
        col: name of the feature
        method: string, 'mean' or 'median'
        upper_bound: the upper bound value
        """
        self.col = col
        self.method = method
        self.upper_bound = upper_bound

    def fit(self, X, y=None):
        # define the value to impute
        if self.method == 'mean':
            self.metric = X[self.col].mean()
        elif self.method == 'median':
            self.metric = X[self.col].median()
        return self

    def transform(self, X):
        X = X.copy()
        X.loc[X[self.col] > self.upper_bound, self.col] = self.metric
        return X


# Define custom transformer for replacment in cathegorical features
class Replace(BaseEstimator, TransformerMixin):
    """
    Replace values in a column
    """
    def __init__(self, col, mapping):
        """
        col: name of the feature
        mapping: tuple, (old_value, new_value)
        """
        self.col = col
        self.mapping = mapping

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.col] = X[self.col].replace(*self.mapping)
        return X

# Define custom transformer for frequancy encoding
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Frequency encoding for categorical features
    """
    def __init__(self, cols):
        """
        cols: list of column names
        """
        self.cols = cols
        self.freq_maps = {}

    def fit(self, X, y=None):
        # Compute frequency maps for each specified column
        for col in self.cols:
            self.freq_maps[col] = X[col].value_counts(normalize=True)
            vals = np.array(self.freq_maps[col].values)
            if np.size(np.unique(vals)) != np.size(vals):
                raise Exception(f"Frequency encoding cannot be applied "
                f"for the feature: '{col}'")
        return self

    def transform(self, X):
        X = X.copy()
        for col, freq_map in self.freq_maps.items():
            X.loc[:, col] = X[col].map(freq_map).astype('float64')
        return X

# Define custom transformer to remove rows with Null values
class DropNaRows(BaseEstimator, TransformerMixin):
    """
    Drop rows with Null values
    """
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.columns:
            return X.dropna(subset=self.columns)
        else:
            return X.dropna()