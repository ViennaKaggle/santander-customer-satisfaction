from functools import wraps
from collections import OrderedDict
import re

import numpy as np
import pandas as pd

from IPython.core.display import display, HTML
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.metrics import make_scorer
from sklearn import cross_validation as cv
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.manifold import TSNE
from sklearn import tree
from sklearn.ensemble import ExtraTreesClassifier


def random_subset(X, y, dims, n_shuffle=10, seed=42):
  """Selects a random subset of X and y according to the dimensions
  
  Params:
    X: n x d pandas dataframe
    y: n x 1 pandas dataframe 
    dims: list of tuples
    n_shuffle: run n_shuffle shuffle operations on the set of indices
    seed: seed the random number generator
  
  Returns:
    X', y': sampled dataframes

  Example:
    Select only 75% of the values with target 0, and all values
    where target is 1
    $ dims = [(0, 0.75), (1, 1.0)]
  """
  np.random.seed(seed)
  idx = []
  
  for target, factor in dims:
    if (0 <= factor < 1.0):
      n_samples = int(len(y[y == target]) * factor)
      idx_sub = np.random.choice(y.index[y == target], n_samples, replace=False)
    else:
      idx_sub = y.index[y == target]
    # Stack the indices together  
    idx = np.hstack((idx, idx_sub))
  for i in range(n_shuffle):
    np.random.shuffle(idx)

  return X.loc[idx.astype(int)], y[idx.astype(int)]


def truncate(value, max_length=100, suffix="...", pre=5):
    if len(value) > max_length:
      return value[0:pre] + suffix + value[pre+len(suffix)+1:max_length+1]
    else:
      return value

def score(*args, **kwargs):
  """Decorator, that transform a function to a scorer.
  A scorer has the arguments estimator, X, y_true, sample_weight=None
  """
  decorator_args = args
  decorator_kwargs = kwargs
  def score_decorator(func):
    @wraps(func)
    def func_wrapper(*args, **kwargs):
      func_args = args
      func_kwargs = kwargs
      scorer = make_scorer(func, *decorator_args, **decorator_kwargs)
      return scorer(*func_args, **func_kwargs)
    return func_wrapper
  return score_decorator

def folds(y, n_folds=4, stratified=False, random_state=42, shuffle=True, **kwargs):
  if stratified:
    return cv.StratifiedKFold(y, n_folds=n_folds, shuffle=shuffle, random_state=random_state, **kwargs)
  return cv.KFold(n=len(y), n_folds=n_folds, shuffle=shuffle, random_state=random_state, **kwargs)

def cross_val(estimator, X, y, n_jobs=-1, n_folds=4, proba=False, **kwargs):
  # Extract values from pandas DF
  if hasattr(X, 'values'):
    X = X.values
  if hasattr(y, 'values'):
    y = y.values

  # Return Cross validation score
  if proba is True:
    estimator.predict = lambda self, *args, **kwargs: self.predict_proba(*args, **kwargs)[:,1]

  return cv.cross_val_score(estimator, X, y, cv=folds(y, n_folds=n_folds), n_jobs=n_jobs, **kwargs)


class BaseTransform(BaseEstimator, ClassifierMixin, TransformerMixin):
  """Transform Interface"""
  def __init__(self):
    pass

  def fit(self, X, y=None, **fit_params):
    return self

  def transform(self, X):
    return X

class PandasTransform(BaseTransform):
  def __init__(self):
    pass

  def transform(self, X):
    return X.values


class Log1pTransform(BaseTransform):
  def __init__(self, columns=None):
    self.columns = columns=None

  def transform(self, X):
    if self.columns:
      for column in self.columns:
        X[column] = np.log1p(X[column])
        return X
    else:
      return np.log1p(X)

  def inverse_transform(self, X):
    if self.columns:
      for column in self.columns:
        X[column] = np.expm1(X[column])
        return X
    else:
      return np.expm1(X)


class NanPreProcessor(TransformerMixin):
  """Fills NaN with class median
  @source: https://www.kaggle.com/cbrogan/titanic/xgboost-example-python/code
  @based: http://stackoverflow.com/a/25562948"""
  def fit(self, X, y=None):
    self.fill = pd.Series([X[c].value_counts().index[0]
      if X[c].dtype == np.dtype('O') else X[c].median() for c in X], index=X.columns)
    return self
  def transform(self, X, y=None):
    return X.fillna(self.fill)


def tsne_plot(X, y, title="", metric='l1', random_state=0, legend_loc='upper left', n_samples=None, n_components=2, _perplexity=30, _n_iter=1000):
  """Plots the first 2 components of the t-distributed Stochastic Neighbor Embedding
  References:
   * http://blog.kaggle.com/2012/11/02/t-distributed-stochastic-neighbor-embedding-wins-merck-viz-challenge/"""

  if n_samples:
      # Get the shape of the training set
      n_samples_orig, n_features = np.shape(X)

      # Select 5000 random indices
      rnd_indices = np.random.choice(n_samples_orig, n_samples)

      X = X[rnd_indices]
      y = y[rnd_indices]

  # Create a t-SNE model
  model = TSNE(n_components=n_components, perplexity=_perplexity, n_iter=_n_iter, random_state=random_state, metric=metric)
  X_trans = model.fit_transform(X)

  # Get a list of unique labels
  labels = np.unique(y)

  # This is only needed to adjust the size of the figure
  # because otherwise it is really small
  plt.figure(figsize=(15, 15), dpi=120)

  # Get a list of color values
  colors = cm.rainbow(np.linspace(0, 1, len(labels)))

  # Loop over labels
  # enumerate also return the index from the list
  for i, label in enumerate(labels):

      # Get a feature vector with the matching label
      # and add a scatter plot with the dataset
      plt.scatter(X_trans[y == label][:,0], X_trans[y == label][:,1], c=colors[i], label=label)

  # Add a legend
  plt.legend(loc=legend_loc)

  # Add axis labels
  plt.xlabel("1st component")
  plt.ylabel("2nd component")

  # Add a title
  plt.title(title)

  # Render the plot
  plt.show()


class Table(object):
  def __init__(self, max_col_width=30):
    self.values = OrderedDict()
    self.size = 0
    self.max_col_width = max_col_width

  def add_column(self, label, values):
    if label in self.values:
      raise ValueError('Duplicate Column')
    self.values[label] = values
    self.size = max(len(values), self.size)

  def max_length(self, col):
    return max(max(list(map(lambda c: len(str(c)), self.values[col]))), len(col))

  def html(self):
    output = ""

    output += "<table>"

    output += "<thead>"
    output += "<tr>"
    for col in self.values:
      output +=  '<th>{name:s}</th>'.format(name=col)
    output += "</tr>"
    output += "</thead>"

    output += "<tbody>"
    for r in range(self.size):
      output += "<tr>"
      for col in self.values:
        output += '<td>{name:s}</td>'.format(name=str(self.values[col][r]))
      output += "</tr>"
    output += "</tbody>"

    output += "</table>"
    return output

  def __str__(self):
    col_sep = " |"
    output = ""

    dim = {col: min(self.max_length(col), self.max_col_width) for col in self.values}

    for col in self.values:
      output +=  ' {name:{fill}<{width}s}'.format(name=truncate(col, dim[col]), fill=" ", width=dim[col])
      output += col_sep
    output += "\n"

    for col in self.values:
      output +=  ' {name:{fill}<{width}s}'.format(name="", fill="-", width=dim[col])
      output += col_sep
    output += "\n"

    for r in range(self.size):
      for col in self.values:
        output += ' {name:{fill}<{width}s}'.format(name=truncate(str(self.values[col][r]), dim[col]), fill=' ', width=dim[col])
        output += col_sep
      output += "\n"

    return output

def pretty_stats(data, stat=None, target_key=None):
  """Generate a pretty statistic about the dataframe *data*"""

  if not stat or stat is 'general':
    table = Table()

    table.add_column('property', values=[
      'Number of features',
      'Number of Samples'
    ])

    table.add_column('values', values=[
      len(data.columns),
      len(data)
    ])

    display(HTML('<h1>General</h1>'))
    display(HTML(table.html()))

  if target_key and (not stat or stat is 'target'):
    table = Table()
    aggregate = data.groupby([target_key]).agg({'ID':len})

    table.add_column('target', values=aggregate.index.values)
    table.add_column('count', values=aggregate.values.flatten())
    table.add_column('percentage', values=aggregate.values.flatten()/len(data))

    display(HTML('<h1>Distribution per Target</h1>'))
    display(HTML(table.html()))

  if not stat or stat is 'distribution':
    table = Table()
    distribution = data.describe()

    table.add_column('feature', values=list(data.columns))
    table.add_column('Unique', values=data.apply(lambda x: len(x.unique())))
    table.add_column('NaN', values=data.isnull().sum().values)
    table.add_column('min', values=distribution.ix['min'].values)
    table.add_column('mean', values=distribution.ix['mean'].values)
    table.add_column('max', values=distribution.ix['max'].values)

    display(HTML('<h1>Distribution</h1>'))
    display(HTML(table.html()))

  if not stat or stat is 'correlation':
    table = Table()
    correlation = data.corr()

    # Create a diagonal condition to filter the correlation of a column with itself
    diag_mask = np.zeros(correlation.shape, dtype='bool')
    np.fill_diagonal(diag_mask, True)

    table.add_column('feature', values=list(data.columns))
    table.add_column('highest value', values=correlation.mask(cond=diag_mask).abs().max(skipna=True).values)
    table.add_column('correlated with', values=correlation.mask(cond=diag_mask).abs().idxmax(skipna=True).values)
    table.add_column('mean', values=correlation.mask(cond=diag_mask).abs().mean().values)

    display(HTML('<h1>Correlation</h1>'))
    display(HTML(table.html()))

def feature_importance(X, y, criterion='entropy', n_estimators=250, random_state=0):
  clf = ExtraTreesClassifier(n_estimators=n_estimators, random_state=random_state, criterion=criterion)
  clf = clf.fit(X, y)
  importances = clf.feature_importances_
  std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
  indices = np.argsort(importances)[::-1]

  return pd.DataFrame({"column":X.columns, "importance":importances, "std": std}).set_index(indices)

def plot_feature_importance(X, y, **kwargs):
  importances = feature_importance(X, y, **kwargs).sort(columns="importance", ascending=False)

  # Plot the feature importances of the forest
  plt.figure(figsize=(15, 5), dpi=120)
  plt.title("Feature importances")
  plt.bar(range(len(importances)), importances['importance'].values, color="r", yerr=importances['std'].values, align="center")
  plt.xticks(range(len(importances)), importances.column.values)
  plt.xticks(rotation=90)
  plt.xlim([-1, len(importances)])
  plt.show()

def feature_hists(data, bins=20):
  uniques = data.apply(lambda x: len(x.unique()))
  bin_options = {col: min(bins, uniques[col]) for col in data.columns}

  for col in data.columns:
      plt.figure(figsize=(15, 5), dpi=120)
      plt.title(col)
      plt.title(col)
      data[col].plot(kind='hist', alpha=0.5, bins=bin_options[col])
