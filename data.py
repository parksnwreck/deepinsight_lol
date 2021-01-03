import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from pyDeepInsight import ImageTransformer, LogScaler
import seaborn as sns
from sklearn import preprocessing
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets


"""
Function to remove outliers from the dataset.
"""

def remove_outliers(df, c, blue_win=True):
  blue_win_data = df[df['blue_win'] == (1 if blue_win else 0)][c]
  q_1 = np.percentile(blue_win_data.values, 25)
  q_3 = np.percentile(blue_win_data, 75)

  iqr = q_3 - q_1
  iqr *= 1.5
  lower = q_1 - iqr
  upper = q_3 + iqr
  outlier_idx = blue_win_data[(blue_win_data < lower) | (blue_win_data > upper)].index
  df.drop(outlier_idx, axis = 0, inplace=True)

  return df

"""
Custom dataset to load RAW match data from csv file
"""

class MatchDataSet(Dataset):
  def __init__(self, f):
    csv_raw = pd.read_csv(f)
    x = csv_raw.iloc[:, 3:].values
    y = csv_raw.iloc[:, 2].values

    self.X_train = torch.tensor(x, dtype=torch.float32)
    self.y_train = torch.tensor(y)

  def __len__(self):
    return len(self.y_train)

  def __getitem__(self, idx):
    return self.X_train[idx], self.y_train[idx]

"""
A custom dataset normalized using norm-1 with outliers removed.
"""

class MatchDataSetCleanedN1(Dataset):
  def __init__(self, f):
    csv_raw = pd.read_csv(f)

    # Apply norm-1 normalization
    min_max_scaler = preprocessing.MinMaxScaler()
    csv_x_norm = min_max_scaler.fit_transform(csv_raw.values)
    csv_norm = pd.DataFrame(csv_x_norm, columns=csv_raw.columns)

    # Remove outliers from normalized data
    for c in csv_norm.columns:
      if c != 'matchId' and c != 'blue_win' and c != 'Unnamed':
        csv_norm = remove_outliers(csv_norm, c, blue_win=True)
        csv_norm = remove_outliers(csv_norm, c, blue_win=False)

    x = csv_norm.iloc[:, 3:].values
    y = csv_norm.iloc[:, 2].values

    self.X_train = torch.tensor(x, dtype=torch.float32)
    self.y_train = torch.tensor(y)

  def __len__(self):
    return len(self.y_train)

  def __getitem__(self, idx):
    return self.X_train[idx], self.y_train[idx]

"""
A custom dataset normalized using norm-2 with outliers removed.
"""

class MatchDataSetCleanedN2(Dataset):
  def __init__(self, f):
    csv_raw = pd.read_csv(f)

    # Apply norm-2 normalization
    ln = LogScaler()
    csv_x_norm = ln.fit_transform(csv_raw.iloc[:, 3:].values)
    csv_norm = pd.DataFrame(csv_x_norm, columns=csv_raw.columns[3:])
    csv_norm.insert(0, 'blue_win', csv_raw.iloc[:, 2].values, True) 

    # Remove outliers from norm-2 data
    for c in csv_norm.columns:
      if c != 'matchId' and c != 'blue_win' and c != 'Unnamed':
        csv_norm = remove_outliers(csv_norm, c, blue_win=True)
        csv_norm = remove_outliers(csv_norm, c, blue_win=False)

    x = csv_norm.iloc[:, 1:].values
    y = csv_norm.iloc[:, 0].values

    self.X_train = torch.tensor(x, dtype=torch.float32)
    self.y_train = torch.tensor(y)

  def __len__(self):
    return len(self.y_train)

  def __getitem__(self, idx):
    return self.X_train[idx], self.y_train[idx]

"""
Custom data set that uses DeepInsight to convert non-image data normalized with norm-1 to image data
"""

class MatchDataImageSetCleanedN1(Dataset):
  def __init__(self, f, p=6, fe='tsne'):
    csv_raw = pd.read_csv(f)

    # Normalize using norm-1 and remove outliers
    min_max_scaler = preprocessing.MinMaxScaler()
    csv_x_norm = min_max_scaler.fit_transform(csv_raw.values)
    csv_norm = pd.DataFrame(csv_x_norm, columns=csv_raw.columns)

    for c in csv_norm.columns:
      if c != 'matchId' and c != 'blue_win' and c != 'Unnamed':
        csv_norm = remove_outliers(csv_norm, c, blue_win=True)
        csv_norm = remove_outliers(csv_norm, c, blue_win=False)

    x = csv_norm.iloc[:, 3:].values
    y = csv_norm.iloc[:, 2].values
    
    # Transform data to images
    it = ImageTransformer(feature_extractor=fe, pixels=p)
    it.fit(x, plot=True)
    x_imgs = it.transform(x)

    # Visualize feature density matrix
    fdm = it.feature_density_matrix()
    fdm[fdm == 0] = np.nan

    plt.figure(figsize=(10, 7))

    ax = sns.heatmap(fdm, cmap="viridis", linewidths=0.01, 
                    linecolor="lightgrey", square=True)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    for _, spine in ax.spines.items():
        spine.set_visible(True)
    _ = plt.title("Features per pixel")

    # Set images to tensor
    self.transform = transforms.Compose([transforms.ToTensor()])
    self.X_train = x_imgs
    self.y_train = torch.tensor(y)

    # Visualize image examples
    fig, ax = plt.subplots(1, 4, figsize=(25, 7))
    for i in range(0,4):
        ax[i].imshow(self.X_train[i])
        ax[i].title.set_text("Test[{}] - class '{}'".format(i, y[i]))
    plt.tight_layout()

  def __len__(self):
    return len(self.y_train)

  def __getitem__(self, idx):
    return self.transform(self.X_train[idx]).double(), self.y_train[idx]

"""
Custom data set that uses DeepInsight to convert non-image data normalized with norm-2 to image data
"""

class MatchDataImageSetCleanedN2(Dataset):
  def __init__(self, f, p=6, fe='tsne'):
    csv_raw = pd.read_csv(f)

    # Apply norm-2 normalization and remove outliers
    ln = LogScaler()
    csv_x_norm = ln.fit_transform(csv_raw.iloc[:, 3:].values)
    csv_norm = pd.DataFrame(csv_x_norm, columns=csv_raw.columns[3:])
    csv_norm.insert(0, 'blue_win', csv_raw.iloc[:, 2].values, True) 

    for c in csv_norm.columns:
      if c != 'matchId' and c != 'blue_win' and c != 'Unnamed':
        csv_norm = remove_outliers(csv_norm, c, blue_win=True)
        csv_norm = remove_outliers(csv_norm, c, blue_win=False)

    x = csv_norm.iloc[:, 1:].values
    y = csv_norm.iloc[:, 0].values
    
    # Transform data to images
    it = ImageTransformer(feature_extractor=fe, pixels=p)
    it.fit(x, plot=True)
    x_imgs = it.transform(x)

    # Visualize feature density matrix
    fdm = it.feature_density_matrix()
    fdm[fdm == 0] = np.nan

    plt.figure(figsize=(10, 7))

    ax = sns.heatmap(fdm, cmap="viridis", linewidths=0.01, 
                    linecolor="lightgrey", square=True)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    for _, spine in ax.spines.items():
        spine.set_visible(True)
    _ = plt.title("Features per pixel")

    # Set images to tensor
    self.transform = transforms.Compose([transforms.ToTensor()])
    self.X_train = x_imgs
    self.y_train = torch.tensor(y)

    # Visualize image examples
    fig, ax = plt.subplots(1, 4, figsize=(25, 7))
    for i in range(0,4):
        ax[i].imshow(self.X_train[i])
        ax[i].title.set_text("Test[{}] - class '{}'".format(i, y[i]))
    plt.tight_layout()

  def __len__(self):
    return len(self.y_train)

  def __getitem__(self, idx):
    return self.transform(self.X_train[idx]).double(), self.y_train[idx]