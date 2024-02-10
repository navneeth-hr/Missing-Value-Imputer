# CS-Foundations-Project

# MissingValImputerDats6450

A Python package for handling missing values in datasets.

## Overview

`MissingValImputerDats6450` is a Python package that provides a `MissingValImputerDats6450` class to handle missing values in datasets. It uses a combination of feature importance and MissForest imputation techniques to impute missing values in a principled manner.

View our package on Test PyPI - https://test.pypi.org/project/MissingValImputerDats6450/

## Usage
Here is an example of how to use MissingValImputerDats6450 with dataset:
```
#Install the package

%pip install -i https://test.pypi.org/simple/ MissingValImputerDats6450

import seaborn as sns
import numpy as np

from MissingValImputerDats6450.MissingValImputerDats6450 import MissingValImputerDats6450

dataframe = sns.load_dataset("diamonds")
dataframe
columns_with_missing = ['depth', 'table', 'x', 'z','y']
missing_percentage = 0.30

for column in columns_with_missing:
    mask = np.random.rand(len(dataframe)) < missing_percentage
    dataframe.loc[mask, column] = np.nan

dataframe
%%time

mvh = MissingValImputerDats6450()
mvh.fit(dataframe, "price", categorical=["cut","color","clarity"])
dataframe = mvh.transform(dataframe)
dataframe

sns.heatmap(dataframe.corr(), annot=True)
df = sns.load_dataset('diamonds')
diff = df.corr()
sns.heatmap(diff,annot  =True)

!pip install xgboost
!pip install MissForest
from missforest.missforest import MissForest

dataframe1 = sns.load_dataset("diamonds")
dataframe1

columns_with_missing = ['depth', 'table', 'x', 'z','y']
missing_percentage = 0.30

for column in columns_with_missing:
    mask = np.random.rand(len(dataframe1)) < missing_percentage
    dataframe1.loc[mask, column] = np.nan

dataframe1
%%time

mvht = MissForest()
mvht.fit(dataframe1, categorical=["cut","color","clarity"])
dataframe1 = mvht.transform(dataframe1)
```
