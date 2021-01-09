import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('./'):
  for filename in filenames:
    # print(os.path.join(dirname, filename))
    continue

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

train_data = pd.read_csv("./data/train.csv", index_col='PassengerId')
test_data = pd.read_csv("./data/test.csv", index_col='PassengerId')

print(train_data.head())

y = train_data.Survived
X_train, X_valid, y_train, y_valid = train_test_split(train_data, y, train_size=0.8, test_size=0.2, random_state=42)

print(X_train.isna().sum())

num_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
num_transformer = SimpleImputer(strategy='constant')

cat_cols = ['Sex', 'Embarked']
cat_transformer = Pipeline(steps=[
  ('imputer', SimpleImputer(strategy='most_frequent')),
  ('onehot', OneHotEncoder(handle_unknown='ignore')),
])

cols = cat_cols + num_cols
X_train = X_train[cols].copy()
X_valid = X_valid[cols].copy()
X_test = test_data[cols].copy()

preprocessor = ColumnTransformer(transformers=[
    ('cat', cat_transformer, cat_cols),
    ('num', num_transformer, num_cols),
  ])
model = RandomForestClassifier(n_estimators=100, random_state=42)

pipeline = Pipeline(steps=[
  ('preprocessor', preprocessor),
  ('model', model),
])

pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_valid)

print('MAE:', mean_absolute_error(y_valid, preds))

preds_test = pipeline.predict(X_test)
output = pd.DataFrame({
  'PassengerId': X_test.index,
  'Survived': preds_test,
})
output.to_csv('./output/submission.csv', index=False)