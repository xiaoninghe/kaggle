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
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score

train_data = pd.read_csv("./data/train.csv", index_col='PassengerId')
test_data = pd.read_csv("./data/test.csv", index_col='PassengerId')

X_train = train_data
y_train = train_data.Survived

num_cols = ['Pclass', 'SibSp', 'Parch', 'Fare']
num_transformer = SimpleImputer(strategy='constant')

avg_cols = ['Age']
avg_transformer = SimpleImputer(strategy='median')

cat_cols = ['Sex', 'Embarked']
cat_transformer = Pipeline(steps=[
  ('imputer', SimpleImputer(strategy='most_frequent')),
  ('onehot', OneHotEncoder(handle_unknown='ignore')),
])

cols = cat_cols + num_cols + avg_cols
X_train = X_train[cols].copy()
X_test = test_data[cols].copy()

preprocessor = ColumnTransformer(transformers=[
    ('cat', cat_transformer, cat_cols),
    ('num', num_transformer, num_cols),
    ('avg', avg_transformer, avg_cols),
  ])

def get_score(n_estimators, preprocessor, X_train, y_train):
  model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
  pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model),
  ])
  scores = -1 * cross_val_score(pipeline, X_train, y_train,
                              cv=5,
                              scoring='neg_mean_absolute_error')
  return scores.mean()

ns = [30, 60, 100, 150, 300, 500]
results = {n: get_score(n, preprocessor, X_train, y_train) for n in range(103, 112, 1)}

# import matplotlib.pyplot as plt

# plt.plot(list(results.keys()), list(results.values()))
# plt.show()

best_value = min(results, key=results.get)
print(best_value)

model = RandomForestClassifier(n_estimators=best_value, random_state=42)
pipeline = Pipeline(steps=[
  ('preprocessor', preprocessor),
  ('model', model),
])

pipeline.fit(X_train, y_train)

preds_test = pipeline.predict(X_test)
output = pd.DataFrame({
  'PassengerId': X_test.index,
  'Survived': preds_test,
})
output.to_csv('./output/submission.csv', index=False)