from sklearn.linear_model import ElasticNet, ElasticNetCV, LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.pipeline import make_pipeline
from sklearn_pandas import DataFrameMapper, CategoricalImputer
from sklearn_pandas import GridSearchCV, DataFrameMapper, RandomizedSearchCV
from catboost import CatBoostRegressor
import pandas as pd
import numpy as np
import pickle

import warnings
warnings.filterwarnings("ignore", category=Warning)

df = pd.read_csv('world-happiness-report-2019.csv')
df.info()

df.columns = [i.lower().replace('\n', ' ').replace(' ', '_').replace('(', '').replace(')', '') for i in df.columns]
df.rename(columns={'country_region': 'country_or_region'}, inplace=True)

predictors = [c for c in df if (c != 'ladder') & (c != 'sd_of_ladder')& (c != 'country_or_region')]
target = df.ladder

X = df[predictors]
y = target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

mapper = DataFrameMapper([
    (['positive_affect'], [SimpleImputer(strategy='most_frequent')]),
    (['negative_affect'], [SimpleImputer(strategy='most_frequent')]),
    (['social_support'], [SimpleImputer(strategy='most_frequent')]),
    (['freedom'], [SimpleImputer(strategy='most_frequent')]),
    (['corruption'], [SimpleImputer(strategy='most_frequent')]),
    (['generosity'], [SimpleImputer(strategy='most_frequent')]),
    (['log_of_gdp_per_capita'], [SimpleImputer(strategy='most_frequent')]),
    (['healthy_life_expectancy'], [SimpleImputer(strategy='most_frequent')])
])

X['healthy_life_expectancy'].dtype
X['log_of_gdp_per_capita'].dtype

# LabelEncoder gives you one column with ranked values (how does it rank them? can you choose?)

model = Ridge()

pipe = make_pipeline(mapper, model)
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
pickle.dump(pipe, open('pipe.pkl', 'wb'))

# # load from a model
# del pipe
# pipe = pickle.load(open('model/pipe.pkl', 'rb'))
