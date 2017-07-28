import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

df = quandl.get("WIKI/GOOGL")
# Pair down the amount of data we have...getting the feaatures we primarily care about
# This leaves us with Adjusted and Volume data
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
# Get percent changed to make data more meaningful
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
# Crude volatility measure
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
# Define a new dataframe with what we want
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

# Our new column that will act as what we want to predict (future price)
forecast_col = 'Adj. Close'
# Replace any gaps in our data with a crazy outlier of -99,999
df.fillna(value=-99999, inplace=True)
# Forecast 1% of our dataset.
# If we have 100 days of data we want to forecast 1 day into the future
forecast_out = int(math.ceil(0.01 * len(df)))

# Assume all current columns are feature and add a new column for our label of future prices
df['label'] = df[forecast_col].shift(-forecast_out)

# Drop any NaN info
df.dropna(inplace=True)

# Since all of our columns are features we can return everything but label
X = np.array(df.drop(['label'], 1))
# We only need our label for y
y = np.array(df['label'])

# Helps with features -1 to 1
X = preprocessing.scale(X)

# Split our data up into training and test data for both X and y
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)