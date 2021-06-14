import pandas as pd
import numpy as np
import xgboost as xgb
from prophet import Prophet


MAX_DATE = pd.Timestamp('2019-10-01')
TIME_PERIOD = 32
OFFSET = 0
L_MEAN = 4


def train_test_split(df, period):
  last_date =  df['ds'].max()
  end_date = pd.Timestamp(MAX_DATE) - pd.DateOffset(weeks=OFFSET)
  end_date = min(pd.Timestamp(MAX_DATE), last_date)
  start_date = end_date - pd.DateOffset(weeks=period)
  test = (df['ds'] >= start_date) & (df['ds'] < end_date)
  train = df['ds'] < start_date
        
  return train, test


def run_nb(df):
  train, test = train_test_split(df)
        
  forecast = df.copy()
  weights = pd.read_csv('weights_l_p.csv')
  weights['sum'] = weights['r'].cumsum()/weights['r'].sum()
  weights = weights[weights['sum'] < .8]
  dfs = []
  for x in weights.index:
    l = weights.loc[x, 'l']
    p = weights.loc[x, 'p']
    try:
      new_df = forecast[(df['location_code']==l) & (df['product_code']==p)].reset_index()
      new_df['yhat'] = forecast[(df['location_code']==l) & (df['product_code']==p) & train]['y'].mean()
      dfs.append(new_df)
    except Exception as e:
      print(e)
  
  return pd.concat(dfs, ignore_index=True)


def run_xgboost(df):
  forecast = df.copy()
  for i in range(TIME_PERIOD - L_MEAN):
    period = TIME_PERIOD - i
    train, test = train_test_split(df, period)
    xg_reg = xgb.XGBRegressor(seed=42, subsample=0.3, objective='reg:squarederror', n_estimators=100, max_depth=20, learning_rate=0.1, gamma=8, colsample_bytree=0.1)
    features = df.columns.difference(['y', 'ds'])
    # features = ['location_code', 'product_code', 'year', 'month', 'week']
    result = xg_reg.fit(df[train][features], df[train]['y'])
    y_pred = result.predict(df[features])
    forecast['yhat_{}'.format(str(i))] = y_pred
  
  return forecast


def run_prophet(df):
  df = df[df['ds'] < MAX_DATE]
  train, test = train_test_split(df)
  forecast = df.copy()
  forecast['yhat'] = np.nan

  features =  ['avg_sales_price'] # df.columns.difference(['ds', 'y'])
  
  weights = pd.read_csv('weights_l_p.csv')
  weights['sum'] = weights['r'].cumsum()/weights['r'].sum()
  weights = weights[weights['sum'] < .8]
  dfs = []
  for x in weights.index:
    l = weights.loc[x, 'l']
    p = weights.loc[x, 'p']
    m = Prophet(daily_seasonality=False, weekly_seasonality=False)
    
    for i in features:
      m.add_regressor(i)

    try:
      m.fit(df[(df['location_code']==l) & (df['product_code']==p) & train])
      new_df = forecast[(df['location_code']==l) & (df['product_code']==p)].reset_index()
      new_df['yhat'] = m.predict(df[(df['location_code']==l) & (df['product_code']==p)])['yhat']
      dfs.append(new_df)
    except Exception as e:
      print(e)

  return pd.concat(dfs, ignore_index=True)


def preprocess(df):
  df['ds'] = pd.to_datetime(df['ds'])
  end_date = pd.Timestamp(MAX_DATE) - pd.DateOffset(weeks=OFFSET)
  df = df[df['ds'] <= end_date]
  df['year'] = df['ds'].dt.year
  df['month'] = df['ds'].dt.month
  df['week'] = df['ds'].dt.week

  return df


def main():
  df = pd.read_csv('reduced_data1.csv').iloc[:, 2:]
  df = preprocess(df)
  forecast = run_xgboost(df)
  forecast.to_csv('forecast_' + str(OFFSET) + '_' + str(TIME_PERIOD) + '.csv')


main()