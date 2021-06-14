import pandas as pd
import numpy as np
import scipy.stats as st
from simulation.simulation import simulate
from simulation import metrics
import click


def get_rmse(df, l_code, p_code):
  df_pl = df[(df['location_code']==l_code) & (df['product_code']==p_code)].iloc[-32:, :]
  error = df_pl['yhat'] - df['y']
  
  return error.std()


def get_safety_stock(std, ci, y_mean):
  return st.norm.ppf(ci) * np.sqrt(4*std + y_mean**2 * 1) # st.norm.ppf(ci, scale=std)


def print_result(result):
  print('\nModel')
  print('-'*25)
  print('%s: %7d' % ('Total demand', np.sum(result['y'])))
  print('%s: %5d' % ('Total forecast', np.sum(result['yhat'])))
  
  print('%s: %8d' % ('Mean demand', np.mean(result['y'])))
  print('%s: %6d' % ('Mean forecast', np.mean(result['yhat'])))

  print('%s: %6.1f' % ('Std of demand', np.std(result['y'])))
  print('%s: %4.1f' % ('Std of forecast', np.std(result['yhat'])))
  
  print('%s: %7d' % ('Safety stock', np.sum(result['safety_stock'])))



@click.command()
@click.option('--index', default=1)
@click.option('--warmup', default=0)
@click.option('--ci', default=0.95)
@click.option('--ma', default=0)
@click.option('--l-mean', 'lead_mean', default=4)
@click.option('--l-std', 'lead_std', default=1)
def main(index, ci, warmup, ma, lead_mean, lead_std):
  weights = pd.read_csv('prediction/weights_l_p.csv')

  df_train = pd.read_csv('prediction/forecast_32_32.csv')
  df_train = df_train[df_train['ds'] < '2020-06-01']

  std = 0
  while not std > 0:
    l_code = weights.loc[index, 'l']
    p_code = weights.loc[index, 'p']
    std = get_rmse(df_train, l_code, p_code)
    index += 1

  print(index-1, l_code, p_code)
  
  df_test = pd.read_csv('prediction/forecast_0_32.csv')
  df_test = df_test[(df_test['location_code']==l_code) & (df_test['product_code']==p_code)]

  safety_stock = get_safety_stock(std, ci, df_test['yhat_0'][-32:].mean())

  review_period = 1
  y = df_test['y'][-(32+warmup):].values
  yhat = df_test.iloc[-(32+warmup):, -28:]
  result = simulate(y, yhat, safety_stock, lead_mean, lead_std, review_period, warmup, ma)

  print_result(result)
  metrics.get_all_metrics(result)


def apply_main(w_rate):
  ci = .95
  warmup = 8
  ma = 1
  weights = pd.read_csv('prediction/weights_l_p.csv')
  weights['sum'] = weights['r'].cumsum()/weights['r'].sum()
  weights = weights[weights['sum'] < w_rate]

  df_train = pd.read_csv('prediction/forecast_32_32.csv')
  df_train = df_train[df_train['ds'] < '2020-06-01']

  data = []

  index = 0
  while index < weights.index.max():
    std = 0
    while not std > 0:
      if index > weights.index.max():
        break
      l_code = weights.loc[index, 'l']
      p_code = weights.loc[index, 'p']
      std = get_rmse(df_train, l_code, p_code)
      index += 1
    
    df_test = pd.read_csv('prediction/forecast_0_32.csv')
    df_test = df_test[(df_test['location_code']==l_code) & (df_test['product_code']==p_code)]

    safety_stock = get_safety_stock(std, ci, df_test['yhat_0'][-32:].mean())

    lead_mean, lead_std, review_period = 4, 1, 1
    y = df_test['y'][-(32+warmup):].values
    yhat = df_test.iloc[-(32+warmup):, -28:]
    try:
      result = simulate(y, yhat, safety_stock, lead_mean, lead_std, review_period, warmup, ma)
      m, rmse, sl, cr, br = metrics.get_all_metrics(result)
      data.append([index-1, l_code, p_code, np.mean(result['y']), result['safety_stock'], m, rmse, sl, cr, br])
    except Exception as e:
      print(e)

  df = pd.DataFrame(data, columns=['w_index', 'l_code', 'p_code', 'mean_demand', 'safety_stock', 'mape', 'rmse', 'service_level', 'cycle_rate', 'bullwhip_rate'])
  print(df.head())
  df.to_csv('sim_results_4.csv')


if __name__ == '__main__':
  main()