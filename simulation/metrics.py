import numpy as np


def index_limit(*args):
  return min([len(i) for i in args])


def accuracy(y, yhat):
  m = np.mean(np.abs((yhat-y)/y)) * 100
  r = np.sqrt(np.mean((yhat - y)**2))
  print('MAPE: \t\t%d%%' % m)
  print('RMSE: \t\t%.1f' % r)
  return m, r


def service_level(y, unmet_demand):
  index = index_limit(y, unmet_demand)
  r = 1 - (np.sum(unmet_demand[:index]) / np.sum(y[:index]))
  print('Service level: \t%.2f' % r)
  return r


def cycle_rate(y, inv):
  index = index_limit(y, inv)
  r = np.sum(inv[:index]) / np.sum(y[:index])
  print('Cycle rate: \t%.2f' % r)
  return r


def bullwhip_rate(y, order_list):
  order_quantity = [i['quantity'] for i in order_list]
  r = (np.std(order_quantity) / np.mean(order_quantity)) / (np.std(y)/np.mean(y))
  print('Bullwhip rate: \t%.2f' % r)
  return r


def get_all_metrics(result):
  print('\nMetrics')
  print('-'*25)
  m, rmse = accuracy(result['y'], result['yhat'])
  sl = service_level(result['y'], result['unmet_demand'])
  cr = cycle_rate(result['y'], result['inv'])
  br = bullwhip_rate(result['y'], result['order_list'])

  return m, rmse, sl, cr, br