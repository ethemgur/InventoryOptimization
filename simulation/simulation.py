import matplotlib.pyplot as plt
import numpy as np


np.random.seed(42)


def adjust_y(forecast):
  return [max(0, i) for i in forecast]

def get_total_order(t, order_list, cumulative=False):
  '''
  If an order arrives at time t it returns the order;
  otherwise, it returns an order with zero quantity.
  '''

  total_quantity = 0

  for order in order_list:
    if cumulative:
      if t <= order['time']:
        total_quantity += order['quantity']
    else:
      if t == order['time']:
        total_quantity += order['quantity']
  
  return total_quantity


def create_lead_time(lead_mean, lead_std):
  '''
  Creates a lead time with the given distribution.
  '''

  return round(np.random.normal(lead_mean, lead_std))


def create_order(new_inv_net, yhat, t, safety_stock, review_period, lead_time):
  '''
  Order
    submission:  submission time of the order
    time      :  arrival time of the order
    quantity  :  amount of the order
  '''

  order_time = t + lead_time + review_period
  order_quantity = sum(yhat[t+1:order_time+1]) + safety_stock - new_inv_net
  new_order = {'submission': t, 'time': order_time, 'quantity': order_quantity}

  return new_order


def get_initial_inv(yhat, safety_stock, lead_mean, review_period):
  return sum(yhat[:lead_mean + review_period + 1]) + safety_stock


def simulate(y, yhats, safety_stock, lead_mean, lead_std, review_period, warmup, ma):
  '''
  inventory     :  level of inventory at the end of each week
  net inventory :  inventory + upcoming orders
  '''

  #yhat = adjust_y(yhat)

  inv = []
  inv_net = []
  unmet_demand = []
  order_list = []

  initial_inv = get_initial_inv(yhats['yhat_0'].values, safety_stock, lead_mean, review_period)

  for t in range(28):
    arriving_order = get_total_order(t, order_list)
    total_order = get_total_order(t, order_list, cumulative=True)
    yhat = yhats['yhat_{}'.format(str(t))]
    if ma > 0:
      if t >= ma:
        yhat[t:] += np.mean(y[t-ma:t] - yhat[t-ma:t])

    if t == 0:
      new_inv = initial_inv - y[t] + arriving_order
      new_inv_net = initial_inv - y[t] + total_order
    else:
      new_inv = inv[t-1] - y[t] + arriving_order
      new_inv_net = inv[t-1] - y[t] + total_order

    lead_time = create_lead_time(lead_mean, lead_std)
    # max(lead_mean, lead_time)
    new_order = create_order(new_inv_net, yhat, t, safety_stock, review_period, lead_time)
    if new_order['quantity'] > 0:
      order_list.append(new_order)
      new_inv_net += new_order['quantity']

    inv.append(max(0, new_inv))
    inv_net.append(max(0, new_inv_net))
    unmet_demand.append(abs(min(0, new_inv)))

  fig, axes = plt.subplots(nrows=2, ncols=1)
  axes[0].plot(inv, label='inventory', c='blue')
  axes[0].plot(inv_net, label='net inventory', c='orange')
  axes[0].plot(unmet_demand, label='unmet demand', c='red')
  axes[0].axhline(safety_stock, c='grey', linestyle='--')
  axes[0].axvline(warmup, c='magenta', linestyle='--')

  #for order in order_list:
  #  axes[0].scatter(order['submission'], order['quantity'], c='black', marker='^')
  #axes[0].legend()
  os_s = [i['submission'] for i in order_list]
  os_q = [i['quantity'] for i in order_list]
  axes[0].scatter(os_s, os_q, c='black', marker='^', label='Order')
  axes[0].legend()

  axes[1].plot(y, label='y', c='black')
  axes[1].scatter(range(len(yhat)), yhat, label='yhat', c='green', marker='x')
  axes[1].legend()
  plt.savefig("fig.pdf")
  
  result = {
    'inv': inv[warmup:],
    'inv_net': inv_net[warmup:],
    'unmet_demand': unmet_demand[warmup:],
    'order_list': [i for i in order_list if i['submission'] >= warmup],
    'y': y[warmup:],
    'yhat': yhat[warmup:],
    'safety_stock': safety_stock
  }

  return result
