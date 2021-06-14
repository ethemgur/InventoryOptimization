import pytest
from pytest_mock import mocker
from simulation import *

import numpy as np


@pytest.fixture
def inv1():
  return [30]


@pytest.fixture
def order_list():
  return [
    {'time': 4, 'quantity': 7},
    {'time': 4, 'quantity': 11},
    {'time': 5, 'quantity': 13},
    {'time': 6, 'quantity': 17}
    ]


@pytest.fixture
def y():
  return [8]*32
  #return [14, 1, 3, 7, 5, 2, 3, 7, 3, 6, 2, 2]
  #      [16,15,12, 5, 0, 0]
  #      [35,34,28,28,26,29]
  #      [35,34,31,28,26,29]
  #      [19, 0, 0, 0, 0, 0]


@pytest.fixture
def yhat():
  return [8]*32
  return [11, 3, 7, 5, 9, 1, 2, 1, 5, 7, 4, 2]


@pytest.fixture
def safety_stock():
  return 0


@pytest.fixture
def review_period():
  return 1


@pytest.fixture
def lead_mean():
  return 3


def test_create_order(safety_stock, review_period, lead_mean):
  new_order = create_order(16, [3, 4, 5, 6, 7, 8, 9], 1, safety_stock, review_period, lead_mean)
  assert new_order == {'submission': 0, 'time': 5, 'quantity': 19 + safety_stock - 16}


@pytest.mark.parametrize('t, result, cumulative', [(4, 18, False), (5, 30, True), (2, 0, False)])
def test_get_total_order(t, result, cumulative, order_list):
  total_order = get_total_order(t, order_list, cumulative)
  assert total_order == result


def test_get_initial_inv(yhat, safety_stock, lead_mean, review_period):
  initial_inv = get_initial_inv(yhat, safety_stock, lead_mean, review_period)
  assert initial_inv == sum(yhat[:lead_mean+review_period]) + safety_stock


def test_simulate(y, yhat, safety_stock, lead_mean, review_period, mocker):
  #mocker.patch('simulation.get_initial_inv', return_value=30)
  mocker.patch('simulation.create_lead_time', return_value=4)
  inv, inv_net, unmet_demand = simulate(y, yhat, safety_stock, lead_mean, 0, review_period)

  #assert inv_net == [35, 34, 31, 28, 26, 29, 29]
  assert inv_net == [32]*13
  