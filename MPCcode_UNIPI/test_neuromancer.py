# import pytest
import torch
import torch.nn as nn
import neuromancer.slim as slim
import pydot
import warnings
from neuromancer.system import Node, System
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.constraint import variable
from neuromancer.dataset import DictDataset
from neuromancer.loss import PenaltyLoss
from neuromancer.modules import blocks
from collections import defaultdict


torch.manual_seed(0)

def example_1():
    """
    define an example 'problem' set-up, e.g. that from
    https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/parametric_programming/Part_1_basics.ipynb
    This will be used to test problem class
    """
    func = blocks.MLP(insize=2, outsize=2,
                      bias=True,
                      linear_map=slim.maps['linear'],
                      nonlin=nn.ReLU,
                      hsizes=[80] * 4)
    # wrap neural net into symbolic representation of the solution map via the Node class: sol_map(xi) -> x
    sol_map = Node(func, ['a', 'p'], ['x'], name='map')
    # trainable components of the problem solution
    components = [sol_map]

    # define decision variables
    x1 = variable("x")[:, [0]]
    x2 = variable("x")[:, [1]]
    # problem parameters sampled in the dataset
    p = variable('p')
    a = variable('a')

    # objective function
    f = (1 - x1) ** 2 + a * (x2 - x1 ** 2) ** 2
    obj = f.minimize(weight=1.0, name='obj')

    # constraints
    Q_con = 100.  # constraint penalty weights
    con_1 = Q_con * (x1 >= x2)
    con_2 = Q_con * ((p / 2) ** 2 <= x1 ** 2 + x2 ** 2)
    con_3 = Q_con * (x1 ** 2 + x2 ** 2 <= p ** 2)
    con_1.name = 'c1'
    con_2.name = 'c2'
    con_3.name = 'c3'

    objectives = [obj]
    constraints = [con_1, con_2, con_3]
    loss = PenaltyLoss(objectives, constraints)

    edges = defaultdict(list,
                        {'in': ['map', 'map', 'obj', 'c2', 'c3'],
                         'map': ['obj', 'c1', 'c2', 'c3'],
                         'obj': ['out'],
                         'c1': ['out'],
                         'c2': ['out'],
                         'c3': ['out']})
    edges = dict(edges)

    return objectives, constraints, components, loss, edges

def test_problem_initialization():
    """
    Pytest testing function to check initialization of a problem, ensuring its class
    attributes are correct.
    """
    objectives, constraints, components, loss, _ = example_1()
    problem = Problem(components, loss, grad_inference=True, check_overwrite=True)

    # assert list_equals_modulelist(components, problem.nodes)
    assert problem.loss == loss
    assert problem.grad_inference == True
    assert problem.check_overwrite == True
    assert isinstance(problem, torch.nn.Module)

    problem = Problem(components, loss)

    # assert list_equals_modulelist(components, problem.nodes)
    assert problem.loss == loss
    assert isinstance(problem.grad_inference, bool)
    assert isinstance(problem.check_overwrite, bool)
    assert isinstance(problem, torch.nn.Module)

    print("Problem initialization test passed.")
