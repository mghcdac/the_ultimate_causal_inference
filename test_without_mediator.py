import numpy as np
from dataset import *
from causal_inference import *


if __name__=='__main__':
    variables = ['A', 'Y']
    methods = ['or', 'ipw', 'dr']
    model_types = {'A':'logistic', 'Y':'linear'}
    random_state = 2020

    L, A, M, Y = create_real_dataset_framing()
    L = (L-L.mean(axis=0, keepdims=True))/L.std(axis=0, keepdims=True)
    M = None
    """
    L, A, M, Y, betaA, betaM, betaY = create_simulated_linear_dataset(model_types, N=1000, D=10, err_std=0.1, with_mediator=False, random_state=random_state)
    for k in variables:
        print(f'beta{k} =', eval(f'beta{k}'))
    te_y = {}
    ya = {}
    for method in methods:
        func = eval(f'TE_Y_{method.upper()}')
        te_y[method], ya[method] = func(L,A,Y,betaA,betaY)
        print(f'TE Y {method.upper()}', te_y[method])

    """
    lambda1 = 1
    lambda2 = 1
    betaA2, betaY2, betaM2, opt_message, ll, R, C = infer_params(model_types, L, A, M, Y, lambda1, lambda2, random_state=random_state)

    print()
    print('lambda1', lambda1)
    print('lambda2', lambda2)
    print('opt_message', opt_message)
    print()

    for k in variables:
        print(f'beta{k}2 =', eval(f'beta{k}2'))
    te_y2 = {}
    ya2 = {}
    for method in methods:
        func = eval(f'TE_Y_{method.upper()}')
        te_y2[method], ya2[method] = func(L,A,Y,betaA2,betaY2)
        print(f'TE Y {method.upper()}2', te_y2[method])
    breakpoint()

