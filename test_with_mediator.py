import numpy as np
from dataset import *
from causal_inference import *


if __name__=='__main__':
    variables = ['A', 'Y', 'Y_AL', 'M']
    methods = ['or', 'ipw', 'dr', 'med']
    model_types = {'A':'logistic', 'M':'logistic', 'Y':'linear'}
    random_state = 100

    """
    #L, A, M, Y = create_real_dataset_framing()
    L, A, M, Y = create_real_dataset_hiv_ba()
    L = (L-L.mean(axis=0, keepdims=True))/L.std(axis=0, keepdims=True)
    """
    L, A, M, Y, betaA, betaM, betaY, betaY_AL = create_simulated_linear_dataset(
        N=1000, D=3, err_std=0.1, coef_std=1, with_mediator=True,
        random_state=random_state)
    for k in variables:
        print(f'beta{k} =', eval(f'beta{k}'))
    te_m = {}
    ma = {}
    for method in methods:
        if method=='med':
            continue
        func = eval(f'TE_{method.upper()}')
        args = (eval('predict_'+model_types['A']),
                None,
                eval('predict_'+model_types['M']),
                L,A,None,M,
                betaA,None,betaM)
        te_m[method], ma[method] = func(*args)
        print(f'TE M {method.upper()}', te_m[method])
    te_y = {}
    ya = {}
    for method in methods:
        func = eval(f'TE_{method.upper()}')
        args = (eval('predict_'+model_types['A']),
                eval('predict_'+model_types['M']),
                eval('predict_'+model_types['Y']),
                L,A,M,Y,
                betaA,betaM)
        if method=='med':
            args = args+(betaY,)
        else:
            args = args+(betaY_AL,)
        te_y[method], ya[method] = func(*args)
        print(f'TE Y {method.upper()}', te_y[method])

    lambda1 = 100
    #lambda2 = 10
    betaA2, betaY2, betaY_AL2, betaM2, opt_message, ll, R = infer_params(
            model_types, L, A, M, Y, lambda1,# lambda2,
            n_iter=1, random_state=random_state)

    print()
    print('lambda1', lambda1)
    #print('lambda2', lambda2)
    print('opt_message', opt_message)
    print()

    for k in variables:
        print(f'beta{k}2 =', eval(f'beta{k}2'))
    te_m2 = {}
    ma2 = {}
    for method in methods:
        if method=='med':
            continue
        func = eval(f'TE_{method.upper()}')
        args = (eval('predict_'+model_types['A']),
                None,
                eval('predict_'+model_types['M']),
                L,A,None,M,
                betaA2,None,betaM2)
        te_m2[method], ma2[method] = func(*args)
        print(f'TE M {method.upper()}2', te_m2[method])
    te_y2 = {}
    ya2 = {}
    for method in methods:
        func = eval(f'TE_{method.upper()}')
        args = (eval('predict_'+model_types['A']),
                eval('predict_'+model_types['M']),
                eval('predict_'+model_types['Y']),
                L,A,M,Y,
                betaA2,betaM2)
        if method=='med':
            args = args+(betaY2,)
        else:
            args = args+(betaY_AL2,)
        te_y2[method], ya2[method] = func(*args)
        print(f'TE Y {method.upper()}2', te_y2[method])
    breakpoint()

