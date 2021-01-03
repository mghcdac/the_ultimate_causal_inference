import numpy as np
from dataset import *
from causal_inference import *


if __name__=='__main__':
    variables = ['A', 'Y']
    methods = ['or', 'ipw', 'dr']
    model_types = {'A':'logistic', 'Y':'linear'}
    random_state = 100

    """
    L, A, M, Y = create_real_dataset_framing()
    #L, A, M, Y = create_real_dataset_hiv_ba()
    L = (L-L.mean(axis=0, keepdims=True))/L.std(axis=0, keepdims=True)
    M = None
    """

    """
    import dowhy.datasets
    data = dowhy.datasets.linear_dataset(
            beta=10,
            num_common_causes=5, 
            num_instruments = 2,
            num_treatments=1,
            num_samples=10000,
            treatment_is_binary=True,
            outcome_is_binary=False)
    L = data["df"][['Z0','Z1','W0','W1','W2','W3','W4']].values.astype(float)
    A = data["df"]['v0'].values.astype(float)
    Y = data["df"]['y'].values.astype(float)
    M = None
    """

    L, A, M, Y, betaA, betaM, betaY, betaY_AL = create_simulated_linear_dataset(
        N=1000, D=3, err_std=0.1, coef_std=1, with_mediator=False,
        random_state=random_state)
    for k in variables:
        print(f'beta{k} =', eval(f'beta{k}'))
    te_y = {}
    ya = {}
    for method in methods:
        func = eval(f'TE_{method.upper()}')
        args = (eval('predict_'+model_types['A']),
                None,
                eval('predict_'+model_types['Y']),
                L,A,M,Y,
                betaA,betaM,betaY)
        te_y[method], ya[method] = func(*args)
        print(f'TE Y {method.upper()}', te_y[method])

    lambda1 = 100
    #lambda2 = 10
    betaA2, betaY2, opt_message, ll, R = infer_params(
            model_types, L, A, M, Y, lambda1,# lambda2,
            n_iter=1, random_state=random_state)

    print()
    print('lambda1', lambda1)
    #print('lambda2', lambda2)
    print('opt_message', opt_message)
    print()

    for k in variables:
        print(f'beta{k}2 =', eval(f'beta{k}2'))
    te_y2 = {}
    ya2 = {}
    for method in methods:
        func = eval(f'TE_{method.upper()}')
        args = (eval('predict_'+model_types['A']),
                None,
                eval('predict_'+model_types['Y']),
                L,A,M,Y,
                betaA2,None,betaY2)
        te_y2[method], ya2[method] = func(*args)
        print(f'TE Y {method.upper()}2', te_y2[method])
    breakpoint()

