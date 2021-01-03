from itertools import product
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit as sigmoid
#from sklearn.linear_model._logistic import _logistic_loss_and_grad
from tqdm import tqdm


Alevels = [0,1]
Mlevels = [0,1]


def addones(X):
    return np.c_[X, np.ones(len(X))]


def predict_logistic(X, beta, proba=False, logit=False):
    y = np.dot(addones(X),beta)
    if not logit:
        if proba:
            y = sigmoid(y)
        else:
            y = (y>0).astype(float)
    return y


def predict_linear(X, beta, proba=False, logit=False):
    y = np.dot(addones(X),beta)
    return y


def cv2(x):
    """
    squared coefficient of variance, or efficiency
    """
    return np.var(x)/(np.mean(x)**2)


def _ll_logistic(y, logit_yp):
    logsig = -np.logaddexp(-logit_yp,0)
    #logsig1m = logsig-logit_yp
    #ll = y*logsig + (1-y)*logsig1m
    ll = logsig-(1-y)*logit_yp
    return ll


def _ll_linear(y, yp):
    ll = -(y-yp)**2
    return ll


def TE_OR(A_func, M_func, Y_func, L, A, M, Y, betaA, betaM, betaY):
    N = len(L)
    ya = {}
    for a in Alevels:
        ya[a] = Y_func(np.c_[np.zeros(N)+a,L],betaY,proba=True)

    te = ya[1].mean() - ya[0].mean()
    return te, ya


def TE_IPW(A_func, M_func, Y_func, L, A, M, Y, betaA, betaM, betaY):
    N = len(L)
    Ap = A_func(L, betaA, proba=True)
    Ap = np.c_[1-Ap, Ap]
    pw = Ap[range(N),A.astype(int)]
    pw = np.maximum(pw, 1e-5)
    Y2 = Y/pw

    ya = {}
    for a in Alevels:
        ya[a] = Y2*((A==a).astype(float))

    te = ya[1].mean() - ya[0].mean()
    return te, ya


def TE_DR(A_func, M_func, Y_func, L, A, M, Y, betaA, betaM, betaY):
    N = len(L)
    Ap = A_func(L, betaA, proba=True)
    Ap = np.c_[1-Ap, Ap]
    pw = Ap[range(N),A.astype(int)]
    pw = np.maximum(pw, 1e-5)

    ya = {}
    for a in Alevels:
        yp = Y_func(np.c_[np.zeros(N)+a,L],betaY,proba=True)
        ya[a] = (Y- yp)/pw*((A==a).astype(float)) + yp

    te = ya[1].mean() - ya[0].mean()
    return te, ya


def TE_MED(A_func, M_func, Y_func, L, A, M, Y, betaA, betaM, betaY):
    """
    Y(a) = Y(a,M(a))
         = Y(a,1)M(a) + Y(a,0)(1-M(a))
         = (Y(a,1)-Y(a,0))M(a) + Y(a,0)
    Y(1) = betaY_m M(1) + Y(1,0)
    Y(0) = betaY_m M(0) + Y(0,0)
    Y(1)-Y(0) = betaY_m (M(1)-M(0)) + betaY_a
    """
    N = len(L)
    Ap = A_func(L, betaA, proba=True)
    Ap = np.c_[1-Ap, Ap]
    Mp = M_func(np.c_[A,L], betaM, proba=True)
    Mp = np.c_[1-Mp, Mp]
    pw = Ap[range(N),A.astype(int)]*Mp[range(N),M.astype(int)]
    pw = np.maximum(pw, 1e-5)

    yam = {}
    for a,m in product(Alevels, Mlevels):
        yp = Y_func(np.c_[np.zeros(N)+a,L,np.zeros(N)+m],betaY,proba=True)
        yam[(a,m)] = np.mean((Y-yp)/pw*((A==a).astype(float))*((M==m).astype(float))+yp)

    pw = Ap[range(N),A.astype(int)]
    pw = np.maximum(pw, 1e-5)
    ma = {}
    ya = {}
    for a in Alevels:
        Mp = M_func(np.c_[np.zeros(N)+a,L],betaM,proba=True)
        ma[a] = ((M- Mp)/pw*((A==a).astype(float)) + Mp).mean()
        ya[a] = yam[(a,1)]*ma[a] + yam[(a,0)]*(1-ma[a])

    te = ya[1] - ya[0]
    return te, ya


def _loss_without_mediator(x, predict_funcs, ll_funcs, L, A, Y, lambda1, return_components=False):#, lambda2
    N, D = L.shape
    betaA = x[:D+1]
    betaY = x[D+1:]

    # loglikelihood

    zA = predict_funcs['A'](L, betaA, logit=True)
    zY = predict_funcs['Y'](np.c_[A,L], betaY, logit=True)
    neg_ll = -sum([
            ll_funcs['A'](A, zA),
            ll_funcs['Y'](Y,zY), ])

    # different TE estimates

    args = (predict_funcs['A'], None, predict_funcs['Y'],
            L,A,None,Y,
            betaA,None,betaY)
    #te_or, ya_or = TE_OR(*args)
    te_ipw, ya_ipw = TE_IPW(*args)
    te_dr, ya_dr = TE_DR(*args)
    R = cv2([te_ipw, te_dr])#, te_or])

    """
    # consistency constraints

    #CY_or = np.mean([( ya_or[a][A==a].mean() - Y[A==a].mean() )**2 for a in Alevels])
    #CY_ipw = np.mean([( ya_ipw[a][A==a].mean() - Y[A==a].mean() )**2 for a in Alevels])
    CY_dr = np.mean([( ya_dr[a][A==a].mean() - Y[A==a].mean() )**2 for a in Alevels])
    C = np.mean([CY_dr])#, CY_or])#, CY_ipw])
    """

    if return_components:
        return neg_ll.mean(), R#, C
    else:
        return neg_ll.mean()+lambda1*R#+lambda2*C


def _loss_with_mediator(x, predict_funcs, ll_funcs, L, A, Y, M, lambda1, return_components=False):#, lambda2
    N, D = L.shape
    betaA = x[:D+1]
    betaY = x[D+1:D+1+1+D+1+1]
    betaY_AL = x[D+1+1+D+1+1:D+1+1+D+1+1+1+D+1]
    betaM = x[D+1+1+D+1+1+1+D+1:]
    
    # loglikelihood

    zA = predict_funcs['A'](L, betaA, logit=True)
    zY = predict_funcs['Y'](np.c_[A,L,M], betaY, logit=True)
    zY_AL = predict_funcs['Y'](np.c_[A,L], betaY_AL, logit=True)
    zM = predict_funcs['M'](np.c_[A,L], betaM, logit=True)
    neg_ll = -sum([
            ll_funcs['A'](A, zA),
            ll_funcs['Y'](Y,zY),
            ll_funcs['Y'](Y,zY_AL),
            ll_funcs['M'](M,zM), ])

    # different TE estimates

    args = (predict_funcs['A'], predict_funcs['M'], predict_funcs['Y'],
            L,A,M,Y,
            betaA,betaM,betaY_AL)
    #te_y_or, ya_or = TE_OR(*args)
    te_y_ipw, ya_ipw = TE_IPW(*args)
    te_y_dr, ya_dr = TE_DR(*args)
    args = (predict_funcs['A'], predict_funcs['M'], predict_funcs['Y'],
            L,A,M,Y,
            betaA,betaM,betaY)
    te_y_med, ya_med = TE_MED(*args)
    RY = cv2([te_y_dr, te_y_med, te_y_ipw])#, te_y_or

    args = (predict_funcs['A'], None, predict_funcs['M'],
            L,A,None,M,
            betaA,None,betaM)
    #te_m_or, ma_or = TE_OR(*args)
    te_m_ipw, ma_ipw = TE_IPW(*args)
    te_m_dr, ma_dr = TE_DR(*args)
    RM = cv2([te_m_dr, te_m_ipw])#, te_m_or

    R = RY+RM

    """
    # consistency constraints

    #CY_or = np.mean([( ya_or[a][A==a].mean() - Y[A==a].mean() )**2 for a in Alevels])
    #CY_ipw = np.mean([( ya_ipw[a][A==a].mean() - Y[A==a].mean() )**2 for a in Alevels])
    CY_dr = np.mean([( ya_dr[a][A==a].mean() - Y[A==a].mean() )**2 for a in Alevels])
    CY_med = np.mean([( ya_med[a][A==a].mean() - Y[A==a].mean() )**2 for a in Alevels])
    CY = np.mean([CY_dr, CY_med])#, CY_ipw]), CY_or

    #CM_or = np.mean([( ma_or[a][A==a].mean() - M[A==a].mean() )**2 for a in Alevels])
    #CM_ipw = np.mean([( ma_ipw[a][A==a].mean() - M[A==a].mean() )**2 for a in Alevels])
    CM_dr = np.mean([( ma_dr[a][A==a].mean() - M[A==a].mean() )**2 for a in Alevels])
    CM = np.mean([CM_dr])#, CM_or])#, CM_ipw])

    C = CY+CM
    """

    if return_components:
        return neg_ll.mean(), R#, C
    else:
        return neg_ll.mean()+lambda1*R#+lambda2*C


def infer_params(model_types, L, A, M, Y, lambda1, n_iter=10, max_iter=1000, verbose=True, random_state=None):#, lambda2
    """
    """
    # some preprocessing
    N, D = L.shape
    with_mediator = M is not None
    assert N==len(A)==len(Y)
    if with_mediator:
        assert N==len(M)
        
    ll_funcs = {
            'A': eval(f'_ll_{model_types["A"]}'),
            'Y': eval(f'_ll_{model_types["Y"]}'),
            }
    predict_funcs = {
            'A': eval(f'predict_{model_types["A"]}'),
            'Y': eval(f'predict_{model_types["Y"]}'),
            }
    if with_mediator:
        ll_funcs['M'] = eval(f'_ll_{model_types["M"]}')
        predict_funcs['M'] = eval(f'predict_{model_types["M"]}')

    # find parameters
    # since it is not convex, run multiple times
    np.random.seed(random_state)
    opt_ress = []
    for _ in tqdm(range(n_iter), disable=not verbose):
        if with_mediator:
            x0 = np.r_[
                    np.random.randn(D+1)/10, # A|L
                    np.random.randn(1+D+1+1)/10, #Y|A,L,M
                    np.random.randn(1+D+1)/10, #Y|A,L
                    np.random.randn(1+D+1)/10, #M|A,L
                    ]
            opt_res = minimize(
                _loss_with_mediator, x0,
                args=(predict_funcs, ll_funcs, L, A, Y, M, lambda1),# lambda2),
                method=None, jac=None,
                options=dict(maxiter=max_iter, disp=False))

        else:
            x0 = np.r_[
                    np.random.randn(D+1)/10, # A|L
                    np.random.randn(1+D+1)/10, #Y|A,L
                    ]
            opt_res = minimize(
                _loss_without_mediator, x0,
                args=(predict_funcs, ll_funcs, L, A, Y, lambda1),# lambda2),
                method=None, jac=None,
                options=dict(maxiter=max_iter, disp=False))
        opt_ress.append(opt_res)

    func_vals = [x.fun for x in opt_ress]
    opt_res = opt_ress[np.argmin(func_vals)]

    if with_mediator:
        betaA = opt_res.x[:D+1]
        betaY = opt_res.x[D+1:D+1+1+D+1+1]
        betaY_AL = opt_res.x[D+1+1+D+1+1:D+1+1+D+1+1+1+D+1]
        betaM = opt_res.x[D+1+1+D+1+1+1+D+1:]
        ll, R = _loss_with_mediator(opt_res.x, predict_funcs, ll_funcs, L, A, Y, M, lambda1, return_components=True)#, lambda2
        return betaA, betaY, betaY_AL, betaM, opt_res.message, ll, R#, C

    else:
        betaA = opt_res.x[:D+1]
        betaY = opt_res.x[D+1:]
        ll, R = _loss_without_mediator(opt_res.x, predict_funcs, ll_funcs, L, A, Y, lambda1, return_components=True)#, lambda2
        return betaA, betaY, opt_res.message, ll, R#, C
    
