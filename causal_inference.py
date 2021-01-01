import numpy as np
from scipy.optimize import minimize
from scipy.special import expit as sigmoid
#from sklearn.linear_model._logistic import _logistic_loss_and_grad


def addones(X):
    return np.c_[X, np.ones(len(X))]


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


def TE_Y_OR(L, A, Y, betaA, betaY):
    N = len(L)
    ya = {}
    for a in [0,1]:
        ya[a] = np.dot(addones(np.c_[np.zeros(N)+a,L]),betaY)
    te = ya[1].mean() - ya[0].mean()
    return te, ya


def TE_Y_IPW(L, A, Y, betaA, betaY):
    N = len(L)
    Ap = sigmoid(np.dot(addones(L),betaA))
    Ap = np.c_[1-Ap, Ap]
    pw = Ap[range(N),A.astype(int)]
    Y2 = Y/pw

    ya = {}
    for a in [0,1]:
        ya[a] = Y2*((A==a).astype(float))
    te = ya[1].mean() - ya[0].mean()
    return te, ya


def TE_Y_DR(L, A, Y, betaA, betaY):
    N = len(L)
    Ap = sigmoid(np.dot(addones(L),betaA))
    Ap = np.c_[1-Ap, Ap]
    pw = Ap[range(N),A.astype(int)]

    ya = {}
    for a in [0,1]:
        yp = np.dot(addones(np.c_[np.zeros(N)+a,L]),betaY)
        ya[a] = (Y - yp)/pw*((A==a).astype(float)) + yp
    te = ya[1].mean() - ya[0].mean()
    return te, ya


def _loss_without_mediator(x, A_ll_func, Y_ll_func, L, A, Y, lambda1, lambda2, return_components=False):
    N, D = L.shape
    betaA = x[:D+1]
    betaY = x[D+1:]
    
    zA = np.dot(addones(L), betaA)
    zY = np.dot(addones(np.c_[A,L]), betaY)

    neg_ll = -A_ll_func(A, zA)-Y_ll_func(Y,zY)

    te_or, ya_or = TE_Y_OR(L,A,Y,betaA,betaY)
    te_ipw, ya_ipw = TE_Y_IPW(L,A,Y,betaA,betaY)
    te_dr, ya_dr = TE_Y_DR(L,A,Y,betaA,betaY)
    RY = cv2([te_or, te_ipw, te_dr])
    R = RY

    CY_or = np.mean([( ya_or[a][A==a].mean() - Y[A==a].mean() )**2 for a in [0,1]])
    #CY_ipw = np.mean([( ya_ipw[a][A==a].mean() - Y[A==a].mean() )**2 for a in [0,1]])
    CY_dr = np.mean([( ya_dr[a][A==a].mean() - Y[A==a].mean() )**2 for a in [0,1]])
    C = np.mean([CY_dr, CY_or])#, CY_ipw])

    if return_components:
        return neg_ll.mean(), R, C
    else:
        return neg_ll.mean()+lambda1*R+lambda2*C


#def _loss_with_mediator():



def infer_params(model_types, L, A, M, Y, lambda1, lambda2, max_iter=1000, verbose=False, random_state=None):
    """
    """
    # some preprocessing
    N, D = L.shape
    with_mediator = M is not None
    assert N==len(A)==len(Y)
    if with_mediator:
        assert N==len(M)
        M_ll_func = eval(f'_ll_{model_types["M"]}')
        
    A_ll_func = eval(f'_ll_{model_types["A"]}')
    Y_ll_func = eval(f'_ll_{model_types["Y"]}')

    np.random.seed(random_state)

    # find parameters
    if with_mediator:
        x0 = np.random.randn(D+1 + 1+D+1 + 1+D+1+1)/10
        opt_res = minimize(
            _loss_with_mediator, x0,
            args=(A_ll_func, M_ll_func, Y_ll_func, L, A, M, Y, lambda1, lambda2),
            method=None, jac=None,
            options=dict(maxiter=max_iter, disp=verbose))
        betaA = opt_res.x[:D+1]
        betaY = opt_res.x[D+1:D+1+1+D+1+1]
        betaM = opt_res.x[D+1+1+D+1+1:]
        ll, R, C =_loss_with_mediator(opt_res.x, A_ll_func, M_ll_func, Y_ll_func, L, A, M, Y, lambda1, lambda2, return_components=True)
    else:
        x0 = np.random.randn(D+1 + 1+D+1)/10
        opt_res = minimize(
            _loss_without_mediator, x0,
            args=(A_ll_func, Y_ll_func, L, A, Y, lambda1, lambda2),
            method=None, jac=None,
            options=dict(maxiter=max_iter, disp=verbose))
        betaA = opt_res.x[:D+1]
        betaY = opt_res.x[D+1:]
        betaM = None
        ll, R, C =_loss_without_mediator(opt_res.x, A_ll_func, Y_ll_func, L, A, Y, lambda1, lambda2, return_components=True)

    return betaA, betaY, betaM, opt_res.message, ll, R, C

