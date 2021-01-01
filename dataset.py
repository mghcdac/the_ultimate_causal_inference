import numpy as np
import pandas as pd
from scipy.stats import bernoulli
from scipy.special import expit as sigmoid
from causal_inference import addones


def create_simulated_linear_dataset(model_types, N=1000, D=3, with_mediator=False, coef_std=0.1, err_std=0.1, random_state=None):
    """
    create simulate data from linear models
    """
    np.random.seed(random_state)
    
    L = np.random.randn(N,D)

    # A|L
    betaA = np.abs(np.random.randn(D+1)*coef_std)
    zA = np.dot(addones(L), betaA)
    if model_types['A']=='logistic':
        A = bernoulli.rvs(sigmoid(zA+np.random.randn(N)*err_std), random_state=random_state)
    elif model_types['A']=='linear':
        A = zA+np.random.randn(N)*err_std
    else:
        raise ValueError(f'Unknown model type: {model_types["A"]}')

    if with_mediator:
        # M|A,L
        betaM = np.abs(np.random.randn(1+D+1)*coef_std)
        zM = np.dot(addones(np.c_[A,L]), betaM)
        if model_types['M']=='logistic':
            M = bernoulli.rvs(sigmoid(zM+np.random.randn(N)*err_std), random_state=random_state)
        elif model_types['M']=='linear':
            M = zM+np.random.randn(N)*err_std
        else:
            raise ValueError(f'Unknown model type: {model_types["M"]}')
    
        # Y|A,L,M
        betaY = np.abs(np.random.randn(1+D+1+1)*coef_std)
        zY = np.dot(addones(np.c_[A,L,M]), betaY)

    else:
        betaM = None
        M = None

        # Y|A,L
        betaY = np.abs(np.random.randn(1+D+1)*coef_std)
        zY = np.dot(addones(np.c_[A,L]), betaY)

    if model_types['Y']=='logistic':
        Y = bernoulli.rvs(sigmoid(zY+np.random.randn(N)*err_std), random_state=random_state)
    elif model_types['Y']=='linear':
        Y = zY+np.random.randn(N)*err_std
    else:
        raise ValueError(f'Unknown model type: {model_types["Y"]}')

    return L, A, M, Y, betaA, betaM, betaY


def create_real_dataset_framing():
    df = pd.read_excel('framing.xlsx')
    A = df['treat'].values.astype(float)

    df.loc[df.educ=='less than high school', 'educ'] = 0
    df.loc[df.educ=='high school', 'educ'] = 1
    df.loc[df.educ=='some college', 'educ'] = 2
    df.loc[df.educ=='bachelor\'s degree or higher', 'educ'] = 3
    df.loc[df.gender=='male', 'gender'] = 1
    df.loc[df.gender=='female', 'gender'] = 0
    L = df[['age', 'educ', 'gender', 'income']].values.astype(float)

    Y = df['immigr'].values.astype(float)

    Mnames = 'emo'#[, 'p_harm']
    df.loc[df.emo<8, 'emo'] = 0
    df.loc[df.emo>=8, 'emo'] = 1
    #df.loc[df.p_harm<7, 'p_harm'] = 0
    #df.loc[df.p_harm>=7, 'p_harm'] = 1
    M = df[Mnames].values.astype(float)

    return L, A, M, Y

