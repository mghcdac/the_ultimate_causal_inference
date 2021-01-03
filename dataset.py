import numpy as np
import pandas as pd
from scipy.stats import bernoulli
from scipy.special import expit as sigmoid
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from causal_inference import addones


def create_simulated_linear_dataset(N=1000, D=3, with_mediator=False, coef_std=0.1, err_std=0.1, random_state=None):
    """
    create simulate data from linear models
    """
    np.random.seed(random_state)
    
    L = np.random.randn(N,D)

    # A|L
    betaA = np.abs(np.random.randn(D+1)*coef_std)
    zA = np.dot(addones(L), betaA)
    A = bernoulli.rvs(sigmoid(zA), random_state=random_state)

    if with_mediator:
        # M|A,L
        betaM = np.abs(np.random.randn(1+D+1)*coef_std)
        zM = np.dot(addones(np.c_[A,L]), betaM)
        pM = sigmoid(zM)
        M = bernoulli.rvs(pM, random_state=random_state)
    
        # Y|A,L,M
        betaY = np.abs(np.random.randn(1+D+1+1)*coef_std)
        zY = np.dot(addones(np.c_[A,L,pM]), betaY) # use p(M) to get unbiased betaY_AL[0]
        Y = zY+np.random.randn(N)*err_std

        model = LinearRegression()
        model.fit(np.c_[A,L],zY)
        betaY_AL = np.r_[model.coef_.flatten()[0], betaY[1:D+1], betaY[-1]]

    else:
        betaM = None
        betaY_AL = None
        M = None

        # Y|A,L
        betaY = np.abs(np.random.randn(1+D+1)*coef_std)
        zY = np.dot(addones(np.c_[A,L]), betaY)
        Y = zY+np.random.randn(N)*err_std
        #Y = bernoulli.rvs(sigmoid(zY), random_state=random_state)

    return L, A, M, Y, betaA, betaM, betaY, betaY_AL


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


def create_real_dataset_hiv_ba():
    res = pd.read_excel('hiv-brain-age.xlsx')
    A = res['HIV'].values.astype(float)

    #res.loc[res.Sex=='M', 'Sex'] = 1
    #res.loc[res.Sex=='F', 'Sex'] = 0
    #race = OneHotEncoder(sparse=False).fit_transform(res.Race.values.astype(str).reshape(-1,1))
    #L = np.c_[res[['Age', 'Sex', 'Tobacco use disorder', 'Alcoholism']].values.astype(float), race]
    L = res[['Age', 'Sex', 'Tobacco use disorder', 'Alcoholism']].values.astype(float)

    Y = res['BAI'].values.astype(float)
    M = res['obesity'].values.astype(float)
    return L, A, M, Y

