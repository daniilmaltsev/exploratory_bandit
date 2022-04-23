import pandas as pd
import numpy as np
import math
from collections import Sequence

def entropy_gradient(n, cvr):
    pip = cvr * (1 - cvr)
    eg = pip / (2 * np.sqrt(n * pip))
    return eg

def get_delta(cvr):
    """
    Calculate the distance from protagonist cvr to the leader.
    Replace the leader's zero distance with the distance between the leader and it's closest rival (i.e. silver).
    """
    leader = np.argmax(cvr)
    cvr_leader = cvr[leader]
    delta = cvr_leader - cvr
    silver = cvr[cvr < cvr_leader].argmax()
    delta[leader] = delta[silver]
    return delta

def get_info_gain(delta, entropy_gradient):
    ig = entropy_gradient / delta
    return ig

def allocate(df):
    df['entropy_gradient'] = entropy_gradient(n=df['n_users'], cvr=df['cvr'])
    df['delta'] = delta(df['cvr'])
    df['info_gain'] = info_gain(df['delta'], df['entropy_gradient'])
    df['allocation'] = df['info_gain'] / df['info_gain'].sum()
    return df

def minimize_entropy(cvr, n_users):
    eg = entropy_gradient(n=n_users, cvr=cvr)
    delta = get_delta(cvr)
    info_gain = get_info_gain(delta, eg)
    allocation = info_gain / info_gain.sum()
    return np.array(allocation)

def sample_conversions(cvr, n_users, resolution):
    alphas = cvr * n_users
    betas = n_users - alphas
    if isinstance(alphas, Sequence) or isinstance(alphas, pd.Series):
        shape = (resolution, len(alphas))
    else:
        shape = resolution
    scvrs = np.random.beta(alphas, betas, shape)
    return scvrs

def minimize_loss(cvr, n_users, resolution=50000):
    """
    SWAG UNDER CONSTRUCTION
    """
    scvrs = sample_conversions(cvr, n_users, resolution)
    
    leader = np.argmax(cvr)
    silver = cvr[cvr < cvr_leader].argmax()
    leader_scvr = scvrs[:,leader]
    silver_scvr = scvrs[:, silver]
    deltas = scvrs - leader_scvr.reshape(resolution, 1)

    winner_cvr = scvrs.max(axis=1).reshape(resolution, 1)
    winner_matrix = (winner_cvr == scvrs)
    deltas[deltas != deltas[winner_matrix].reshape(resolution, 1)] = 0
    loss_contribution = deltas.mean(axis=0)
    
    # SWAG: the leader's loss contribution is 1/2 of loss in all lost clashes
    leader_loss_contribution = loss_contribution.sum()
    loss_contribution[leader] = leader_loss_contribution
    allocation = loss_contribution / loss_contribution.sum()
    return allocation

def batch_thompson(cvr, n_users, resolution=50000):
    """
    A batch version of Thompson sampling for slow feedback loops.
    Use at your own risk.
    """
    alphas = cvr * n_users
    betas = n_users - alphas
    scvrs = np.random.beta(alphas, betas, (resolution, len(alphas)))
    winner_cvr = scvrs.max(axis=1).reshape(resolution, 1)
    winner_matrix = (winner_cvr == scvrs)
    allocation = winner_matrix.mean(axis=0)
    return allocation

def get_loss_gradient(cvr, n_users, batch_size=100, resolution=100000):
    scvrs = sample_conversions(cvr, n_users, resolution)
    leader = np.argmax(cvr)
    winner_cvr = scvrs.max(axis=1).reshape(resolution, 1)
    deltas = winner_cvr - scvrs
    base_loss = deltas[:, leader].mean()

    loss_gradients = {}
    cnt = 0
    for i in n_users.index:
        tweaked_users = n_users[i] + batch_size
        tweaked_scvrs = scvrs.copy()
        tweaked_scvrs[:, cnt] = sample_conversions(cvr[i], tweaked_users, resolution).ravel()
        leader = np.argmax(cvr)
        winner_cvr = tweaked_scvrs.max(axis=1).reshape(resolution, 1)
        deltas = winner_cvr - tweaked_scvrs
        loss = deltas[:, leader].mean()
        loss_gradients[i] = (base_loss - loss) / base_loss
        cnt += 1
    loss_gradients = pd.Series(loss_gradients)
    loss_gradients = loss_gradients.clip(0, None)
    norma_loss_gradients = loss_gradients / loss_gradients.sum()
    return np.array(norma_loss_gradients)
