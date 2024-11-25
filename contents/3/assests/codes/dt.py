import numpy as np
import pandas as pd

def gini(S):
    N = len(S)
    y = S[:, -1].reshape(N)
    gini = 1 - ((pd.Series(y).value_counts()/N)**2).sum()
    return gini


def split(G):
    m = G.shape[0]
    gmini = gini(G)
    pair = None
    if gini(G) != 0:
        numOffeatures = G.shape[1] - 1
        for k in range(numOffeatures):
            for t in range(m):
                Gl = G[G[:, k] <= G[t, k]]
                Gr = G[G[:, k] > G[t, k]]
                gl = gini(Gl)
                gr = gini(Gr)
                ml = Gl.shape[0]
                mr = Gr.shape[0]
                g = gl*ml/m + gr*mr/m
                if g < gmini:
                    gmini = g
                    pair = (k, G[t, k])
                    Glm = Gl
                    Grm = Gr
        res = {'split': True,
               'pair': pair,
               'sets': (Glm, Grm)}
    else:
        res = {'split': False,
               'pair': pair,
               'sets': G}
    return res


def countlabels(S):
    y = S[:, -1].reshape(S.shape[0])
    labelCount = dict(pd.Series(y).value_counts())
    return labelCount