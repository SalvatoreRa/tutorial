# -*- coding: utf-8 -*-
from numpy import array, random, arange
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

def xicor(X, Y, ties=True):
    '''
    Calculate the correlation coefficient between two NumPy arrays.
    based on this article: https://www.tandfonline.com/doi/full/10.1080/01621459.2020.1758115
    
    Parameters:
    x (np.array): First input array.
    y (np.array): Second input array.
    ties (Bool); if there are ties
    '''
    random.seed(42)
    n = len(X)
    order = array([i[0] for i in sorted(enumerate(X), key=lambda x: x[1])])
    if ties:
        l = array([sum(y >= Y[order]) for y in Y[order]])
        r = l.copy()
        for j in range(n):
            if sum([r[j] == r[i] for i in range(n)]) > 1:
                tie_index = array([r[j] == r[i] for i in range(n)])
                r[tie_index] = random.choice(r[tie_index] - arange(0, sum([r[j] == r[i] for i in range(n)])), sum(tie_index), replace=False)
        return 1 - n*sum( abs(r[1:] - r[:n-1]) ) / (2*sum(l*(n - l)))
    else:
        r = array([sum(y >= Y[order]) for y in Y[order]])
        return 1 - 3 * sum( abs(r[1:] - r[:n-1]) ) / (n**2 - 1)

    
def kendall_tau_correlation(x, y):
    """
    Calculate the Kendall Tau correlation coefficient
    
    Parameters:
    x (np.array): First input array.
    y (np.array): Second input array.
    
    Returns:
    float: Kendall Tau correlation coefficient.
    """
    n = len(x)
    if n != len(y):
        raise ValueError("Both arrays must be of the same length")
    
    index_pairs = np.array(np.triu_indices(n, 1)).T
    
    concordant = 0
    discordant = 0
    
    for i, j in index_pairs:
        dx = np.sign(x[i] - x[j])
        dy = np.sign(y[i] - y[j])
        prod = dx * dy
        
        if prod > 0:
            concordant += 1
        elif prod < 0:
            discordant += 1
    
    coeff = (2 * (concordant - discordant)) / (n * (n - 1))
    return coeff



def spearman_rank_correlation(x, y):
    """
    Calculate the Spearman rank correlation coefficient.
    
    Parameters:
    x (np.array): First input array.
    y (np.array): Second input array.
    
    Returns:
    float: Spearman rank correlation coefficient.
    """
    if len(x) != len(y):
        raise ValueError("Both arrays must be of the same length")
    
    
    def rank(data):
        sorted_indices = np.argsort(data)
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(len(data)) + 1
        # Handle ties by averaging the ranks
        for val in np.unique(data):
            tie_indices = np.where(data == val)
            tie_rank = np.mean(ranks[tie_indices])
            ranks[tie_indices] = tie_rank
        return ranks
    
    
    x_ranks = rank(x)
    y_ranks = rank(y)
    
    
    d = x_ranks - y_ranks
    d_squared = d**2
    sum_d_squared = np.sum(d_squared)
    
    
    n = len(x)
    
    
    corr = 1 - (6 * sum_d_squared) / (n * (n**2 - 1))
    return corr


def point_biserial_correlation(binary_var, continuous_var):
    """
    Calculate the point-biserial correlation coefficient.
    
    Parameters:
    binary_var (np.array): Binary nominal variable (array of 0s and 1s).
    continuous_var (np.array): Continuous interval or ratio variable.
    
    Returns:
    float: Point-biserial correlation coefficient.
    """
    if len(binary_var) != len(continuous_var):
        raise ValueError("Both arrays must be of the same length")

        group1 = continuous_var[binary_var == 1]
    group0 = continuous_var[binary_var == 0]

    M1 = np.mean(group1)
    M0 = np.mean(group0)

    
    s = np.std(continuous_var, ddof=1)  

   
    n1 = len(group1)
    n0 = len(group0)
    n = len(continuous_var)

    
    corr = ((M1 - M0) / s) * np.sqrt((n1 * n0) / (n**2))

    return corr


def goodman_kruskal_gamma(x, y):
    """
    Calculate Goodman and Kruskal's gamma correlation coefficient.
    
    Parameters:
    x (np.array): First ordinal variable.
    y (np.array): Second ordinal variable.
    
    Returns:
    float: Goodman and Kruskal's gamma correlation coefficient.
    """
    if len(x) != len(y):
        raise ValueError("Both arrays must be of the same length")

    n = len(x)
    concordant = 0
    discordant = 0

    
    for i in range(n):
        for j in range(i + 1, n):
            
            dx = np.sign(x[i] - x[j])
            dy = np.sign(y[i] - y[j])
            product = dx * dy

            
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1

    
    if (concordant + discordant) == 0:
        return 0 
    corr = (concordant - discordant) / (concordant + discordant)
    return corr


def cramers_v(table):
    """
    Calculate Cramér's V statistic for a given contingency table.
    
    Parameters:
    table (np.array): A 2D numpy array representing the contingency table where
                      rows are categories of one variable and columns are categories of the other variable.
    
    Returns:
    float: Cramér's V statistic.
    """
    chi2 = np.sum((table - np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum())**2 / 
                  (np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()))
    n = table.sum()
    phi2 = chi2 / n
    r, k = table.shape
    phi2_corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))     # Correction for bias
    r_corr = r - ((r-1)**2)/(n-1)
    k_corr = k - ((k-1)**2)/(n-1)
    corr = np.sqrt(phi2_corr / min((k_corr-1), (r_corr-1)))
    return corr