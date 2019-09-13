# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 12:13:17 2019

@author: jcondori
"""


RANDOM_STATE = 31415
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
from scipy.stats import kstest
import scipy.stats
import pandas as pd


distributions=[
            scipy.stats.alpha	,
           # scipy.stats.anglit	,
           # scipy.stats.arcsine	,
           # scipy.stats.argus	,
            scipy.stats.beta	,
            #scipy.stats.betaprime	,
            #scipy.stats.bradford	,
            #scipy.stats.burr	,
            #scipy.stats.burr12	,
            scipy.stats.cauchy	,
            scipy.stats.chi	,
            scipy.stats.chi2	,
            #scipy.stats.cosine	,
          #  scipy.stats.crystalball	,
            #scipy.stats.dgamma	,
            #scipy.stats.dweibull	,
            #scipy.stats.erlang	,
            scipy.stats.expon	,
            #scipy.stats.exponnorm	,
            #scipy.stats.exponweib	,
            #scipy.stats.exponpow	,
            scipy.stats.f	,
            #scipy.stats.fatiguelife	,
            #scipy.stats.fisk	,
            #scipy.stats.foldcauchy	,
            #scipy.stats.foldnorm	,
            #scipy.stats.frechet_r	,
            #scipy.stats.frechet_l	,
            #scipy.stats.genlogistic	,
            #scipy.stats.gennorm	,
            #scipy.stats.genpareto	,
            #scipy.stats.genexpon	,
            #scipy.stats.genextreme	,
            #scipy.stats.gausshyper	,
            scipy.stats.gamma	,
            #scipy.stats.gengamma	,
            #scipy.stats.genhalflogistic	,
            #scipy.stats.gilbrat	,
            #scipy.stats.gompertz	,
            #scipy.stats.gumbel_r	,
            #scipy.stats.gumbel_l	,
            #scipy.stats.halfcauchy	,
            #scipy.stats.halflogistic	,
            #scipy.stats.halfnorm	,
            #scipy.stats.halfgennorm	,
            #scipy.stats.hypsecant	,
            #scipy.stats.invgamma	,
            #scipy.stats.invgauss	,
            #scipy.stats.invweibull	,
            #scipy.stats.johnsonsb	,
            #scipy.stats.johnsonsu	,
            #scipy.stats.kappa4	,
            #scipy.stats.kappa3	,
            #scipy.stats.ksone	,
            #scipy.stats.kstwobign	,
            #scipy.stats.laplace	,
            #scipy.stats.levy	,
            #scipy.stats.levy_l	,
            #scipy.stats.levy_stable	,
            scipy.stats.logistic	,
            #scipy.stats.loggamma	,
            #scipy.stats.loglaplace	,
            scipy.stats.lognorm	,
            #scipy.stats.lomax	,
            #scipy.stats.maxwell	,
            #scipy.stats.mielke	,
           # scipy.stats.moyal	,
            #scipy.stats.nakagami	,
            #scipy.stats.ncx2	,
            #scipy.stats.ncf	,
            #scipy.stats.nct	,
            scipy.stats.norm	,
           # scipy.stats.norminvgauss	,
            scipy.stats.pareto	,
            #scipy.stats.pearson3	,
            #scipy.stats.powerlaw	,
            #scipy.stats.powerlognorm	,
            #scipy.stats.powernorm	,
            #scipy.stats.rdist	,
            #scipy.stats.reciprocal	,
            #scipy.stats.rayleigh	,
            #scipy.stats.rice	,
            #scipy.stats.recipinvgauss	,
            #scipy.stats.semicircular	,
            #scipy.stats.skewnorm	,
            #scipy.stats.t	,
            #scipy.stats.trapz	,
            #scipy.stats.triang	,
            #scipy.stats.truncexpon	,
            #scipy.stats.truncnorm	,
            #scipy.stats.tukeylambda	,
            scipy.stats.uniform	,
            #scipy.stats.vonmises	,
            #scipy.stats.vonmises_line	,
            #scipy.stats.wald	,
            #scipy.stats.weibull_min	,
            #scipy.stats.weibull_max	,
            #scipy.stats.wrapcauchy	
            ]



    
def fit(data,dist,plot=False):
    params = dist.fit(data)
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]
    if plot:
        x = np.linspace(min(data), max(data), 80)
        _, ax = plt.subplots(1, 1)
        plt.hist(data, bins = 80, range=(min(data), max(data)))
        ax2 = ax.twinx()
        ax2.plot(x, dist.pdf(x, loc=loc, scale=scale, *arg), '-', color = "r", lw=2)
        plt.show()
    return dist, loc, scale, arg
    


def goodness_of_fit(data,dist,plot=False):
    dist, loc, scale, arg = fit(data,dist,plot=False)    
    d, pvalue = kstest(data.tolist(), lambda x: dist.cdf(x, loc = loc, scale = scale, *arg), alternative="two-sided")
    if plot:        
        x = np.linspace(min(data), max(data), 80)
        _, ax = plt.subplots(1, 1)
        counts, bin_edges = np.histogram(data, bins=80, normed=True)
        cdf = np.cumsum(counts)
        plt.plot(bin_edges[1:], cdf)
        ax2 = ax.twinx()
        ax2.plot(x, dist.cdf(x, loc=loc, scale=scale, *arg), '-', color = "r", lw=2)
        plt.show()
    return d,pvalue

def search_fit(data,distributions):
    ds=[]
    pvalues=[]
    dists=[]
    for dist in distributions:
        try:
            d,pvalue=goodness_of_fit(data,dist)
        except NotImplementedError:
            pass
        except KeyboardInterrupt:
            break
        else:
            pass
        ds.append(d)
        pvalues.append(pvalue)
        dists.append(dist)
    return {'dists':dists,'pvalues':pvalues,'ds':ds}

def best_distribution(data,plot=False):
    base=search_fit(data,distributions)
    minpos = base['ds'].index(min(base['ds']))
    dist=base['dists'][minpos]
    #p=base['pvalues'][minpos]
    dist,loc,scale,arg=fit(data,dist,plot=True)
    return arg,loc,scale,dist#,p

def simulation(n,arg,loc,scale,dist):
    try:
        seed = dist(c=arg,loc = loc, scale = scale)
    except:
        seed = dist(loc = loc, scale = scale)
    results = seed.rvs(n)
    return results


