# -*- coding: utf-8 -*-
import pymc as pm
import arviz as az
import numpy as np
import scipy as sp
import pandas as pd
import pytensor.tensor as pt
import matplotlib.pyplot as plt
from scipy.special import gamma

#####plotting parameters
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'figure.titlesize': 16})
plt.rcParams['font.family'] = "DeJavu Serif"
plt.rcParams['font.serif'] = "Cambria Math"

np.random.seed(27)

df = pd.read_csv("Netherlands_Mpox.csv")

N = len(df)
tStartExposure = df['Start date of exposure'].values.astype("int")
tEndExposure = df['End date of exposure'].values.astype("int")
tSymptomOnset = df['Symptom onset'].values.astype("int")

# Ti = tSymptomOnset - tStartExposure #incubation period

# uE = np.linspace(0,1,N)
# tE = tStartExposure + uE * (tEndExposure - tStartExposure)
# obs = tSymptomOnset - tE

obs = tSymptomOnset - tStartExposure

with pm.Model() as mod_l:
    u = pm.Beta("u", 1, 1)
    e = u * (tEndExposure - tStartExposure)
    a = pm.Gamma("a", 1, 1)
    b = pm.Gamma("b", 1, 1)
    y = pm.LogNormal("y", mu=a+e, sigma=b, observed=obs)
    idata_l = pm.sample(1000, idata_kwargs={"log_likelihood": True}, random_seed=27)

with pm.Model() as mod_g:
    u = pm.Beta("u", 1, 1)
    e = u * (tEndExposure - tStartExposure)
    a = pm.Gamma("a", 1, 1)
    b = pm.Gamma("b", 1, 1)
    y = pm.Gamma("y", mu=a+e, sigma=b, observed=obs)
    idata_g = pm.sample(1000, idata_kwargs={"log_likelihood": True}, random_seed=27)

with pm.Model() as mod_w:
    u = pm.Beta("u", 1, 1)
    e = u * (tEndExposure - tStartExposure)
    a = pm.Gamma("a", 1, 1)
    b = pm.Gamma("b", 1, 1)
    y = pm.Weibull("y", alpha=a+e, beta=b, observed=obs)
    idata_w = pm.sample(1000, idata_kwargs={"log_likelihood": True}, random_seed=27)
    
with pm.Model() as mod_n:
    u = pm.Beta("u", 1, 1)
    e = u * (tEndExposure - tStartExposure)
    a = pm.Gamma("a", 1, 1)
    b = pm.Gamma("b", 1, 1)
    y = pm.NegativeBinomial("y", mu=a+e, alpha=b, observed=obs)
    idata_n = pm.sample(1000, idata_kwargs={"log_likelihood": True}, random_seed=27)


###compare loo
mods = {"LN":idata_l, "Gamma":idata_g, "Weibull":idata_w, "NB":idata_n}
loo = az.compare(mods, ic='loo')

az.plot_compare(loo, insample_dev=True, plot_kwargs={'color_insample_dev':'crimson', 'color_dse':'steelblue'})
plt.xlabel("ELPD LOO")
plt.title("LOO Model Comparison (Netherlands)", size=12)
plt.grid(alpha=0.3)
plt.legend(prop={'size': 12})
plt.tight_layout()
plt.savefig('NE_model_comp_loo.png', dpi=600)
plt.show()
plt.close()
loo_df = pd.DataFrame(loo)
loo_df.to_csv("NE_model_comp_loo.csv")


###compare Waic
mods = {"LN":idata_l, "Gamma":idata_g, "Weibull":idata_w, "NB":idata_n}
waic = az.compare(mods, ic='waic')

az.plot_compare(loo, insample_dev=True, plot_kwargs={'color_insample_dev':'crimson', 'color_dse':'steelblue'})
plt.xlabel("Log")
plt.title("Waic Model Comparison (Netherlands)", size=12)
plt.grid(alpha=0.3)
plt.legend(prop={'size': 12})
plt.tight_layout()
plt.savefig('NE_model_comp_waic.png', dpi=600)
plt.show()
plt.close()
loo_df = pd.DataFrame(loo)
loo_df.to_csv("NE_model_comp_waic.csv")


pos_l_a = az.extract(idata_l.posterior)['a'].values
pos_l_b = az.extract(idata_l.posterior)['b'].values
means_l = np.exp(pos_l_a + (pos_l_b**2)/2)

pos_g_a = az.extract(idata_g.posterior)['a'].values
pos_g_b = az.extract(idata_g.posterior)['b'].values
ag = (pos_g_a**2)/(pos_g_b**2)
bg = pos_g_a/(pos_g_b**2)
means_g = ag / bg

pos_w_a = az.extract(idata_w.posterior)['a'].values
pos_w_b = az.extract(idata_w.posterior)['b'].values
means_w = pos_w_b*gamma(1+(1/pos_w_a))

pos_n_a = az.extract(idata_n.posterior)['a'].values
pos_n_b = az.extract(idata_n.posterior)['b'].values
means_n = pos_n_a

mod_names = ['LogNormal','Gamma','Weibull','NegativeBinomial']
mod_means = [means_l.mean(),means_g.mean(),means_w.mean(),means_n.mean()]
mod_stds = [means_l.std(),means_g.std(),means_w.std(),means_n.std()]
mod_hls = [az.hdi(means_l.T, hdi_prob=0.95)[0], az.hdi(means_g.T, hdi_prob=0.95)[0],
           az.hdi(means_w.T, hdi_prob=0.95)[0], az.hdi(means_n.T, hdi_prob=0.95)[0]]
mod_hus = [az.hdi(means_l.T, hdi_prob=0.95)[1], az.hdi(means_g.T, hdi_prob=0.95)[1],
           az.hdi(means_w.T, hdi_prob=0.95)[1], az.hdi(means_n.T, hdi_prob=0.95)[1]]
ne_means = pd.DataFrame({"Model":mod_names, "Mean":mod_means, "SD":mod_stds,
                         "HDI 2.5%":mod_hls, "HDI 97.5%":mod_hus})
ne_means.to_csv("NE_means.csv")

ne_means = ne_means.round(2)

fig, ax = plt.subplots(1, figsize=(5,5))
ax.axis('off')
ax.axis('tight')
table = ax.table(cellText=ne_means.values, colLabels=ne_means.columns, loc='center')
table.set_fontsize(25)
table.scale(4.5, 4.5) 
plt.suptitle("Table 1. Estimated Mean Incubation Period (Netherlands)", size=30, y=1)
plt.tight_layout()
plt.savefig("NE_table1.png", dpi=600, bbox_inches="tight")
plt.show()


### plot
erf = sp.special.erf
ginc = sp.special.gammainc
Gamma = sp.special.gamma
Phi = sp.stats.norm.cdf
binc = sp.special.betainc

def ln_cdf(x, m, s):
   return 0.5*(1 + erf( ((np.log(x)-m))/(s*np.sqrt(2))) )    
   
def gam_cdf(x, m, s):
    a = (m**2)/(s**2)
    return sp.stats.gamma.cdf(x,a,scale=s)
    
def wei_cdf(x, a, b):
    return 1 - np.exp(-(x/b)**a) 

def nb_cdf(x, m, a):
    p = a/(m+a)
    n = a
    return sp.stats.nbinom.cdf(x, n, p)
    
# def wal_cdf(x, m, l):
#     p1 = Phi(np.sqrt(l/x)*((x/m) - 1))
#     p2 = np.exp(2*l/m)*Phi(-np.sqrt(l/x)*((x/m) + 1))
#     return p1 + p2
    
x = np.array([np.arange(30, step=0.1) for i in range(pos_l_a.shape[0])]).T

l_cdf = ln_cdf(x, pos_l_a, pos_l_b)
l_cdf_m = np.median(l_cdf, axis=1)
l_cdf_5, l_cdf_95 = az.hdi(l_cdf.T, hdi_prob=0.95).T

g_cdf = gam_cdf(x, pos_g_a, pos_g_b)
g_cdf_m = np.median(g_cdf, axis=1)
g_cdf_5, g_cdf_95 = az.hdi(g_cdf.T, hdi_prob=0.95).T

w_cdf = wei_cdf(x, pos_w_a, pos_w_b)
w_cdf_m = np.median(w_cdf, axis=1)
w_cdf_5, w_cdf_95 = az.hdi(w_cdf.T, hdi_prob=0.95).T

n_cdf = nb_cdf(x, pos_n_a, pos_n_b)
n_cdf_m = np.median(n_cdf, axis=1)
n_cdf_5, n_cdf_95 = az.hdi(n_cdf.T, hdi_prob=0.95).T

inc_day = ((tSymptomOnset-tEndExposure)+(tSymptomOnset-tStartExposure))/2

num_bins = x.shape[0]
counts, bin_edges = np.histogram(np.sort(inc_day-inc_day.min()), bins=num_bins, density=True)
e_cdf = np.cumsum(counts/counts.sum())

fig, ax = plt.subplots(2,2, figsize=(10,10))
ax[0,0].step(x.mean(axis=1), e_cdf, color='k', linestyle=":", label="Empirical CDF")
ax[0,0].plot(x.mean(axis=1), l_cdf_m, color="slateblue", label="LogNormal CDF")
ax[0,0].fill_between(x.mean(axis=1), l_cdf_5, l_cdf_95, color="slateblue", alpha=0.2, label="95% HDI")
ax[0,0].set_ylabel("Proporton")
ax[0,0].set_xlabel("Incubation period (days)")
ax[0,0].spines[['right', 'top']].set_visible(False)
ax[0,0].legend(loc="lower right")
ax[0,0].grid(alpha=0.2)
ax[0,0].set_title("A. LogNormal")
ax[0,1].step(x.mean(axis=1), e_cdf, color='k', linestyle=":", label="Empirical CDF")
ax[0,1].plot(x.mean(axis=1), g_cdf_m, color="crimson", label="Gamma CDF")
ax[0,1].fill_between(x.mean(axis=1), g_cdf_5, g_cdf_95, color="crimson", alpha=0.2, label="95% HDI")
ax[0,1].set_ylabel("Proporton")
ax[0,1].set_xlabel("Incubation period (days)")
ax[0,1].spines[['right', 'top']].set_visible(False)
ax[0,1].legend(loc="lower right")
ax[0,1].grid(alpha=0.2)
ax[0,1].set_title("B. Gamma")
ax[1,0].step(x.mean(axis=1), e_cdf, color='k', linestyle=":", label="Empirical CDF")
ax[1,0].plot(x.mean(axis=1), w_cdf_m, color="orange", label="Weibull CDF")
ax[1,0].fill_between(x.mean(axis=1), w_cdf_5, w_cdf_95, color="orange", alpha=0.2, label="95% HDI")
ax[1,0].set_ylabel("Proporton")
ax[1,0].set_xlabel("Incubation period (days)")
ax[1,0].spines[['right', 'top']].set_visible(False)
ax[1,0].legend(loc="lower right")
ax[1,0].grid(alpha=0.2)
ax[1,0].set_title("C. Weibull")
ax[1,1].step(x.mean(axis=1), e_cdf, color='k', linestyle=":", label="Empirical CDF")
ax[1,1].plot(x.mean(axis=1), n_cdf_m, color="forestgreen", label="NegativeBinomial CDF")
ax[1,1].fill_between(x.mean(axis=1), n_cdf_5, n_cdf_95, color="forestgreen", alpha=0.2, label="95% HDI")
ax[1,1].set_ylabel("Proporton")
ax[1,1].set_xlabel("Incubation period (days)")
ax[1,1].spines[['right', 'top']].set_visible(False)
ax[1,1].legend(loc="lower right")
ax[1,1].grid(alpha=0.2)
ax[1,1].set_title("D. Negative Binomial")
plt.suptitle("Models CDFs (Netherlands)")
plt.tight_layout()
plt.savefig("NE_cdfs_plots.png", dpi=600)
plt.show()
plt.close()

l_summ = az.summary(idata_l, hdi_prob=0.95)
g_summ = az.summary(idata_g, hdi_prob=0.95)
w_summ = az.summary(idata_w, hdi_prob=0.95)
n_summ = az.summary(idata_n, hdi_prob=0.95)

l_summ['model'] = np.repeat("LogNormal", len(l_summ))
g_summ['model'] = np.repeat("Gamma", len(g_summ)) 
w_summ['model'] = np.repeat("Weibull", len(w_summ))
n_summ['model'] = np.repeat("NegativeBinomial", len(n_summ))

posteriors = pd.concat([l_summ, g_summ, w_summ, n_summ])

posteriors.to_csv("NE_posteriors.csv")

