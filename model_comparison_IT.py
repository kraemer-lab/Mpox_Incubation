# -*- coding: utf-8 -*-
import pymc as pm
import arviz as az
import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import pytensor.tensor as pt
import matplotlib.pyplot as plt
from scipy.special import gamma

#####plotting parameters
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'figure.titlesize': 16})
plt.rcParams['font.family'] = "DeJavu Serif"
plt.rcParams['font.serif'] = "Cambria Math"


np.random.seed(27)

df = pd.read_csv("./data/Italy_Mpox.csv")

priors = pd.read_csv("./summaries/NE_posteriors.csv")
priors = priors.rename(columns={"Unnamed: 0":"p"})
ps_l = priors[priors.model=="LogNormal"]
ps_l.reset_index(inplace=True,drop=True)
ps_g = priors[priors.model=="Gamma"]
ps_g.reset_index(inplace=True,drop=True)
ps_w = priors[priors.model=="Weibull"]
ps_w.reset_index(inplace=True,drop=True)
ps_n = priors[priors.model=="NegativeBinomial"]
ps_n.reset_index(inplace=True,drop=True)

N = len(df)
tStartExposure = df['start'].values.astype("int")
tEndExposure = df['end'].values.astype("int")
tSymptomOnset = df['onset'].values.astype("int")

obs = tSymptomOnset - tStartExposure

with pm.Model() as mod_l:
    #u = pm.Gamma("u", ps_l['mean'][0], ps_l['sd'][0])
    u = pm.Beta("u", 1, 1)
    e = u * (tEndExposure - tStartExposure)
    a = pm.Gamma("a", mu=ps_l['mean'][1], sigma=ps_l['sd'][1])
    b = pm.Gamma("b", mu=ps_l['mean'][2], sigma=ps_l['sd'][2])
    y = pm.LogNormal("y", mu=a+e, sigma=b, observed=obs)
    idata_l = pm.sample(1000, idata_kwargs={"log_likelihood": True}, target_accept=0.99, random_seed=27)

with pm.Model() as mod_g:
    #u = pm.Gamma("u", ps_g['mean'][0], ps_g['sd'][0])
    u = pm.Beta("u", 1, 1)
    e = u * (tEndExposure - tStartExposure)
    a = pm.Gamma("a", mu=ps_g['mean'][1], sigma=ps_g['sd'][1])
    b = pm.Gamma("b", mu=ps_g['mean'][2], sigma=ps_g['sd'][2])
    y = pm.Gamma("y", mu=a+e, sigma=b, observed=obs)
    idata_g = pm.sample(1000, idata_kwargs={"log_likelihood": True}, random_seed=27)

with pm.Model() as mod_w:
    #u = pm.Gamma("u", ps_w['mean'][0], ps_w['sd'][0])
    u = pm.Beta("u", 1, 1)
    e = u * (tEndExposure - tStartExposure)
    a = pm.Gamma("a", mu=ps_w['mean'][1], sigma=ps_w['sd'][1])
    b = pm.Gamma("b", mu=ps_w['mean'][2], sigma=ps_w['sd'][2])
    y = pm.Weibull("y", alpha=a+e, beta=b, observed=obs)
    idata_w = pm.sample(1000, idata_kwargs={"log_likelihood": True}, target_accept=0.99, random_seed=27)
    
with pm.Model() as mod_wa:
    #u = pm.Gamma("u", ps_n['mean'][0], ps_n['sd'][0])
    u = pm.Beta("u", 1, 1)
    e = u * (tEndExposure - tStartExposure)
    a = pm.Gamma("a", mu=ps_n['mean'][1], sigma=ps_n['sd'][1])
    b = pm.Gamma("b", mu=ps_n['mean'][2], sigma=ps_n['sd'][2])
    y = pm.NegativeBinomial("y", mu=a+e, alpha=b, observed=obs)
    idata_n = pm.sample(1000, idata_kwargs={"log_likelihood": True}, random_seed=27)


###compare loo
mods = {"LN":idata_l, "Gamma":idata_g, "Weibull":idata_w,  "NB":idata_n}
loo = az.compare(mods, ic='loo')

az.plot_compare(loo, insample_dev=True, plot_kwargs={'color_insample_dev':'crimson', 'color_dse':'steelblue'})
plt.xlabel("ELPD LOO")
plt.title("LOO Model Comparison (Italy)", size=12)
plt.grid(alpha=0.3)
plt.legend(prop={'size': 12})
plt.tight_layout()
plt.savefig('./plots/IT_model_comp_loo.png', dpi=600)
plt.show()
plt.close()
loo_df = pd.DataFrame(loo)
loo_df.to_csv("./summaries/IT_model_comp_loo.csv")


###compare Waic
mods = {"LN":idata_l, "Gamma":idata_g, "Weibull":idata_w, "NB":idata_n}
waic = az.compare(mods, ic='waic')

az.plot_compare(loo, insample_dev=True, plot_kwargs={'color_insample_dev':'crimson', 'color_dse':'steelblue'})
plt.xlabel("Log")
plt.title("Waic Model Comparison (Italy)", size=12)
plt.grid(alpha=0.3)
plt.legend(prop={'size': 12})
plt.tight_layout()
plt.savefig('./plots/IT_model_comp_waic.png', dpi=600)
plt.show()
plt.close()
loo_df = pd.DataFrame(loo)
loo_df.to_csv("./summaries/IT_model_comp_waic.csv")


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
ne_means.to_csv("./summaries/IT_means.csv")

ne_means = ne_means.round(2)

fig, ax = plt.subplots(1, figsize=(5,5))
ax.axis('off')
ax.axis('tight')
table = ax.table(cellText=ne_means.values, colLabels=ne_means.columns, loc='center')
table.set_fontsize(25)
table.scale(4.5, 4.5) 
plt.suptitle("Table 3. Estimated Mean Incubation Period (Italy)", size=30, y=1)
plt.tight_layout()
plt.savefig("./plots/IT_table3.png", dpi=600, bbox_inches="tight")
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
    b = m/s**2
    return sp.stats.gamma.cdf(x,a,scale=s.mean()/2)
    
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

l = np.round(((30-18)/2), 0)
r = 30-18 - l

#inc_day = list(np.repeat(min(inc_day), l)) + list(inc_day) + list(np.repeat(max(inc_day), r))

num_bins = x.shape[0]
counts, bin_edges = np.histogram(np.sort(inc_day), bins=num_bins, density=True)
e_cdf = np.cumsum(counts/counts.sum())

fig, ax = plt.subplots(2,2, figsize=(10,10))
sns.ecdfplot(inc_day, ax=ax[0,0], color='k', linestyle=":", label="Empirical CDF")
ax[0,0].plot(x.mean(axis=1), l_cdf_m, color="slateblue", label="LogNormal CDF")
ax[0,0].fill_between(x.mean(axis=1), l_cdf_5, l_cdf_95, color="slateblue", alpha=0.2, label="95% HDI")
ax[0,0].set_ylabel("Proporton")
ax[0,0].set_xlabel("Incubation period (days)")
ax[0,0].spines[['right', 'top']].set_visible(False)
ax[0,0].legend(loc="lower right")
ax[0,0].grid(alpha=0.2)
ax[0,0].set_title("A. LogNormal")
sns.ecdfplot(inc_day, ax=ax[0,1], color='k', linestyle=":", label="Empirical CDF")
ax[0,1].plot(x.mean(axis=1), g_cdf_m, color="crimson", label="Gamma CDF")
ax[0,1].fill_between(x.mean(axis=1), g_cdf_5, g_cdf_95, color="crimson", alpha=0.2, label="95% HDI")
ax[0,1].set_ylabel("Proporton")
ax[0,1].set_xlabel("Incubation period (days)")
ax[0,1].spines[['right', 'top']].set_visible(False)
ax[0,1].legend(loc="lower right")
ax[0,1].grid(alpha=0.2)
ax[0,1].set_title("B. Gamma")
sns.ecdfplot(inc_day, ax=ax[1,0], color='k', linestyle=":", label="Empirical CDF")
ax[1,0].plot(x.mean(axis=1), w_cdf_m, color="orange", label="Weibull CDF")
ax[1,0].fill_between(x.mean(axis=1), w_cdf_5, w_cdf_95, color="orange", alpha=0.2, label="95% HDI")
ax[1,0].set_ylabel("Proporton")
ax[1,0].set_xlabel("Incubation period (days)")
ax[1,0].spines[['right', 'top']].set_visible(False)
ax[1,0].legend(loc="lower right")
ax[1,0].grid(alpha=0.2)
ax[1,0].set_title("C. Weibull")
sns.ecdfplot(inc_day, ax=ax[1,1], color='k', linestyle=":", label="Empirical CDF")
ax[1,1].plot(x.mean(axis=1), n_cdf_m, color="forestgreen", label="NegativeBinomial CDF")
ax[1,1].fill_between(x.mean(axis=1), n_cdf_5, n_cdf_95, color="forestgreen", alpha=0.2, label="95% HDI")
ax[1,1].set_ylabel("Proporton")
ax[1,1].set_xlabel("Incubation period (days)")
ax[1,1].spines[['right', 'top']].set_visible(False)
ax[1,1].legend(loc="lower right")
ax[1,1].grid(alpha=0.2)
ax[1,1].set_title("D. Negative Binomial")
plt.suptitle("Models CDFs (Italy)")
plt.tight_layout()
plt.savefig("./plots/IT_cdfs_plots.png", dpi=600)
plt.show()
plt.close()


l_summ = az.summary(idata_l, hdi_prob=0.95)
g_summ = az.summary(idata_g, hdi_prob=0.95)
w_summ = az.summary(idata_w, hdi_prob=0.95)
n_summ = az.summary(idata_n, hdi_prob=0.95)

l_summ['model'] = np.repeat("LogNormal", len(l_summ))
g_summ['model'] = np.repeat("Gamma", len(g_summ)) 
w_summ['model'] = np.repeat("Weibull", len(w_summ))
n_summ['model'] = np.repeat("NegativeBinomial", len(w_summ))

posteriors = pd.concat([l_summ, g_summ, w_summ, n_summ])

posteriors.to_csv("./summaries/IT_posteriors.csv")


### save summary plots
fig, ax = plt.subplots(2,2, figsize=(12,12))
az.plot_energy(idata_l, ax=ax[0,0])
ax[0,0].set_title("LogNormal")
az.plot_energy(idata_g, ax=ax[0,1])
ax[0,1].set_title("Gamma")
az.plot_energy(idata_w, ax=ax[1,0])
ax[1,0].set_title("Weibull")
az.plot_energy(idata_n, ax=ax[1,1])
ax[1,1].set_title("NegativeBinomial")
plt.suptitle("Italy")
plt.tight_layout()
plt.savefig("./summary_plots/IT_energy_plots.png", dpi=300)
plt.close()


fig, ax = plt.subplots(2,2, figsize=(12,12))
az.plot_trace(idata_l, kind="rank_vlines")
plt.suptitle("LogNormal (Italy)")
plt.tight_layout()
plt.savefig("./summary_plots/IT_lognormal_rankplot.png", dpi=300)

az.plot_trace(idata_g, kind="rank_vlines")
plt.suptitle("Gamma (Italy)")
plt.tight_layout()
plt.savefig("./summary_plots/IT_gamma_rankplot.png", dpi=300)

az.plot_trace(idata_w, kind="rank_vlines")
plt.suptitle("Weibull (Italy)")
plt.tight_layout()
plt.savefig("./summary_plots/IT_weibull_rankplot.png", dpi=300)

az.plot_trace(idata_l, kind="rank_vlines")
plt.suptitle("NegativeBinomial (Italy)")
plt.tight_layout()
plt.savefig("./summary_plots/IT_negativebinom_rankplot.png", dpi=300)