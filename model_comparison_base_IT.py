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
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'figure.titlesize': 20})
plt.rcParams['font.family'] = "DeJavu Serif"
plt.rcParams['font.serif'] = "Cambria Math"


np.random.seed(27)

df = pd.read_csv("./data/Italy_Mpox.csv")


N = len(df)
tStartExposure = df['start'].values.astype("int")
tEndExposure = df['end'].values.astype("int")
tSymptomOnset = df['onset'].values.astype("int")

obs = tSymptomOnset - tStartExposure

#Lognormal model
with pm.Model() as mod_l:
    r = pm.Beta("r", 1, 1) 
    e = pm.Deterministic("e", r * (tEndExposure - tStartExposure)) 
    a = pm.Gamma("a", 1, 1) 
    m = pm.Deterministic("m", a+e) 
    s = pm.Gamma("s", 1, 1) 
    y = pm.LogNormal("y", mu=m, sigma=s, observed=obs) 
    ppc_l = pm.sample_prior_predictive(1000, random_seed=27) #prior predictives
    
#Gamma model
with pm.Model() as mod_g:
    r = pm.Beta("r", 1, 1) 
    e = pm.Deterministic("e", r * (tEndExposure - tStartExposure)) #exposure period effect
    a = pm.Gamma("a", 1, 1) 
    m = pm.Deterministic("m", a+e) 
    s = pm.Gamma("s", 1, 1) 
    y = pm.Gamma("y", mu=m, sigma=s, observed=obs)
    ppc_g = pm.sample_prior_predictive(1000, random_seed=27) #prior predictives

#Weibull model
with pm.Model() as mod_w:
    r = pm.Beta("r", 1, 1) 
    e = pm.Deterministic("e", r * (tEndExposure - tStartExposure)) 
    a = pm.Gamma("a", 1, 1) 
    m = pm.Deterministic("m", a+e) 
    s = pm.Gamma("s", 1, 1) 
    y = pm.Weibull("y", alpha=m, beta=s, observed=obs)
    ppc_w = pm.sample_prior_predictive(1000, random_seed=27) #prior predictives

#Negative binomial model
with pm.Model() as mod_n:
    r = pm.Beta("r", 1, 1) 
    e = pm.Deterministic("e", r * (tEndExposure - tStartExposure)) 
    a = pm.Gamma("a", 1, 1) 
    m = pm.Deterministic("m", a+e) 
    s = pm.Gamma("s", 1, 1) 
    y = pm.NegativeBinomial("y", mu=m, alpha=s, observed=obs)
    ppc_n = pm.sample_prior_predictive(1000, random_seed=27) #prior predictives


## plot priors
colors = ["#44AA99", "k", "#AA4499"]
ppcs = [ppc_l, ppc_g, ppc_w, ppc_n]
ppc_names = ["LogNormal", "Gamma", "Weibull", "NegativeBinomial"]
samps = np.random.randint(1000, size=100)
for p in range(len(ppcs)):
    y = az.extract(ppcs[p].prior_predictive)['y'].values
    r = az.extract(ppcs[p].prior)['r'].values
    m = az.extract(ppcs[p].prior)['m'].values.mean(axis=0)
    s = az.extract(ppcs[p].prior)['s'].values
    fig, ax, = plt.subplots(2,2, figsize=(10,10))
    ax[0,0].plot(np.arange(len(obs)), obs, color=colors[1], label="Observed")
    for i in samps:
        ax[0,0].plot(np.arange(len(obs)), y[:,i], alpha=0.1, color=colors[0])
    ax[0,0].plot(np.nan, np.nan, color=colors[0], label="Prior predictive")
    ax[0,0].plot(np.arange(len(obs)), y.mean(axis=1), linestyle="--", color=colors[2], label="Prior predictive mean")
    ax[0,0].set_axisbelow(True)
    ax[0,0].grid(0.1)
    ax[0,0].legend(loc="upper right", fontsize=12)
    ax[0,0].set_xlabel("Case")
    ax[0,0].set_ylabel("Symptom Onset - Start Exposure")
    ax[0,0].set_title("A. Predicted y")
    ax[0,1].hist(r, bins=100, color="sienna", alpha=0.5)
    ax[0,1].set_xlabel("Days")
    ax[0,1].set_ylabel("Density")
    ax[0,1].set_title("B. Prior r")
    ax[0,1].set_axisbelow(True)
    ax[0,1].grid(0.1)
    ax[1,0].hist(m, bins=100, color="orangered", alpha=0.5)
    ax[1,0].set_xlabel("Days")
    ax[1,0].set_ylabel("Density")
    ax[1,0].set_title("C. Prior m (mean)")
    ax[1,0].set_axisbelow(True)
    ax[1,0].grid(0.1)
    ax[1,1].hist(s, bins=100, color="slateblue", alpha=0.5)
    ax[1,1].set_xlabel("Days")
    ax[1,1].set_ylabel("Density")
    ax[1,1].set_title("D. Prior s")
    ax[1,1].set_axisbelow(True)
    ax[1,1].grid(0.1)
    plt.suptitle(ppc_names[p]+" Priors (Italy: no prior update)")
    plt.tight_layout()
    plt.savefig("./prior_predictive/IT_base_priors_"+ppc_names[p]+".png", dpi=600)
    

###### sample models #########
with mod_l:
    idata_l = pm.sample(1000, idata_kwargs={"log_likelihood": True}, random_seed=27) #NUTS sampler

with mod_g:
    idata_g = pm.sample(1000, idata_kwargs={"log_likelihood": True}, random_seed=27) #NUTS sampler

with mod_w:
    idata_w = pm.sample(1000, idata_kwargs={"log_likelihood": True}, random_seed=27) #NUTS sampler

with mod_n:
    idata_n = pm.sample(1000, idata_kwargs={"log_likelihood": True}, random_seed=27) #NUTS sampler



###compare models with PSIS-LOO
mods = {"LN":idata_l, "Gamma":idata_g, "Weibull":idata_w, "NB":idata_n}
loo = az.compare(mods, ic='loo')

az.plot_compare(loo, insample_dev=True, plot_kwargs={'color_insample_dev':'crimson', 'color_dse':'steelblue'})
plt.xlabel("ELPD LOO")
plt.title("LOO Model Comparison (Italy: no prior update)", size=12)
plt.grid(alpha=0.3)
plt.legend(prop={'size': 12})
plt.tight_layout()
plt.savefig('./model_comparison/IT_base_model_comp_loo.png', dpi=600)
plt.show()
plt.close()
loo_df = pd.DataFrame(loo)
loo_df.to_csv("./model_comparison/IT_base_model_comp_loo.csv")

###compare models with WAIC
mods = {"LN":idata_l, "Gamma":idata_g, "Weibull":idata_w, "NB":idata_n}
waic = az.compare(mods, ic='waic')

az.plot_compare(loo, insample_dev=True, plot_kwargs={'color_insample_dev':'crimson', 'color_dse':'steelblue'})
plt.xlabel("Log")
plt.title("Waic Model Comparison (Italy: no prior update)", size=12)
plt.grid(alpha=0.3)
plt.legend(prop={'size': 12})
plt.tight_layout()
plt.savefig('./model_comparison/IT_base_model_comp_waic.png', dpi=600)
plt.show()
plt.close()
loo_df = pd.DataFrame(loo)
loo_df.to_csv("./model_comparison/IT_base_model_comp_waic.csv")


##### Take models' estimated mean incubation period ####
# for Gamma and NegativeBinomial distributions this should be
# the same as their location/central tendecy parameters a.

pos_l_a = az.extract(idata_l.posterior)['m'].values.mean(axis=0)
pos_l_b = az.extract(idata_l.posterior)['s'].values
means_l = np.exp(pos_l_a + (pos_l_b**2)/2)

pos_g_a = az.extract(idata_g.posterior)['m'].values.mean(axis=0)
pos_g_b = az.extract(idata_g.posterior)['s'].values
ag = (pos_g_a**2)/(pos_g_b**2)
bg = pos_g_a/(pos_g_b**2)
means_g = ag / bg

pos_w_a = az.extract(idata_w.posterior)['m'].values.mean(axis=0)
pos_w_b = az.extract(idata_w.posterior)['s'].values
means_w = pos_w_b*gamma(1+(1/pos_w_a))

pos_n_a = az.extract(idata_n.posterior)['m'].values.mean(axis=0)
pos_n_b = az.extract(idata_n.posterior)['s'].values
means_n = pos_n_a

mod_names = ['LogNormal','Gamma','Weibull','NegativeBinomial']
mod_means = [means_l.mean(),means_g.mean(),means_w.mean(),means_n.mean()]
mod_stds = [means_l.std(),means_g.std(),means_w.std(),means_n.std()]
mod_hls = [az.hdi(means_l.T, hdi_prob=0.95)[0], az.hdi(means_g.T, hdi_prob=0.95)[0],
           az.hdi(means_w.T, hdi_prob=0.95)[0], az.hdi(means_n.T, hdi_prob=0.95)[0]]
mod_hus = [az.hdi(means_l.T, hdi_prob=0.95)[1], az.hdi(means_g.T, hdi_prob=0.95)[1],
           az.hdi(means_w.T, hdi_prob=0.95)[1], az.hdi(means_n.T, hdi_prob=0.95)[1]]
IT_means = pd.DataFrame({"Model":mod_names, "Mean":mod_means, "SD":mod_stds,
                         "HDI 2.5%":mod_hls, "HDI 97.5%":mod_hus})
IT_means.to_csv("./posteriors/IT_base_means.csv")

IT_means = IT_means.round(2)

## Save estimates as table
fig, ax = plt.subplots(1, figsize=(5,5))
ax.axis('off')
ax.axis('tight')
table = ax.table(cellText=IT_means.values, colLabels=IT_means.columns, loc='center')
table.set_fontsize(25)
table.scale(4.5, 4.5) 
plt.suptitle("Table 2. Estimated Mean Incubation Period (Italy: no prior update)", size=30, y=1)
plt.tight_layout()
plt.savefig("./posteriors/IT_base_table2.png", dpi=600, bbox_inches="tight")
plt.show()


### Produce and save cumulative density function (CDF) plots
# functions for cdf calculation
erf = sp.special.erf
ginc = sp.special.gammainc
Gamma = sp.special.gamma
Phi = sp.stats.norm.cdf
binc = sp.special.betainc

#Lognormal cdf
def ln_cdf(x, m, s):
   return 0.5*(1 + erf( ((np.log(x)-m))/(s*np.sqrt(2))) )    

#Gamma cdf   
def gam_cdf(x, m, s):
    a = (m**2)/(s**2)
    b = m/s**2
    return sp.stats.gamma.cdf(x,a,scale=1/b)

#Weibull cdf    
def wei_cdf(x, a, b):
    return 1 - np.exp(-(x/b)**a) 

#Negative binomial cdf
def nb_cdf(x, m, a):
    p = a/(m+a)
    n = a
    return sp.stats.nbinom.cdf(x, n, p)

# In case a Wald distribution is also tried
# def wal_cdf(x, m, l):
#     p1 = Phi(np.sqrt(l/x)*((x/m) - 1))
#     p2 = np.exp(2*l/m)*Phi(-np.sqrt(l/x)*((x/m) + 1))
#     return p1 + p2


# period of x = 30 days for computing cdfs
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


# save plots
fig, ax = plt.subplots(2,2, figsize=(10,10))
sns.ecdfplot(inc_day, ax=ax[0,0], color='k',  label="Empirical CDF")
ax[0,0].plot(x.mean(axis=1), l_cdf_m, color="#0072B2", linestyle="--", label="Posterior mean")
ax[0,0].fill_between(x.mean(axis=1), l_cdf_5, l_cdf_95, color="#56B4E9", alpha=0.3, label="95% HDI")
ax[0,0].set_ylabel("Proportion")
ax[0,0].set_xlabel("Incubation period (days)")
ax[0,0].spines[['right', 'top']].set_visible(False)
ax[0,0].legend(loc="lower right")
ax[0,0].grid(alpha=0.2)
ax[0,0].set_title("A. LogNormal")
sns.ecdfplot(inc_day, ax=ax[0,1], color='k', label="Empirical CDF")
ax[0,1].plot(x.mean(axis=1), g_cdf_m, color="#D55E00", linestyle="--", label="Posterior mean")
ax[0,1].fill_between(x.mean(axis=1), g_cdf_5, g_cdf_95, color="#CC79A7", alpha=0.3, label="95% HDI")
ax[0,1].set_ylabel("Proportion")
ax[0,1].set_xlabel("Incubation period (days)")
ax[0,1].spines[['right', 'top']].set_visible(False)
ax[0,1].legend(loc="lower right")
ax[0,1].grid(alpha=0.2)
ax[0,1].set_title("B. Gamma")
sns.ecdfplot(inc_day, ax=ax[1,0], color='k', label="Empirical CDF")
ax[1,0].plot(x.mean(axis=1), w_cdf_m, color="#E69F00", linestyle="--", label="Posterior mean")
ax[1,0].fill_between(x.mean(axis=1), w_cdf_5, w_cdf_95, color="#F0E442", alpha=0.3, label="95% HDI")
ax[1,0].set_ylabel("Proportion")
ax[1,0].set_xlabel("Incubation period (days)")
ax[1,0].spines[['right', 'top']].set_visible(False)
ax[1,0].legend(loc="lower right")
ax[1,0].grid(alpha=0.2)
ax[1,0].set_title("C. Weibull")
sns.ecdfplot(inc_day, ax=ax[1,1], color='k', label="Empirical CDF")
ax[1,1].plot(x.mean(axis=1), n_cdf_m, color="#117733", linestyle="--", label="Posterior mean")
ax[1,1].fill_between(x.mean(axis=1), n_cdf_5, n_cdf_95, color="#44AA99", alpha=0.3, label="95% HDI")
ax[1,1].set_ylabel("Proportion")
ax[1,1].set_xlabel("Incubation period (days)")
ax[1,1].spines[['right', 'top']].set_visible(False)
ax[1,1].legend(loc="lower right")
ax[1,1].grid(alpha=0.2)
ax[1,1].set_title("D. Negative Binomial")
plt.suptitle("Models CDFs (Italy: no prior update)")
plt.tight_layout()
plt.savefig("./posteriors/IT_base_cdfs_plots.png", dpi=600)
plt.show()
plt.close()


## produce and save summaries of models outputs
l_summ = az.summary(idata_l, hdi_prob=0.95)
g_summ = az.summary(idata_g, hdi_prob=0.95)
w_summ = az.summary(idata_w, hdi_prob=0.95)
n_summ = az.summary(idata_n, hdi_prob=0.95)

l_summ['model'] = np.repeat("LogNormal", len(l_summ))
g_summ['model'] = np.repeat("Gamma", len(g_summ)) 
w_summ['model'] = np.repeat("Weibull", len(w_summ))
n_summ['model'] = np.repeat("NegativeBinomial", len(n_summ))

posteriors = pd.concat([l_summ, g_summ, w_summ, n_summ])

posteriors.to_csv("./posteriors/IT_base_posteriors.csv")


##################### save summary plots #########################
#plot bfmis
fig, ax = plt.subplots(2,2, figsize=(12,12))
az.plot_energy(idata_l, ax=ax[0,0])
ax[0,0].set_title("LogNormal")
az.plot_energy(idata_g, ax=ax[0,1])
ax[0,1].set_title("Gamma")
az.plot_energy(idata_w, ax=ax[1,0])
ax[1,0].set_title("Weibull")
az.plot_energy(idata_n, ax=ax[1,1])
ax[1,1].set_title("NegativeBinomial")
plt.suptitle("Italy: no prior update")
plt.tight_layout()
plt.savefig("./convergence_checks/IT_base_energy_plots.png", dpi=300)
plt.close()

# plot traces
fig, ax = plt.subplots(5,2, figsize=(12,12))
az.plot_trace(idata_l, kind="rank_vlines", axes=ax)
for a in ax[:,0]:
    t = a.get_title()
    a.set_title(t, fontsize=22)
for a in ax[:,1]:
    t = a.get_title()
    a.set_title(t, fontsize=22)
plt.suptitle("LogNormal (Italy: no prior update)", size=22)
plt.tight_layout()
plt.savefig("./convergence_checks/IT_base_lognormal_rankplot.png", dpi=300)

fig, ax = plt.subplots(5,2, figsize=(12,12))
az.plot_trace(idata_g, kind="rank_vlines", axes=ax)
for a in ax[:,0]:
    t = a.get_title()
    a.set_title(t, fontsize=22)
for a in ax[:,1]:
    t = a.get_title()
    a.set_title(t, fontsize=22)
plt.suptitle("Gamma (Italy: no prior update)", size=22)
plt.tight_layout()
plt.savefig("./convergence_checks/IT_base_gamma_rankplot.png", dpi=300)

fig, ax = plt.subplots(5,2, figsize=(12,12))
az.plot_trace(idata_w, kind="rank_vlines", axes=ax)
for a in ax[:,0]:
    t = a.get_title()
    a.set_title(t, fontsize=22)
for a in ax[:,1]:
    t = a.get_title()
    a.set_title(t, fontsize=22)
plt.suptitle("Weibull (Italy: no prior update)", size=22)
plt.tight_layout()
plt.savefig("./convergence_checks/IT_base_weibull_rankplot.png", dpi=300)

fig, ax = plt.subplots(5,2, figsize=(12,12))
az.plot_trace(idata_l, kind="rank_vlines", axes=ax)
for a in ax[:,0]:
    t = a.get_title()
    a.set_title(t, fontsize=22)
for a in ax[:,1]:
    t = a.get_title()
    a.set_title(t, fontsize=22)
plt.suptitle("NegativeBinomial (Italy: no prior update)", size=22)
plt.tight_layout()
plt.savefig("./convergence_checks/IT_base_negativebinom_rankplot.png", dpi=300)


########### Posterior predictive checks #################

pred_l = pm.sample_posterior_predictive(idata_l, model=mod_l, random_seed=27) #osterior predictives

pred_g = pm.sample_posterior_predictive(idata_g, model=mod_g, random_seed=27) #posterior predictives

pred_w = pm.sample_posterior_predictive(idata_w, model=mod_w, random_seed=27) #posterior predictives

pred_n = pm.sample_posterior_predictive(idata_n, model=mod_n, random_seed=27) #posterior predictives


#plot posterior predictions
ppcs = [pred_l, pred_g, pred_w, pred_n]
ppc_names = ["LogNormal", "Gamma", "Weibull", "NegativeBinomial"]
letters=["A. ", "B. ", "C. ", "D. "]
colors = ["#44AA99", "k", "#AA4499"]
fig, ax = plt.subplots(2,2, figsize=(10,10))
axes = [(0,0), (0,1), (1,0), (1,1)]
for p in range(len(ppcs)):
    az.plot_ppc(ppcs[p], kind="cumulative", colors=colors, ax=ax[axes[p]], random_seed=27)
    ax[axes[p]].set_title(letters[p]+ppc_names[p])
    ax[axes[p]].spines[['right', 'top']].set_visible(False)
    ax[axes[p]].set_xlabel("y (days)")
    ax[axes[p]].set_ylabel("Proportion")
    ax[axes[p]].grid(0.2)
plt.suptitle("Posterior Predictives (Italy: no prior update)")
plt.tight_layout()
plt.savefig("./posterior_predictive/IT_base_predictions.png", dpi=600)

# ppcs = [pred_l, pred_g, pred_w, pred_n]
# ppc_names = ["LogNormal", "Gamma", "Weibull", "NegativeBinomial"]
# letters=["A. ", "B. ", "C. ", "D. "]
# colors = [["#56B4E9", "k", "#0072B2"], ["#CC79A7", "k", "#D55E00"],
#           ["#F0E442", "k", "#E69F00"], ["#44AA99", "k", "#117733"] ]
# fig, ax = plt.subplots(2,2, figsize=(10,10))
# axes = [(0,0), (0,1), (1,0), (1,1)]
# for p in range(len(ppcs)):
#     az.plot_ppc(ppcs[p], kind="cumulative", colors=colors[p], ax=ax[axes[p]], random_seed=27)
#     ax[axes[p]].set_title(letters[p]+ppc_names[p])
#     ax[axes[p]].spines[['right', 'top']].set_visible(False)
#     ax[axes[p]].set_xlabel("y (days)")
#     ax[axes[p]].set_ylabel("Proportion")
#     ax[axes[p]].grid(0.2)
# plt.suptitle("Posterior Predictives (Italy: no prior update)")
# plt.tight_layout()
# plt.savefig("./posterior_predictive/IT_base_predictions.png", dpi=600)
