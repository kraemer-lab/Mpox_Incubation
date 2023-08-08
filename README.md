<h1> Mpox Incubation Estimation </h1>

<p>The present repository reproduces and expands the analyses carried on by Miura and colleagues (2022) [see : https://github.com/fmiura/MpxInc_2022/tree/main]. We expand the analyses by providing fully parametrised models obtaining very close results to Miura et al. (2022), we also provide an example of prior updating where we fit models to data from Italy by adjusting priors from Netherlands parameters posterior information </p>


<h1> Models </h1>

<p> We implement the three models used in previous analyses: Lognormal, Gamma and Weibull models, and we add a NegativeBinomial model. All models use the same priors for parameters, with the exception of the sampling distribution (likelihood): </p>
<i>
<p align="center"> u ~ Beta(1, 1) </p>
<p align="center"> a ~ Gamma(1, 1) </p>
<p align="center"> b ~ Gamma(1, 1) </p>
<p align="center"> e = u(E - S) </p>
<p align="center"> y ~ LogNormal(&mu;=a + e, &sigma;=b) </p>
</i>

<p> Where, E is a vector containing the end of the exposure period, S is a vector containing the beginning of exposure period, <i>u</i> is the exposure period rate, and the observed data for <i>y</i> corresponds to the onset of symptoms vector O - S. The model above is repeated three more times but replacing <i>y</i> with <i>y</i> ~ Gamma(&mu;=a+e, &sigma;=b), parametrised via mean &mu; and standard deviation &sigma;;  <i>y</i> ~ Weibull(&alpha;=a+e, &beta;=b), and <i>y</i> ~ NegativeBinomial(&mu;=a+e, &alpha;=b) with mean &mu; and shape &sigma;. </p>


<h1> Results </h1>
<p> Models sampled well with all R_hats = 1, and ESS > 1000. The estimated mean incubation periods from the Netherlands data reproduce previous results closely, as Table 1 shows. CDF plots and PSIS-LOO model comparisons also indicate similarity of results, though the show the additional variation expected from a fully parametrised model.</p>

<p align="center">
	<img src="NE_table1.png" width="800" height="200" />
</p>

<p align="center">
	<img src="NE_cdfs_plots.png" width="800" height="700" />
</p>

<p align="center">
	<img src="NE_model_comp_loo.png" width="800" height="500" />
</p>

<p> After this, we used the posterior mean and standard deviation from each Gamma prior for parameters <i>a</i> and <i>b</i> from each model and used it to parameterise the respective model's Gamma priors for the Italy models. Results showed improved accuracy and precision. As expected from small data models, updating priors changes the predictive capacity of models, though it is relevant to emphasise that models show high ELPD overlap. </p>

<p align="center">
	<img src="IT_table2.png" width="800" height="200" />
</p>

<p align="center">
	<img src="IT_cdfs_plots.png" width="800" height="700" />
</p>

<p align="center">
	<img src="IT_model_comp_loo.png" width="800" height="500" />
</p>

<h1> Conclusion </h1>

<p> Results indicate that prior updating on models applied to new data (excluding previous data) can aid the estimation process. However, more testing is needed to asses what are the concrete advantages. Present analyses attempted to reproduce and expand previous results, but crucial parts of the analysis process need to be included. Crucially, a prior sensitivity analysis is required, as models are very sensitive to prior variation. This can be accomplished via a full Bayesian workflow. </p>

<h1> References </h1>
<p>
 Miura Fuminari, van Ewijk Catharina E, Backer Jantien A, Xiridou Maria, Franz Eelco, Op de Coul Eline, Brandwagt Diederik, van Cleef Brigitte, van Rijckevorsel Gini, Swaan Corien, van den Hof Susan, Wallinga Jacco. Estimated incubation period for monkeypox cases confirmed in the Netherlands, May 2022. Euro Surveill. 2022;27(24):pii=2200448. https://doi.org/10.2807/1560-7917.ES.2022.27.24.2200448
</p>
