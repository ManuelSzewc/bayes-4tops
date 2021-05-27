# Posterior inference of a simple probabilistic model of 4-top production at the LHC

Based on arxiv:XXXXX

We consider the 4-top production and one of its main irreducible backgrounds ttW, measured at the LHC on 2LSS++ final state. As input, we consider the number of light and b-tagged jets measured per event. Using a mixture model, we assume conditional independence of both measurements. This forces the model to encode the correlation between the two variables in the theme assignments. After we specify the priors, we can infer the posterior distributions over the relevant parameters: the theme fractions and the theme distributions (two per theme). 

In the "necessary_functions" module we include a series of functions which allow us to search for the MAP using EM+priors, to approximate the posterior using VI and to obtain samples from the posterior by means of a Gibbs Sampler. Usage of the latter two is shown in run_vb.py and run_gibbs_sampler_one_walker.py respectively. In the run_gibbs_sampler.bash we show how to recover the plots shown in arxiv:XXX. Having turned the file into an executable one needs to run

./run_gibbs_sampler.bash "data directory" "output directory" "Number of events considered" "Number of walkers" "Number of saved iterations per walker" "Burn-in samples" "Separation between saved samples"
