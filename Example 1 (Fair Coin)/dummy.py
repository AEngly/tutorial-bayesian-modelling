# 1. Importer nødvendige biblioteker
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS 
import jax.random as random

# 2. Opsætning af model
def coin_model(flips=None):

    # Specifikér prior for p
    prior_alpha = 50
    prior_beta = 50

    # Sample p fra Beta fordelingen
    p = numpyro.sample("p", dist.Beta(prior_alpha, prior_beta))

    # Ved uafhængighed mellem kast kan plate bruges til at optimere sampling
    with numpyro.plate("data", len(flips)):
        numpyro.sample("obs", dist.Bernoulli(probs=p), obs=flips)

# 3. Kørsel af inferens
def run_inference(flips):

    # Konstruer en No-U-Turn Sampler (NUTS) kernel
    kernel = NUTS(coin_model)

    # Kør MCMC med den specificerede kernel
    mcmc = MCMC(kernel, num_warmup=200, num_samples=500, progress_bar=False)

    # Kør procedure (flips er de observerede data og skal passe med modellens argument)
    mcmc.run(random.PRNGKey(1), flips=flips)

    # Returner samples for p
    return mcmc.get_samples()["p"]
   


