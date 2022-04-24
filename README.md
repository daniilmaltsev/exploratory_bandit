# exploratory_bandits
Pure exploration N-armed Bernoulli bayesian bandits aimed to minimize the experiment time regardless of the expoitation benefits.
Slow feedback loop is assumed, i.e. some number of events (batch_size) happen between the allocation decision and the feedback signal.
A batch-adjusted Thompson bandit is provided for a baseline.

All the bandits in this repository share a similar interface:

Input: conversions vector and n_users vector (each vector contains the values for all arms in the same order), batch_size.

Output: allocation vector with suggested traffic fractions to be allocated to each group.

The desired time constraint is ~500ms per allocation for 4 arms. All the bandits follow single-process to allow for external multiprocess calls.

## Squid: quantile intersection gradient bandit
Squid calculates the distance between the lower 3-sigma quantile of the leading arm and the upper 3-sigma quantiles of its rivals, and finds the gradient for this distance reduction relative to the number of users directed to the arm. 
The quantile intersections are clipped to zero for the non-intersecting arms. These arms are deemed useless at this iteration and their gradients turn to zero, too.

The quantile intersection distance is the measure of the entropy we care about. The resulting allocation is proportional to the entropy reduction expected from each arm on having been fed the batch_size number of users.

Drawbacks: 
- 3-sigma quantiles are a rough simplification, they don't reflect the actiual quantiles for heavy-tailed beta distributions close to 0 and 1.
- the allocation is conservative, perhaps we would want to feed all the batch to the best arm (in terms of entropy minimization)

## Helga: expected loss gradient bandit
Similarly to Squid, Helga seeks to minimize the entropy, but the measure of the entropy is bidimensional: the expected loss. Expected loss is the negative part of the leading arm's uplift distribution. 

Helga is not prone to the Squid's simplified distribution model aberrations, because it fairly samples from the relevant beta distributions.
Unfortunately, it is unstable on small batches (<100).

## Multi-shot allocation optimization

