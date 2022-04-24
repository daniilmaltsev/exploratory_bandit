# exploratory_bandits
Pure exploration N-armed Bernoulli bayesian bandits aimed to minimize the experiment time regardless of the expoitation benefits.
Slow feedback loop is assumed, i.e. some number of events (batch_size) happen between the allocation decision and the feedback signal.
A batch-adjusted Thompson bandit is provided for a baseline.
All the bandits in this repository share a similar interface:
Input: conversions vector and n_users vector (each vector contains the values for all arms in the same order), batch_size.
Output: allocation vector with suggested traffic fractions to be allocated to each group.

## Quantile intersection gradient bandit

## Expected loss gradient bandit

## Multi-shot allocation optimization
