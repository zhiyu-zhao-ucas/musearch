# `GwgPairSampler` 
## `_calc_local_diff` 
- need autogradient
- input is an one-hot tensor
- output is a tensor
## `_gibbs_sampler`
- input is an one-hot tensor
- use `_calc_local_diff` to calculate the logits
- use Gibbs sampler function for proposing mutations
## `_make_one_hot`
- input is a seq: Tensor of sequence indices
- output is a one-hot tensor
## `_evaluate_one_hot`
- evaluate predictor on one-hot tensor
## `_decode`
- decode one-hot tensor to sequence using Encoder
## `_metropolis_hastings`
- input:
    - mutants: Proposed mutant sequences
    - source_one_hot: One-hot encoding of source sequence
    - delta_score: Change in predictor score for mutants
- output:
    - mh_step: Boolean mask of accepted mutations
## `_evaluate_mutants`
- input:
    - mutants: Proposed mutant sequences
    - score: Score of source sequence
    - source_one_hot: One-hot encoding of source sequence
- output:
    - DataFrame containing accepted mutants and their scores, Tensor of accepted mutant sequences
- call `_metropolis_hastings` to accept mutations
- delta_score is the difference between the score of the mutant and the source sequence
## `compute_mutant_stats`
- Compute number of mutations in proposed sequences
## `forward`
- input:
    - batch: Dictionary containing input sequences (not one-hot encoded)
- output:
    - DataFrame of accepted mutant pairs and their scores
    - Overall acceptance rate
### pipeline
- get one-hot encoding
- get predictor score
- propose mutations using Gibbs sampler
---
# CheckList
- Is the CNN input one-hot encoded?
    - If not, construct a similar one-hot encoding CNN


