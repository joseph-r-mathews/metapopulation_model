# Retired susceptibility/infectiousness model

The previous 51-location, three-age, four-vaccination model used one log beta,
11 free log susceptibility modifiers, and 11 free log infectiousness modifiers
per location (23 parameters; 1173 total). It was retired in favor of the
age x vaccination x variant SEIR model.

The completed September 16, 2026 learning curve used 100 global simulations,
with 80 training, 10 validation, and 10 test simulations. At the largest
training size (8160 local examples), log-parameter recovery correlations were
0.937900 for beta, 0.044539 for susceptibility, and 0.029987 for infectiousness.
Posterior/prior standard-deviation ratios were 0.421027, 1.005529, and 1.003054,
respectively. The learned modifier posteriors showed essentially no contraction.
These flow results motivated retirement; they alone do not prove structural
nonidentifiability of the mechanistic model.

Deleted artifacts: results/local_flow_pilot/, results/local_flow_scaling/,
results/stratified_smoke/ (old version), results/local_flow_training.csv,
results/local_flow_evaluation.json, results/local_flow_scaling_pipeline.log,
results/local_flow_scaling_pipeline_errors.log, and old flow checkpoints.
The earlier SIR proposal/block workflow and its partition result were also
removed from the active project. Tracked earlier code remains in Git history;
untracked generated data/checkpoints are not recoverable from Git.
