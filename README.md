# Full duplex pixel antenna codebook

Train a small set of diode configurations for specified **complex S21 environment points** using `results.mat` from `scalable_pixel_sebo/run_reduced_port_sebo.m` in [fd-pixel-capability-reduction](https://github.com/Alikhanum/fd-pixel-capability-reduction). A codeword is one physical on/off setting for every reconfigurable diode. The same settings may serve multiple environment points.

## Start

1. Run scalable SEBO and locate its `results/K_..._C_.../results.mat` file. The SEBO run evaluates the architecture and diode states; this repository selects a subset of those states. No RF Toolbox or dependency on the old repository's MATLAB functions is required for training.
2. Edit `seboResultFile` in `example_train_codebook.m`, then run that script in MATLAB. The script can be called from any working directory.
3. Inspect the output folder: `codebook.mat`, `codebook_cst_states.csv`, `environment_assignments.csv`, and both `.fig` and `.png` versions of the environment and coverage plots.

Or call the trainer directly:

```matlab
addpath('/path/to/fd-pixel-codebook');
c.resultsFile = '/path/to/scalable_pixel_sebo/results/K_10_C_18/results.mat';
c.environmentS21 = complex([0.03; -0.04],[0.12; 0.08]);
c.maxCodewords = 8;
c.residualLimitDb = -20;
c.selectionMetric = 'COVERAGE';
c.matchEnabled = true;
c.limitS11Db = -10;
c.limitS22Db = -10;
c.outputFolder = '/path/to/output';
model = train_fd_pixel_codebook(c);
out = apply_fd_pixel_codebook(model,0.03+0.12i);
```

`cfg.environmentS21` may be omitted to reuse the environment points in `results.mat`. It must contain complex **linear S21 values**, including phase, rather than dB magnitudes. The source's `centerS(2,1,1,q)` is the loaded antenna response for evaluated state `q`. For environment `e`, the estimated residual is `abs(e + centerS(2,1,1,q))`; the environment is added once. The model selects codewords only from the states evaluated in the source result. Check `results.evaluation.sampled` if the source run sampled rather than exhausted the diode combinations.

The greedy selection adds at most `maxCodewords` states. The default `COVERAGE` metric maximizes the count of training points meeting `residualLimitDb` at each step; ties prefer smaller mean residual, then smaller worst residual, then source row order. `MINIMAX` instead prioritizes the worst residual, and `MEAN` prioritizes mean residual. Greedy selection is deterministic but does not guarantee a globally optimal subset. The trainer batches candidates with `batchSize` (default 256) so it does not allocate a full point-by-state matrix. Training still takes work proportional to points × eligible states × codewords.

With `matchEnabled=true`, each codeword must meet `limitS11Db` and `limitS22Db` based on the saved antenna S-parameters. You can set `matchEnabled=false`. These limits refer to the antenna at the saved center frequency; the supplied environment perturbs S21 only. This codebook does not certify broadband performance, the influence of an object on S11/S22, or DC routing for a different architecture. The underlying architecture is the one in the saved SEBO result.

`codebook_cst_states.csv` includes `Codeword`, `SourceStateRow`, complex S-parameter components, and `LoadNNN_PortP` columns for **all original load ports**. Values 0 and 1 mean open/off and short/on respectively. The columns where `fullArchitecture==2` are the diode switches; all other columns preserve fixed states. Use this manifest with the original port order for CST cases. `environment_assignments.csv` maps every training point to a codeword and its residual. `apply_fd_pixel_codebook` maps new points to the best stored word and returns `diodeBits`, `fullLoadStates`, `touchstoneLoadPorts`, and residuals; it does not retrain.

## Verification

Run `tests/test_codebook.m` in MATLAB for a small synthetic result with a known coverage optimum and a port-mapping check. Training uses saved center-frequency responses, so validate chosen configurations with your full-network verification and CST comparison workflows before treating the predicted residual as a measurement.
