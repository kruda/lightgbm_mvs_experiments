# lightgbm_mvs_experiments
MVS experiments in lightgbm
hyperparameters for baseline, datasets and data preprocessing are taken from https://github.com/catboost/benchmarks/tree/master/quality_benchmarks
Hyperamaters for MVS and GOSS are taken from https://arxiv.org/abs/1910.13204
Summary table (columns subsample size):
|0.05|0.1625|0.275|0.3875|0.5|0.6125|0.725|0.8375|0.95|
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|mvs|8.314|3.032|0.775|0.061|-0.391|-0.403|-0.26|-0.185|0.084|
|sgb|13.037|4.516|3.169|2.569|2.105|1.521|1.08|0.343|-0.155|
|goss|17.263|4.672|3.285|2.223|1.196|0.488|0.007|-0.05|-0.079|
|mvs_adaptive|8.314|2.919|0.904|-0.195|-0.451|-0.466|-0.274|-0.079|0.206|

