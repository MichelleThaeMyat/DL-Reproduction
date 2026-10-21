# Reference output provenance

These are outputs of the original implementation, not independent reproduction results and not values transcribed from the paper.

The MNIST files were copied unchanged from the original project's results/20261019-114047-653352/mnist_run directory. The summary CSV and run.log are the inputs used by reproduction.compare. PNGs are included for visual inspection.

The small-experiment metrics were copied from the verification capture in output/audit-20261021/reference-captured. That capture reran Himmelblau_Optimization_14.py, calr_comparison_toy.py, and calr_mnist_compare.py with one CPU thread, disabling plot saving/display. The original scripts' recorded numerical variables were exported to JSON. Those original scripts are available in the credited upstream repository; they are not included as newly authored code here.

The original MNIST archive and the small-experiment capture come from separate runs. Comparing these saved outputs does not rerun the original implementation. No upstream commit identifier was recorded in this package, so these reference files should be treated as a preserved local reference snapshot.

The retained LICENSE credits the original author. See the main README for scope, commands, and precision limitations.
