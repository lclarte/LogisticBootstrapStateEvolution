using LogisticBootstrapStateEvolution.New: state_evolution, to
using StaticArrays
using TimerOutputs

alpha  = 1.0
lambda = 0.5

@time state_evolution(alpha, lambda, 6, max_iteration=10);
@profview state_evolution(alpha, lambda, 3, max_iteration=5)
# @profview_allocs state_evolution(alpha, lambda, 3, max_iteration=5)
