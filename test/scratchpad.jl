using LogisticBootstrapStateEvolution

alpha  = 1.0
lambda = 2.0

state_evolution(alpha, lambda, 2, max_iteration=1);
@time state_evolution(alpha, lambda, 6, max_iteration=10);
@profview state_evolution(alpha, lambda, 3, max_iteration=5)
@profview_allocs state_evolution(alpha, lambda, 3, max_iteration=5)
