using LogisticBootstrapStateEvolution

alpha  = 1.0
lambda = 0.5

result_1 = state_evolution(alpha, lambda, 2, max_iteration=10);
@time result = state_evolution(alpha, lambda, 6, max_iteration=20);
result
@profview state_evolution(alpha, lambda, 3, max_iteration=5)
@profview_allocs state_evolution(alpha, lambda, 3, max_iteration=5)
