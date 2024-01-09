using LogisticBootstrapStateEvolution

alpha = 2.0
lambda= 1.0
@time LogisticBootstrapStateEvolution.state_evolution_bootstrap_bootstrap(alpha, lambda, max_weight=10, reltol=1e-4)
@time LogisticBootstrapStateEvolution.state_evolution_bootstrap_full(alpha, lambda, max_weight=10, reltol=1e-4)

alpha = 1.0
lambda= 1e-3
@time LogisticBootstrapStateEvolution.state_evolution_bootstrap_bootstrap(alpha, lambda, max_weight=10, reltol=1e-4)