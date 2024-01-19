-
using LogisticBootstrapStateEvolution: SingleLogisticStateEvolution

alpha = 2.0
lambda= 0.1

@time SingleLogisticStateEvolution.state_evolution(alpha, lambda, max_iteration=1000, reltol=1e-3)
@profview SingleLogisticStateEvolution.state_evolution(alpha, lambda, max_iteration=1000, reltol=1e-3)
@time SingleLogisticStateEvolution.state_evolution_bootstrap(alpha, lambda, max_weight=10, max_iteration=1000, reltol=1e-3)

alpha = 1.0
lambda= 1e-3

@time SingleLogisticStateEvolution.state_evolution(alpha, lambda, max_iteration=1000, reltol=1e-5)
@time SingleLogisticStateEvolution.state_evolution_bootstrap(alpha, lambda, max_weight=10, max_iteration=1000, reltol=1e-5)