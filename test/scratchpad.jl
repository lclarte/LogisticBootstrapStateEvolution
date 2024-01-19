using LogisticBootstrapStateEvolution
using LogisticBootstrapStateEvolution: MultipleLogisticStateEvolution, SingleLogisticStateEvolution
using StaticArrays
using Plots

@profview LogisticBootstrapStateEvolution.state_evolution_bootstrap_bootstrap(1.0, 2.0; max_weight=6, reltol=1e-2)

lambda = 1e-1
overlaps = []
alpha_range = 1.0:0.25:6.0
for alpha in alpha_range
    println("alpha = $alpha")
    res = SingleLogisticStateEvolution.state_evolution(alpha, lambda; rho = 1.0, max_iteration=100, reltol=1e-4)
    push!(overlaps, res)
end

print([(a, o["q"]) for (a, o) in zip(alpha_range, overlaps)])