using LogisticBootstrapStateEvolution
using LogisticBootstrapStateEvolution: MultipleLogisticStateEvolution
using StaticArrays

@profview LogisticBootstrapStateEvolution.state_evolution_bootstrap_bootstrap(1.0, 2.0; max_weight=6, reltol=1e-2)