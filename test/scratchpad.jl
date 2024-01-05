using LogisticBootstrapStateEvolution.New: state_evolution, to
using StaticArrays
using TimerOutputs

m_vec = SVector(0.0, 0.0);
q_mat = SMatrix{2,2}([1.0 0.01; 0.01  1.0]);
v_mat = SMatrix{2,2}([1.0 0.01; 0.01  1.0]);

alpha  = 1.0
lambda = 1.0

@time state_evolution(alpha, lambda, 3, max_iteration=5);
@profview state_evolution(alpha, lambda, 3, max_iteration=5)
# @profview_allocs state_evolution(alpha, lambda, 3, max_iteration=5)
