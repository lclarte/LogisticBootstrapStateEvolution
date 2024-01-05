using LogisticBootstrapStateEvolution.New: state_evolution
using StaticArrays

m_vec = SVector(0.0, 0.0);
q_mat = SMatrix{2,2}([1.0 0.01; 0.01  1.0]);
v_mat = SMatrix{2,2}([1.0 0.01; 0.01  1.0]);

alpha  = 1.0
lambda = 1.0

result = state_evolution(m_vec, q_mat, v_mat, alpha, 2)
@time result = state_evolution(alpha, lambda, 7, max_iteration=10)
@profview state_evolution(alpha, lambda, 3, max_iteration=5)