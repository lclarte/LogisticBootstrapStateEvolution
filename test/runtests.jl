using LogisticBootstrapStateEvolution.New: state_evolution
using LogisticBootstrapStateEvolution.Old: old_state_evolution
using StaticArrays
using Test

m_vec = MVector(0.0, 0.0);
q_mat = MMatrix{2,2}([1.0 0.01; 0.01  1.0]);
v_mat = MMatrix{2,2}([1.0 0.01; 0.01  1.0]);
result = state_evolution(m_vec, q_mat, v_mat, 1.0, 2)

m_vec_old = [0.0, 0.0];
q_mat_old = [1.0 0.01; 0.01  1.0];
v_mat_old = [1.0 0.01; 0.01  1.0];
result_old = old_state_evolution(m_vec_old, q_mat_old, v_mat_old, 1.0, 2)

@test result ≈ result_old rtol=1e-3
