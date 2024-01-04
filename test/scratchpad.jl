using LogisticBootstrapStateEvolution.New: state_evolution
using StaticArrays

m_vec = MVector(0.0, 0.0);
q_mat = MMatrix{2,2}([1.0 0.01; 0.01  1.0]);
v_mat = MMatrix{2,2}([1.0 0.01; 0.01  1.0]);
result = state_evolution(m_vec, q_mat, v_mat, 1.0, 2)
@time result = state_evolution(m_vec, q_mat, v_mat, 1.0, 2)
@profview for i in 1:10; state_evolution(m_vec, q_mat, v_mat, 1.0, 2); end
