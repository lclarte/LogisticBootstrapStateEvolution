using LogisticResamplingSE: state_evolution
using StaticArrays
# Assuming state_evolution is the function you want to run
m_vec = MVector{2}([0.0, 0.0])
q_mat = MMatrix{2,2}([1.0 0.01; 0.01  1.0])
v_mat = MMatrix{2,2}([1.0 0.01; 0.01  1.0])
@time result = state_evolution(m_vec, q_mat, v_mat, 1.0, 2)
@profview for i in 1:100; state_evolution(m_vec, q_mat, v_mat, 1.0, 2); end
# Print or use the result as needed
println("Result of state_evolution: $result")
