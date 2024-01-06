using LogisticBootstrapStateEvolution
using StaticArrays

#= 
alpha  = 1.0
lambda = 2.0

# hardcoded values for this value of alpha = 1.0 and lambda = 2.0
m        = SVector{2}([ 0.07756231047458477, 0.07756231047458477 ] )
qdiag    = SVector{2}([ 0.07034273363434002, 0.07034273363434002 ])
v        = SVector{2}([ 0.4551180048548463 , 0.4551180048548463  ])
mhat     = SVector{2}([ 0.17042241714722361, 0.17042241714722361 ])
qhatdiag = SVector{2}([ 0.31055895718276305, 0.31055895718276305 ])
vhat     = SVector{2}([ 0.19723234267327278, 0.19723234267327278 ])
=#

alpha = 2.0
lambda = 0.1

m        = SVector{2}([0.7629706003134091,  0.7629706003134091])
qdiag    = SVector{2}([3.8746065374638277,  3.8746065374638277])
v        = SVector{2}([4.447796051796358,   4.447796051796358])
mhat     = SVector{2}([0.1715390254922466,  0.1715390254922466])
qhatdiag = SVector{2}([0.16643081204981966, 0.16643081204981966])
vhat     = SVector{2}([0.12483045273537757, 0.12483045273537757])


result_1 = state_evolution(m, qdiag, v, mhat, qhatdiag, vhat, alpha, lambda, 6, max_iteration=30);
result_1
@time state_evolution(m, qdiag, v, mhat, qhatdiag, vhat, alpha, lambda, 6, max_iteration=10);
result
@profview state_evolution(m, qdiag, v, mhat, qhatdiag, vhat, alpha, lambda, 6, max_iteration=10)
@profview_allocs state_evolution(alpha, lambda, 3, max_iteration=5)
