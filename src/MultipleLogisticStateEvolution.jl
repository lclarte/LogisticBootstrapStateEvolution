module MultipleLogisticStateEvolution

using Integrals: Integrals, HCubatureJL
using LinearAlgebra
using NLSolvers
using QuadGK
using SpecialFunctions
using StaticArrays
using StatsFuns: normpdf, poispdf, logistic

include("LogisticChannel.jl")

export state_evolution_bootstrap_bootstrap, state_evolution_bootstrap_full

# 5.0 seems good enough for now :) and is twice faster than Inf 
const Bound = 7.5
const LogisticProbitFactor = 0.5875651988237005

function weights_proba_function_bootstrap(w::Number)
    return poispdf(1, w)
end

## 

function integrand_qhat(
    x::AbstractVector,
    y::Number,
    m::AbstractVector,
    q_sqrt::AbstractMatrix,
    q_inv::AbstractMatrix,
    v::AbstractVector,
    v_star_float,
    weights::AbstractVector,
)
    omega = q_sqrt * x
    conditional_mean = m' * q_inv * omega
    g1 = LogisticChannel.gout_logistic_univariate(y, omega[1], v[1], weights[1])
    g2 = LogisticChannel.gout_logistic_univariate(y, omega[2], v[2], weights[2])
    return LogisticChannel.logistic_z0_approximate(y, conditional_mean, v_star_float) * g1 * g2
end

function update_qhat(
    m::AbstractVector, q::AbstractMatrix, v::AbstractVector, rho, weight1_range, weight2_range, weight1_function::Function, weight2_function::Function
)
    result = 0.0
    q_sqrt = sqrt(q)
    q_inv  = inv(q)
    q_inv = inv(q)
    v_star_float = rho - m' * q_inv * m

    for w1 in weight1_range
        for w2 in weight2_range
            weights = SVector(w1, w2)
            for label in (-1, 1)
                function fq(x, p)
                    a = integrand_qhat(
                        x, label, m, q_sqrt, q_inv, v, v_star_float, weights
                    )
                    b = prod(normpdf, x)
                    return a * b
                end
                prob = Integrals.IntegralProblem(
                    fq, SVector(-Bound, -Bound), SVector(Bound, Bound)
                )
                sol = Integrals.solve(prob, HCubatureJL(); reltol=1e-3)
                result += sol.u * weight1_function(w1) * weight2_function(w2)
            end
        end
    end

    return result
end

##  

function update_q(
    mhat::AbstractVector, qhat::AbstractMatrix, vhat::AbstractVector, lambda::Number
)
    tmp = diagm( 1.0 ./ (lambda .+ vhat))
    return tmp * (mhat * mhat' .+ qhat) * tmp
end

function state_evolution_bootstrap_bootstrap(
    m::AbstractVector, qdiag::AbstractVector, v::AbstractVector, mhat::AbstractVector, qhatdiag::AbstractVector, vhat::AbstractVector, sampling_ratio::Number, regularisation::Number; max_weight::Number=5, max_iteration=2, reltol=1e-3
)
    #= 
    Compute the correlation between two bootstrap resamples 
    We only need to compute the off-diagonal elements of q and qhat because the matrices V and Vhat are diagonal,
    and their diagonal is computed with the usual state evolution code.
    =#
    rho = 1.0

    q = MMatrix{2,2}(diagm(qdiag))
    qhat = MMatrix{2,2}(diagm(qhatdiag))

    weight1_range = Array(0:max_weight)
    weight2_range = 0:max_weight

    for i in 0:max_iteration
        old_q_off = q[1, 2]

        update = sampling_ratio * update_qhat(m, q, v, rho, weight1_range, weight2_range, weights_proba_function_bootstrap, weights_proba_function_bootstrap)
        qhat[1, 2] = update
        qhat[2, 1] = update

        q    = update_q(mhat, qhat, vhat, regularisation)
        if abs(q[1, 2] - old_q_off) / abs(q[1, 2]) < reltol
            return q, qhat
        end
    end

    println("Warning: state evolution did not converge after $max_iteration iterations")
    return q, qhat
end

function state_evolution_bootstrap_full(
    m::AbstractVector, qdiag::AbstractVector, v::AbstractVector, mhat::AbstractVector, qhatdiag::AbstractVector, vhat::AbstractVector, sampling_ratio::Number, regularisation::Number, max_weight::Number=5; max_iteration=2, reltol=1e-3
)
    #=
    Compute the correlation between bootstrap resamples and learner trained on the full dataset
    - 1st component corresponds to bootstrap resample, 2nd component corresponds to full dataset
    - Returns:
        * q
        * qhat 
    =#
    rho = 1.0

    q = MMatrix{2,2}(diagm(qdiag))
    qhat = MMatrix{2,2}(diagm(qhatdiag))

    weight1_range = 0:max_weight
    weight2_range = [1]

    for i in 0:max_iteration
        old_q_off = q[1, 2]

        update = sampling_ratio * update_qhat(m, q, v, rho, weight1_range, weight2_range, weights_proba_function_bootstrap, z -> 1.0)
        qhat[1, 2] = update
        qhat[2, 1] = update

        q    = update_q(mhat, qhat, vhat, regularisation)
        if abs(q[1, 2] - old_q_off) / abs(q[1, 2]) < reltol
            return q, qhat
        end
    end

    println("Warning: state evolution did not converge after $max_iteration iterations")
    return q, qhat
end

end  # module