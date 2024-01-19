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
const Bound = 10.0
const LogisticProbitFactor = 0.5875651988237005

function weights_proba_function_bootstrap(w::Number)
    return poispdf(1, w)
end

## 

function integrand_qhat(
    x::AbstractVector, y::Number, m::AbstractVector, q_sqrt::AbstractMatrix, q_inv::AbstractMatrix, v::AbstractVector, v_star_float::Number, weights::AbstractVector
)
    omega = q_sqrt * x
    conditional_mean = m' * q_inv * omega
    g1 = LogisticChannel.gout_logistic_univariate(y, omega[1], v[1], weights[1])
    g2 = LogisticChannel.gout_logistic_univariate(y, omega[2], v[2], weights[2])
    return LogisticChannel.logistic_z0(y, conditional_mean, v_star_float) * g1 * g2
end

function update_qhat(
    m::AbstractVector, q::AbstractMatrix, v::AbstractVector, rho, weight1_range, weight2_range, weight_function::Function
)
    result = 0.0
    q_sqrt = sqrt(q)
    q_inv  = inv(q)
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
                result += sol.u * weight_function(w1, w2)
            end
        end
    end

    return result
end

# Below are the update function of qhat for y resampling, which is different from the usual state evolution because there are two labels we want to sum on 

function integrand_qhat_y_resampling(
    x::AbstractVector, ys::AbstractVector, m::AbstractVector, q_sqrt::AbstractMatrix, q_inv::AbstractMatrix, v::AbstractVector, v_star_float::Number
    )
    omega = q_sqrt * x
    conditional_mean = m' * q_inv * omega
    g1 = LogisticChannel.gout_logistic_univariate(ys[1], omega[1], v[1], 1.0)
    g2 = LogisticChannel.gout_logistic_univariate(ys[2], omega[2], v[2], 1.0)
    return LogisticChannel.logistic_z0_approximate(ys[1], conditional_mean, v_star_float) * LogisticChannel.logistic_z0_approximate(ys[2], conditional_mean, v_star_float) * g1 * g2 

end

function update_qhat_y_resampling(
    m::AbstractVector, q::AbstractMatrix, v::AbstractVector, rho::Number
)
    result = 0.0
    q_sqrt = sqrt(q)
    q_inv  = inv(q)
    q_inv = inv(q)
    v_star_float = rho - m' * q_inv * m

    for label1 in (-1, 1)
        for label2 in (-1, 1)
            function fq(x, p)
                labels = SVector{2}([label1, label2])
                a = integrand_qhat_y_resampling(x, labels, m, q_sqrt, q_inv, v, v_star_float)
                b = prod(normpdf, x)
                return a * b
            end
            prob = Integrals.IntegralProblem(
                fq, SVector(-Bound, -Bound), SVector(Bound, Bound)
            )
            sol = Integrals.solve(prob, HCubatureJL(); reltol=1e-3)
            result += sol.u
        end
    end
    return result
end

## Function to update q : it's the same regardless of the method use and only depends on the Rdige prior 

function update_q(
    mhat::AbstractVector, qhat::AbstractMatrix, vhat::AbstractVector, lambda::Number; rho = 1.0
)
    tmp = diagm( 1.0 ./ (lambda .+ vhat))
    return tmp * (rho .* mhat * mhat' .+ qhat) * tmp
end

###### 

function state_evolution_bootstrap_bootstrap_from_single_overlaps(
    m::AbstractVector, qdiag::AbstractVector, v::AbstractVector, mhat::AbstractVector, qhatdiag::AbstractVector, vhat::AbstractVector, sampling_ratio::Number, regularisation::Number; 
    max_weight::Number=5, max_iteration=100, reltol=1e-3, rho = 1.0, verbose = false
)
    #= 
    Compute the correlation between two bootstrap resamples 
    We only need to compute the off-diagonal elements of q and qhat because the matrices V and Vhat are diagonal,
    and their diagonal is computed with the usual state evolution code.
    =#
    q = MMatrix{2,2}(diagm(qdiag))
    qhat = MMatrix{2,2}(diagm(qhatdiag))

    weight1_range = Array(0:max_weight)
    weight2_range = 0:max_weight

    for i in 0:max_iteration
        old_q_off = q[1, 2]

        update = sampling_ratio * update_qhat(m, q, v, rho, weight1_range, weight2_range, (w1, w2) -> weights_proba_function_bootstrap(w1) * weights_proba_function_bootstrap(w2))
        qhat[1, 2] = update
        qhat[2, 1] = update

        q    = update_q(mhat, qhat, vhat, regularisation, rho = rho)
        reldiff = abs(q[1, 2] - old_q_off) / abs(q[1, 2])
        if reldiff < reltol
            if verbose
                println("Converged in $i iterations, last relative difference is $reldiff")
            end
            return q, qhat
        end
    end

    println("Warning: state evolution did not converge after $max_iteration iterations")
    return q, qhat
end

function state_evolution_bootstrap_full_from_single_overlaps(
    m::AbstractVector, qdiag::AbstractVector, v::AbstractVector, mhat::AbstractVector, qhatdiag::AbstractVector, vhat::AbstractVector, sampling_ratio::Number, regularisation::Number; 
    max_weight::Number=5, max_iteration=100, reltol=1e-3, rho = 1.0
)
    #=
    Compute the correlation between bootstrap resamples and learner trained on the full dataset
    - 1st component corresponds to bootstrap resample, 2nd component corresponds to full dataset
    - Returns:
        * q
        * qhat 
    =#
    q = MMatrix{2,2}(diagm(qdiag))
    q[1, 2] = m[1] * m[2]
    qhat = MMatrix{2,2}(diagm(qhatdiag))

    for i in 0:max_iteration
        old_q_off = q[1, 2]

        update = sampling_ratio * update_qhat(m, q, v, rho, 0:max_weight, [1], (w1, w2) -> weights_proba_function_bootstrap(w1) * (w2 == 1.0))
        qhat[1, 2] = update
        qhat[2, 1] = update

        q    = update_q(mhat, qhat, vhat, regularisation, rho = rho)
        if abs(q[1, 2] - old_q_off) / abs(q[1, 2]) < reltol
            return q, qhat
        end
    end

    println("Warning: state evolution did not converge after $max_iteration iterations")
    return q, qhat
end

function state_evolution_full_full_from_single_overlaps(
    m::AbstractVector, qdiag::AbstractVector, v::AbstractVector, mhat::AbstractVector, qhatdiag::AbstractVector, vhat::AbstractVector, sampling_ratio::Number, regularisation::Number;
    max_iteration=100, reltol=1e-3, rho = 1.0, verbose::Bool = false
)
    q       = MMatrix{2,2}(diagm(qdiag))
    q[1, 2] = m[1] * m[2] # we do this to avoid vstar to be negative maybe ? 
    qhat    = MMatrix{2,2}(diagm(qhatdiag))

    weight = 0:1
    weight_function = (w1, w2) -> (w1 == 1.0 && w2 == 0.0) || (w1 == 0.0 && w2 == 1.0) ? 0.5 : 0.0

    for i in 0:max_iteration
        old_q_off = q[1, 2]

        # We multiply the sampling ratio by 2 to be consistent with the other methods, as here we'll split the dataset into two
        update = (2.0 * sampling_ratio) * update_qhat(m, q, v, rho, weight, weight, weight_function)
        qhat[1, 2] = update
        qhat[2, 1] = update
        
        # if update^2 > abs(qhat[1] * qhat[2])
        #     update = 0.9 * sign(update) * sqrt(abs(qhat[1] * qhat[2]))
        # end

        q    = update_q(mhat, qhat, vhat, regularisation, rho = rho)
        reldiff = abs(q[1, 2] - old_q_off) / abs(q[1, 2])
        if reldiff < reltol
            if verbose
                println("Converged in $i iterations, last relative difference is $reldiff")
            end
            return q, qhat
        end
    end

    println("Warning: state evolution did not converge after $max_iteration iterations")
    return q, qhat
end

function state_evolution_y_resampling_from_single_overlaps(
    m::AbstractVector, qdiag::AbstractVector, v::AbstractVector, mhat::AbstractVector, qhatdiag::AbstractVector, vhat::AbstractVector, sampling_ratio::Number, regularisation::Number; 
        rho = 1.0, max_iteration=100, reltol=1e-3, verbose::Bool = false
)
    #=
    Compute the correlation between bootstrap resamples and learner trained on the full dataset
    - 1st component corresponds to bootstrap resample, 2nd component corresponds to full dataset
    - Returns:
        * q
        * qhat 
    =#
    q    = MMatrix{2,2}(diagm(qdiag))
    q[1, 2] = m[1] * m[2] # we do this to avoid vstar to be negative maybe ? 
    qhat = MMatrix{2,2}(diagm(qhatdiag))

    for i in 0:max_iteration
        old_q_off = q[1, 2]

        update = sampling_ratio * update_qhat_y_resampling(m, q, v, rho)
        # if update^2 > abs(qhat[1] * qhat[2])
        #     update = 0.9 * sign(update) * sqrt(abs(qhat[1] * qhat[2]))
        # end
        qhat[1, 2] = update
        qhat[2, 1] = update
        
        # NOTE : normally the update_q ensures always that the resulting matrix is positive definite, if qhat is positive definite
        # which it's not apparently
        q    = update_q(mhat, qhat, vhat, regularisation, rho = rho)

        if abs(q[1, 2] - old_q_off) / abs(q[1, 2]) < reltol
            if verbose
                println("Converged in $i iterations")
            end
            return q, qhat
        end
    end

    println("Warning: state evolution did not converge after $max_iteration iterations")
    return q, qhat
end

end  # module