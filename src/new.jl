module New

export state_evolution

using Distributions
using ForwardDiff
using Integrals
using LinearAlgebra
using Optim
using QuadGK
using SpecialFunctions
using StaticArrays

# 5.0 seems good enough for now :) and is twice faster than Inf 
const Bound = 5.0
const LogisticProbitFactor = 0.5875651988237005

function weights_proba_function_bootstrap(w1::Number, w2::Number)
    return pdf(Poisson(1), w1) * pdf(Poisson(1), w2)
end

function sigmoid(x::Number)
    return inv(1 + exp(-x))
end

function hessian_logistic_loss(y::Number, z::AbstractVector, weights::AbstractVector)
    return Diagonal(weights .* inv.(abs2.(cosh.(y .* z))) ./ 4)
end

function gradient_logistic_loss(y::Number, z::AbstractVector, weights::AbstractVector)
    return y .* weights .* (1 .- sigmoid.(y .* z))
end

function prox_logistic_multivariate(
    y::Number, omega::AbstractVector, v_inv::AbstractMatrix, weights::AbstractVector
)
    if all(iszero, weights)
        return omega
    end

    function prox_logistic_multivariate_objective(z::AbstractVector)
        a = dot(weights, log1p.(exp.(-y .* z)))
        b = dot(z - omega, v_inv * (z - omega)) / 2
        return a + b
    end

    res = optimize(prox_logistic_multivariate_objective, Vector(omega), LBFGS(); autodiff=:forward)
    prox = res.minimizer
    return prox
end

function gout_logistic_multivariate(
    y::Number, omega::AbstractVector, v_inv::AbstractMatrix, weights::AbstractVector
)
    prox = prox_logistic_multivariate(y, omega, v_inv, weights)
    return v_inv * (prox - omega)
end

function dwgout_logistic_multivariate(
    y::Number,
    omega::AbstractVector,
    v_inv::AbstractMatrix,
    weights::AbstractVector,
    v::AbstractMatrix,
)
    prox = prox_logistic_multivariate(y, omega, v_inv, weights)
    derivative_prox = inv(I + v * hessian_logistic_loss(y, prox, weights))
    return v_inv * (derivative_prox - I)
end

function logistic_z0(y::Number, mean::Number, variance::Number)
    # integrate the sigmoid multiplied by the Gaussian of mean and variance 
    # from -BOUND to BOUND
    f(x) = sigmoid(y * (x * sqrt(variance) + mean)) * pdf(Normal(0, 1), x)
    return quadgk(f, -Inf, Inf)[1]
end

function logistic_z0_approximate(y::Number, mean::Number, variance::Number)
    # integrate the sigmoid multiplied by the Gaussian of mean and variance 
    # from -BOUND to BOUND
    return sigmoid(y * mean / sqrt(1.0 + variance * LogisticProbitFactor^2))
end

## 

function integrand_qhat(
    x::AbstractVector,
    y::Number,
    m::AbstractVector,
    q_sqrt::AbstractMatrix,
    q_inv_sqrt::AbstractMatrix,
    v_inv::AbstractMatrix,
    v_star_float,
    weights::AbstractVector,
)
    omega = q_sqrt * x
    conditional_mean = m' * q_inv_sqrt * x
    g = gout_logistic_multivariate(y, omega, v_inv, weights)
    return logistic_z0_approximate(y, conditional_mean, v_star_float) * g[1] * g[2]
end

function update_qhat(
    m::AbstractVector, q::AbstractMatrix, v::AbstractMatrix, rho, max_weight=2
)
    result = 0.0
    q_sqrt = sqrt(q)
    q_inv_sqrt = inv(q_sqrt)
    v_inv = inv(v)

    for w1 in 0:(max_weight - 1)
        for w2 in 0:(max_weight - 1)
            weights = SVector(w1, w2)
            for label in (-1, 1)
                function f(x, p)
                    a = integrand_qhat(x, label, m, q_sqrt, q_inv_sqrt, v_inv, rho, weights)
                    b = pdf(MvNormal(SVector(0, 0), I), x)
                    return a * b
                end
                prob = IntegralProblem(f, SVector(-Bound, -Bound), SVector(Bound, Bound))
                sol = solve(prob, HCubatureJL(); reltol=1e-3)
                result += sol.u * weights_proba_function_bootstrap(w1, w2)
            end
        end
    end

    return result
end

function state_evolution(m_vec, q_mat, v_mat, rho, max_weight=2)
    qhat_1_2 = update_qhat(m_vec, q_mat, v_mat, rho, max_weight)
    return qhat_1_2
end

end