module LogisticBootstrapStateEvolution

using Integrals: Integrals, HCubatureJL
using LinearAlgebra
using NLSolvers
using QuadGK
using SpecialFunctions
using StaticArrays
using StatsFuns: normpdf, poispdf, logistic

export state_evolution

# 5.0 seems good enough for now :) and is twice faster than Inf 
const Bound = 7.5
const LogisticProbitFactor = 0.5875651988237005

sigmoid(x::Number) = logistic(x)

function weights_proba_function_bootstrap(w1::Number, w2::Number)
    return poispdf(1, w1) * poispdf(1, w2)
end

function gradient_logistic_loss(y::Number, z::Number)
    return (sigmoid(y * z) - 1) * y
end

function hessian_logistic_loss(z::Number)
    return 1.0 ./ (cosh.(z ./ 2.0) .^ 2 .* 4.0)
end

function prox_logistic_univariate(
    y::Number, omega::Number, v::Number, weight::Number
)
    if weight == 0.0
        return omega
    end

    function objective(z::Number)
        a = weight * log1p(exp(-y * z))
        b = (z - omega)^2 / (2.0 * v)
        return a + b
    end

    function gradient(g::Number, z::Number)
        ga = weight * gradient_logistic_loss(y, z)
        gb = (z - omega) / v
        g = ga + gb
        return g
    end

    function hessian(H::Number, z::Number)
        Ha = weight * hessian_logistic_loss(z)
        Hb = 1.0 / v
        H = Ha + Hb
        return H
    end

    f(x) = objective(x)
    fgh(g, H, x) = objective(x), gradient(g, x), hessian(H, x)

    scalarobj = NLSolvers.ScalarObjective(; f=objective, fgh)
    optprob = NLSolvers.OptimizationProblem(scalarobj; inplace=false)
    init = omega
    res = NLSolvers.solve(
        optprob,
        init,
        NLSolvers.LineSearch(NLSolvers.Newton()),
        NLSolvers.OptimizationOptions(),
    )
    return res.info.solution
end

function gout_logistic_univariate(
    y::Number, omega::Number, v::Number, weight::Number
)
    return (prox_logistic_univariate(y, omega, v, weight) - omega) / v
end

function dwgout_logistic_univariate(
    y::Number,
    omega::Number,
    v::Number,
    weight::Number,
)
    prox::MVector{2} = prox_logistic_univariate(y, omega, v, weight)
    derivative_prox = 1.0 / (1.0 + v * weight * hessian_logistic_loss(prox))
    return (derivative_prox - 1.0) / v
end

function logistic_z0(y::Number, mean::Number, variance::Number)
    # integrate the sigmoid multiplied by the Gaussian of mean and variance 
    # from -BOUND to BOUND
    f(x) = sigmoid(y * (x * sqrt(variance) + mean)) * normpdf(x)
    return quadgk(f, -10.0, 10.0)[1]
end

function logistic_z0_approximate(y::Number, mean::Number, variance::Number)
    # integrate the sigmoid multiplied by the Gaussian of mean and variance 
    # from -BOUND to BOUND
    return sigmoid(y * mean / sqrt(1.0 + variance * LogisticProbitFactor^2))
end

function logistic_dz0(y::Number, mean::Number, variance::Number)::Number
    integrand(z) = z * sigmoid(y * (z * sqrt(variance) + mean)) * normpdf(z)
    result = quadgk(integrand, -10.0, 10.0)[1]
    return result / sqrt(variance)
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
    g1 = gout_logistic_univariate(y, omega[1], v[1], weights[1])
    g2 = gout_logistic_univariate(y, omega[2], v[2], weights[2])
    return logistic_z0_approximate(y, conditional_mean, v_star_float) * g1 * g2
end

function update_qhat(
    m::AbstractVector, q::AbstractMatrix, v::AbstractVector, rho, max_weight=2
)
    result = 0.0
    q_sqrt = sqrt(q)
    q_inv  = inv(q)
    q_inv = inv(q)
    v_star_float = rho - m' * q_inv * m

    for w1 in 0:(max_weight - 1)
        for w2 in 0:(max_weight - 1)
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
                result += sol.u * weights_proba_function_bootstrap(w1, w2)
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

function state_evolution(
    m::AbstractVector, qdiag::AbstractVector, v::AbstractVector, mhat::AbstractVector, qhatdiag::AbstractVector, vhat::AbstractVector, sampling_ratio::Number, regularisation::Number, max_weight::Number=5; max_iteration=2
)
    rho = 1.0

    q = MMatrix{2,2}(diagm(qdiag))
    qhat = MMatrix{2,2}(diagm(qhatdiag))

    for i in 0:max_iteration
        # copy m into old_m to compute the difference at the end of the loop

        update = sampling_ratio * update_qhat(m, q, v, rho, max_weight)
        qhat[1, 2] = update
        qhat[2, 1] = update

        q    = update_q(mhat, qhat, vhat, regularisation)
    end

    println("Warning: state evolution did not converge after $max_iteration iterations")
    return q, qhat
end

end  # module
