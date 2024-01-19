#=
loss and prox. operator for the logistic channel
=#

module LogisticChannel

using StatsFuns: normpdf, poispdf, logistic
using NLSolvers
using QuadGK
using SpecialFunctions
using StaticArrays

sigmoid(x::Number) = logistic(x)
# TODO : This should be Inf
const Bound = 10.0 
const LogisticProbitFactor = 0.5875651988237005

function gradient_logistic_loss(y::Number, z::Number)
    return (sigmoid(y * z) - 1) * y
end

function hessian_logistic_loss(z::Number)
    return 1.0 / (cosh(z / 2.0)^ 2 * 4.0)
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

    # TODO : No control on the precision of the solution
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
    y::Number, omega::Number, v::Number, weight::Number,
)
    prox = prox_logistic_univariate(y, omega, v, weight)
    derivative_prox = 1.0 / (1.0 + v * weight * hessian_logistic_loss(prox))
    return (derivative_prox - 1.0) / v
end

function logistic_z0(y::Number, mean::Number, variance::Number)
    # integrate the sigmoid multiplied by the Gaussian of mean and variance 
    # from -BOUND to BOUND
    f(x) = sigmoid(y * (x * sqrt(variance) + mean)) * normpdf(x)
    return quadgk(f, -Bound, Bound)[1]
end

# TODO : That0s a fast approximation of logistic_z0
function logistic_z0_approximate(y::Number, mean::Number, variance::Number)
    # integrate the sigmoid multiplied by the Gaussian of mean and variance 
    # from -BOUND to BOUND
    return sigmoid(y * mean / sqrt(1.0 + variance * LogisticProbitFactor^2))
end

function logistic_dz0(y::Number, mean::Number, variance::Number)::Number
    integrand(z) = z * sigmoid(y * (z * sqrt(variance) + mean)) * normpdf(z)
    result = quadgk(integrand, -Bound, Bound)[1]
    return result / sqrt(variance)
end

end 