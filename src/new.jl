module New

export state_evolution

using ForwardDiff
using Integrals
using LinearAlgebra
using LineSearches
using Optim
using QuadGK
using SpecialFunctions
using StaticArrays
using StatsFuns: normpdf, poispdf
using TimerOutputs

const to = TimerOutput()

# 5.0 seems good enough for now :) and is twice faster than Inf 
const Bound = 7.5
const LogisticProbitFactor = 0.5875651988237005

function weights_proba_function_bootstrap(w1::Number, w2::Number)
    return poispdf(1, w1) * poispdf(1, w2)
end

function sigmoid(x::Number)
    return inv(1 + exp(-x))
end

function gradient_logistic_loss(y::Number, z::AbstractVector, weights::AbstractVector)
    return y .* weights .* (1 .- sigmoid.(y .* z))
end

function hessian_logistic_loss(y::Number, z::AbstractVector, weights::AbstractVector)
    return Diagonal(weights ./ (cosh.(y .* z / 2.0)) .^ 2 ./ 4)
end

@timeit to function prox_logistic_multivariate(
    y::Number, omega::AbstractVector, v_inv::AbstractMatrix, weights::AbstractVector
)
    if all(iszero, weights)
        return omega
    end

    @timeit to "objective" function objective(z::AbstractVector)
        a = dot(weights, log1p.(exp.(-y .* z)))
        b = dot(z - omega, v_inv * (z - omega)) / 2.0
        return a + b
    end

    @timeit to "gradient" function objective_grad!(g::AbstractVector, z::AbstractVector)
        ForwardDiff.gradient!(g, objective, z)
        return nothing
    end

    @timeit to "hessian" function objective_hess!(h::AbstractMatrix, z::AbstractVector)
        ForwardDiff.hessian!(h, objective, z)
        return nothing
    end

    init = Vector(omega)
    algo = Newton(;
        alphaguess=LineSearches.InitialStatic(), linesearch=LineSearches.Static()
    )
    algo = BFGS(;
        alphaguess=LineSearches.InitialStatic(), linesearch=LineSearches.BackTracking()
    )
    options = Optim.Options(;
        store_trace=false, x_tol=0.0, f_tol=0.0, g_tol=1e-8, iterations=1000
    )
    res = @timeit to "optimize" optimize(
        objective,  #
        # objective_grad!,  #
        # objective_hess!,  #
        init,  #
        algo,  #
        options,  #
        autodiff=:forward,  #
    )
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
    result = quadgk(integrand, -Bound, Bound)[1]
    return result / sqrt(variance)
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
    return logistic_z0_approximate(y, conditional_mean, v_star_float) * g * g'
end

function update_qhat(
    m::AbstractVector, q::AbstractMatrix, v::AbstractMatrix, rho, max_weight=2
)
    result = MMatrix{2,2}(zeros(2, 2))
    q_sqrt = sqrt(q)
    q_inv_sqrt = inv(q_sqrt)
    v_inv = inv(v)
    q_inv = inv(q)
    v_star_float = rho - m' * q_inv * m

    for w1 in 0:(max_weight - 1)
        for w2 in 0:(max_weight - 1)
            weights = SVector(w1, w2)
            for label in (-1, 1)
                @timeit to "integrand" function fq(x, p)
                    a = integrand_qhat(
                        x, label, m, q_sqrt, q_inv_sqrt, v_inv, v_star_float, weights
                    )
                    b = prod(normpdf, x)
                    return a * b
                end
                prob = IntegralProblem(fq, SVector(-Bound, -Bound), SVector(Bound, Bound))
                sol = @timeit to "solve" solve(prob, HCubatureJL(); reltol=1e-3)
                result += sol.u * weights_proba_function_bootstrap(w1, w2)
            end
        end
    end

    return result
end

## 

function integrand_mhat(
    x::AbstractVector,
    y::Number,
    m::AbstractVector,
    q_sqrt::AbstractMatrix,
    q_inv_sqrt::AbstractMatrix,
    v_inv::AbstractMatrix,
    v_star_float::Number,
    weights::AbstractVector,
)
    omega = q_sqrt * x
    conditional_mean = m' * q_inv_sqrt * x
    g = gout_logistic_multivariate(y, omega, v_inv, weights)
    return logistic_dz0(y, conditional_mean, v_star_float) * g
end

function update_mhat(
    m::AbstractVector, q::AbstractMatrix, v::AbstractMatrix, rho, max_weight=2
)
    result = Vector{Float64}(zeros(2))
    q_sqrt = sqrt(q)
    q_inv = inv(q)
    q_inv_sqrt = inv(q_sqrt)
    v_inv = inv(v)
    v_star_float = rho - m' * q_inv * m

    for w1 in 0:(max_weight - 1)
        for w2 in 0:(max_weight - 1)
            weights = SVector(w1, w2)
            for label in (-1, 1)
                @timeit to "integrand" function fm(x, p)
                    a = integrand_mhat(
                        x, label, m, q_sqrt, q_inv_sqrt, v_inv, v_star_float, weights
                    )
                    b = prod(normpdf, x)
                    return a * b
                end
                prob = IntegralProblem(fm, SVector(-Bound, -Bound), SVector(Bound, Bound))
                sol = @timeit to "solve" solve(prob, HCubatureJL(); reltol=1e-3)
                result += sol.u * weights_proba_function_bootstrap(w1, w2)
            end
        end
    end

    return result
end

# 

function integrand_vhat(
    x::AbstractVector,
    y::Number,
    m::AbstractVector,
    q_sqrt::AbstractMatrix,
    q_inv_sqrt::AbstractMatrix,
    v::AbstractMatrix,
    v_inv::AbstractMatrix,
    v_star_float,
    weights::AbstractVector,
)
    #= 
    Compute the 2x2 qhat matrix and 2 x 2 vhat matrix, returns a 4 x 2 matrix so that 
    we can integrate both at the same time
    =#
    omega = q_sqrt * x
    conditional_mean = m' * q_inv_sqrt * x
    dg = dwgout_logistic_multivariate(y, omega, v_inv, weights, v)
    return logistic_z0(y, conditional_mean, v_star_float) * dg
end

function update_vhat(
    m::AbstractVector, q::AbstractMatrix, v::AbstractMatrix, rho, max_weight=2
)
    result = MMatrix{2,2}(zeros(2, 2))
    q_sqrt = sqrt(q)
    q_inv = inv(q)
    q_inv_sqrt = inv(q_sqrt)
    v_inv = inv(v)
    v_star_float = rho - m' * q_inv * m

    for w1 in 0:(max_weight - 1)
        for w2 in 0:(max_weight - 1)
            weights = SVector(w1, w2)
            for label in (-1, 1)
                @timeit to "integrand" function fv(x, p)
                    a = integrand_vhat(
                        x, label, m, q_sqrt, q_inv_sqrt, v, v_inv, v_star_float, weights
                    )
                    b = prod(normpdf, x)
                    return a * b
                end
                prob = IntegralProblem(fv, SVector(-Bound, -Bound), SVector(Bound, Bound))
                sol = @timeit to "solve" solve(prob, HCubatureJL(); reltol=1e-3)
                result += sol.u * weights_proba_function_bootstrap(w1, w2)
            end
        end
    end
    return result
end

## functions to update the overlap

function update_overlaps(
    mhat::AbstractVector, qhat::AbstractMatrix, vhat::AbstractMatrix, lambda::Number
)
    tmp = inv(lambda * I + vhat)
    m = tmp * mhat
    q = tmp * (mhat * mhat' + qhat) * tmp'
    v = tmp

    return m, q, v
end

function state_evolution(
    sampling_ratio, regularisation, max_weight=2; relative_tolerance=1e-4, max_iteration=100
)
    reset_timer!(to)

    rho = 1.0

    old_m = SVector(0.0, 0.0)
    m = SVector(0.0, 0.0)
    q = SMatrix{2,2}([1.0 0.01; 0.01 1.0])
    v = SMatrix{2,2}([1.0 0.01; 0.01 1.0])

    mhat = SVector(0.0, 0.0)
    qhat = SMatrix{2,2}([1.0 0.01; 0.01 1.0])
    vhat = SMatrix{2,2}([1.0 0.01; 0.01 1.0])

    for i in 0:max_iteration
        # copy m into old_m to compute the difference at the end of the loop
        old_m = copy(m)

        mhat = sampling_ratio * update_mhat(m, q, v, rho, max_weight)
        qhat = sampling_ratio * update_qhat(m, q, v, rho, max_weight)
        vhat = sampling_ratio * update_vhat(m, q, v, rho, max_weight)

        m, q, v = update_overlaps(mhat, qhat, vhat, regularisation)

        # compute the relative difference between old and new m 
        difference = norm(m - old_m) / norm(m)
        if difference < relative_tolerance
            show(to)
            println()
            return m, q, v, mhat, qhat, vhat
        end
    end

    println("Warning: state evolution did not converge after $max_iteration iterations")
    show(to)
    println()
    return m, q, v, mhat, qhat, vhat
end

end
