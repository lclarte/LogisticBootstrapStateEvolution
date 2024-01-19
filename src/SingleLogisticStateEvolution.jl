#=
Code for state evolution of a single learner in logistic regression with a logistic teacher
=#

module SingleLogisticStateEvolution

include("LogisticChannel.jl")

using StatsFuns: normpdf, poispdf, logistic
using QuadGK

# TODO : Normally the bound should be at infinity
const Bound = 10.0

function weights_proba_function_bootstrap(w::Number)
    return poispdf(1, w)
end

function update_mhat(m::Number, q::Number, v::Number, weight_range, weight_function::Function; rho = 1.0)
    v_star::Number = rho - m^2 / q

    result = 0.0
    for weight in weight_range
        for label in [-1, 1]
            result += quadgk(
                z -> LogisticChannel.logistic_dz0(label, m / sqrt(q) * z, v_star) * LogisticChannel.gout_logistic_univariate(label, sqrt(q) * z, v, weight) * normpdf(z),
                -Bound,
                Bound,
            )[1] * weight_function(weight)
        end
    end
    return result
end

function update_qhat(m::Number, q::Number, v::Number, weight_range, weight_function::Function; rho = 1.0)
    v_star::Number = rho - m^2 / q

    result = 0.0
    for weight in weight_range
        for label in [-1, 1]
            result += quadgk(
                z -> LogisticChannel.logistic_z0_approximate(label, m / sqrt(q) * z, v_star) * LogisticChannel.gout_logistic_univariate(label, sqrt(q) * z, v, weight)^2 * normpdf(z),
                -Bound,
                Bound,
            )[1] * weight_function(weight)
        end
    end
    return result
end

function update_vhat(m::Number, q::Number, v::Number, weight_range, weight_function::Function; rho = 1.0)
    v_star::Number = rho - m^2 / q

    result = 0.0
    for weight in weight_range
        for label in [-1, 1]
            result += quadgk(
                z -> LogisticChannel.logistic_z0_approximate(label, m / sqrt(q) * z, v_star) * LogisticChannel.dwgout_logistic_univariate(label, sqrt(q) * z, v, weight) * normpdf(z),
                -Bound,
                Bound,
            )[1] * weight_function(weight)
        end
    end
    return -result
end

function update_m(mhat::Number, qhat::Number, vhat::Number, lambda::Number; rho = 1.0)
    return rho * mhat / (lambda + vhat)
end

function update_q(mhat::Number, qhat::Number, vhat::Number, lambda::Number; rho = 1.0)
    return (rho .* mhat^2 + qhat) / (lambda + vhat)^2
end

function update_v(mhat::Number, qhat::Number, vhat::Number, lambda::Number; rho = 1.0)
    return 1.0 / (lambda + vhat)
end

### 

function state_evolution_bootstrap(sampling_ratio::Number, regularisation::Number; rho = 1.0, max_weight::Int=6, max_iteration::Int=1000, reltol::Number=1e-3,
    minit::Union{Nothing, Number}=nothing, qinit::Union{Nothing, Number}=nothing, vinit::Union{Nothing, Number}=nothing)
    #= 
    Code for state evolution of a single learner in logistic regression with a logistic teacher with bootstrap resamples
    =#
    m::Number = minit === nothing ? 0.00 : minit
    q::Number = qinit === nothing ? 1.0  : qinit
    v::Number = vinit === nothing ? 1.0  : vinit

    bootstrap_range = 0:max_weight

    for iteration in 0:max_iteration
        old_q = q

        qhat = sampling_ratio * update_qhat(m, q, v, bootstrap_range, weights_proba_function_bootstrap, rho = rho)
        vhat = sampling_ratio * update_vhat(m, q, v, bootstrap_range, weights_proba_function_bootstrap, rho = rho)
        mhat = sampling_ratio * update_mhat(m, q, v, bootstrap_range, weights_proba_function_bootstrap, rho = rho)

        m = update_m(mhat, qhat, vhat, regularisation, rho = rho)
        q = update_q(mhat, qhat, vhat, regularisation, rho = rho)
        v = update_v(mhat, qhat, vhat, regularisation, rho = rho)

        if abs(q - old_q) / abs(q) < reltol
            return Dict(
                "m" => m,
                "q" => q,
                "v" => v,
                "mhat" => mhat,
                "qhat" => qhat,
                "vhat" => vhat,
            )
        end
    end

    println("Warning: state evolution did not converge in $max_iteration iterations")
    return Dict(
                "m" => m,
                "q" => q,
                "v" => v,
                "mhat" => mhat,
                "qhat" => qhat,
                "vhat" => vhat,
            )
end

function state_evolution(sampling_ratio::Number, regularisation::Number; rho = 1.0, max_iteration=1000, reltol=1e-3,
    minit::Union{Nothing, Number}=nothing, qinit::Union{Nothing, Number}=nothing, vinit::Union{Nothing, Number}=nothing)
    #= 
    Code for state evolution of a single learner in logistic regression with a logistic teacher with bootstrap resamples
    =#
    m::Number = minit === nothing ? 0.00 : minit
    q::Number = qinit === nothing ? 1.0 : qinit
    v::Number = vinit === nothing ? 1.0 : vinit

    mhat::Number = 0.0
    qhat::Number = 0.0
    vhat::Number = 0.0

    erm_range = [1]
    erm_weight_function = z -> 1.0

    for iteration in 0:max_iteration
        old_q = q
        
        mhat = sampling_ratio * update_mhat(m, q, v, erm_range, erm_weight_function, rho = rho)
        qhat = sampling_ratio * update_qhat(m, q, v, erm_range, erm_weight_function, rho = rho)
        vhat = sampling_ratio * update_vhat(m, q, v, erm_range, erm_weight_function, rho = rho)
        
        m = update_m(mhat, qhat, vhat, regularisation, rho = rho)
        q = update_q(mhat, qhat, vhat, regularisation, rho = rho)
        v = update_v(mhat, qhat, vhat, regularisation, rho = rho)
        
        if abs(q - old_q) / abs(q) < reltol
            return Dict(
                "m" => m,
                "q" => q,
                "v" => v,
                "mhat" => mhat,
                "qhat" => qhat,
                "vhat" => vhat,
            )
        end
    end

    println("Warning: state evolution did not converge in $max_iteration iterations")
    return Dict(
                "m" => m,
                "q" => q,
                "v" => v,
                "mhat" => mhat,
                "qhat" => qhat,
                "vhat" => vhat,
            )
end

end  # module