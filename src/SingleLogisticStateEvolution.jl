#=
Code for state evolution of a single learner in logistic regression with a logistic teacher
=#

module SingleLogisticStateEvolution

include("LogisticChannel.jl")

using StatsFuns: normpdf, poispdf, logistic
using QuadGK

const Bound = 10.0

function weights_proba_function_bootstrap(w::Number)
    return poispdf(1, w)
end

function update_mhat(m::Number, q::Number, v::Number, weight_range, weight_function::Function)
    rho = 1.0
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

function update_qhat(m::Number, q::Number, v::Number, weight_range, weight_function::Function)
    rho = 1.0
    v_star::Number = rho - m^2 / q

    result = 0.0
    for weight in weight_range
        for label in [-1, 1]
            result += quadgk(
                z -> LogisticChannel.logistic_z0(label, m / sqrt(q) * z, v_star) * LogisticChannel.gout_logistic_univariate(label, sqrt(q) * z, v, weight)^2 * normpdf(z),
                -Bound,
                Bound,
            )[1] * weight_function(weight)
        end
    end
    return result
end

function update_vhat(m::Number, q::Number, v::Number, weight_range, weight_function::Function)
    rho = 1.0
    v_star::Number = rho - m^2 / q

    result = 0.0
    for weight in weight_range
        for label in [-1, 1]
            result += quadgk(
                z -> LogisticChannel.logistic_z0(label, m / sqrt(q) * z, v_star) * LogisticChannel.dwgout_logistic_univariate(label, sqrt(q) * z, v, weight) * normpdf(z),
                -Bound,
                Bound,
            )[1] * weight_function(weight)
        end
    end
    return -result
end

function update_m(mhat::Number, qhat::Number, vhat::Number, lambda::Number)
    return mhat / (lambda + vhat)
end

function update_q(mhat::Number, qhat::Number, vhat::Number, lambda::Number)
    return (mhat^2 + qhat) / (lambda + vhat)^2
end

function update_v(mhat::Number, qhat::Number, vhat::Number, lambda::Number)
    return 1.0 / (lambda + vhat)
end

### 

function state_evolution_bootstrap(sampling_ratio::Number, regularisation::Number; max_weight=6, max_iteration=1000, reltol=1e-3)
    #= 
    Code for state evolution of a single learner in logistic regression with a logistic teacher with bootstrap resamples
    =#
    m = 0.01
    q = 0.99
    v = 0.99

    for iteration in 0:max_iteration
        old_q = q

        mhat = sampling_ratio * update_mhat(m, q, v, 0:max_weight, weights_proba_function_bootstrap)
        qhat = sampling_ratio * update_qhat(m, q, v, 0:max_weight, weights_proba_function_bootstrap)
        vhat = sampling_ratio * update_vhat(m, q, v, 0:max_weight, weights_proba_function_bootstrap)

        m = update_m(mhat, qhat, vhat, regularisation)
        q = update_q(mhat, qhat, vhat, regularisation)
        v = update_v(mhat, qhat, vhat, regularisation)

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

function state_evolution(sampling_ratio::Number, regularisation::Number; max_iteration=1000, reltol=1e-3)
    #= 
    Code for state evolution of a single learner in logistic regression with a logistic teacher with bootstrap resamples
    =#
    m = 0.01
    q = 0.99
    v = 0.99

    for iteration in 0:max_iteration
        old_q = q
        
        mhat = sampling_ratio * update_mhat(m, q, v, [1], z -> 1.0)
        qhat = sampling_ratio * update_qhat(m, q, v, [1], z -> 1.0)
        vhat = sampling_ratio * update_vhat(m, q, v, [1], z -> 1.0)
        
        m = update_m(mhat, qhat, vhat, regularisation)
        q = update_q(mhat, qhat, vhat, regularisation)
        v = update_v(mhat, qhat, vhat, regularisation)
        
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