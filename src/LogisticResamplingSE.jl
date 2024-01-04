module LogisticResamplingSE

export state_evolution

using ForwardDiff
using LinearAlgebra
using QuadGK
using Integrals
using SpecialFunctions
using Optim
using Distributions
using StaticArrays

const BOUND = Inf

function sigmoid(x)
    return inv(1.0 + exp(-x))
end

function hessian_logistic_loss(y, z_vec, weights_vec)
    return diagm(weights_vec .* (1.0 ./ cosh.(y .* z_vec) .^ 2) ./ 4.0)
end

function gradient_logistic_loss(y, z_vec, weights_vec)
    return y .* weights_vec .* (1.0 .- sigmoid.(y .* z_vec))
end

function prox_logistic_multivariate(y, omega_vec, v_inv_mat, weights_vec)
    if all(iszero, weights_vec)
        return omega_vec
    end
    function aux(z_vec)
        return dot(weights_vec, log1p.(exp.(.-y .* z_vec))) + 0.5 * dot(z_vec - omega_vec, v_inv_mat * (z_vec - omega_vec))
    end
    function jac_aux!(result, z_vec)
        result .= gradient_logistic_loss(y, z_vec, weights_vec) + v_inv_mat * (z_vec - omega_vec)
    end

    # TODO : Change this 
    res = optimize(aux, omega_vec, LBFGS(); autodiff=:forward)
    return res.minimizer
end

function gout_logistic_multivariate(y, omega_vec, v_inv_mat, weights_vec)
    return v_inv_mat * (prox_logistic_multivariate(y, omega_vec, v_inv_mat, weights_vec) - omega_vec)
end

function dwgout_logistic_multivariate(y, omega_vec, v_inv_mat, weights_vec, v_mat)
    prox = prox_logistic_multivariate(y, omega_vec, v_inv_mat, weights_vec)
    derivative_prox = inv(I + v_mat * hessian_logistic_loss(y, prox, weights_vec))
    return v_inv_mat * (derivative_prox - I)
end

weights_proba_function_bootstrap(w1, w2) = pdf(Poisson(1.0), w1) * pdf(Poisson(1.0), w2)
weights_proba_function_full_resample(w1, w2) = (w1 != w2) ? 0.5 : 0.0
function get_weights_proba_function_cross_validation(k)
    @assert k >= 2 "k must be >= 2"
    function weights_proba_function(w1, w2)
        if w1 != w2
            return 1.0 / k
        elseif w1 == w2 == 0.0
            return 1.0 - 2.0 / k
        else
            return 0.0
        end
    end
    return weights_proba_function
end

function logistic_z0(y, mean, variance)
    # integrate the sigmoid multiplied by the Gaussian of mean and variance 
    # from -BOUND to BOUND
    logistic_z0_integrand(x) = sigmoid(y * (x * sqrt(variance) + mean)) * pdf(Normal(0.0, 1.0), x)
    return quadgk(logistic_z0_integrand, -BOUND, BOUND)[1]
end

## 

function integrand_qhat(x1, x2, y, m_vec, q_sqrt_mat, q_inv_sqrt_mat, v_inv_mat, v_star_float, weights_vec)
    omega = q_sqrt_mat * MVector{2,Float64}(x1, x2)
    conditional_mean = m_vec' * q_inv_sqrt_mat * MVector{2,Float64}(x1, x2)
    g = gout_logistic_multivariate(y, omega, v_inv_mat, weights_vec)
    return logistic_z0(y, conditional_mean, v_star_float) * g[1] * g[2]
end

function update_qhat(m_vec, q_mat, v_mat, rho, max_weight=2)
    result = 0.0
    q_sqrt_mat = sqrt(q_mat)
    q_inv_sqrt_mat = inv(q_sqrt_mat)
    v_inv_mat = inv(v_mat)

    dist = MvNormal(SVector{2,Float64}(0.0, 0.0), I)
    for w1 in 0:max_weight-1
        for w2 in 0:max_weight-1
            for label in (-1, 1)
                f(x, p) = integrand_qhat(x[1], x[2], label, m_vec, q_sqrt_mat, q_inv_sqrt_mat, v_inv_mat, rho, MVector{2,Float64}(w1, w2))
                lower = SVector{2,Float64}(-BOUND, -BOUND)
                upper = SVector{2,Float64}(BOUND, BOUND)
                prob = IntegralProblem(f, lower, upper)
                sol = solve(prob, HCubatureJL(); reltol=1e-3, maxiters=10_000)
                result += sol.u * weights_proba_function_bootstrap(w1, w2)
                # NOTE : For now only done for the bootstrap case
                # result += quadgk(x -> quadgk(y -> pdf(dist, MVector{2,Float64}(x, y)) *
                #                                   integrand_qhat(x, y, label, m_vec, q_sqrt_mat, q_inv_sqrt_mat, v_inv_mat, rho, MVector{2,Float64}(w1, w2)),
                #         -BOUND, BOUND, rtol=1e-3, maxevals=100)[1],
                #     -BOUND, BOUND, rtol=1e-3, maxevals=100)[1] * weights_proba_function_bootstrap(w1, w2)
            end
        end
    end

    return result
end

# 

# function integrand_vhat(x1, x2, y, m_vec, q_sqrt_mat, q_inv_sqrt_mat, v_mat, v_inv_mat, v_star_float, weights_vec)
#     omega = q_sqrt_mat * [x1, x2]
#     conditional_mean = m_vec' * q_inv_sqrt_mat * [x1, x2]
#     dg = dwgout_logistic_multivariate(y, omega, v_inv_mat, weights_vec, v_mat)

#     return logistic_z0(y, conditional_mean, v_star_float) * dg[1] * dg[2]
# end

# function update_vhat(m_vec, q_mat, v_mat, rho, max_weight=2)
#     result = 0.0
#     q_sqrt_mat = sqrt(q_mat)
#     q_inv_sqrt_mat = inv(q_sqrt_mat)
#     v_inv_mat = inv(v_mat)

#     for w1 in 0:max_weight-1
#         for w2 in 0:max_weight-1
#             for label in [-1, 1]
#                 result += quadgk(x -> quadgk(y -> pdf(MvNormal(zeros(2), I), [x, y]) *
#                                                   integrand_vhat(x, y, label, m_vec, q_sqrt_mat, q_inv_sqrt_mat, v_mat, v_inv_mat, rho, [w1, w2]),
#                         -BOUND, BOUND, rtol=1e-3, maxevals=100)[1],
#                     -BOUND, BOUND, rtol=1e-3, maxevals=100)[1]
#             end
#         end
#     end

#     return result
# end

#### 

function state_evolution(m_vec, q_mat, v_mat, rho, max_weight=2)
    qhat_1_2 = update_qhat(m_vec, q_mat, v_mat, rho, max_weight)
    # vhat_1_2 = update_vhat(m_vec, q_mat, v_mat, rho, max_weight)
    # return [qhat_1_2, vhat_1_2]
    return nothing
end

end  # module
