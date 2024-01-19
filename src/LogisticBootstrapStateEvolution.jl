module LogisticBootstrapStateEvolution

using StaticArrays

include("SingleLogisticStateEvolution.jl")
include("MultipleLogisticStateEvolution.jl")
include("LogisticChannel.jl")

#=
The overall philosophy is to compute the s-e equations first for single estimators (i.e we don't compute their correlation)
which corresponds to computing the diagonal of the q matrix, and then computing the off-diagonal term
=#

function state_evolution_bootstrap_bootstrap(sampling_ratio::Number, regularisation::Number;
    rho = 1.0, max_iteration=100, max_weight=6, reltol=1e-3, bootstrap_minit=nothing, bootstrap_qinit=nothing, bootstrap_vinit=nothing, verbose = false)
    single_overlaps = SingleLogisticStateEvolution.state_evolution_bootstrap(sampling_ratio, regularisation, max_weight=max_weight, reltol=reltol,
                                                    minit=bootstrap_minit, qinit=bootstrap_qinit, vinit=bootstrap_vinit, rho = rho, max_iteration=max_iteration)
    m        = SVector{2}([single_overlaps["m"], single_overlaps["m"]])
    qdiag    = SVector{2}([single_overlaps["q"], single_overlaps["q"]])
    v        = SVector{2}([single_overlaps["v"], single_overlaps["v"]])
    mhat     = SVector{2}([single_overlaps["mhat"], single_overlaps["mhat"]])
    qhatdiag = SVector{2}([single_overlaps["qhat"], single_overlaps["qhat"]])
    vhat     = SVector{2}([single_overlaps["vhat"], single_overlaps["vhat"]])

    q, qhat = MultipleLogisticStateEvolution.state_evolution_bootstrap_bootstrap_from_single_overlaps(m, qdiag, v, mhat, qhatdiag, vhat, sampling_ratio, regularisation,
                                                    max_iteration=max_iteration, max_weight=max_weight, reltol=reltol, rho = rho, verbose=verbose)

    return Dict(
        "m" => m,
        "q" => q,
        "v" => v,
        "mhat" => mhat,
        "qhat" => qhat,
        "vhat" => vhat,
    )
end

# 

function state_evolution_bootstrap_full(sampling_ratio::Number, regularisation::Number;
    rho = 1.0, max_iteration=100, max_weight=6, reltol=1e-3,bootstrap_minit=nothing, bootstrap_qinit=nothing, bootstrap_vinit=nothing)
    #= 
    1st index of vectors corresponds to bootstrap, 2nd index corresponds to ERM on full dataset
    =#
    bootstrap_overlaps = SingleLogisticStateEvolution.state_evolution_bootstrap(sampling_ratio, regularisation, rho = rho, max_weight=max_weight, reltol=reltol,
                                                                                minit=bootstrap_minit, qinit=bootstrap_qinit, vinit=bootstrap_vinit, max_iteration=max_iteration)
    full_overlaps      = SingleLogisticStateEvolution.state_evolution(sampling_ratio, regularisation, reltol=reltol, max_iteration=max_iteration, rho = rho)
    
    m        = SVector{2}([bootstrap_overlaps["m"],    full_overlaps["m"]])
    qdiag    = SVector{2}([bootstrap_overlaps["q"],    full_overlaps["q"]])
    v        = SVector{2}([bootstrap_overlaps["v"],    full_overlaps["v"]])
    mhat     = SVector{2}([bootstrap_overlaps["mhat"], full_overlaps["mhat"]])
    qhatdiag = SVector{2}([bootstrap_overlaps["qhat"], full_overlaps["qhat"]])
    vhat     = SVector{2}([bootstrap_overlaps["vhat"], full_overlaps["vhat"]])

    q, qhat = MultipleLogisticStateEvolution.state_evolution_bootstrap_full_from_single_overlaps(m, qdiag, v, mhat, qhatdiag, vhat, sampling_ratio, regularisation, max_weight=max_weight, reltol=reltol, max_iteration=max_iteration)
    # return the overlaps in a dictionnary
    return Dict(
        "m" => m,
        "q" => q,
        "v" => v,
        "mhat" => mhat,
        "qhat" => qhat,
        "vhat" => vhat,
    )
end

function state_evolution_full_full(sampling_ratio::Number, regularisation::Number; 
    rho = 1.0, reltol=1e-3, max_iteration=100, verbose = false)
    overlaps = SingleLogisticStateEvolution.state_evolution(sampling_ratio, regularisation, reltol=reltol, max_iteration=max_iteration, rho = rho)
    
    m        = SVector{2}([overlaps["m"],    overlaps["m"]])
    qdiag    = SVector{2}([overlaps["q"],    overlaps["q"]])
    v        = SVector{2}([overlaps["v"],    overlaps["v"]])
    mhat     = SVector{2}([overlaps["mhat"], overlaps["mhat"]])
    qhatdiag = SVector{2}([overlaps["qhat"], overlaps["qhat"]])
    vhat     = SVector{2}([overlaps["vhat"], overlaps["vhat"]])

    q, qhat = MultipleLogisticStateEvolution.state_evolution_full_full_from_single_overlaps(m, qdiag, v, mhat, qhatdiag, vhat, sampling_ratio, regularisation,
                        reltol=reltol, max_iteration=max_iteration, verbose = verbose)
    return Dict(
        "m" => m,
        "q" => q,
        "v" => v,
        "mhat" => mhat,
        "qhat" => qhat,
        "vhat" => vhat,
    )
end

function state_evolution_y_resampling(sampling_ratio::Number, regularisation::Number;
            rho = 1.0, reltol=1e-3, max_iteration=1000, verbose = false)
    overlaps = SingleLogisticStateEvolution.state_evolution(sampling_ratio, regularisation, reltol=reltol, max_iteration=max_iteration, rho = rho)
    
    m        = SVector{2}([overlaps["m"],    overlaps["m"]])
    qdiag    = SVector{2}([overlaps["q"],    overlaps["q"]])
    v        = SVector{2}([overlaps["v"],    overlaps["v"]])
    mhat     = SVector{2}([overlaps["mhat"], overlaps["mhat"]])
    qhatdiag = SVector{2}([overlaps["qhat"], overlaps["qhat"]])
    vhat     = SVector{2}([overlaps["vhat"], overlaps["vhat"]])

    if verbose
        print("single overlaps are $overlaps\n")
    end
        
    q, qhat = MultipleLogisticStateEvolution.state_evolution_y_resampling_from_single_overlaps(m, qdiag, v, mhat, qhatdiag, vhat, sampling_ratio, regularisation,
                reltol=reltol, max_iteration=max_iteration, rho=rho, verbose=verbose)
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
