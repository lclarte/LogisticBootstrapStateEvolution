module LogisticBootstrapStateEvolution

using StaticArrays

include("SingleLogisticStateEvolution.jl")
include("MultipleLogisticStateEvolution.jl")
include("LogisticChannel.jl")

function state_evolution_bootstrap_bootstrap(sampling_ratio::Number, regularisation::Number; max_weight=6, reltol=1e-3, bootstrap_minit=nothing, bootstrap_qinit=nothing, bootstrap_vinit=nothing)
    single_overlaps = SingleLogisticStateEvolution.state_evolution_bootstrap(sampling_ratio, regularisation, max_weight=max_weight, reltol=reltol,
                                                                            minit=bootstrap_minit, qinit=bootstrap_qinit, vinit=bootstrap_vinit)
    m        = SVector{2}([single_overlaps["m"], single_overlaps["m"]])
    qdiag    = SVector{2}([single_overlaps["q"], single_overlaps["q"]])
    v        = SVector{2}([single_overlaps["v"], single_overlaps["v"]])
    mhat     = SVector{2}([single_overlaps["mhat"], single_overlaps["mhat"]])
    qhatdiag = SVector{2}([single_overlaps["qhat"], single_overlaps["qhat"]])
    vhat     = SVector{2}([single_overlaps["vhat"], single_overlaps["vhat"]])

    q, qhat = MultipleLogisticStateEvolution.state_evolution_bootstrap_bootstrap_from_single_overlaps(m, qdiag, v, mhat, qhatdiag, vhat, sampling_ratio, regularisation, max_weight=max_weight, reltol=reltol)
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

# 

function state_evolution_bootstrap_full(sampling_ratio::Number, regularisation::Number; max_weight=6, reltol=1e-3,
    bootstrap_minit=nothing, bootstrap_qinit=nothing, bootstrap_vinit=nothing)
    #= 
    1st index of vectors corresponds to bootstrap, 2nd index corresponds to ERM on full dataset
    =#
    bootstrap_overlaps = SingleLogisticStateEvolution.state_evolution_bootstrap(sampling_ratio, regularisation, max_weight=max_weight, reltol=reltol,
                                                                                minit=bootstrap_minit, qinit=bootstrap_qinit, vinit=bootstrap_vinit)
    full_overlaps      = SingleLogisticStateEvolution.state_evolution(sampling_ratio, regularisation, reltol=reltol)
    
    m        = SVector{2}([bootstrap_overlaps["m"],    full_overlaps["m"]])
    qdiag    = SVector{2}([bootstrap_overlaps["q"],    full_overlaps["q"]])
    v        = SVector{2}([bootstrap_overlaps["v"],    full_overlaps["v"]])
    mhat     = SVector{2}([bootstrap_overlaps["mhat"], full_overlaps["mhat"]])
    qhatdiag = SVector{2}([bootstrap_overlaps["qhat"], full_overlaps["qhat"]])
    vhat     = SVector{2}([bootstrap_overlaps["vhat"], full_overlaps["vhat"]])

    q, qhat = MultipleLogisticStateEvolution.state_evolution_bootstrap_bootstrap_from_single_overlaps(m, qdiag, v, mhat, qhatdiag, vhat, sampling_ratio, regularisation, max_weight=max_weight, reltol=reltol)
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

function state_evolution_y_resampling(sampling_ratio::Number, regularisation::Number; reltol=1e-3)
    overlaps = SingleLogisticStateEvolution.state_evolution(sampling_ratio, regularisation, reltol=reltol)
    
    m        = SVector{2}([bootstrap_overlaps["m"],    full_overlaps["m"]])
    qdiag    = SVector{2}([bootstrap_overlaps["q"],    full_overlaps["q"]])
    v        = SVector{2}([bootstrap_overlaps["v"],    full_overlaps["v"]])
    mhat     = SVector{2}([bootstrap_overlaps["mhat"], full_overlaps["mhat"]])
    qhatdiag = SVector{2}([bootstrap_overlaps["qhat"], full_overlaps["qhat"]])
    vhat     = SVector{2}([bootstrap_overlaps["vhat"], full_overlaps["vhat"]])

    q, qhat = MultipleLogisticStateEvolution.state_evolution_y_resampling_from_single_overlaps(m, qdiag, v, mhat, qhatdiag, vhat, sampling_ratio, regularisation, reltol=reltol)
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
