module LogisticBootstrapStateEvolution

using StaticArrays

include("SingleLogisticStateEvolution.jl")
include("MultipleLogisticStateEvolution.jl")
include("LogisticChannel.jl")

function state_evolution_bootstrap_bootstrap(sampling_ratio::Number, regularisation::Number; max_weight=6, reltol=1e-3)
    single_overlaps = SingleLogisticStateEvolution.state_evolution_bootstrap(sampling_ratio, regularisation, max_weight=max_weight, reltol=1e-3)
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

end  # module
