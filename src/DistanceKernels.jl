module DistanceKernels

using Profile, PProf
using Statistics
using ChainRules
using Zygote
using CUDA
using CUDAKernels
using KernelAbstractions
using KernelGradients
using Tullio

newaxis = [CartesianIndex()]

CUDA.allowscalar(false)

const AAAF = AbstractArray{<:AbstractFloat}

fp32(x::CuArray) = convert(CuArray{Float32}, x)
fp32(x::Array) = convert(Array{Float32}, x)

#######################################################################
# Euclidean Distance using Linear Algebra Primitives
#######################################################################

function euclidean_distance(x::AAAF, y::AAAF)
    x² = sum(x.^2, dims=1)
    y² = sum(y.^2, dims=1)
    xy = x' * y
    return sqrt.(x²' .+ y² .- 2xy)
end

function euclidean_distance2_zygote(w::AAAF, x::AAAF)
    @tullio result[i, j] := (w[k, i] - x[k, j])^2
    return result
end

#######################################################################
# Euclidean Distance using Kernels
#######################################################################

function euclidean_distance2(w::AAAF, x::AAAF)
    @tullio grad=false result[i, j] := (w[k, i] - x[k, j])^2
    return result
end

function euclidean_distance2_pb(w, x)
    result = euclidean_distance2(w, x)
    return result, (Δresult) -> begin
        @assert size(result) == size(Δresult) "$size(result) != $size(Δresult)"
        @tullio grad=false Δw[k, i] := Δresult[i, j] * 2 * (w[k, i] - x[k, j])
        @tullio grad=false Δx[k, j] := - Δresult[i, j] * 2 * (w[k, i] - x[k, j])
        return (ChainRules.NoTangent(), Δw, Δx)
    end
end

ChainRules.rrule(::typeof(euclidean_distance2), w, x) = euclidean_distance2_pb(w, x)

#######################################################################
# p-Distance
#######################################################################

function p_distance_zygote(w::AAAF, x::AAAF; p::Int64=2)
    @tullio result[i, j] := abs(w[k, i] - x[k, j])^p
    return result
end

function p_distance(w::AAAF, x::AAAF; p::Int64=2)
    @tullio grad=false result[i, j] := abs(w[k, i] - x[k, j])^p
    return result
end

function p_distance_pb(w::AAAF, x::AAAF; p::Int64=2)
    result = p_distance(w, x; p=p)
    return result, (Δresult) -> begin
        @assert size(result) == size(Δresult) "$size(result) != $size(Δresult)"
        @tullio grad=false Δw[k, i] := + Δresult[i, j] * p * abs(w[k, i] - x[k, j])^(p-1) * sign(w[k, i] - x[k, j])
        @tullio grad=false Δx[k, j] := - Δresult[i, j] * p * abs(w[k, i] - x[k, j])^(p-1) * sign(w[k, i] - x[k, j])
        return (ChainRules.NoTangent(), Δw, Δx)
    end
end

ChainRules.rrule(::typeof(p_distance), w, x; p=2) = p_distance_pb(w, x; p=p)

#######################################################################
# Weighted p-Distance
#######################################################################

function weighted_distance_zygote(w::AAAF, s::AAAF, x::AAAF; p::Int64=2)
    @tullio result[i, j] := abs(s[k, i] * (w[k, i] - x[k, j]))^p
    return result
end

function weighted_distance(w::AAAF, s::AAAF, x::AAAF; p::Int64=2)
    @tullio grad=false result[i, j] := abs(s[k, i] * (w[k, i] - x[k, j]))^p
    return result
end

function weighted_distance_pb(w::AAAF, s::AAAF, x::AAAF; p::Int64=2)
    result = weighted_distance(w, s, x; p=p)
    return result, (Δresult) -> begin
        @assert size(result) == size(Δresult) "$size(result) != $size(Δresult)"
        @tullio grad=false Δw[k, i] := + Δresult[i, j] * p * abs(s[k, i] * (w[k, i] - x[k, j]))^(p-1) * sign(s[k, i] * (w[k, i] - x[k, j])) * s[k, i]
        @tullio grad=false Δs[k, i] := + Δresult[i, j] * p * abs(s[k, i] * (w[k, i] - x[k, j]))^(p-1) * sign(s[k, i] * (w[k, i] - x[k, j])) * (w[k, i] - x[k, j])
        @tullio grad=false Δx[k, j] := - Δresult[i, j] * p * abs(s[k, i] * (w[k, i] - x[k, j]))^(p-1) * sign(s[k, i] * (w[k, i] - x[k, j])) * s[k, i]
        return (ChainRules.NoTangent(), Δw, Δs, Δx)
    end
end

ChainRules.rrule(::typeof(weighted_distance), w, s, x; p=2) = weighted_distance_pb(w, s, x; p=p)

#######################################################################
# mp distance
#######################################################################

function mp_distance(w::AAAF, x::AAAF)
    @tullio (max) result[i, j] := abs(w[k, i] - x[k, j])
    return result
end

#######################################################################
# weighted mp distance
#######################################################################

heaviside(x::AbstractFloat) = ifelse(x < 0, zero(x), ifelse(x > 0, one(x), oftype(x,0.5)))

function wmp_distance(w::AAAF, s::AAAF, x::AAAF)
    @tullio (max) result[i, j] := max(abs(w[k, i] - x[k, j]) - s[k, i], 0)
    return result
end

end
