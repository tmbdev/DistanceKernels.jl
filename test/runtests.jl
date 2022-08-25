##

using BenchmarkTools
using Profile, PProf
using Statistics
using CUDA
using CUDAKernels
using Test
using Zygote
using DistanceKernels

DK = DistanceKernels

print(CUDA.functional())
print(Threads.nthreads())


# include("DistanceKernels.jl"); DK = DistanceKernels

mode = cu;

xs = randn(Float32, 700, 900) |> mode;
ws = randn(Float32, 700, 900) |> mode;
ys = randn(Float32, 700, 1100) |> mode;
zs = randn(Float32, 900, 1100) |> mode;

println("euclidean_distance")

@btime CUDA.@sync DK.euclidean_distance(xs, ys);
@btime CUDA.@sync Zygote.pullback(DK.euclidean_distance, xs, ys)[2](zs);

println("euclidean_distance2")

@btime CUDA.@sync DK.euclidean_distance2(xs, ys);
@btime CUDA.@sync Zygote.pullback(DK.euclidean_distance2_zygote, xs, ys)[2](zs);
@btime CUDA.@sync Zygote.pullback(DK.euclidean_distance2, xs, ys)[2](zs);

xz, yz = Zygote.pullback(DK.euclidean_distance2_zygote, xs, ys)[2](zs);
xd, yd = Zygote.pullback(DK.euclidean_distance2, xs, ys)[2](zs);
@test xz ≈ xd
@test yz ≈ yd

println("p_distance")

@btime CUDA.@sync DK.p_distance(xs, ys);
@btime CUDA.@sync Zygote.pullback(DK.p_distance_zygote, xs, ys)[2](zs);
@btime CUDA.@sync Zygote.pullback(DK.p_distance, xs, ys)[2](zs);

xz, yz = Zygote.pullback(DK.p_distance_zygote, xs, ys)[2](zs);
xd, yd = Zygote.pullback(DK.p_distance, xs, ys)[2](zs);
@test xz ≈ xd
@test yz ≈ yd

println("q_distance")

@btime CUDA.@sync DK.q_distance(xs, ys);
@btime CUDA.@sync Zygote.pullback(DK.q_distance_zygote, xs, ys)[2](zs);
@btime CUDA.@sync Zygote.pullback(DK.q_distance, xs, ys)[2](zs);

xz, yz = Zygote.pullback((xs, ys) -> DK.q_distance_zygote(xs, ys; q=3.2f0), xs, ys)[2](zs);
xd, yd = Zygote.pullback((xs, ys) -> DK.q_distance(xs, ys; q=3.2f0), xs, ys)[2](zs);
@test xz ≈ xd
@test yz ≈ yd

println("weighted_distance")

@btime CUDA.@sync DK.weighted_distance(xs, ws, ys);
@btime CUDA.@sync Zygote.pullback(DK.weighted_distance_zygote, xs, ws, ys)[2](zs);
@btime CUDA.@sync Zygote.pullback(DK.weighted_distance, xs, ws, ys)[2](zs);

xz, wz, yz = Zygote.pullback(DK.weighted_distance_zygote, xs, ws, ys)[2](zs);
xd, wd, yd = Zygote.pullback(DK.weighted_distance, xs, ws, ys)[2](zs);
@test xz ≈ xd
@test wz ≈ wd
@test yz ≈ yd

println("mp distance")

@btime CUDA.@sync DK.mp_distance(xs, ys);
@btime CUDA.@sync Zygote.pullback(DK.mp_distance, xs, ys)[2](zs);

println("weighted mp distance")

@btime CUDA.@sync DK.wmp_distance(xs, ws, ys);
@btime CUDA.@sync Zygote.pullback(DK.wmp_distance, xs, ws, ys)[2](zs);

##

