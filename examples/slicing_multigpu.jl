using Distributed
using OMEinsum.OMEinsumContractionOrders, OMEinsum, CUDA

@info "find $(length(devices())) GPU devices"
const procs = addprocs(length(devices())-nprocs()+1)
const gpus = collect(devices())
const process_device_map = Dict(zip(procs, gpus))
@info "mapping processes to GPUs: $process_device_map"

@everywhere begin  # these packages/functions should be accessible on all processes
    using OMEinsum.OMEinsumContractionOrders, OMEinsum, CUDA
    CUDA.allowscalar(false)

    function do_work(f, jobs, results) # define work function everywhere
        while true
            job = take!(jobs)
            @info "running $job on device $(Distributed.myid())"
            res = f(job)
            put!(results, res)
        end
    end
end

"""
    multiprocess_run(func, inputs::AbstractVector)

Run `func` in parallel for a vector f `inputs`.
Returns a vector of results.
"""
function multiprocess_run(func, inputs::AbstractVector{T}) where T
    n = length(inputs)
    jobs = RemoteChannel(()->Channel{T}(n));
    results = RemoteChannel(()->Channel{Any}(n));
    for i in 1:n
        put!(jobs, inputs[i])
    end
    for p in workers() # start tasks on the workers to process requests in parallel
        remote_do(do_work, p, func, jobs, results)
    end
    return Any[take!(results) for i=1:n]
end

"""
    multigpu_einsum(code::SlicedEinsum, xs::AbstractArray...; size_info = nothing, process_device_map::Dict)

Multi-GPU contraction of a sliced einsum specified by `code`.
Each time, the program take the slice and upload them to a specific GPU device and do the contraction.
Other arguments are

* `xs` are input tensors allocated in **main memory**,
* `size_info` specifies extra size information,
* `process_device_map` is a map between processes and GPU devices.
"""
function multigpu_einsum(se::SlicedEinsum{LT,ET}, @nospecialize(xs::AbstractArray...); size_info = nothing, process_device_map::Dict) where {LT, ET}
    length(se.slicing) == 0 && return se.eins(xs...; size_info=size_info)
    size_dict = size_info===nothing ? Dict{OMEinsum.labeltype(se),Int}() : copy(size_info)
    OMEinsum.get_size_dict!(se, xs, size_dict)

    it = OMEinsum.SliceIterator(se, size_dict)
    res = OMEinsum.get_output_array(xs, getindex.(Ref(size_dict), it.iyv), true)
    eins_sliced = OMEinsum.drop_slicedim(se.eins, se.slicing)
    inputs = collect(enumerate([copy(x) for x in it]))
    @info "start multiple process contraction!"
    results = multiprocess_run(inputs) do (k, slicemap)
        @info "computing slice $k/$(length(it))"
        device!(process_device_map[Distributed.myid()])
        xsi = ntuple(i->CuArray(OMEinsum.take_slice(xs[i], it.ixsv[i], slicemap)), length(xs))
        Array(einsum(eins_sliced, xsi, it.size_dict_sliced))
    end
    # accumulate results to `res`
    for (resi, (k, slicemap)) in zip(results, inputs)
        OMEinsum.fill_slice!(res, it.iyv, resi, slicemap)
    end
    return res
end

# A using case
# ---------------------------------------
using Yao, Yao.YaoToEinsum, Yao.EasyBuild

# I. create a quantum circuit
nbit = 20
c = Yao.dispatch!(variational_circuit(nbit, 10), :random)

# II. convert a tensor network
# 1. specify input and output states as product states,
prod_state = Dict(zip(1:nbit, zeros(Int, nbit)))
# 2. convert the circuit to einsum code,
code, xs = YaoToEinsum.yao2einsum(c; initial_state=prod_state, final_state=prod_state, optimizer=nothing)
# 3. optimize the contraction order
size_dict = OMEinsum.get_size_dict(getixsv(code), xs)
slicedcode = optimize_code(code, size_dict, TreeSA(); simplifier=MergeGreedy(), slicer=TreeSASlicer(score=ScoreFunction(sc_target=15)))

# III. do the contraction on multiple GPUs in parallel
@info "time/space complexity is $(timespace_complexity(slicedcode, size_dict)), number of slices: $(length(slicedcode.slicing))"
multigpu_einsum(slicedcode, xs...; process_device_map=process_device_map)
