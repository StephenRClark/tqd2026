# --- TQD2026 summer school ---
#
# MPS_MPO_helpers.jl
#
# Helper routines for the TQD2026 summer school short-course on 
# "Tensor networks methods for quantum (thermo)dynamics". The
# functions are divided into 4 sections:
#
# (1) Trotter circuit to MPO
# Routines for constructing local Trotter circuits from ITensor OpSum
# Hamiltonians and compiling them into MPO time-step operators. The code supports 
# OpSum terms acting on contiguous blocks of up to a chosen range (typically 4), 
# including nearest-neighbour and short-range extended interactions. For fermionic 
# site types, local terms are dressed with Jordan-Wigner parity strings following 
# ITensor's OpSum-to-MPO convention. These routines are used in all 4 notebooks.
#
# (2) Thermal purification
# A small number of helper functions for building maximally entangled ancilla states,
# constructing MPO series expansions and computing expectation values.
#
# (3) Lindbladians 
# Functions for constructing Lindbladian superoperators from OpSum definitions
# of Hamiltonians and jump operators forming MPOs with "fat" operator indices.
#
# (4) Chain-mapping
# A wide range of helper functions for constructing a chain-mapping of a continuous
# fermionic thermal bath defined by a spectral function. Numerical and analytical 
# solutions to key test cases are also implemented.
#
# Author: Stephen R. Clark
# Contributions: David Strachan
#
# Copyright (c) 2026 Stephen R. Clark, University of Bristol
# License: MIT

module MPS_MPO_helpers

using ITensors
using ITensorMPS
using LinearAlgebra
using PolyChaos
using DSP

export GateLayer,
  second_order_trotter_circuit,
  flatten_gates,
  circuit_mpo,
  apply_mpo_step,
  half_chain_entropy,
  dense_hamiltonian_matrix,
  dense_state_vector,
  dense_half_chain_entropy,
  expect_MPO,
  exp_series,
  maximally_entangled_pairs,
  lindbladian_opsum,
  product_operator_mps,
  trace_mps,
  trace_value,
  expect_liouville,
  expect_liouville_complex,
  normalize_trace!,
  bath_parameters,
  system_parameters,
  TDVP_parameters,
  build_chain_hamiltonian,
  thermofield_vacuum,
  propagate_correlations,
  LB_current

"""
    --------------------------------------------
    Struct definitions used for chain-mappings
    --------------------------------------------
"""

const I2 = ComplexF64[1 0; 0 1]
const Sp = ComplexF64[0 1; 0 0]
const Sm = ComplexF64[0 0; 1 0]
const Sz = ComplexF64[0.5 0; 0 -0.5]
const Sx = 0.5 * ComplexF64[0 1; 1 0]
const Sy = 0.5 * ComplexF64[0 -im; im 0]

Base.@kwdef struct bath_parameters
    Γ::Float64 #Coupling to system
    β::Float64 #inverse temperature
    μ::Float64 #chemical potential
    D::Float64 #bandwidth
    N::Int     #Number of chain modes
end

Base.@kwdef struct system_parameters
    ϵ::Vector{Float64} #Vector of onsite energies
    t::Vector{Float64} #Vector of couplings
    U::Vector{Float64} #Vector of interactions
    occupations::Vector{String} # Vector of initial occupations
end

Base.@kwdef struct TDVP_parameters
    tdvp_cutoff::Float64 #Numerical cutoff for tdvp
    minbonddim::Int      #Minimum bond dimension for tdvp
    maxbonddim::Int      #Maximum bond dimension for tdvp
    δt::Float64          #Time step
    total_simulation_time::Float64 #Evolution time
end

abstract type ThermofieldSector end
struct Filled <: ThermofieldSector end
struct Empty <: ThermofieldSector end

struct ChainLayout
    left_filled
    left_empty

    system

    right_filled
    right_empty
end

"""
    --------------------------------------------
    Trotter circuit to MPO helper functions:
    --------------------------------------------
"""

"""
    LocalBlock(sites, h)

Container for one local Hamiltonian piece. `sites` is a contiguous interval
such as `3:5`, and `h` is the ITensor representation of the Hamiltonian terms
assigned to that interval.
"""
struct LocalBlock
  sites::UnitRange{Int}
  h::ITensor
end

"""
    GateLayer(name, time_fraction, blocks, gates)

A layer of mutually non-overlapping local gates. Since the gates in one layer
act on disjoint site intervals, they can be applied in any order and compiled
into a single MPO layer without extra Trotter error.
"""
struct GateLayer
  name::Symbol
  time_fraction::Float64
  blocks::Vector{UnitRange{Int}}
  gates::Vector{ITensor}
end

"""
   opsum_terms(os), term_ops(term)

These thin wrappers isolate the few places where we inspect ITensor's OpSum
representation. Keeping them here makes it easier to change the parser later.
""" 
opsum_terms(os) = ITensors.terms(os)
term_ops(term) = ITensors.terms(term)

"""
    op_site_number(o)
Return the site number that a one-site operator `o` acts on. This is used to sort 
the operators in an OpSum term into site order, and to detect any skipped sites 
that require identity operators to be inserted.
"""
function op_site_number(o)
  op_sites = ITensors.sites(o)
  length(op_sites) == 1 ||
    error("The operator $o acts on more than one site; use ordinary one-site OpSum factors.")
  return only(op_sites)
end

"""
    term_sites(term), term_interval(term), interval_length(interval)

Return the set of site numbers that an OpSum term acts on, the contiguous interval 
that covers those sites, and the length of a contiguous interval.
"""
term_sites(term) = sort(unique(op_site_number.(term_ops(term))))
term_interval(term) = first(term_sites(term)):last(term_sites(term))
interval_length(interval::UnitRange{Int}) = last(interval) - first(interval) + 1

"""
    intervals_overlap(a, b) 

Check whether two contiguous site intervals overlap. This is used to group local 
Hamiltonian blocks into non-overlapping layers.
"""
function intervals_overlap(a::UnitRange{Int}, b::UnitRange{Int})
  return first(a) <= last(b) && first(b) <= last(a)
end

"""
    check_supported_opsum(os, sites; max_range=4)

Validate the class of Hamiltonians handled by this tutorial helper. Terms may
act on any subset of a contiguous interval of at most `max_range` sites. This
includes onsite, nearest-neighbour, next-nearest-neighbour, next-next-nearest
neighbour, and genuine 3- or 4-site product terms. Longer-range terms are
rejected instead of silently building an invalid Trotter circuit.
"""
function check_supported_opsum(os, sites; max_range::Int=4)
  N = length(sites)
  N >= 1 || error("This helper expects at least one site.")
  max_range >= 1 || error("max_range must be at least 1.")

  for term in opsum_terms(os)
    ops = term_ops(term)
    isempty(ops) && error("Constant identity terms are not supported by this helper.")

    op_sites = op_site_number.(ops)
    unique_sites = sort(unique(op_sites))
    all(1 .<= unique_sites .<= N) || error("The term $term refers to a site outside 1:$N.")

    length(unique_sites) == length(op_sites) ||
      error("The term $term has more than one operator on the same site.")

    interval = first(unique_sites):last(unique_sites)
    interval_length(interval) <= max_range ||
      error("The term $term spans $(interval_length(interval)) sites, larger than max_range=$max_range.")
  end
  return nothing
end

"""
    local_operator(o, sites)  

Return the ITensor representation of a one-site operator `o` from an OpSum.
"""
function local_operator(o, sites)
  i = op_site_number(o)
  return op(ITensors.which_op(o), sites[i]; ITensors.params(o)...)
end

"""
    op_has_fermion_string(o, sites)

Check whether a one-site operator `o` is parity-odd and therefore requires a 
Jordan-Wigner string of `"F"` operators to the left. This is only relevant for 
fermionic site types, and is ignored if ITensor's auto-fermion mode is enabled.
"""
function op_has_fermion_string(o, sites)
  i = op_site_number(o)
  return ITensors.has_fermion_string(ITensors.which_op(o), sites[i]; ITensors.params(o)...)
end

"""
    local_operator_with_fermion_string(o, sites)

Return the ITensor representation of a one-site operator `o` from an OpSum, 
together with a Jordan-Wigner string of `"F"` operators to the left. This is 
only relevant for fermionic site types, and is ignored if ITensor's auto-fermion 
mode is enabled.
"""
function local_operator_with_fermion_string(o, sites)
  i = op_site_number(o)
  opname = ITensors.which_op(o)
  opname isa AbstractArray &&
    error("Matrix-valued OpSum terms inside a fermionic Jordan-Wigner string are not supported.")
  return op("$(opname) * F", sites[i]; ITensors.params(o)...)
end

"""
    sorted_term_ops_and_fermion_sign(term, sites)

Sort the operators in an OpSum term into site order, and return the corresponding 
fermionic sign. This is only relevant for fermionic site types, and is ignored if 
ITensor's auto-fermion mode is enabled.
"""
function sorted_term_ops_and_fermion_sign(term, sites)
  ops = collect(term_ops(term))
  perm = Vector{Int}(undef, length(ops))
  sortperm!(perm, ops; alg=InsertionSort, lt=(o1, o2) -> op_site_number(o1) < op_site_number(o2))
  sorted_ops = ops[perm]

  fermion_perm = Int[]
  for (n, o) in enumerate(sorted_ops)
    op_has_fermion_string(o, sites) && push!(fermion_perm, perm[n])
  end

  isodd(length(fermion_perm)) &&
    error("Parity-odd fermionic terms are not supported by this OpSum Trotter helper.")

  sign = isempty(fermion_perm) ? 1 : ITensors.parity_sign(fermion_perm)
  return sorted_ops, sign
end

"""
    fermion_string_sites(sorted_ops, sites)

Return the set of site numbers that require a Jordan-Wigner string of `"F"` operators 
to the left. This is only relevant for fermionic site types, and is ignored if ITensor's 
auto-fermion mode is enabled.
"""
function fermion_string_sites(sorted_ops, sites)
  string_sites = Set{Int}()
  ITensors.using_auto_fermion() && return string_sites

  prevsite = typemax(Int)
  odd_parity_to_right = false
  for o in reverse(sorted_ops)
    i = op_site_number(o)
    if odd_parity_to_right && i < prevsite
      union!(string_sites, i:(prevsite - 1))
    end
    prevsite = i
    op_has_fermion_string(o, sites) && (odd_parity_to_right = !odd_parity_to_right)
  end
  return string_sites
end

"""
    identity_on_interval(interval, sites)

Return the ITensor representation of the identity operator on a contiguous interval of sites. 
This is used to fill in skipped sites when materializing an OpSum term.
"""
function identity_on_interval(interval::UnitRange{Int}, sites)
  T = op("Id", sites[first(interval)])
  for i in (first(interval) + 1):last(interval)
    T *= op("Id", sites[i])
  end
  return T
end

"""
    term_itensor(term, sites, interval)

Materialize one OpSum term as an ITensor on the full contiguous `interval`.
If the OpSum term skips sites inside that interval, identities are inserted on
the skipped sites. For example, a term on sites `(j, j+2)` becomes a 3-site
operator on `j:j+2`.

For fermionic site types, this follows the same convention as ITensor's
`MPO(::OpSum, sites)` constructor: parity-odd local operators are detected with
`has_fermion_string`, the term is sorted into site order with the corresponding
fermionic sign, and explicit Jordan-Wigner `"F"` factors are inserted when
ITensor's auto-fermion mode is disabled.
"""
function term_itensor(term, sites, interval::UnitRange{Int})
  sorted_ops, fermion_sign = sorted_term_ops_and_fermion_sign(term, sites)
  ops_by_site = Dict(op_site_number(o) => o for o in sorted_ops)
  string_sites = fermion_string_sites(sorted_ops, sites)

  T = ITensors.coefficient(term) * fermion_sign
  for i in interval
    if haskey(ops_by_site, i)
      local_op = if i in string_sites
        local_operator_with_fermion_string(ops_by_site[i], sites)
      else
        local_operator(ops_by_site[i], sites)
      end
      T *= local_op
    else
      T *= i in string_sites ? op("F", sites[i]) : op("Id", sites[i])
    end
  end
  return T
end

"""
    local_hamiltonian_blocks(os, sites; max_range=4)

Group an `OpSum` into local Hamiltonian blocks. Terms with the same contiguous
support interval are summed together. A next-nearest-neighbour term on `j` and
`j+2`, for instance, is assigned to the interval `j:j+2`.
"""
function local_hamiltonian_blocks(os, sites; max_range::Int=4)
  check_supported_opsum(os, sites; max_range)
  terms_by_interval = Dict{Tuple{Int,Int},Vector{Any}}()

  for term in opsum_terms(os)
    interval = term_interval(term)
    key = (first(interval), last(interval))
    push!(get!(terms_by_interval, key, Any[]), term)
  end

  blocks = LocalBlock[]
  for key in sort(collect(keys(terms_by_interval)))
    interval = key[1]:key[2]
    h = 0.0 * identity_on_interval(interval, sites)
    for term in terms_by_interval[key]
      h += term_itensor(term, sites, interval)
    end
    push!(blocks, LocalBlock(interval, h))
  end
  return blocks
end

# Backwards-compatible name from the nearest-neighbour-only draft. It now
# returns general local interval blocks rather than only two-site bonds.
bond_hamiltonians(os, sites; max_range::Int=4) = local_hamiltonian_blocks(os, sites; max_range)

"""
    color_local_blocks(blocks)

Greedily group local Hamiltonian blocks into colours/layers. Blocks in the same
colour do not overlap on any physical site, so their exponentials commute and
can be applied as one layer of local gates.
"""
function color_local_blocks(blocks::Vector{LocalBlock})
  groups = Vector{Vector{Int}}()
  for block_index in eachindex(blocks)
    interval = blocks[block_index].sites
    placed = false
    for group in groups
      if all(!intervals_overlap(interval, blocks[i].sites) for i in group)
        push!(group, block_index)
        placed = true
        break
      end
    end
    placed || push!(groups, [block_index])
  end
  return groups
end

"""
    second_order_schedule(nlayers)

Return the layer indices and time fractions for a symmetric second-order
Strang splitting.
"""
function second_order_schedule(nlayers::Int)
  nlayers >= 1 || error("Need at least one Trotter layer.")
  nlayers == 1 && return [(1, 1.0)]

  schedule = [(i, 0.5) for i in 1:(nlayers - 1)]
  push!(schedule, (nlayers, 1.0))
  append!(schedule, [(i, 0.5) for i in (nlayers - 1):-1:1])
  return schedule
end

"""
    second_order_trotter_circuit(os, sites, tau; max_range=4)

Build a second-order Strang Trotter circuit from an `OpSum`. Local Hamiltonian
blocks are greedily coloured into non-overlapping layers. If there are layers
`H1, H2, ..., Hm`, the returned circuit applies
`exp(tau H1/2) exp(tau H2/2) ... exp(tau Hm) ... exp(tau H2/2) exp(tau H1/2)`.
"""
function second_order_trotter_circuit(os, sites, tau; evolution_factor=-im, max_range::Int=4)
  blocks = local_hamiltonian_blocks(os, sites; max_range)
  color_groups = color_local_blocks(blocks)
  schedule = second_order_schedule(length(color_groups))

  layers = GateLayer[]
  for (step, (group_index, frac)) in enumerate(schedule)
    block_indices = color_groups[group_index]
    layer_blocks = [blocks[i].sites for i in block_indices]
    gates = [exp(evolution_factor * (frac * tau) * blocks[i].h) for i in block_indices]
    push!(layers, GateLayer(Symbol("layer_$step"), frac, layer_blocks, gates))
  end
  return layers
end

"""
    flatten_gates(layers)

Return the local ITensor gates in the order they appear in the circuit. This
is useful for checking the compiled MPO against direct gate application.
"""
function flatten_gates(layers)
  gates = ITensor[]
  for layer in layers
    append!(gates, layer.gates)
  end
  return gates
end

"""
    gate_segment_tensors(gate, interval, sites; cutoff=1e-14)

Split a k-site gate into k MPO tensors by sweeping left to right with SVDs.
For k=1 the gate is already a single MPO tensor.
"""
function gate_segment_tensors(gate::ITensor, interval::UnitRange{Int}, sites; cutoff=1e-14)
  block_sites = collect(interval)
  length(block_sites) == 1 && return [gate]

  T = gate
  tensors = ITensor[]
  for site_number in block_sites[1:(end - 1)]
    left_inds = Index[]
    if !isempty(tensors)
      append!(left_inds, commoninds(last(tensors), T))
    end
    push!(left_inds, sites[site_number]')
    push!(left_inds, sites[site_number])

    U, S, V = svd(T, left_inds; cutoff)
    push!(tensors, U * S)
    T = V
  end
  push!(tensors, T)
  return tensors
end

"""
    layer_mpo(layer, sites; cutoff=1e-14)

Compile one non-overlapping gate layer into an MPO. Gates may act on 1, 2, 3,
or 4 contiguous sites. Multi-site gates are split into an MPO segment by
successive SVDs from left to right.
"""
function layer_mpo(layer::GateLayer, sites; cutoff=1e-14)
  N = length(sites)
  W = MPO(N)
  for n in 1:N
    W[n] = op("Id", sites[n])
  end

  occupied = Set{Int}()
  for (interval, gate) in zip(layer.blocks, layer.gates)
    all(i -> !(i in occupied), interval) ||
      error("Layer $(layer.name) contains overlapping gates; cannot compile it as one MPO layer.")
    union!(occupied, interval)

    segment = gate_segment_tensors(gate, interval, sites; cutoff)
    for (offset, site_number) in enumerate(interval)
      W[site_number] = segment[offset]
    end
  end
  return W
end

"""
    mpo_product(A, B; cutoff=1e-13, maxdim=10000)

Compose two MPOs while avoiding any dependence on a zip-up MPO product. This is
only used when compiling the local gate layers into one one-step MPO.
"""
function mpo_product(A::MPO, B::MPO; cutoff=1e-13, maxdim=10_000)
  return apply(A, B; alg="naive", cutoff, maxdim)
end

"""
    circuit_mpo(layers, sites; cutoff=1e-13, maxdim=10000)

Compile all Trotter layers into a single one-step MPO. The individual layer
MPOs are multiplied in the same order that their gates act on an MPS.
"""
function circuit_mpo(layers, sites; cutoff=1e-13, maxdim=10_000)
  layer_mpos = [layer_mpo(layer, sites; cutoff) for layer in layers]
  U = first(layer_mpos)
  for W in layer_mpos[2:end]
    U = mpo_product(W, U; cutoff, maxdim)
  end
  return U
end

"""
    apply_mpo_step(Utau, psi; cutoff=1e-10, maxdim=200)

Apply the one-step MPO to an MPS using ITensor's density-matrix MPO x MPS
contraction algorithm.
"""
function apply_mpo_step(Utau::MPO, psi::MPS; cutoff=1e-10, maxdim=200, normalize=true)
  return apply(Utau, psi; alg="densitymatrix", cutoff, maxdim, normalize)
end

"""
    half_chain_entropy(psi; bond=div(length(psi), 2))

Compute the von Neumann entropy across a chosen MPS bond. The default bond is
the middle cut of the chain.
"""
function half_chain_entropy(psi::MPS; bond::Int=div(length(psi), 2))
  1 <= bond < length(psi) || error("The entropy bond must be between 1 and length(psi)-1.")
  psi_centered = orthogonalize(psi, bond)
  _, S, _ = svd(psi_centered[bond], (linkinds(psi_centered, bond - 1)..., siteinds(psi_centered, bond)...))

  entropy = 0.0
  for n in 1:ITensors.dim(S, 1)
    p = abs2(S[n, n])
    p > 0 && (entropy -= p * log(p))
  end
  return entropy
end


"""
    --------------------------------------------
    Full state/operator helper functions:
    --------------------------------------------  
"""

function dense_operator_matrix(T::ITensor, sites)
  row_inds = prime.(sites)
  col_inds = sites
  A = array(T, row_inds..., col_inds...)
  return reshape(A, prod(ITensors.dim.(row_inds)), prod(ITensors.dim.(col_inds)))
end

function dense_mpo_matrix(W::MPO, sites)
  old_warn_order = ITensors.disable_warn_order()
  try
    T = W[1]
    for n in 2:length(W)
      T *= W[n]
    end
    return dense_operator_matrix(T, sites)
  finally
    ITensors.set_warn_order(old_warn_order)
  end
end

"""
    dense_hamiltonian_matrix(os, sites)

Construct the full dense Hamiltonian matrix represented by an `OpSum` on
`sites`. This is intended only for exact-diagonalization checks of small
systems. Internally it first builds `MPO(os, sites)`, so it follows ITensor's
standard OpSum conventions, including Jordan-Wigner strings for fermionic
operators.
"""
function dense_hamiltonian_matrix(os::OpSum, sites)
  return dense_mpo_matrix(MPO(os, sites), sites)
end

"""
    dense_state_vector(psi, sites)

Contract an MPS into a dense state vector ordered according to `sites`. This is
only practical for small systems, but is useful for comparing MPS results with
exact diagonalization.
"""
function dense_state_vector(psi::MPS, sites)
  old_warn_order = ITensors.disable_warn_order()
  try
    T = psi[1]
    for n in 2:length(psi)
      T *= psi[n]
    end
    return vec(array(T, sites...))
  finally
    ITensors.set_warn_order(old_warn_order)
  end
end

"""
    dense_half_chain_entropy(state, sites; bond=div(length(sites), 2))

Compute the bipartite von Neumann entropy of a dense state vector across
`bond`, where sites `1:bond` define the left subsystem.
"""
function dense_half_chain_entropy(state::AbstractVector, sites; bond::Integer=div(length(sites), 2))
  1 <= bond < length(sites) || error("The entropy bond must be between 1 and length(sites)-1.")

  dims = ITensors.dim.(sites)
  left_dim = prod(dims[1:bond])
  right_dim = prod(dims[(bond + 1):end])
  _, singular_values, _ = svd(reshape(state, left_dim, right_dim))

  entropy = 0.0
  for s in singular_values
    p = abs2(s)
    p > 0 && (entropy -= p * log(p))
  end
  return entropy
end

"""
    --------------------------------------------
    Thermal purification helper functions:
    --------------------------------------------  
"""

"""
    expect_MPO(psi::MPS,H::MPO) 

Compute the expectation value of an MPO for an MPS.
"""
function expect_MPO(psi::MPS,H::MPO)

  E = real(inner(psi, Apply(H, psi)) / inner(psi, psi))
  return E
end

"""
    exp_series(H, β; order=70)

Compute the low-order series expansion of a MPO H exponential exp(-β H).
"""
function exp_series(H, β, sites; order=3)

  ρT = MPO(sites, "Id")
  term = deepcopy(ρT)
  for n in 1:order
    term = (-β / n) * apply(H,term)
    ρT += term
  end
  return ρT
end

"""
     bell_pair_gate(s1, s2)

Return an ITensor representing a Bell pair gate on two sites. This is used to 
prepare a maximally entangled initial state for testing the Trotter circuit.
"""
function bell_pair_gate(s1, s2)
  G = ITensor(s1', s2', s1, s2)
  G[s1' => 1, s2' => 1, s1 => 1, s2 => 1] = 1 / sqrt(2)
  G[s1' => 2, s2' => 2, s1 => 1, s2 => 1] = 1 / sqrt(2)
  return G
end

"""
     maximally_entangled_pairs(sites)

Prepare an MPS where every pair of adjacent sites is in a Bell pair. This is 
a highly entangled state used for purified thermal state preparation.
""" 
function maximally_entangled_pairs(sites)
  psi = MPS(ComplexF64, sites, "Up")
  gates = [bell_pair_gate(sites[j], sites[j + 1]) for j in 1:2:length(sites)]
  psi = apply(gates, psi; cutoff=1e-14)
  normalize!(psi)
  return psi
end

"""
    --------------------------------------------
    Lindbladian helper functions:
    --------------------------------------------  
"""

"""
    physical_op(name)

Return the matrix representation of a physical operator given its name.
"""
function physical_op(name)
  table = Dict(
    "Id" => I2,
    "I" => I2,
    "S+" => Sp,
    "S-" => Sm,
    "Sz" => Sz,
    "Sx" => Sx,
    "Sy" => Sy,
  )
  return table[String(name)]
end

"""
    left_super(A), right_super(A), sandwich_super(A, B)

Return the superoperator representation of a physical operator A, or a sandwich
superoperator of A and B. These are used to build the Lindbladian in Liouville space.
"""
left_super(A) = kron(I2, A)
right_super(A) = kron(transpose(A), I2)
sandwich_super(A, B) = kron(transpose(B), A)

"""
    add_matrix_term!(os, coeff, mats_by_site)

Add a term to an OpSum given its coefficient and a dictionary of site numbers to matrices. 
This is used to build the Lindbladian in Liouville space.
"""
function add_matrix_term!(os, coeff, mats_by_site)
  abs(coeff) < 1e-14 && return os
  args = Any[coeff]
  for site in sort(collect(keys(mats_by_site)))
    push!(args, mats_by_site[site])
    push!(args, site)
  end
  add!(os, Tuple(args))
  return os
end

"""
    term_matrix_data(term)

Return the coefficient and a dictionary of site numbers to matrices for a given OpSum term.
This is used to build the Lindbladian in Liouville space.
"""
function term_matrix_data(term)
  coeff = ITensors.coefficient(term)
  mats = Dict{Int,Matrix{ComplexF64}}()
  for o in ITensors.terms(term)
    site = only(ITensors.sites(o))
    A = physical_op(ITensors.which_op(o))
    mats[site] = haskey(mats, site) ? mats[site] * A : A
  end
  return coeff, mats
end

"""
    multiply_local_data(coeff1, mats1, coeff2, mats2)

Multiply two local data representations of OpSum terms. This is used to build the 
Lindbladian in Liouville space.
"""
function multiply_local_data(coeff1, mats1, coeff2, mats2)
  mats = Dict(site => Matrix(A) for (site, A) in mats1)
  for (site, A) in mats2
    mats[site] = haskey(mats, site) ? mats[site] * A : A
  end
  return coeff1 * coeff2, mats
end

"""
    adjoint_local_data(coeff, mats)
    add_left_term!(L, coeff, mats)
    add_right_term!(L, coeff, mats)
    add_sandwich_term!(L, coeff, left_mats, right_mats)

These functions are used to build the Lindbladian in Liouville space. They handle
the adjoint of local data, and add left, right, and sandwich terms to the Lindbladian OpSum.
"""
function adjoint_local_data(coeff, mats)
  return conj(coeff), Dict(site => Matrix(A') for (site, A) in mats)
end

function add_left_term!(L, coeff, mats)
  return add_matrix_term!(L, coeff, Dict(site => left_super(A) for (site, A) in mats))
end

function add_right_term!(L, coeff, mats)
  return add_matrix_term!(L, coeff, Dict(site => right_super(A) for (site, A) in mats))
end

function add_sandwich_term!(L, coeff, left_mats, right_mats)
  support = sort(collect(union(keys(left_mats), keys(right_mats))))
  mats = Dict{Int,Matrix{ComplexF64}}()
  for site in support
    A = get(left_mats, site, I2)
    B = get(right_mats, site, I2)
    mats[site] = sandwich_super(A, B)
  end
  return add_matrix_term!(L, coeff, mats)
end

"""
    lindbladian_opsum(H, jumps)

Construct the Lindbladian superoperator in Liouville space from a Hamiltonian
`H` and a list of jump operators `jumps`. The Lindbladian is represented as an
`OpSum` that includes the Hamiltonian contribution and the dissipative 
contributions from the jump operators. The Hamiltonian contribution is added as 
left and right superoperators, while the jump operators contribute both sandwich 
terms and left/right terms to the Lindbladian.
"""
function lindbladian_opsum(H::OpSum, jumps::Vector{<:OpSum})
  L = OpSum()

  for hterm in ITensors.terms(H)
    coeff, mats = term_matrix_data(hterm)
    add_left_term!(L, -im * coeff, mats)
    add_right_term!(L, im * coeff, mats)
  end

  for J in jumps
    jterms = [term_matrix_data(term) for term in ITensors.terms(J)]
    for (ca, A) in jterms, (cb, B) in jterms
      cd, Bd = adjoint_local_data(cb, B)
      add_sandwich_term!(L, ca * cd, A, Bd)

      ck, K = multiply_local_data(cd, Bd, ca, A)
      add_left_term!(L, -0.5 * ck, K)
      add_right_term!(L, -0.5 * ck, K)
    end
  end
  return L
end

"""
    product_operator_mps(sites, local_mats)

Construct an MPS that represents a product operator from a list of local matrices.
Each local matrix is converted into an ITensor and assigned to the corresponding site.
This function is useful for creating MPS representations of operators that act 
independently on each site.
"""
function product_operator_mps(sites, local_mats)
  return MPS([itensor(vec(Matrix{ComplexF64}(A)), sites[j]) for (j, A) in enumerate(local_mats)])
end

"""
    trace_mps(sites), trace_value(rho, trmps)
These functions compute the trace of a density matrix MPS `rho` by contracting it 
with a trace MPS `trmps` that represents the identity operator on all sites. The 
trace value is calculated as the inner product of `trmps` and `rho`.
"""
trace_mps(sites) = product_operator_mps(sites, [I2 for _ in sites])
trace_value(rho, trmps) = inner(trmps, rho)

"""
    observable_mps(sites, ops)
    expect_liouville_complex(rho, trmps, sites, ops)
    expect_liouville(rho, trmps, sites, ops)

Together these functions compute the expectation value of an observable in Liouville space. 
The observable is specified by a list of tuples `(site_index, operator_matrix)`, where `site_index` 
is the  index of the site and `operator_matrix` is the matrix representation of the operator acting 
on that site. The function returns the real part of the expectation value, which is calculated as 
the inner product of the observable MPS and the density matrix MPS `rho`, normalized by the trace 
of `rho` with respect to the trace MPS `trmps`.
"""
function observable_mps(sites, ops)
  local_ops = [I2 for _ in sites]
  for (j, A) in ops
    local_ops[j] = transpose(A)
  end
  return MPS([itensor(conj(vec(local_ops[j])), sites[j]) for j in eachindex(sites)])
end

function expect_liouville_complex(rho, trmps, sites, ops)
  return inner(observable_mps(sites, ops), rho) / trace_value(rho, trmps)
end

function expect_liouville(rho, trmps, sites, ops)
  return real(expect_liouville_complex(rho, trmps, sites, ops))
end

"""
    normalize_trace!(rho, trmps)

Normalize the trace of a density matrix MPS `rho` by dividing it by its trace value
calculated with respect to the trace MPS `trmps` and returning the result.
"""
function normalize_trace!(rho, trmps)
  rho[1] *= inv(trace_value(rho, trmps))
  return rho
end

"""
    --------------------------------------------
    Chain-mapping helper functions:
    --------------------------------------------  
"""

"""
    fermi_factor(ω,β,μ)

Calculates the Fermi-Dirac distribution function for a given frequency `ω`, inverse temperature `β`, 
and chemical potential `μ`. This function is used to determine the occupation probability of fermionic 
modes in the bath.
"""
fermi_factor(ω,β,μ) = 1 / (exp(β*(ω-μ)) + 1)

"""
    heaviside(t)

Calculates the Heaviside step function for a given input `t`. The Heaviside function is defined as 0 
for negative inputs and 1 for non-negative inputs. 
"""
heaviside(t) = 0.5 * (sign.(t) .+ 1)

"""
    semicircular_density(Γ,ω,D)

Calculates the semicircular spectral density for a given coupling strength `Γ`, frequency `ω`, 
and bandwidth `D`. The semicircular spectral density is defined as a semicircle function within 
the frequency range `[-D, D]` and zero outside this range.
""" 
function semicircular_density(Γ,ω,D)
    J = real((2*Γ/(π^2))*sqrt.(Complex.(1 .-(ω/D).^2)))
    return J
end

"""
    box_spectral_density(Γ,ω,D)

Calculates the box spectral density for a given coupling strength `Γ`, frequency `ω`, 
and bandwidth `D`. The box spectral density is defined as a constant value within the 
frequency range `[-D, D]` and zero outside this range.
"""
function box_spectral_density(Γ,ω,D)
    #Box spectral density
    return J = (Γ/(2*π))*(heaviside(ω .+ D) .- heaviside(ω .- D))
end

"""
    thermofield_spectral_density(ω,bath::bath_parameters,::Filled)

Calculates the spectral density for a filled thermofield chain. The function takes
the frequency `ω`, bath parameters, and a `Filled` thermofield sector as inputs. 
It computes the spectral density using a box spectral density function and multiplies 
it by the Fermi factor, which accounts for the occupation probability of the bath modes. 
The resulting spectral density is used in the chain mapping process to determine the chain 
coefficients for the filled thermofield chain representation of the bath.
"""
function thermofield_spectral_density(ω,bath::bath_parameters,::Filled)
    #Spectral density for a filled chain
    J = box_spectral_density(bath.Γ,ω,bath.D)
   # J = semicircular_density(bath.Γ,ω,bath.D)
    return J * fermi_factor(ω,bath.β,bath.μ)
end

"""
    thermofield_spectral_density(ω,bath::bath_parameters,::Empty)

Calculates the spectral density for an empty thermofield chain. The function takes
the frequency `ω`, bath parameters, and an `Empty` thermofield sector as inputs. 
It computes the spectral density using a box spectral density function and multiplies 
it by the complement of the Fermi factor, which accounts for the occupation probability 
of the bath modes. The resulting spectral density is used in the chain mapping process 
to determine the chain coefficients for the empty thermofield chain representation of the bath.
"""
function thermofield_spectral_density(ω,bath::bath_parameters,::Empty)
    #Spectral density for an empty chain
    J = box_spectral_density(bath.Γ,ω,bath.D)
  #  J = semicircular_density(bath.Γ,ω,bath.D)
    return J * (1 - fermi_factor(ω,bath.β,bath.μ))
end

"""
    chain_mapping(bath::bath_parameters,sector::ThermofieldSector)

Calculates the chain coefficients using PolyChaos.jl for a given bath and thermofield 
sector. The function returns the chain coefficients α and β, which are used to construct 
the thermofield chain representation of the bath. The spectral density is computed based 
on the bath parameters and the specified sector (Filled or Empty). The support for the 
spectral function is set to be larger than the bath's bandwidth for numerical stability. 
The function utilizes the Measure and OrthoPoly classes from PolyChaos.jl to perform the 
necessary calculations and obtain the chain coefficients.
"""
function chain_mapping(bath::bath_parameters,sector::ThermofieldSector)

    spec_fun(ω) = thermofield_spectral_density(ω,bath,sector)

    #support needs to be larger than spectral function for numerical reasons.
    support = (-2*bath.D,2*bath.D)

    meas = Measure("thermofield",spec_fun,support,false,Dict())
    op = OrthoPoly("chain",bath.N-1,meas;Nquad=100000) #Calculates the chain coefficients using PolyChaos.jl

    α = coeffs(op)[:,1]
    β = coeffs(op)[:,2]

    return α,sqrt.(β)
end

"""
    chain_layout(N_left_bath,N_right_bath,Nsys)
Arranges the chains in interleaved fashion, with the system at the centre. The 
left and right baths are represented by their respective number of modes, and 
the system is defined by its number of sites. The function returns a 
`ChainLayout` struct that contains the indices of the left filled chain, left 
empty chain, system, right filled chain, and right empty chain.
"""
function chain_layout(N_left_bath,N_right_bath,Nsys)
  
    N = 2*(N_left_bath+N_right_bath) + Nsys

    ChainLayout(
        1:2:2*N_left_bath,
        2:2:2*N_left_bath,
        2*N_left_bath+1:2*N_left_bath+Nsys,
        2*N_left_bath+Nsys+1:2:N,
        2*N_left_bath+Nsys+2:2:N,
    )
end

"""
    add_chain(H,os,inds,energies,hoppings)

Adds MPO terms and associated single particle Hamiltonian elements for the 
thermofield chain.
"""
function add_chain(H,os,inds,energies,hoppings)
    ##Adds MPO terms and associated single particle hamiltonian elements 
    ##for the thermofield chain

    N = length(inds)
    for i in 1:N
        os += energies[i],"N",inds[i]

        H[inds[i],inds[i]] = energies[i]

        if i < N
            t = hoppings[i]
            os += t,"Cdag",inds[i],"C",inds[i+1]
            os += t,"Cdag",inds[i+1],"C",inds[i]

            H[inds[i],inds[i+1]] = t
            H[inds[i+1],inds[i]] = t
        end
    end
    return H,os
end

"""
    couple_sites(H,os,i,j,t)

Couples two sites, used for the system-chain coupling. This function adds the 
hopping terms between sites `i` and `j` to both the single-particle Hamiltonian 
matrix `H` and the OpSum representation `os`. The hopping amplitude is given by `t`.
"""
function couple_sites(H,os,i,j,t)
    #Couples two sites, used for the system-chain coupling

    os += t,"Cdag",i,"C",j
    os += t,"Cdag",j,"C",i

    H[i,j] = t
    H[j,i] = t
    return H,os
end

"""
    build_chain_hamiltonian(sites,left,right,sys)

Builds the single-particle Hamiltonian and many-body MPO for a system coupled to two 
thermofield chains. The left and right baths are represented by their respective 
parameters, and the system is defined by its energies, hoppings, and interactions. 
The function returns the MPO representation of the Hamiltonian, the single-particle 
Hamiltonian matrix, and the OpSum representation of the Hamiltonian.
"""
function build_chain_hamiltonian(sites,left,right,sys)
    #builds single particle hamiltonian and many body MPO

    layout = chain_layout(left.N,right.N,length(sys.ϵ))
    N = length(sites)
    Hsingle = zeros(ComplexF64,N,N)
    os = OpSum()

    #chain coefficients for the four chains
    εLF,tLF = chain_mapping(left,Filled())
    εLE,tLE = chain_mapping(left,Empty())
    εRF,tRF = chain_mapping(right,Filled())
    εRE,tRE = chain_mapping(right,Empty())

    #adds the terms for the chains
    Hsingle,os = add_chain(Hsingle,os,layout.left_filled,reverse(εLF),reverse(tLF))
    Hsingle,os = add_chain(Hsingle,os,layout.left_empty,reverse(εLE),reverse(tLE))
    Hsingle,os = add_chain(Hsingle,os,layout.right_filled,εRF,tRF[2:end])
    Hsingle,os = add_chain(Hsingle,os,layout.right_empty,εRE,tRE[2:end])

    #system
    for i in eachindex(sys.ϵ)
        ##energies
        os += sys.ϵ[i],"N",layout.system[i]
        Hsingle[layout.system[i],layout.system[i]] = sys.ϵ[i]
        if i < length(sys.ϵ)
            #hoppings
            Hsingle,os = couple_sites(Hsingle,os,layout.system[i],layout.system[i+1],sys.t[i])
            #interactions
            os += sys.U[i],"N",layout.system[i],"N",layout.system[i+1]
        end
    end

    #bath-system couplings
    Hsingle,os = couple_sites(Hsingle,os,last(layout.left_filled),first(layout.system),first(tLF))
    Hsingle,os = couple_sites(Hsingle,os,last(layout.left_empty),first(layout.system),first(tLE))
    Hsingle,os = couple_sites(Hsingle,os,first(layout.right_filled),last(layout.system),first(tRF))
    Hsingle,os = couple_sites(Hsingle,os,first(layout.right_empty),last(layout.system),first(tRE))
    return MPO(os,sites), Hsingle, os
end

"""
    thermofield_vacuum(left,right,system)

Defines the initial state for a system coupled to two thermofield chains. The left 
and right baths are represented by their respective parameters, and the system is 
defined by its occupations. The function returns a vector of occupation states for 
the left bath, system, and right bath, arranged in an interleaved ordering with the 
filled mode from the left.
"""
function thermofield_vacuum(left,right,system)
    #defines the initial state given both baths have an interleaved ordering
    #for the filled and empty chains. Both start with the filled mode from the left.

    Nbath = left.N
    Nsys = length(system.ϵ)
    left_bath_occ =[isodd(i) ? "Emp" : "Occ" for i in 1:2*left.N]
    right_bath_occ =[isodd(i) ? "Emp" : "Occ" for i in 1:2*right.N]
    sys_occ = system.occupations

    return [left_bath_occ;sys_occ;right_bath_occ]
end

"""
    propagate_correlations(corr0,H_single,times)

Compute the time evolution of a correlation matrix under a single-particle Hamiltonian. 
The correlation matrix C_ij = <c†_j c_i> evolves according to C(t) = U C(0) U†, where 
U = exp(-i H_single t). The function returns the correlation matrices at the specified 
times, transposed to match the standard definition G_ij = <c†_i c_j>.
"""
function propagate_correlations(corr0,H_single,times)
    
    δt = times[2] - times[1]
    U_step = exp(-im*δt*H_single)

    corrs = Vector{Any}(undef,length(times))
    corrs[1] = U_step*corr0*U_step'
    for i in 2:length(times)
        corrs[i] = U_step*corrs[i-1]*U_step'
    end
    return transpose.(corrs)
end

"""
    LB_current(left,right,system)

Performs a Landauer-Büttiker current calculation of the steady state particle 
current Jp of a system. This assumes both baths have the same spectral function, 
given by the box function.
"""
function LB_current(left,right,system)

    eta = 0.005 
    wsamp = 10000; # Frequency sampling.
    w = range(-10*left.D,10*left.D,wsamp); # Frequency axis for Landauer calculations (larger than the band).
    Γ = left.Γ+right.Γ
    dw = w[2] - w[1]; # Frequency increment.

    ρ = (1/(2*left.D))*(heaviside(w .+ left.D) .- heaviside(w .- left.D))
    Δ_L = (-left.Γ/(2*π))*log.((w.-left.D .+im*eta)./(w .+left.D .+im*eta))    #exact result (only valid for box)
    Δ_R = (-right.Γ/(2*π))*log.((w.-right.D .+im*eta)./(w .+right.D .+im*eta)) #exact result (only valid for box)
        Δ_L = -im*left.D*left.Γ*hilbert(ρ) #more generally
        Δ_R = -im*left.D*right.Γ*hilbert(ρ) #more generally

    ####Calculate determinant of M matrix (N_sys x N_sys)
    N_sys = length(system.ϵ)
    if N_sys == 1
        Δ = Δ_L + Δ_R
        detM = w .- system.ϵ[1] .- Δ
        G = 1 ./detM
        A_den = (-1/π).*imag.(G)    
    elseif N_sys == 2
        detM = (w .-system.ϵ[1] .-Δ_L).*(w.-system.ϵ[2].-Δ_R)
        detM = detM .- abs.(system.t[1]^2) 
        G = ((w .-system.ϵ[2] .-Δ_R))./detM
        A_den = (-1/π).*imag.(G)    
    elseif N_sys == 3
        detM = (w .-system.ϵ[1] .-Δ_L).*(w.-system.ϵ[3].-Δ_R).*(w.-system.ϵ[2])
        detM = detM - (abs.(system.t[2])^2)*(w.-system.ϵ[1].-Δ_L)
        detM = detM - (abs.(system.t[1])^2)*(w.-system.ϵ[3].-Δ_R)
        G =  (w .-system.ϵ[1] .-Δ_L).*(w.-system.ϵ[3].-Δ_R)./detM
        A_den = (-1/π)*imag.(G)
    end

    A = (left.D*Γ*ρ/π) ./(abs.(detM).^2)
    if length(system.t)>0
        coupling_factor = prod(abs.(system.t).^2)
        A = A.*coupling_factor
    end    
    f_L = fermi_factor.(w,left.β,left.μ)
    f_R = fermi_factor.(w,right.β,right.μ)

    prefactor = (4*π*left.D*left.Γ*right.Γ)/(2*π*Γ)
    Jp = prefactor*sum(ρ.*A.*(f_L - f_R))*dw;

    return Jp
end

end