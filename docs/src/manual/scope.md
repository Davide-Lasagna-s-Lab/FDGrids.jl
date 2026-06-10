# Scope and Limitations

`FDGrids.jl` is a compact finite-difference building block, not a full PDE
framework. It focuses on one-dimensional differentiation matrices that can be
reused efficiently across vectors and array fibers.

## Designed For

- Non-uniform finite-difference coefficients from Fornberg's algorithm.
- Compact `DiffMatrix` storage with fixed-width row stencils.
- Generated `mul!` kernels for vectors and arrays of arbitrary rank.
- Ordinary and weighted adjoints without forming dense matrices.
- Grid constructors with quadrature weights that match the node distribution.
- Compact no-pivot banded LU for repeated solves.
- Slab-local multiplication when an external code manages halo values.
- Optional CUDA GPU kernels for forward and adjoint application,
  loaded automatically when `using CUDA` is brought into scope.

This makes the package a good fit for stability calculations, collocation or
finite-difference boundary value problems, and time-stepping codes that already
own their state arrays and boundary-condition logic.

## Not Provided

The package intentionally leaves these layers to user code or companion
packages:

- boundary-condition objects,
- time integrators,
- multi-dimensional operator assembly,
- automatic halo exchange,
- sparse-matrix conversion helpers,
- periodic grids and periodic stencils,
- adaptive or moving grids,
- interpolation between grids,
- GPU linear solves (only `mul!` paths are GPU-enabled).

The important consequence is that `FDGrids.jl` gives you fast local operators,
but it does not decide the physics or communication pattern of a larger solver.

## Reading Path

Start with [Grids and Quadrature](grids.md) and
[Finite-Difference Operators](diffmatrix.md). Read [Adjoints](adjoints.md) if
you use weighted inner products, [Linear Solves](linear-solves.md) if you need
boundary value problems or compact LU, and [Decomposed Domains](decomposed-domains.md)
if your arrays include halo storage. If you run on NVIDIA hardware, see
[GPU Support](gpu.md). The mathematical and implementation details are
separated into [Numerical Methods](methods.md) and
[Internal Layout and Kernels](internals.md).
