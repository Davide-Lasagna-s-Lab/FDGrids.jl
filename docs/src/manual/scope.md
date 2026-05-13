# Scope and Limitations

`FDGrids.jl` is deliberately narrow: it builds compact one-dimensional
finite-difference operators and applies them efficiently to vectors or to fibers
of higher-dimensional arrays.

It is a good fit when you already know the grid direction, stencil width, and
boundary treatment you want, and you need to apply the same operator many times
inside a solver, stability calculation, or time-stepping loop.

## What the Package Provides

- Non-uniform finite-difference coefficients from Fornberg's algorithm.
- Compact `DiffMatrix` storage with fixed-width row stencils.
- Generated `mul!` kernels for vectors and arrays of arbitrary rank.
- Ordinary and weighted adjoints without forming dense matrices.
- Grid constructors with matching quadrature weights.
- Compact no-pivot banded LU routines for repeated solves.
- Lower-level hooks for slab-local or decomposed-domain storage.

## What It Does Not Try to Be

`FDGrids.jl` is not a full PDE framework. In particular, it does not currently
provide:

- time integrators,
- boundary-condition objects,
- multi-dimensional operator assembly,
- automatic halo exchange,
- GPU kernels,
- sparse-matrix conversion helpers,
- periodic grids or periodic stencils,
- adaptive grids,
- interpolation/resampling between grids.

Those are natural layers to build on top of the current package, but keeping
them separate helps the core stay small and predictable.

## Recommended Reading Order

Start with [Grids and Quadrature](grids.md) and
[Finite-Difference Operators](diffmatrix.md). Then read [Adjoints](adjoints.md)
if you use quadrature-weighted inner products,
[Decomposed Domains](decomposed-domains.md) if you own halo management outside
the package, and [Linear Solves](linear-solves.md) if you solve banded systems
with the compact operator.
