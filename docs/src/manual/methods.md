# Numerical Methods

This page describes the numerical algorithms used by `FDGrids.jl` and the
assumptions behind them. The package is built around one idea: construct a
one-dimensional finite-difference operator once, store only its local stencil
coefficients, and reuse it efficiently.

## Finite-Difference Weights

Finite-difference coefficients are computed with Fornberg's recursive
algorithm for arbitrarily spaced nodes. Given distinct nodes

```math
x_0, x_1, \ldots, x_n
```

and an evaluation point `\xi`, the algorithm computes coefficients `c_{j,k}`
such that

```math
f^{(k)}(\xi) \approx \sum_{j=0}^n c_{j,k} f(x_j),
\qquad k = 0,\ldots,m.
```

The coefficients are exact on polynomials up to degree `n`:

```math
p^{(k)}(\xi) = \sum_{j=0}^n c_{j,k} p(x_j),
\qquad \deg p \le n.
```

This polynomial exactness is the local property used to assemble every row of a
`DiffMatrix`. For smooth functions on well-resolved grids, increasing the
stencil width usually improves accuracy. On arbitrary non-uniform grids,
however, the error constants depend on the node spacing and can grow if the
stencil nodes are poorly distributed.

The implementation in `utils.jl` follows Fornberg's recurrence and returns the
full coefficient table for derivative orders `0:m`. `get_coeffs(xs, width,
order)` then selects the column for the requested derivative order and stores it
row-by-row.

## Stencil Selection

For an odd stencil width `W`, define

```math
H = \lfloor W/2 \rfloor.
```

For row `i`, the first stencil index is

```math
\ell_i = \operatorname{clamp}(i-H,\,1,\,N-W+1),
```

and the row uses the nodes

```math
x_{\ell_i}, x_{\ell_i+1}, \ldots, x_{\ell_i+W-1}.
```

Interior rows are centered around `x_i`. Boundary rows use the nearest valid
one-sided stencil of the same width. This choice gives every forward row the
same number of stored coefficients, which is essential for the compact
`DiffMatrix` layout and generated multiplication kernels.

The package does not attach semantic boundary conditions to a derivative
matrix. Boundary conditions are imposed later by modifying rows, as shown in
[Linear Solves](linear-solves.md#Boundary-Value-Problem).

## Differentiation Matrices

A `DiffMatrix` represents the logical dense matrix

```math
D_{i,j}
```

but stores only the `W` nonzero stencil entries in each row. Applying the
operator to a vector is mathematically the usual matrix-vector product:

```math
y_i = \sum_j D_{i,j} x_j.
```

For arrays, the same one-dimensional matrix is applied independently to every
fiber along the selected dimension:

```math
y[\ldots,i,\ldots]
=
\sum_j D_{i,j}x[\ldots,j,\ldots].
```

The generated kernels specialize on array rank, differentiated dimension, and
stencil width. This makes the inner dot product fixed-size and allows the
compiler to unroll it.

## Quadrature

`grid` returns quadrature weights paired with the nodes:

```math
\int_l^h f(x)\,dx \approx \sum_i f(x_i) w_i.
```

The available rules are:

- `UniformGrid()`: equally spaced nodes with composite trapezoidal weights.
- `MappedGrid(alpha, order)`: mapped Chebyshev-like nodes with composite
  Newton-Cotes weights on local panels.
- `GaussLobattoGrid()`: Chebyshev-Lobatto nodes with Clenshaw-Curtis weights.

For a single Newton-Cotes panel with nodes `x_1,\ldots,x_s`, the weights are
computed by moment matching:

```math
\sum_{i=1}^s w_i x_i^d
=
\frac{x_s^{d+1}-x_1^{d+1}}{d+1},
\qquad d=0,\ldots,s-1.
```

Composite Newton-Cotes panels are then accumulated across the grid. These
weights can be negative on high-order or strongly non-uniform panels.

For `GaussLobattoGrid`, the nodes are Chebyshev-Lobatto points and the weights
are Clenshaw-Curtis weights computed from the explicit cosine-series formula
described by Waldvogel. These weights are positive for the supported grids and
are the recommended choice when a positive weighted inner product is needed.

## Adjoints

The ordinary adjoint represents the transpose:

```math
D^* = D^T.
```

For a weighted inner product with positive weights `w` and
`W = Diagonal(w)`, the weighted adjoint is

```math
D^+ = W^{-1}D^T W.
```

It satisfies

```math
(Du)^T W v = u^T W (D^+v).
```

The factors `w_i/w_j` are inserted into the adjoint coefficients once during
construction, so applying a weighted adjoint uses the same multiplication
kernel as applying an unweighted adjoint.

## Linear Solves

The compact in-place LU factorisation uses the standard no-pivot banded
algorithm. The strict lower band stores the multipliers of `L`, while the
diagonal and upper band store `U`. Generated triangular solves specialize on
the stencil width encoded in the `DiffMatrix` type.

The LAPACK path, `lu(D)`, converts to LAPACK's general banded format and uses
pivoted `gbtrf!`/`gbtrs!`. The compact path, `lu!(copy(D))`, avoids that
workspace conversion but does not pivot. Use it when the problem is known to be
well behaved without row exchanges. See [Linear Solves](linear-solves.md) for
the detailed discussion.

## References

- B. Fornberg, "Generation of Finite Difference Formulas on Arbitrarily Spaced
  Grids", *Mathematics of Computation*, 51(184), 699-706, 1988.
- B. Fornberg, *A Practical Guide to Pseudospectral Methods*, Cambridge
  University Press, 1998.
- J. M. Lopez, D. Feldmann, M. Rampp, A. Vela-Martin, L. Shi, and M. Avila,
  "nsCouette: A High-Performance Code for Direct Numerical Simulations of
  Turbulent Taylor-Couette Flow", *SoftwareX*, 11, 100395, 2020.
- J. Waldvogel, "Fast Construction of the Fejer and Clenshaw-Curtis Quadrature
  Rules", *BIT Numerical Mathematics*, 46, 195-202, 2006.
- G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th edition, Johns
  Hopkins University Press, 2013.
