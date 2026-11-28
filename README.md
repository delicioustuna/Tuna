# Tuna

[![NuGet](https://img.shields.io/nuget/v/Tuna)](https://www.nuget.org/packages/Tuna)
[![Releases](https://img.shields.io/github/release/delicioustuna/Tuna)](https://github.com/delicioustuna/Tuna/releases)

[日本語 (Japanese)](README_ja.md)

**T**iny **U**tilities for **N**umerical **A**nalysis — A pure C#, SIMD-accelerated numerical computing library for .NET 10.

Tuna provides fast linear algebra, special functions, and deterministic random/quasi-random generators with zero native dependencies, alongside modular extension packages for statistics, signal processing, optimization, integration, automatic differentiation, MCMC, and neural networks.

## Key Features

- **Pure C# & Zero Native Dependencies**: Runs anywhere .NET 10 runs, with SIMD-accelerated kernels and no native BLAS/LAPACK binaries to deploy.
- **Zero-Copy Views (`readonly ref struct`)**: Transpose (`A.T`), slice (`A[1..3, ..]`), row/column/diagonal views, and explicit zero-copy broadcasting (`v.Broadcast(cols)`) without allocating.
- **Allocation-Aware API**: Elementwise and linear algebra operations in `XMath` provide allocating, `destination`-based, and `XMath.InPlace` overloads—plus reusable decomposition buffers (`Recompute`).
- **Modular Packages**: Use `Tuna` for core numerics, and combine domain-specific `Tuna.*` packages as needed.

## Installation

This library is distributed via [NuGet](https://www.nuget.org/packages/Tuna) (requires .NET 10).

```bash
dotnet add package Tuna
```

Add optional extension packages as needed:

```bash
dotnet add package Tuna.Statistics
dotnet add package Tuna.SignalProcessing
dotnet add package Tuna.Optimization
dotnet add package Tuna.Integration
dotnet add package Tuna.Differentiation
dotnet add package Tuna.Probabilistic
dotnet add package Tuna.NeuralNetwork
```

## Packages

| Package | NuGet | Description |
| --- | --- | --- |
| [`Tuna`](src/Tuna) | [![NuGet](https://img.shields.io/nuget/v/Tuna)](https://www.nuget.org/packages/Tuna) | Core package: `Vector<T>`, `Matrix<T>`, `Tensor<T>`, `XMath`, BLAS & decompositions (LU/Cholesky/QR/Eigen/SVD), special functions, PRNG/QRNG |
| [`Tuna.Statistics`](src/Tuna.Statistics) | [![NuGet](https://img.shields.io/nuget/v/Tuna.Statistics)](https://www.nuget.org/packages/Tuna.Statistics) | Continuous, discrete, and multivariate probability distributions (PDF/CDF/quantiles/sampling) |
| [`Tuna.SignalProcessing`](src/Tuna.SignalProcessing) | [![NuGet](https://img.shields.io/nuget/v/Tuna.SignalProcessing)](https://www.nuget.org/packages/Tuna.SignalProcessing) | 1D/2D FFT, real FFT, DCT/DST, convolution, correlation, and discrete/continuous wavelets |
| [`Tuna.Optimization`](src/Tuna.Optimization) | [![NuGet](https://img.shields.io/nuget/v/Tuna.Optimization)](https://www.nuget.org/packages/Tuna.Optimization) | L-BFGS / L-BFGS-B, SGD / Adam / AdamW, proximal methods (FISTA/ADMM), global optimization, genetic algorithms |
| [`Tuna.Integration`](src/Tuna.Integration) | [![NuGet](https://img.shields.io/nuget/v/Tuna.Integration)](https://www.nuget.org/packages/Tuna.Integration) | 1D adaptive quadrature, multidimensional cubature, and quasi-Monte Carlo integration |
| [`Tuna.Differentiation`](src/Tuna.Differentiation) | [![NuGet](https://img.shields.io/nuget/v/Tuna.Differentiation)](https://www.nuget.org/packages/Tuna.Differentiation) | Tape-based reverse-mode automatic differentiation for scalars, vectors, matrices, tensors, and decompositions |
| [`Tuna.Probabilistic`](src/Tuna.Probabilistic) | [![NuGet](https://img.shields.io/nuget/v/Tuna.Probabilistic)](https://www.nuget.org/packages/Tuna.Probabilistic) | MCMC samplers (HMC, NUTS, slice, ensemble, parallel tempering) and convergence diagnostics ($\hat{R}$, ESS, MCSE) |
| [`Tuna.NeuralNetwork`](src/Tuna.NeuralNetwork) | [![NuGet](https://img.shields.io/nuget/v/Tuna.NeuralNetwork)](https://www.nuget.org/packages/Tuna.NeuralNetwork) | 2D convolution, Mamba blocks, normalization layers, and SafeTensors reader |

## Quick Start

### Linear Algebra & Zero-Copy Views

```csharp
using Tuna;
using Tuna.LinearAlgebra;

var A = Matrix<double>.FromRowMajor(3, 3,
[
    4, 1, 0,
    1, 3, 1,
    0, 1, 2,
]);
var b = new Vector<double>([1.0, 2.0, 3.0]);

// Solve A * x = b
Vector<double> x = XMath.Solve(A, b);

// Cholesky decomposition
var chol = A.Cholesky();
double logDet = chol.LogDeterminant;

// Zero-copy transpose, slicing, and broadcasting
Matrix<double> C = A.T * A[0..3, 0..2];
Vector<double> rowMeans = XMath.RowMeans(A);
XMath.Subtract(A, rowMeans.Broadcast(A.Columns), A); // center columns in-place
```

### Allocation Control (`XMath`)

Most `XMath` operations provide three consistent forms:

```csharp
var v = new Vector<double>([0.0, 1.0, 2.0]);
var buffer = new Vector<double>(v.Length);

Vector<double> y = XMath.Exp(v);     // 1. Allocate a new container
XMath.Exp(v, buffer);                // 2. Write into an existing destination buffer
XMath.InPlace.Exp(v);                // 3. Mutate in-place
```

### Special Functions & Deterministic RNG

```csharp
using Tuna;
using Tuna.RandomSource;

double logGamma = XMath.Special.LogGamma(200.0);
double tail = XMath.Special.Erfc(6.0);

var rng = new Pcg64DxsmRandom(seed: 42);
double u = rng.NextDouble();
```

### Extension Packages (e.g. Reverse-Mode Autodiff)

```csharp
using Tuna;
using Tuna.Differentiation;

var tape = new Tape();
var x = tape.Leaf(2.0);
var y = XMath.Sin(x) + x * x;
tape.Backward(y);
double dydx = tape.GetGradient(x); // cos(2) + 4
```

## Design Highlights

- **Column-Major Layout**: `Matrix<T>` stores elements in column-major order. Datasets follow the convention of **1 column = 1 sample** (`dimensions × samples`).
- **No Implicit Broadcasting**: Shapes must match explicitly; use `v.Broadcast(cols)` / `v.BroadcastRows(rows)` to create zero-copy broadcast views.
- **Strict Aliasing Rules**: Elementwise operations allow exact-match in-place updates (partial overlap throws `ArgumentException`). Matrix products, transposes, and reductions forbid input/output overlap.

## Requirements

- .NET 10 (`net10.0`)

## Documentation

Detailed guides, design notes, and generated API references are available in [`documents/`](documents/README.md) (Japanese):

- [Getting Started (Containers & Views)](documents/getting-started.md)
- [`XMath` Guide](documents/xmath.md)
- [Linear Algebra & Matrix Decompositions](documents/linear-algebra.md)
- [Special Functions (`XMath.Special`)](documents/special-functions.md)
- [Random & Quasi-Random Generators](documents/random.md)
- [Extension Packages Overview](documents/extensions.md)
- [Performance & Memory Optimization](documents/appendix/performance-and-memory.md)
- [Design Notes](documents/design/README.md)
- [API Reference](documents/api/README.md)

## License

This library is licensed under the [MIT License](LICENSE.txt).
