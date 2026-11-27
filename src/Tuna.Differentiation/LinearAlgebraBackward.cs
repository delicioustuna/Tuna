using System.Runtime.InteropServices;
using Tuna.LinearAlgebra;

namespace Tuna.Differentiation;

internal static partial class LinearAlgebraBackward
{
    // Fixed block size depends only on order so results do not vary with thread count.
    internal const int CholeskyBlock = 64;

    public static void Backward(ArrayStore<double> store, in ArrayRecord record, double[] adjoints)
    {
        switch (record.Operation)
        {
            case ArrayOperation.LuSolveVector:
            case ArrayOperation.LuSolveMatrix:
            case ArrayOperation.LuSolveTransposedVector:
            case ArrayOperation.LuSolveTransposedMatrix:
                LuSolve(store, record);
                break;
            case ArrayOperation.LuLogAbsDeterminant:
                LuLogAbsDeterminant(store, record, adjoints[record.Output]);
                break;
            case ArrayOperation.SolveLowerTriangularVector:
            case ArrayOperation.SolveLowerTriangularMatrix:
            case ArrayOperation.SolveUpperTriangularVector:
            case ArrayOperation.SolveUpperTriangularMatrix:
                SolveTriangular(store, record);
                break;
            case ArrayOperation.Cholesky:
                Cholesky(store, record);
                break;
            case ArrayOperation.CholeskyLogDeterminant:
                CholeskyLogDeterminant(store, record, adjoints[record.Output]);
                break;
            case ArrayOperation.CholeskySolveVector:
            case ArrayOperation.CholeskySolveMatrix:
                CholeskySolve(store, record);
                break;
            case ArrayOperation.CholeskyInverseQuadraticForm:
                CholeskyInverseQuadraticForm(store, record, adjoints[record.Output]);
                break;
            case ArrayOperation.Qr:
                Qr(store, record);
                break;
            case ArrayOperation.SymmetricEigen:
                SymmetricEigen(store, record);
                break;
            case ArrayOperation.SingularValueDecomposition:
                SingularValue(store, record);
                break;
            case ArrayOperation.SvdSolve:
                SvdSolve(store, record);
                break;
            case ArrayOperation.SvdPseudoInverse:
                SvdPseudoInverse(store, record);
                break;
        }
    }

    private static void LuSolve(ArrayStore<double> store, in ArrayRecord record)
    {
        int output = record.Output;
        if (!store.Touched(output))
        {
            return;
        }

        ArrayView a = record.Left, b = record.Right;
        bool transposed = record.Operation is ArrayOperation.LuSolveTransposedVector or ArrayOperation.LuSolveTransposedMatrix;
        bool vector = record.Operation is ArrayOperation.LuSolveVector or ArrayOperation.LuSolveTransposedVector;
        var lu = store.Decompositions.Lu(record.ScalarInput);
        int n = b.Rows, k = b.Columns;
        var g = new ReadOnlyMatrixSpan<double>(store.OutputGradient(output), n, k);
        var u = new MatrixSpan<double>(store.Scratch(ArrayStore<double>.SolveAdjoint, n * k), n, k);
        if (vector)
        {
            var uVector = u.Column(0);
            if (transposed)
            {
                lu.Solve(g.Column(0), uVector);
            }
            else
            {
                lu.SolveTransposed(g.Column(0), uVector);
            }
        }
        else if (transposed)
        {
            lu.Solve(g, u);
        }
        else
        {
            lu.SolveTransposed(g, u);
        }

        if (store.RequiresGradient(b.Slot))
        {
            store.AddGradient(b, u.Data.Slice(0, n * k), negate: false);
        }

        if (store.RequiresGradient(a.Slot))
        {
            ReadOnlySpan<double> x = store.Values(output).Slice(0, n * k);
            Span<double> contribution = store.Scratch(ArrayStore<double>.MatrixContribution, n * n);
            ReadOnlySpan<double> left = transposed ? x : u.Data, right = transposed ? u.Data : x;
            Product(n, k, left, right, contribution, triangle: Triangle.Full);
            store.AddGradient(a, contribution, negate: true);
        }
    }

    private static void LuLogAbsDeterminant(ArrayStore<double> store, in ArrayRecord record, double adjoint)
    {
        ArrayView a = record.Left;
        if (adjoint == 0 || !store.RequiresGradient(a.Slot))
        {
            return;
        }

        var lu = store.Decompositions.Lu(record.ScalarInput);
        int n = lu.Order;
        Span<double> contribution = store.Scratch(ArrayStore<double>.MatrixContribution, n * n);
        if (lu.IsSingular)
        {
            contribution.Fill(double.NaN);
        }
        else
        {
            var inverse = new MatrixSpan<double>(contribution, n, n);
            lu.Inverse(inverse);
            TransposeInPlace(contribution, n);
            Scale(contribution, adjoint);
        }

        store.AddGradient(a, contribution, negate: false);
    }

    private static void SolveTriangular(ArrayStore<double> store, in ArrayRecord record)
    {
        int output = record.Output;
        if (!store.Touched(output))
        {
            return;
        }

        ArrayView a = record.Left, b = record.Right;
        bool lower = record.Operation is ArrayOperation.SolveLowerTriangularVector or ArrayOperation.SolveLowerTriangularMatrix;
        bool vector = record.Operation is ArrayOperation.SolveLowerTriangularVector or ArrayOperation.SolveUpperTriangularVector;
        int n = b.Rows, k = b.Columns;
        // MatrixContribution is only written after the triangular solve finishes, so broadcast materialization can safely reuse it here.
        ReadOnlyMatrixSpan<double> coefficientT = store.CoefficientView(a, ArrayStore<double>.MatrixContribution, transpose: true);
        var g = new ReadOnlyMatrixSpan<double>(store.OutputGradient(output), n, k);
        var w = new MatrixSpan<double>(store.Scratch(ArrayStore<double>.SolveAdjoint, n * k), n, k);
        if (vector)
        {
            if (lower)
            {
                XMath.SolveUpperTriangular(coefficientT, g.Column(0), w.Column(0));
            }
            else
            {
                XMath.SolveLowerTriangular(coefficientT, g.Column(0), w.Column(0));
            }
        }
        else if (lower)
        {
            XMath.SolveUpperTriangular(coefficientT, g, w);
        }
        else
        {
            XMath.SolveLowerTriangular(coefficientT, g, w);
        }

        if (store.RequiresGradient(b.Slot))
        {
            store.AddGradient(b, w.Data.Slice(0, n * k), negate: false);
        }

        if (store.RequiresGradient(a.Slot))
        {
            ReadOnlySpan<double> x = store.Values(output).Slice(0, n * k);
            Span<double> contribution = store.Scratch(ArrayStore<double>.MatrixContribution, n * n);
            Product(n, k, w.Data, x, contribution, lower ? Triangle.Lower : Triangle.Upper);
            store.AddGradient(a, contribution, negate: true);
        }
    }

    private static void CholeskySolve(ArrayStore<double> store, in ArrayRecord record)
    {
        int output = record.Output;
        if (!store.Touched(output))
        {
            return;
        }

        ArrayView a = record.Left, b = record.Right;
        var cholesky = store.Decompositions.Cholesky(record.ScalarInput);
        int n = b.Rows, k = b.Columns;
        var g = new ReadOnlyMatrixSpan<double>(store.OutputGradient(output), n, k);
        var u = new MatrixSpan<double>(store.Scratch(ArrayStore<double>.SolveAdjoint, n * k), n, k);
        if (record.Operation == ArrayOperation.CholeskySolveVector)
        {
            cholesky.Solve(g.Column(0), u.Column(0));
        }
        else
        {
            cholesky.Solve(g, u);
        }

        if (store.RequiresGradient(b.Slot))
        {
            store.AddGradient(b, u.Data.Slice(0, n * k), negate: false);
        }

        if (store.RequiresGradient(a.Slot))
        {
            ReadOnlySpan<double> x = store.Values(output).Slice(0, n * k);
            Span<double> contribution = store.Scratch(ArrayStore<double>.MatrixContribution, n * n);
            Product(n, k, u.Data, x, contribution, Triangle.Symmetric);
            store.AddGradient(a, contribution, negate: true);
        }
    }

    private static void CholeskyInverseQuadraticForm(ArrayStore<double> store, in ArrayRecord record, double adjoint)
    {
        if (adjoint == 0)
        {
            return;
        }

        ArrayView a = record.Left, x = record.Right;
        var cholesky = store.Decompositions.Cholesky(record.ScalarInput);
        int n = x.Rows;
        Span<double> w = store.Scratch(ArrayStore<double>.SolveAdjoint, n);
        cholesky.Solve(store.VectorView(x), new VectorSpan<double>(w));
        if (store.RequiresGradient(a.Slot))
        {
            Span<double> contribution = store.Scratch(ArrayStore<double>.MatrixContribution, n * n);
            Product(n, 1, w, w, contribution, Triangle.Symmetric);
            Scale(contribution, -adjoint);
            store.AddGradient(a, contribution, negate: false);
        }

        if (store.RequiresGradient(x.Slot))
        {
            // Multiply by adjoint before doubling to avoid overflowing 2 * adjoint when adjoint * w[i] is finite.
            for (int i = 0; i < w.Length; i++)
            {
                w[i] = 2 * (adjoint * w[i]);
            }

            store.AddGradient(x, w, negate: false);
        }
    }

    private static void CholeskyLogDeterminant(ArrayStore<double> store, in ArrayRecord record, double adjoint)
    {
        ArrayView a = record.Left;
        if (adjoint == 0 || !store.RequiresGradient(a.Slot))
        {
            return;
        }

        var cholesky = store.Decompositions.Cholesky(record.ScalarInput);
        int n = cholesky.L.Rows;
        Span<double> contribution = store.Scratch(ArrayStore<double>.MatrixContribution, n * n);
        cholesky.Inverse(new MatrixSpan<double>(contribution, n, n));
        // Multiply off-diagonals by adjoint before doubling so 2 * adjoint cannot overflow prematurely.
        for (int j = 0; j < n; j++)
        {
            Span<double> column = contribution.Slice(j * n, n);
            column.Slice(0, j).Clear();
            column[j] *= adjoint;
            for (int i = j + 1; i < n; i++)
            {
                column[i] = 2 * (column[i] * adjoint);
            }
        }

        store.AddGradient(a, contribution, negate: false);
    }

    // Blocked reverse-mode Cholesky differentiation (Murray 2016) traversing left-looking blocks in reverse.
    // Works on transposed upper-triangular W = tril(dL)^T so all triangular solves become in-place left solves.
    private static void Cholesky(ArrayStore<double> store, in ArrayRecord record)
    {
        int output = record.Output;
        ArrayView a = record.Left;
        if (!store.Touched(output) || !store.RequiresGradient(a.Slot))
        {
            return;
        }

        int n = a.Rows;
        ReadOnlySpan<double> l = store.Values(output).Slice(0, n * n);
        ReadOnlySpan<double> dl = store.OutputGradient(output);
        Span<double> work = store.Scratch(ArrayStore<double>.MatrixContribution, n * n);
        for (int j = 0; j < n; j++)
        {
            Span<double> column = work.Slice(j * n, n);
            for (int i = 0; i <= j; i++)
            {
                column[i] = dl[j + i * n];
            }

            column.Slice(j + 1).Clear();
        }

        var factor = new ReadOnlyMatrixSpan<double>(l, n, n);
        var w = new MatrixSpan<double>(work, n, n);
        int last = n == 0 ? 0 : (n - 1) / CholeskyBlock * CholeskyBlock;
        for (int k = last; k >= 0 && n > 0; k -= CholeskyBlock)
        {
            int b = Math.Min(CholeskyBlock, n - k), m = n - k - b;
            int below = k + b;
            ReadOnlyMatrixSpan<double> diagonalT = factor[k..below, k..below].T;
            if (m > 0)
            {
                var meanT = w[k..below, below..n];
                XMath.SolveUpperTriangular(diagonalT, meanT.AsReadOnly(), meanT);
                Gemm(true, true, b, b, m, l, below + k * n, n, work, k + below * n, n, work, k + k * n, n);
                if (k > 0)
                {
                    Gemm(true, false, k, m, b, l, k, n, work, k + below * n, n, work, below * n, n);
                    Gemm(true, true, k, b, m, l, below, n, work, k + below * n, n, work, k * n, n);
                }
            }

            DiagonalBlock(store, l, work, n, k, b);
        }

        TransposeInPlace(work, n);
        store.AddGradient(a, work, negate: false);
    }

    private static void DiagonalBlock(ArrayStore<double> store, ReadOnlySpan<double> l, Span<double> work, int n, int k, int b)
    {
        var factor = new ReadOnlyMatrixSpan<double>(l, n, n);
        ReadOnlyMatrixSpan<double> diagonalT = factor[k..(k + b), k..(k + b)].T;
        Span<double> first = store.Scratch(ArrayStore<double>.BlockFirst, b * b);
        Span<double> second = store.Scratch(ArrayStore<double>.BlockSecond, b * b);
        for (int j = 0; j < b; j++)
        {
            for (int i = 0; i < b; i++)
            {
                first[i + j * b] = i >= j ? work[k + j + (k + i) * n] : 0;
            }
        }

        Blas.Gemm(BlasTranspose.Transpose, BlasTranspose.None, b, b, b, 1.0, l.Slice(k + k * n), n, first, b, 0.0, second, b);
        for (int j = 0; j < b; j++)
        {
            Span<double> column = second.Slice(j * b, b);
            column.Slice(0, j).Clear();
            column[j] *= 0.5;
        }

        var y = new MatrixSpan<double>(second, b, b);
        XMath.SolveUpperTriangular(diagonalT, y.AsReadOnly(), y);
        var sT = new MatrixSpan<double>(first, b, b);
        y.AsReadOnly().T.CopyTo(sT);
        XMath.SolveUpperTriangular(diagonalT, sT.AsReadOnly(), sT);
        for (int j = 0; j < b; j++)
        {
            for (int i = 0; i < b; i++)
            {
                second[i + j * b] = first[i + j * b] + first[j + i * b];
            }
        }

        if (k > 0)
        {
            Blas.Gemm(BlasTranspose.Transpose, BlasTranspose.None, k, b, b, -1.0, l.Slice(k), n, second, b, 1.0, work.Slice(k * n), n);
        }

        // Read diagonal 0.5 * (S + S^T)[j, j] directly from S^T (first) to avoid overflowing 2 * S[j, j].
        for (int j = 0; j < b; j++)
        {
            Span<double> column = work.Slice(k + (k + j) * n, b);
            for (int i = 0; i < j; i++)
            {
                column[i] = second[i + j * b];
            }

            column[j] = first[j + j * b];
            column.Slice(j + 1).Clear();
        }
    }

    // Thin QR differentiation (Liao et al. 2019), partitioning R = [X | V] and A = [. | Y] when m < n; strictly lower entries of dR are ignored.
    private static void Qr(ArrayStore<double> store, in ArrayRecord record)
    {
        ArrayView a = record.Left;
        var pool = store.Decompositions;
        ref DecompositionInfo info = ref pool.Info(record.ScalarInput);
        int qSlot = info.Factor, rSlot = info.Second;
        bool qTouched = store.Touched(qSlot), rTouched = store.Touched(rSlot);
        if (!(qTouched || rTouched) || !store.RequiresGradient(a.Slot))
        {
            return;
        }

        var qr = pool.Qr(record.ScalarInput);
        int m = qr.Rows, n = qr.Columns, k = Math.Min(m, n);
        if (k == 0)
        {
            return;
        }

        ReadOnlySpan<double> q = store.Values(qSlot).Slice(0, m * k);
        ReadOnlySpan<double> r = store.Values(rSlot).Slice(0, k * n);
        ReadOnlySpan<double> dr = rTouched ? store.OutputGradient(rSlot) : default;
        Span<double> bt = store.Scratch(ArrayStore<double>.DecompositionFirst, k * m);
        Span<double> mm = store.Scratch(ArrayStore<double>.DecompositionSecond, k * k);
        Span<double> contribution = store.Scratch(ArrayStore<double>.MatrixContribution, m * n);
        if (qTouched)
        {
            Transpose(m, k, store.OutputGradient(qSlot), bt);
        }
        else
        {
            bt.Clear();
        }

        if (rTouched && n > k)
        {
            Span<double> y = store.Scratch(ArrayStore<double>.DecompositionThird, m * (n - k));
            store.MatrixView(a)[.., k..].CopyTo(new MatrixSpan<double>(y, m, n - k));
            Blas.Gemm(BlasTranspose.None, BlasTranspose.Transpose, k, m, n - k, 1.0, dr.Slice(k * k), k, y, m, 1.0, bt, k);
        }

        double beta = 0.0;
        if (rTouched)
        {
            Span<double> upper = store.Scratch(ArrayStore<double>.DecompositionThird, k * k);
            for (int j = 0; j < k; j++)
            {
                Span<double> column = upper.Slice(j * k, k);
                dr.Slice(j * k, j + 1).CopyTo(column);
                column.Slice(j + 1).Clear();
            }

            Blas.Gemm(BlasTranspose.None, BlasTranspose.Transpose, k, k, k, 1.0, r, k, upper, k, 0.0, mm, k);
            beta = 1.0;
        }

        Blas.Gemm(BlasTranspose.None, BlasTranspose.None, k, k, m, -1.0, bt, k, q, m, beta, mm, k);
        for (int j = 1; j < k; j++)
        {
            for (int i = 0; i < j; i++)
            {
                mm[i + j * k] = mm[j + i * k];
            }
        }

        Blas.Gemm(BlasTranspose.None, BlasTranspose.Transpose, k, m, k, 1.0, mm, k, q, m, 1.0, bt, k);
        var system = new MatrixSpan<double>(bt, k, m);
        XMath.SolveUpperTriangular(new ReadOnlyMatrixSpan<double>(r.Slice(0, k * k), k, k), system.AsReadOnly(), system);
        Transpose(k, m, bt, contribution);
        if (n > k)
        {
            Span<double> rest = contribution.Slice(m * k, m * (n - k));
            if (rTouched)
            {
                Blas.Gemm(BlasTranspose.None, BlasTranspose.None, m, n - k, k, 1.0, q, m, dr.Slice(k * k), k, 0.0, rest, m);
            }
            else
            {
                rest.Clear();
            }
        }

        store.AddGradient(a, contribution, negate: false);
    }

    // Symmetric eigendecomposition differentiation (Giles 2008); folds off-diagonal gradients into the lower triangle.
    private static void SymmetricEigen(ArrayStore<double> store, in ArrayRecord record)
    {
        ArrayView a = record.Left;
        ref DecompositionInfo info = ref store.Decompositions.Info(record.ScalarInput);
        int valuesSlot = info.Factor, vectorsSlot = info.Second;
        bool valuesTouched = store.Touched(valuesSlot), vectorsTouched = store.Touched(vectorsSlot);
        if (!(valuesTouched || vectorsTouched) || !store.RequiresGradient(a.Slot))
        {
            return;
        }

        int n = a.Rows;
        if (n == 0)
        {
            return;
        }

        ReadOnlySpan<double> lambda = store.Values(valuesSlot).Slice(0, n);
        ReadOnlySpan<double> v = store.Values(vectorsSlot).Slice(0, n * n);
        ReadOnlySpan<double> dl = valuesTouched ? store.OutputGradient(valuesSlot) : default;
        Span<double> t = store.Scratch(ArrayStore<double>.DecompositionSecond, n * n);
        Span<double> contribution = store.Scratch(ArrayStore<double>.MatrixContribution, n * n);
        if (vectorsTouched)
        {
            Span<double> kernel = store.Scratch(ArrayStore<double>.DecompositionFirst, n * n);
            Blas.Gemm(BlasTranspose.Transpose, BlasTranspose.None, n, n, n, 1.0, v, n, store.OutputGradient(vectorsSlot), n, 0.0, kernel, n);
            for (int j = 0; j < n; j++)
            {
                Span<double> column = kernel.Slice(j * n, n);
                double lj = lambda[j];
                for (int i = 0; i < n; i++)
                {
                    double c = column[i], difference = lj - lambda[i];
                    column[i] = i == j ? (valuesTouched ? dl[j] : 0) : c == 0 ? 0 : double.IsFinite(difference) ? c / difference : HalvedQuotient(c, lj, lambda[i]);
                }
            }

            Blas.Gemm(BlasTranspose.None, BlasTranspose.None, n, n, n, 1.0, v, n, kernel, n, 0.0, t, n);
        }
        else
        {
            for (int j = 0; j < n; j++)
            {
                double d = dl[j];
                ReadOnlySpan<double> source = v.Slice(j * n, n);
                Span<double> destination = t.Slice(j * n, n);
                for (int i = 0; i < n; i++)
                {
                    destination[i] = source[i] * d;
                }
            }
        }

        Blas.Gemm(BlasTranspose.None, BlasTranspose.Transpose, n, n, n, 1.0, t, n, v, n, 0.0, contribution, n);
        FoldSymmetric(contribution, n);
        store.AddGradient(a, contribution, negate: false);
    }

    // Thin SVD differentiation (Townsend 2016) with rectangular orthogonal-complement projections folded into N'.
    private static void SingularValue(ArrayStore<double> store, in ArrayRecord record)
    {
        ArrayView a = record.Left;
        var pool = store.Decompositions;
        ref DecompositionInfo info = ref pool.Info(record.ScalarInput);
        int sSlot = info.Factor, uSlot = info.Second, vSlot = info.Third;
        bool sTouched = store.Touched(sSlot), uTouched = store.Touched(uSlot), vTouched = store.Touched(vSlot);
        if (!(sTouched || uTouched || vTouched) || !store.RequiresGradient(a.Slot))
        {
            return;
        }

        var svd = pool.SingularValue(record.ScalarInput);
        int m = svd.U!.Rows, n = svd.V!.Rows, k = svd.Values.Length;
        if (k == 0)
        {
            return;
        }

        ReadOnlySpan<double> s = store.Values(sSlot).Slice(0, k);
        ReadOnlySpan<double> u = store.Values(uSlot).Slice(0, m * k);
        ReadOnlySpan<double> v = store.Values(vSlot).Slice(0, n * k);
        ReadOnlySpan<double> ds = sTouched ? store.OutputGradient(sSlot) : default;
        ReadOnlySpan<double> du = uTouched ? store.OutputGradient(uSlot) : default;
        ReadOnlySpan<double> dv = vTouched ? store.OutputGradient(vSlot) : default;
        bool projectU = uTouched && m > k, projectV = vTouched && n > k;
        Span<double> inner = store.Scratch(ArrayStore<double>.DecompositionFirst, k * k);
        Span<double> w = store.Scratch(ArrayStore<double>.DecompositionSecond, k * k);
        Span<double> side = store.Scratch(ArrayStore<double>.DecompositionThird, Math.Max(m, n) * k);
        Span<double> contribution = store.Scratch(ArrayStore<double>.MatrixContribution, m * n);
        if (uTouched)
        {
            Blas.Gemm(BlasTranspose.Transpose, BlasTranspose.None, k, k, m, 1.0, u, m, du, m, 0.0, inner, k);
        }

        if (vTouched)
        {
            Blas.Gemm(BlasTranspose.Transpose, BlasTranspose.None, k, k, n, 1.0, v, n, dv, n, 0.0, w, k);
        }

        // Overwrite inner in place per (i, j) / (j, i) pair since both entries depend on the same two P and W elements.
        for (int j = 0; j < k; j++)
        {
            double sj = s[j];
            for (int i = 0; i < j; i++)
            {
                double si = s[i];
                double pij = uTouched ? inner[i + j * k] : 0, pji = uTouched ? inner[j + i * k] : 0;
                double wij = vTouched ? w[i + j * k] : 0, wji = vTouched ? w[j + i * k] : 0;
                double d = pij - pji, e = wij - wji;
                double denominator = (sj - si) * (sj + si);
                double upper = d * sj + si * e, lower = d * si + sj * e;
                if (double.IsNormal(denominator) && (d == 0 && e == 0 || double.IsNormal(upper) && double.IsNormal(lower)))
                {
                    upper = upper == 0 ? 0 : upper / denominator;
                    lower = lower == 0 ? 0 : lower / denominator;
                }
                else
                {
                    ScaledSingularValueQuotients(pij, pji, wij, wji, si, sj, out upper, out lower);
                }

                if (projectU)
                {
                    upper -= pij / sj;
                    lower -= pji / si;
                }

                if (projectV)
                {
                    upper -= wji / si;
                    lower -= wij / sj;
                }

                inner[i + j * k] = upper;
                inner[j + i * k] = lower;
            }

            double diagonal = sTouched ? ds[j] : 0;
            if (projectU)
            {
                diagonal -= inner[j + j * k] / sj;
            }

            if (projectV)
            {
                diagonal -= w[j + j * k] / sj;
            }

            inner[j + j * k] = diagonal;
        }

        Span<double> product = side.Slice(0, m * k);
        Blas.Gemm(BlasTranspose.None, BlasTranspose.None, m, k, k, 1.0, u, m, inner, k, 0.0, product, m);
        if (projectU)
        {
            for (int j = 0; j < k; j++)
            {
                double sj = s[j];
                ReadOnlySpan<double> source = du.Slice(j * m, m);
                Span<double> destination = product.Slice(j * m, m);
                for (int i = 0; i < m; i++)
                {
                    destination[i] += source[i] / sj;
                }
            }
        }

        Blas.Gemm(BlasTranspose.None, BlasTranspose.Transpose, m, n, k, 1.0, product, m, v, n, 0.0, contribution, m);
        if (projectV)
        {
            Span<double> scaled = side.Slice(0, n * k);
            for (int j = 0; j < k; j++)
            {
                double sj = s[j];
                ReadOnlySpan<double> source = dv.Slice(j * n, n);
                Span<double> destination = scaled.Slice(j * n, n);
                for (int i = 0; i < n; i++)
                {
                    destination[i] = source[i] / sj;
                }
            }

            Blas.Gemm(BlasTranspose.None, BlasTranspose.Transpose, m, n, k, 1.0, u, m, scaled, n, 1.0, contribution, m);
        }

        store.AddGradient(a, contribution, negate: false);
    }

    // Halves finite operands whose difference overflows without losing precision; non-finite eigenvalues divide directly.
    private static double HalvedQuotient(double c, double lj, double li) =>
        double.IsFinite(lj) && double.IsFinite(li) ? 0.5 * c / (0.5 * lj - 0.5 * li) : c / (lj - li);

    // Evaluates (d * sj + si * e, d * si + sj * e) / ((sj - si) * (sj + si)) via binary exponent separation
    // so intermediate underflow/overflow cannot corrupt finite quotients.
    private static void ScaledSingularValueQuotients(double pij, double pji, double wij, double wji, double si, double sj, out double upper, out double lower)
    {
        double d = pij - pji, e = wij - wji;
        if (!(double.IsFinite(pij) && double.IsFinite(pji) && double.IsFinite(wij) && double.IsFinite(wji) && double.IsFinite(si) && double.IsFinite(sj)))
        {
            double denominator = (sj - si) * (sj + si);
            upper = d * sj + si * e;
            lower = d * si + sj * e;
            upper = upper == 0 ? 0 : upper / denominator;
            lower = lower == 0 ? 0 : lower / denominator;
            return;
        }

        int dShift = 0, eShift = 0;
        if (!double.IsFinite(d))
        {
            d = 0.5 * pij - 0.5 * pji;
            dShift = 1;
        }

        if (!double.IsFinite(e))
        {
            e = 0.5 * wij - 0.5 * wji;
            eShift = 1;
        }

        double largest = Math.Max(Math.Abs(si), Math.Abs(sj));
        int scale = largest == 0 ? 0 : Math.ILogB(largest);
        double ti = Math.ScaleB(si, -scale), tj = Math.ScaleB(sj, -scale);
        double scaledDenominator = (tj - ti) * (tj + ti);
        upper = ScaledQuotient(d, dShift, sj, si, e, eShift, scaledDenominator, 2 * scale);
        lower = ScaledQuotient(d, dShift, si, sj, e, eShift, scaledDenominator, 2 * scale);
    }

    private static double ScaledQuotient(double d, int dShift, double x, double y, double e, int eShift, double denominator, int denominatorShift)
    {
        double first = SplitProduct(d, x, out int firstExponent), second = SplitProduct(y, e, out int secondExponent);
        firstExponent += dShift;
        secondExponent += eShift;
        int exponent = Math.Max(firstExponent, secondExponent);
        double numerator = Math.ScaleB(first, firstExponent - exponent) + Math.ScaleB(second, secondExponent - exponent);
        return numerator == 0 ? 0 : Math.ScaleB(numerator / denominator, exponent - denominatorShift);
    }

    private static double SplitProduct(double a, double b, out int exponent)
    {
        if (a == 0 || b == 0)
        {
            // Keep far below any valid exponent sum so the non-zero term determines the alignment exponent.
            exponent = int.MinValue / 4;
            return 0;
        }

        int ea = Math.ILogB(a), eb = Math.ILogB(b);
        exponent = ea + eb;
        return Math.ScaleB(a, -ea) * Math.ScaleB(b, -eb);
    }

    private static void Gemm(
        bool transA, bool transB, int m, int n, int k,
        ReadOnlySpan<double> a, int aOffset, int lda, ReadOnlySpan<double> b, int bOffset, int ldb, Span<double> c, int cOffset, int ldc)
    {
        Blas.Gemm(transA ? BlasTranspose.Transpose : BlasTranspose.None, transB ? BlasTranspose.Transpose : BlasTranspose.None, m, n, k,
            -1.0, a.Slice(aOffset), lda, b.Slice(bOffset), ldb, 1.0, c.Slice(cOffset), ldc);
    }

    private enum Triangle
    {
        Full,
        Lower,
        Upper,
        Symmetric,
    }

    private static void Product(int n, int k, ReadOnlySpan<double> u, ReadOnlySpan<double> x, Span<double> contribution, Triangle triangle)
    {
        if (n == 0)
        {
            return;
        }

        if (k == 1 && triangle != Triangle.Full)
        {
            OuterProduct(n, u, x, contribution, triangle);
            return;
        }

        Blas.Gemm(BlasTranspose.None, BlasTranspose.Transpose, n, n, k, 1.0, u, n, x, n, 0.0, contribution, n);
        switch (triangle)
        {
            case Triangle.Lower:
                for (int j = 1; j < n; j++)
                {
                    contribution.Slice(j * n, j).Clear();
                }

                break;
            case Triangle.Upper:
                for (int j = 0; j < n - 1; j++)
                {
                    contribution.Slice(j * n + j + 1, n - j - 1).Clear();
                }

                break;
            case Triangle.Symmetric:
                FoldSymmetric(contribution, n);
                break;
        }
    }

    private static void FoldSymmetric(Span<double> contribution, int n)
    {
        for (int j = 0; j < n; j++)
        {
            for (int i = j + 1; i < n; i++)
            {
                contribution[i + j * n] += contribution[j + i * n];
            }

            contribution.Slice(j * n, j).Clear();
        }
    }

    private static void Transpose(int rows, int columns, ReadOnlySpan<double> source, Span<double> destination)
    {
        for (int j = 0; j < columns; j++)
        {
            ReadOnlySpan<double> column = source.Slice(j * rows, rows);
            for (int i = 0; i < rows; i++)
            {
                destination[j + i * columns] = column[i];
            }
        }
    }

    private static void OuterProduct(int n, ReadOnlySpan<double> u, ReadOnlySpan<double> x, Span<double> contribution, Triangle triangle)
    {
        for (int j = 0; j < n; j++)
        {
            Span<double> column = contribution.Slice(j * n, n);
            double xj = x[j], uj = u[j];
            switch (triangle)
            {
                case Triangle.Lower:
                    column.Slice(0, j).Clear();
                    for (int i = j; i < n; i++)
                    {
                        column[i] = u[i] * xj;
                    }

                    break;
                case Triangle.Upper:
                    for (int i = 0; i <= j; i++)
                    {
                        column[i] = u[i] * xj;
                    }

                    column.Slice(j + 1).Clear();
                    break;
                default:
                    column.Slice(0, j).Clear();
                    column[j] = uj * xj;
                    for (int i = j + 1; i < n; i++)
                    {
                        column[i] = u[i] * xj + x[i] * uj;
                    }

                    break;
            }
        }
    }

    private static void TransposeInPlace(Span<double> matrix, int n)
    {
        for (int j = 0; j < n; j++)
        {
            for (int i = j + 1; i < n; i++)
            {
                (matrix[i + j * n], matrix[j + i * n]) = (matrix[j + i * n], matrix[i + j * n]);
            }
        }
    }

    private static void Scale(Span<double> values, double factor)
    {
        for (int i = 0; i < values.Length; i++)
        {
            values[i] *= factor;
        }
    }
}
