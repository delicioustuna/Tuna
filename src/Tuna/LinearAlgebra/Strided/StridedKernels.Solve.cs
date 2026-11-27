using System.Numerics;
using System.Runtime.InteropServices;

namespace Tuna.LinearAlgebra;

// 連立一次方程式（三角行列・LU 分解）をストライド付きビューで解く。
// 係数と解は BLAS に渡せる配置（係数は列優先かその転置、解は列優先）ならそのまま、そうでなければ連続領域へ写して計算する。
internal static partial class StridedKernels
{
    // 三角連立方程式 AX = B を解く（lower で三角側を選択。vector が true なら TRSV 経路、false なら TRSM 経路）。
    internal static void SolveTriangular<T>(ReadOnlyMatrixSpan<T> a, ReadOnlyMatrixSpan<T> b, MatrixSpan<T> x, bool lower, bool vector)
        where T : unmanaged, INumberBase<T>
    {
        RequireSystemShape(a, b, x);
        RequireDisjoint(a, x);
        CopyRightHandSide(b, x);
        SolveTriangularInPlace(a, x, lower, vector);
    }

    // AX = B をインプレースで解く（入力 x は右辺 B）。
    internal static void SolveTriangularInPlace<T>(ReadOnlyMatrixSpan<T> a, MatrixSpan<T> x, bool lower, bool vector)
        where T : unmanaged, INumberBase<T>
    {
        int n = a.Rows;
        int nrhs = x.Columns;
        if (n == 0 || nrhs == 0)
        {
            return;
        }

        if (!TryGetBlasLayout(a, out bool transpose, out int lda))
        {
            a = a.ToMatrix();
            transpose = false;
            lda = n;
        }

        if (!TryGetBlasLayout(x.AsReadOnly(), out bool transposeX, out int ldx) || transposeX)
        {
            var temp = x.ToMatrix();
            SolveTriangularInPlace(a, temp, lower, vector);
            temp.AsMatrixSpan().CopyTo(x);
            return;
        }

        var aData = a.Data;
        var xData = x.Data;
        if (typeof(T) == typeof(double) && !vector)
        {
            TriangularKernels.SolveLeft(lower, transpose, unitDiagonal: false, n, nrhs,
                MemoryMarshal.Cast<T, double>(aData), lda, MemoryMarshal.Cast<T, double>(xData), ldx);
            return;
        }

        // 転置ビューは、反対側の三角部分に格納された行列の転置として渡す
        var triangle = lower != transpose ? BlasTriangle.Lower : BlasTriangle.Upper;
        var operation = transpose ? BlasTranspose.Transpose : BlasTranspose.None;
        for (int j = 0; j < nrhs; j++)
        {
            ParallelLevel2<T>.Trsv(triangle, operation, BlasDiagonal.NonUnit, n, aData, lda, xData.Slice(j * ldx, n));
        }
    }

    // 部分ピボット付き LU 分解で AX = B を解く（特異行列時は ArgumentException）。
    internal static void Solve(ReadOnlyMatrixSpan<double> a, ReadOnlyMatrixSpan<double> b, MatrixSpan<double> x, bool vector)
    {
        RequireSystemShape(a, b, x);
        RequireDisjoint(a, x);
        int n = a.Rows;
        var lu = GC.AllocateUninitializedArray<double>(checked(n * n));
        var pivots = new int[n];
        CopyDense(a, lu);
        if (LuKernels.Factor(n, n, lu, n, pivots) != 0)
        {
            throw new ArgumentException("The matrix is singular.", nameof(a));
        }

        CopyRightHandSide(b, x);
        SolveFactoredInPlace(n, lu, pivots, x, vector);
    }

    // 部分ピボット付き LU 分解で逆行列 A^-1 を求める（特異時は x を変更せず ArgumentException）。
    internal static void Inverse(ReadOnlyMatrixSpan<double> a, MatrixSpan<double> x)
    {
        RequireSystemShape(a, a, x);
        RequireDisjoint(a, x);
        int n = a.Rows;
        var lu = GC.AllocateUninitializedArray<double>(checked(n * n));
        var pivots = new int[n];
        CopyDense(a, lu);
        if (LuKernels.Factor(n, n, lu, n, pivots) != 0)
        {
            throw new ArgumentException("The matrix is singular.", nameof(a));
        }

        if (TryGetBlasLayout(x.AsReadOnly(), out bool transposeX, out int ldx) && !transposeX)
        {
            var data = x.Data;
            for (int j = 0; j < n; j++)
            {
                data.Slice(j * ldx, n).Clear();
                data[j + j * ldx] = 1;
            }
        }
        else
        {
            x.Fill(0);
            x.Diagonal().Fill(1);
        }

        SolveFactoredInPlace(n, lu, pivots, x, vector: false);
    }

    internal static double CholeskyLogDeterminant(ReadOnlyMatrixSpan<double> a)
    {
        int n = a.Rows;
        if (n != a.Columns)
        {
            throw new ArgumentException("The matrix must be square.", nameof(a));
        }

        const int stackLimit = 256;
        int length = checked(n * n);
        double[]? rented = null;
        Span<double> l = length <= stackLimit
            ? stackalloc double[length]
            : (rented = WorkArrayPool<double>.Rent(length)).AsSpan(0, length);
        try
        {
            CopyDense(a, l);
            if (CholeskyKernels.Factor(n, l, n) != 0)
            {
                throw new ArgumentException("The matrix is not positive definite.", nameof(a));
            }

            return CholeskyKernels.LogDeterminant(n, l);
        }
        finally
        {
            if (rented is not null)
            {
                WorkArrayPool<double>.Return(rented);
            }
        }
    }

    // 分解済み LU で AX = B（transpose 時は A^T X = B）をインプレースで解く。
    internal static void SolveFactoredInPlace(
        int n, ReadOnlySpan<double> lu, ReadOnlySpan<int> pivots, MatrixSpan<double> x, bool vector, bool transpose = false)
    {
        if (!TryGetBlasLayout(x.AsReadOnly(), out bool transposeX, out int ldx) || transposeX)
        {
            var temp = x.ToMatrix();
            LuKernels.Solve(n, lu, n, pivots, temp.Columns, temp.Data, n, vector, transpose);
            temp.AsMatrixSpan().CopyTo(x);
            return;
        }

        LuKernels.Solve(n, lu, n, pivots, x.Columns, x.Data, ldx, vector, transpose);
    }

    // 再分解する行列 a を分解用ストレージ（列優先連続）へコピーする（完全一致なら無処理、部分重複や転置重複は例外）。
    internal static void CopyIntoFactorStorage(ReadOnlyMatrixSpan<double> a, Span<double> storage)
    {
        var destination = new MatrixSpan<double>(storage, a.Rows, a.Columns);
        if (Overlaps(a, destination, out bool exact))
        {
            if (!exact)
            {
                throw new ArgumentException(
                    "The matrix partially overlaps the storage of the decomposition. Only the storage itself is allowed.", nameof(a));
            }

            return;
        }

        // 列優先で連続な入力は、要素ごとの写しを通さずにまとめて写す（Compute が配列を写すのと同じ速さにする）
        if (a.TryGetDense(out var dense))
        {
            dense.CopyTo(storage);
            return;
        }

        Copy(a, destination);
    }

    // コレスキー因子 l から A^-1 を計算する（列優先連続または転置は対称性により直接書き込み、それ以外は作業領域経由）。
    internal static void InverseCholesky(int n, ReadOnlySpan<double> l, MatrixSpan<double> destination)
    {
        if (n == 0)
        {
            return;
        }

        if (!TryGetBlasLayout(destination.AsReadOnly(), out _, out int ld))
        {
            var temp = Matrix<double>.Uninitialized(n, n);
            CholeskyKernels.Inverse(n, l, n, temp.Data, n);
            temp.AsMatrixSpan().CopyTo(destination);
            return;
        }

        CholeskyKernels.Inverse(n, l, n, destination.Data, ld);
    }

    // コレスキー因子 L で AX = B をインプレースで解く。
    internal static void SolveCholeskyInPlace(int n, ReadOnlySpan<double> l, MatrixSpan<double> x, bool vector)
    {
        if (n == 0 || x.Columns == 0)
        {
            return;
        }

        if (!TryGetBlasLayout(x.AsReadOnly(), out bool transposeX, out int ldx) || transposeX)
        {
            var temp = x.ToMatrix();
            CholeskyKernels.Solve(n, l, n, temp.Columns, temp.Data, n, vector);
            temp.AsMatrixSpan().CopyTo(x);
            return;
        }

        CholeskyKernels.Solve(n, l, n, x.Columns, x.Data, ldx, vector);
    }

    // 反復改良の引数（形状および解と係数・右辺・因子の非重複）を検証する。
    internal static void RequireRefinementArguments(
        int n, ReadOnlyMatrixSpan<double> a, ReadOnlyMatrixSpan<double> b, MatrixSpan<double> x, int maxIterations, ReadOnlyMatrixSpan<double> factor)
    {
        if (a.Rows != n || a.Columns != n)
        {
            throw new ArgumentException("The matrix must have the order of the decomposition.", nameof(a));
        }

        if (b.Rows != n)
        {
            throw new ArgumentException("The right-hand side must have as many rows as the order of the matrix.", nameof(b));
        }

        if (x.Rows != n || x.Columns != b.Columns)
        {
            throw new ArgumentException("The solution must have the shape of the right-hand side.", nameof(x));
        }

        if (Overlaps(a, x, out _) || Overlaps(b, x, out _) || Overlaps(factor, x, out _))
        {
            throw new ArgumentException("The solution must not overlap the matrix, the right-hand side or the factors.", nameof(x));
        }

        ArgumentOutOfRangeException.ThrowIfNegative(maxIterations);
    }

    // 右辺の列ごとに反復改良を実行して x を上書きし、各列の後退誤差の最大値を返す。
    internal static double Refine(
        ReadOnlyMatrixSpan<double> a, bool symmetricLower, ReadOnlyMatrixSpan<double> b, MatrixSpan<double> x, int maxIterations, ConditionKernels.Apply solve)
    {
        int n = a.Rows;
        if (n == 0 || x.Columns == 0)
        {
            return 0;
        }

        // 係数は列優先の配置ならそのまま読み、そうでなければ写す
        ReadOnlySpan<double> coefficients;
        if (TryGetBlasLayout(a, out bool transpose, out int lda) && !transpose)
        {
            coefficients = a.Data;
        }
        else
        {
            var copy = GC.AllocateUninitializedArray<double>(checked(n * n));
            CopyDense(a, copy);
            coefficients = copy;
            lda = n;
        }

        var rightHandSide = new double[n];
        var solution = new double[n];
        double error = 0;
        for (int c = 0; c < x.Columns; c++)
        {
            for (int i = 0; i < n; i++)
            {
                rightHandSide[i] = b[i, c];
                solution[i] = x[i, c];
            }

            double columnError = ConditionKernels.Refine(n, coefficients, lda, symmetricLower, rightHandSide, solution, maxIterations, solve);
            for (int i = 0; i < n; i++)
            {
                x[i, c] = solution[i];
            }

            if (!(columnError <= error))
            {
                error = columnError;
            }
        }

        return error;
    }

    internal static void CopyDense(ReadOnlyMatrixSpan<double> a, Span<double> destination) =>
        Copy(a, new MatrixSpan<double>(destination, a.Rows, a.Columns));

    private static void RequireSystemShape<T>(ReadOnlyMatrixSpan<T> a, ReadOnlyMatrixSpan<T> b, MatrixSpan<T> x)
        where T : unmanaged, INumberBase<T>
    {
        if (a.Rows != a.Columns)
        {
            throw new ArgumentException("The coefficient matrix must be square.", nameof(a));
        }

        if (b.Rows != a.Rows)
        {
            throw new ArgumentException("The right-hand side must have as many rows as the coefficient matrix.", nameof(b));
        }

        if (x.Rows != b.Rows || x.Columns != b.Columns)
        {
            throw new ArgumentException("Destination shape must agree.", "destination");
        }
    }

    // 右辺を出力先へコピーする（完全一致なら無処理、部分重複は例外）。
    internal static void CopyRightHandSide<T>(ReadOnlyMatrixSpan<T> b, MatrixSpan<T> x)
        where T : unmanaged, INumberBase<T>
    {
        if (Overlaps(b, x, out bool exact))
        {
            if (!exact)
            {
                throw new ArgumentException("The destination partially overlaps the right-hand side. Only an exact alias is allowed.", "destination");
            }

            return;
        }

        Copy(b, x);
    }
}
