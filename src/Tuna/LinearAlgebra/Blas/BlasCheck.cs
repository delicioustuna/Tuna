using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace Tuna.LinearAlgebra;

// Blas の引数検証と補助ユーティリティ。
internal static class BlasCheck
{
    internal static void SameLength(int x, int y)
    {
        if (x != y)
        {
            throw new ArgumentException($"Vector lengths differ ({x} and {y}).");
        }
    }

    // m × n 行列（leading dimension: lda）が長さ length のスパンに収まるか検証する。
    internal static void Matrix(int m, int n, int length, int lda, string name = "a", string ldName = "lda")
    {
        ArgumentOutOfRangeException.ThrowIfNegative(m);
        ArgumentOutOfRangeException.ThrowIfNegative(n);
        if (lda < Math.Max(1, m))
        {
            throw new ArgumentOutOfRangeException(ldName, lda, $"{ldName} must be at least max(1, rows) = {Math.Max(1, m)}.");
        }

        if (m > 0 && n > 0 && length < ((long)(n - 1) * lda) + m)
        {
            throw new ArgumentException($"The matrix span is too short: {length} < {((long)(n - 1) * lda) + m}.", name);
        }
    }

    // 行列積 C <- α*op(A)*op(B) + β*C の引数および出力 C と入力 A/B の非重複を検証する（A と B の重複は許容）。
    internal static void MatrixProduct<T>(
        bool transA, bool transB, int m, int n, int k, ReadOnlySpan<T> a, int lda, ReadOnlySpan<T> b, int ldb, ReadOnlySpan<T> c, int ldc)
    {
        // 行列ごとの検査は行数・列数を m・n の名前で報告するので、負の大きさは先にここで引数名どおりに報告する
        ArgumentOutOfRangeException.ThrowIfNegative(m);
        ArgumentOutOfRangeException.ThrowIfNegative(n);
        ArgumentOutOfRangeException.ThrowIfNegative(k);
        var (rowsA, columnsA) = transA ? (k, m) : (m, k);
        var (rowsB, columnsB) = transB ? (n, k) : (k, n);
        Matrix(rowsA, columnsA, a.Length, lda, nameof(a), nameof(lda));
        Matrix(rowsB, columnsB, b.Length, ldb, nameof(b), nameof(ldb));
        Matrix(m, n, c.Length, ldc, nameof(c), nameof(ldc));
        Disjoint(a, rowsA, columnsA, lda, c, m, n, ldc);
        Disjoint(b, rowsB, columnsB, ldb, c, m, n, ldc);
    }

    // 要素単位で重複を検査する（同一配列の異なる行など、範囲が重なっても要素を共有しない部分行列は許容）。
    private static void Disjoint<T>(ReadOnlySpan<T> x, int rows, int columns, int ld, ReadOnlySpan<T> c, int m, int n, int ldc)
    {
        if (StridedLayout.Overlaps(
            in MemoryMarshal.GetReference(x), rows, columns, StridedLayout.Normalize(rows, 1), StridedLayout.Normalize(columns, ld),
            in MemoryMarshal.GetReference(c), m, n, StridedLayout.Normalize(m, 1), StridedLayout.Normalize(n, ldc), out _))
        {
            throw new ArgumentException("The output matrix must not overlap the input matrices.", nameof(c));
        }
    }

    internal static void Vector(int length, int required, string name)
    {
        if (length < required)
        {
            throw new ArgumentException($"The vector is too short: {length} < {required}.", name);
        }
    }

    internal static void Swap<T>(Span<T> x, Span<T> y)
    {
        // スタック上の 4KB の一時領域を使って区切りながら入れ替える
        Span<byte> buffer = stackalloc byte[4096];
        int chunk = Math.Max(1, buffer.Length / Unsafe.SizeOf<T>());
        if (RuntimeHelpers.IsReferenceOrContainsReferences<T>())
        {
            for (int i = 0; i < x.Length; i++)
            {
                (x[i], y[i]) = (y[i], x[i]);
            }

            return;
        }

        // 処理した長さだけ進める（i += chunk は、長さが int の上限に近いと最後の区切りの後に負へ折り返す）
        for (int i = 0, len; i < x.Length; i += len)
        {
            len = Math.Min(chunk, x.Length - i);
            var t = MemoryMarshal.CreateSpan(ref Unsafe.As<byte, T>(ref MemoryMarshal.GetReference(buffer)), len);
            x.Slice(i, len).CopyTo(t);
            y.Slice(i, len).CopyTo(x.Slice(i, len));
            t.CopyTo(y.Slice(i, len));
        }
    }

    internal static ReadOnlySpan<double> AsDoubles(ReadOnlySpan<Complex> x) =>
        MemoryMarshal.CreateReadOnlySpan(ref Unsafe.As<Complex, double>(ref MemoryMarshal.GetReference(x)), 2 * x.Length);

    internal static Span<double> AsDoubles(Span<Complex> x) =>
        MemoryMarshal.CreateSpan(ref Unsafe.As<Complex, double>(ref MemoryMarshal.GetReference(x)), 2 * x.Length);
}

// 単精度ベクトルのノルム（倍精度で平方和を累積するため float 範囲ではオーバーフロー・アンダーフローしない）。
internal static class BlasNorms
{
    internal static float Norm2(ReadOnlySpan<float> x) => (float)Math.Sqrt(SumOfSquares(x));

    // 倍精度で累積した Σ x[i]^2。
    internal static double SumOfSquares(ReadOnlySpan<float> x)
    {
        int n = x.Length;
        ref float px = ref MemoryMarshal.GetReference(x);
        int i = 0;
        double sum = 0;
        if (Vector512.IsHardwareAccelerated && n >= 16)
        {
            var acc = Vector512<double>.Zero;
            for (; i <= n - 16; i += 16)
            {
                var (lo, hi) = Vector512.Widen(Vector512.LoadUnsafe(ref px, (nuint)i));
                acc = Vector512.FusedMultiplyAdd(lo, lo, acc);
                acc = Vector512.FusedMultiplyAdd(hi, hi, acc);
            }

            sum = Vector512.Sum(acc);
        }
        else if (Vector256.IsHardwareAccelerated && n >= 8)
        {
            var acc = Vector256<double>.Zero;
            for (; i <= n - 8; i += 8)
            {
                var (lo, hi) = Vector256.Widen(Vector256.LoadUnsafe(ref px, (nuint)i));
                acc = RealVectorOps.MulAdd(lo, lo, acc);
                acc = RealVectorOps.MulAdd(hi, hi, acc);
            }

            sum = Vector256.Sum(acc);
        }
        else if (Vector128.IsHardwareAccelerated && n >= 4)
        {
            var acc = Vector128<double>.Zero;
            for (; i <= n - 4; i += 4)
            {
                var (lo, hi) = Vector128.Widen(Vector128.LoadUnsafe(ref px, (nuint)i));
                acc = RealVectorOps.MulAdd(lo, lo, acc);
                acc = RealVectorOps.MulAdd(hi, hi, acc);
            }

            sum = Vector128.Sum(acc);
        }

        for (; i < n; i++)
        {
            double v = Unsafe.Add(ref px, i);
            sum += v * v;
        }

        return sum;
    }
}
