using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace Tuna.LinearAlgebra;

// BLAS Level 2（列優先行列 A[i, j] = a[i + j*lda]）のジェネリック実装。
// 4 列融合カーネルおよび固定分割並列化により、並列数によらずビット一致する結果を保つ。
internal static unsafe class ParallelLevel2<T>
    where T : unmanaged, INumberBase<T>
{
    private const double Grain = 16 * 1024; // 並列化の最小仕事量（8 バイト換算要素数）
    private const int MaxPartialCount = 16; // 部分和数の上限（スレッド数非依存な分割のため固定）

    private static int PartialCount(int m, int n) =>
        (int)Math.Clamp((double)m * n * Unsafe.SizeOf<T>() / (8.0 * Grain), 1, MaxPartialCount);

    private static int Threads(int m, int n, double grain = Grain) =>
        Parallelism.PlannedDegree((double)m * n * Unsafe.SizeOf<T>() / 8.0, grain);

    private static bool IsComplex => typeof(T) == typeof(Complex);

    // count 要素を align 要素境界で parts 分割した index 番目の範囲。
    private static (int Lo, int Hi) Split(int count, int parts, int index, int align)
    {
        int lo = (int)((long)count * index / parts);
        int hi = (int)((long)count * (index + 1) / parts);
        if (index > 0)
        {
            lo = (int)Math.Min(count, ((long)lo + align - 1) / align * align);
        }

        if (index < parts - 1)
        {
            hi = (int)Math.Min(count, ((long)hi + align - 1) / align * align);
        }

        return (lo, hi);
    }

    // 64 行境界で分割し、端数は末尾ブロックに含めて SIMD 経路を維持する。
    private static (int Lo, int Hi) SplitRows(int count, int parts, int index)
    {
        int blocks = count / 64;
        int lo = (int)((long)blocks * index / parts) * 64;
        int hi = index == parts - 1 ? count : (int)((long)blocks * (index + 1) / parts) * 64;
        return (lo, hi);
    }

    // y <- β*y（β == 0 なら y を読まずゼロクリア）。
    private static void ScaleOrClear(T beta, Span<T> y)
    {
        if (beta == T.Zero)
        {
            y.Clear();
        }
        else if (beta != T.One)
        {
            VectorOps.Scale(beta, y);
        }
    }

    // A <- β*A（β == 0 はゼロクリア、β == 1 は無処理。パディング領域は触らず、大規模複素行列は列単位で SIMD 端数を揃える）。
    internal static void ScaleOrClear(T beta, int m, int n, Span<T> a, int lda)
    {
        if (m == 0 || n == 0 || beta == T.One)
        {
            return;
        }

        int threads = Math.Min(Threads(m, n), n);
        if (threads > 1)
        {
            fixed (T* pa = a)
            {
                ScaleOrClearColumnsParallel(m, n, threads, beta, pa, lda);
            }
        }
        else if (lda == m && (!IsComplex || n == 1 || (double)m * n * Unsafe.SizeOf<T>() < 16 * Grain))
        {
            ScaleOrClear(beta, a[..(m * n)]);
        }
        else
        {
            for (int j = 0; j < n; j++)
            {
                ScaleOrClear(beta, a.Slice(j * lda, m));
            }
        }
    }

    private static void ScaleOrClearColumnsParallel(int m, int n, int threads, T beta, T* a, int lda)
    {
        var work = new ScaleColumnsWork(m, n, threads, beta, a, lda);
        Parallelism.For(threads, threads, ref work);
    }

    private readonly struct ScaleColumnsWork(int m, int n, int threads, T beta, T* a, int lda) : IParallelWork
    {
        public void Invoke(int t)
        {
            var (lo, hi) = Split(n, threads, t, 1);
            for (int j = lo; j < hi; j++)
            {
                ScaleOrClear(beta, new Span<T>(a + ((nint)j * lda), m));
            }
        }
    }

    // y <- α*op(A)*x + β*y（A: m×n）。
    internal static void Gemv(BlasTranspose trans, int m, int n, T alpha, ReadOnlySpan<T> a, int lda, ReadOnlySpan<T> x, T beta, Span<T> y)
    {
        bool notrans = trans == BlasTranspose.None;
        int leny = notrans ? m : n;
        ScaleOrClear(beta, y[..leny]);
        if (m == 0 || n == 0 || alpha == T.Zero)
        {
            return;
        }

        bool conj = trans == BlasTranspose.ConjugateTranspose;
        int threads = Threads(m, n);
        fixed (T* pa = a)
        fixed (T* px = x)
        fixed (T* py = y)
        {
            if (notrans)
            {
                int parts = 1;
                if ((m < 2048 || (double)m * Unsafe.SizeOf<T>() < RowSplitMinBytes) && (double)m * Unsafe.SizeOf<T>() <= ColumnSplitBytes)
                {
                    int candidateParts = Math.Min(PartialCount(m, n), ((n - 1) / 4) + 1);
                    if (n / candidateParts >= MinColumnsPerPartial)
                    {
                        parts = candidateParts;
                    }
                }

                if (parts > 1)
                {
                    MultiplyColumnSplit(m, n, parts, alpha, pa, lda, px, py);
                }
                else if (threads > 1)
                {
                    MultiplyRowsParallel(m, n, threads, alpha, pa, lda, px, py);
                }
                else
                {
                    MultiplyRows(0, m, n, alpha, pa, lda, px, py);
                }
            }
            else if (threads > 1)
            {
                MultiplyColumnsTransposedParallel(m, n, threads, alpha, pa, lda, px, py, conj);
            }
            else
            {
                MultiplyColumnsTransposed(0, n, m, alpha, pa, lda, px, py, conj);
            }
        }
    }

    private static void MultiplyRowsParallel(int m, int n, int threads, T alpha, T* a, int lda, T* x, T* y)
    {
        var work = new MultiplyRowsWork(m, n, threads, alpha, a, lda, x, y);
        Parallelism.For(threads, threads, ref work);
    }

    private readonly struct MultiplyRowsWork(int m, int n, int threads, T alpha, T* a, int lda, T* x, T* y) : IParallelWork
    {
        public void Invoke(int t)
        {
            var (lo, hi) = SplitRows(m, threads, t);
            MultiplyRows(lo, hi, n, alpha, a, lda, x, y);
        }
    }

    private static void MultiplyColumnsTransposedParallel(int m, int n, int threads, T alpha, T* a, int lda, T* x, T* y, bool conj)
    {
        var work = new MultiplyColumnsWork(m, n, threads, alpha, a, lda, x, y, conj);
        Parallelism.For(threads, threads, ref work);
    }

    private readonly struct MultiplyColumnsWork(int m, int n, int threads, T alpha, T* a, int lda, T* x, T* y, bool conj) : IParallelWork
    {
        public void Invoke(int t)
        {
            var (lo, hi) = Split(n, threads, t, 4);
            MultiplyColumnsTransposed(lo, hi, m, alpha, a, lda, x, y, conj);
        }
    }

    private const double RowSplitMinBytes = 16 * 1024;   // 2048 行以上かつ列がこのバイト数以上なら行分割を優先
    private const int MinColumnsPerPartial = 32;         // 部分和 1 本あたりの最小列数目安
    private const double ColumnSplitBytes = 256 * 1024;  // 列分割時の作業ベクトル（キャッシュ収容）上限

    // 列分割版 y += α*A*x（分割 0 は y に直接加算し、他は作業ベクトルへ累積後に番号順で加算）。
    private static void MultiplyColumnSplit(int m, int n, int parts, T alpha, T* a, int lda, T* x, T* y)
    {
        T[] partial = WorkArrayPool<T>.Rent(checked((parts - 1) * m));
        try
        {
            fixed (T* pp = partial)
            {
                var work = new MultiplyColumnSplitWork(m, n, parts, alpha, a, lda, x, y, pp);
                Parallelism.For(parts, parts, ref work);

                var dest = new Span<T>(y, m);
                for (int t = 1; t < parts; t++)
                {
                    VectorOps.Axpy(T.One, new ReadOnlySpan<T>(pp + ((nint)(t - 1) * m), m), dest);
                }
            }
        }
        finally
        {
            WorkArrayPool<T>.Return(partial);
        }
    }

    private readonly struct MultiplyColumnSplitWork(int m, int n, int parts, T alpha, T* a, int lda, T* x, T* y, T* partial) : IParallelWork
    {
        public void Invoke(int t)
        {
            var (lo, hi) = Split(n, parts, t, 4);
            T* w = y;
            if (t > 0)
            {
                w = partial + ((nint)(t - 1) * m);
                new Span<T>(w, m).Clear();
            }

            MultiplyRows(0, m, hi - lo, alpha, a + ((nint)lo * lda), lda, x + lo, w);
        }
    }

    private static void MultiplyRows(int r0, int r1, int n, T alpha, T* a, int lda, T* x, T* y)
    {
        int rows = r1 - r0;
        if (rows <= 0)
        {
            return;
        }

        nint ld = lda;
        int j = 0;
        if (!IsComplex)
        {
            for (; j <= n - 4; j += 4)
            {
                RealVectorOps.AddColumns4(
                    rows, a + r0 + (j * ld), ld, alpha * x[j], alpha * x[j + 1], alpha * x[j + 2], alpha * x[j + 3], y + r0);
            }
        }
        else
        {
            Span<Complex> t4 = stackalloc Complex[4];
            for (; j <= n - 4; j += 4)
            {
                for (int k = 0; k < 4; k++)
                {
                    T tk = alpha * x[j + k];
                    t4[k] = Unsafe.As<T, Complex>(ref tk);
                }

                ComplexVectorOps.AddColumns4(rows, (Complex*)(a + r0 + (j * ld)), ld, t4, (Complex*)(y + r0));
            }
        }

        for (; j < n; j++)
        {
            T t = alpha * x[j];
            if (t != T.Zero)
            {
                VectorOps.Axpy(t, new ReadOnlySpan<T>(a + r0 + (j * ld), rows), new Span<T>(y + r0, rows));
            }
        }
    }

    private static void MultiplyColumnsTransposed(int c0, int c1, int m, T alpha, T* a, int lda, T* x, T* y, bool conj)
    {
        nint ld = lda;
        var xs = new ReadOnlySpan<T>(x, m);
        int j = c0;
        if (!IsComplex)
        {
            for (; j <= c1 - 4; j += 4)
            {
                RealVectorOps.Dot4(m, a + (j * ld), ld, x, out T r0, out T r1, out T r2, out T r3);
                y[j] += alpha * r0;
                y[j + 1] += alpha * r1;
                y[j + 2] += alpha * r2;
                y[j + 3] += alpha * r3;
            }
        }

        else
        {
            Span<Complex> r = stackalloc Complex[4];
            for (; j <= c1 - 4; j += 4)
            {
                ComplexVectorOps.Dot4(m, (Complex*)(a + (j * ld)), ld, (Complex*)x, conj, r);
                for (int k = 0; k < 4; k++)
                {
                    y[j + k] += alpha * Unsafe.As<Complex, T>(ref r[k]);
                }
            }
        }

        for (; j < c1; j++)
        {
            y[j] += alpha * VectorOps.Dot(new ReadOnlySpan<T>(a + (j * ld), m), xs, conj);
        }
    }

    // A <- α*x*op(y)^T + A（conjugateY が true なら共役）。
    internal static void Ger(int m, int n, T alpha, ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> a, int lda, bool conjugateY)
    {
        if (m == 0 || n == 0 || alpha == T.Zero)
        {
            return;
        }

        int threads = Threads(m, n);
        fixed (T* pa = a)
        fixed (T* px = x)
        fixed (T* py = y)
        {
            if (threads > 1)
            {
                AddOuterColumnsParallel(m, n, threads, alpha, px, py, pa, lda, conjugateY);
            }
            else
            {
                AddOuterColumns(0, n, m, alpha, px, py, pa, lda, conjugateY);
            }
        }
    }

    private static void AddOuterColumnsParallel(int m, int n, int threads, T alpha, T* x, T* y, T* a, int lda, bool conjugateY)
    {
        var work = new AddOuterColumnsWork(m, n, threads, alpha, x, y, a, lda, conjugateY);
        Parallelism.For(threads, threads, ref work);
    }

    private readonly struct AddOuterColumnsWork(int m, int n, int threads, T alpha, T* x, T* y, T* a, int lda, bool conjugateY) : IParallelWork
    {
        public void Invoke(int t)
        {
            var (lo, hi) = Split(n, threads, t, 1);
            AddOuterColumns(lo, hi, m, alpha, x, y, a, lda, conjugateY);
        }
    }

    private static void AddOuterColumns(int c0, int c1, int m, T alpha, T* x, T* y, T* a, int lda, bool conjugateY)
    {
        var xs = new ReadOnlySpan<T>(x, m);
        for (int j = c0; j < c1; j++)
        {
            T t = alpha * (conjugateY ? VectorOps.Conjugate(y[j]) : y[j]);
            if (t != T.Zero)
            {
                VectorOps.Axpy(t, xs, new Span<T>(a + ((nint)j * lda), m));
            }
        }
    }

    // y <- α*A*x + β*y（対称またはエルミート行列。hermitian 時は対角の虚部を参照しない）。
    internal static void Symv(BlasTriangle uplo, int n, T alpha, ReadOnlySpan<T> a, int lda, ReadOnlySpan<T> x, T beta, Span<T> y, bool hermitian)
    {
        ScaleOrClear(beta, y[..n]);
        if (n == 0 || alpha == T.Zero)
        {
            return;
        }

        int parts = PartialCount(n, n / 2);
        fixed (T* pa = a)
        fixed (T* px = x)
        fixed (T* py = y)
        {
            if (parts <= 1)
            {
                SymmetricColumns(uplo, 0, n, n, alpha, pa, lda, px, py, hermitian);
            }
            else
            {
                SymmetricColumnsPartitioned(uplo, n, parts, alpha, pa, lda, px, py, hermitian);
            }
        }
    }

    // 三角形の面積が等しくなる列範囲に分割し、各作業ベクトルを番号順に加算する。
    private static void SymmetricColumnsPartitioned(BlasTriangle uplo, int n, int parts, T alpha, T* a, int lda, T* x, T* y, bool hermitian)
    {
        T[] partial = WorkArrayPool<T>.Rent(checked(parts * n));
        try
        {
            fixed (T* pp = partial)
            {
                var work = new SymmetricColumnsWork(uplo, n, parts, alpha, a, lda, x, pp, hermitian);
                Parallelism.For(parts, parts, ref work);

                var dest = new Span<T>(y, n);
                for (int t = 0; t < parts; t++)
                {
                    VectorOps.Axpy(T.One, new ReadOnlySpan<T>(pp + ((nint)t * n), n), dest);
                }
            }
        }
        finally
        {
            WorkArrayPool<T>.Return(partial);
        }
    }

    private readonly struct SymmetricColumnsWork(BlasTriangle uplo, int n, int parts, T alpha, T* a, int lda, T* x, T* partial, bool hermitian) : IParallelWork
    {
        public void Invoke(int t)
        {
            // 下三角では列 j の長さが n - j、上三角では j + 1
            int lo = TriangleBound(n, parts, t, uplo);
            int hi = TriangleBound(n, parts, t + 1, uplo);
            T* w = partial + ((nint)t * n);
            new Span<T>(w, n).Clear();
            SymmetricColumns(uplo, lo, hi, n, alpha, a, lda, x, w, hermitian);
        }
    }

    private static int TriangleBound(int n, int parts, int index, BlasTriangle uplo)
    {
        if (index <= 0)
        {
            return 0;
        }

        if (index >= parts)
        {
            return n;
        }

        // 下三角: 列 [0, c) の面積 ≈ n² - (n - c)² が全体 n² の index/parts になる c
        double f = (double)index / parts;
        double c = uplo == BlasTriangle.Lower ? n * (1 - Math.Sqrt(1 - f)) : n * Math.Sqrt(f);
        return Math.Clamp((int)Math.Round(c), 0, n);
    }

    private static void SymmetricColumns(BlasTriangle uplo, int c0, int c1, int n, T alpha, T* a, int lda, T* x, T* y, bool hermitian)
    {
        nint ld = lda;
        int j = c0;
        if (!IsComplex)
        {
            // 実数型は2列ずつ: 2×2 の対角ブロックを先に処理し、残りの部分で x・y の読み書きを共有する
            for (; j <= c1 - 2; j += 2)
            {
                T* ca = a + (j * ld);
                T* cb = ca + ld;
                T ta = alpha * x[j];
                T tb = alpha * x[j + 1];
                T sa, sb;
                if (uplo == BlasTriangle.Lower)
                {
                    // 対角ブロック [[A_jj, ·], [A_j+1,j, A_j+1,j+1]]
                    T off = ca[j + 1];
                    y[j] += (ta * ca[j]) + (tb * off);
                    y[j + 1] += (ta * off) + (tb * cb[j + 1]);
                    int len = n - j - 2;
                    RealVectorOps.AxpyDot2(len, ta, ca + j + 2, tb, cb + j + 2, x + j + 2, y + j + 2, out sa, out sb);
                }
                else
                {
                    T off = cb[j];
                    y[j] += (ta * ca[j]) + (tb * off);
                    y[j + 1] += (ta * off) + (tb * cb[j + 1]);
                    RealVectorOps.AxpyDot2(j, ta, ca, tb, cb, x, y, out sa, out sb);
                }

                y[j] += alpha * sa;
                y[j + 1] += alpha * sb;
            }
        }

        else if (Vector256.IsHardwareAccelerated)
        {
            // 複素数も2列ずつ（対角ブロックのエルミート側は共役、対角は実部だけを使う）
            Complex* za = (Complex*)a, zx = (Complex*)x, zy = (Complex*)y;
            Complex al = Unsafe.As<T, Complex>(ref alpha);
            for (; j <= c1 - 2; j += 2)
            {
                Complex* ca = za + (j * ld);
                Complex* cb = ca + ld;
                Complex ta = al * zx[j];
                Complex tb = al * zx[j + 1];
                Complex da = hermitian ? ca[j].Real : ca[j];
                Complex db = hermitian ? cb[j + 1].Real : cb[j + 1];
                Complex sa, sb;
                if (uplo == BlasTriangle.Lower)
                {
                    // 下三角: A[j+1, j] を格納、A[j, j+1] はその（共役）
                    Complex off = ca[j + 1];
                    Complex upper = hermitian ? Complex.Conjugate(off) : off;
                    zy[j] += (ta * da) + (tb * upper);
                    zy[j + 1] += (ta * off) + (tb * db);
                    ComplexVectorOps.AxpyDot2(n - j - 2, ta, ca + j + 2, tb, cb + j + 2, zx + j + 2, zy + j + 2, hermitian, out sa, out sb);
                }
                else
                {
                    // 上三角: A[j, j+1] を格納、A[j+1, j] はその（共役）
                    Complex off = cb[j];
                    Complex lower = hermitian ? Complex.Conjugate(off) : off;
                    zy[j] += (ta * da) + (tb * off);
                    zy[j + 1] += (ta * lower) + (tb * db);
                    ComplexVectorOps.AxpyDot2(j, ta, ca, tb, cb, zx, zy, hermitian, out sa, out sb);
                }

                zy[j] += al * sa;
                zy[j + 1] += al * sb;
            }
        }

        for (; j < c1; j++)
        {
            T* col = a + (j * ld);
            T t1 = alpha * x[j];
            T diag = hermitian ? VectorOps.RealPart(col[j]) : col[j];
            T t2;
            if (uplo == BlasTriangle.Lower)
            {
                int len = n - j - 1;
                t2 = len > 0
                    ? VectorOps.AxpyDot(t1, new ReadOnlySpan<T>(col + j + 1, len), new ReadOnlySpan<T>(x + j + 1, len), new Span<T>(y + j + 1, len), hermitian)
                    : T.Zero;
            }
            else
            {
                t2 = j > 0
                    ? VectorOps.AxpyDot(t1, new ReadOnlySpan<T>(col, j), new ReadOnlySpan<T>(x, j), new Span<T>(y, j), hermitian)
                    : T.Zero;
            }

            y[j] += (t1 * diag) + (alpha * t2);
        }
    }

    private const int TriangularBlock = 64;
    private const double UpdateGrain = 16 * 1024; // 三角行列の非対角ブロック更新の並列化閾値

    private static void UpdateRows(int r0, int r1, int n, T alpha, T* a, int lda, T* x, T* y)
    {
        int threads = Threads(r1 - r0, n, UpdateGrain);
        if (threads <= 1)
        {
            MultiplyRows(r0, r1, n, alpha, a, lda, x, y);
            return;
        }

        // 行 r0 からを先頭とみなせば、各行の計算は MultiplyRows(r0 + lo, r0 + hi, …) と同じになる
        MultiplyRowsParallel(r1 - r0, n, threads, alpha, a + r0, lda, x, y + r0);
    }

    private static void UpdateColumnsTransposed(int c0, int c1, int m, T alpha, T* a, int lda, T* x, T* y, bool conj)
    {
        int threads = Math.Min((c1 - c0) / 4, Threads(m, c1 - c0, UpdateGrain));
        if (threads <= 1)
        {
            MultiplyColumnsTransposed(c0, c1, m, alpha, a, lda, x, y, conj);
            return;
        }

        // 列 c0 からを先頭とみなせば、各列の計算は MultiplyColumnsTransposed(c0 + lo, c0 + hi, …) と同じになる
        MultiplyColumnsTransposedParallel(m, c1 - c0, threads, alpha, a + ((nint)c0 * lda), lda, x, y + c0, conj);
    }

    // x <- op(A)*x（三角行列。非対角ブロックを 4 列融合積で処理）。
    internal static void Trmv(BlasTriangle uplo, BlasTranspose trans, BlasDiagonal diag, int n, ReadOnlySpan<T> a, int lda, Span<T> x)
    {
        if (n <= TriangularBlock)
        {
            TrmvUnblocked(uplo, trans, diag, n, a, lda, x);
            return;
        }

        bool notrans = trans == BlasTranspose.None;
        bool conj = trans == BlasTranspose.ConjugateTranspose;
        bool upper = uplo == BlasTriangle.Upper;
        nint ld = lda;
        fixed (T* pa = a)
        fixed (T* px = x)
        {
            // 元の x を読む部分が先に済むように、ブロックを進める向きを選ぶ
            bool ascending = upper == notrans;
            int count = (n + TriangularBlock - 1) / TriangularBlock;
            for (int bi = 0; bi < count; bi++)
            {
                int b = ascending ? bi : count - 1 - bi;
                int b0 = b * TriangularBlock;
                int b1 = Math.Min(n, b0 + TriangularBlock);
                int nb = b1 - b0;
                if (notrans)
                {
                    // 対角ブロックより上（下）の行に、このブロックの列の寄与を足してから対角ブロックを掛ける
                    if (upper)
                    {
                        UpdateRows(0, b0, nb, T.One, pa + (b0 * ld), lda, px + b0, px);
                    }
                    else
                    {
                        UpdateRows(b1, n, nb, T.One, pa + (b0 * ld), lda, px + b0, px);
                    }

                    TrmvUnblocked(uplo, trans, diag, nb, new ReadOnlySpan<T>(pa + b0 + (b0 * ld), ((nb - 1) * lda) + nb), lda, new Span<T>(px + b0, nb));
                }
                else
                {
                    TrmvUnblocked(uplo, trans, diag, nb, new ReadOnlySpan<T>(pa + b0 + (b0 * ld), ((nb - 1) * lda) + nb), lda, new Span<T>(px + b0, nb));
                    if (upper)
                    {
                        UpdateColumnsTransposed(0, nb, b0, T.One, pa + (b0 * ld), lda, px, px + b0, conj);
                    }
                    else
                    {
                        UpdateColumnsTransposed(0, nb, n - b1, T.One, pa + b1 + (b0 * ld), lda, px + b1, px + b0, conj);
                    }
                }
            }
        }
    }

    // op(A)*x = b を解き x を解で上書きする（対角ブロック求解ごとに残余右辺を 4 列融合積で更新）。
    internal static void Trsv(BlasTriangle uplo, BlasTranspose trans, BlasDiagonal diag, int n, ReadOnlySpan<T> a, int lda, Span<T> x)
    {
        if (n <= TriangularBlock)
        {
            TrsvUnblocked(uplo, trans, diag, n, a, lda, x);
            return;
        }

        bool notrans = trans == BlasTranspose.None;
        bool conj = trans == BlasTranspose.ConjugateTranspose;
        bool upper = uplo == BlasTriangle.Upper;
        nint ld = lda;
        T minusOne = -T.One;
        fixed (T* pa = a)
        fixed (T* px = x)
        {
            // 解が決まった成分から順に使えるように、ブロックを進める向きを選ぶ
            bool ascending = upper != notrans;
            int count = (n + TriangularBlock - 1) / TriangularBlock;
            for (int bi = 0; bi < count; bi++)
            {
                int b = ascending ? bi : count - 1 - bi;
                int b0 = b * TriangularBlock;
                int b1 = Math.Min(n, b0 + TriangularBlock);
                int nb = b1 - b0;
                var block = new ReadOnlySpan<T>(pa + b0 + (b0 * ld), ((nb - 1) * lda) + nb);
                if (notrans)
                {
                    TrsvUnblocked(uplo, trans, diag, nb, block, lda, new Span<T>(px + b0, nb));
                    if (upper)
                    {
                        UpdateRows(0, b0, nb, minusOne, pa + (b0 * ld), lda, px + b0, px);
                    }
                    else
                    {
                        UpdateRows(b1, n, nb, minusOne, pa + (b0 * ld), lda, px + b0, px);
                    }
                }
                else
                {
                    if (upper)
                    {
                        UpdateColumnsTransposed(0, nb, b0, minusOne, pa + (b0 * ld), lda, px, px + b0, conj);
                    }
                    else
                    {
                        UpdateColumnsTransposed(0, nb, n - b1, minusOne, pa + b1 + (b0 * ld), lda, px + b1, px + b0, conj);
                    }

                    TrsvUnblocked(uplo, trans, diag, nb, block, lda, new Span<T>(px + b0, nb));
                }
            }
        }
    }

    private static void TrmvUnblocked(BlasTriangle uplo, BlasTranspose trans, BlasDiagonal diag, int n, ReadOnlySpan<T> a, int lda, Span<T> x)
    {
        bool unit = diag == BlasDiagonal.Unit;
        bool conj = trans == BlasTranspose.ConjugateTranspose;
        bool upper = uplo == BlasTriangle.Upper;
        if (trans == BlasTranspose.None)
        {
            if (upper)
            {
                for (int j = 0; j < n; j++)
                {
                    T xj = x[j];
                    if (xj != T.Zero)
                    {
                        VectorOps.Axpy(xj, a.Slice(j * lda, j), x[..j]);
                        if (!unit)
                        {
                            x[j] = xj * a[j + (j * lda)];
                        }
                    }
                }
            }
            else
            {
                for (int j = n - 1; j >= 0; j--)
                {
                    T xj = x[j];
                    if (xj != T.Zero)
                    {
                        VectorOps.Axpy(xj, a.Slice(j + 1 + (j * lda), n - j - 1), x[(j + 1)..n]);
                        if (!unit)
                        {
                            x[j] = xj * a[j + (j * lda)];
                        }
                    }
                }
            }
        }
        else if (upper)
        {
            for (int j = n - 1; j >= 0; j--)
            {
                T d = unit ? T.One : Op(a[j + (j * lda)], conj);
                x[j] = (d * x[j]) + VectorOps.Dot(a.Slice(j * lda, j), x[..j], conj);
            }
        }
        else
        {
            for (int j = 0; j < n; j++)
            {
                T d = unit ? T.One : Op(a[j + (j * lda)], conj);
                x[j] = (d * x[j]) + VectorOps.Dot(a.Slice(j + 1 + (j * lda), n - j - 1), x[(j + 1)..n], conj);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void TrsvUnblocked(BlasTriangle uplo, BlasTranspose trans, BlasDiagonal diag, int n, ReadOnlySpan<T> a, int lda, Span<T> x)
    {
        bool unit = diag == BlasDiagonal.Unit;
        bool conj = trans == BlasTranspose.ConjugateTranspose;
        bool upper = uplo == BlasTriangle.Upper;
        if (trans == BlasTranspose.None)
        {
            if (upper)
            {
                for (int j = n - 1; j >= 0; j--)
                {
                    if (x[j] != T.Zero)
                    {
                        if (!unit)
                        {
                            x[j] /= a[j + (j * lda)];
                        }

                        VectorOps.Axpy(-x[j], a.Slice(j * lda, j), x[..j]);
                    }
                }
            }
            else
            {
                for (int j = 0; j < n; j++)
                {
                    if (x[j] != T.Zero)
                    {
                        if (!unit)
                        {
                            x[j] /= a[j + (j * lda)];
                        }

                        VectorOps.Axpy(-x[j], a.Slice(j + 1 + (j * lda), n - j - 1), x[(j + 1)..n]);
                    }
                }
            }
        }
        else if (upper)
        {
            for (int j = 0; j < n; j++)
            {
                T t = x[j] - VectorOps.Dot(a.Slice(j * lda, j), x[..j], conj);
                x[j] = unit ? t : t / Op(a[j + (j * lda)], conj);
            }
        }
        else
        {
            for (int j = n - 1; j >= 0; j--)
            {
                T t = x[j] - VectorOps.Dot(a.Slice(j + 1 + (j * lda), n - j - 1), x[(j + 1)..n], conj);
                x[j] = unit ? t : t / Op(a[j + (j * lda)], conj);
            }
        }
    }

    // A <- α*x*op(x)^T + A（uplo 側のみ更新。hermitian 時は対角の虚部を 0 にする）。
    internal static void Syr(BlasTriangle uplo, int n, T alpha, ReadOnlySpan<T> x, Span<T> a, int lda, bool hermitian)
    {
        if (n == 0 || alpha == T.Zero)
        {
            return;
        }

        for (int j = 0; j < n; j++)
        {
            T t = alpha * (hermitian ? VectorOps.Conjugate(x[j]) : x[j]);
            var col = a.Slice(j * lda, n);
            if (t != T.Zero)
            {
                if (uplo == BlasTriangle.Lower)
                {
                    VectorOps.Axpy(t, x[j..n], col[j..n]);
                }
                else
                {
                    VectorOps.Axpy(t, x[..(j + 1)], col[..(j + 1)]);
                }
            }

            if (hermitian)
            {
                col[j] = VectorOps.RealPart(col[j]);
            }
        }
    }

    // A <- α*x*op(y)^T + op(α)*y*op(x)^T + A（uplo 側のみ更新。hermitian 時は対角の虚部を 0 にする）。
    internal static void Syr2(BlasTriangle uplo, int n, T alpha, ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> a, int lda, bool hermitian)
    {
        if (n == 0 || alpha == T.Zero)
        {
            return;
        }

        T alpha2 = hermitian ? VectorOps.Conjugate(alpha) : alpha;
        for (int j = 0; j < n; j++)
        {
            T t1 = alpha * (hermitian ? VectorOps.Conjugate(y[j]) : y[j]);
            T t2 = alpha2 * (hermitian ? VectorOps.Conjugate(x[j]) : x[j]);
            var col = a.Slice(j * lda, n);
            int lo = uplo == BlasTriangle.Lower ? j : 0;
            int hi = uplo == BlasTriangle.Lower ? n : j + 1;
            if (t1 != T.Zero)
            {
                VectorOps.Axpy(t1, x[lo..hi], col[lo..hi]);
            }

            if (t2 != T.Zero)
            {
                VectorOps.Axpy(t2, y[lo..hi], col[lo..hi]);
            }

            if (hermitian)
            {
                col[j] = VectorOps.RealPart(col[j]);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static T Op(T v, bool conj) => conj ? VectorOps.Conjugate(v) : v;
}
