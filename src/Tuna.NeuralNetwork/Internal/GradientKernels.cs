using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace Tuna.NeuralNetwork;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む。
using Tuna.LinearAlgebra;

internal static unsafe class GradientKernels
{
    // c += a·b。c が列優先でも行優先でもないビューなら、積を作業領域に求めてから加える。
    internal static void AddProduct<T>(MatrixSpan<T> c, ReadOnlyMatrixSpan<T> a, ReadOnlyMatrixSpan<T> b)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (c.Rows == 0 || c.Columns == 0 || a.Columns == 0)
        {
            return;
        }

        var view = c.AsReadOnly();
        if (TryColumnMajor(view, out int ldc))
        {
            Gemm(a, b, T.One, c, ldc);
            return;
        }

        if (TryColumnMajor(view.T, out _))
        {
            AddProduct(c.T, b.T, a.T);
            return;
        }

        int length = c.Rows * c.Columns;
        T[] rented = ArrayPool<T>.Shared.Rent(length);
        try
        {
            var product = new MatrixSpan<T>(rented.AsSpan(0, length), c.Rows, c.Columns);
            Gemm(a, b, T.Zero, product, c.Rows);
            XMath.Add(c, product, c);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(rented);
        }
    }

    // destination += 各行の和（XMath.RowSums と同じ値）。
    internal static void AddRowSums<T>(ReadOnlyMatrixSpan<T> x, VectorSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (destination.Length == 0 || x.Columns == 0)
        {
            return;
        }

        T[] rented = ArrayPool<T>.Shared.Rent(destination.Length);
        try
        {
            var sums = new VectorSpan<T>(rented.AsSpan(0, destination.Length));
            XMath.RowSums(x, sums);
            XMath.Add(destination, sums, destination);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(rented);
        }
    }

    // 列ごとの正規化の逆伝播。x̂ = ((x − K) − δ)·r を順伝播と同じ式で再計算し、
    // g = dy ⊙ scale から dx = r·((g − mean(g)) − x̂·mean(g ⊙ x̂)) を加える（RMSNorm は K = δ = 0 で mean(g) を引かない）。
    // terms には dy ⊙ x̂、shift の勾配は呼び出し側で dy の行ごとの和として求める。
    internal static void NormalizeBackward<T>(
        ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> dy, ReadOnlyVectorSpan<T> scale, ReadOnlySpan<T> stats, bool center,
        MatrixSpan<T> dx, bool hasDx, VectorSpan<T> dscale, bool hasDscale)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        int rows = x.Rows, columns = x.Columns;
        if (rows == 0 || columns == 0 || !(hasDx || hasDscale))
        {
            return;
        }

        T[]? rx = null, rdy = null, rs = null, rdx = null, rterms = null;
        try
        {
            var xd = Dense.Columns(x, ref rx);
            var dyd = Dense.Columns(dy, ref rdy);
            var sd = Dense.Contiguous(scale, ref rs);
            var target = dx;
            bool direct = hasDx && TryColumnMajor(dx.AsReadOnly(), out _);
            if (hasDx && !direct)
            {
                rdx = ArrayPool<T>.Shared.Rent(rows * columns);
                target = new MatrixSpan<T>(rdx.AsSpan(0, rows * columns), rows, columns);
            }

            MatrixSpan<T> terms = default;
            if (hasDscale)
            {
                rterms = ArrayPool<T>.Shared.Rent(rows * columns);
                terms = new MatrixSpan<T>(rterms.AsSpan(0, rows * columns), rows, columns);
            }

            int threads = Split.Threads(8.0 * rows * columns);
            fixed (T* px = &Dense.Ref(xd), pdy = &Dense.Ref(dyd), ps = &Dense.Ref(sd), pst = stats, pdx = &Dense.Ref(target),
                pterms = &Dense.Ref(terms))
            {
                var work = new NormalizeBackwardWork<T>
                {
                    X = px,
                    Dy = pdy,
                    Scale = scale.Length == 0 ? null : ps,
                    Stats = pst,
                    Dx = hasDx ? pdx : null,
                    Terms = hasDscale ? pterms : null,
                    Rows = rows,
                    Ldx = xd.ColumnStride,
                    Lddy = dyd.ColumnStride,
                    Ldt = hasDx ? target.AsReadOnly().ColumnStride : 0,
                    Accumulate = direct,
                    Center = center,
                };
                Split.For(columns, Split.Chunks(columns, threads, 16), threads, ref work);
            }

            if (hasDx && !direct)
            {
                XMath.Add(dx, target, dx);
            }

            if (hasDscale)
            {
                AddRowSums(terms, dscale);
            }
        }
        finally
        {
            Dense.Return(rx);
            Dense.Return(rdy);
            Dense.Return(rs);
            Dense.Return(rdx);
            Dense.Return(rterms);
        }
    }

    // PReLU の逆伝播。dx に (x < 0 ? a : 1)·dy を加え（局所微分がちょうど 0 の要素は加えない）、terms に (x < 0 ? x·dy : 0) を書き込む。
    internal static void PreluBackward<T>(
        ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> dy, ReadOnlyVectorSpan<T> slopes,
        MatrixSpan<T> dx, bool hasDx, VectorSpan<T> dslopes, bool hasDslopes)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        int rows = x.Rows, columns = x.Columns;
        if (rows == 0 || columns == 0 || !(hasDx || hasDslopes))
        {
            return;
        }

        T[]? rx = null, rdy = null, rs = null, rdx = null, rterms = null;
        try
        {
            var xd = Dense.Columns(x, ref rx);
            var dyd = Dense.Columns(dy, ref rdy);
            var sd = Dense.Contiguous(slopes, ref rs);
            var target = dx;
            bool direct = hasDx && TryColumnMajor(dx.AsReadOnly(), out _);
            if (hasDx && !direct)
            {
                rdx = ArrayPool<T>.Shared.Rent(rows * columns);
                target = new MatrixSpan<T>(rdx.AsSpan(0, rows * columns), rows, columns);
            }

            MatrixSpan<T> terms = default;
            if (hasDslopes)
            {
                rterms = ArrayPool<T>.Shared.Rent(rows * columns);
                terms = new MatrixSpan<T>(rterms.AsSpan(0, rows * columns), rows, columns);
            }

            bool perRow = slopes.Length == rows && rows > 1;
            int threads = Split.Threads(2.0 * rows * columns);
            fixed (T* px = &Dense.Ref(xd), pdy = &Dense.Ref(dyd), ps = &Dense.Ref(sd), pdx = &Dense.Ref(target), pterms = &Dense.Ref(terms))
            {
                var work = new PreluBackwardWork<T>
                {
                    X = px,
                    Dy = pdy,
                    Slopes = perRow ? ps : null,
                    Slope = sd[0],
                    Dx = hasDx ? pdx : null,
                    Terms = hasDslopes ? pterms : null,
                    Rows = rows,
                    Ldx = xd.ColumnStride,
                    Lddy = dyd.ColumnStride,
                    Ldt = hasDx ? target.AsReadOnly().ColumnStride : 0,
                    Accumulate = direct,
                };
                Split.For(columns, Split.Chunks(columns, threads, 16), threads, ref work);
            }

            if (hasDx && !direct)
            {
                XMath.Add(dx, target, dx);
            }

            if (hasDslopes)
            {
                if (dslopes.Length == rows && rows > 1)
                {
                    AddRowSums(terms, dslopes);
                }
                else
                {
                    var all = new ReadOnlyVectorSpan<T>(rterms.AsSpan(0, rows * columns));
                    dslopes[0] += XMath.Sum(all);
                }
            }
        }
        finally
        {
            Dense.Return(rx);
            Dense.Return(rdy);
            Dense.Return(rs);
            Dense.Return(rdx);
            Dense.Return(rterms);
        }
    }

    // 1 列分の正規化の逆伝播。縮約は順伝播と同じく決定的な加算順序のため Vector256 に固定する。
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void NormalizeBackwardColumn<T>(T* x, T* dy, int n, T* scale, T* stats, bool center, T* dx, bool accumulate, T* terms)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        int w = Vector256<T>.Count;
        T origin = stats[0], offset = stats[1], inverse = stats[2];
        var o = Vector256.Create(origin);
        var m = Vector256.Create(offset);
        var r = Vector256.Create(inverse);

        // Σ g と Σ g·x̂
        T sum = T.Zero, dot = T.Zero;
        int i = 0;
        if (n >= w)
        {
            var accSum = Vector256<T>.Zero;
            var accDot = Vector256<T>.Zero;
            for (; i <= n - w; i += w)
            {
                var xh = ((Vector256.Load(x + i) - o) - m) * r;
                var d = Vector256.Load(dy + i);
                var g = scale != null ? d * Vector256.Load(scale + i) : d;
                accSum += g;
                accDot = Simd.Fma(g, xh, accDot);
                if (terms != null)
                {
                    (d * xh).Store(terms + i);
                }
            }

            sum = Vector256.Sum(accSum);
            dot = Vector256.Sum(accDot);
        }

        for (; i < n; i++)
        {
            T xh = ((x[i] - origin) - offset) * inverse;
            T d = dy[i];
            T g = scale != null ? d * scale[i] : d;
            sum += g;
            dot = T.FusedMultiplyAdd(g, xh, dot);
            if (terms != null)
            {
                terms[i] = d * xh;
            }
        }

        if (dx == null)
        {
            return;
        }

        T count = T.CreateTruncating(n);
        T meanG = center ? sum / count : T.Zero;
        T meanDot = dot / count;
        var mg = Vector256.Create(meanG);
        var md = Vector256.Create(meanDot);
        i = 0;
        for (; i <= n - w; i += w)
        {
            var xh = ((Vector256.Load(x + i) - o) - m) * r;
            var d = Vector256.Load(dy + i);
            var g = scale != null ? d * Vector256.Load(scale + i) : d;
            var v = ((g - mg) - xh * md) * r;
            (accumulate ? Vector256.Load(dx + i) + v : v).Store(dx + i);
        }

        for (; i < n; i++)
        {
            T xh = ((x[i] - origin) - offset) * inverse;
            T d = dy[i];
            T g = scale != null ? d * scale[i] : d;
            T v = ((g - meanG) - xh * meanDot) * inverse;
            dx[i] = accumulate ? dx[i] + v : v;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void PreluBackwardColumn<T>(T* x, T* dy, int n, T* slopes, T slope, T* dx, bool accumulate, T* terms)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        int i = 0;
        var s = Vector256.Create(slope);
        for (; i <= n - Vector256<T>.Count; i += Vector256<T>.Count)
        {
            var v = Vector256.Load(x + i);
            var d = Vector256.Load(dy + i);
            var negative = Vector256.LessThan(v, Vector256<T>.Zero);
            if (dx != null)
            {
                var a = slopes != null ? Vector256.Load(slopes + i) : s;
                var derivative = Vector256.ConditionalSelect(negative, a, Vector256<T>.One);
                var contribution = derivative * d;
                var zero = Vector256.Equals(derivative, Vector256<T>.Zero);
                var previous = accumulate ? Vector256.Load(dx + i) : Vector256<T>.Zero;
                Vector256.ConditionalSelect(zero, previous, previous + contribution).Store(dx + i);
            }

            if (terms != null)
            {
                Vector256.ConditionalSelect(negative, v * d, Vector256<T>.Zero).Store(terms + i);
            }
        }

        for (; i < n; i++)
        {
            T v = x[i];
            T d = dy[i];
            bool negative = v < T.Zero;
            if (dx != null)
            {
                T derivative = negative ? (slopes != null ? slopes[i] : slope) : T.One;
                T previous = accumulate ? dx[i] : T.Zero;
                dx[i] = derivative == T.Zero ? previous : previous + derivative * d;
            }

            if (terms != null)
            {
                terms[i] = negative ? v * d : T.Zero;
            }
        }
    }

    // 列優先（各列が連続で列の間隔が行数以上）なら、その列の間隔を返す。
    internal static bool TryColumnMajor<T>(ReadOnlyMatrixSpan<T> x, out int leading)
        where T : unmanaged, INumberBase<T>
    {
        int rows = x.Rows, columns = x.Columns;
        leading = columns <= 1 ? Math.Max(1, rows) : x.ColumnStride;
        return (rows <= 1 || x.RowStride == 1) && (columns <= 1 || x.ColumnStride >= Math.Max(1, rows));
    }

    private static void Gemm<T>(ReadOnlyMatrixSpan<T> a, ReadOnlyMatrixSpan<T> b, T beta, MatrixSpan<T> c, int ldc)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        int m = c.Rows, n = c.Columns, k = a.Columns;
        T[]? ra = null, rb = null;
        try
        {
            var transA = Operand(a, ref ra, out var sa, out int lda);
            var transB = Operand(b, ref rb, out var sb, out int ldb);
            var sc = MemoryMarshal.CreateSpan(ref Dense.Ref(c), Extent(m, n, ldc));
            if (typeof(T) == typeof(float))
            {
                Blas.Gemm(transA, transB, m, n, k, 1f, MemoryMarshal.Cast<T, float>(sa), lda, MemoryMarshal.Cast<T, float>(sb), ldb,
                    float.CreateTruncating(beta), MemoryMarshal.Cast<T, float>(sc), ldc);
            }
            else
            {
                Blas.Gemm(transA, transB, m, n, k, 1.0, MemoryMarshal.Cast<T, double>(sa), lda, MemoryMarshal.Cast<T, double>(sb), ldb,
                    double.CreateTruncating(beta), MemoryMarshal.Cast<T, double>(sc), ldc);
            }
        }
        finally
        {
            Dense.Return(ra);
            Dense.Return(rb);
        }
    }

    // BLAS に渡せる配置（列優先、またはその転置）ならそのまま、そうでなければ列優先に詰め直して返す。
    private static BlasTranspose Operand<T>(ReadOnlyMatrixSpan<T> x, ref T[]? rented, out ReadOnlySpan<T> span, out int leading)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (TryColumnMajor(x, out leading))
        {
            span = MemoryMarshal.CreateReadOnlySpan(in x.Reference, Extent(x.Rows, x.Columns, leading));
            return BlasTranspose.None;
        }

        var t = x.T;
        if (TryColumnMajor(t, out leading))
        {
            span = MemoryMarshal.CreateReadOnlySpan(in t.Reference, Extent(t.Rows, t.Columns, leading));
            return BlasTranspose.Transpose;
        }

        int length = x.Rows * x.Columns;
        rented = ArrayPool<T>.Shared.Rent(length);
        x.CopyTo(new MatrixSpan<T>(rented.AsSpan(0, length), x.Rows, x.Columns));
        span = rented.AsSpan(0, length);
        leading = Math.Max(1, x.Rows);
        return BlasTranspose.None;
    }

    private static int Extent(int rows, int columns, int leading) =>
        rows == 0 || columns == 0 ? 0 : checked((columns - 1) * leading + rows);

    private struct NormalizeBackwardWork<T> : Split.IRangeWork where T : unmanaged, IFloatingPointIeee754<T>
    {
        public T* X, Dy, Scale, Stats, Dx, Terms;
        public int Rows, Ldx, Lddy, Ldt;
        public bool Accumulate, Center;

        public readonly void Invoke(int start, int end)
        {
            for (int j = start; j < end; j++)
            {
                NormalizeBackwardColumn(X + (nint)j * Ldx, Dy + (nint)j * Lddy, Rows, Scale, Stats + (nint)3 * j, Center,
                    Dx == null ? null : Dx + (nint)j * Ldt, Accumulate, Terms == null ? null : Terms + (nint)j * Rows);
            }
        }
    }

    private struct PreluBackwardWork<T> : Split.IRangeWork where T : unmanaged, IFloatingPointIeee754<T>
    {
        public T* X, Dy, Slopes, Dx, Terms;
        public T Slope;
        public int Rows, Ldx, Lddy, Ldt;
        public bool Accumulate;

        public readonly void Invoke(int start, int end)
        {
            for (int j = start; j < end; j++)
            {
                PreluBackwardColumn(X + (nint)j * Ldx, Dy + (nint)j * Lddy, Rows, Slopes, Slope,
                    Dx == null ? null : Dx + (nint)j * Ldt, Accumulate, Terms == null ? null : Terms + (nint)j * Rows);
            }
        }
    }
}
