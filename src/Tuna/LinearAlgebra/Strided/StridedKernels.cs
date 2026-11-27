using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace Tuna.LinearAlgebra;

// ストライド付きビューの演算カーネル（ベクトルは Length × 1 行列として共通化）。
// 要素演算は完全一致のインプレースのみ許容し、積と縮約は重なりをすべて拒否する。
internal static partial class StridedKernels
{
    internal static void Copy<T>(ReadOnlyMatrixSpan<T> source, MatrixSpan<T> destination)
        where T : unmanaged, INumberBase<T> =>
        Map(source, destination, default(IdentityOperator<T>));

    internal static void Map<T, TOperator>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination, TOperator op)
        where T : unmanaged, INumberBase<T>
        where TOperator : IStatefulUnaryOperator<T>
    {
        RequireSameShape(x, destination);
        RequireElementwiseAlias(x, destination);
        if (x.TryGetDense(out var xs) && destination.TryGetDense(out var ds))
        {
            if (DenseElementwise.TryMap(xs, ds, op))
            {
                return;
            }

            for (int i = 0; i < ds.Length; i++)
            {
                ds[i] = op.Invoke(xs[i]);
            }

            return;
        }

        if (TryMapByLine(x, destination, op))
        {
            return;
        }

        if (TryMapBuffered(x, destination, op))
        {
            return;
        }

        for (int j = 0; j < x.Columns; j++)
        {
            for (int i = 0; i < x.Rows; i++)
            {
                Element(destination, i, j) = op.Invoke(x.ElementAt(i, j));
            }
        }
    }

    // 対応 SIMD 実装がある演算子について、非連続ビューを作業領域へパックして DenseElementwise.TryMap で計算する（レイアウト非依存なビット一致維持）。
    private static bool TryMapBuffered<T, TOperator>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination, TOperator op)
        where T : unmanaged, INumberBase<T>
    {
        if (!DenseElementwise.TryMap<T, TOperator>([], [], op))
        {
            return false;
        }

        if (x.Rows == 1)
        {
            x = x.T;
            destination = destination.T;
        }

        const int Chunk = 64;
        Span<T> buffer = stackalloc T[Chunk];
        var output = destination.AsReadOnly();
        int rows = x.Rows;
        int sourceStride = x.RowStride, destinationStride = output.RowStride;
        long total = (long)rows * x.Columns;
        for (long start = 0; start < total; start += Chunk)
        {
            var part = buffer[..(int)Math.Min(Chunk, total - start)];
            var (firstColumn, firstRow) = Math.DivRem(start, rows);
            int row = (int)firstRow, column = (int)firstColumn;
            for (int copied = 0; copied < part.Length;)
            {
                int count = Math.Min(rows - row, part.Length - copied);
                ref readonly T source = ref x.ElementAt(row, column);
                for (int i = 0; i < count; i++)
                {
                    part[copied + i] = Unsafe.Add(ref Unsafe.AsRef(in source), (nint)i * sourceStride);
                }

                copied += count;
                row = 0;
                column++;
            }

            DenseElementwise.TryMap<T, TOperator>(part, part, op);
            row = (int)firstRow;
            column = (int)firstColumn;
            for (int copied = 0; copied < part.Length;)
            {
                int count = Math.Min(rows - row, part.Length - copied);
                ref T target = ref Unsafe.AsRef(in output.ElementAt(row, column));
                for (int i = 0; i < count; i++)
                {
                    Unsafe.Add(ref target, (nint)i * destinationStride) = part[copied + i];
                }

                copied += count;
                row = 0;
                column++;
            }
        }

        return true;
    }

    // 列または行が連続なビューに対し、全ラインをまとめて DenseElementwise.TryMapLines へ渡す。
    private static bool TryMapByLine<T, TOperator>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination, TOperator op)
        where T : unmanaged, INumberBase<T>
    {
        var output = destination.AsReadOnly();
        if (x.Rows > 1 && x.RowStride == 1 && output.RowStride == 1)
        {
            return DenseElementwise.TryMapLines(in x.Reference, x.ColumnStride,
                ref Unsafe.AsRef(in output.Reference), output.ColumnStride, x.Rows, x.Columns, op);
        }

        if (x.Columns > 1 && x.ColumnStride == 1 && output.ColumnStride == 1)
        {
            return DenseElementwise.TryMapLines(in x.Reference, x.RowStride,
                ref Unsafe.AsRef(in output.Reference), output.RowStride, x.Columns, x.Rows, op);
        }

        return false;
    }

    private static bool TryZipByLine<T, TOperator>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y, MatrixSpan<T> destination, TOperator op)
        where T : unmanaged, INumberBase<T>
    {
        var output = destination.AsReadOnly();
        if (x.Rows > 1 && x.RowStride == 1 && y.RowStride == 1 && output.RowStride == 1)
        {
            return DenseElementwise.TryZipLines<T, TOperator>(in x.Reference, x.ColumnStride, in y.Reference, y.ColumnStride,
                ref Unsafe.AsRef(in output.Reference), output.ColumnStride, x.Rows, x.Columns);
        }

        if (x.Columns > 1 && x.ColumnStride == 1 && y.ColumnStride == 1 && output.ColumnStride == 1)
        {
            return DenseElementwise.TryZipLines<T, TOperator>(in x.Reference, x.RowStride, in y.Reference, y.RowStride,
                ref Unsafe.AsRef(in output.Reference), output.RowStride, x.Columns, x.Rows);
        }

        return false;
    }

    internal static void Zip<T, TOperator>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y, MatrixSpan<T> destination, TOperator op)
        where T : unmanaged, INumberBase<T>
        where TOperator : IStatefulBinaryOperator<T>
    {
        RequireSameShape(x, y);
        RequireSameShape(x, destination);
        RequireElementwiseAlias(x, destination);
        RequireElementwiseAlias(y, destination);
        if (x.TryGetDense(out var xs) && y.TryGetDense(out var ys) && destination.TryGetDense(out var ds))
        {
            if (typeof(TOperator) == typeof(AddOperator<T>) || typeof(TOperator) == typeof(SubtractOperator<T>))
            {
                DenseElementwise.Add(xs, ys, ds, subtract: typeof(TOperator) == typeof(SubtractOperator<T>));
                return;
            }

            if (DenseElementwise.TryZip(xs, ys, ds, op))
            {
                return;
            }

            for (int i = 0; i < ds.Length; i++)
            {
                ds[i] = op.Invoke(xs[i], ys[i]);
            }

            return;
        }

        if (TryZipByLine(x, y, destination, op))
        {
            return;
        }

        if (TryZipBroadcastByLine(x, y, destination, op))
        {
            return;
        }

        for (int j = 0; j < x.Columns; j++)
        {
            for (int i = 0; i < x.Rows; i++)
            {
                Element(destination, i, j) = op.Invoke(x.ElementAt(i, j), y.ElementAt(i, j));
            }
        }
    }

    // 片方が列（または行）ごとに定数なら、連続軸をスカラーとのブロードキャスト SIMD 演算に通す。
    private static bool TryZipBroadcastByLine<T, TOperator>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y, MatrixSpan<T> destination, TOperator op)
        where T : unmanaged, INumberBase<T>
    {
        var output = destination.AsReadOnly();
        if (x.Rows > 1 && output.RowStride == 1
            && ((x.RowStride == 0 && y.RowStride == 1) || (y.RowStride == 0 && x.RowStride == 1)))
        {
            bool scalarOnLeft = x.RowStride == 0;
            var scalars = scalarOnLeft ? x : y;
            var values = scalarOnLeft ? y : x;
            return DenseElementwise.TryZipScalarLines<T, TOperator>(in values.Reference, values.ColumnStride, in scalars.Reference, scalars.ColumnStride,
                ref Unsafe.AsRef(in output.Reference), output.ColumnStride, x.Rows, x.Columns, scalarOnLeft);
        }

        if (x.Columns > 1 && output.ColumnStride == 1
            && ((x.ColumnStride == 0 && y.ColumnStride == 1) || (y.ColumnStride == 0 && x.ColumnStride == 1)))
        {
            bool scalarOnLeft = x.ColumnStride == 0;
            var scalars = scalarOnLeft ? x : y;
            var values = scalarOnLeft ? y : x;
            return DenseElementwise.TryZipScalarLines<T, TOperator>(in values.Reference, values.RowStride, in scalars.Reference, scalars.RowStride,
                ref Unsafe.AsRef(in output.Reference), output.RowStride, x.Columns, x.Rows, scalarOnLeft);
        }

        return false;
    }

    internal static void Zip<T, TOperator>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y, ReadOnlyMatrixSpan<T> z, MatrixSpan<T> destination, TOperator op)
        where T : unmanaged, INumberBase<T>
        where TOperator : IStatefulTernaryOperator<T>
    {
        RequireSameShape(x, y);
        RequireSameShape(x, z);
        RequireSameShape(x, destination);
        RequireElementwiseAlias(x, destination);
        RequireElementwiseAlias(y, destination);
        RequireElementwiseAlias(z, destination);
        if (x.TryGetDense(out var xs) && y.TryGetDense(out var ys) && z.TryGetDense(out var zs) && destination.TryGetDense(out var ds))
        {
            for (int i = 0; i < ds.Length; i++)
            {
                ds[i] = op.Invoke(xs[i], ys[i], zs[i]);
            }

            return;
        }

        for (int j = 0; j < x.Columns; j++)
        {
            for (int i = 0; i < x.Rows; i++)
            {
                Element(destination, i, j) = op.Invoke(x.ElementAt(i, j), y.ElementAt(i, j), z.ElementAt(i, j));
            }
        }
    }

    internal static void Scale<T>(ReadOnlyMatrixSpan<T> x, T scalar, MatrixSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        if (x.TryGetDense(out var xs) && destination.TryGetDense(out var ds))
        {
            RequireSameShape(x, destination);
            RequireElementwiseAlias(x, destination);
            DenseElementwise.Scale(xs, scalar, ds);
            return;
        }

        Map(x, destination, new ScaleOperator<T>(scalar));
    }

    // 軟判定しきい値 sign(x) * max(|x| - threshold, 0)（threshold >= 0 前提。連続軸があればライン単位で SIMD を使用）。
    internal static void SoftThreshold<T>(ReadOnlyMatrixSpan<T> x, T threshold, MatrixSpan<T> destination)
        where T : unmanaged, INumber<T>
    {
        RequireSameShape(x, destination);
        RequireElementwiseAlias(x, destination);
        if (x.TryGetDense(out var xs) && destination.TryGetDense(out var ds))
        {
            DenseElementwise.SoftThreshold(xs, threshold, ds);
            return;
        }

        var output = destination.AsReadOnly();
        if (x.Rows > 1 && x.RowStride == 1 && output.RowStride == 1)
        {
            DenseElementwise.SoftThresholdLines(in x.Reference, x.ColumnStride, threshold,
                ref Unsafe.AsRef(in output.Reference), output.ColumnStride, x.Rows, x.Columns);
            return;
        }

        if (x.Columns > 1 && x.ColumnStride == 1 && output.ColumnStride == 1)
        {
            DenseElementwise.SoftThresholdLines(in x.Reference, x.RowStride, threshold,
                ref Unsafe.AsRef(in output.Reference), output.RowStride, x.Columns, x.Rows);
            return;
        }

        Map(x, destination, new SoftThresholdOperator<T>(threshold));
    }

    // C = A * B（配置が合えば GEMM / 列ごと GEMV、それ以外は汎用ループ）。
    internal static void Multiply<T>(ReadOnlyMatrixSpan<T> a, ReadOnlyMatrixSpan<T> b, MatrixSpan<T> c)
        where T : unmanaged, INumberBase<T>
    {
        int m = a.Rows, k = a.Columns, n = b.Columns;
        if (b.Rows != k)
        {
            throw new ArgumentException("Inner dimensions must agree.");
        }

        if (c.Rows != m || c.Columns != n)
        {
            throw new ArgumentException("Destination shape must agree.", "destination");
        }

        RequireDisjoint(a, c);
        RequireDisjoint(b, c);
        MultiplyUnchecked(a, b, c);
    }

    internal static unsafe void MultiplyUnchecked<T>(ReadOnlyMatrixSpan<T> a, ReadOnlyMatrixSpan<T> b, MatrixSpan<T> c)
        where T : unmanaged, INumberBase<T>
    {
        int m = a.Rows, k = a.Columns, n = b.Columns;
        if (m == 0 || n == 0)
        {
            return;
        }

        if (k == 0)
        {
            c.Fill(T.Zero);
            return;
        }

        bool layoutA = TryGetBlasLayout(a, out bool transA, out int lda);
        bool layoutC = TryGetBlasLayout(c.AsReadOnly(), out bool transC, out int ldc) && !transC;
        if (layoutA && layoutC && TryGetBlasLayout(b, out bool transB, out int ldb))
        {
            if (typeof(T) == typeof(double) && !transA && UseSmallProduct && ((long)m * k <= SmallProductMK || m < 4))
            {
                fixed (T* pa = &Unsafe.AsRef(in a.Reference))
                fixed (T* pb = &Unsafe.AsRef(in b.Reference))
                fixed (T* pc = &Unsafe.AsRef(in c.AsReadOnly().Reference))
                {
                    MultiplySmall(m, n, k, (double*)pa, lda, (double*)pb, transB ? ldb : 1, transB ? 1 : ldb, (double*)pc, ldc);
                }

                return;
            }

            if (typeof(T) == typeof(double))
            {
                // GEMM は C に加算するため、先に 0 にする。
                for (int j = 0; j < n; j++)
                {
                    ColumnSpan(c, j).Clear();
                }

                Gemm.Multiply(transA, transB, m, n, k, 1.0,
                    MemoryMarshal.Cast<T, double>(a.Data), lda,
                    MemoryMarshal.Cast<T, double>(b.Data), ldb,
                    MemoryMarshal.Cast<T, double>(c.Data), ldc);
                return;
            }

            if (typeof(T) == typeof(float) && GemmSingle.Prefers(m, k))
            {
                for (int j = 0; j < n; j++)
                {
                    ColumnSpan(c, j).Clear();
                }

                GemmSingle.Multiply(transA, transB, m, n, k, 1f,
                    MemoryMarshal.Cast<T, float>(a.Data), lda,
                    MemoryMarshal.Cast<T, float>(b.Data), ldb,
                    MemoryMarshal.Cast<T, float>(c.Data), ldc);
                return;
            }

            if (typeof(T) == typeof(Complex) && GemmComplex.Prefers(m, n, k))
            {
                // 4M 法で計算し、非有限要素が生じた列のみ複素数の直接経路で再計算する（列独立の判定により列結果を他列に依存させない）。
                if (!GemmComplex.Multiply(transA, transB, m, n, k,
                    MemoryMarshal.Cast<T, Complex>(a.Data), lda,
                    MemoryMarshal.Cast<T, Complex>(b.Data), ldb,
                    MemoryMarshal.Cast<T, Complex>(c.Data), ldc))
                {
                    for (int j = 0; j < n; j++)
                    {
                        if (!IsFinite(MemoryMarshal.Cast<T, Complex>(ColumnSpan(c, j))))
                        {
                            MultiplyColumns(a, b, c, j, j + 1, layout: true, transA, lda);
                        }
                    }
                }

                return;
            }
        }

        MultiplyColumns(a, b, c, 0, n, layoutA && layoutC, transA, lda);
    }

    // C = A * B の列 [first, end) を計算する（経路は n 非依存）。
    private static void MultiplyColumns<T>(ReadOnlyMatrixSpan<T> a, ReadOnlyMatrixSpan<T> b, MatrixSpan<T> c, int first, int end, bool layout, bool transA, int lda)
        where T : unmanaged, INumberBase<T>
    {
        int m = a.Rows, k = a.Columns;

        // 既存の BLAS は出力をゼロビットで初期化するので、T.Zero がゼロビットの型だけ渡す。
        if (T.Zero == default(T) && layout && (k <= 1 || b.RowStride == 1))
        {
            for (int j = first; j < end; j++)
            {
                var column = MemoryMarshal.CreateReadOnlySpan(in b.ElementAt(0, j), k);
                Gemv(a, transA, lda, column, ColumnSpan(c, j));
            }

            return;
        }

        for (int j = first; j < end; j++)
        {
            for (int i = 0; i < m; i++)
            {
                T sum = T.Zero;
                for (int p = 0; p < k; p++)
                {
                    sum = unchecked(sum + a.ElementAt(i, p) * b.ElementAt(p, j));
                }

                Element(c, i, j) = sum;
            }
        }
    }

    private static bool IsFinite(ReadOnlySpan<Complex> x)
    {
        foreach (var z in x)
        {
            if (!double.IsFinite(z.Real) || !double.IsFinite(z.Imaginary))
            {
                return false;
            }
        }

        return true;
    }

    // 列数 n と combined で同一の列計算経路になるか判定する（n 依存は GemmComplex.Prefers のみ）。
    internal static bool SameColumnPath<T>(int m, int n, int combined, int k) =>
        typeof(T) != typeof(Complex) || GemmComplex.Prefers(m, n, k) == GemmComplex.Prefers(m, combined, k);

    // FMA 命令対応時のみ小規模行列積カーネルを使用（ソフトウェア FMA の低速化を避けるため）。
    private static readonly bool UseSmallProduct = System.Runtime.Intrinsics.X86.Fma.IsSupported || System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.IsSupported;

    private const long SmallProductMK = 24 * 24; // MultiplySmall の適用上限（m * k）

    // 小規模行列積 C = A * B（C の列をレジスタ上で直接 FMA 累積し、ゼロクリアと列ごと axpy のオーバーヘッドを省く）。
    private static unsafe void MultiplySmall(int m, int n, int k, double* a, int lda, double* b, nint bRow, nint bColumn, double* c, int ldc)
    {
        nint ld = lda;
        for (int j = 0; j < n; j++)
        {
            double* bj = b + j * bColumn;
            double* cj = c + (nint)j * ldc;
            int i = 0;
            if (Vector512.IsHardwareAccelerated)
            {
                for (; i + 8 <= m; i += 8)
                {
                    var sum = Vector512<double>.Zero;
                    double* ai = a + i;
                    for (int p = 0; p < k; p++, ai += ld)
                    {
                        sum = Vector512.FusedMultiplyAdd(Vector512.Load(ai), Vector512.Create(bj[p * bRow]), sum);
                    }

                    sum.Store(cj + i);
                }
            }

            if (Vector256.IsHardwareAccelerated)
            {
                for (; i + 4 <= m; i += 4)
                {
                    var sum = Vector256<double>.Zero;
                    double* ai = a + i;
                    for (int p = 0; p < k; p++, ai += ld)
                    {
                        sum = Vector256.FusedMultiplyAdd(Vector256.Load(ai), Vector256.Create(bj[p * bRow]), sum);
                    }

                    sum.Store(cj + i);
                }
            }

            if (Vector128.IsHardwareAccelerated)
            {
                for (; i + 2 <= m; i += 2)
                {
                    var sum = Vector128<double>.Zero;
                    double* ai = a + i;
                    for (int p = 0; p < k; p++, ai += ld)
                    {
                        sum = Vector128.FusedMultiplyAdd(Vector128.Load(ai), Vector128.Create(bj[p * bRow]), sum);
                    }

                    sum.Store(cj + i);
                }
            }

            for (; i < m; i++)
            {
                double sum = 0;
                double* ai = a + i;
                for (int p = 0; p < k; p++, ai += ld)
                {
                    sum = Math.FusedMultiplyAdd(*ai, bj[p * bRow], sum);
                }

                cj[i] = sum;
            }
        }
    }

    // y = A * x（配置が合えば BLAS GEMV を使用）。
    internal static void Multiply<T>(ReadOnlyMatrixSpan<T> a, ReadOnlyVectorSpan<T> x, VectorSpan<T> y)
        where T : unmanaged, INumberBase<T>
    {
        if (a.Columns != x.Length)
        {
            throw new ArgumentException("Inner dimensions must agree.");
        }

        if (y.Length != a.Rows)
        {
            throw new ArgumentException("Destination length must agree.", "destination");
        }

        var output = y.AsMatrix();
        RequireDisjoint(a, output);
        RequireDisjoint(x.AsMatrix(), output);
        if (T.Zero == default(T) && TryGetBlasLayout(a, out bool transA, out int lda)
            && (x.Length <= 1 || x.Stride == 1) && output.TryGetDense(out var ys))
        {
            Gemv(a, transA, lda, MemoryMarshal.CreateReadOnlySpan(in x.Reference, x.Length), ys);
            return;
        }

        for (int i = 0; i < a.Rows; i++)
        {
            T sum = T.Zero;
            for (int p = 0; p < a.Columns; p++)
            {
                sum = unchecked(sum + a.ElementAt(i, p) * x.ElementAt(p));
            }

            Element(output, i, 0) = sum;
        }
    }

    private static void Gemv<T>(ReadOnlyMatrixSpan<T> a, bool transpose, int lda, ReadOnlySpan<T> x, Span<T> y)
        where T : unmanaged, INumberBase<T>
    {
        // 転置ビューは、列優先で格納された Columns × Rows の行列の転置として渡す。
        var (rows, columns) = transpose ? (a.Columns, a.Rows) : (a.Rows, a.Columns);
        ParallelLevel2<T>.Gemv(transpose ? BlasTranspose.Transpose : BlasTranspose.None, rows, columns, T.One,
            a.Data, lda, x, T.Zero, y);
    }

    // BLAS 互換配置か判定する（列優先なら transpose=false、行優先なら転置として transpose=true）。
    private static bool TryGetBlasLayout<T>(ReadOnlyMatrixSpan<T> view, out bool transpose, out int leadingDimension)
        where T : unmanaged, INumberBase<T>
    {
        int rows = view.Rows, columns = view.Columns;
        if ((rows <= 1 || view.RowStride == 1) && (columns <= 1 || view.ColumnStride >= Math.Max(1, rows)))
        {
            transpose = false;
            leadingDimension = columns <= 1 ? Math.Max(1, rows) : view.ColumnStride;
            return true;
        }

        if ((columns <= 1 || view.ColumnStride == 1) && (rows <= 1 || view.RowStride >= Math.Max(1, columns)))
        {
            transpose = true;
            leadingDimension = rows <= 1 ? Math.Max(1, columns) : view.RowStride;
            return true;
        }

        transpose = false;
        leadingDimension = 0;
        return false;
    }

    private static Span<T> ColumnSpan<T>(MatrixSpan<T> view, int column) where T : unmanaged, INumberBase<T> =>
        MemoryMarshal.CreateSpan(ref Element(view, 0, column), view.Rows);

    private static ref T Element<T>(MatrixSpan<T> view, int row, int column) where T : unmanaged, INumberBase<T> =>
        ref Unsafe.AsRef(in view.AsReadOnly().ElementAt(row, column));

    private static void RequireSameShape<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y) where T : unmanaged, INumberBase<T>
    {
        if (x.Rows != y.Rows || x.Columns != y.Columns)
        {
            throw new ArgumentException("Matrix dimensions must agree.");
        }
    }

    private static void RequireElementwiseAlias<T>(ReadOnlyMatrixSpan<T> input, MatrixSpan<T> destination) where T : unmanaged, INumberBase<T>
    {
        if (Overlaps(input, destination, out bool exact) && !exact)
        {
            throw new ArgumentException("The destination partially overlaps an input. Only an exact alias is allowed.", nameof(destination));
        }
    }

    internal static void RequireDisjoint<T>(ReadOnlyMatrixSpan<T> input, MatrixSpan<T> destination) where T : unmanaged, INumberBase<T>
    {
        if (Overlaps(input, destination, out _))
        {
            throw new ArgumentException("The destination must not overlap the inputs.", nameof(destination));
        }
    }

    internal static bool Overlaps<T>(ReadOnlyMatrixSpan<T> input, MatrixSpan<T> destination, out bool exact) where T : unmanaged, INumberBase<T>
    {
        var output = destination.AsReadOnly();
        return StridedLayout.Overlaps(
            in input.Reference, input.Rows, input.Columns, input.RowStride, input.ColumnStride,
            in output.Reference, output.Rows, output.Columns, output.RowStride, output.ColumnStride, out exact);
    }
}
