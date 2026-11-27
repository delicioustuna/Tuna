using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Tuna.LinearAlgebra;
using Tuna.RandomSource;
using Simd = System.Numerics.Vector;
using SimdDouble = System.Numerics.Vector<double>;

namespace Tuna.Statistics.Distributions;

// 行列は列優先の密配列（要素 (i, j) は i + j * n）で扱う。
internal static class MultivariateMath
{
    // 対称性判定の相対許容誤差 √ε（|a_ij - a_ji| <= √ε √|a_ii| √|a_jj|）
    internal const double SymmetryTolerance = 1.4901161193847656e-08;

    internal const int StackLimit = 256;

    internal static double[] FiniteVector(ReadOnlyVectorSpan<double> values, string name)
    {
        if (values.IsEmpty)
        {
            throw new ArgumentException("The dimension must be at least 1.", name);
        }

        var copy = new double[values.Length];
        for (int i = 0; i < copy.Length; i++)
        {
            double value = values[i];
            if (!double.IsFinite(value))
            {
                throw new ArgumentOutOfRangeException(name, value, "Each element must be finite.");
            }

            copy[i] = value;
        }

        return copy;
    }

    internal static void RequireShape(ReadOnlyMatrixSpan<double> values, int rows, int columns, string name)
    {
        if (values.Rows != rows || values.Columns != columns)
        {
            throw new ArgumentException($"The matrix must be {rows} × {columns}.", name);
        }
    }

    internal static double[] FiniteMatrix(ReadOnlyMatrixSpan<double> values, int rows, int columns, string name)
    {
        RequireShape(values, rows, columns, name);
        var copy = new double[DenseStorage.Length(rows, columns)];
        for (int j = 0; j < columns; j++)
        {
            for (int i = 0; i < rows; i++)
            {
                double value = values[i, j];
                if (!double.IsFinite(value))
                {
                    throw new ArgumentOutOfRangeException(name, value, "Each element must be finite.");
                }

                copy[i + j * rows] = value;
            }
        }

        return copy;
    }

    internal static double[] SymmetricMatrix(ReadOnlyMatrixSpan<double> values, int n, string name)
    {
        var copy = FiniteMatrix(values, n, n, name);
        if (!IsSymmetric(values))
        {
            throw new ArgumentException("The matrix must be symmetric.", name);
        }

        MirrorLower(copy, n);
        return copy;
    }

    internal static double[] PositiveDefiniteFactor(double[] symmetric, int n, string name)
    {
        var factor = (double[])symmetric.Clone();
        if (!FactorInPlace(factor, n))
        {
            throw new ArgumentException("The matrix must be positive definite.", name);
        }

        return factor;
    }

    // 下三角のみを検証してコピーし、上三角は 0 のままにする
    internal static double[] LowerFactor(ReadOnlyMatrixSpan<double> values, int n, string name)
    {
        RequireShape(values, n, n, name);
        var factor = new double[DenseStorage.Length(n, n)];
        for (int j = 0; j < n; j++)
        {
            for (int i = j; i < n; i++)
            {
                double value = values[i, j];
                if (!double.IsFinite(value) || (i == j && !(value > 0)))
                {
                    throw new ArgumentOutOfRangeException(name, value, "The lower triangle must be finite and the diagonal must be positive.");
                }

                factor[i + j * n] = value;
            }
        }

        return factor;
    }

    internal static double[] MultiplyByTranspose(double[] l, int n, string name)
    {
        var product = new double[l.Length];
        for (int j = 0; j < n; j++)
        {
            for (int i = j; i < n; i++)
            {
                double sum = 0;
                for (int k = 0; k <= j; k++)
                {
                    sum += l[i + k * n] * l[j + k * n];
                }

                if (!double.IsFinite(sum))
                {
                    throw new ArgumentOutOfRangeException(name, sum, "The product of the factor and its transpose must be finite.");
                }

                product[i + j * n] = sum;
                product[j + i * n] = sum;
            }
        }

        return product;
    }

    internal static bool IsSymmetric(ReadOnlyMatrixSpan<double> a)
    {
        int n = a.Rows;
        for (int j = 0; j < n; j++)
        {
            double scale = Math.Sqrt(Math.Abs(a[j, j]));
            for (int i = j + 1; i < n; i++)
            {
                double tolerance = SymmetryTolerance * (Math.Sqrt(Math.Abs(a[i, i])) * scale);
                if (!(Math.Abs(a[i, j] - a[j, i]) <= tolerance))
                {
                    return false;
                }
            }
        }

        return true;
    }

    // NaN を含む場合は NaN、±∞ を含む場合は +∞、全要素が有限なら 0 を返す
    internal static double NonFiniteStatus(ReadOnlyMatrixSpan<double> x)
    {
        bool infinite = false;
        for (int j = 0; j < x.Columns; j++)
        {
            for (int i = 0; i < x.Rows; i++)
            {
                double value = x[i, j];
                if (double.IsNaN(value))
                {
                    return double.NaN;
                }

                infinite |= double.IsInfinity(value);
            }
        }

        return infinite ? double.PositiveInfinity : 0;
    }

    internal static void MirrorLower(Span<double> a, int n)
    {
        for (int j = 0; j < n; j++)
        {
            for (int i = j + 1; i < n; i++)
            {
                a[j + i * n] = a[i + j * n];
            }
        }
    }

    internal static void CopyLowerSymmetric(ReadOnlyMatrixSpan<double> x, Span<double> destination, int n)
    {
        for (int j = 0; j < n; j++)
        {
            for (int i = j; i < n; i++)
            {
                double value = x[i, j];
                destination[i + j * n] = value;
                destination[j + i * n] = value;
            }
        }
    }

    // 列優先の対称行列をインプレースでコレスキー分解し、上三角を 0 クリアする
    internal static bool FactorInPlace(Span<double> a, int n)
    {
        if (CholeskyKernels.Factor(n, a, n) != 0)
        {
            return false;
        }

        for (int j = 1; j < n; j++)
        {
            a.Slice(j * n, j).Clear();
        }

        return true;
    }

    internal static double SumLogDiagonal(ReadOnlySpan<double> l, int n)
    {
        double sum = 0;
        for (int i = 0; i < n; i++)
        {
            sum += Math.Log(l[i + i * n]);
        }

        return sum;
    }

    // 列指向の前進代入で Lz = b をインプレースで解く（先頭 start 要素は 0 とみなす）
    internal static void SolveLowerInPlace(ReadOnlySpan<double> l, int n, Span<double> b, int offset = 0, int stride = 1, int start = 0)
    {
        if (start >= 0 && (long)n - start >= 16 && stride == 1 && l.Length >= (long)n * n)
        {
            SolveLowerContiguousInPlace(l, b.Slice(offset, n), start);
            return;
        }

        for (int j = start; j < n; j++)
        {
            double z = b[offset + j * stride] / l[j + j * n];
            b[offset + j * stride] = z;
            for (int i = j + 1; i < n; i++)
            {
                b[offset + i * stride] -= l[i + j * n] * z;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void SolveLowerContiguousInPlace(ReadOnlySpan<double> l, Span<double> b, int start)
    {
        int n = b.Length;
        ref double factor = ref MemoryMarshal.GetReference(l);
        ref double rhs = ref MemoryMarshal.GetReference(b);
        int width = SimdDouble.Count;
        for (int j = start; j < n; j++)
        {
            ref double column = ref Unsafe.Add(ref factor, (nint)j * n);
            ref double pivot = ref Unsafe.Add(ref rhs, j);
            double z = pivot / Unsafe.Add(ref column, j);
            pivot = z;
            int i = j + 1;

            // 行ごとに更新は独立。乗算と減算を分けてスカラー経路と同じ丸め順序を保つ。
            if (Simd.IsHardwareAccelerated && i <= n - width)
            {
                var scale = new SimdDouble(z);
                for (; i <= n - width; i += width)
                {
                    var coefficients = Simd.LoadUnsafe(ref column, (nuint)i);
                    var values = Simd.LoadUnsafe(ref rhs, (nuint)i);
                    Simd.StoreUnsafe(values - coefficients * scale, ref rhs, (nuint)i);
                }
            }

            for (; i < n; i++)
            {
                Unsafe.Add(ref rhs, i) -= Unsafe.Add(ref column, i) * z;
            }
        }
    }

    // z = L⁻¹(x - μ) を求める。NaN を含めば NaN、±∞ または非有限解なら +∞、有限なら 0 を返す。
    // x - μ のみがオーバーフローする場合は 1/2 スケーリングで解いて 2 倍に戻す。
    internal static double Standardize(ReadOnlyVectorSpan<double> x, ReadOnlySpan<double> mean, ReadOnlySpan<double> l, Span<double> z)
    {
        int n = mean.Length;
        bool infinite = false;
        for (int i = 0; i < n; i++)
        {
            double value = x[i];
            if (double.IsNaN(value))
            {
                return double.NaN;
            }

            infinite |= double.IsInfinity(value);
        }

        if (infinite)
        {
            return double.PositiveInfinity;
        }

        bool halved = false;
        for (int i = 0; i < n; i++)
        {
            double difference = x[i] - mean[i];
            if (double.IsInfinity(difference))
            {
                halved = true;
                break;
            }

            z[i] = difference;
        }

        if (halved)
        {
            for (int i = 0; i < n; i++)
            {
                z[i] = 0.5 * x[i] - 0.5 * mean[i];
            }
        }

        SolveLowerInPlace(l, n, z);
        for (int i = 0; i < n; i++)
        {
            double value = halved ? z[i] * 2 : z[i];
            if (!double.IsFinite(value))
            {
                return double.PositiveInfinity;
            }

            z[i] = value;
        }

        return 0;
    }

    // ‖z‖²/2 を計算する（‖z‖² がオーバーフローしても半分が表現可能なら有限値を返す）
    internal static double HalfSquaredNorm(ReadOnlySpan<double> z)
    {
        double sum = 0;
        foreach (double value in z)
        {
            sum += 0.5 * value * value;
        }

        return sum;
    }

    // destination = μ + L t。和のみがオーバーフローする場合は 1/2 スケーリングで回避する
    internal static void AddLowerProduct(ReadOnlySpan<double> mean, ReadOnlySpan<double> l, ReadOnlySpan<double> t, VectorSpan<double> destination)
    {
        int n = mean.Length;
        for (int i = 0; i < n; i++)
        {
            double sum = l[i] * t[0];
            bool finite = double.IsFinite(t[0]);
            for (int k = 1; k <= i; k++)
            {
                sum += l[i + k * n] * t[k];
                finite &= double.IsFinite(t[k]);
            }

            double x = mean[i] + sum;
            if (double.IsInfinity(x) && finite)
            {
                double half = 0.5 * l[i] * t[0];
                for (int k = 1; k <= i; k++)
                {
                    half += 0.5 * l[i + k * n] * t[k];
                }

                x = 2 * (0.5 * mean[i] + half);
            }

            destination[i] = x;
        }
    }

    // destination = mean + scale * (L t)。L t の成分が 0 のときは scale = ∞ でも 0 * ∞ を避ける
    internal static void AddScaledLowerProduct(ReadOnlySpan<double> mean, ReadOnlySpan<double> l, ReadOnlySpan<double> t, double scale, VectorSpan<double> destination)
    {
        int n = mean.Length;
        for (int i = 0; i < n; i++)
        {
            double sum = 0;
            for (int k = 0; k <= i; k++)
            {
                sum += l[i + k * n] * t[k];
            }

            destination[i] = sum == 0 ? mean[i] : mean[i] + scale * sum;
        }
    }

    // y = a²/2 に対する ln(y^k e^(-y) / Γ(k)) と ln y を求める（y が正規化数なら LogGammaPrefix で桁落ちを避ける）
    internal static double LogGammaPrefixOfHalfSquare(double shape, double a, out double logY)
    {
        double y = 0.5 * a * a;
        if (y >= DistributionMath.MinNormal && y < double.PositiveInfinity)
        {
            logY = Math.Log(y);
            return XMath.Special.LogGammaPrefix(shape, y);
        }

        logY = 2 * Math.Log(a) - DistributionMath.Ln2;
        return shape * logY - y - XMath.Special.LogGamma(shape);
    }

    // 自由度 2 * shape の √χ² 標本。shape < 1 では χ² のアンダーフローを避けるため対数から生成する
    internal static double ChiSample(IRandomSource random, double shape) =>
        shape >= 1
            ? Math.Sqrt(2 * Sampling.StandardGamma(random, shape))
            : Math.Exp(0.5 * (DistributionMath.Ln2 + Sampling.LogStandardGamma(random, shape)));

    // 下三角 B の B Bᵀ を対称に格納する
    internal static void WriteLowerGram(ReadOnlySpan<double> b, int n, MatrixSpan<double> destination)
    {
        for (int j = 0; j < n; j++)
        {
            for (int i = j; i < n; i++)
            {
                double sum = 0;
                for (int k = 0; k <= j; k++)
                {
                    sum += b[i + k * n] * b[j + k * n];
                }

                destination[i, j] = sum;
                destination[j, i] = sum;
            }
        }
    }

    // 下三角行列の積 L B を B にインプレースで格納する
    internal static void MultiplyLowerInPlace(ReadOnlySpan<double> l, int n, Span<double> b)
    {
        // 行 i の積は B の行 j..i のみを参照するため、下の行から更新すれば上書き競合が起きない
        for (int i = n - 1; i >= 0; i--)
        {
            for (int j = 0; j <= i; j++)
            {
                double sum = 0;
                for (int k = j; k <= i; k++)
                {
                    sum += l[i + k * n] * b[k + j * n];
                }

                b[i + j * n] = sum;
            }
        }
    }

    internal static void RequireDisjoint(ReadOnlyMatrixSpan<double> points, VectorSpan<double> destination)
    {
        var output = destination.AsMatrix().AsReadOnly();
        if (StridedLayout.Overlaps(
            in points.Reference, points.Rows, points.Columns, points.RowStride, points.ColumnStride,
            in output.Reference, output.Rows, output.Columns, output.RowStride, output.ColumnStride, out _))
        {
            throw new ArgumentException("The destination must not overlap the points.", nameof(destination));
        }
    }

    internal static void RequireDisjoint(ReadOnlyMatrixSpan<long> points, VectorSpan<double> destination)
    {
        var output = destination.AsMatrix().AsReadOnly();
        if (StridedLayout.OverlapsAcrossTypes(
            in points.Reference, points.Rows, points.Columns, points.RowStride, points.ColumnStride,
            in output.Reference, output.Rows, output.Columns, output.RowStride, output.ColumnStride))
        {
            throw new ArgumentException("The destination must not overlap the points.", nameof(destination));
        }
    }

    // 許容誤差 16 K ε 以内で総和が 1 になる非負確率配列を検証し、総和で正規化したコピーを返す
    internal static double[] Probabilities(ReadOnlySpan<double> values, string name)
    {
        if (values.IsEmpty)
        {
            throw new ArgumentException("At least one category is required.", name);
        }

        double sum = 0;
        foreach (double value in values)
        {
            if (!(value >= 0 && value < double.PositiveInfinity))
            {
                throw new ArgumentOutOfRangeException(name, value, "Each probability must be non-negative and finite.");
            }

            sum += value;
        }

        if (!(Math.Abs(sum - 1) <= 16 * values.Length * DistributionMath.Epsilon))
        {
            throw new ArgumentException($"The probabilities must sum to one (the sum is {sum}).", name);
        }

        var copy = new double[values.Length];
        for (int i = 0; i < copy.Length; i++)
        {
            copy[i] = values[i] / sum;
        }

        return copy;
    }

    // suffix[i] = Σ_{j>=i} v_j と others[i] = Σ_{j!=i} v_j を求める（総和からの減算を避けて相対精度を保つ）
    internal static (double[] Suffix, double[] Others) SuffixAndOthers(ReadOnlySpan<double> values)
    {
        var suffix = new double[values.Length + 1];
        for (int i = values.Length - 1; i >= 0; i--)
        {
            suffix[i] = suffix[i + 1] + values[i];
        }

        var others = new double[values.Length];
        double prefix = 0;
        for (int i = 0; i < values.Length; i++)
        {
            others[i] = prefix + suffix[i + 1];
            prefix += values[i];
        }

        return (suffix, others);
    }

    internal static bool IsComposition(ReadOnlyVectorSpan<long> x, long total)
    {
        long remaining = total;
        for (int i = 0; i < x.Length; i++)
        {
            long value = x[i];
            if (value < 0 || value > remaining)
            {
                return false;
            }

            remaining -= value;
        }

        return remaining == 0;
    }
}
