using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Tuna.LinearAlgebra;

// 縮約で加算する index 番目の項（入力ビューを保持するため ref struct で実装）。
internal interface ISumTerm<T>
{
    T Get(int index);
}

internal interface IRowSumTerm<T>
{
    T Get(int row, int column);
}

internal readonly ref struct VectorTerm<T, TOperator> : ISumTerm<T>
    where T : unmanaged, INumberBase<T>
    where TOperator : IBinaryOperator<T>
{
    private readonly ReadOnlyVectorSpan<T> x;
    private readonly T parameter;

    public VectorTerm(ReadOnlyVectorSpan<T> x, T parameter)
    {
        this.x = x;
        this.parameter = parameter;
    }

    public T Get(int index) => TOperator.Invoke(x.ElementAt(index), parameter);
}

internal readonly ref struct ProductTerm<T> : ISumTerm<T> where T : unmanaged, INumberBase<T>
{
    private readonly ReadOnlyVectorSpan<T> x;
    private readonly ReadOnlyVectorSpan<T> y;

    public ProductTerm(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y)
    {
        this.x = x;
        this.y = y;
    }

    public T Get(int index) => unchecked(x.ElementAt(index) * y.ElementAt(index));
}

internal readonly ref struct MatrixTerm<T, TOperator> : IRowSumTerm<T>
    where T : unmanaged, INumberBase<T>
    where TOperator : IBinaryOperator<T>
{
    private readonly ReadOnlyMatrixSpan<T> a;

    public MatrixTerm(ReadOnlyMatrixSpan<T> a) => this.a = a;

    public T Get(int row, int column) => TOperator.Invoke(a.ElementAt(row, column), T.Zero);
}

internal readonly ref struct RowTerm<T, TOperator> : IRowSumTerm<T>
    where T : unmanaged, INumberBase<T>
    where TOperator : IBinaryOperator<T>
{
    private readonly ReadOnlyMatrixSpan<T> a;
    private readonly ReadOnlySpan<T> parameters;

    public RowTerm(ReadOnlyMatrixSpan<T> a, ReadOnlySpan<T> parameters)
    {
        this.a = a;
        this.parameters = parameters;
    }

    public T Get(int row, int column) => TOperator.Invoke(a.ElementAt(row, column), parameters[row]);
}

// ペアワイズ総和（SumBlockSize 要素ブロック内を順次加算し、ブロック和を二分木状に結合）。
// 加算木は長さのみで決まり、レイアウト・向き・スレッド数によらずビット一致する。
internal static partial class StridedKernels
{
    internal const int SumBlockSize = 64;

    // ブロックの数は 2³¹ 未満なので、部分和の段数は 31 以下
    private const int MaxSumDepth = 32;

    [SkipLocalsInit]
    internal static T PairwiseSum<T, TTerm>(int length, TTerm term)
        where T : unmanaged, INumberBase<T>
        where TTerm : ISumTerm<T>, allows ref struct
    {
        Span<T> partial = stackalloc T[MaxSumDepth];
        int depth = 0;
        for (int start = 0, block = 0; start < length; block++)
        {
            int end = length - start <= SumBlockSize ? length : start + SumBlockSize;
            T sum = T.Zero;
            for (int i = start; i < end; i++)
            {
                sum = unchecked(sum + term.Get(i));
            }
            // ブロック番号の末尾に並ぶ 1 の数だけ、同じ大きさの部分和と結合する（前の部分和を左に置く）
            for (int k = block; (k & 1) != 0; k >>= 1)
            {
                sum = unchecked(partial[--depth] + sum);
            }

            partial[depth++] = sum;
            start = end;
        }

        if (depth == 0)
        {
            return T.Zero;
        }

        T result = partial[--depth];
        while (depth > 0)
        {
            result = unchecked(partial[--depth] + result);
        }

        return result;
    }

    // PairwiseSum で Σ TOperator(x[i], parameter) を計算する（連続な float / double はビット一致の SIMD 経路）。
    private static T SumOf<T, TOperator>(ReadOnlyVectorSpan<T> x, T parameter)
        where T : unmanaged, INumberBase<T>
        where TOperator : IBinaryOperator<T> =>
        DenseReductions.TrySum<T, TOperator>(x, parameter, out T sum)
            ? sum
            : PairwiseSum<T, VectorTerm<T, TOperator>>(x.Length, new VectorTerm<T, TOperator>(x, parameter));

    // 各行の総和を RowPairwiseSums と同じ木構造で計算する（引数不要な演算子では parameters は空可）。
    private static void RowSumsOf<T, TOperator>(ReadOnlyMatrixSpan<T> a, ReadOnlySpan<T> parameters, Span<T> destination)
        where T : unmanaged, INumberBase<T>
        where TOperator : IBinaryOperator<T>
    {
        if (DenseReductions.TryRowSums<T, TOperator>(a, parameters, destination))
        {
            return;
        }

        if (parameters.IsEmpty)
        {
            RowPairwiseSums(a.Rows, a.Columns, new MatrixTerm<T, TOperator>(a), destination);
        }
        else
        {
            RowPairwiseSums(a.Rows, a.Columns, new RowTerm<T, TOperator>(a, parameters), destination);
        }
    }

    // 2^k が T の正規化数ならその値を返す（double・float だけ）。ScaleB(x, k) と x·2^k は同じビットになるので、要素ごとの ScaleB を乗算に置き換えられる
    private static bool TryPowerOfTwo<T>(int k, out T factor) where T : IFloatingPointIeee754<T>
    {
        factor = T.One;
        if (!(typeof(T) == typeof(double) || typeof(T) == typeof(float)))
        {
            return false;
        }

        factor = T.ScaleB(T.One, k);
        return T.IsNormal(factor);
    }

    // 列優先に沿って走査しつつ、各行に PairwiseSum と同じ木構造で総和を計算する。
    private static void RowPairwiseSums<T, TTerm>(int rows, int columns, TTerm term, Span<T> destination)
        where T : unmanaged, INumberBase<T>
        where TTerm : IRowSumTerm<T>, allows ref struct
    {
        if (rows == 0)
        {
            return;
        }

        int blocks = (int)(((long)columns + SumBlockSize - 1) / SumBlockSize);
        if (blocks <= 1)
        {
            destination.Fill(T.Zero);
            for (int j = 0; j < columns; j++)
            {
                for (int i = 0; i < rows; i++)
                {
                    destination[i] = unchecked(destination[i] + term.Get(i, j));
                }
            }

            return;
        }

        // 作業領域: 先頭の rows 個がブロックの和、続いて段ごとの部分和
        int levels = BitOperations.Log2((uint)blocks) + 1;
        T[] rented = ArrayPool<T>.Shared.Rent(checked(rows * (levels + 1)));
        try
        {
            var sum = rented.AsSpan(0, rows);
            Span<T> Level(int level) => rented.AsSpan(rows * (level + 1), rows);
            int depth = 0;
            for (int start = 0, block = 0; start < columns; block++)
            {
                int end = columns - start <= SumBlockSize ? columns : start + SumBlockSize;
                sum.Fill(T.Zero);
                for (int j = start; j < end; j++)
                {
                    for (int i = 0; i < rows; i++)
                    {
                        sum[i] = unchecked(sum[i] + term.Get(i, j));
                    }
                }

                for (int k = block; (k & 1) != 0; k >>= 1)
                {
                    var left = Level(--depth);
                    for (int i = 0; i < rows; i++)
                    {
                        sum[i] = unchecked(left[i] + sum[i]);
                    }
                }

                sum.CopyTo(Level(depth++));
                start = end;
            }

            Level(--depth).CopyTo(destination);
            while (depth > 0)
            {
                var left = Level(--depth);
                for (int i = 0; i < rows; i++)
                {
                    destination[i] = unchecked(left[i] + destination[i]);
                }
            }
        }
        finally
        {
            ArrayPool<T>.Shared.Return(rented);
        }
    }

    private static T[] Rent<T>(int length, int count) => ArrayPool<T>.Shared.Rent(Math.Max(1, checked(length * count)));

    private static void WriteTo<T>(ReadOnlySpan<T> values, VectorSpan<T> destination) where T : unmanaged, INumberBase<T>
    {
        for (int i = 0; i < values.Length; i++)
        {
            destination[i] = values[i];
        }
    }

    private static void RequireLength<T>(VectorSpan<T> destination, int length) where T : unmanaged, INumberBase<T>
    {
        if (destination.Length != length)
        {
            throw new ArgumentException("Destination length must agree.", nameof(destination));
        }
    }

    private static void RequireNotEmpty(int length, string name)
    {
        if (length == 0)
        {
            throw new ArgumentException("The input must not be empty.", name);
        }
    }

    // ---------- 和・平均・分散 ----------

    internal static T Sum<T>(ReadOnlyVectorSpan<T> x) where T : unmanaged, INumberBase<T> =>
        SumOf<T, FirstOperator<T>>(x, T.Zero);

    internal static void ColumnSums<T>(ReadOnlyMatrixSpan<T> a, VectorSpan<T> destination) where T : unmanaged, INumberBase<T>
    {
        RequireLength(destination, a.Columns);
        RequireDisjoint(a, destination.AsMatrix());
        for (int j = 0; j < a.Columns; j++)
        {
            destination[j] = Sum(a.Column(j));
        }
    }

    internal static void RowSums<T>(ReadOnlyMatrixSpan<T> a, VectorSpan<T> destination) where T : unmanaged, INumberBase<T> =>
        RowSumsOf<T, FirstOperator<T>>(a, destination);

    private static void RowSumsOf<T, TOperator>(ReadOnlyMatrixSpan<T> a, VectorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
        where TOperator : IBinaryOperator<T>
    {
        RequireLength(destination, a.Rows);
        RequireDisjoint(a, destination.AsMatrix());
        T[] rented = Rent<T>(a.Rows, 1);
        try
        {
            var sums = rented.AsSpan(0, a.Rows);
            RowSumsOf<T, TOperator>(a, default, sums);
            WriteTo<T>(sums, destination);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(rented);
        }
    }

    internal static T SumOfSquares<T>(ReadOnlyVectorSpan<T> x) where T : unmanaged, INumberBase<T> =>
        SumOf<T, SquareOperator<T>>(x, T.Zero);

    internal static void ColumnSumsOfSquares<T>(ReadOnlyMatrixSpan<T> a, VectorSpan<T> destination) where T : unmanaged, INumberBase<T>
    {
        RequireLength(destination, a.Columns);
        RequireDisjoint(a, destination.AsMatrix());
        for (int j = 0; j < a.Columns; j++)
        {
            destination[j] = SumOfSquares(a.Column(j));
        }
    }

    internal static void RowSumsOfSquares<T>(ReadOnlyMatrixSpan<T> a, VectorSpan<T> destination) where T : unmanaged, INumberBase<T> =>
        RowSumsOf<T, SquareOperator<T>>(a, destination);

    internal static T Mean<T>(ReadOnlyVectorSpan<T> x) where T : unmanaged, INumberBase<T>
    {
        T count = T.CreateChecked(x.Length);
        T mean = Sum(x) / count;
        if (T.IsFinite(mean) || x.Length == 0 || !(typeof(T) == typeof(double) || typeof(T) == typeof(float)))
        {
            return mean;
        }

        // 途中の和だけがオーバーフローした場合（有限の入力なら平均は表現できる）: 要素を 2^-k 倍して足し直す。
        // 2^k ≥ 2n なので縮尺した和はオーバーフローしない。入力に ∞・NaN があれば同じ ∞・NaN になる
        int k = 65 - BitOperations.LeadingZeroCount((ulong)x.Length);
        T down = T.CreateChecked(Math.ScaleB(1.0, -k));
        T up = T.CreateChecked(Math.ScaleB(1.0, k));
        return unchecked(SumOf<T, ScaledOperator<T>>(x, down) / count * up);
    }

    internal static void ColumnMeans<T>(ReadOnlyMatrixSpan<T> a, VectorSpan<T> destination) where T : unmanaged, INumberBase<T>
    {
        RequireLength(destination, a.Columns);
        RequireDisjoint(a, destination.AsMatrix());
        for (int j = 0; j < a.Columns; j++)
        {
            destination[j] = Mean(a.Column(j));
        }
    }

    internal static void RowMeans<T>(ReadOnlyMatrixSpan<T> a, VectorSpan<T> destination) where T : unmanaged, INumberBase<T>
    {
        RowSums(a, destination);
        T count = T.CreateChecked(a.Columns);
        for (int i = 0; i < a.Rows; i++)
        {
            destination[i] /= count;
            // 和がオーバーフローした行は Mean と同じく縮尺して求め直す（オーバーフローしない行は Mean と同じビット）
            if (!T.IsFinite(destination[i]))
            {
                destination[i] = Mean(a.Row(i));
            }
        }
    }

    // 補正付き 2 パス法による分散・標準偏差。オーバーフローや非正規化数領域に近づく場合のみ 2 のべき乗でスケーリングして再計算し、
    // 結果を v * 2^(2e) で保持することで標準偏差のオーバーフロー回避と精度維持を両立する（Half は float に拡張して計算）。

    internal static T Variance<T>(ReadOnlyVectorSpan<T> x, int correction) where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (typeof(T) == typeof(Half))
        {
            return WidenedDispersion(x, correction, standardDeviation: false);
        }

        T variance = Variance(x, correction, out int exponent);
        return exponent == 0 ? variance : T.ScaleB(variance, 2 * exponent);
    }

    internal static T StandardDeviation<T>(ReadOnlyVectorSpan<T> x, int correction) where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (typeof(T) == typeof(Half))
        {
            return WidenedDispersion(x, correction, standardDeviation: true);
        }

        T variance = Variance(x, correction, out int exponent);
        return exponent == 0 ? T.Sqrt(variance) : T.ScaleB(T.Sqrt(variance), exponent);
    }

    // Half は指数範囲が狭く大きな n で平方和が 65504 を超えるため、float に拡張して計算し最後に 1 回だけ Half へ丸める。
    private static T WidenedDispersion<T>(ReadOnlyVectorSpan<T> x, int correction, bool standardDeviation) where T : unmanaged, IFloatingPointIeee754<T>
    {
        ArgumentOutOfRangeException.ThrowIfNegative(correction);
        var view = Widen(x, out float[] rented);
        try
        {
            return Narrow<T>(standardDeviation ? StandardDeviation(view, correction) : Variance(view, correction));
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rented);
        }
    }

    private static ReadOnlyVectorSpan<float> Widen<T>(ReadOnlyVectorSpan<T> x, out float[] rented) where T : unmanaged, INumberBase<T>
    {
        rented = ArrayPool<float>.Shared.Rent(Math.Max(1, x.Length));
        var widened = rented.AsSpan(0, x.Length);
        if (x.Length <= 1 || x.Stride == 1)
        {
            HalfConversion.Widen(HalfSpan(in x.Reference, x.Length), widened);
        }
        else
        {
            for (int i = 0; i < x.Length; i++)
            {
                widened[i] = (float)Unsafe.BitCast<T, Half>(x.ElementAt(i));
            }
        }

        return new ReadOnlyVectorSpan<float>(widened);
    }

    private static MatrixSpan<float> Widen<T>(ReadOnlyMatrixSpan<T> a, out float[] rented) where T : unmanaged, INumberBase<T>
    {
        rented = Rent<float>(a.Rows, a.Columns);
        var widened = new MatrixSpan<float>(rented.AsSpan(0, a.Rows * a.Columns), a.Rows, a.Columns);
        for (int j = 0; j < a.Columns; j++)
        {
            var column = rented.AsSpan(j * a.Rows, a.Rows);
            if (a.Rows <= 1 || a.RowStride == 1)
            {
                HalfConversion.Widen(HalfSpan(in a.ElementAt(0, j), a.Rows), column);
            }
            else
            {
                for (int i = 0; i < a.Rows; i++)
                {
                    column[i] = (float)Unsafe.BitCast<T, Half>(a.ElementAt(i, j));
                }
            }
        }

        return widened;
    }

    private static void NarrowTo<T>(ReadOnlyMatrixSpan<float> x, MatrixSpan<T> destination) where T : unmanaged, INumberBase<T>
    {
        for (int j = 0; j < x.Columns; j++)
        {
            var column = MemoryMarshal.CreateReadOnlySpan(in x.ElementAt(0, j), x.Rows);
            if (x.Rows <= 1 || destination.AsReadOnly().RowStride == 1)
            {
                HalfConversion.Narrow(column, MemoryMarshal.Cast<T, Half>(MemoryMarshal.CreateSpan(ref Element(destination, 0, j), x.Rows)));
            }
            else
            {
                for (int i = 0; i < x.Rows; i++)
                {
                    Element(destination, i, j) = Narrow<T>(column[i]);
                }
            }
        }
    }

    // T が Half のときだけ呼ぶ
    private static ReadOnlySpan<Half> HalfSpan<T>(ref readonly T reference, int length) where T : unmanaged =>
        MemoryMarshal.CreateReadOnlySpan(in Unsafe.As<T, Half>(ref Unsafe.AsRef(in reference)), length);

    private static T Narrow<T>(float x) where T : unmanaged => Unsafe.BitCast<Half, T>((Half)x);

    // 分散を (戻り値) * 2^(2 * exponent) として返す（非スケーリング時は exponent = 0）。
    private static T Variance<T>(ReadOnlyVectorSpan<T> x, int correction, out int exponent) where T : unmanaged, IFloatingPointIeee754<T>
    {
        ArgumentOutOfRangeException.ThrowIfNegative(correction);
        exponent = 0;
        int n = x.Length;
        if (n <= correction)
        {
            return T.NaN;
        }

        T count = T.CreateChecked(n);
        T denominator = T.CreateChecked(n - correction);
        T mean = Mean(x);
        T squares = SumOf<T, SquaredDeviationOperator<T>>(x, mean);
        if (NeedsRescaling(squares, mean, n) && TryRescaledVariance(x, count, denominator, out T variance, out exponent))
        {
            return variance;
        }

        T deviations = SumOf<T, DeviationOperator<T>>(x, mean);
        return CorrectedVariance(squares, deviations, count, denominator);
    }

    // T の仮数桁数 p と最小正規化指数（double: 53/-1022, float: 24/-126, Half: 11/-14）。
    private static class FloatingPointFormat<T> where T : IFloatingPointIeee754<T>
    {
        internal static readonly int Precision = 1 - T.ILogB(T.BitIncrement(T.One) - T.One);

        // 最小の非正規化数を 2^(p-1) 倍した最小の正規化数の指数（非正規化数の ILogB には頼らない。Exponent の注）
        internal static readonly int MinExponent = T.ILogB(T.Epsilon * T.ScaleB(T.One, Precision - 1));
    }

    // 0 でない有限の値の指数（T.ILogB と同じ値）。非正規化数は 2^(p-1) 倍した正規化数の指数から求める
    // （.NET 10 の Half.ILogB は非正規化数で誤った値を返す。Half.ILogB(Half.Epsilon) が -24 でなく -40）
    private static int Exponent<T>(T x) where T : IFloatingPointIeee754<T>
    {
        if (!T.IsSubnormal(x))
        {
            return T.ILogB(x);
        }

        int shift = FloatingPointFormat<T>.Precision - 1;
        return T.ILogB(x * T.ScaleB(T.One, shift)) - shift;
    }

    // 縮尺して求め直すか。平均が有限でない（入力に ∞・NaN があるか、double・float 以外で和がオーバーフローした）か、偏差の平方和がオーバーフローしたか、
    // 平方和が 2^(最小の指数 + p + ⌊log₂n⌋ + 1) 未満のとき。それ以上なら、非正規化数に落ちた項の丸め（合わせて n·2^(最小の指数 - p) 以下）は
    // 平方和の 2^-2p 倍以下で、分散 = 平方和 / (n - correction) も正規化数の範囲に p 桁以上の余裕がある。
    // ただし平方和がちょうど 0 で |平均| ≥ 2^(p + 最小の指数/2) なら、偏差はすべて 0 なので縮尺しない
    // （平均に近い数どうしの差は 0 か、ulp(平均/2) ≥ 2^(最小の指数/2 + 1) 以上で、平方が 0 に丸まらない）
    private static bool NeedsRescaling<T>(T squares, T mean, int n) where T : IFloatingPointIeee754<T>
    {
        if (!T.IsFinite(mean) || !T.IsFinite(squares))
        {
            return true;
        }

        int precision = FloatingPointFormat<T>.Precision, minimum = FloatingPointFormat<T>.MinExponent;
        if (Exponent(squares) >= minimum + precision + (32 - BitOperations.LeadingZeroCount((uint)n)))
        {
            return false;
        }

        return !(squares == T.Zero && Exponent(mean) >= precision + minimum / 2);
    }

    // 入力を 2^-e₁ 倍して最大の絶対値を [1, 2) にそろえ、その平均からの偏差をさらに 2^-e₂ 倍して最大の絶対値を [1, 2) にそろえてから、
    // 補正つき2パス法の平方和と偏差の和をとる（縮尺した偏差の平方和は 1 以上 16n 未満で、double・float ではオーバーフローも非正規化数への落ち込みもない。Half は float に広げて求める）。
    // 2 の冪の縮尺は、拡大では丸めず、縮小では非正規化数に落ちる小さな値だけを丸める（最大の絶対値の 2^-p 倍未満の誤差で、分散には効かない）。
    // 分散は 戻り値·2^(2(e₁ + e₂))。入力に ∞・NaN があれば縮尺しても変わらないので false を返す
    private static bool TryRescaledVariance<T>(ReadOnlyVectorSpan<T> x, T count, T denominator, out T variance, out int exponent)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        variance = T.Zero;
        exponent = 0;
        T largest = NormInfinity(x);
        if (!T.IsFinite(largest))
        {
            return false;
        }

        if (largest == T.Zero)
        {
            return true;
        }

        int n = x.Length;
        T[] rented = Rent<T>(n, 1);
        try
        {
            var y = rented.AsSpan(0, n);
            int first = Exponent(largest);
            MultiplyByPowerOfTwo(x, -first, y);
            var scaled = new ReadOnlyVectorSpan<T>(y);
            T mean = Mean(scaled);
            T spread = T.Zero;
            for (int i = 0; i < n; i++)
            {
                y[i] -= mean;
                spread = T.Max(spread, T.Abs(y[i]));
            }

            // 縮尺した値がすべて平均に等しい（入力の値がすべて等しい）
            if (spread == T.Zero)
            {
                return true;
            }

            int second = Exponent(spread);
            MultiplyByPowerOfTwo(scaled, -second, y);
            variance = CorrectedVariance(SumOfSquares(scaled), Sum(scaled), count, denominator);
            exponent = first + second;
            return true;
        }
        finally
        {
            ArrayPool<T>.Shared.Return(rented);
        }
    }

    // destination[i] = x[i]·2^k（2 の冪を掛けるのは ScaleB と同じく1回の丸め）。
    // 2^k が表せない（非正規化数の最大値をそろえる 2^1074 など）ときは、表せる2つの冪に分けて掛ける（拡大なので途中で丸めない）
    private static void MultiplyByPowerOfTwo<T>(ReadOnlyVectorSpan<T> x, int k, Span<T> destination) where T : unmanaged, IFloatingPointIeee754<T>
    {
        T factor = T.ScaleB(T.One, k);
        if (T.IsFinite(factor))
        {
            for (int i = 0; i < destination.Length; i++)
            {
                destination[i] = x.ElementAt(i) * factor;
            }

            return;
        }

        T low = T.ScaleB(T.One, k / 2), high = T.ScaleB(T.One, k - k / 2);
        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = x.ElementAt(i) * low * high;
        }
    }

    // Chan, Golub & LeVeque (1983) の補正: Σ(x - m)² - (Σ(x - m))²/n。数学的には負にならないので、丸めで負になれば 0 にする
    private static T CorrectedVariance<T>(T squares, T deviations, T count, T denominator) where T : IFloatingPointIeee754<T>
    {
        T correction = deviations * deviations / count;
        // (Σ(x - m))² だけがオーバーフローした場合（|平均| が 1e160 程度を超えるとき）は先に n で割る。数学的に (Σ(x - m))²/n ≤ Σ(x - m)² なのでオーバーフローしない
        if (T.IsPositiveInfinity(correction))
        {
            correction = deviations / count * deviations;
        }

        T sum = squares - correction;
        return (sum < T.Zero ? T.Zero : sum) / denominator;
    }

    internal static void ColumnVariances<T>(ReadOnlyMatrixSpan<T> a, int correction, VectorSpan<T> destination) where T : unmanaged, IFloatingPointIeee754<T>
    {
        ArgumentOutOfRangeException.ThrowIfNegative(correction);
        RequireLength(destination, a.Columns);
        RequireDisjoint(a, destination.AsMatrix());
        for (int j = 0; j < a.Columns; j++)
        {
            destination[j] = Variance(a.Column(j), correction);
        }
    }

    internal static void RowVariances<T>(ReadOnlyMatrixSpan<T> a, int correction, VectorSpan<T> destination) where T : unmanaged, IFloatingPointIeee754<T> =>
        RowVariances(a, correction, destination, standardDeviation: false);

    internal static void ColumnStandardDeviations<T>(ReadOnlyMatrixSpan<T> a, int correction, VectorSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        ArgumentOutOfRangeException.ThrowIfNegative(correction);
        RequireLength(destination, a.Columns);
        RequireDisjoint(a, destination.AsMatrix());
        for (int j = 0; j < a.Columns; j++)
        {
            destination[j] = StandardDeviation(a.Column(j), correction);
        }
    }

    internal static void RowStandardDeviations<T>(ReadOnlyMatrixSpan<T> a, int correction, VectorSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T> =>
        RowVariances(a, correction, destination, standardDeviation: true);

    // 各行の分散（standardDeviation なら標準偏差）。縮尺が要る行は Variance・StandardDeviation で求め直す（縮尺しない行も同じビット）
    private static void RowVariances<T>(ReadOnlyMatrixSpan<T> a, int correction, VectorSpan<T> destination, bool standardDeviation)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        ArgumentOutOfRangeException.ThrowIfNegative(correction);
        RequireLength(destination, a.Rows);
        RequireDisjoint(a, destination.AsMatrix());
        int rows = a.Rows, n = a.Columns;
        if (typeof(T) == typeof(Half))
        {
            WidenedRows(a, destination, standardDeviation ? WidenedRowReduction.StandardDeviation : WidenedRowReduction.Variance, correction);
            return;
        }

        if (n <= correction)
        {
            destination.Fill(T.NaN);
            return;
        }

        T[] rented = Rent<T>(rows, 3);
        try
        {
            var means = rented.AsSpan(0, rows);
            var squares = rented.AsSpan(rows, rows);
            var deviations = rented.AsSpan(2 * rows, rows);
            T count = T.CreateChecked(n);
            RowSumsOf<T, FirstOperator<T>>(a, default, means);
            for (int i = 0; i < rows; i++)
            {
                means[i] /= count;
            }

            RowSumsOf<T, SquaredDeviationOperator<T>>(a, means, squares);
            RowSumsOf<T, DeviationOperator<T>>(a, means, deviations);
            T denominator = T.CreateChecked(n - correction);
            for (int i = 0; i < rows; i++)
            {
                if (NeedsRescaling(squares[i], means[i], n))
                {
                    destination[i] = standardDeviation ? StandardDeviation(a.Row(i), correction) : Variance(a.Row(i), correction);
                }
                else
                {
                    T variance = CorrectedVariance(squares[i], deviations[i], count, denominator);
                    destination[i] = standardDeviation ? T.Sqrt(variance) : variance;
                }
            }
        }
        finally
        {
            ArrayPool<T>.Shared.Return(rented);
        }
    }

    // ---------- 最大・最小 ----------

    private static T Extremum<T, TOperator>(ReadOnlyVectorSpan<T> x)
        where T : unmanaged, INumber<T>
        where TOperator : IBinaryOperator<T>
    {
        RequireNotEmpty(x.Length, nameof(x));
        if (DenseReductions.TryExtremum<T, TOperator>(x, out T extremum))
        {
            return extremum;
        }

        T result = x.ElementAt(0);
        for (int i = 1; i < x.Length; i++)
        {
            result = TOperator.Invoke(result, x.ElementAt(i));
        }

        return result;
    }

    private static void ColumnExtrema<T, TOperator>(ReadOnlyMatrixSpan<T> a, VectorSpan<T> destination)
        where T : unmanaged, INumber<T>
        where TOperator : IBinaryOperator<T>
    {
        RequireLength(destination, a.Columns);
        RequireDisjoint(a, destination.AsMatrix());
        if (a.Columns > 0)
        {
            RequireNotEmpty(a.Rows, "x");
        }

        for (int j = 0; j < a.Columns; j++)
        {
            destination[j] = Extremum<T, TOperator>(a.Column(j));
        }
    }

    private static void RowExtrema<T, TOperator>(ReadOnlyMatrixSpan<T> a, Span<T> result)
        where T : unmanaged, INumber<T>
        where TOperator : IBinaryOperator<T>
    {
        if (a.Rows > 0)
        {
            RequireNotEmpty(a.Columns, "x");
        }

        if (DenseReductions.TryRowExtrema<T, TOperator>(a, result[..a.Rows]))
        {
            return;
        }

        for (int i = 0; i < a.Rows; i++)
        {
            result[i] = a.ElementAt(i, 0);
        }

        for (int j = 1; j < a.Columns; j++)
        {
            for (int i = 0; i < a.Rows; i++)
            {
                result[i] = TOperator.Invoke(result[i], a.ElementAt(i, j));
            }
        }
    }

    private static void RowExtrema<T, TOperator>(ReadOnlyMatrixSpan<T> a, VectorSpan<T> destination)
        where T : unmanaged, INumber<T>
        where TOperator : IBinaryOperator<T>
    {
        RequireLength(destination, a.Rows);
        RequireDisjoint(a, destination.AsMatrix());
        T[] rented = Rent<T>(a.Rows, 1);
        try
        {
            var result = rented.AsSpan(0, a.Rows);
            RowExtrema<T, TOperator>(a, result);
            WriteTo<T>(result, destination);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(rented);
        }
    }

    internal static T Max<T>(ReadOnlyVectorSpan<T> x) where T : unmanaged, INumber<T> => Extremum<T, MaximumOperator<T>>(x);

    internal static T Min<T>(ReadOnlyVectorSpan<T> x) where T : unmanaged, INumber<T> => Extremum<T, MinimumOperator<T>>(x);

    internal static void ColumnMax<T>(ReadOnlyMatrixSpan<T> a, VectorSpan<T> destination) where T : unmanaged, INumber<T> =>
        ColumnExtrema<T, MaximumOperator<T>>(a, destination);

    internal static void ColumnMin<T>(ReadOnlyMatrixSpan<T> a, VectorSpan<T> destination) where T : unmanaged, INumber<T> =>
        ColumnExtrema<T, MinimumOperator<T>>(a, destination);

    internal static void RowMax<T>(ReadOnlyMatrixSpan<T> a, VectorSpan<T> destination) where T : unmanaged, INumber<T> =>
        RowExtrema<T, MaximumOperator<T>>(a, destination);

    internal static void RowMin<T>(ReadOnlyMatrixSpan<T> a, VectorSpan<T> destination) where T : unmanaged, INumber<T> =>
        RowExtrema<T, MinimumOperator<T>>(a, destination);

    // ---------- NaN を除く和・平均・最大・最小 ----------

    // NaN を +0 に置き換えて、同じ木で足す（NaN を飛ばして足すのと同じ値。ブロックの区切りは元の添字のまま）
    internal static T SumIgnoringNaN<T>(ReadOnlyVectorSpan<T> x) where T : unmanaged, INumberBase<T> =>
        SumOf<T, IgnoreNaNOperator<T>>(x, T.Zero);

    internal static void ColumnSumsIgnoringNaN<T>(ReadOnlyMatrixSpan<T> a, VectorSpan<T> destination) where T : unmanaged, INumberBase<T>
    {
        RequireLength(destination, a.Columns);
        RequireDisjoint(a, destination.AsMatrix());
        for (int j = 0; j < a.Columns; j++)
        {
            destination[j] = SumIgnoringNaN(a.Column(j));
        }
    }

    internal static void RowSumsIgnoringNaN<T>(ReadOnlyMatrixSpan<T> a, VectorSpan<T> destination) where T : unmanaged, INumberBase<T> =>
        RowSumsOf<T, IgnoreNaNOperator<T>>(a, destination);

    private static int CountNaN<T>(ReadOnlyVectorSpan<T> x) where T : unmanaged, INumberBase<T>
    {
        if (DenseReductions.TryCountNaN(x, out int count))
        {
            return count;
        }

        for (int i = 0; i < x.Length; i++)
        {
            if (T.IsNaN(x.ElementAt(i)))
            {
                count++;
            }
        }

        return count;
    }

    // NaN を除く和を NaN でない要素の数で割る（すべて NaN か空なら NaN）。途中の和だけがオーバーフローした場合は Mean と同じく縮尺して足し直す
    internal static T MeanIgnoringNaN<T>(ReadOnlyVectorSpan<T> x) where T : unmanaged, INumberBase<T>
    {
        int n = x.Length - CountNaN(x);
        T count = T.CreateChecked(n);
        T mean = SumIgnoringNaN(x) / count;
        if (T.IsFinite(mean) || n == 0 || !(typeof(T) == typeof(double) || typeof(T) == typeof(float)))
        {
            return mean;
        }

        int k = 65 - BitOperations.LeadingZeroCount((ulong)n);
        T down = T.CreateChecked(Math.ScaleB(1.0, -k));
        T up = T.CreateChecked(Math.ScaleB(1.0, k));
        return unchecked(SumOf<T, ScaledIgnoreNaNOperator<T>>(x, down) / count * up);
    }

    internal static void ColumnMeansIgnoringNaN<T>(ReadOnlyMatrixSpan<T> a, VectorSpan<T> destination) where T : unmanaged, INumberBase<T>
    {
        RequireLength(destination, a.Columns);
        RequireDisjoint(a, destination.AsMatrix());
        for (int j = 0; j < a.Columns; j++)
        {
            destination[j] = MeanIgnoringNaN(a.Column(j));
        }
    }

    internal static void RowMeansIgnoringNaN<T>(ReadOnlyMatrixSpan<T> a, VectorSpan<T> destination) where T : unmanaged, INumberBase<T>
    {
        RowSumsIgnoringNaN(a, destination);
        int rows = a.Rows;
        int[] rented = ArrayPool<int>.Shared.Rent(rows);
        try
        {
            var counts = rented.AsSpan(0, rows);
            counts.Clear();
            for (int j = 0; j < a.Columns; j++)
            {
                for (int i = 0; i < rows; i++)
                {
                    counts[i] += T.IsNaN(a.ElementAt(i, j)) ? 0 : 1;
                }
            }

            for (int i = 0; i < rows; i++)
            {
                destination[i] /= T.CreateChecked(counts[i]);
                // 和がオーバーフローした行は MeanIgnoringNaN と同じく縮尺して求め直す（オーバーフローしない行は MeanIgnoringNaN と同じビット）
                if (!T.IsFinite(destination[i]))
                {
                    destination[i] = MeanIgnoringNaN(a.Row(i));
                }
            }
        }
        finally
        {
            ArrayPool<int>.Shared.Return(rented);
        }
    }

    internal static T MaxIgnoringNaN<T>(ReadOnlyVectorSpan<T> x) where T : unmanaged, INumber<T> => Extremum<T, MaximumNumberOperator<T>>(x);

    internal static T MinIgnoringNaN<T>(ReadOnlyVectorSpan<T> x) where T : unmanaged, INumber<T> => Extremum<T, MinimumNumberOperator<T>>(x);

    internal static void ColumnMaxIgnoringNaN<T>(ReadOnlyMatrixSpan<T> a, VectorSpan<T> destination) where T : unmanaged, INumber<T> =>
        ColumnExtrema<T, MaximumNumberOperator<T>>(a, destination);

    internal static void ColumnMinIgnoringNaN<T>(ReadOnlyMatrixSpan<T> a, VectorSpan<T> destination) where T : unmanaged, INumber<T> =>
        ColumnExtrema<T, MinimumNumberOperator<T>>(a, destination);

    internal static void RowMaxIgnoringNaN<T>(ReadOnlyMatrixSpan<T> a, VectorSpan<T> destination) where T : unmanaged, INumber<T> =>
        RowExtrema<T, MaximumNumberOperator<T>>(a, destination);

    internal static void RowMinIgnoringNaN<T>(ReadOnlyMatrixSpan<T> a, VectorSpan<T> destination) where T : unmanaged, INumber<T> =>
        RowExtrema<T, MinimumNumberOperator<T>>(a, destination);

    // ---------- 最大・最小の位置 ----------

    // 最大・最小の値を求めてから、同じ値の最初の位置を探す。NaN があれば最大・最小は NaN なので最初の NaN の位置になり、
    // ±0 は Max・Min と同じく +0 を -0 より大きいとする（値と符号が同じものを探す。decimal などでも演算子が -0 < +0 で選ぶ。SignedZeroOrder）
    private static bool IsSame<T>(T x, T value) where T : INumber<T> =>
        T.IsNaN(value) ? T.IsNaN(x) : x == value && T.IsNegative(x) == T.IsNegative(value);

    private static int IndexOfExtremum<T, TOperator>(ReadOnlyVectorSpan<T> x)
        where T : unmanaged, INumber<T>
        where TOperator : IBinaryOperator<T>
    {
        T value = Extremum<T, TOperator>(x);
        if (DenseReductions.TryIndexOf(x, value, out int index))
        {
            return index;
        }

        for (int i = 0; i < x.Length; i++)
        {
            if (IsSame(x.ElementAt(i), value))
            {
                return i;
            }
        }

        return -1;
    }

    private static (int Row, int Column) IndexOfExtremum<T, TOperator>(ReadOnlyMatrixSpan<T> x)
        where T : unmanaged, INumber<T>
        where TOperator : IBinaryOperator<T>
    {
        RequireNotEmpty(x.Rows, nameof(x));
        RequireNotEmpty(x.Columns, nameof(x));
        if (x.TryGetDense(out var data))
        {
            int index = IndexOfExtremum<T, TOperator>(new ReadOnlyVectorSpan<T>(data));
            return (index % x.Rows, index / x.Rows);
        }

        int bestRow = IndexOfExtremum<T, TOperator>(x.Column(0));
        int bestColumn = 0;
        T best = x.ElementAt(bestRow, bestColumn);
        for (int column = 1; column < x.Columns && !T.IsNaN(best); column++)
        {
            int row = IndexOfExtremum<T, TOperator>(x.Column(column));
            T candidate = x.ElementAt(row, column);
            if (!IsSame(best, TOperator.Invoke(best, candidate)))
            {
                best = candidate;
                bestRow = row;
                bestColumn = column;
            }
        }

        return (bestRow, bestColumn);
    }

    private static void RequireLength(Span<int> destination, int length)
    {
        if (destination.Length != length)
        {
            throw new ArgumentException("Destination length must agree.", nameof(destination));
        }
    }

    // 出力先が入力と重なっても（T が int のとき）よいよう、作業領域に求めてから書き写す
    private static void ColumnIndexOfExtrema<T, TOperator>(ReadOnlyMatrixSpan<T> a, Span<int> destination)
        where T : unmanaged, INumber<T>
        where TOperator : IBinaryOperator<T>
    {
        RequireLength(destination, a.Columns);
        if (a.Columns > 0)
        {
            RequireNotEmpty(a.Rows, "x");
        }

        int[] rented = ArrayPool<int>.Shared.Rent(a.Columns);
        try
        {
            var result = rented.AsSpan(0, a.Columns);
            for (int j = 0; j < a.Columns; j++)
            {
                result[j] = IndexOfExtremum<T, TOperator>(a.Column(j));
            }

            result.CopyTo(destination);
        }
        finally
        {
            ArrayPool<int>.Shared.Return(rented);
        }
    }

    private static void RowIndexOfExtrema<T, TOperator>(ReadOnlyMatrixSpan<T> a, Span<int> destination)
        where T : unmanaged, INumber<T>
        where TOperator : IBinaryOperator<T>
    {
        RequireLength(destination, a.Rows);
        int rows = a.Rows;
        T[] rented = Rent<T>(rows, 1);
        int[] found = ArrayPool<int>.Shared.Rent(rows);
        try
        {
            var extrema = rented.AsSpan(0, rows);
            var result = found.AsSpan(0, rows);
            RowExtrema<T, TOperator>(a, extrema);
            result.Fill(-1);
            int remaining = rows;
            for (int j = 0; j < a.Columns && remaining > 0; j++)
            {
                for (int i = 0; i < rows; i++)
                {
                    if (result[i] < 0 && IsSame(a.ElementAt(i, j), extrema[i]))
                    {
                        result[i] = j;
                        remaining--;
                    }
                }
            }

            result.CopyTo(destination);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(rented);
            ArrayPool<int>.Shared.Return(found);
        }
    }

    internal static int IndexOfMax<T>(ReadOnlyVectorSpan<T> x) where T : unmanaged, INumber<T> => IndexOfExtremum<T, MaximumOperator<T>>(x);

    internal static int IndexOfMin<T>(ReadOnlyVectorSpan<T> x) where T : unmanaged, INumber<T> => IndexOfExtremum<T, MinimumOperator<T>>(x);

    internal static (int Row, int Column) IndexOfMax<T>(ReadOnlyMatrixSpan<T> x) where T : unmanaged, INumber<T> =>
        IndexOfExtremum<T, MaximumOperator<T>>(x);

    internal static (int Row, int Column) IndexOfMin<T>(ReadOnlyMatrixSpan<T> x) where T : unmanaged, INumber<T> =>
        IndexOfExtremum<T, MinimumOperator<T>>(x);

    internal static void ColumnIndexOfMax<T>(ReadOnlyMatrixSpan<T> a, Span<int> destination) where T : unmanaged, INumber<T> =>
        ColumnIndexOfExtrema<T, MaximumOperator<T>>(a, destination);

    internal static void ColumnIndexOfMin<T>(ReadOnlyMatrixSpan<T> a, Span<int> destination) where T : unmanaged, INumber<T> =>
        ColumnIndexOfExtrema<T, MinimumOperator<T>>(a, destination);

    internal static void RowIndexOfMax<T>(ReadOnlyMatrixSpan<T> a, Span<int> destination) where T : unmanaged, INumber<T> =>
        RowIndexOfExtrema<T, MaximumOperator<T>>(a, destination);

    internal static void RowIndexOfMin<T>(ReadOnlyMatrixSpan<T> a, Span<int> destination) where T : unmanaged, INumber<T> =>
        RowIndexOfExtrema<T, MinimumOperator<T>>(a, destination);

    // ---------- 内積・ノルム ----------

    internal static T Dot<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y) where T : unmanaged, INumberBase<T>
    {
        if (x.Length != y.Length)
        {
            throw new ArgumentException("Vector lengths must agree.");
        }

        return DenseReductions.TryDot(x, y, out T sum) ? sum : PairwiseSum<T, ProductTerm<T>>(x.Length, new ProductTerm<T>(x, y));
    }

    internal static T Norm1<T>(ReadOnlyVectorSpan<T> x) where T : unmanaged, INumberBase<T> =>
        SumOf<T, AbsoluteOperator<T>>(x, T.Zero);

    internal static T NormInfinity<T>(ReadOnlyVectorSpan<T> x) where T : unmanaged, INumber<T>
    {
        if (DenseReductions.TryNormInfinity(x, out T norm))
        {
            return norm;
        }

        T result = T.Zero;
        for (int i = 0; i < x.Length; i++)
        {
            result = T.Max(result, T.Abs(x.ElementAt(i)));
        }

        return result;
    }

    // 最大絶対値（NaN があれば NaN）
    private static T LargestMagnitude<T>(T largest, T x) where T : IFloatingPointIeee754<T>
    {
        T magnitude = T.Abs(x);
        return T.IsNaN(magnitude) || magnitude > largest ? magnitude : largest;
    }

    // ユークリッドノルム。最大絶対値で 2 のべき乗スケーリングしてオーバーフロー・アンダーフローを回避（Half は float 拡張経路）。
    internal static T Norm<T>(ReadOnlyVectorSpan<T> x) where T : unmanaged, IFloatingPointIeee754<T> =>
        typeof(T) == typeof(Half) ? WidenedReduction(x, logSumExp: false) : NormCore(x);

    private static T NormCore<T>(ReadOnlyVectorSpan<T> x) where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (!DenseReductions.TryLargestMagnitude(x, out T largest))
        {
            largest = T.Zero;
            for (int i = 0; i < x.Length; i++)
            {
                largest = LargestMagnitude(largest, x.ElementAt(i));
            }
        }

        if (!T.IsFinite(largest) || largest == T.Zero)
        {
            return largest;
        }

        int exponent = Exponent(largest);
        T sum = TryPowerOfTwo(-exponent, out T factor)
            ? SumOf<T, ScaledSquareByFactorOperator<T>>(x, factor)
            : SumOf<T, ScaledSquareOperator<T>>(x, T.CreateTruncating(-exponent));
        return T.ScaleB(T.Sqrt(sum), exponent);
    }

    internal static void ColumnNorms<T>(ReadOnlyMatrixSpan<T> a, VectorSpan<T> destination) where T : unmanaged, IFloatingPointIeee754<T>
    {
        RequireLength(destination, a.Columns);
        RequireDisjoint(a, destination.AsMatrix());
        for (int j = 0; j < a.Columns; j++)
        {
            destination[j] = Norm(a.Column(j));
        }
    }

    internal static void RowNorms<T>(ReadOnlyMatrixSpan<T> a, VectorSpan<T> destination) where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (typeof(T) == typeof(Half))
        {
            WidenedRowReductions(a, destination, logSumExp: false);
        }
        else
        {
            RowNormsCore(a, destination);
        }
    }

    private static void RowNormsCore<T>(ReadOnlyMatrixSpan<T> a, VectorSpan<T> destination) where T : unmanaged, IFloatingPointIeee754<T>
    {
        RequireLength(destination, a.Rows);
        RequireDisjoint(a, destination.AsMatrix());
        int rows = a.Rows;
        T[] rented = Rent<T>(rows, 4);
        try
        {
            var largest = rented.AsSpan(0, rows);
            var scales = rented.AsSpan(rows, rows);
            var sums = rented.AsSpan(2 * rows, rows);
            var factors = rented.AsSpan(3 * rows, rows);
            largest.Fill(T.Zero);
            for (int j = 0; j < a.Columns; j++)
            {
                for (int i = 0; i < rows; i++)
                {
                    largest[i] = LargestMagnitude(largest[i], a.ElementAt(i, j));
                }
            }

            bool multiply = true;
            for (int i = 0; i < rows; i++)
            {
                scales[i] = T.IsFinite(largest[i]) && largest[i] != T.Zero ? T.CreateTruncating(-Exponent(largest[i])) : T.Zero;
                multiply &= TryPowerOfTwo(int.CreateTruncating(scales[i]), out factors[i]);
            }

            if (multiply)
            {
                RowSumsOf<T, ScaledSquareByFactorOperator<T>>(a, factors, sums);
            }
            else
            {
                RowSumsOf<T, ScaledSquareOperator<T>>(a, scales, sums);
            }

            for (int i = 0; i < rows; i++)
            {
                destination[i] = !T.IsFinite(largest[i]) || largest[i] == T.Zero
                    ? largest[i]
                    : T.ScaleB(T.Sqrt(sums[i]), -int.CreateTruncating(scales[i]));
            }
        }
        finally
        {
            ArrayPool<T>.Shared.Return(rented);
        }
    }

    // ---------- 対数領域 ----------

    // ln Σ exp(x - max) は、最大値に等しい要素の個数 c と、残りの項の和 s から ln(1 + ((c - 1) + s)) として log1p で求める。
    // 1 + s を先に丸めると、最大の項が支配的（s ≪ 1）なときに s の精度が落ち、LogSumExp([0, -40]) が 4.2e-18 でなく 0 になる
    // （SciPy 1.15 以降の logsumexp と同じ）。T.LogP1 は ln(1 + x) をそのまま求めるので、XMath.Log1p と同じく double の log1p を通す。
    private static T LogOnePlusCountAndSum<T>(int count, T rest) where T : INumberBase<T> =>
        T.CreateTruncating(XMath.Special.Log1p((count - 1) + double.CreateTruncating(rest)));

    // ln Σ exp(x[i] - max)（max は有限の最大値）。
    private static T LogSumOfShiftedExp<T>(ReadOnlyVectorSpan<T> x, T max) where T : unmanaged, IFloatingPointIeee754<T>
    {
        // 連続な float・double は、最大値に等しい要素を指数の項を作る走査の中で数える
        if (!DenseReductions.TrySumOfShiftedExp(x, max, out T rest, out int count))
        {
            rest = SumOf<T, ShiftedExpOperator<T>>(x, max);
            for (int i = 0; i < x.Length; i++)
            {
                if (x.ElementAt(i) == max)
                {
                    count++;
                }
            }
        }

        return LogOnePlusCountAndSum(count, rest);
    }

    // 各行の ln Σ_j exp(a[i, j] - max[i])（最大値が非有限の行の値は未使用）。
    private static void RowLogSumsOfShiftedExp<T>(ReadOnlyMatrixSpan<T> a, ReadOnlySpan<T> max, Span<T> destination) where T : unmanaged, IFloatingPointIeee754<T>
    {
        int rows = a.Rows;
        int[] counts = ArrayPool<int>.Shared.Rent(rows);
        try
        {
            var count = counts.AsSpan(0, rows);
            count.Clear();
            if (!DenseReductions.TryRowSumsOfShiftedExp(a, max, destination, count))
            {
                RowSumsOf<T, ShiftedExpOperator<T>>(a, max, destination);
                for (int j = 0; j < a.Columns; j++)
                {
                    for (int i = 0; i < rows; i++)
                    {
                        if (a.ElementAt(i, j) == max[i])
                        {
                            count[i]++;
                        }
                    }
                }
            }

            for (int i = 0; i < rows; i++)
            {
                destination[i] = LogOnePlusCountAndSum(count[i], destination[i]);
            }
        }
        finally
        {
            ArrayPool<int>.Shared.Return(counts);
        }
    }

    // ln Σ exp(x[i])（空なら -∞、最大値が ±∞ または NaN ならその値。Half は float 拡張経路）。
    internal static T LogSumExp<T>(ReadOnlyVectorSpan<T> x) where T : unmanaged, IFloatingPointIeee754<T> =>
        typeof(T) == typeof(Half) ? WidenedReduction(x, logSumExp: true) : LogSumExpCore(x);

    private static T LogSumExpCore<T>(ReadOnlyVectorSpan<T> x) where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (x.Length == 0)
        {
            return T.NegativeInfinity;
        }

        T max = Max(x);
        return T.IsFinite(max) ? max + LogSumOfShiftedExp(x, max) : max;
    }

    internal static void ColumnLogSumExp<T>(ReadOnlyMatrixSpan<T> a, VectorSpan<T> destination) where T : unmanaged, IFloatingPointIeee754<T>
    {
        RequireLength(destination, a.Columns);
        RequireDisjoint(a, destination.AsMatrix());
        for (int j = 0; j < a.Columns; j++)
        {
            destination[j] = LogSumExp(a.Column(j));
        }
    }

    internal static void RowLogSumExp<T>(ReadOnlyMatrixSpan<T> a, VectorSpan<T> destination) where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (typeof(T) == typeof(Half))
        {
            WidenedRowReductions(a, destination, logSumExp: true);
        }
        else
        {
            RowLogSumExpCore(a, destination);
        }
    }

    private static void RowLogSumExpCore<T>(ReadOnlyMatrixSpan<T> a, VectorSpan<T> destination) where T : unmanaged, IFloatingPointIeee754<T>
    {
        RequireLength(destination, a.Rows);
        RequireDisjoint(a, destination.AsMatrix());
        T[] rented = Rent<T>(a.Rows, 2);
        try
        {
            var result = rented.AsSpan(0, a.Rows);
            var max = rented.AsSpan(a.Rows, a.Rows);
            if (a.Columns == 0)
            {
                result.Fill(T.NegativeInfinity);
            }
            else
            {
                RowExtrema<T, MaximumOperator<T>>(a, max);
                RowLogSumsOfShiftedExp(a, max, result);
                for (int i = 0; i < a.Rows; i++)
                {
                    result[i] = T.IsFinite(max[i]) ? max[i] + result[i] : max[i];
                }
            }

            WriteTo<T>(result, destination);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(rented);
        }
    }

    // 各列のソフトマックス（最大値シフトによりオーバーフローを回避）。
    internal static void ColumnSoftmax<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination)
        where T : unmanaged, INumber<T>, IExponentialFunctions<T>
    {
        if (typeof(T) == typeof(Half))
        {
            WidenedSoftmax(x, destination, rows: false, logarithm: false);
        }
        else
        {
            ColumnSoftmaxCore(x, destination);
        }
    }

    private static void ColumnSoftmaxCore<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination)
        where T : unmanaged, INumber<T>, IExponentialFunctions<T>
    {
        RequireSameShape(x, destination);
        RequireElementwiseAlias(x, destination);
        if (x.Rows == 0)
        {
            return;
        }

        for (int j = 0; j < x.Columns; j++)
        {
            if (DenseReductions.TrySoftmax(x.Column(j), destination.Column(j)))
            {
                continue;
            }

            T max = Max(x.Column(j));
            // その場更新でも、読んだ要素を同じ位置へ書き戻すだけなので順序は崩れない。
            for (int i = 0; i < x.Rows; i++)
            {
                Element(destination, i, j) = DenseElementwise.Exp(x.ElementAt(i, j) - max);
            }

            T sum = Sum(destination.Column(j).AsReadOnly());
            for (int i = 0; i < x.Rows; i++)
            {
                Element(destination, i, j) /= sum;
            }
        }
    }

    // 各行のソフトマックス（列優先に沿って走査し、行ごとの最大値と和を作業領域に保持）。
    internal static void RowSoftmax<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination)
        where T : unmanaged, INumber<T>, IExponentialFunctions<T>
    {
        if (typeof(T) == typeof(Half))
        {
            WidenedSoftmax(x, destination, rows: true, logarithm: false);
        }
        else
        {
            RowSoftmaxCore(x, destination);
        }
    }

    private static void RowSoftmaxCore<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination)
        where T : unmanaged, INumber<T>, IExponentialFunctions<T>
    {
        RequireSameShape(x, destination);
        RequireElementwiseAlias(x, destination);
        int rows = x.Rows, columns = x.Columns;
        if (rows == 0 || columns == 0)
        {
            return;
        }

        T[] rented = Rent<T>(rows, 2);
        try
        {
            var max = rented.AsSpan(0, rows);
            var sum = rented.AsSpan(rows, rows);
            RowExtrema<T, MaximumOperator<T>>(x, max);
            if (!DenseReductions.TrySubtractRows<T>(x, max, destination, exponentiate: true))
            {
                for (int j = 0; j < columns; j++)
                {
                    for (int i = 0; i < rows; i++)
                    {
                        Element(destination, i, j) = DenseElementwise.Exp(x.ElementAt(i, j) - max[i]);
                    }
                }
            }

            RowSumsOf<T, FirstOperator<T>>(destination.AsReadOnly(), default, sum);
            if (!DenseReductions.TryDivideRows<T>(destination, sum))
            {
                for (int j = 0; j < columns; j++)
                {
                    for (int i = 0; i < rows; i++)
                    {
                        Element(destination, i, j) /= sum[i];
                    }
                }
            }
        }
        finally
        {
            ArrayPool<T>.Shared.Return(rented);
        }
    }

    // 対数ソフトマックスは (x - max) - ln Σ exp(x - max) として計算し、max の桁での丸めによる平行移動不変性の崩れを防ぐ。
    // 最大値が ±∞ または NaN のときは ln Σ = 0 として x - max を返す。
    internal static void ColumnLogSoftmax<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination) where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (typeof(T) == typeof(Half))
        {
            WidenedSoftmax(x, destination, rows: false, logarithm: true);
        }
        else
        {
            ColumnLogSoftmaxCore(x, destination);
        }
    }

    private static void ColumnLogSoftmaxCore<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination) where T : unmanaged, IFloatingPointIeee754<T>
    {
        RequireSameShape(x, destination);
        RequireElementwiseAlias(x, destination);
        if (x.Rows == 0)
        {
            return;
        }

        for (int j = 0; j < x.Columns; j++)
        {
            var column = x.Column(j);
            T max = Max(column);
            T logSum = T.IsFinite(max) ? LogSumOfShiftedExp(column, max) : T.Zero;
            var output = destination.Column(j);
            if (DenseReductions.TrySubtract(column, max, logSum, output))
            {
                continue;
            }

            for (int i = 0; i < x.Rows; i++)
            {
                Element(destination, i, j) = (x.ElementAt(i, j) - max) - logSum;
            }
        }
    }

    internal static void RowLogSoftmax<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination) where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (typeof(T) == typeof(Half))
        {
            WidenedSoftmax(x, destination, rows: true, logarithm: true);
        }
        else
        {
            RowLogSoftmaxCore(x, destination);
        }
    }

    private static void RowLogSoftmaxCore<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination) where T : unmanaged, IFloatingPointIeee754<T>
    {
        RequireSameShape(x, destination);
        RequireElementwiseAlias(x, destination);
        int rows = x.Rows;
        if (rows == 0 || x.Columns == 0)
        {
            return;
        }

        T[] rented = Rent<T>(rows, 2);
        try
        {
            var max = rented.AsSpan(0, rows);
            var logSum = rented.AsSpan(rows, rows);
            RowExtrema<T, MaximumOperator<T>>(x, max);
            RowLogSumsOfShiftedExp(x, max, logSum);
            for (int i = 0; i < rows; i++)
            {
                if (!T.IsFinite(max[i]))
                {
                    logSum[i] = T.Zero;
                }
            }

            if (DenseReductions.TrySubtractRows<T>(x, max, logSum, destination))
            {
                return;
            }

            for (int j = 0; j < x.Columns; j++)
            {
                for (int i = 0; i < rows; i++)
                {
                    Element(destination, i, j) = (x.ElementAt(i, j) - max[i]) - logSum[i];
                }
            }
        }
        finally
        {
            ArrayPool<T>.Shared.Return(rented);
        }
    }

    // ---------- Half -> float 拡張経路 ----------
    // Half は要素数が多いと和が 65504 を超えるため float で縮約し最後に 1 回だけ丸める（他型の IL 膨張を防ぐため本体と分離）。
    private static T WidenedReduction<T>(ReadOnlyVectorSpan<T> x, bool logSumExp) where T : unmanaged, INumberBase<T>
    {
        var widened = Widen(x, out float[] rented);
        try
        {
            return Narrow<T>(logSumExp ? LogSumExp(widened) : Norm(widened));
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rented);
        }
    }

    private static void WidenedRowReductions<T>(ReadOnlyMatrixSpan<T> a, VectorSpan<T> destination, bool logSumExp) where T : unmanaged, INumberBase<T>
    {
        RequireLength(destination, a.Rows);
        RequireDisjoint(a, destination.AsMatrix());
        WidenedRows(a, destination, logSumExp ? WidenedRowReduction.LogSumExp : WidenedRowReduction.Norm, correction: 0);
    }

    private enum WidenedRowReduction { Norm, LogSumExp, Variance, StandardDeviation }

    // 行を写す単位（RowBlock 行 × 列数の float を作業領域に置く）
    private const int RowBlock = 64;

    // 行連続時は行単位、列優先時は RowBlock 行ブロック単位で Half->float 変換して縮約する（どちらも同一ビット結果）。
    private static void WidenedRows<T>(ReadOnlyMatrixSpan<T> a, VectorSpan<T> destination, WidenedRowReduction reduction, int correction)
        where T : unmanaged, INumberBase<T>
    {
        int rows = a.Rows, columns = a.Columns;
        if (columns <= 1 || a.ColumnStride == 1)
        {
            for (int i = 0; i < rows; i++)
            {
                var row = Widen(a.Row(i), out float[] rented);
                try
                {
                    destination[i] = Narrow<T>(Reduce(row, reduction, correction));
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(rented);
                }
            }

            return;
        }

        float[] block = Rent<float>(Math.Min(RowBlock, rows), columns);
        try
        {
            Span<float> column = stackalloc float[RowBlock];
            ref float rowMajor = ref MemoryMarshal.GetArrayDataReference(block);
            // 処理した行数だけ進める（first += RowBlock は、行数が int の上限に近いと負へ折り返す）
            for (int first = 0, count; first < rows; first += count)
            {
                count = Math.Min(RowBlock, rows - first);
                for (int j = 0; j < columns; j++)
                {
                    if (count == 1 || a.RowStride == 1)
                    {
                        HalfConversion.Widen(HalfSpan(in a.ElementAt(first, j), count), column[..count]);
                    }
                    else
                    {
                        for (int r = 0; r < count; r++)
                        {
                            column[r] = (float)Unsafe.BitCast<T, Half>(a.ElementAt(first + r, j));
                        }
                    }
                    // block の大きさは count·columns 以上（Rent）
                    for (int r = 0; r < count; r++)
                    {
                        Unsafe.Add(ref rowMajor, (nint)r * columns + j) = column[r];
                    }
                }

                for (int r = 0; r < count; r++)
                {
                    var row = new ReadOnlyVectorSpan<float>(block.AsSpan(r * columns, columns));
                    destination[first + r] = Narrow<T>(Reduce(row, reduction, correction));
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(block);
        }

        static float Reduce(ReadOnlyVectorSpan<float> row, WidenedRowReduction reduction, int correction) => reduction switch
        {
            WidenedRowReduction.Norm => Norm(row),
            WidenedRowReduction.LogSumExp => LogSumExp(row),
            WidenedRowReduction.Variance => Variance(row, correction),
            _ => StandardDeviation(row, correction),
        };
    }

    // 行列全体を float バッファへ展開してから計算する（インプレース安全）。
    private static void WidenedSoftmax<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination, bool rows, bool logarithm)
        where T : unmanaged, INumberBase<T>
    {
        RequireSameShape(x, destination);
        RequireElementwiseAlias(x, destination);
        if (x.Rows == 0 || x.Columns == 0)
        {
            return;
        }

        var widened = Widen(x, out float[] rented);
        try
        {
            var input = widened.AsReadOnly();
            if (logarithm)
            {
                if (rows)
                {
                    RowLogSoftmax(input, widened);
                }
                else
                {
                    ColumnLogSoftmax(input, widened);
                }
            }
            else
            {
                if (rows)
                {
                    RowSoftmax(input, widened);
                }
                else
                {
                    ColumnSoftmax(input, widened);
                }
            }

            NarrowTo(input, destination);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rented);
        }
    }
}
