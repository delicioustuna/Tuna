using System.Buffers;
using System.Numerics;
using Tuna.LinearAlgebra;
using Tuna.RandomSource;
using Simd = System.Numerics.Vector;
using SimdVector = System.Numerics.Vector<ulong>;

namespace Tuna.QuasiRandom;

/// <summary>
/// Represents a Sobol' low-discrepancy point sequence, either unscrambled or randomized via linear matrix scrambling and digital shifting.
/// <para>非スクランブル、または線形行列スクランブルとデジタルシフトによりランダム化された Sobol' 低食い違い量点列を表します。</para>
/// </summary>
/// <remarks>
/// <para>Supports up to <see cref="MaxDimension"/> dimensions and <c>2⁶⁴</c> points (indices <c>0</c> through <c>ulong.MaxValue</c>) in Gray-code order. Coordinates are mapped to <c>[2⁻⁵³, 1 - 2⁻⁵³]</c> and never equal <c>0</c> or <c>1</c>, so even the unscrambled point at index <c>0</c> can be passed directly to inverse CDF functions.</para>
/// <para>For optimal balance properties, use a power-of-two number of points starting from index <c>0</c> without discarding the first point. Creating multiple independently scrambled instances allows estimating integration error from the sample variance of the estimates. Instances are immutable after construction and thread-safe.</para>
/// <para>最大 <see cref="MaxDimension"/> 次元、最大 <c>2⁶⁴</c> 点（インデックス <c>0</c> から <c>ulong.MaxValue</c>）を Gray code 順で生成します。各座標は <c>[2⁻⁵³, 1 - 2⁻⁵³]</c> に写像され <c>0</c> および <c>1</c> を取らないため、非スクランブルの第 0 点もそのまま分布の逆関数（<c>InverseCdf</c>）へ渡せます。</para>
/// <para>均等分布性を保つため、点の個数は 2 のべき乗とし、第 0 点を捨てずに先頭から使用してください。独立にスクランブルした複数の列で推定を繰り返すことで、推定値の標本分散から積分誤差を見積もれます。構築後は不変であり、すべてのメソッドはスレッドセーフです。</para>
/// </remarks>
public sealed class SobolSequence : IPointSequence
{
    /// <summary>
    /// The maximum number of dimensions supported by <see cref="SobolSequence"/>.
    /// <para><see cref="SobolSequence"/> がサポートする最大次元数。</para>
    /// </summary>
    public const int MaxDimension = SobolDirectionNumbers.Count;

    private const int Digits = SobolDirectionNumbers.Digits;
    private const int StackLimit = 256;
    private const double Unit = 1.0 / (1UL << 52);
    private const double HalfUnit = 0.5 / (1UL << 52);

    // directions[k * Dimension + j] に第 j 次元の方向数 v_(k+1) を配置し、Gray code 遷移時に同一 k の全次元を連続アクセスできるようにする。
    private readonly ulong[] directions;
    private readonly ulong[] shifts;

    /// <summary>
    /// Initializes a new unscrambled <see cref="SobolSequence"/> with the specified dimensionality.
    /// <para>指定した次元数の非スクランブル Sobol' 点列を初期化します。</para>
    /// </summary>
    /// <param name="dimension">The number of dimensions (between 1 and <see cref="MaxDimension"/>, inclusive). / 次元数（1 以上 <see cref="MaxDimension"/> 以下）。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="dimension"/> is less than 1 or greater than <see cref="MaxDimension"/>. / <paramref name="dimension"/> が 1 未満または <see cref="MaxDimension"/> より大きい場合。</exception>
    public SobolSequence(int dimension)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(dimension, 1);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(dimension, MaxDimension);
        Dimension = dimension;
        directions = new ulong[dimension * Digits];
        SobolDirectionNumbers.Fill(dimension, directions);
        shifts = new ulong[dimension];
    }

    /// <summary>
    /// Initializes a new randomized <see cref="SobolSequence"/> using linear matrix scrambling and a random digital shift.
    /// <para>線形行列スクランブルとランダムなデジタルシフトによりランダム化された Sobol' 点列を初期化します。</para>
    /// </summary>
    /// <param name="dimension">The number of dimensions (between 1 and <see cref="MaxDimension"/>, inclusive). / 次元数（1 以上 <see cref="MaxDimension"/> 以下）。</param>
    /// <param name="random">The random source used to generate the scrambling matrices and digital shifts. / スクランブル行列およびデジタルシフトの生成に使用する乱数源。</param>
    /// <remarks>
    /// <para>Random sources in the same state produce identical scrambled sequences.</para>
    /// <para>同一の状態にある乱数源からは同一のスクランブル点列が生成されます。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="dimension"/> is less than 1 or greater than <see cref="MaxDimension"/>. / <paramref name="dimension"/> が 1 未満または <see cref="MaxDimension"/> より大きい場合。</exception>
    /// <exception cref="ArgumentNullException"><paramref name="random"/> is <see langword="null"/>. / <paramref name="random"/> が <see langword="null"/> の場合。</exception>
    public SobolSequence(int dimension, IRandomSource random) : this(dimension)
    {
        ArgumentNullException.ThrowIfNull(random);
        Span<ulong> rows = stackalloc ulong[Digits];
        Span<ulong> column = stackalloc ulong[Digits];
        for (int j = 0; j < dimension; j++)
        {
            // Matoušek (1998) の線形行列スクランブル: 第 r 行は対角ビット 1 と上位 r ビットのランダム係数で構成する。
            rows[0] = 1UL << (Digits - 1);
            for (int r = 1; r < Digits; r++)
            {
                rows[r] = (random.NextUInt64() & (ulong.MaxValue << (Digits - r))) | (1UL << (Digits - 1 - r));
            }

            shifts[j] = random.NextUInt64();

            for (int k = 0; k < Digits; k++)
            {
                column[k] = directions[k * dimension + j];
            }

            for (int k = 0; k < Digits; k++)
            {
                ulong scrambled = 0;
                for (int r = 0; r < Digits; r++)
                {
                    scrambled |= (ulong)(BitOperations.PopCount(column[k] & rows[r]) & 1) << (Digits - 1 - r);
                }

                directions[k * dimension + j] = scrambled;
            }
        }

        IsScrambled = true;
    }

    /// <inheritdoc/>
    public int Dimension { get; }

    /// <summary>
    /// Gets a value indicating whether this sequence is randomized by scrambling and digital shifting.
    /// <para>この点列がスクランブルおよびデジタルシフトによりランダム化されているかどうかを示す値を取得します。</para>
    /// </summary>
    public bool IsScrambled { get; }

    /// <inheritdoc/>
    public void GetPoint(ulong index, Span<double> destination)
    {
        if (destination.Length != Dimension)
        {
            throw new ArgumentException("The destination length must equal the dimension.", nameof(destination));
        }

        ulong[]? rented = null;
        Span<ulong> state = Dimension <= StackLimit ? stackalloc ulong[Dimension] : (rented = ArrayPool<ulong>.Shared.Rent(Dimension)).AsSpan(0, Dimension);
        try
        {
            Start(index, state);
            for (int j = 0; j < Dimension; j++)
            {
                destination[j] = ToDouble(state[j]);
            }
        }
        finally
        {
            if (rented is not null)
            {
                ArrayPool<ulong>.Shared.Return(rented);
            }
        }
    }

    /// <inheritdoc/>
    public MatrixSpan<double> GetPoints(ulong start, MatrixSpan<double> destination)
    {
        if (destination.Rows != Dimension)
        {
            throw new ArgumentException("The destination must have one row per dimension.", nameof(destination));
        }

        int count = destination.Columns;
        if (count == 0)
        {
            return destination;
        }

        if ((ulong)(count - 1) > ulong.MaxValue - start)
        {
            throw new ArgumentOutOfRangeException(nameof(start), "The points exceed the 2^64 points of the sequence.");
        }

        ulong[]? rented = null;
        Span<ulong> state = Dimension <= StackLimit ? stackalloc ulong[Dimension] : (rented = ArrayPool<ulong>.Shared.Rent(Dimension)).AsSpan(0, Dimension);
        try
        {
            Start(start, state);
            for (int column = 0; ; column++)
            {
                for (int j = 0; j < Dimension; j++)
                {
                    destination[j, column] = ToDouble(state[j]);
                }

                if (column == count - 1)
                {
                    break;
                }

                // Gray code 順（Antonov & Saleev, 1979）では、第 i 点から第 i+1 点への遷移は i+1 の最下位セットビットに対応する方向数の XOR のみで求まる。
                int k = BitOperations.TrailingZeroCount(start + (ulong)column + 1);
                Xor(state, directions.AsSpan(k * Dimension, Dimension));
            }
        }
        finally
        {
            if (rented is not null)
            {
                ArrayPool<ulong>.Shared.Return(rented);
            }
        }

        return destination;
    }

    /// <inheritdoc/>
    public Matrix<double> GetPoints(ulong start, int count)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(count);
        var points = new Matrix<double>(Dimension, count);
        GetPoints(start, points.AsMatrixSpan());
        return points;
    }

    private void Start(ulong index, Span<ulong> state)
    {
        shifts.CopyTo(state);
        ulong gray = index ^ (index >> 1);
        while (gray != 0)
        {
            int k = BitOperations.TrailingZeroCount(gray);
            Xor(state, directions.AsSpan(k * Dimension, Dimension));
            gray &= gray - 1;
        }
    }

    private static void Xor(Span<ulong> state, ReadOnlySpan<ulong> direction)
    {
        int j = 0;
        if (Simd.IsHardwareAccelerated)
        {
            for (; j <= state.Length - SimdVector.Count; j += SimdVector.Count)
            {
                (new SimdVector(state[j..]) ^ new SimdVector(direction[j..])).CopyTo(state[j..]);
            }
        }

        for (; j < state.Length; j++)
        {
            state[j] ^= direction[j];
        }
    }

    // 上位 52 ビット k を (k + 0.5) * 2^-52 に写像し、開区間 [2^-53, 1 - 2^-53] に収める。
    private static double ToDouble(ulong digits) => (digits >> 12) * Unit + HalfUnit;
}
