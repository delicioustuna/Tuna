namespace Tuna.LinearAlgebra;

public sealed partial class SingularValueDecomposition
{
    private const double Epsilon = 2.220446049250313e-16;
    private const int StackLimit = 256;
    private static readonly double LargeQuotient = Math.ScaleB(1.0, 1000);
    private static readonly double SmallRightHandSide = Math.ScaleB(1.0, -970);

    /// <summary>
    /// Returns the minimum-norm least-squares solution to <c>Ax = b</c> in a new vector.
    /// <para><c>Ax = b</c> の最小二乗解のうち、ノルムが最小の解を新しいベクトルとして返します。</para>
    /// </summary>
    /// <remarks>
    /// <para>The method supports rectangular and rank-deficient matrices. It treats singular values at or below the cutoff as zero. If no singular values are retained, the solution is zero. Otherwise, non-finite entries of <paramref name="b"/> participate in floating-point arithmetic.</para>
    /// <para>長方形行列やランク落ち行列にも適用できます。カットオフ以下の特異値を 0 として扱います。保持する特異値がなければ解はゼロになります。それ以外では、<paramref name="b"/> に含まれる有限でない値も浮動小数点演算の対象になります。</para>
    /// </remarks>
    /// <param name="b">The right-hand side, with length equal to the number of rows of the original matrix. / 元の行列の行数と同じ長さの右辺。</param>
    /// <param name="tolerance">The relative tolerance. A value of <see langword="null"/> selects <c>max(m, n) × 2⁻⁵²</c> for an <c>m × n</c> matrix. The method retains <c>σᵢ &gt; tolerance × σ_max</c>. A value of 0 retains every positive singular value. / 相対許容誤差。<see langword="null"/> の場合、<c>m × n</c> 行列に対して <c>max(m, n) × 2⁻⁵²</c> を使います。<c>σᵢ &gt; tolerance × σ_max</c> を満たす特異値を保持します。0 を指定すると正の特異値をすべて保持します。</param>
    /// <returns>The solution, with length equal to the number of columns of the original matrix. / 元の行列の列数と同じ長さの解。</returns>
    /// <exception cref="ArgumentException">The length of <paramref name="b"/> does not match the number of rows. / <paramref name="b"/> の長さが元の行列の行数と一致しない場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="tolerance"/> is negative or non-finite. / <paramref name="tolerance"/> が負または有限でない場合。</exception>
    /// <exception cref="InvalidOperationException">Singular vectors were omitted, or the singular values are non-finite, negative, or out of descending order. / 特異ベクトルを計算していないか、特異値に有限でない値や負の値があるか、特異値が降順に並んでいない場合。</exception>
    public Vector<double> Solve(ReadOnlyVectorSpan<double> b, double? tolerance = null)
    {
        var (u, v, rank) = RequireFactors(tolerance);
        RequireRightHandSideRows(b.Length, u.Rows);
        var result = Vector<double>.Uninitialized(v.Rows);
        SolveCore(u, v, rank, b.AsMatrix(), result.AsVectorSpan().AsMatrix());
        return result;
    }

    /// <summary>
    /// Stores the minimum-norm least-squares solution to <c>Ax = b</c> in the specified vector view.
    /// <para><c>Ax = b</c> の最小ノルム最小二乗解を指定したベクトルビューに格納します。</para>
    /// </summary>
    /// <param name="b">The right-hand side, with length equal to the number of rows of the original matrix. / 元の行列の行数と同じ長さの右辺。</param>
    /// <param name="destination">The output view, with length equal to the number of columns of the original matrix. It may overlap <paramref name="b"/>. / 元の行列の列数と同じ長さの出力先ビュー。<paramref name="b"/> と領域を共有できます。</param>
    /// <param name="tolerance">The relative tolerance, as in <see cref="Solve(ReadOnlyVectorSpan{double}, double?)"/>. / <see cref="Solve(ReadOnlyVectorSpan{double}, double?)"/> と同じ相対許容誤差。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException">A length does not match, or <paramref name="destination"/> overlaps <see cref="U"/>, <see cref="V"/>, or <see cref="Values"/>. / 長さが一致しないか、<paramref name="destination"/> が <see cref="U"/>、<see cref="V"/>、<see cref="Values"/> のいずれかと重複する場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="tolerance"/> is negative or non-finite. / <paramref name="tolerance"/> が負または有限でない場合。</exception>
    /// <exception cref="InvalidOperationException">Singular vectors were omitted, or the singular values are non-finite, negative, or out of descending order. / 特異ベクトルを計算していないか、特異値に有限でない値や負の値があるか、特異値が降順に並んでいない場合。</exception>
    public VectorSpan<double> Solve(ReadOnlyVectorSpan<double> b, VectorSpan<double> destination, double? tolerance = null)
    {
        var (u, v, rank) = RequireFactors(tolerance);
        RequireRightHandSideRows(b.Length, u.Rows);
        SolveCore(u, v, rank, b.AsMatrix(), destination.AsMatrix());
        return destination;
    }

    /// <summary>
    /// Returns the minimum-norm least-squares solutions for the columns of <paramref name="b"/> in a new matrix.
    /// <para><paramref name="b"/> の列ごとに最小ノルム最小二乗解を求め、新しい行列として返します。</para>
    /// </summary>
    /// <param name="b">The right-hand sides, with the same number of rows as the original matrix. / 元の行列と同じ行数の右辺行列。</param>
    /// <param name="tolerance">The relative tolerance, as in <see cref="Solve(ReadOnlyVectorSpan{double}, double?)"/>. / <see cref="Solve(ReadOnlyVectorSpan{double}, double?)"/> と同じ相対許容誤差。</param>
    /// <returns>The solution matrix, with <c>n</c> rows for an <c>m × n</c> input and the same number of columns as <paramref name="b"/>. / 元の行列が <c>m × n</c> の場合に、<c>n</c> 行で、<paramref name="b"/> と同じ数の列を持つ解行列。</returns>
    /// <exception cref="ArgumentException">The number of rows of <paramref name="b"/> does not match. / <paramref name="b"/> の行数が元の行列の行数と一致しない場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="tolerance"/> is negative or non-finite. / <paramref name="tolerance"/> が負または有限でない場合。</exception>
    /// <exception cref="InvalidOperationException">Singular vectors were omitted, or the singular values are non-finite, negative, or out of descending order. / 特異ベクトルを計算していないか、特異値に有限でない値や負の値があるか、特異値が降順に並んでいない場合。</exception>
    /// <exception cref="OverflowException">The number of elements in the result exceeds <see cref="int.MaxValue"/>. / 結果の要素数が <see cref="int.MaxValue"/> を超える場合。</exception>
    public Matrix<double> Solve(ReadOnlyMatrixSpan<double> b, double? tolerance = null)
    {
        var (u, v, rank) = RequireFactors(tolerance);
        RequireRightHandSideRows(b.Rows, u.Rows);
        var result = Matrix<double>.Uninitialized(v.Rows, b.Columns);
        SolveCore(u, v, rank, b, result);
        return result;
    }

    /// <summary>
    /// Stores the minimum-norm least-squares solutions for the columns of <paramref name="b"/> in the specified matrix view.
    /// <para><paramref name="b"/> の列ごとに最小ノルム最小二乗解を求め、指定した行列ビューに格納します。</para>
    /// </summary>
    /// <param name="b">The right-hand sides, with the same number of rows as the original matrix. / 元の行列と同じ行数の右辺行列。</param>
    /// <param name="destination">The <c>n × b.Columns</c> output view for an <c>m × n</c> input. It may overlap <paramref name="b"/>. / 元の行列が <c>m × n</c> の場合の <c>n × b.Columns</c> の出力先ビュー。<paramref name="b"/> と領域を共有できます。</param>
    /// <param name="tolerance">The relative tolerance, as in <see cref="Solve(ReadOnlyVectorSpan{double}, double?)"/>. / <see cref="Solve(ReadOnlyVectorSpan{double}, double?)"/> と同じ相対許容誤差。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException">A shape does not match, or <paramref name="destination"/> overlaps <see cref="U"/>, <see cref="V"/>, or <see cref="Values"/>. / 形状が一致しないか、<paramref name="destination"/> が <see cref="U"/>、<see cref="V"/>、<see cref="Values"/> のいずれかと重複する場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="tolerance"/> is negative or non-finite. / <paramref name="tolerance"/> が負または有限でない場合。</exception>
    /// <exception cref="InvalidOperationException">Singular vectors were omitted, or the singular values are non-finite, negative, or out of descending order. / 特異ベクトルを計算していないか、特異値に有限でない値や負の値があるか、特異値が降順に並んでいない場合。</exception>
    public MatrixSpan<double> Solve(ReadOnlyMatrixSpan<double> b, MatrixSpan<double> destination, double? tolerance = null)
    {
        var (u, v, rank) = RequireFactors(tolerance);
        RequireRightHandSideRows(b.Rows, u.Rows);
        SolveCore(u, v, rank, b, destination);
        return destination;
    }

    /// <summary>
    /// Returns the Moore–Penrose pseudoinverse computed from the retained singular values in a new matrix.
    /// <para>保持した特異値からムーア・ペンローズ疑似逆行列を求め、新しい行列として返します。</para>
    /// </summary>
    /// <param name="tolerance">The relative tolerance, as in <see cref="Solve(ReadOnlyVectorSpan{double}, double?)"/>. / <see cref="Solve(ReadOnlyVectorSpan{double}, double?)"/> と同じ相対許容誤差。</param>
    /// <returns>The <c>n × m</c> pseudoinverse of an <c>m × n</c> input. / 元の行列が <c>m × n</c> の場合の <c>n × m</c> の疑似逆行列。</returns>
    /// <remarks>
    /// <para>When solving linear or least-squares systems, prefer <see cref="Solve(ReadOnlyMatrixSpan{double}, double?)"/> over forming the pseudoinverse explicitly.</para>
    /// <para>連立一次方程式や最小二乗問題を解くだけであれば、疑似逆行列を明示的に求めるよりも <see cref="Solve(ReadOnlyMatrixSpan{double}, double?)"/> の方が高速かつ高精度です。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="tolerance"/> is negative or non-finite. / <paramref name="tolerance"/> が負または有限でない場合。</exception>
    /// <exception cref="InvalidOperationException">Singular vectors were omitted, or the singular values are non-finite, negative, or out of descending order. / 特異ベクトルを計算していないか、特異値に有限でない値や負の値があるか、特異値が降順に並んでいない場合。</exception>
    public Matrix<double> PseudoInverse(double? tolerance = null)
    {
        var (u, v, rank) = RequireFactors(tolerance);
        var result = Matrix<double>.Uninitialized(v.Rows, u.Rows);
        PseudoInverseCore(u, v, rank, result);
        return result;
    }

    /// <summary>
    /// Stores the Moore–Penrose pseudoinverse computed from the retained singular values in the specified matrix view.
    /// <para>保持した特異値からムーア・ペンローズ疑似逆行列を求め、指定した行列ビューに格納します。</para>
    /// </summary>
    /// <param name="destination">The <c>n × m</c> output view for an <c>m × n</c> input. / 元の行列が <c>m × n</c> の場合の <c>n × m</c> の出力先ビュー。</param>
    /// <param name="tolerance">The relative tolerance, as in <see cref="Solve(ReadOnlyVectorSpan{double}, double?)"/>. / <see cref="Solve(ReadOnlyVectorSpan{double}, double?)"/> と同じ相対許容誤差。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException">The shape of <paramref name="destination"/> does not match, or it overlaps <see cref="U"/>, <see cref="V"/>, or <see cref="Values"/>. / <paramref name="destination"/> の形状が一致しないか、<paramref name="destination"/> が <see cref="U"/>、<see cref="V"/>、<see cref="Values"/> のいずれかと重複する場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="tolerance"/> is negative or non-finite. / <paramref name="tolerance"/> が負または有限でない場合。</exception>
    /// <exception cref="InvalidOperationException">Singular vectors were omitted, or the singular values are non-finite, negative, or out of descending order. / 特異ベクトルを計算していないか、特異値に有限でない値や負の値があるか、特異値が降順に並んでいない場合。</exception>
    public MatrixSpan<double> PseudoInverse(MatrixSpan<double> destination, double? tolerance = null)
    {
        var (u, v, rank) = RequireFactors(tolerance);
        PseudoInverseCore(u, v, rank, destination);
        return destination;
    }

    private int CountAboveCutoff(double tolerance)
    {
        if (Values.Length == 0)
        {
            return 0;
        }

        double cutoff = tolerance * Values[0];
        // カットオフが非正規数になるときは、丸めの前に縮尺を変えて比較する。
        int shift = double.IsSubnormal(cutoff) ? 54 : 0;
        if (shift != 0)
        {
            cutoff = tolerance * Math.ScaleB(Values[0], shift);
        }

        int rank = 0;
        foreach (double sigma in Values)
        {
            if ((shift == 0 ? sigma : Math.ScaleB(sigma, shift)) > cutoff)
            {
                rank++;
            }
        }

        return rank;
    }

    private (Matrix<double> U, Matrix<double> V, int Rank) RequireFactors(double? tolerance)
    {
        double relativeTolerance = tolerance ?? (_maxDimension * Epsilon);
        if (!double.IsFinite(relativeTolerance) || relativeTolerance < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(tolerance), "The tolerance must be finite and nonnegative.");
        }

        if (U is not { } u || V is not { } v)
        {
            throw new InvalidOperationException("The decomposition must include singular vectors.");
        }

        double previous = double.PositiveInfinity;
        foreach (double sigma in Values)
        {
            if (!double.IsFinite(sigma) || sigma < 0 || sigma > previous)
            {
                throw new InvalidOperationException("The singular values must be finite, nonnegative and sorted in descending order.");
            }

            previous = sigma;
        }

        return (u, v, CountAboveCutoff(relativeTolerance));
    }

    private static void RequireRightHandSideRows(int rows, int expected)
    {
        if (rows != expected)
        {
            throw new ArgumentException("The right-hand side must have as many rows as the original matrix.", "b");
        }
    }

    private void RequireDestination(Matrix<double> u, Matrix<double> v, int columns, MatrixSpan<double> destination)
    {
        if (destination.Rows != v.Rows || destination.Columns != columns)
        {
            throw new ArgumentException("Destination shape must agree.", nameof(destination));
        }

        StridedKernels.RequireDisjoint<double>(u, destination);
        StridedKernels.RequireDisjoint<double>(v, destination);
        StridedKernels.RequireDisjoint(new ReadOnlyMatrixSpan<double>(Values, Values.Length, 1), destination);
    }

    private void SolveCore(Matrix<double> u, Matrix<double> v, int rank, ReadOnlyMatrixSpan<double> b, MatrixSpan<double> destination)
    {
        RequireDestination(u, v, b.Columns, destination);
        if (rank == 0 || destination.IsEmpty)
        {
            destination.Fill(0);
            return;
        }

        // UᵀB を先にすべて計算する。出力先と右辺は領域を共有できる。
        int length = checked(rank * b.Columns);
        double[]? rented = null;
        Span<double> work = length <= StackLimit ? stackalloc double[length] : (rented = WorkArrayPool<double>.Rent(length)).AsSpan(0, length);
        try
        {
            ReadOnlyMatrixSpan<double> left = u[.., ..rank];
            var projected = new MatrixSpan<double>(work, rank, b.Columns);
            MultiplyInto(left.T, b, projected);

            int[]?[]? exponents = null;
            for (int j = 0; j < b.Columns; j++)
            {
                if (DivideProjection(left, b.Column(j), work.Slice(j * rank, rank)) is { } columnExponents)
                {
                    (exponents ??= new int[b.Columns][])[j] = columnExponents;
                }
            }

            ReadOnlyMatrixSpan<double> right = v[.., ..rank];
            if (exponents is null)
            {
                MultiplyInto(right, projected, destination);
            }
            else
            {
                MultiplyScaledColumns(right, projected, destination, exponents);
            }
        }
        finally
        {
            if (rented is not null)
            {
                WorkArrayPool<double>.Return(rented);
            }
        }
    }

    private int[]? DivideProjection(ReadOnlyMatrixSpan<double> u, ReadOnlyVectorSpan<double> b, Span<double> projected)
    {
        bool needsScaling = false;
        bool zeroProjection = false;
        for (int i = 0; i < projected.Length; i++)
        {
            double dot = projected[i];
            double sigma = Values[i];
            double quotient = dot / sigma;
            needsScaling |= !(Math.Abs(quotient) <= LargeQuotient) || (sigma < 1 && double.IsSubnormal(dot));
            zeroProjection |= sigma < 1 && dot == 0;
            projected[i] = quotient;
        }

        if (!needsScaling && !zeroProjection)
        {
            return null;
        }

        // 極小の右辺では、内積の結果が 0 に丸められることもある。
        double max = XMath.NormInfinity(b);
        if (max == 0 || !double.IsFinite(max) || (!needsScaling && max >= SmallRightHandSide))
        {
            return null;
        }

        // 射影の各成分を仮数と指数に分け、桁の離れた成分を同じ倍率で縮めずに済むようにする。
        var exponents = new int[projected.Length];
        for (int i = 0; i < projected.Length; i++)
        {
            var (mantissa, exponent) = ScaledDot(u.Column(i), b);
            int sigmaExponent = Math.ILogB(Values[i]);
            double sigma = Math.ScaleB(Values[i], -sigmaExponent);
            projected[i] = mantissa / sigma;
            exponents[i] = exponent - sigmaExponent;
        }

        return exponents;
    }

    private static void MultiplyScaledColumns(ReadOnlyMatrixSpan<double> v, ReadOnlyMatrixSpan<double> projected,
        MatrixSpan<double> destination, int[]?[] exponents)
    {
        int first = 0;
        for (int j = 0; j < exponents.Length; j++)
        {
            if (exponents[j] is not { } columnExponents)
            {
                continue;
            }

            if (first < j)
            {
                MultiplyInto(v, projected[.., first..j], destination[.., first..j]);
            }

            var input = projected.Column(j);
            var output = destination.Column(j);
            for (int i = 0; i < output.Length; i++)
            {
                var (mantissa, exponent) = ScaledDot(v.Row(i), input, columnExponents);
                output[i] = Math.ScaleB(mantissa, exponent);
            }

            first = j + 1;
        }

        if (first < exponents.Length)
        {
            MultiplyInto(v, projected[.., first..], destination[.., first..]);
        }
    }

    // 積の指数に合わせて部分和を縮尺し、最後に仮数と指数を返す。
    private static (double Mantissa, int Exponent) ScaledDot(ReadOnlyVectorSpan<double> x, ReadOnlyVectorSpan<double> y,
        ReadOnlySpan<int> yExponents = default)
    {
        double sum = 0;
        int exponent = 0;
        for (int i = 0; i < x.Length; i++)
        {
            double a = x.ElementAt(i);
            double b = y.ElementAt(i);
            if (!double.IsFinite(a) || !double.IsFinite(b))
            {
                return (double.NaN, 0);
            }

            if (a == 0 || b == 0)
            {
                continue;
            }

            int ea = Math.ILogB(a);
            int eb = Math.ILogB(b);
            int e = ea + eb + (yExponents.IsEmpty ? 0 : yExponents[i]);
            double term = Math.ScaleB(a, -ea) * Math.ScaleB(b, -eb);
            if (sum == 0 || e > exponent)
            {
                sum = Math.ScaleB(sum, exponent - e);
                exponent = e;
            }

            sum += Math.ScaleB(term, e - exponent);
        }

        return (sum, exponent);
    }

    private void PseudoInverseCore(Matrix<double> u, Matrix<double> v, int rank, MatrixSpan<double> destination)
    {
        RequireDestination(u, v, u.Rows, destination);
        if (rank == 0)
        {
            destination.Fill(0);
            return;
        }

        ReadOnlyMatrixSpan<double> left = v[.., ..rank];
        ReadOnlyMatrixSpan<double> right = u[.., ..rank];
        bool scaleLeft = left.Rows <= right.Rows;
        var thin = scaleLeft ? left : right;
        int length = checked(thin.Rows * rank);
        double[]? rented = null;
        double[]? rentedOther = null;
        Span<double> work = length <= StackLimit ? stackalloc double[length] : (rented = WorkArrayPool<double>.Rent(length)).AsSpan(0, length);
        try
        {
            var scaled = new MatrixSpan<double>(work, thin.Rows, rank);
            bool split = double.IsSubnormal(Values[rank - 1]);
            DivideColumns(thin, scaled, split);

            if (split)
            {
                // 逆数がオーバーフローしうるときは、1/√σ を両方の因子に掛ける。
                var other = scaleLeft ? right : left;
                int otherLength = checked(other.Rows * rank);
                rentedOther = WorkArrayPool<double>.Rent(otherLength);
                var scaledOther = new MatrixSpan<double>(rentedOther.AsSpan(0, otherLength), other.Rows, rank);
                DivideColumns(other, scaledOther, split);
                ReadOnlyMatrixSpan<double> scaledV = scaleLeft ? scaled : scaledOther;
                ReadOnlyMatrixSpan<double> scaledU = scaleLeft ? scaledOther : scaled;
                MultiplyInto(scaledV, scaledU.T, destination);
                RecomputeNonFiniteProducts(scaledV, scaledU, destination);
            }
            else
            {
                MultiplyInto(scaleLeft ? scaled : left, (scaleLeft ? right : scaled).T, destination);
            }
        }
        finally
        {
            if (rentedOther is not null)
            {
                WorkArrayPool<double>.Return(rentedOther);
            }

            if (rented is not null)
            {
                WorkArrayPool<double>.Return(rented);
            }
        }
    }

    private static void RecomputeNonFiniteProducts(ReadOnlyMatrixSpan<double> v, ReadOnlyMatrixSpan<double> u, MatrixSpan<double> destination)
    {
        for (int j = 0; j < destination.Columns; j++)
        {
            var output = destination.Column(j);
            for (int i = 0; i < output.Length; i++)
            {
                if (!double.IsFinite(output[i]))
                {
                    var (mantissa, exponent) = ScaledDot(v.Row(i), u.Row(j));
                    output[i] = Math.ScaleB(mantissa, exponent);
                }
            }
        }
    }

    private void DivideColumns(ReadOnlyMatrixSpan<double> source, MatrixSpan<double> destination, bool split)
    {
        for (int j = 0; j < source.Columns; j++)
        {
            double sigma = split ? Math.Sqrt(Values[j]) : Values[j];
            var input = source.Column(j).Data;
            var output = destination.Column(j).Data;
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = input[i] / sigma;
            }
        }
    }

    private static void MultiplyInto(ReadOnlyMatrixSpan<double> left, ReadOnlyMatrixSpan<double> right, MatrixSpan<double> destination)
    {
        var output = destination.AsReadOnly();
        if (destination.Columns == 1)
        {
            StridedKernels.Multiply(left, right.Column(0), destination.Column(0));
        }
        else if (output.RowStride != 1 && output.ColumnStride == 1)
        {
            StridedKernels.MultiplyUnchecked(right.T, left.T, destination.T);
        }
        else
        {
            StridedKernels.MultiplyUnchecked(left, right, destination);
        }
    }
}
