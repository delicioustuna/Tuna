using System.Numerics;
using System.Runtime.CompilerServices;

namespace Tuna;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む。
using Tuna.LinearAlgebra;

public static partial class XMath
{
    /// <summary>
    /// Computes the matrix product <c>AB</c> and returns a new matrix.
    /// <para>行列積 <c>AB</c> を計算し、新しい行列を返します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="a">The left input matrix. / 左側の入力行列。</param>
    /// <param name="b">The right input matrix. / 右側の入力行列。</param>
    /// <returns>A new matrix containing <c>AB</c>. / 行列積 <c>AB</c> を格納した新しい行列。</returns>
    /// <remarks>
    /// <para>Transposed views (such as <c>A.T</c>) are passed directly as GEMM transpose flags without copying.</para>
    /// <para>転置ビュー（<c>A.T</c> など）はコピーせず GEMM の転置フラグとして渡されます。</para>
    /// </remarks>
    /// <exception cref="OverflowException">The number of elements <c>a.Rows · b.Columns</c> exceeds <see cref="int.MaxValue"/>. / 結果の要素数 <c>a.Rows · b.Columns</c> が <see cref="int.MaxValue"/> を超える場合。</exception>
    [InferenceOverloads]
    public static Matrix<T> Multiply<T>(ReadOnlyMatrixSpan<T> a, ReadOnlyMatrixSpan<T> b)
        where T : unmanaged, INumberBase<T>
    {
        var result = Matrix<T>.Uninitialized(a.Rows, b.Columns);
        Multiply(a, b, result);
        return result;
    }

    /// <summary>
    /// Computes the matrix product <c>AB</c> and writes the result to <paramref name="destination"/>.
    /// <para>行列積 <c>AB</c> を計算し、結果を <paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="a">The left input matrix. / 左側の入力行列。</param>
    /// <param name="b">The right input matrix. / 右側の入力行列。</param>
    /// <param name="destination">The destination span to receive the product. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> must not overlap either input.</para>
    /// <para><paramref name="destination"/> はいずれの入力とも領域が重なってはなりません。</para>
    /// </remarks>
    [InferenceOverloads]
    public static MatrixSpan<T> Multiply<T>(ReadOnlyMatrixSpan<T> a, ReadOnlyMatrixSpan<T> b, MatrixSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        StridedKernels.Multiply(a, b, destination);
        return destination;
    }

    /// <summary>
    /// Computes the matrix-vector product <c>Ax</c> and returns a new vector.
    /// <para>行列ベクトル積 <c>Ax</c> を計算し、新しいベクトルを返します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="a">The input matrix. / 入力行列。</param>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <returns>A new vector containing <c>Ax</c>. / 行列ベクトル積 <c>Ax</c> を格納した新しいベクトル。</returns>
    /// <remarks>
    /// <para>Transposed views are passed directly as GEMV transpose flags without copying.</para>
    /// <para>転置ビューはコピーせず GEMV の転置フラグとして渡されます。</para>
    /// </remarks>
    [InferenceOverloads]
    public static Vector<T> Multiply<T>(ReadOnlyMatrixSpan<T> a, ReadOnlyVectorSpan<T> x)
        where T : unmanaged, INumberBase<T>
    {
        var result = Vector<T>.Uninitialized(a.Rows);
        Multiply(a, x, result);
        return result;
    }

    /// <summary>
    /// Computes the matrix-vector product <c>Ax</c> and writes the result to <paramref name="destination"/>.
    /// <para>行列ベクトル積 <c>Ax</c> を計算し、結果を <paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="a">The input matrix. / 入力行列。</param>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="destination">The destination span to receive the product. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> must not overlap either input.</para>
    /// <para><paramref name="destination"/> はいずれの入力とも領域が重なってはなりません。</para>
    /// </remarks>
    [InferenceOverloads]
    public static VectorSpan<T> Multiply<T>(ReadOnlyMatrixSpan<T> a, ReadOnlyVectorSpan<T> x, VectorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        StridedKernels.Multiply(a, x, destination);
        return destination;
    }

    /// <summary>
    /// Computes the outer product <c>u vᵀ</c> of two vectors.
    /// <para>2 つのベクトルの外積（直積）<c>u vᵀ</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="u">The vector that determines the rows. / 行に対応するベクトル。</param>
    /// <param name="v">The vector that determines the columns. / 列に対応するベクトル。</param>
    /// <returns>A new <c>u.Length × v.Length</c> matrix with entries <c>u[i] * v[j]</c>. / 各要素が <c>u[i] * v[j]</c> の <c>u.Length × v.Length</c> 行列。</returns>
    /// <remarks>
    /// <para>Complex inputs use ordinary (non-conjugate) transposition. An empty input produces a matrix with the corresponding dimension equal to zero.</para>
    /// <para>複素数入力には共役転置ではなく通常の転置を用います。いずれかの入力の長さが 0 の場合、対応する次元が 0 の空行列を返します。</para>
    /// </remarks>
    /// <exception cref="OverflowException">The number of result elements exceeds <see cref="int.MaxValue"/>. / 結果の要素数が <see cref="int.MaxValue"/> を超える場合。</exception>
    [InferenceOverloads]
    public static Matrix<T> Outer<T>(ReadOnlyVectorSpan<T> u, ReadOnlyVectorSpan<T> v)
        where T : unmanaged, INumberBase<T>
    {
        var result = Matrix<T>.Uninitialized(u.Length, v.Length);
        Outer(u, v, result);
        return result;
    }

    /// <summary>
    /// Writes the outer product <c>u vᵀ</c> of two vectors to <paramref name="destination"/>.
    /// <para>2 つのベクトルの外積（直積）<c>u vᵀ</c> を <paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="u">The vector that determines the rows. / 行に対応するベクトル。</param>
    /// <param name="v">The vector that determines the columns. / 列に対応するベクトル。</param>
    /// <param name="destination">A <c>u.Length × v.Length</c> matrix view to receive the product. / 結果を書き込む <c>u.Length × v.Length</c> の行列ビュー。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> must not overlap either input. Complex inputs use ordinary (non-conjugate) transposition.</para>
    /// <para><paramref name="destination"/> はいずれの入力とも領域が重なってはなりません。複素数入力には共役転置ではなく通常の転置を用います。</para>
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has the wrong shape or overlaps either input. / <paramref name="destination"/> の形状が一致しないか、入力と領域が重なる場合。</exception>
    [InferenceOverloads]
    public static MatrixSpan<T> Outer<T>(ReadOnlyVectorSpan<T> u, ReadOnlyVectorSpan<T> v, MatrixSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        if (destination.Rows != u.Length || destination.Columns != v.Length)
        {
            throw new ArgumentException("Destination shape must agree.", nameof(destination));
        }

        if (u.Length == 0 || v.Length == 0)
        {
            return destination;
        }

        StridedKernels.RequireDisjoint(u.AsMatrix(), destination);
        StridedKernels.RequireDisjoint(v.AsMatrix(), destination);
        return Hadamard(u.Broadcast(v.Length), v.BroadcastRows(u.Length), destination);
    }

    /// <summary>
    /// Computes the wedge product <c>u vᵀ − v uᵀ</c> of two vectors of the same length.
    /// <para>同じ長さの 2 つのベクトルのウェッジ積 <c>u vᵀ − v uᵀ</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="u">The first vector. / 第 1 ベクトル。</param>
    /// <param name="v">The second vector. / 第 2 ベクトル。</param>
    /// <returns>A new <c>u.Length × u.Length</c> skew-symmetric matrix. / 新しい <c>u.Length × u.Length</c> の反対称行列。</returns>
    /// <remarks>
    /// <para>Empty vectors produce a 0 × 0 matrix. Off-diagonal entries come in opposite-sign pairs, and a diagonal entry is zero when both corresponding elements are finite.</para>
    /// <para>For finite <see langword="double"/>, <see langword="float"/>, and <see cref="Half"/> operands, the difference of products uses extra precision to reduce cancellation and avoid intermediate overflow and underflow. Non-finite operands use ordinary multiplication and subtraction. Complex inputs use ordinary (non-conjugate) transposition.</para>
    /// <para>入力の長さが 0 の場合は 0 × 0 の行列を返します。非対角成分は反対符号の組になり、対応する両方の要素が有限であれば対角成分は 0 になります。</para>
    /// <para>有限な <see langword="double"/>、<see langword="float"/>、<see cref="Half"/> の積の差は、桁落ちや途中のオーバーフロー、アンダーフローを抑えて計算します。非有限値を含む成分には通常の乗算と減算を用います。複素数入力には共役転置ではなく通常の転置を用います。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">The vectors have different lengths. / ベクトルの長さが一致しない場合。</exception>
    /// <exception cref="OverflowException">The number of result elements exceeds <see cref="int.MaxValue"/>. / 結果の要素数が <see cref="int.MaxValue"/> を超える場合。</exception>
    [InferenceOverloads]
    public static Matrix<T> Wedge<T>(ReadOnlyVectorSpan<T> u, ReadOnlyVectorSpan<T> v)
        where T : unmanaged, INumberBase<T>
    {
        RequireWedgeLengths(u.Length, v.Length);
        var result = Matrix<T>.Uninitialized(u.Length, u.Length);
        WedgeCore(u, v, result);
        return result;
    }

    /// <summary>
    /// Writes the wedge product <c>u vᵀ − v uᵀ</c> of two vectors to <paramref name="destination"/>.
    /// <para>2 つのベクトルのウェッジ積 <c>u vᵀ − v uᵀ</c> を <paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="u">The first vector. / 第 1 ベクトル。</param>
    /// <param name="v">The second vector. / 第 2 ベクトル。</param>
    /// <param name="destination">A <c>u.Length × u.Length</c> matrix view to receive the product. / 結果を書き込む <c>u.Length × u.Length</c> の行列ビュー。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> must not overlap either input. The calculation follows <see cref="Wedge{T}(ReadOnlyVectorSpan{T}, ReadOnlyVectorSpan{T})"/>.</para>
    /// <para><paramref name="destination"/> はいずれの入力とも領域が重なってはなりません。計算方法は <see cref="Wedge{T}(ReadOnlyVectorSpan{T}, ReadOnlyVectorSpan{T})"/> と同じです。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">The vectors have different lengths, or <paramref name="destination"/> has the wrong shape or overlaps either input. / ベクトルの長さが一致しないか、<paramref name="destination"/> の形状が一致しない、あるいは入力と領域が重なる場合。</exception>
    [InferenceOverloads]
    public static MatrixSpan<T> Wedge<T>(scoped ReadOnlyVectorSpan<T> u, scoped ReadOnlyVectorSpan<T> v, MatrixSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        int n = u.Length;
        RequireWedgeLengths(n, v.Length);
        if (destination.Rows != n || destination.Columns != n)
        {
            throw new ArgumentException("Destination shape must agree.", nameof(destination));
        }

        StridedKernels.RequireDisjoint(u.AsMatrix(), destination);
        StridedKernels.RequireDisjoint(v.AsMatrix(), destination);
        WedgeCore(u, v, destination);
        return destination;
    }

    private static void RequireWedgeLengths(int uLength, int vLength)
    {
        if (uLength != vLength)
        {
            throw new ArgumentException("The vectors must have the same length.", "v");
        }
    }

    private const int WedgeBlockSize = 16;

    private static void WedgeCore<T>(ReadOnlyVectorSpan<T> u, ReadOnlyVectorSpan<T> v, MatrixSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        int n = u.Length;
        if (n > WedgeBlockSize && u.Stride == 1 && v.Stride == 1 && destination.TryGetDense(out var dense))
        {
            WedgeDense(u.Data, v.Data, dense);
            return;
        }

        var output = destination.AsReadOnly();
        for (int j = 0; j < n; j++)
        {
            T uj = u.ElementAt(j), vj = v.ElementAt(j);
            Unsafe.AsRef(in output.ElementAt(j, j)) = WedgeDiagonal(uj, vj);
            for (int i = 0; i < j; i++)
            {
                T value = DifferenceOfProducts(u.ElementAt(i), vj, v.ElementAt(i), uj);
                Unsafe.AsRef(in output.ElementAt(i, j)) = value;
                Unsafe.AsRef(in output.ElementAt(j, i)) = -value;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void WedgeDense<T>(ReadOnlySpan<T> u, ReadOnlySpan<T> v, Span<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        int n = u.Length;
        for (int j = 0; j < n; j++)
        {
            destination[j * n + j] = WedgeDiagonal(u[j], v[j]);
        }

        // 符号反転先の列をブロック内に限定し、書き込みの局所性を保つ。
        for (int start = 0; start < n; start += WedgeBlockSize)
        {
            int end = Math.Min(start + WedgeBlockSize, n);
            for (int j = start + 1; j < n; j++)
            {
                T uj = u[j], vj = v[j];
                int column = j * n, mirror = start * n + j;
                int stop = Math.Min(end, j);
                for (int i = start; i < stop; i++, mirror += n)
                {
                    T value = DifferenceOfProducts(u[i], vj, v[i], uj);
                    destination[column + i] = value;
                    destination[mirror] = -value;
                }
            }
        }
    }

    private static T WedgeDiagonal<T>(T u, T v) where T : unmanaged, INumberBase<T> =>
        T.IsFinite(u) && T.IsFinite(v) ? T.Zero : u * v - v * u;

    /// <summary>
    /// Computes the cross product <c>u × v</c> of two three-dimensional vectors.
    /// <para>2 つの 3 次元ベクトルの外積 <c>u × v</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="u">The first vector, of length 3. / 長さ 3 の第 1 ベクトル。</param>
    /// <param name="v">The second vector, of length 3. / 長さ 3 の第 2 ベクトル。</param>
    /// <returns>A new vector of length 3 containing <c>u × v</c>. / <c>u × v</c> を格納した長さ 3 の新しいベクトル。</returns>
    /// <remarks>
    /// <para>For finite <see langword="double"/>, <see langword="float"/>, and <see cref="Half"/> operands, each component is computed with extra precision to reduce cancellation and avoid intermediate overflow and underflow. Other types and non-finite operands use ordinary multiplication and subtraction.</para>
    /// <para>有限な <see langword="double"/>、<see langword="float"/>、<see cref="Half"/> の成分は、桁落ちや途中のオーバーフロー、アンダーフローを抑えて計算します。その他の型や非有限値には通常の乗算と減算を用います。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">Either input has a length other than 3. / いずれかの入力の長さが 3 以外の場合。</exception>
    [InferenceOverloads]
    public static Vector<T> Cross<T>(ReadOnlyVectorSpan<T> u, ReadOnlyVectorSpan<T> v)
        where T : unmanaged, INumberBase<T>
    {
        RequireCrossInputs(u.Length, v.Length);
        var result = Vector<T>.Uninitialized(3);
        CrossCore(u, v, result);
        return result;
    }

    /// <summary>
    /// Writes the cross product <c>u × v</c> to <paramref name="destination"/>.
    /// <para>外積 <c>u × v</c> を <paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="u">The first vector, of length 3. / 長さ 3 の第 1 ベクトル。</param>
    /// <param name="v">The second vector, of length 3. / 長さ 3 の第 2 ベクトル。</param>
    /// <param name="destination">The destination vector view, of length 3. / 結果を書き込む長さ 3 のベクトルビュー。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> may overlap either or both inputs.</para>
    /// <para><paramref name="destination"/> は片方または両方の入力と領域が重なっていても構いません。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">An input or <paramref name="destination"/> has a length other than 3. / 入力または <paramref name="destination"/> の長さが 3 以外の場合。</exception>
    [InferenceOverloads]
    public static VectorSpan<T> Cross<T>(scoped ReadOnlyVectorSpan<T> u, scoped ReadOnlyVectorSpan<T> v, VectorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        RequireCrossInputs(u.Length, v.Length);
        if (destination.Length != 3)
        {
            throw new ArgumentException("Destination length must be 3.", nameof(destination));
        }

        CrossCore(u, v, destination);
        return destination;
    }

    private static void RequireCrossInputs(int uLength, int vLength)
    {
        if (uLength != 3)
        {
            throw new ArgumentException("Vector length must be 3.", "u");
        }

        if (vLength != 3)
        {
            throw new ArgumentException("Vector length must be 3.", "v");
        }
    }

    private static void CrossCore<T>(ReadOnlyVectorSpan<T> u, ReadOnlyVectorSpan<T> v, VectorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        T u0 = u[0], u1 = u[1], u2 = u[2];
        T v0 = v[0], v1 = v[1], v2 = v[2];
        T x = DifferenceOfProducts(u1, v2, u2, v1);
        T y = DifferenceOfProducts(u2, v0, u0, v2);
        T z = DifferenceOfProducts(u0, v1, u1, v0);
        destination[0] = x;
        destination[1] = y;
        destination[2] = z;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static T DifferenceOfProducts<T>(T a, T b, T c, T d) where T : unmanaged, INumberBase<T>
    {
        if (typeof(T) == typeof(double))
        {
            double difference = DifferenceOfDoubleProducts(Unsafe.BitCast<T, double>(a), Unsafe.BitCast<T, double>(b),
                Unsafe.BitCast<T, double>(c), Unsafe.BitCast<T, double>(d));
            return Unsafe.BitCast<double, T>(difference);
        }

        if (typeof(T) == typeof(float))
        {
            double left = (double)Unsafe.BitCast<T, float>(a) * Unsafe.BitCast<T, float>(b);
            double right = (double)Unsafe.BitCast<T, float>(c) * Unsafe.BitCast<T, float>(d);
            double difference = left - right;
            return double.IsFinite(difference) ? Unsafe.BitCast<float, T>((float)difference) : a * b - c * d;
        }

        if (typeof(T) == typeof(Half))
        {
            double left = (double)Unsafe.BitCast<T, Half>(a) * (double)Unsafe.BitCast<T, Half>(b);
            double right = (double)Unsafe.BitCast<T, Half>(c) * (double)Unsafe.BitCast<T, Half>(d);
            double difference = left - right;
            return double.IsFinite(difference) ? Unsafe.BitCast<Half, T>((Half)difference) : a * b - c * d;
        }

        return a * b - c * d;
    }

    private static double DifferenceOfDoubleProducts(double a, double b, double c, double d)
    {
        double left = a * b, right = c * d;
        double difference = left - right;
        // 53 ビット同士の積の丸め誤差を下位ビットまで保持できる大きさを目安にする。
        const double minUnscaledProduct = double.Epsilon * (1L << 53) * (1L << 53);
        if (double.IsFinite(difference) && Math.Max(Math.Abs(left), Math.Abs(right)) >= minUnscaledProduct)
        {
            return difference + ProductErrorDifference(a, b, c, d, left, right);
        }

        if (a == 0 || b == 0 || c == 0 || d == 0 || !double.IsFinite(a) || !double.IsFinite(b)
            || !double.IsFinite(c) || !double.IsFinite(d))
        {
            return difference;
        }

        // 積ごとに指数を求め、共通の指数でスケーリングしてから差を取る。
        int aExponent = Math.ILogB(a), cExponent = Math.ILogB(c);
        int exponent = Math.Max(aExponent + Math.ILogB(b), cExponent + Math.ILogB(d));
        a = Math.ScaleB(a, -aExponent);
        b = Math.ScaleB(b, aExponent - exponent);
        c = Math.ScaleB(c, -cExponent);
        d = Math.ScaleB(d, cExponent - exponent);
        left = a * b;
        right = c * d;
        return Math.ScaleB((left - right) + ProductErrorDifference(a, b, c, d, left, right), exponent);
    }

    private static double ProductErrorDifference(double a, double b, double c, double d, double left, double right)
        => Math.FusedMultiplyAdd(a, b, -left) - Math.FusedMultiplyAdd(c, d, -right);

    /// <summary>
    /// Returns a newly allocated transposed copy <c>Aᵀ</c> of the input matrix.
    /// <para>入力行列を転置した新しいコピー <c>Aᵀ</c> を返します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <returns>A new matrix containing the transpose of <paramref name="x"/>. / <paramref name="x"/> を転置した新しい行列。</returns>
    /// <remarks>
    /// <para>For a zero-copy transposed view, use <c>x.T</c>.</para>
    /// <para>コピーを伴わない転置ビューが必要な場合は <c>x.T</c> を使用してください。</para>
    /// </remarks>
    /// <exception cref="OverflowException">The number of elements exceeds <see cref="int.MaxValue"/> (for example, when passing a large broadcast view). / 結果の要素数が <see cref="int.MaxValue"/> を超える場合（要素数が <c>int</c> に収まらないブロードキャストビューを渡した場合など）。</exception>
    [InferenceOverloads]
    public static Matrix<T> Transpose<T>(ReadOnlyMatrixSpan<T> x)
        where T : unmanaged, INumberBase<T>
    {
        var result = Matrix<T>.Uninitialized(x.Columns, x.Rows);
        Transpose(x, result);
        return result;
    }

    /// <summary>
    /// Writes the transpose <c>Aᵀ</c> of the input matrix into <paramref name="destination"/>.
    /// <para>入力行列の転置 <c>Aᵀ</c> を <paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="destination">The destination span to receive the transposed matrix. / 転置結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> must not overlap <paramref name="x"/> (in-place transposition is not supported).</para>
    /// <para><paramref name="destination"/> は <paramref name="x"/> と領域が重なってはなりません（インプレース転置には対応していません）。</para>
    /// </remarks>
    [InferenceOverloads]
    public static MatrixSpan<T> Transpose<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        StridedKernels.Copy(x.T, destination);
        return destination;
    }
}
