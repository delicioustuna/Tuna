using System.Numerics;

namespace Tuna;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む。
using Tuna.LinearAlgebra;

public static partial class XMath
{
    /// <summary>
    /// Computes the squared Euclidean distance <c>Σᵢ (xᵢ − yᵢ)²</c> between two vectors.
    /// <para>2点間の二乗ユークリッド距離 <c>Σᵢ (xᵢ − yᵢ)²</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The first point. / 第1ベクトル。</param>
    /// <param name="y">The second point. / 第2ベクトル。</param>
    /// <returns>The squared Euclidean distance. / 二乗ユークリッド距離。</returns>
    /// <remarks>
    /// <para>For pairwise distance matrices across columns, use <c>SquaredDistance(X, Y)</c> or <c>PairwiseSymmetric&lt;Operators.SquaredDistance&lt;double&gt;&gt;(X)</c>.</para>
    /// <para>列ごとの距離行列には <c>SquaredDistance(X, Y)</c>、または同一集合の対称距離行列には <c>PairwiseSymmetric&lt;Operators.SquaredDistance&lt;double&gt;&gt;(X)</c> を使用してください。</para>
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="x"/> and <paramref name="y"/> have different lengths. / <paramref name="x"/> と <paramref name="y"/> の次元が異なる場合。</exception>
    [PairwiseOperation(OperatorName = "SquaredDistance", OperatorContainer = "Operators")]
    [InferenceOverloads]
    public static T SquaredDistance<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y)
        where T : unmanaged, INumber<T>
    {
        if (x.Length != y.Length)
        {
            throw new ArgumentException("Both points must have the same dimension.", nameof(y));
        }

        T sum = T.Zero;
        for (int i = 0; i < x.Length; i++)
        {
            T d = x[i] - y[i];
            sum += d * d;
        }

        return sum;
    }

    /// <summary>
    /// Computes the Euclidean distance <c>√Σᵢ (xᵢ − yᵢ)²</c> between two vectors without undue intermediate overflow or underflow.
    /// <para>途中のオーバーフローやアンダーフローを回避して、2点間のユークリッド距離 <c>√Σᵢ (xᵢ − yᵢ)²</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type. / 浮動小数点要素型。</typeparam>
    /// <param name="x">The first point. / 第1ベクトル。</param>
    /// <param name="y">The second point. / 第2ベクトル。</param>
    /// <returns>The Euclidean distance. / ユークリッド距離。</returns>
    /// <remarks>
    /// <para>For pairwise distance matrices across columns, use <c>Distance(X, Y)</c> or <c>PairwiseSymmetric&lt;Operators.Distance&lt;double&gt;&gt;(X)</c>.</para>
    /// <para>列ごとの距離行列には <c>Distance(X, Y)</c>、または同一集合の対称距離行列には <c>PairwiseSymmetric&lt;Operators.Distance&lt;double&gt;&gt;(X)</c> を使用してください。</para>
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="x"/> and <paramref name="y"/> have different lengths. / <paramref name="x"/> と <paramref name="y"/> の次元が異なる場合。</exception>
    [PairwiseOperation(OperatorName = "Distance", OperatorContainer = "Operators")]
    [InferenceOverloads]
    public static T Distance<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        T sum = SquaredDistance(x, y);
        if (T.IsNormal(sum) || T.IsZero(sum) && AllEqual(x, y) || T.IsNaN(sum))
        {
            return T.Sqrt(sum);
        }

        // オーバーフローまたは非正規化数になった場合は、最大差 s でスケーリングして s·√Σ(dᵢ/s)² を計算する。
        T scale = T.Zero;
        for (int i = 0; i < x.Length; i++)
        {
            T d = T.Abs(x[i] - y[i]);
            if (T.IsNaN(d))
            {
                return d;
            }

            scale = T.Max(scale, d);
        }

        if (T.IsInfinity(scale) || T.IsZero(scale))
        {
            return scale;
        }

        T scaled = T.Zero;
        for (int i = 0; i < x.Length; i++)
        {
            T d = (x[i] - y[i]) / scale;
            scaled += d * d;
        }

        return scale * T.Sqrt(scaled);
    }

    private static bool AllEqual<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y)
        where T : unmanaged, INumber<T>
    {
        for (int i = 0; i < x.Length; i++)
        {
            if (x[i] != y[i])
            {
                return false;
            }
        }

        return true;
    }

    public static partial class Operators
    {
        /// <summary>
        /// Stateful pairwise operator that computes the axis-scaled squared distance <c>Σᵢ ((xᵢ − yᵢ) / ℓᵢ)²</c> for use with <c>XMath.Pairwise</c>.
        /// <para>軸ごとの長さ尺度でスケーリングした二乗距離 <c>Σᵢ ((xᵢ − yᵢ) / ℓᵢ)²</c> を <c>XMath.Pairwise</c> で計算するための演算子です。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type. / 浮動小数点要素型。</typeparam>
        /// <remarks>
        /// <para>Useful for ARD kernel evaluation (<c>r²</c>). Pass by value to <c>XMath.Pairwise</c> or <c>XMath.PairwiseSymmetric</c>.</para>
        /// <para>ARD カーネルの <c>r²</c> の計算などに使用できます。<c>XMath.Pairwise</c> や <c>XMath.PairwiseSymmetric</c> に値として渡します。</para>
        /// </remarks>
        public readonly struct ScaledSquaredDistance<T> : IStatefulPairwiseOperator<T>
            where T : unmanaged, IFloatingPointIeee754<T>
        {
            // 乗算経路では 1/ℓᵢ、除算経路（逆数が正規化数で表せない場合）では ℓᵢ を保持する。
            private readonly T[] scales;
            private readonly bool divide;

            /// <summary>
            /// Initializes a new instance with a copy of the specified per-axis length scales.
            /// <para>指定した軸ごとの長さ尺度 <c>ℓᵢ</c> をコピーしてインスタンスを初期化します。</para>
            /// </summary>
            /// <param name="lengthScales">Positive, finite length scales with length equal to the point dimension. / 正かつ有限な長さ尺度（長さは点の次元）。</param>
            /// <exception cref="ArgumentException">Any length scale is non-positive, infinite, or <see cref="double.NaN"/>. / 長さ尺度に正かつ有限でない値が含まれる場合。</exception>
            public ScaledSquaredDistance(ReadOnlySpan<T> lengthScales)
            {
                scales = new T[lengthScales.Length];
                for (int i = 0; i < lengthScales.Length; i++)
                {
                    T l = lengthScales[i];
                    if (!(l > T.Zero && T.IsFinite(l)))
                    {
                        throw new ArgumentException("Length scales must be positive and finite.", nameof(lengthScales));
                    }

                    scales[i] = T.One / l;
                    divide |= !T.IsNormal(scales[i]);
                }

                if (divide)
                {
                    lengthScales.CopyTo(scales);
                }
            }

            /// <summary>
            /// Gets the expected point dimension (the number of length scales).
            /// <para>点の次元（長さ尺度の個数）を取得します。</para>
            /// </summary>
            public int Dimension => scales?.Length ?? 0;

            /// <summary>
            /// Computes the scaled squared distance between two points.
            /// <para>2点間の尺度付き二乗距離を計算します。</para>
            /// </summary>
            /// <param name="x">The first point. / 第1ベクトル。</param>
            /// <param name="y">The second point. / 第2ベクトル。</param>
            /// <returns>The scaled squared distance. / 尺度付き二乗距離。</returns>
            /// <exception cref="ArgumentException">The length of <paramref name="x"/> or <paramref name="y"/> does not equal <see cref="Dimension"/>. / <paramref name="x"/> または <paramref name="y"/> の次元が <see cref="Dimension"/> と一致しない場合。</exception>
            public T Invoke(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y)
            {
                var w = scales ?? [];
                if (x.Length != w.Length || y.Length != w.Length)
                {
                    throw new ArgumentException("The dimension of the points must equal the number of length scales.");
                }

                T sum = T.Zero;
                if (divide)
                {
                    for (int i = 0; i < w.Length; i++)
                    {
                        T d = (x[i] - y[i]) / w[i];
                        sum += d * d;
                    }
                }
                else
                {
                    for (int i = 0; i < w.Length; i++)
                    {
                        T d = (x[i] - y[i]) * w[i];
                        sum += d * d;
                    }
                }

                // 有限座標の差 xᵢ − yᵢ のみがオーバーフローして和が非有限になった場合のみ、先にスケーリングしてから再計算する。
                return T.IsFinite(sum) ? sum : RescaledSum(x, y, w, divide);
            }

            private static T RescaledSum(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y, T[] w, bool divide)
            {
                T sum = T.Zero;
                for (int i = 0; i < w.Length; i++)
                {
                    T xi = x[i];
                    T yi = y[i];
                    T difference = xi - yi;
                    T d = T.IsInfinity(difference) && T.IsFinite(xi) && T.IsFinite(yi)
                        ? (divide ? xi / w[i] - yi / w[i] : xi * w[i] - yi * w[i])
                        : (divide ? difference / w[i] : difference * w[i]);
                    sum += d * d;
                }

                return sum;
            }
        }
    }
}
