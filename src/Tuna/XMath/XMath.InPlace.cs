using System.Numerics;

namespace Tuna;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む。
using Tuna.LinearAlgebra;

public static partial class XMath
{
    public static partial class InPlace
    {
        /// <summary>
        /// Adds a scalar to every element of a vector in place (<c>x ← x + scalar</c>).
        /// <para>ベクトルの全要素にスカラーをインプレースで加算します（<c>x ← x + scalar</c>）。</para>
        /// </summary>
        /// <typeparam name="T">The element type. / 要素型。</typeparam>
        /// <param name="x">The vector to modify in place. / インプレースで更新するベクトル。</param>
        /// <param name="scalar">The scalar value to add. / 加算するスカラー値。</param>
        public static void Add<T>(VectorSpan<T> x, T scalar)
            where T : unmanaged, INumberBase<T> =>
            StridedKernels.Map(x.AsMatrix(), x.AsMatrix(), new AddScalarOperator<T>(scalar));

        /// <summary>
        /// Adds a scalar to every element of a matrix in place (<c>x ← x + scalar</c>).
        /// <para>行列の全要素にスカラーをインプレースで加算します（<c>x ← x + scalar</c>）。</para>
        /// </summary>
        /// <typeparam name="T">The element type. / 要素型。</typeparam>
        /// <param name="x">The matrix to modify in place. / インプレースで更新する行列。</param>
        /// <param name="scalar">The scalar value to add. / 加算するスカラー値。</param>
        public static void Add<T>(MatrixSpan<T> x, T scalar)
            where T : unmanaged, INumberBase<T> =>
            StridedKernels.Map(x, x, new AddScalarOperator<T>(scalar));

        /// <summary>
        /// Subtracts a scalar from every element of a vector in place (<c>x ← x - scalar</c>).
        /// <para>ベクトルの全要素からスカラーをインプレースで減算します（<c>x ← x - scalar</c>）。</para>
        /// </summary>
        /// <typeparam name="T">The element type. / 要素型。</typeparam>
        /// <param name="x">The vector to modify in place. / インプレースで更新するベクトル。</param>
        /// <param name="scalar">The scalar value to subtract. / 減算するスカラー値。</param>
        public static void Subtract<T>(VectorSpan<T> x, T scalar)
            where T : unmanaged, INumberBase<T> =>
            StridedKernels.Map(x.AsMatrix(), x.AsMatrix(), new SubtractScalarOperator<T>(scalar));

        /// <summary>
        /// Subtracts a scalar from every element of a matrix in place (<c>x ← x - scalar</c>).
        /// <para>行列の全要素からスカラーをインプレースで減算します（<c>x ← x - scalar</c>）。</para>
        /// </summary>
        /// <typeparam name="T">The element type. / 要素型。</typeparam>
        /// <param name="x">The matrix to modify in place. / インプレースで更新する行列。</param>
        /// <param name="scalar">The scalar value to subtract. / 減算するスカラー値。</param>
        public static void Subtract<T>(MatrixSpan<T> x, T scalar)
            where T : unmanaged, INumberBase<T> =>
            StridedKernels.Map(x, x, new SubtractScalarOperator<T>(scalar));

        /// <summary>
        /// Multiplies every element of a vector by a scalar in place (<c>x ← x · scalar</c>).
        /// <para>ベクトルの全要素をインプレースでスカラー倍します（<c>x ← x · scalar</c>）。</para>
        /// </summary>
        /// <typeparam name="T">The element type. / 要素型。</typeparam>
        /// <param name="x">The vector to modify in place. / インプレースで更新するベクトル。</param>
        /// <param name="scalar">The scalar multiplier. / 乗算するスカラー値。</param>
        public static void Multiply<T>(VectorSpan<T> x, T scalar)
            where T : unmanaged, INumberBase<T> =>
            StridedKernels.Scale(x.AsMatrix(), scalar, x.AsMatrix());

        /// <summary>
        /// Multiplies every element of a matrix by a scalar in place (<c>x ← x · scalar</c>).
        /// <para>行列の全要素をインプレースでスカラー倍します（<c>x ← x · scalar</c>）。</para>
        /// </summary>
        /// <typeparam name="T">The element type. / 要素型。</typeparam>
        /// <param name="x">The matrix to modify in place. / インプレースで更新する行列。</param>
        /// <param name="scalar">The scalar multiplier. / 乗算するスカラー値。</param>
        public static void Multiply<T>(MatrixSpan<T> x, T scalar)
            where T : unmanaged, INumberBase<T> =>
            StridedKernels.Scale(x, scalar, x);

        /// <summary>
        /// Applies elementwise soft thresholding to a vector in place (<c>x ← sign(x) · max(|x| - threshold, 0)</c>).
        /// <para>ベクトルに対して要素ごとの軟判定しきい値をインプレースで適用します（<c>x ← sign(x) · max(|x| - threshold, 0)</c>）。</para>
        /// </summary>
        /// <typeparam name="T">The element type. / 要素型。</typeparam>
        /// <param name="x">The vector to modify in place. / インプレースで更新するベクトル。</param>
        /// <param name="threshold">The non-negative threshold value. / 0 以上の軟判定しきい値。</param>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="threshold"/> is negative or <see cref="double.NaN"/>. / <paramref name="threshold"/> が負または NaN の場合。</exception>
        public static void SoftThreshold<T>(VectorSpan<T> x, T threshold)
            where T : unmanaged, INumber<T> =>
            StridedKernels.SoftThreshold(x.AsMatrix(), RequireThreshold(threshold), x.AsMatrix());

        /// <summary>
        /// Applies elementwise soft thresholding to a matrix in place (<c>x ← sign(x) · max(|x| - threshold, 0)</c>).
        /// <para>行列に対して要素ごとの軟判定しきい値をインプレースで適用します（<c>x ← sign(x) · max(|x| - threshold, 0)</c>）。</para>
        /// </summary>
        /// <typeparam name="T">The element type. / 要素型。</typeparam>
        /// <param name="x">The matrix to modify in place. / インプレースで更新する行列。</param>
        /// <param name="threshold">The non-negative threshold value. / 0 以上の軟判定しきい値。</param>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="threshold"/> is negative or <see cref="double.NaN"/>. / <paramref name="threshold"/> が負または NaN の場合。</exception>
        public static void SoftThreshold<T>(MatrixSpan<T> x, T threshold)
            where T : unmanaged, INumber<T> =>
            StridedKernels.SoftThreshold(x, RequireThreshold(threshold), x);

        /// <summary>
        /// Computes the elementwise exponential of a vector in place (<c>x ← exp(x)</c>).
        /// <para>ベクトルの要素ごとの指数関数をインプレースで計算します（<c>x ← exp(x)</c>）。</para>
        /// </summary>
        /// <typeparam name="T">The element type. / 要素型。</typeparam>
        /// <param name="x">The vector to modify in place. / インプレースで更新するベクトル。</param>
        [InferenceOverloads]
        public static void Exp<T>(VectorSpan<T> x)
            where T : unmanaged, INumberBase<T>, IExponentialFunctions<T> =>
            StridedKernels.Map(x.AsMatrix(), x.AsMatrix(), default(ExpOperator<T>));

        /// <summary>
        /// Computes the elementwise exponential of a matrix in place (<c>x ← exp(x)</c>).
        /// <para>行列の要素ごとの指数関数をインプレースで計算します（<c>x ← exp(x)</c>）。</para>
        /// </summary>
        /// <typeparam name="T">The element type. / 要素型。</typeparam>
        /// <param name="x">The matrix to modify in place. / インプレースで更新する行列。</param>
        [InferenceOverloads]
        public static void Exp<T>(MatrixSpan<T> x)
            where T : unmanaged, INumberBase<T>, IExponentialFunctions<T> =>
            StridedKernels.Map(x, x, default(ExpOperator<T>));

        /// <summary>
        /// Applies a stateless unary operator elementwise to a vector in place (<c>x ← TOperator(x)</c>).
        /// <para>状態を持たない1引数演算子をベクトルの各要素にインプレースで適用します（<c>x ← TOperator(x)</c>）。</para>
        /// </summary>
        /// <typeparam name="T">The element type. / 要素型。</typeparam>
        /// <typeparam name="TOperator">The stateless unary operator type. / 状態を持たない1引数演算子の型。</typeparam>
        /// <param name="x">The vector to modify in place. / インプレースで更新するベクトル。</param>
        [FixedElementTypeOverloads(typeof(Half), typeof(float), typeof(double), typeof(decimal), typeof(Complex))]
        public static void Map<T, TOperator>(VectorSpan<T> x)
            where T : unmanaged, INumberBase<T>
            where TOperator : IUnaryOperator<T> =>
            StridedKernels.Map(x.AsMatrix(), x.AsMatrix(), default(StaticUnaryOperator<T, TOperator>));

        /// <summary>
        /// Applies a stateless unary operator elementwise to a matrix in place (<c>x ← TOperator(x)</c>).
        /// <para>状態を持たない1引数演算子を行列の各要素にインプレースで適用します（<c>x ← TOperator(x)</c>）。</para>
        /// </summary>
        /// <typeparam name="T">The element type. / 要素型。</typeparam>
        /// <typeparam name="TOperator">The stateless unary operator type. / 状態を持たない1引数演算子の型。</typeparam>
        /// <param name="x">The matrix to modify in place. / インプレースで更新する行列。</param>
        [FixedElementTypeOverloads(typeof(Half), typeof(float), typeof(double), typeof(decimal), typeof(Complex))]
        public static void Map<T, TOperator>(MatrixSpan<T> x)
            where T : unmanaged, INumberBase<T>
            where TOperator : IUnaryOperator<T> =>
            StridedKernels.Map(x, x, default(StaticUnaryOperator<T, TOperator>));

        /// <summary>
        /// Applies a stateful unary operator elementwise to a vector in place (<c>x ← op(x)</c>).
        /// <para>状態を持つ1引数演算子をベクトルの各要素にインプレースで適用します（<c>x ← op(x)</c>）。</para>
        /// </summary>
        /// <typeparam name="T">The element type. / 要素型。</typeparam>
        /// <typeparam name="TOperator">The stateful unary operator type. / 状態を持つ1引数演算子の型。</typeparam>
        /// <param name="x">The vector to modify in place. / インプレースで更新するベクトル。</param>
        /// <param name="op">The operator to apply. / 適用する演算子。</param>
        [InferenceOverloads]
        public static void Map<T, TOperator>(VectorSpan<T> x, TOperator op)
            where T : unmanaged, INumberBase<T>
            where TOperator : IStatefulUnaryOperator<T> =>
            StridedKernels.Map(x.AsMatrix(), x.AsMatrix(), op);

        /// <summary>
        /// Applies a stateful unary operator elementwise to a matrix in place (<c>x ← op(x)</c>).
        /// <para>状態を持つ1引数演算子を行列の各要素にインプレースで適用します（<c>x ← op(x)</c>）。</para>
        /// </summary>
        /// <typeparam name="T">The element type. / 要素型。</typeparam>
        /// <typeparam name="TOperator">The stateful unary operator type. / 状態を持つ1引数演算子の型。</typeparam>
        /// <param name="x">The matrix to modify in place. / インプレースで更新する行列。</param>
        /// <param name="op">The operator to apply. / 適用する演算子。</param>
        [InferenceOverloads]
        public static void Map<T, TOperator>(MatrixSpan<T> x, TOperator op)
            where T : unmanaged, INumberBase<T>
            where TOperator : IStatefulUnaryOperator<T> =>
            StridedKernels.Map(x, x, op);
    }
}
