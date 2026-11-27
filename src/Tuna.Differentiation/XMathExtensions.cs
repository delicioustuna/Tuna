using System.ComponentModel;
using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using System.Runtime.CompilerServices;
using Tuna.LinearAlgebra;

namespace Tuna.Differentiation;

/// <summary>
/// Provides <see cref="XMath"/> extension methods for building differentiable scalar, vector, matrix, and tensor expressions.
/// <para>スカラー・ベクトル・行列・テンソルの微分可能な計算式を組み立てるための <see cref="XMath"/> 拡張メソッドを提供します。</para>
/// </summary>
[EditorBrowsable(EditorBrowsableState.Never)]
public static partial class XMathExtensions
{
    extension(XMath)
    {
        /// <summary>
        /// Computes the matrix product <c>AB</c>.
        /// <para>行列積 <c>AB</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="a">The left matrix node. / 左から掛ける行列ノード。</param>
        /// <param name="b">The right matrix node. / 右から掛ける行列ノード。</param>
        /// <returns>A matrix node holding the product <c>AB</c>. / 積 <c>AB</c> を格納した行列ノード。</returns>
        /// <exception cref="ArgumentException">Thrown if the column count of <paramref name="a"/> does not equal the row count of <paramref name="b"/>. / <paramref name="a"/> の列数と <paramref name="b"/> の行数が一致しない場合。</exception>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> Multiply<T>(MatrixNode<T> a, MatrixNode<T> b) where T : unmanaged, IFloatingPointIeee754<T>
            => MultiplyCore(a, b);

        /// <summary>
        /// Computes the matrix-vector product <c>Ax</c>.
        /// <para>行列ベクトル積 <c>Ax</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="a">The matrix node. / 左から掛ける行列ノード。</param>
        /// <param name="x">The vector node. / 右から掛けるベクトルノード。</param>
        /// <returns>A vector node holding the product <c>Ax</c>. / 積 <c>Ax</c> を格納したベクトルノード。</returns>
        /// <exception cref="ArgumentException">Thrown if the column count of <paramref name="a"/> does not equal the length of <paramref name="x"/>. / <paramref name="a"/> の列数と <paramref name="x"/> の要素数が一致しない場合。</exception>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> Multiply<T>(MatrixNode<T> a, VectorNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => MultiplyCore(a, x);

        /// <summary>
        /// Computes the outer product <c>u vᵀ</c>.
        /// <para>2 つのベクトルの外積（直積）<c>u vᵀ</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The element type (<see langword="float"/> or <see langword="double"/>). / 要素型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="u">The vector that determines the rows. / 行に対応するベクトルノード。</param>
        /// <param name="v">The vector that determines the columns. / 列に対応するベクトルノード。</param>
        /// <returns>A matrix node holding <c>u vᵀ</c>. / 外積 <c>u vᵀ</c> を格納した行列ノード。</returns>
        /// <remarks>
        /// <para>The pullback accumulates <c>dZ v</c> into the gradient of <paramref name="u"/> and <c>dZᵀ u</c> into the gradient of <paramref name="v"/>.</para>
        /// <para>逆伝播では <paramref name="u"/> の勾配に <c>dZ v</c>、<paramref name="v"/> の勾配に <c>dZᵀ u</c> を加算します。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> Outer<T>(VectorNode<T> u, VectorNode<T> v) where T : unmanaged, IFloatingPointIeee754<T>
        {
            Tape tape = Owner(u.Tape, u.Generation, v.Tape, v.Generation);
            var store = tape.Store<T>();
            bool record = store.RequiresGradient(u.View.Slot) || store.RequiresGradient(v.View.Slot);
            tape.EnsureCanApply(record);

            int slot = store.AddPooled(u.Length, v.Length, record);
            XMath.Outer(store.VectorView(u.View), store.VectorView(v.View), store.WritableMatrix(slot));
            if (record)
            {
                tape.AppendArrayRecord<T>(ArrayOperation.Outer, slot, u.View, v.View);
            }

            return new(tape, ArrayView.Whole(slot, u.Length, v.Length));
        }

        /// <summary>
        /// Computes the cross product <c>u × v</c> of two three-dimensional vector nodes.
        /// <para>2 つの 3 次元ベクトルノードの外積 <c>u × v</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The element type (<see langword="float"/> or <see langword="double"/>). / 要素型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="u">The first vector node, of length 3. / 長さ 3 の第 1 ベクトルノード。</param>
        /// <param name="v">The second vector node, of length 3. / 長さ 3 の第 2 ベクトルノード。</param>
        /// <returns>A vector node holding <c>u × v</c>. / <c>u × v</c> を格納したベクトルノード。</returns>
        /// <remarks>
        /// <para>For the output adjoint <c>dz</c>, backpropagation accumulates <c>v × dz</c> into the gradient of <paramref name="u"/> and <c>dz × u</c> into the gradient of <paramref name="v"/>.</para>
        /// <para>出力の随伴を <c>dz</c> とすると、逆伝播では <paramref name="u"/> の勾配に <c>v × dz</c>、<paramref name="v"/> の勾配に <c>dz × u</c> を加算します。</para>
        /// </remarks>
        /// <exception cref="ArgumentException">Either vector node has a length other than 3. / いずれかのベクトルノードの長さが 3 以外の場合。</exception>
        /// <exception cref="InvalidOperationException">An input node is uninitialized, stale or belongs to a different tape, or recording requires a reset. / 入力ノードが未初期化、世代が古い、別のテープに属する、または記録を続けるためにテープのリセットが必要な場合。</exception>
        public static VectorNode<T> Cross<T>(VectorNode<T> u, VectorNode<T> v) where T : unmanaged, IFloatingPointIeee754<T>
        {
            Tape tape = Owner(u.Tape, u.Generation, v.Tape, v.Generation);
            if (u.Length != 3)
            {
                throw new ArgumentException("Vector length must be 3.", nameof(u));
            }

            if (v.Length != 3)
            {
                throw new ArgumentException("Vector length must be 3.", nameof(v));
            }

            var store = tape.Store<T>();
            bool record = store.RequiresGradient(u.View.Slot) || store.RequiresGradient(v.View.Slot);
            tape.EnsureCanApply(record);
            int slot = store.AddPooled(3, 1, record);
            XMath.Cross(store.VectorView(u.View), store.VectorView(v.View), store.WritableVector(slot));
            if (record)
            {
                tape.AppendArrayRecord<T>(ArrayOperation.Cross, slot, u.View, v.View);
            }

            return new(tape, ArrayView.Whole(slot, 3, 1));
        }

        /// <summary>
        /// Computes the sum of all elements, <c>Σ x[i]</c>.
        /// <para>全要素の総和 <c>Σ x[i]</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input vector node. / 入力ベクトルノード。</param>
        /// <returns>A scalar node holding the sum widened to <see langword="double"/>. / 総和を <see langword="double"/> に拡幅したスカラーノード。</returns>
        /// <remarks>
        /// <para>The local derivative is 1 for every element (<c>dx += ds</c>).</para>
        /// <para>各要素に対する局所微分は 1 とします（逆伝播は <c>dx += ds</c>）。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static ScalarNode Sum<T>(VectorNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
        {
            Tape tape = x.Owner();
            var store = tape.Store<T>();
            double value = double.CreateTruncating(XMath.Sum(store.VectorView(x.View)));
            return store.RequiresGradient(x.View.Slot) ? RecordScalar<T>(tape, ArrayOperation.Sum, value, x.View, default) : new(value);
        }

        /// <summary>
        /// Computes the dot product <c>Σ x[i]·y[i]</c>.
        /// <para>内積 <c>Σ x[i]·y[i]</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The first vector node. / 第 1 ベクトルノード。</param>
        /// <param name="y">The second vector node. / 第 2 ベクトルノード。</param>
        /// <returns>A scalar node holding the dot product widened to <see langword="double"/>. / 内積を <see langword="double"/> に拡幅したスカラーノード。</returns>
        /// <remarks>
        /// <para>The local derivatives are <c>y</c> and <c>x</c> respectively (<c>dx += ds·y</c>, <c>dy += ds·x</c>).</para>
        /// <para>局所微分は順に <c>y</c>、<c>x</c> とします（逆伝播は <c>dx += ds·y</c>、<c>dy += ds·x</c>）。</para>
        /// </remarks>
        /// <exception cref="ArgumentException">Thrown if the vectors do not have the same length. / ベクトルの要素数が一致しない場合。</exception>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static ScalarNode Dot<T>(VectorNode<T> x, VectorNode<T> y) where T : unmanaged, IFloatingPointIeee754<T>
        {
            Tape tape = Owner(x.Tape, x.Generation, y.Tape, y.Generation);
            if (x.Length != y.Length)
            {
                throw new ArgumentException("The vectors must have the same length.", nameof(y));
            }

            var store = tape.Store<T>();
            double value = double.CreateTruncating(XMath.Dot(store.VectorView(x.View), store.VectorView(y.View)));
            bool record = store.RequiresGradient(x.View.Slot) || store.RequiresGradient(y.View.Slot);
            return record ? RecordScalar<T>(tape, ArrayOperation.Dot, value, x.View, y.View) : new(value);
        }
    }

    internal static MatrixNode<T> MultiplyCore<T>(MatrixNode<T> a, MatrixNode<T> b) where T : unmanaged, IFloatingPointIeee754<T>
    {
        Tape tape = Owner(a.Tape, a.Generation, b.Tape, b.Generation);
        if (a.Columns != b.Rows)
        {
            throw new ArgumentException("The number of columns of A must equal the number of rows of B.", nameof(b));
        }

        var store = tape.Store<T>();
        bool record = store.RequiresGradient(a.View.Slot) || store.RequiresGradient(b.View.Slot);
        tape.EnsureCanApply(record);

        int slot = store.AddPooled(a.Rows, b.Columns, record);
        XMath.Multiply(store.MatrixView(a.View), store.MatrixView(b.View), store.WritableMatrix(slot));
        if (record)
        {
            tape.AppendArrayRecord<T>(ArrayOperation.MatrixMultiply, slot, a.View, b.View);
        }

        return new(tape, ArrayView.Whole(slot, a.Rows, b.Columns));
    }

    internal static VectorNode<T> MultiplyCore<T>(MatrixNode<T> a, VectorNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
    {
        Tape tape = Owner(a.Tape, a.Generation, x.Tape, x.Generation);
        if (a.Columns != x.Length)
        {
            throw new ArgumentException("The number of columns of A must equal the length of x.", nameof(x));
        }

        var store = tape.Store<T>();
        bool record = store.RequiresGradient(a.View.Slot) || store.RequiresGradient(x.View.Slot);
        tape.EnsureCanApply(record);

        int slot = store.AddPooled(a.Rows, 1, record);
        XMath.Multiply(store.MatrixView(a.View), store.VectorView(x.View), store.WritableVector(slot));
        if (record)
        {
            tape.AppendArrayRecord<T>(ArrayOperation.MatrixVectorMultiply, slot, a.View, x.View);
        }

        return new(tape, ArrayView.Whole(slot, a.Rows, 1));
    }

    private static ScalarNode RecordScalar<T>(Tape tape, ArrayOperation operation, double value, in ArrayView left, in ArrayView right)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        int index = tape.AppendScalarOutput();
        tape.AppendArrayRecord<T>(operation, index, left, right);
        return new(value, tape, index, tape.Generation);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Tape Owner(Tape? first, int firstGeneration, Tape? second, int secondGeneration)
    {
        if (first is null || second is null)
        {
            VectorNode<double>.ThrowUninitialized();
        }

        first.ValidateGeneration(firstGeneration);
        second.ValidateGeneration(secondGeneration);
        if (first != second)
        {
            ThrowDifferentTapes();
        }

        return first;
    }

    [DoesNotReturn]
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void ThrowDifferentTapes() => throw new InvalidOperationException("Nodes must belong to the same tape.");
}
