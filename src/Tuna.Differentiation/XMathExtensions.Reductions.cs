using System.Numerics;
using Tuna.LinearAlgebra;

namespace Tuna.Differentiation;

public static partial class XMathExtensions
{
    internal static ScalarNode ElementAt<T>(VectorNode<T> vector, int index) where T : unmanaged, IFloatingPointIeee754<T>
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(index, vector.Length);
        Tape tape = vector.Owner();
        var store = tape.Store<T>();
        double value = double.CreateTruncating(store.VectorView(vector.View)[index]);
        if (!store.RequiresGradient(vector.View.Slot))
        {
            return value;
        }

        // 1 要素の平均の逆伝播は、同じ要素に随伴を加える。
        var view = vector[index..(index + 1)].View;
        return RecordScalar<T>(tape, ArrayOperation.Mean, value, view, default);
    }

    extension(XMath)
    {
        /// <summary>
        /// Computes the sum of squares <c>Σ x[i]²</c>.
        /// <para>2 乗和 <c>Σ x[i]²</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input vector node. / 入力ベクトルノード。</param>
        /// <returns>A scalar node holding the sum of squares widened to <see langword="double"/>. / 2 乗和を <see langword="double"/> に拡幅したスカラーノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>2x</c>.</para>
        /// <para>局所微分は <c>2x</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static ScalarNode SumOfSquares<T>(VectorNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
        {
            Tape tape = x.Owner();
            var store = tape.Store<T>();
            double value = double.CreateTruncating(XMath.SumOfSquares(store.VectorView(x.View)));
            return store.RequiresGradient(x.View.Slot) ? RecordScalar<T>(tape, ArrayOperation.SumOfSquares, value, x.View, default) : new(value);
        }

        /// <summary>
        /// Computes the arithmetic mean <c>Σ x[i] / n</c>.
        /// <para>算術平均 <c>Σ x[i] / n</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input vector node. / 入力ベクトルノード。</param>
        /// <returns>A scalar node holding the mean widened to <see langword="double"/>, or <c>NaN</c> if <paramref name="x"/> is empty. / 平均を <see langword="double"/> に拡幅したスカラーノード。入力が空の場合は <c>NaN</c>。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>1/n</c> for each element.</para>
        /// <para>各要素に対する局所微分は <c>1/n</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static ScalarNode Mean<T>(VectorNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
        {
            Tape tape = x.Owner();
            var store = tape.Store<T>();
            double value = double.CreateTruncating(XMath.Mean(store.VectorView(x.View)));
            return store.RequiresGradient(x.View.Slot) ? RecordScalar<T>(tape, ArrayOperation.Mean, value, x.View, default) : new(value);
        }

        /// <summary>
        /// Computes the log-sum-exp <c>ln Σ exp(x[i])</c>.
        /// <para>指数の総和の対数 <c>ln Σ exp(x[i])</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input vector node. / 入力ベクトルノード。</param>
        /// <returns>A scalar node holding <c>ln Σ exp(x[i])</c> widened to <see langword="double"/>, or <c>-∞</c> if <paramref name="x"/> is empty. / 指数の総和の対数を <see langword="double"/> に拡幅したスカラーノード。入力が空の場合は <c>-∞</c>。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>softmax(x)</c>. When the maximum is non-finite, weights follow the scalar <c>LogSumExp</c> limit rules (equal share among <c>+∞</c> entries, equal share across all entries if all are <c>-∞</c>, and <c>NaN</c> if any entry is <c>NaN</c>).</para>
        /// <para>局所微分は <c>softmax(x)</c> とします。最大値が有限でない場合はスカラー版 <c>LogSumExp</c> と同じ極限の扱い（<c>+∞</c> がある場合はその要素のみで等分、すべて <c>-∞</c> なら全要素で等分、<c>NaN</c> を含む場合はすべて <c>NaN</c>）に従います。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static ScalarNode LogSumExp<T>(VectorNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
        {
            Tape tape = x.Owner();
            var store = tape.Store<T>();
            double value = double.CreateTruncating(XMath.LogSumExp(store.VectorView(x.View)));
            return store.RequiresGradient(x.View.Slot) ? RecordScalar<T>(tape, ArrayOperation.LogSumExp, value, x.View, default) : new(value);
        }

        /// <summary>
        /// Computes the sum of each column.
        /// <para>各列の総和を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input matrix node. / 入力行列ノード。</param>
        /// <returns>A vector node of length <c>x.Columns</c> holding each column sum. / 各列の総和を格納した長さ <c>x.Columns</c> のベクトルノード。</returns>
        /// <remarks>
        /// <para>Backpropagation accumulates <c>dX[i, j] += ds[j]</c>.</para>
        /// <para>逆伝播では <c>dX[i, j] += ds[j]</c> を加算します。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> ColumnSums<T>(MatrixNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => ReductionCore(x, ArrayOperation.ColumnSums);

        /// <summary>
        /// Computes the sum of each row.
        /// <para>各行の総和を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input matrix node. / 入力行列ノード。</param>
        /// <returns>A vector node of length <c>x.Rows</c> holding each row sum. / 各行の総和を格納した長さ <c>x.Rows</c> のベクトルノード。</returns>
        /// <remarks>
        /// <para>Backpropagation accumulates <c>dX[i, j] += ds[i]</c>.</para>
        /// <para>逆伝播では <c>dX[i, j] += ds[i]</c> を加算します。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> RowSums<T>(MatrixNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => ReductionCore(x, ArrayOperation.RowSums);

        /// <summary>
        /// Computes the sum of squares of each column.
        /// <para>各列の 2 乗和を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input matrix node. / 入力行列ノード。</param>
        /// <returns>A vector node of length <c>x.Columns</c> holding each column's sum of squares. / 各列の 2 乗和を格納した長さ <c>x.Columns</c> のベクトルノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>2X</c>.</para>
        /// <para>局所微分は <c>2X</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> ColumnSumsOfSquares<T>(MatrixNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => ReductionCore(x, ArrayOperation.ColumnSumsOfSquares);

        /// <summary>
        /// Computes the sum of squares of each row.
        /// <para>各行の 2 乗和を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input matrix node. / 入力行列ノード。</param>
        /// <returns>A vector node of length <c>x.Rows</c> holding each row's sum of squares. / 各行の 2 乗和を格納した長さ <c>x.Rows</c> のベクトルノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>2X</c>.</para>
        /// <para>局所微分は <c>2X</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> RowSumsOfSquares<T>(MatrixNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => ReductionCore(x, ArrayOperation.RowSumsOfSquares);

        /// <summary>
        /// Computes the arithmetic mean of each column.
        /// <para>各列の算術平均を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input matrix node. / 入力行列ノード。</param>
        /// <returns>A vector node of length <c>x.Columns</c> holding each column mean (<c>NaN</c> for empty columns). / 各列の平均を格納した長さ <c>x.Columns</c> のベクトルノード（要素が空の列は <c>NaN</c>）。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>1 / x.Rows</c>.</para>
        /// <para>局所微分は <c>1 / x.Rows</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> ColumnMeans<T>(MatrixNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => ReductionCore(x, ArrayOperation.ColumnMeans);

        /// <summary>
        /// Computes the arithmetic mean of each row.
        /// <para>各行の算術平均を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input matrix node. / 入力行列ノード。</param>
        /// <returns>A vector node of length <c>x.Rows</c> holding each row mean (<c>NaN</c> for empty rows). / 各行の平均を格納した長さ <c>x.Rows</c> のベクトルノード（要素が空の行は <c>NaN</c>）。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>1 / x.Columns</c>.</para>
        /// <para>局所微分は <c>1 / x.Columns</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> RowMeans<T>(MatrixNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => ReductionCore(x, ArrayOperation.RowMeans);

        /// <summary>
        /// Computes <c>ln Σ exp</c> for each column.
        /// <para>各列の <c>ln Σ exp</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input matrix node. / 入力行列ノード。</param>
        /// <returns>A vector node of length <c>x.Columns</c> holding each column's log-sum-exp. / 各列の <c>ln Σ exp</c> を格納した長さ <c>x.Columns</c> のベクトルノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>ColumnSoftmax(X)</c>, using the same non-finite limit weights as vector <c>LogSumExp</c>.</para>
        /// <para>局所微分は <c>ColumnSoftmax(X)</c> とし、最大値が有限でない列にはベクトル版 <c>LogSumExp</c> と同じ極限重みを適用します。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> ColumnLogSumExp<T>(MatrixNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => ReductionCore(x, ArrayOperation.ColumnLogSumExp);

        /// <summary>
        /// Computes <c>ln Σ exp</c> for each row.
        /// <para>各行の <c>ln Σ exp</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input matrix node. / 入力行列ノード。</param>
        /// <returns>A vector node of length <c>x.Rows</c> holding each row's log-sum-exp. / 各行の <c>ln Σ exp</c> を格納した長さ <c>x.Rows</c> のベクトルノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>RowSoftmax(X)</c>, using the same non-finite limit weights as vector <c>LogSumExp</c>.</para>
        /// <para>局所微分は <c>RowSoftmax(X)</c> とし、最大値が有限でない行にはベクトル版 <c>LogSumExp</c> と同じ極限重みを適用します。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> RowLogSumExp<T>(MatrixNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => ReductionCore(x, ArrayOperation.RowLogSumExp);

        /// <summary>
        /// Computes the softmax transformation <c>exp(x[i]) / Σ exp(x[k])</c> over the vector.
        /// <para>ベクトル全体のソフトマックス変換 <c>exp(x[i]) / Σ exp(x[k])</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input vector node. / 入力ベクトルノード。</param>
        /// <returns>A vector node of the same length holding the softmax probabilities. / ソフトマックス確率を格納した同じ長さのベクトルノード。</returns>
        /// <remarks>
        /// <para>Backpropagation computes <c>dx += y ∘ (dy - Σ y[k]·dy[k])</c> from the forward output <c>y</c>.</para>
        /// <para>逆伝播では出力 <c>y</c> から <c>dx += y ∘ (dy - Σ y[k]·dy[k])</c> を計算します。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> Softmax<T>(VectorNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => SoftmaxCore(x, ArrayOperation.ColumnSoftmax);

        /// <summary>
        /// Computes the log-softmax transformation <c>x - ln Σ exp(x[k])</c> over the vector.
        /// <para>ベクトル全体の対数ソフトマックス変換 <c>x - ln Σ exp(x[k])</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input vector node. / 入力ベクトルノード。</param>
        /// <returns>A vector node of the same length holding the log-softmax values. / 対数ソフトマックス値を格納した同じ長さのベクトルノード。</returns>
        /// <remarks>
        /// <para>Backpropagation computes <c>dx += dz - exp(z)·Σ dz[k]</c> from the forward output <c>z</c>.</para>
        /// <para>逆伝播では出力 <c>z</c> から <c>dx += dz - exp(z)·Σ dz[k]</c> を計算します。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> LogSoftmax<T>(VectorNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => SoftmaxCore(x, ArrayOperation.ColumnLogSoftmax);

        /// <summary>
        /// Computes the softmax transformation along each column.
        /// <para>各列に対してソフトマックス変換を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input matrix node. / 入力行列ノード。</param>
        /// <returns>A matrix node of the same shape holding column-wise softmax probabilities. / 列ごとのソフトマックス確率を格納した同じ形状の行列ノード。</returns>
        /// <remarks>
        /// <para>Backpropagation computes <c>dX += Y ∘ (dY - Σᵢ Y[i, j]·dY[i, j])</c> per column from the forward output <c>Y</c>.</para>
        /// <para>逆伝播では出力 <c>Y</c> から列ごとに <c>dX += Y ∘ (dY - Σᵢ Y[i, j]·dY[i, j])</c> を計算します。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> ColumnSoftmax<T>(MatrixNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => SoftmaxCore(x, ArrayOperation.ColumnSoftmax);

        /// <summary>
        /// Computes the softmax transformation along each row.
        /// <para>各行に対してソフトマックス変換を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input matrix node. / 入力行列ノード。</param>
        /// <returns>A matrix node of the same shape holding row-wise softmax probabilities. / 行ごとのソフトマックス確率を格納した同じ形状の行列ノード。</returns>
        /// <remarks>
        /// <para>Backpropagation computes <c>dX += Y ∘ (dY - Σⱼ Y[i, j]·dY[i, j])</c> per row from the forward output <c>Y</c>.</para>
        /// <para>逆伝播では出力 <c>Y</c> から行ごとに <c>dX += Y ∘ (dY - Σⱼ Y[i, j]·dY[i, j])</c> を計算します。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> RowSoftmax<T>(MatrixNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => SoftmaxCore(x, ArrayOperation.RowSoftmax);

        /// <summary>
        /// Computes the log-softmax transformation along each column.
        /// <para>各列に対して対数ソフトマックス変換を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input matrix node. / 入力行列ノード。</param>
        /// <returns>A matrix node of the same shape holding column-wise log-softmax values. / 列ごとの対数ソフトマックス値を格納した同じ形状の行列ノード。</returns>
        /// <remarks>
        /// <para>Backpropagation computes <c>dX += dZ - exp(Z)·Σᵢ dZ[i, j]</c> per column from the forward output <c>Z</c>.</para>
        /// <para>逆伝播では出力 <c>Z</c> から列ごとに <c>dX += dZ - exp(Z)·Σᵢ dZ[i, j]</c> を計算します。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> ColumnLogSoftmax<T>(MatrixNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => SoftmaxCore(x, ArrayOperation.ColumnLogSoftmax);

        /// <summary>
        /// Computes the log-softmax transformation along each row.
        /// <para>各行に対して対数ソフトマックス変換を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input matrix node. / 入力行列ノード。</param>
        /// <returns>A matrix node of the same shape holding row-wise log-softmax values. / 行ごとの対数ソフトマックス値を格納した同じ形状の行列ノード。</returns>
        /// <remarks>
        /// <para>Backpropagation computes <c>dX += dZ - exp(Z)·Σⱼ dZ[i, j]</c> per row from the forward output <c>Z</c>.</para>
        /// <para>逆伝播では出力 <c>Z</c> から行ごとに <c>dX += dZ - exp(Z)·Σⱼ dZ[i, j]</c> を計算します。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> RowLogSoftmax<T>(MatrixNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => SoftmaxCore(x, ArrayOperation.RowLogSoftmax);
    }

    private static VectorNode<T> ReductionCore<T>(MatrixNode<T> x, ArrayOperation operation) where T : unmanaged, IFloatingPointIeee754<T>
    {
        Tape tape = x.Owner();
        var store = tape.Store<T>();
        bool record = store.RequiresGradient(x.View.Slot);
        tape.EnsureCanApply(record);

        bool columns = operation is ArrayOperation.ColumnSums or ArrayOperation.ColumnSumsOfSquares
            or ArrayOperation.ColumnMeans or ArrayOperation.ColumnLogSumExp;
        int length = columns ? x.Columns : x.Rows;
        int slot = store.AddPooled(length, 1, record);
        ReadOnlyMatrixSpan<T> values = store.MatrixView(x.View);
        VectorSpan<T> destination = store.WritableVector(slot);
        switch (operation)
        {
            case ArrayOperation.ColumnSums:
                XMath.ColumnSums(values, destination);
                break;
            case ArrayOperation.RowSums:
                XMath.RowSums(values, destination);
                break;
            case ArrayOperation.ColumnSumsOfSquares:
                XMath.ColumnSumsOfSquares(values, destination);
                break;
            case ArrayOperation.RowSumsOfSquares:
                XMath.RowSumsOfSquares(values, destination);
                break;
            case ArrayOperation.ColumnMeans:
                XMath.ColumnMeans(values, destination);
                break;
            case ArrayOperation.RowMeans:
                XMath.RowMeans(values, destination);
                break;
            case ArrayOperation.ColumnLogSumExp:
                XMath.ColumnLogSumExp(values, destination);
                break;
            default:
                XMath.RowLogSumExp(values, destination);
                break;
        }

        if (record)
        {
            tape.AppendArrayRecord<T>(operation, slot, x.View, default);
        }

        return new(tape, ArrayView.Whole(slot, length, 1));
    }

    private static VectorNode<T> SoftmaxCore<T>(VectorNode<T> x, ArrayOperation operation) where T : unmanaged, IFloatingPointIeee754<T>
    {
        Tape tape = x.Owner();
        var store = tape.Store<T>();
        bool record = store.RequiresGradient(x.View.Slot);
        tape.EnsureCanApply(record);

        int slot = store.AddPooled(x.Length, 1, record);
        if (operation == ArrayOperation.ColumnSoftmax)
        {
            XMath.Softmax(store.VectorView(x.View), store.WritableVector(slot));
        }
        else
        {
            XMath.LogSoftmax(store.VectorView(x.View), store.WritableVector(slot));
        }

        if (record)
        {
            tape.AppendArrayRecord<T>(operation, slot, x.View, default);
        }

        return new(tape, ArrayView.Whole(slot, x.Length, 1));
    }

    private static MatrixNode<T> SoftmaxCore<T>(MatrixNode<T> x, ArrayOperation operation) where T : unmanaged, IFloatingPointIeee754<T>
    {
        Tape tape = x.Owner();
        var store = tape.Store<T>();
        bool record = store.RequiresGradient(x.View.Slot);
        tape.EnsureCanApply(record);

        int slot = store.AddPooled(x.Rows, x.Columns, record);
        ReadOnlyMatrixSpan<T> values = store.MatrixView(x.View);
        MatrixSpan<T> destination = store.WritableMatrix(slot);
        switch (operation)
        {
            case ArrayOperation.ColumnSoftmax:
                XMath.ColumnSoftmax(values, destination);
                break;
            case ArrayOperation.RowSoftmax:
                XMath.RowSoftmax(values, destination);
                break;
            case ArrayOperation.ColumnLogSoftmax:
                XMath.ColumnLogSoftmax(values, destination);
                break;
            default:
                XMath.RowLogSoftmax(values, destination);
                break;
        }

        if (record)
        {
            tape.AppendArrayRecord<T>(operation, slot, x.View, default);
        }

        return new(tape, ArrayView.Whole(slot, x.Rows, x.Columns));
    }
}
