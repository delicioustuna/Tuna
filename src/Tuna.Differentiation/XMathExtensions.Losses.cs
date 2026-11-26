using System.Numerics;
using Tuna.LinearAlgebra;

namespace Tuna.Differentiation;

public static partial class XMathExtensions
{
    extension(XMath)
    {
        /// <summary>
        /// Computes the summed softmax cross-entropy loss <c>-Σₙ ln softmax(Z[:, n])[t[n]]</c> for integer class labels.
        /// <para>整数クラスラベルに対するソフトマックス交差エントロピーの総和 <c>-Σₙ ln softmax(Z[:, n])[t[n]]</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="logits">The class-score matrix node where each column corresponds to one sample. / 各列が 1 標本のクラススコアを表す行列ノード。</param>
        /// <param name="labels">The zero-based class indices for each sample; the caller may mutate the span after this call returns. / 各標本の 0 始まりクラスインデックス列（呼び出し後に書き換えても影響ありません）。</param>
        /// <returns>A scalar node holding the sum of per-sample losses (<c>+∞</c> if a selected log-softmax is <c>-∞</c>, <c>NaN</c> if <c>NaN</c>). / 全標本の損失の和を値とするスカラーノード（選択クラスの対数ソフトマックスが <c>-∞</c> なら <c>+∞</c>、<c>NaN</c> なら <c>NaN</c>）。</returns>
        /// <remarks>
        /// <para>
        /// The local derivative is <c>P - onehot(t)</c>, where <c>P = exp(logsoftmax(Z))</c>.
        /// Columns whose maximum is <c>+∞</c> split weight equally among the <c>+∞</c> entries; columns whose entries are all <c>-∞</c> split weight equally among all entries; columns containing <c>NaN</c> receive <c>NaN</c> weights.
        /// </para>
        /// <para>
        /// 局所微分は <c>P = exp(logsoftmax(Z))</c> として <c>P - onehot(t)</c> とします。
        /// 最大値が <c>+∞</c> の列は <c>+∞</c> の要素のみで、全要素が <c>-∞</c> の列は全要素で重みを等分します。<c>NaN</c> を含む列の重みはすべて <c>NaN</c> とします。
        /// </para>
        /// </remarks>
        /// <exception cref="ArgumentException">Thrown if <paramref name="labels"/> length does not equal the column count of <paramref name="logits"/>. / <paramref name="labels"/> の長さが <paramref name="logits"/> の列数と一致しない場合。</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown if any label is negative or greater than or equal to the row count of <paramref name="logits"/>. / ラベルが負、または <paramref name="logits"/> の行数以上の場合。</exception>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static ScalarNode SoftmaxCrossEntropy<T>(MatrixNode<T> logits, ReadOnlySpan<int> labels) where T : unmanaged, IFloatingPointIeee754<T>
        {
            Tape tape = logits.Owner();
            if (labels.Length != logits.Columns)
            {
                throw new ArgumentException("The number of labels must equal the number of columns of the logits.", nameof(labels));
            }

            ValidateLabels(labels, logits.Rows, nameof(labels));
            return SoftmaxCrossEntropyCore<T>(tape, logits.View, labels);
        }

        /// <summary>
        /// Computes the softmax cross-entropy loss <c>-ln softmax(z)[t]</c> for a single sample.
        /// <para>単一標本のソフトマックス交差エントロピー <c>-ln softmax(z)[t]</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="logits">The class-score vector node for one sample. / 1 標本のクラススコアを表すベクトルノード。</param>
        /// <param name="label">The zero-based class index. / 0 始まりのクラスインデックス。</param>
        /// <returns>A scalar node holding the loss for the sample. / 1 標本の損失を値とするスカラーノード。</returns>
        /// <remarks>
        /// <para>Produces the same value and local derivative as calling the matrix overload with a single-column matrix.</para>
        /// <para>1 列の行列を渡した場合と同一の値および局所微分を計算します。</para>
        /// </remarks>
        /// <exception cref="ArgumentOutOfRangeException">Thrown if <paramref name="label"/> is negative or greater than or equal to the length of <paramref name="logits"/>. / <paramref name="label"/> が負、または <paramref name="logits"/> の要素数以上の場合。</exception>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static ScalarNode SoftmaxCrossEntropy<T>(VectorNode<T> logits, int label) where T : unmanaged, IFloatingPointIeee754<T>
        {
            Tape tape = logits.Owner();
            ReadOnlySpan<int> labels = new(in label);
            ValidateLabels(labels, logits.Length, nameof(label));
            return SoftmaxCrossEntropyCore<T>(tape, logits.View, labels);
        }

        /// <summary>
        /// Computes the summed softmax cross-entropy loss <c>-Σₙ Σₖ T[k, n]·ln softmax(Z[:, n])[k]</c> against target distributions.
        /// <para>目標分布に対するソフトマックス交差エントロピーの総和 <c>-Σₙ Σₖ T[k, n]·ln softmax(Z[:, n])[k]</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="logits">The class-score matrix node where each column corresponds to one sample. / 各列が 1 標本のクラススコアを表す行列ノード。</param>
        /// <param name="targets">The target distribution matrix node with the same shape as <paramref name="logits"/>. / <paramref name="logits"/> と同じ形状の目標分布行列ノード。</param>
        /// <returns>A scalar node holding the sum of per-sample losses (entries where the target is exact 0 contribute 0). / 全標本の損失の和を値とするスカラーノード（目標値が厳密に 0 の項は 0 とします）。</returns>
        /// <remarks>
        /// <para>With <c>L = logsoftmax(Z)</c>, the local derivatives are <c>P ∘ s - T</c> with respect to <paramref name="logits"/> and <c>-L</c> with respect to <paramref name="targets"/>, where <c>s</c> broadcasts the column sums of <c>T</c> across rows. Non-finite weight rules for <c>P</c> match the integer-label overload.</para>
        /// <para><c>L = logsoftmax(Z)</c> として、局所微分は <paramref name="logits"/> について <c>P ∘ s - T</c>、<paramref name="targets"/> について <c>-L</c> とします（<c>s</c> は <c>T</c> の列和を各行へブロードキャストしたもの）。<c>P</c> の非有限値における重みの扱いは整数ラベル向けオーバーロードと同一です。</para>
        /// </remarks>
        /// <exception cref="ArgumentException">Thrown if <paramref name="targets"/> does not have the same shape as <paramref name="logits"/>. / <paramref name="logits"/> と <paramref name="targets"/> の形状が一致しない場合。</exception>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static ScalarNode SoftmaxCrossEntropy<T>(MatrixNode<T> logits, MatrixNode<T> targets) where T : unmanaged, IFloatingPointIeee754<T>
        {
            Tape tape = Owner(logits.Tape, logits.Generation, targets.Tape, targets.Generation);
            if (logits.Rows != targets.Rows || logits.Columns != targets.Columns)
            {
                throw new ArgumentException("The targets must have the same shape as the logits.", nameof(targets));
            }

            return SoftmaxCrossEntropyCore<T>(tape, logits.View, targets.View);
        }

        /// <summary>
        /// Computes the softmax cross-entropy loss <c>-Σₖ t[k]·ln softmax(z)[k]</c> for a single sample against a target distribution.
        /// <para>単一標本における目標分布とのソフトマックス交差エントロピー <c>-Σₖ t[k]·ln softmax(z)[k]</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="logits">The class-score vector node for one sample. / 1 標本のクラススコアを表すベクトルノード。</param>
        /// <param name="targets">The target distribution vector node with the same length as <paramref name="logits"/>. / <paramref name="logits"/> と同じ長さの目標分布ベクトルノード。</param>
        /// <returns>A scalar node holding the loss for the sample. / 1 標本の損失を値とするスカラーノード。</returns>
        /// <remarks>
        /// <para>Produces the same value and local derivatives as calling the matrix overload with single-column matrices.</para>
        /// <para>1 列の行列を渡した場合と同一の値および局所微分を計算します。</para>
        /// </remarks>
        /// <exception cref="ArgumentException">Thrown if <paramref name="targets"/> does not have the same length as <paramref name="logits"/>. / <paramref name="logits"/> と <paramref name="targets"/> の要素数が一致しない場合。</exception>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static ScalarNode SoftmaxCrossEntropy<T>(VectorNode<T> logits, VectorNode<T> targets) where T : unmanaged, IFloatingPointIeee754<T>
        {
            Tape tape = Owner(logits.Tape, logits.Generation, targets.Tape, targets.Generation);
            if (logits.Length != targets.Length)
            {
                throw new ArgumentException("The targets must have the same length as the logits.", nameof(targets));
            }

            return SoftmaxCrossEntropyCore<T>(tape, logits.View, targets.View);
        }

        /// <summary>
        /// Computes the summed logistic (binary) cross-entropy loss <c>Σᵢ [yᵢ·softplus(-ηᵢ) + (1 - yᵢ)·softplus(ηᵢ)]</c>.
        /// <para>ロジスティック交差エントロピーの総和 <c>Σᵢ [yᵢ·softplus(-ηᵢ) + (1 - yᵢ)·softplus(ηᵢ)]</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="logits">The vector node of per-sample logits <c>η</c>. / 標本ごとのロジット <c>η</c> を格納したベクトルノード。</param>
        /// <param name="targets">The target values (0, 1, or probabilities) with the same length as <paramref name="logits"/>. / 各要素の目標値（0、1、または確率）を表す同じ長さのベクトルノード。</param>
        /// <returns>A scalar node holding the sum of elementwise losses (terms with exact zero target coefficients contribute 0). / 各要素の損失の和を値とするスカラーノード（目標値係数が厳密に 0 の項は 0 とします）。</returns>
        /// <remarks>
        /// <para>The local derivatives are <c>σ(η) - y</c> with respect to <paramref name="logits"/> and <c>-η</c> with respect to <paramref name="targets"/>, evaluated stably in the tails.</para>
        /// <para>局所微分は <paramref name="logits"/> について <c>σ(η) - y</c>、<paramref name="targets"/> について <c>-η</c> とし、裾の領域でも精度を保って計算します。</para>
        /// </remarks>
        /// <exception cref="ArgumentException">Thrown if <paramref name="targets"/> does not have the same length as <paramref name="logits"/>. / <paramref name="logits"/> と <paramref name="targets"/> の要素数が一致しない場合。</exception>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static ScalarNode LogisticCrossEntropy<T>(VectorNode<T> logits, VectorNode<T> targets) where T : unmanaged, IFloatingPointIeee754<T>
        {
            Tape tape = Owner(logits.Tape, logits.Generation, targets.Tape, targets.Generation);
            if (logits.Length != targets.Length)
            {
                throw new ArgumentException("The targets must have the same length as the logits.", nameof(targets));
            }

            return LogisticCrossEntropyCore<T>(tape, logits.View, targets.View, matrix: false);
        }

        /// <summary>
        /// Computes the summed logistic (binary) cross-entropy loss <c>Σᵢⱼ [Yᵢⱼ·softplus(-Hᵢⱼ) + (1 - Yᵢⱼ)·softplus(Hᵢⱼ)]</c> over a matrix of independent binary logits.
        /// <para>独立な 2 値ロジット行列に対するロジスティック交差エントロピーの総和 <c>Σᵢⱼ [Yᵢⱼ·softplus(-Hᵢⱼ) + (1 - Yᵢⱼ)·softplus(Hᵢⱼ)]</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="logits">The matrix node of logits where each column corresponds to one sample. / 各列が 1 標本のロジットを表す行列ノード。</param>
        /// <param name="targets">The target values (0, 1, or probabilities) with the same shape as <paramref name="logits"/>. / 各要素の目標値（0、1、または確率）を表す同じ形状の行列ノード。</param>
        /// <returns>A scalar node holding the sum of elementwise losses (terms with exact zero target coefficients contribute 0). / 各要素の損失の和を値とするスカラーノード（目標値係数が厳密に 0 の項は 0 とします）。</returns>
        /// <remarks>
        /// <para>The local derivatives are <c>σ(H) - Y</c> with respect to <paramref name="logits"/> and <c>-H</c> with respect to <paramref name="targets"/>.</para>
        /// <para>局所微分は <paramref name="logits"/> について <c>σ(H) - Y</c>、<paramref name="targets"/> について <c>-H</c> とします。</para>
        /// </remarks>
        /// <exception cref="ArgumentException">Thrown if <paramref name="targets"/> does not have the same shape as <paramref name="logits"/>. / <paramref name="logits"/> と <paramref name="targets"/> の形状が一致しない場合。</exception>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static ScalarNode LogisticCrossEntropy<T>(MatrixNode<T> logits, MatrixNode<T> targets) where T : unmanaged, IFloatingPointIeee754<T>
        {
            Tape tape = Owner(logits.Tape, logits.Generation, targets.Tape, targets.Generation);
            if (logits.Rows != targets.Rows || logits.Columns != targets.Columns)
            {
                throw new ArgumentException("The targets must have the same shape as the logits.", nameof(targets));
            }

            return LogisticCrossEntropyCore<T>(tape, logits.View, targets.View, matrix: true);
        }
    }

    private static void ValidateLabels(ReadOnlySpan<int> labels, int classes, string parameterName)
    {
        foreach (int label in labels)
        {
            if ((uint)label >= (uint)classes)
            {
                throw new ArgumentOutOfRangeException(parameterName, label, "Each label must be at least 0 and less than the number of classes.");
            }
        }
    }

    private static ScalarNode SoftmaxCrossEntropyCore<T>(Tape tape, in ArrayView z, ReadOnlySpan<int> labels) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var store = tape.Store<T>();
        bool record = store.RequiresGradient(z.Slot);
        int keep = -1;
        tape.EnsureCanApply(record);
        if (record)
        {
            keep = store.AddPooled(z.Rows, z.Columns, requiresGradient: false);
        }

        double value = store.SoftmaxCrossEntropy(z, labels, keep);
        return record ? RecordLoss<T>(tape, ArrayOperation.SoftmaxCrossEntropyLabels, value, z, default, keep) : new(value);
    }

    private static ScalarNode SoftmaxCrossEntropyCore<T>(Tape tape, in ArrayView z, in ArrayView t) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var store = tape.Store<T>();
        bool record = store.RequiresGradient(z.Slot) || store.RequiresGradient(t.Slot);
        int keep = -1;
        tape.EnsureCanApply(record);
        if (record)
        {
            keep = store.AddPooled(z.Rows, z.Columns, requiresGradient: false);
        }

        double value = store.SoftmaxCrossEntropy(z, t, keep);
        return record ? RecordLoss<T>(tape, ArrayOperation.SoftmaxCrossEntropyTargets, value, z, t, keep) : new(value);
    }

    private static ScalarNode LogisticCrossEntropyCore<T>(Tape tape, in ArrayView eta, in ArrayView y, bool matrix) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var store = tape.Store<T>();
        bool record = store.RequiresGradient(eta.Slot) || store.RequiresGradient(y.Slot);
        tape.EnsureCanApply(record);

        double value = store.LogisticCrossEntropy(eta, y, matrix);
        return record ? RecordLoss<T>(tape, ArrayOperation.LogisticCrossEntropy, value, eta, y, -1) : new(value);
    }

    private static ScalarNode RecordLoss<T>(Tape tape, ArrayOperation operation, double value, in ArrayView left, in ArrayView right, int keep)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        int index = tape.AppendScalarOutput();
        tape.AppendArrayRecord<T>(operation, index, left, right, 0, keep);
        return new(value, tape, index, tape.Generation);
    }
}
