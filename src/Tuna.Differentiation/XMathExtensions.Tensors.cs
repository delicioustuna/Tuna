using System.Numerics;
using Tuna.LinearAlgebra;

namespace Tuna.Differentiation;

public static partial class XMathExtensions
{
    // ブロードキャスト随伴の一時テンソル上限（float で 16 MiB、double で 32 MiB）。これを超えるバッチはチャンク分割で集約する。
    internal const int MaxBroadcastScratchElements = 1 << 22;

    extension(XMath)
    {
        /// <summary>
        /// Records the batched product of the matrices at matching indices.
        /// <para>インデックスが一致する行列ごとの積を記録します。</para>
        /// </summary>
        /// <typeparam name="T">The element type, either float or double. / 要素型。float または double。</typeparam>
        /// <param name="a">The left operand. / 左辺のテンソル。</param>
        /// <param name="b">The right operand. / 右辺のテンソル。</param>
        /// <returns>The differentiable batch of products. / 行列積のバッチを表す微分可能なノード。</returns>
        /// <remarks>
        /// <para>The forward pass uses the tensor overload of <see cref="XMath.Multiply{T}(ReadOnlyTensorSpan{T}, ReadOnlyTensorSpan{T})"/>. Backpropagation sums the contributions to a broadcast operand over the matrix axis.</para>
        /// <para>順伝播ではテンソルを受け取る <see cref="XMath.Multiply{T}(ReadOnlyTensorSpan{T}, ReadOnlyTensorSpan{T})"/> を使用します。逆伝播ではブロードキャストされた入力への寄与を行列軸に沿って合計します。</para>
        /// </remarks>
        /// <exception cref="ArgumentException">The inner dimensions or matrix counts do not match. / 積の内側の次元または行列数が一致しない場合。</exception>
        /// <exception cref="OverflowException">The output element count or flattened column count exceeds <see cref="int.MaxValue"/>. / 出力の要素数またはフラットな列数が <see cref="int.MaxValue"/> を超える場合。</exception>
        /// <exception cref="InvalidOperationException">An input is invalid for the tape or recording requires a reset. / 入力がテープで無効、または記録にリセットが必要な場合。</exception>
        public static TensorNode<T> Multiply<T>(TensorNode<T> a, TensorNode<T> b)
            where T : unmanaged, IFloatingPointIeee754<T>
        {
            _ = Owner(a.Storage.Tape, a.Storage.Generation, b.Storage.Tape, b.Storage.Generation);
            if (a.Columns != b.Rows || a.MatrixCount != b.MatrixCount)
            {
                throw new ArgumentException("The inner dimensions and matrix counts must match.", nameof(b));
            }

            var output = TensorView.Whole(a.Rows, b.Columns, a.MatrixCount);
            var node = MatrixNode<T>.Record(new BatchProduct<T>(a.View, b.View, output), [a.Storage, b.Storage],
                output.Rows, output.FlatColumns);
            return new(node, output);
        }

        /// <summary>
        /// Records the sum over the matrix index axis of a tensor.
        /// <para>テンソルの行列インデックス軸に沿う総和を記録します。</para>
        /// </summary>
        /// <typeparam name="T">The element type, either float or double. / 要素型。float または double。</typeparam>
        /// <param name="x">The input tensor node. / 入力のテンソルノード。</param>
        /// <returns>The matrix of sums, or a zero matrix for an empty batch. / 総和の行列。バッチが空の場合はゼロ行列。</returns>
        /// <exception cref="InvalidOperationException">The node is invalid or recording requires a reset. / ノードが無効、または記録にリセットが必要な場合。</exception>
        public static MatrixNode<T> SumMatrices<T>(TensorNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => MatrixNode<T>.Record(new BatchSum<T>(x.View), [x.Storage], x.Rows, x.Columns);
    }

    private readonly struct BatchProduct<T>(TensorView a, TensorView b, TensorView output) : IDifferentiableOperation<T>
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        public void Forward(DifferentiableOperationContext<T> context, MatrixSpan<T> destination)
            => XMath.Multiply(a.Read(context[0]), b.Read(context[1]), output.Write(destination));

        public void PullBack(DifferentiableOperationContext<T> context, ReadOnlyMatrixSpan<T> value, ReadOnlyMatrixSpan<T> gradient)
        {
            if (output.IsEmpty)
            {
                return;
            }

            var dy = output.Read(gradient);
            if (!a.IsEmpty && context.RequiresGradient(0))
            {
                AddBatchProduct(context, 0, a, dy, b.Read(context[1]).T);
            }

            if (!b.IsEmpty && context.RequiresGradient(1))
            {
                AddBatchProduct(context, 1, b, a.Read(context[0]).T, dy);
            }
        }
    }

    private readonly struct BatchSum<T>(TensorView x) : IDifferentiableOperation<T>
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        public void Forward(DifferentiableOperationContext<T> context, MatrixSpan<T> destination)
            => XMath.SumMatrices(x.Read(context[0]), destination);

        public void PullBack(DifferentiableOperationContext<T> context, ReadOnlyMatrixSpan<T> value, ReadOnlyMatrixSpan<T> gradient)
        {
            if (!x.IsEmpty)
            {
                bool allowOverwrite = x.IsWhole || x.CoversRepeatedSource;
                var storage = context.GetGradient(0, allowOverwrite, out bool overwrite);
                AddBatchGradient(context.Store, x, storage, gradient.Broadcast(x.Count), overwrite);
            }
        }
    }

    internal static MatrixNode<T> PackMatrices<T>(TensorNode<T> x, int columns) where T : unmanaged, IFloatingPointIeee754<T>
        => MatrixNode<T>.Record(new PackMatricesOperation<T>(x.View), [x.Storage], x.Rows, columns);

    private readonly struct PackMatricesOperation<T>(TensorView source) : IDifferentiableOperation<T>
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        public void Forward(DifferentiableOperationContext<T> context, MatrixSpan<T> destination)
            => source.Read(context[0]).CopyTo(new TensorSpan<T>(destination.Data, source.Rows, source.Columns, source.Count));

        public void PullBack(DifferentiableOperationContext<T> context, ReadOnlyMatrixSpan<T> output, ReadOnlyMatrixSpan<T> gradient)
        {
            if (!source.IsEmpty)
            {
                var dy = new ReadOnlyTensorSpan<T>(gradient.Data, source.Rows, source.Columns, source.Count);
                bool allowOverwrite = source.IsWhole || source.CoversRepeatedSource;
                var storage = context.GetGradient(0, allowOverwrite, out bool overwrite);
                AddBatchGradient(context.Store, source, storage, dy, overwrite);
            }
        }
    }

    private static void AddBatchProduct<T>(DifferentiableOperationContext<T> context, int inputIndex, TensorView target,
        ReadOnlyTensorSpan<T> a, ReadOnlyTensorSpan<T> b)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        var store = context.Store;
        if (!target.Repeated)
        {
            var storage = context.GetGradient(inputIndex, target.IsWhole, out bool overwrite);
            if (overwrite)
            {
                XMath.Multiply(a, b, target.Write(storage));
                return;
            }

            int length = checked(target.Rows * target.Columns * target.Count);
            var product = new TensorSpan<T>(store.Scratch(ArrayStore<T>.MatrixContribution, length), target.Rows, target.Columns, target.Count);
            XMath.Multiply(a, b, product);
            var destination = target.Write(storage);
            XMath.Add(destination.AsReadOnly(), product.AsReadOnly(), destination);
            return;
        }

        var targetStorage = context.GetGradient(inputIndex, target.CoversRepeatedSource, out bool overwriteRepeated);
        var repeatedDestination = target.RepeatedSource(targetStorage);
        int matrixLength = checked(target.Rows * target.Columns);
        int chunkSize = Math.Max(1, MaxBroadcastScratchElements / matrixLength);
        if (chunkSize >= target.Count)
        {
            int length = checked(matrixLength * target.Count);
            var product = new TensorSpan<T>(store.Scratch(ArrayStore<T>.MatrixContribution, length), target.Rows, target.Columns, target.Count);
            XMath.Multiply(a, b, product);
            AddRepeatedSum(store, repeatedDestination, product.AsReadOnly(), matrixLength, overwriteRepeated);
            return;
        }

        for (int start = 0; start < target.Count; start += chunkSize)
        {
            int count = Math.Min(chunkSize, target.Count - start);
            var range = start..(start + count);
            var product = new TensorSpan<T>(store.Scratch(ArrayStore<T>.MatrixContribution, checked(matrixLength * count)),
                target.Rows, target.Columns, count);
            XMath.Multiply(a[.., .., range], b[.., .., range], product);
            AddRepeatedSum(store, repeatedDestination, product.AsReadOnly(), matrixLength, overwriteRepeated && start == 0);
        }
    }

    private static void AddBatchGradient<T>(ArrayStore<T> store, TensorView target, MatrixSpan<T> storage,
        ReadOnlyTensorSpan<T> gradient, bool overwrite)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (!target.Repeated)
        {
            var destination = target.Write(storage);
            if (overwrite)
            {
                gradient.CopyTo(destination);
            }
            else
            {
                XMath.Add(destination.AsReadOnly(), gradient, destination);
            }

            return;
        }

        int matrixLength = checked(target.Rows * target.Columns);
        AddRepeatedSum(store, target.RepeatedSource(storage), gradient, matrixLength, overwrite);
    }

    private static void AddRepeatedSum<T>(ArrayStore<T> store, MatrixSpan<T> destination, ReadOnlyTensorSpan<T> gradient,
        int matrixLength, bool overwrite)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (overwrite)
        {
            XMath.SumMatrices(gradient, destination);
            return;
        }

        var sum = new MatrixSpan<T>(store.Scratch(ArrayStore<T>.ReducedValues, matrixLength), destination.Rows, destination.Columns);
        XMath.SumMatrices(gradient, sum);
        XMath.Add(destination.AsReadOnly(), sum.AsReadOnly(), destination);
    }
}
