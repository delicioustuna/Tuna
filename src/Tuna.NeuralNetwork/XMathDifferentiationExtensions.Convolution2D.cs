using System.Numerics;
using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.NeuralNetwork;

public static partial class XMathDifferentiationExtensions
{
    extension(XMath)
    {
        /// <summary>
        /// Records a two-dimensional convolution with optional bias.
        /// <para>省略可能なバイアスを持つ2次元畳み込みを記録します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see cref="float"/> or <see cref="double"/>). / 浮動小数点の要素型（<see cref="float"/> または <see cref="double"/>）。</typeparam>
        /// <param name="x">Features of shape <c>[inChannels, width, height]</c>. / 形状 <c>[入力チャネル, 幅, 高さ]</c> の特徴マップ。</param>
        /// <param name="weights">A matrix with one output channel per row. Columns are ordered by kernel row, kernel column, and input channel, with input channel varying fastest. / 出力チャネルを行とする重み行列。列はカーネル行、カーネル列、入力チャネルの順で、入力チャネルが最も速く変化します。</param>
        /// <param name="kernelHeight">The kernel height. / カーネルの高さ。</param>
        /// <param name="kernelWidth">The kernel width. / カーネルの幅。</param>
        /// <param name="padding">The zero-padding on each edge. / 各辺のゼロパディング。</param>
        /// <param name="bias">An optional bias with one element per output channel. / 出力チャネルごとに 1 要素を持つ省略可能なバイアス。</param>
        /// <param name="strideHeight">The stride along the height axis. / 高さ方向のストライド。</param>
        /// <param name="strideWidth">The stride along the width axis. / 幅方向のストライド。</param>
        /// <param name="dilationHeight">The dilation along the height axis. / 高さ方向のダイレーション。</param>
        /// <param name="dilationWidth">The dilation along the width axis. / 幅方向のダイレーション。</param>
        /// <returns>A tensor node of shape <c>[outChannels, outputWidth, outputHeight]</c>. / 形状 <c>[出力チャネル, 出力幅, 出力高さ]</c> のテンソルノード。</returns>
        /// <remarks>
        /// <para>The forward pass uses the same patch packing and matrix product as <see cref="Convolution2D{T}"/>. The pullback recomputes input patches when a weight gradient is needed.</para>
        /// <para>順伝播では <see cref="Convolution2D{T}"/> と同じパッチの詰め直しと行列積を使用します。重み勾配が必要な場合は、逆伝播で入力パッチを再計算します。</para>
        /// </remarks>
        /// <exception cref="ArgumentOutOfRangeException">A channel count, kernel dimension, stride, or dilation is less than one, or padding is negative. / チャネル数、カーネルサイズ、ストライド、ダイレーションのいずれかが 1 未満、またはパディングが負の場合。</exception>
        /// <exception cref="ArgumentException">The weight or bias shape does not match the convolution. / 重みまたはバイアスの形状が畳み込みと一致しない場合。</exception>
        /// <exception cref="OverflowException">The input, kernel, or output element count exceeds <see cref="int.MaxValue"/>. / 入力、カーネル、または出力の要素数が <see cref="int.MaxValue"/> を超える場合。</exception>
        /// <exception cref="InvalidOperationException">A node is invalid for the tape or recording requires a reset. / ノードがテープで無効、または記録にリセットが必要な場合。</exception>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> is neither <see cref="float"/> nor <see cref="double"/>. / <typeparamref name="T"/> が <see cref="float"/> でも <see cref="double"/> でもない場合。</exception>
        public static TensorNode<T> Convolve2D<T>(TensorNode<T> x, MatrixNode<T> weights,
            int kernelHeight, int kernelWidth, Padding2D padding, VectorNode<T> bias = default,
            int strideHeight = 1, int strideWidth = 1, int dilationHeight = 1, int dilationWidth = 1)
            where T : unmanaged, IFloatingPointIeee754<T>
        {
            Guard.RequireElementType<T>();
            _ = x.RequiresGradient;
            _ = weights.RequiresGradient;
            bool hasBias = bias.Length != 0;
            MatrixNode<T> biasColumn = hasBias ? bias.AsColumnMatrix() : default;

            Guard.RequirePositive(x.Rows, nameof(x));
            Guard.RequirePositive(weights.Rows, nameof(weights));
            Guard.RequirePositive(kernelHeight, nameof(kernelHeight));
            Guard.RequirePositive(kernelWidth, nameof(kernelWidth));
            Guard.RequirePositive(strideHeight, nameof(strideHeight));
            Guard.RequirePositive(strideWidth, nameof(strideWidth));
            Guard.RequirePositive(dilationHeight, nameof(dilationHeight));
            Guard.RequirePositive(dilationWidth, nameof(dilationWidth));
            Convolution2D<T>.RequirePadding(padding);
            var shape = new ConvolutionGeometry(x.Rows, kernelHeight, kernelWidth, strideHeight, strideWidth, dilationHeight, dilationWidth);
            Guard.RequireLength(shape.ColumnLength, weights.Columns, nameof(weights));
            Guard.RequireOptionalLength(weights.Rows, bias.Length, nameof(bias));

            int outWidth = Convolution2D<T>.OutputLength(x.Columns, padding.Left, padding.Right, kernelWidth, strideWidth, dilationWidth);
            int outHeight = Convolution2D<T>.OutputLength(x.MatrixCount, padding.Top, padding.Bottom, kernelHeight, strideHeight, dilationHeight);
            _ = checked(weights.Rows * checked(outWidth * outHeight));
            var operation = new ConvolutionOperation<T>(shape, padding, x.Columns, x.MatrixCount, outWidth, outHeight);
            var input = x.ToMatrix();
            return hasBias
                ? TensorNode<T>.Record(operation, [input, weights, biasColumn], weights.Rows, outWidth, outHeight)
                : TensorNode<T>.Record(operation, [input, weights], weights.Rows, outWidth, outHeight);
        }
    }

    private readonly struct ConvolutionOperation<T>(ConvolutionGeometry shape, Padding2D padding,
        int width, int height, int outWidth, int outHeight) : IDifferentiableOperation<T>
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        public void Forward(DifferentiableOperationContext<T> context, MatrixSpan<T> destination)
        {
            var input = new ReadOnlyTensorSpan<T>(context[0].Data, shape.Channels, width, height);
            var output = new TensorSpan<T>(destination.Data, destination.Rows, outWidth, outHeight);
            Convolution2D<T>.ForwardCore(input, context[1], context.Count == 3 ? context[2].Column(0) : default, shape, padding, output);
        }

        public void PullBack(DifferentiableOperationContext<T> context, ReadOnlyMatrixSpan<T> output, ReadOnlyMatrixSpan<T> dy)
        {
            if (dy.IsEmpty)
            {
                return;
            }

            var dx = context.RequiresGradient(0) ? context.GetGradient(0) : default;
            var dw = context.RequiresGradient(1) ? context.GetGradient(1) : default;
            Convolution2D<T>.BackwardCore(context[0], context[1], dy, shape, padding, width, height, outWidth, dx, dw);
            if (context.Count == 3 && context.RequiresGradient(2))
            {
                GradientKernels.AddRowSums(dy, context.GetGradient(2).Column(0));
            }
        }
    }
}
