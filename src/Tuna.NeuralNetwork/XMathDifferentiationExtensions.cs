using System.ComponentModel;
using System.Numerics;
using Tuna.Differentiation;

namespace Tuna.NeuralNetwork;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む。
using Tuna.LinearAlgebra;

/// <summary>
/// Provides neural network layers on automatic differentiation nodes as <see cref="XMath"/> extensions, so that their gradients can be computed.
/// <para>自動微分のノードに対するニューラルネットワークの層を、勾配を計算できる形で記録する <see cref="XMath"/> 拡張として提供します。</para>
/// </summary>
/// <remarks>
/// <para>Each layer records one operation on the tape of its inputs. Output values are bit-for-bit identical to the corresponding <see cref="XMath"/> inference operation applied to the input values. When every input is a constant, only the forward value is computed and nothing is recorded. Gradients are bit-for-bit identical regardless of the degree of parallelism.</para>
/// <para>各層は入力のテープに1つの演算として記録されます。出力値は、入力値に対応する <see cref="XMath"/> の推論演算を適用した結果とビット単位で一致します。入力がすべて定数の場合は値だけを計算し、記録しません。勾配は並列数によらずビット単位で一致します。</para>
/// <para>Feature matrices use rows for channels and columns for positions, as in the <see cref="XMath"/> inference operations. Supported element types are <see cref="float"/> and <see cref="double"/>.</para>
/// <para>特徴量行列は <see cref="XMath"/> の推論演算と同じく、行をチャネル、列を位置として扱います。対応する要素型は <see cref="float"/> および <see cref="double"/> です。</para>
/// </remarks>
[EditorBrowsable(EditorBrowsableState.Never)]
public static partial class XMathDifferentiationExtensions
{
    extension(XMath)
    {
        /// <summary>
        /// Records the linear (affine) transformation <c>y = W * x + b</c>.
        /// <para>全結合（アフィン変換）<c>y = W * x + b</c> を記録します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see cref="float"/> or <see cref="double"/>). / 浮動小数点の要素型（<see cref="float"/> または <see cref="double"/>）。</typeparam>
        /// <param name="weight">The weight node of shape <c>[outChannels × inChannels]</c>. / 重みのノード <c>[出力チャネル × 入力チャネル]</c>。</param>
        /// <param name="x">The input node of shape <c>[inChannels × positions]</c>. / 入力のノード <c>[入力チャネル × 位置]</c>。</param>
        /// <param name="bias">The per-output-channel bias node of length <c>weight.Rows</c>. / 出力チャネルごとのバイアスのノード（長さは重みの行数）。</param>
        /// <returns>The output node of shape <c>[outChannels × positions]</c>. / <c>[出力チャネル × 位置]</c> の出力ノード。</returns>
        /// <remarks>
        /// <para>The pullback adds <c>dY·xᵀ</c> to the weight gradient, <c>Wᵀ·dY</c> to the input gradient, and the row sums of <c>dY</c> to the bias gradient. Each is computed only for inputs that require a gradient.</para>
        /// <para>逆伝播では、重みの勾配に <c>dY·xᵀ</c>、入力の勾配に <c>Wᵀ·dY</c>、バイアスの勾配に <c>dY</c> の行ごとの和を加算します。いずれも勾配が必要な入力についてだけ計算します。</para>
        /// </remarks>
        /// <exception cref="ArgumentException"><c>x.Rows</c> does not match <c>weight.Columns</c>, or the length of <paramref name="bias"/> is not <c>weight.Rows</c>. / 入力の行数が重みの列数と一致しない場合、または <paramref name="bias"/> の長さが重みの行数と一致しない場合。</exception>
        /// <exception cref="InvalidOperationException">A node is uninitialized or stale, the nodes belong to different tapes, or the tape cannot record the operation. / ノードが未初期化または古い世代の場合、ノードが別のテープに属する場合、またはテープが演算を記録できない場合。</exception>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> is neither <see cref="float"/> nor <see cref="double"/>. / <typeparamref name="T"/> が <see cref="float"/> でも <see cref="double"/> でもない場合。</exception>
        public static MatrixNode<T> Linear<T>(MatrixNode<T> weight, MatrixNode<T> x, VectorNode<T> bias)
            where T : unmanaged, IFloatingPointIeee754<T>
        {
            Guard.RequireElementType<T>();
            _ = weight.RequiresGradient;
            _ = x.RequiresGradient;
            var biasColumn = bias.AsColumnMatrix();
            RequireLinearShape(weight, x);
            Guard.RequireLength(weight.Rows, bias.Length, nameof(bias));
            return MatrixNode<T>.Record(new LinearOperation<T>(hasBias: true), [weight, x, biasColumn], weight.Rows, x.Columns);
        }

        /// <summary>
        /// Records the linear transformation <c>y = W * x</c> without bias.
        /// <para>バイアスなしの全結合 <c>y = W * x</c> を記録します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see cref="float"/> or <see cref="double"/>). / 浮動小数点の要素型（<see cref="float"/> または <see cref="double"/>）。</typeparam>
        /// <param name="weight">The weight node of shape <c>[outChannels × inChannels]</c>. / 重みのノード <c>[出力チャネル × 入力チャネル]</c>。</param>
        /// <param name="x">The input node of shape <c>[inChannels × positions]</c>. / 入力のノード <c>[入力チャネル × 位置]</c>。</param>
        /// <returns>The output node of shape <c>[outChannels × positions]</c>. / <c>[出力チャネル × 位置]</c> の出力ノード。</returns>
        /// <exception cref="ArgumentException"><c>x.Rows</c> does not match <c>weight.Columns</c>. / 入力の行数が重みの列数と一致しない場合。</exception>
        /// <exception cref="InvalidOperationException">A node is uninitialized or stale, the nodes belong to different tapes, or the tape cannot record the operation. / ノードが未初期化または古い世代の場合、ノードが別のテープに属する場合、またはテープが演算を記録できない場合。</exception>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> is neither <see cref="float"/> nor <see cref="double"/>. / <typeparamref name="T"/> が <see cref="float"/> でも <see cref="double"/> でもない場合。</exception>
        public static MatrixNode<T> Linear<T>(MatrixNode<T> weight, MatrixNode<T> x)
            where T : unmanaged, IFloatingPointIeee754<T>
        {
            Guard.RequireElementType<T>();
            _ = weight.RequiresGradient;
            _ = x.RequiresGradient;
            RequireLinearShape(weight, x);
            return MatrixNode<T>.Record(new LinearOperation<T>(hasBias: false), [weight, x], weight.Rows, x.Columns);
        }

        /// <summary>
        /// Records column-wise LayerNorm, <c>y = (x - μ) / √(σ² + ε) ⊙ scale + shift</c>.
        /// <para>列ごとの LayerNorm <c>y = (x - μ) / √(σ² + ε) ⊙ scale + shift</c> を記録します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see cref="float"/> or <see cref="double"/>). / 浮動小数点の要素型（<see cref="float"/> または <see cref="double"/>）。</typeparam>
        /// <param name="x">The input node of shape <c>[channels × positions]</c>. / 入力のノード <c>[チャネル × 位置]</c>。</param>
        /// <param name="scale">The per-channel scale node of length <c>x.Rows</c>. / チャネルごとのスケールのノード（長さは入力の行数）。</param>
        /// <param name="shift">The per-channel shift node of length <c>x.Rows</c>. / チャネルごとのシフトのノード（長さは入力の行数）。</param>
        /// <param name="epsilon">The non-negative constant added to the variance. / 分散に加える 0 以上の定数。</param>
        /// <returns>The output node with the same shape as <paramref name="x"/>. / <paramref name="x"/> と同じ形状の出力ノード。</returns>
        /// <remarks>
        /// <para>Saves three values per column (the shifted mean and <c>1/√(σ² + ε)</c>) and recomputes the normalized values in the pullback with the same expression as the forward pass.</para>
        /// <para>列ごとに 3 個の値（平均を 2 つに分けた値と <c>1/√(σ² + ε)</c>）を保存し、逆伝播では正規化した値を順伝播と同じ式で再計算します。</para>
        /// </remarks>
        /// <exception cref="ArgumentException">The length of <paramref name="scale"/> or <paramref name="shift"/> is not <c>x.Rows</c>. / <paramref name="scale"/> または <paramref name="shift"/> の長さが入力の行数と一致しない場合。</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="epsilon"/> is negative or <c>NaN</c>. / <paramref name="epsilon"/> が負または <c>NaN</c> の場合。</exception>
        /// <exception cref="InvalidOperationException">A node is uninitialized or stale, the nodes belong to different tapes, or the tape cannot record the operation. / ノードが未初期化または古い世代の場合、ノードが別のテープに属する場合、またはテープが演算を記録できない場合。</exception>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> is neither <see cref="float"/> nor <see cref="double"/>. / <typeparamref name="T"/> が <see cref="float"/> でも <see cref="double"/> でもない場合。</exception>
        /// <seealso href="https://arxiv.org/abs/1607.06450">Layer Normalization (2016)</seealso>
        public static MatrixNode<T> LayerNorm<T>(MatrixNode<T> x, VectorNode<T> scale, VectorNode<T> shift, T epsilon)
            where T : unmanaged, IFloatingPointIeee754<T>
        {
            Guard.RequireElementType<T>();
            _ = x.RequiresGradient;
            var scaleColumn = scale.AsColumnMatrix();
            var shiftColumn = shift.AsColumnMatrix();
            RequireNormalization(epsilon);
            Guard.RequireLength(x.Rows, scale.Length, nameof(scale));
            Guard.RequireLength(x.Rows, shift.Length, nameof(shift));
            return MatrixNode<T>.Record(new NormalizationOperation<T>(center: true, parameters: 2, epsilon),
                [x, scaleColumn, shiftColumn], x.Rows, x.Columns);
        }

        /// <summary>
        /// Records unscaled column-wise LayerNorm, <c>y = (x - μ) / √(σ² + ε)</c>.
        /// <para>スケール・シフトなしの列ごとの LayerNorm <c>y = (x - μ) / √(σ² + ε)</c> を記録します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see cref="float"/> or <see cref="double"/>). / 浮動小数点の要素型（<see cref="float"/> または <see cref="double"/>）。</typeparam>
        /// <param name="x">The input node of shape <c>[channels × positions]</c>. / 入力のノード <c>[チャネル × 位置]</c>。</param>
        /// <param name="epsilon">The non-negative constant added to the variance. / 分散に加える 0 以上の定数。</param>
        /// <returns>The output node with the same shape as <paramref name="x"/>. / <paramref name="x"/> と同じ形状の出力ノード。</returns>
        /// <remarks>
        /// <para>Saves three values per column (the shifted mean and <c>1/√(σ² + ε)</c>) and recomputes the normalized values in the pullback with the same expression as the forward pass.</para>
        /// <para>列ごとに 3 個の値（平均を 2 つに分けた値と <c>1/√(σ² + ε)</c>）を保存し、逆伝播では正規化した値を順伝播と同じ式で再計算します。</para>
        /// </remarks>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="epsilon"/> is negative or <c>NaN</c>. / <paramref name="epsilon"/> が負または <c>NaN</c> の場合。</exception>
        /// <exception cref="InvalidOperationException">The node is uninitialized or stale, or the tape cannot record the operation. / ノードが未初期化または古い世代の場合、またはテープが演算を記録できない場合。</exception>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> is neither <see cref="float"/> nor <see cref="double"/>. / <typeparamref name="T"/> が <see cref="float"/> でも <see cref="double"/> でもない場合。</exception>
        /// <seealso href="https://arxiv.org/abs/1607.06450">Layer Normalization (2016)</seealso>
        public static MatrixNode<T> LayerNorm<T>(MatrixNode<T> x, T epsilon)
            where T : unmanaged, IFloatingPointIeee754<T>
        {
            Guard.RequireElementType<T>();
            _ = x.RequiresGradient;
            RequireNormalization(epsilon);
            return MatrixNode<T>.Record(new NormalizationOperation<T>(center: true, parameters: 0, epsilon), [x], x.Rows, x.Columns);
        }

        /// <summary>
        /// Records column-wise RMSNorm, <c>y = x / √(mean(x²) + ε) ⊙ scale</c>.
        /// <para>列ごとの RMSNorm <c>y = x / √(mean(x²) + ε) ⊙ scale</c> を記録します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see cref="float"/> or <see cref="double"/>). / 浮動小数点の要素型（<see cref="float"/> または <see cref="double"/>）。</typeparam>
        /// <param name="x">The input node of shape <c>[channels × positions]</c>. / 入力のノード <c>[チャネル × 位置]</c>。</param>
        /// <param name="scale">The per-channel scale node of length <c>x.Rows</c>. / チャネルごとのスケールのノード（長さは入力の行数）。</param>
        /// <param name="epsilon">The non-negative constant added to the mean square. / 二乗平均に加える 0 以上の定数。</param>
        /// <returns>The output node with the same shape as <paramref name="x"/>. / <paramref name="x"/> と同じ形状の出力ノード。</returns>
        /// <remarks>
        /// <para>Saves per-column normalization statistics and recomputes the normalized values in the pullback with the same expression as the forward pass.</para>
        /// <para>列ごとの正規化統計量を保存し、逆伝播では正規化した値を順伝播と同じ式で再計算します。</para>
        /// </remarks>
        /// <exception cref="ArgumentException">The length of <paramref name="scale"/> is not <c>x.Rows</c>. / <paramref name="scale"/> の長さが入力の行数と一致しない場合。</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="epsilon"/> is negative or <c>NaN</c>. / <paramref name="epsilon"/> が負または <c>NaN</c> の場合。</exception>
        /// <exception cref="InvalidOperationException">A node is uninitialized or stale, the nodes belong to different tapes, or the tape cannot record the operation. / ノードが未初期化または古い世代の場合、ノードが別のテープに属する場合、またはテープが演算を記録できない場合。</exception>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> is neither <see cref="float"/> nor <see cref="double"/>. / <typeparamref name="T"/> が <see cref="float"/> でも <see cref="double"/> でもない場合。</exception>
        /// <seealso href="https://arxiv.org/abs/1910.07467">Root Mean Square Layer Normalization (2019)</seealso>
        public static MatrixNode<T> RmsNorm<T>(MatrixNode<T> x, VectorNode<T> scale, T epsilon)
            where T : unmanaged, IFloatingPointIeee754<T>
        {
            Guard.RequireElementType<T>();
            _ = x.RequiresGradient;
            var scaleColumn = scale.AsColumnMatrix();
            RequireNormalization(epsilon);
            Guard.RequireLength(x.Rows, scale.Length, nameof(scale));
            return MatrixNode<T>.Record(new NormalizationOperation<T>(center: false, parameters: 1, epsilon),
                [x, scaleColumn], x.Rows, x.Columns);
        }

        /// <summary>
        /// Records unscaled column-wise RMSNorm, <c>y = x / √(mean(x²) + ε)</c>.
        /// <para>スケールなしの列ごとの RMSNorm <c>y = x / √(mean(x²) + ε)</c> を記録します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see cref="float"/> or <see cref="double"/>). / 浮動小数点の要素型（<see cref="float"/> または <see cref="double"/>）。</typeparam>
        /// <param name="x">The input node of shape <c>[channels × positions]</c>. / 入力のノード <c>[チャネル × 位置]</c>。</param>
        /// <param name="epsilon">The non-negative constant added to the mean square. / 二乗平均に加える 0 以上の定数。</param>
        /// <returns>The output node with the same shape as <paramref name="x"/>. / <paramref name="x"/> と同じ形状の出力ノード。</returns>
        /// <remarks>
        /// <para>Saves per-column normalization statistics and recomputes the normalized values in the pullback with the same expression as the forward pass.</para>
        /// <para>列ごとの正規化統計量を保存し、逆伝播では正規化した値を順伝播と同じ式で再計算します。</para>
        /// </remarks>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="epsilon"/> is negative or <c>NaN</c>. / <paramref name="epsilon"/> が負または <c>NaN</c> の場合。</exception>
        /// <exception cref="InvalidOperationException">The node is uninitialized or stale, or the tape cannot record the operation. / ノードが未初期化または古い世代の場合、またはテープが演算を記録できない場合。</exception>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> is neither <see cref="float"/> nor <see cref="double"/>. / <typeparamref name="T"/> が <see cref="float"/> でも <see cref="double"/> でもない場合。</exception>
        /// <seealso href="https://arxiv.org/abs/1910.07467">Root Mean Square Layer Normalization (2019)</seealso>
        public static MatrixNode<T> RmsNorm<T>(MatrixNode<T> x, T epsilon)
            where T : unmanaged, IFloatingPointIeee754<T>
        {
            Guard.RequireElementType<T>();
            _ = x.RequiresGradient;
            RequireNormalization(epsilon);
            return MatrixNode<T>.Record(new NormalizationOperation<T>(center: false, parameters: 0, epsilon), [x], x.Rows, x.Columns);
        }

        /// <summary>
        /// Records parametric ReLU (PReLU), <c>y = x</c> for <c>x ≥ 0</c> and <c>aᵢ * x</c> for <c>x &lt; 0</c>.
        /// <para>PReLU（<c>x ≥ 0</c> で <c>x</c>、<c>x &lt; 0</c> で <c>aᵢ * x</c>）を記録します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see cref="float"/> or <see cref="double"/>). / 浮動小数点の要素型（<see cref="float"/> または <see cref="double"/>）。</typeparam>
        /// <param name="x">The input node of shape <c>[channels × positions]</c>. / 入力のノード <c>[チャネル × 位置]</c>。</param>
        /// <param name="slopes">The negative-region slope node of length <c>x.Rows</c> (per channel) or 1 (shared across all channels). / 負の領域の傾きのノード（長さは入力の行数、または全チャネル共通の 1）。</param>
        /// <returns>The output node with the same shape as <paramref name="x"/>. / <paramref name="x"/> と同じ形状の出力ノード。</returns>
        /// <remarks>
        /// <para>The local derivative with respect to <c>x</c> is <c>aᵢ</c> for <c>x &lt; 0</c> and 1 otherwise, including <c>x = 0</c> and <c>NaN</c>. As with other elementwise operations, no contribution is added where the local derivative is exactly zero. The slope gradient is the sum of <c>x·dy</c> over the negative elements.</para>
        /// <para><c>x</c> についての局所微分は、<c>x &lt; 0</c> で <c>aᵢ</c>、それ以外（<c>x = 0</c> と <c>NaN</c> を含む）で 1 です。ほかの要素ごとの演算と同じく、局所微分がちょうど 0 の要素には寄与を加えません。傾きの勾配は、負の要素についての <c>x·dy</c> の和です。</para>
        /// </remarks>
        /// <exception cref="ArgumentException">The length of <paramref name="slopes"/> is neither <c>x.Rows</c> nor 1. / <paramref name="slopes"/> の長さが入力の行数でも 1 でもない場合。</exception>
        /// <exception cref="InvalidOperationException">A node is uninitialized or stale, the nodes belong to different tapes, or the tape cannot record the operation. / ノードが未初期化または古い世代の場合、ノードが別のテープに属する場合、またはテープが演算を記録できない場合。</exception>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> is neither <see cref="float"/> nor <see cref="double"/>. / <typeparamref name="T"/> が <see cref="float"/> でも <see cref="double"/> でもない場合。</exception>
        /// <seealso href="https://arxiv.org/abs/1502.01852">Delving Deep into Rectifiers (2015)</seealso>
        public static MatrixNode<T> Prelu<T>(MatrixNode<T> x, VectorNode<T> slopes)
            where T : unmanaged, IFloatingPointIeee754<T>
        {
            Guard.RequireElementType<T>();
            _ = x.RequiresGradient;
            var slopesColumn = slopes.AsColumnMatrix();
            if (slopes.Length != x.Rows && slopes.Length != 1)
            {
                throw new ArgumentException($"The slopes must have length {x.Rows} (one per row) or 1 but got {slopes.Length}.", nameof(slopes));
            }

            return MatrixNode<T>.Record(new PreluOperation<T>(), [x, slopesColumn], x.Rows, x.Columns);
        }
    }

    private static void RequireLinearShape<T>(MatrixNode<T> weight, MatrixNode<T> x)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (weight.Columns != x.Rows)
        {
            throw new ArgumentException($"The weight has {weight.Columns} columns but the input has {x.Rows} rows.", nameof(x));
        }
    }

    private static void RequireNormalization<T>(T epsilon)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (!(epsilon >= T.Zero))
        {
            throw new ArgumentOutOfRangeException(nameof(epsilon), "The epsilon must be non-negative.");
        }
    }

    private readonly struct LinearOperation<T>(bool hasBias) : IDifferentiableOperation<T>
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        public void Forward(DifferentiableOperationContext<T> context, MatrixSpan<T> destination) =>
            XMathExtensions.LinearCore(context[0], context[1], hasBias ? context[2].Column(0) : default, destination);

        public void PullBack(DifferentiableOperationContext<T> context, ReadOnlyMatrixSpan<T> output, ReadOnlyMatrixSpan<T> outputGradient)
        {
            if (context.RequiresGradient(0))
            {
                GradientKernels.AddProduct(context.GetGradient(0), outputGradient, context[1].T);
            }

            if (context.RequiresGradient(1))
            {
                GradientKernels.AddProduct(context.GetGradient(1), context[0].T, outputGradient);
            }

            if (hasBias && context.RequiresGradient(2))
            {
                GradientKernels.AddRowSums(outputGradient, context.GetGradient(2).Column(0));
            }
        }
    }

    // parameters は 0（なし）、1（スケール）、2（スケールとシフト）。
    private readonly struct NormalizationOperation<T>(bool center, int parameters, T epsilon) : IDifferentiableOperation<T>
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        public void Forward(DifferentiableOperationContext<T> context, MatrixSpan<T> destination)
        {
            var x = context[0];
            bool needStats = x.Rows > 0 && x.Columns > 0 &&
                (context.RequiresGradient(0) || (parameters >= 1 && context.RequiresGradient(1)));
            Span<T> stats = needStats ? context.AllocateSaved(checked(3 * x.Columns)) : default;
            XMathExtensions.Normalize(x, parameters >= 1 ? context[1].Column(0) : default, parameters >= 2 ? context[2].Column(0) : default,
                epsilon, center, destination, stats);
        }

        public void PullBack(DifferentiableOperationContext<T> context, ReadOnlyMatrixSpan<T> output, ReadOnlyMatrixSpan<T> outputGradient)
        {
            bool hasDx = context.RequiresGradient(0);
            bool hasDscale = parameters >= 1 && context.RequiresGradient(1);
            if (hasDx || hasDscale)
            {
                var x = context[0];
                if (x.Rows > 0 && x.Columns > 0)
                {
                    var scale = parameters >= 1 ? context[1].Column(0) : default;
                    GradientKernels.NormalizeBackward(x, outputGradient, scale, context.GetSaved(0), center,
                        hasDx ? context.GetGradient(0) : default, hasDx, hasDscale ? context.GetGradient(1).Column(0) : default, hasDscale);
                }
            }

            if (parameters >= 2 && context.RequiresGradient(2))
            {
                GradientKernels.AddRowSums(outputGradient, context.GetGradient(2).Column(0));
            }
        }
    }

    private readonly struct PreluOperation<T> : IDifferentiableOperation<T>
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        public void Forward(DifferentiableOperationContext<T> context, MatrixSpan<T> destination) =>
            XMathExtensions.PreluCore(context[0], context[1].Column(0), destination);

        public void PullBack(DifferentiableOperationContext<T> context, ReadOnlyMatrixSpan<T> output, ReadOnlyMatrixSpan<T> outputGradient)
        {
            bool hasDx = context.RequiresGradient(0);
            bool hasDslopes = context.RequiresGradient(1);
            GradientKernels.PreluBackward(context[0], outputGradient, context[1].Column(0),
                hasDx ? context.GetGradient(0) : default, hasDx, hasDslopes ? context.GetGradient(1).Column(0) : default, hasDslopes);
        }
    }
}
