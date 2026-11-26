using System.Buffers;
using System.Numerics;

namespace Tuna.NeuralNetwork;

using Tuna.Differentiation;
using Tuna.LinearAlgebra;

public static partial class XMathDifferentiationExtensions
{
    private const int FullScanHistoryLimit = 64;
    private const int ScanCheckpointInterval = 32;

    extension(XMath)
    {
        /// <summary>
        /// Records causal depthwise convolution with optional bias.
        /// <para>省略可能なバイアスを持つ因果的な深さ方向の畳み込みを記録します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see cref="float"/> or <see cref="double"/>). / 浮動小数点の要素型（<see cref="float"/> または <see cref="double"/>）。</typeparam>
        /// <param name="x">The features with channels on rows and positions on columns. / 行がチャネル、列が位置の特徴量。</param>
        /// <param name="kernel">One row per channel and one column per tap, from oldest to latest. / チャネルごとに 1 行を持ち、各列のタップを過去から現在の順に並べたカーネル。</param>
        /// <param name="bias">Optional per-channel bias. / チャネルごとの省略可能なバイアス。</param>
        /// <returns>The convolution node with the same shape as <paramref name="x"/>. / <paramref name="x"/> と同じ形状の畳み込みノード。</returns>
        /// <exception cref="ArgumentException">The channel counts do not match. / チャネル数が一致しない場合。</exception>
        /// <exception cref="ArgumentOutOfRangeException">The kernel has no taps. / カーネルにタップがない場合。</exception>
        /// <exception cref="InvalidOperationException">A node is invalid for the tape or recording requires a reset. / ノードがテープで無効、または記録にリセットが必要な場合。</exception>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> is neither <see cref="float"/> nor <see cref="double"/>. / <typeparamref name="T"/> が <see cref="float"/> でも <see cref="double"/> でもない場合。</exception>
        public static MatrixNode<T> CausalDepthwiseConvolve<T>(MatrixNode<T> x, MatrixNode<T> kernel, VectorNode<T> bias = default)
            where T : unmanaged, IFloatingPointIeee754<T>
        {
            Guard.RequireElementType<T>();
            _ = x.RequiresGradient;
            _ = kernel.RequiresGradient;
            bool hasBias = bias.Length != 0;
            MatrixNode<T> biasColumn = hasBias ? bias.AsColumnMatrix() : default;

            Guard.RequireLength(x.Rows, kernel.Rows, nameof(kernel));
            Guard.RequirePositive(kernel.Columns, nameof(kernel));
            Guard.RequireOptionalLength(x.Rows, bias.Length, nameof(bias));

            return hasBias
                ? MatrixNode<T>.Record(new DepthwiseOperation<T>(), [x, kernel, biasColumn], x.Rows, x.Columns)
                : MatrixNode<T>.Record(new DepthwiseOperation<T>(), [x, kernel], x.Rows, x.Columns);
        }

        /// <summary>
        /// Records the selective state-space scan from a zero initial state.
        /// <para>初期状態がゼロの選択的状態空間モデルの走査を記録します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see cref="float"/> or <see cref="double"/>). / 浮動小数点の要素型（<see cref="float"/> または <see cref="double"/>）。</typeparam>
        /// <param name="x">The input of shape <c>[channels × length]</c>. / 形状 <c>[チャネル数 × 系列長]</c> の入力。</param>
        /// <param name="delta">The step sizes with the same shape as <paramref name="x"/>. / <paramref name="x"/> と同じ形状の刻み幅。</param>
        /// <param name="a">The diagonal state matrix of shape <c>[channels × states]</c>. / 形状 <c>[チャネル数 × 状態数]</c> の対角状態行列。</param>
        /// <param name="b">The input projection of shape <c>[states × length]</c>. / 形状 <c>[状態数 × 系列長]</c> の入力行列。</param>
        /// <param name="c">The output projection of shape <c>[states × length]</c>. / 形状 <c>[状態数 × 系列長]</c> の出力行列。</param>
        /// <param name="d">Optional per-channel skip connection. / チャネルごとの省略可能なスキップ接続。</param>
        /// <param name="approximate">Whether to use the approximate exponential in the recurrence. / 漸化式で指数関数の近似を使うかどうか。</param>
        /// <returns>The scan node with the same shape as <paramref name="x"/>. / <paramref name="x"/> と同じ形状の走査ノード。</returns>
        /// <remarks>
        /// <para>The forward pass uses the inference recurrence. The pullback uses the same exponential values for the derivative of the decay and recomputes intermediate states between checkpoints.</para>
        /// <para>順伝播には推論と同じ漸化式を使用します。逆伝播では減衰係数の導関数に同じ指数値を用い、チェックポイント間の状態を再計算します。</para>
        /// </remarks>
        /// <exception cref="ArgumentException">The input shapes do not agree. / 入力の形状が一致しない場合。</exception>
        /// <exception cref="InvalidOperationException">A node is invalid for the tape or recording requires a reset. / ノードがテープで無効、または記録にリセットが必要な場合。</exception>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> is neither <see cref="float"/> nor <see cref="double"/>. / <typeparamref name="T"/> が <see cref="float"/> でも <see cref="double"/> でもない場合。</exception>
        public static MatrixNode<T> SelectiveScan<T>(MatrixNode<T> x, MatrixNode<T> delta, MatrixNode<T> a,
            MatrixNode<T> b, MatrixNode<T> c, VectorNode<T> d = default, bool approximate = false)
            where T : unmanaged, IFloatingPointIeee754<T>
        {
            Guard.RequireElementType<T>();
            _ = x.RequiresGradient;
            _ = delta.RequiresGradient;
            _ = a.RequiresGradient;
            _ = b.RequiresGradient;
            _ = c.RequiresGradient;
            bool hasSkip = d.Length != 0;
            MatrixNode<T> skipColumn = hasSkip ? d.AsColumnMatrix() : default;

            int channels = x.Rows, length = x.Columns, n = a.Columns;
            Guard.RequireShape(channels, length, delta.Rows, delta.Columns, nameof(delta));
            if (a.Rows != channels)
            {
                throw new ArgumentException($"A must have {channels} rows (one per channel) but has {a.Rows}.", nameof(a));
            }

            Guard.RequireShape(n, length, b.Rows, b.Columns, nameof(b));
            Guard.RequireShape(n, length, c.Rows, c.Columns, nameof(c));
            Guard.RequireOptionalLength(channels, d.Length, nameof(d));

            var operation = new ScanOperation<T>(approximate);
            return hasSkip
                ? MatrixNode<T>.Record(operation, [x, delta, a, b, c, skipColumn], x.Rows, x.Columns)
                : MatrixNode<T>.Record(operation, [x, delta, a, b, c], x.Rows, x.Columns);
        }
    }

    private readonly struct DepthwiseOperation<T> : IDifferentiableOperation<T>
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        public void Forward(DifferentiableOperationContext<T> context, MatrixSpan<T> destination)
            => XMathExtensions.CausalDepthwiseCore(context[0], context[1], context.Count == 3 ? context[2].Column(0) : default, destination);

        public void PullBack(DifferentiableOperationContext<T> context, ReadOnlyMatrixSpan<T> output, ReadOnlyMatrixSpan<T> dy)
        {
            if (dy.IsEmpty)
            {
                return;
            }

            var x = context[0];
            var kernel = context[1];
            var dx = context.RequiresGradient(0) ? context.GetGradient(0) : default;
            var dk = context.RequiresGradient(1) ? context.GetGradient(1) : default;
            if (!dx.IsEmpty || !dk.IsEmpty)
            {
                for (int t = 0; t < x.Columns; t++)
                {
                    int first = Math.Max(0, kernel.Columns - 1 - t);
                    var upstream = dy.Column(t);
                    for (int k = first; k < kernel.Columns; k++)
                    {
                        int source = t - kernel.Columns + 1 + k;
                        for (int i = 0; i < x.Rows; i++)
                        {
                            if (!dx.IsEmpty)
                            {
                                dx[i, source] += upstream[i] * kernel[i, k];
                            }

                            if (!dk.IsEmpty)
                            {
                                dk[i, k] += upstream[i] * x[i, source];
                            }
                        }
                    }
                }
            }

            if (context.Count == 3 && context.RequiresGradient(2))
            {
                GradientKernels.AddRowSums(dy, context.GetGradient(2).Column(0));
            }
        }
    }

    private readonly struct ScanOperation<T>(bool approximate) : IDifferentiableOperation<T>
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        public void Forward(DifferentiableOperationContext<T> context, MatrixSpan<T> destination)
        {
            var x = context[0];
            var a = context[2];
            int length = x.Columns;
            int stateSize = checked(a.Rows * a.Columns);
            int interval = length <= FullScanHistoryLimit ? 1 : ScanCheckpointInterval;
            bool save = length > 0 && x.Rows > 0 && stateSize > 0 && NeedsStateGradient(context);
            var history = save ? context.AllocateSaved(checked(((length - 1) / interval + 2) * stateSize)) : default;
            var buffer = ArrayPool<T>.Shared.Rent(stateSize);
            try
            {
                var state = new MatrixSpan<T>(buffer.AsSpan(0, stateSize), a.Rows, a.Columns);
                state.Fill(T.Zero);
                XMathExtensions.ScanCore(x, context[1], a, context[3], context[4], context.Count == 6 ? context[5].Column(0) : default,
                    state, approximate, destination, history, interval);
            }
            finally
            {
                ArrayPool<T>.Shared.Return(buffer);
            }
        }

        public void PullBack(DifferentiableOperationContext<T> context, ReadOnlyMatrixSpan<T> output, ReadOnlyMatrixSpan<T> dy)
        {
            if (dy.IsEmpty)
            {
                return;
            }

            var a = context[2];
            var gradients = new ScanGradients<T>
            {
                X = context[0],
                Delta = context[1],
                A = a,
                B = context[3],
                C = context[4],
                Skip = context.Count == 6 ? context[5].Column(0) : default,
                Dy = dy,
                Dx = context.RequiresGradient(0) ? context.GetGradient(0) : default,
                DSkip = context.Count == 6 && context.RequiresGradient(5) ? context.GetGradient(5).Column(0) : default,
            };

            if (a.Columns == 0 || !NeedsStateGradient(context))
            {
                for (int t = dy.Columns - 1; t >= 0; t--)
                {
                    gradients.AddSkip(t);
                }

                return;
            }

            gradients.DDelta = context.RequiresGradient(1) ? context.GetGradient(1) : default;
            gradients.Da = context.RequiresGradient(2) ? context.GetGradient(2) : default;
            gradients.Db = context.RequiresGradient(3) ? context.GetGradient(3) : default;
            gradients.Dc = context.RequiresGradient(4) ? context.GetGradient(4) : default;

            int interval = dy.Columns <= FullScanHistoryLimit ? 1 : ScanCheckpointInterval;
            gradients.Run(context.GetSaved(0), interval, approximate);
        }

        private static bool NeedsStateGradient(DifferentiableOperationContext<T> context)
        {
            for (int i = 0; i < 5; i++)
            {
                if (context.RequiresGradient(i))
                {
                    return true;
                }
            }

            return false;
        }
    }

    private ref struct ScanGradients<T> where T : unmanaged, IFloatingPointIeee754<T>
    {
        internal ReadOnlyMatrixSpan<T> X, Delta, A, B, C, Dy;
        internal ReadOnlyVectorSpan<T> Skip;
        internal MatrixSpan<T> Dx, DDelta, Da, Db, Dc;
        internal VectorSpan<T> DSkip;

        internal void Run(ReadOnlySpan<T> history, int interval, bool approximate)
        {
            int channels = X.Rows, length = X.Columns, n = A.Columns;
            int stateSize = checked(channels * n);
            int blockCount = (length - 1) / interval + 1;
            int capacity = interval == 1 ? 0 : Math.Min(interval, length);
            int replayLength = capacity == 0 ? 0 : checked((capacity + 1) * stateSize);
            int size = checked(2 * stateSize + replayLength + (capacity == 0 ? 0 : stateSize + capacity * channels));
            var buffer = ArrayPool<T>.Shared.Rent(size);
            try
            {
                var adjoint = buffer.AsSpan(0, stateSize);
                adjoint.Clear();
                var decay = buffer.AsSpan(stateSize, stateSize);
                for (int block = blockCount - 1; block >= 0; block--)
                {
                    int first = block * interval, count = Math.Min(interval, length - first);
                    ReadOnlySpan<T> states = history.Slice(block * stateSize, stateSize);
                    if (interval == 1)
                    {
                        states = history.Slice(block * stateSize, 2 * stateSize);
                    }
                    else
                    {
                        var replay = buffer.AsSpan(2 * stateSize, (count + 1) * stateSize);
                        var initial = buffer.AsSpan(2 * stateSize + replayLength, stateSize);
                        states.CopyTo(initial);
                        var state = new MatrixSpan<T>(initial, channels, n);
                        var output = new MatrixSpan<T>(buffer.AsSpan(3 * stateSize + replayLength, channels * count), channels, count);
                        var range = first..(first + count);
                        XMathExtensions.ScanCore(X[.., range], Delta[.., range], A, B[.., range], C[.., range], default, state,
                            approximate, output, replay);
                        states = replay;
                    }

                    for (int offset = count - 1; offset >= 0; offset--)
                    {
                        AddStep(first + offset, states.Slice(offset * stateSize, stateSize),
                            states.Slice((offset + 1) * stateSize, stateSize), adjoint, decay, approximate);
                    }
                }
            }
            finally
            {
                ArrayPool<T>.Shared.Return(buffer);
            }
        }

        internal void AddSkip(int t)
        {
            if (Skip.Length == 0)
            {
                return;
            }

            for (int d = 0; d < X.Rows; d++)
            {
                T dy = Dy[d, t];
                if (!Dx.IsEmpty)
                {
                    Dx[d, t] += dy * Skip[d];
                }

                if (DSkip.Length != 0)
                {
                    DSkip[d] += dy * X[d, t];
                }
            }
        }

        private void AddStep(int t, ReadOnlySpan<T> previous, ReadOnlySpan<T> current, Span<T> adjoint,
            Span<T> decay, bool approximate)
        {
            AddSkip(t);
            int channels = X.Rows;
            for (int k = 0; k < A.Columns; k++)
            {
                for (int d = 0; d < channels; d++)
                {
                    decay[k * channels + d] = Delta[d, t] * A[d, k];
                }
            }

            XMath.InPlace.Exp(new VectorSpan<T>(decay), approximate);
            for (int k = 0; k < A.Columns; k++)
            {
                T b = B[k, t], c = C[k, t];
                for (int d = 0; d < channels; d++)
                {
                    int index = k * channels + d;
                    T dy = Dy[d, t], delta = Delta[d, t], x = X[d, t];
                    T g = adjoint[index] + dy * c;
                    T propagated = g * decay[index];
                    adjoint[index] = propagated;
                    if (!Dx.IsEmpty)
                    {
                        Dx[d, t] += (g * b) * delta;
                    }

                    if (!DDelta.IsEmpty)
                    {
                        DDelta[d, t] += (g * b) * x + (propagated * previous[index]) * A[d, k];
                    }

                    if (!Da.IsEmpty)
                    {
                        Da[d, k] += (propagated * previous[index]) * delta;
                    }

                    if (!Db.IsEmpty)
                    {
                        Db[k, t] += g * (delta * x);
                    }

                    if (!Dc.IsEmpty)
                    {
                        Dc[k, t] += dy * current[index];
                    }
                }
            }
        }
    }
}
