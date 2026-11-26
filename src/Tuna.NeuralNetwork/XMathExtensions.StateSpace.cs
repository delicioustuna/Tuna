using System.Buffers;
using System.Numerics;

namespace Tuna.NeuralNetwork;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む。
using Tuna.LinearAlgebra;

public static partial class XMathExtensions
{
    // 選択的走査のチャネル分割単位（Vector512<float> のレーン数の倍数）
    private const int ChannelUnit = 16;

    extension(XMath)
    {
        /// <summary>
        /// Computes channel-wise causal depthwise 1D convolution, <c>y(d, t) = bias_d + Σ_{k=0}^{K-1} w(d, k) * x(d, t - K + 1 + k)</c>, and returns a new matrix.
        /// <para>チャネルごとの因果的な depthwise 1次元畳み込み <c>y(d, t) = bias_d + Σ_{k=0}^{K-1} w(d, k) * x(d, t - K + 1 + k)</c> を計算し、新しい行列として返します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see cref="float"/> or <see cref="double"/>). / 浮動小数点の要素型（<see cref="float"/> または <see cref="double"/>）。</typeparam>
        /// <param name="x">The input sequence matrix of shape <c>[channels D × length L]</c>. / 入力系列行列 <c>[チャネル数 D × 系列長 L]</c>。</param>
        /// <param name="kernel">The kernel weight matrix of shape <c>[D × K]</c>, where the last column multiplies the current position. / カーネル重み行列 <c>[D × K]</c>（最終列が現在位置に対応）。</param>
        /// <param name="bias">Per-channel bias of length <c>D</c>, or empty for zero bias. / チャネルごとのバイアス（長さ <c>D</c>。空の場合は 0）。</param>
        /// <returns>A new matrix of shape <c>[D × L]</c> containing the convolution result. / 畳み込み結果を格納した新しい行列 <c>[D × L]</c>。</returns>
        /// <remarks>
        /// <para>Positions before the start of the sequence are zero-padded. Row-major PyTorch <c>Conv1d</c> weights of shape <c>[D, 1, K]</c> can be wrapped via <c>new ReadOnlyMatrixSpan&lt;float&gt;(w, K, D).T</c>.</para>
        /// <para>系列先頭より前の入力は 0 として扱います。行優先の PyTorch <c>Conv1d</c> 重み <c>[D, 1, K]</c> は <c>new ReadOnlyMatrixSpan&lt;float&gt;(w, K, D).T</c> でコピーせずに参照できます。</para>
        /// </remarks>
        /// <exception cref="ArgumentException">The input, kernel, or bias dimensions do not match. / 入力、カーネル、またはバイアスの形状が一致しない場合。</exception>
        /// <exception cref="ArgumentOutOfRangeException"><c>kernel.Columns</c> is less than 1. / <c>kernel.Columns</c> が 1 未満の場合。</exception>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> is neither <see cref="float"/> nor <see cref="double"/>. / <typeparamref name="T"/> が <see cref="float"/> でも <see cref="double"/> でもない場合。</exception>
        [InferenceOverloads]
        public static Matrix<T> CausalDepthwiseConvolve<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> kernel, ReadOnlyVectorSpan<T> bias)
            where T : unmanaged, IFloatingPointIeee754<T>
        {
            var result = Matrix<T>.Uninitialized(x.Rows, x.Columns);
            CausalDepthwiseCore(x, kernel, bias, result);
            return result;
        }

        /// <summary>
        /// Computes channel-wise causal depthwise 1D convolution without bias and returns a new matrix.
        /// <para>バイアスなしのチャネルごとの因果的な depthwise 1次元畳み込みを計算し、新しい行列として返します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see cref="float"/> or <see cref="double"/>). / 浮動小数点の要素型（<see cref="float"/> または <see cref="double"/>）。</typeparam>
        /// <param name="x">The input sequence matrix of shape <c>[channels D × length L]</c>. / 入力系列行列 <c>[チャネル数 D × 系列長 L]</c>。</param>
        /// <param name="kernel">The kernel weight matrix of shape <c>[D × K]</c>, where the last column multiplies the current position. / カーネル重み行列 <c>[D × K]</c>（最終列が現在位置に対応）。</param>
        /// <returns>A new matrix of shape <c>[D × L]</c> containing the convolution result. / 畳み込み結果を格納した新しい行列 <c>[D × L]</c>。</returns>
        /// <remarks>
        /// <para>Positions before the start of the sequence are zero-padded.</para>
        /// <para>系列先頭より前の入力は 0 として扱います。</para>
        /// </remarks>
        /// <exception cref="ArgumentException"><c>kernel.Rows</c> does not match <c>x.Rows</c>. / <c>kernel.Rows</c> が <c>x.Rows</c> と一致しない場合。</exception>
        /// <exception cref="ArgumentOutOfRangeException"><c>kernel.Columns</c> is less than 1. / <c>kernel.Columns</c> が 1 未満の場合。</exception>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> is neither <see cref="float"/> nor <see cref="double"/>. / <typeparamref name="T"/> が <see cref="float"/> でも <see cref="double"/> でもない場合。</exception>
        [InferenceOverloads]
        public static Matrix<T> CausalDepthwiseConvolve<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> kernel)
            where T : unmanaged, IFloatingPointIeee754<T> =>
            XMath.CausalDepthwiseConvolve(x, kernel, default(ReadOnlyVectorSpan<T>));

        /// <summary>
        /// Computes channel-wise causal depthwise 1D convolution, <c>y(d, t) = bias_d + Σ_{k=0}^{K-1} w(d, k) * x(d, t - K + 1 + k)</c>, into <paramref name="destination"/>.
        /// <para>チャネルごとの因果的な depthwise 1次元畳み込み <c>y(d, t) = bias_d + Σ_{k=0}^{K-1} w(d, k) * x(d, t - K + 1 + k)</c> を計算し、<paramref name="destination"/> に書き込みます。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see cref="float"/> or <see cref="double"/>). / 浮動小数点の要素型（<see cref="float"/> または <see cref="double"/>）。</typeparam>
        /// <param name="x">The input sequence matrix of shape <c>[channels D × length L]</c>. / 入力系列行列 <c>[チャネル数 D × 系列長 L]</c>。</param>
        /// <param name="kernel">The kernel weight matrix of shape <c>[D × K]</c>, where the last column multiplies the current position. / カーネル重み行列 <c>[D × K]</c>（最終列が現在位置に対応）。</param>
        /// <param name="bias">Per-channel bias of length <c>D</c>, or empty for zero bias. / チャネルごとのバイアス（長さ <c>D</c>。空の場合は 0）。</param>
        /// <param name="destination">The output matrix span of shape <c>[D × L]</c>. Must not overlap <paramref name="x"/>, <paramref name="kernel"/>, or <paramref name="bias"/>. / 出力行列 <c>[D × L]</c>。<paramref name="x"/>、<paramref name="kernel"/>、<paramref name="bias"/> と重複してはなりません。</param>
        /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
        /// <remarks>
        /// <para>Positions before the start of the sequence are zero-padded.</para>
        /// <para>系列先頭より前の入力は 0 として扱います。</para>
        /// </remarks>
        /// <exception cref="ArgumentException">The shapes do not match, or <paramref name="destination"/> overlaps an input. / 形状が一致しない場合、または <paramref name="destination"/> が入力と重複する場合。</exception>
        /// <exception cref="ArgumentOutOfRangeException"><c>kernel.Columns</c> is less than 1. / <c>kernel.Columns</c> が 1 未満の場合。</exception>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> is neither <see cref="float"/> nor <see cref="double"/>. / <typeparamref name="T"/> が <see cref="float"/> でも <see cref="double"/> でもない場合。</exception>
        [InferenceOverloads]
        public static MatrixSpan<T> CausalDepthwiseConvolve<T>(
            ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> kernel, ReadOnlyVectorSpan<T> bias, MatrixSpan<T> destination)
            where T : unmanaged, IFloatingPointIeee754<T> =>
            CausalDepthwiseCore(x, kernel, bias, destination);

        /// <summary>
        /// Runs the Mamba S6 selective state-space scan from a zero initial state and returns a new matrix.
        /// <para>初期状態 0 から Mamba の選択的状態空間モデル（S6）を走査し、新しい行列として返します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see cref="float"/> or <see cref="double"/>). / 浮動小数点の要素型（<see cref="float"/> または <see cref="double"/>）。</typeparam>
        /// <param name="x">The input matrix of shape <c>[channels D × length L]</c>. / 入力行列 <c>[チャネル数 D × 系列長 L]</c>。</param>
        /// <param name="delta">The step sizes <c>Δ</c> of shape <c>[D × L]</c> (already transformed to positive values, e.g. via softplus). / 刻み幅 <c>Δ</c> <c>[D × L]</c>（softplus 等で正にした後の値）。</param>
        /// <param name="a">The continuous-time diagonal state matrix <c>A</c> of shape <c>[D × N]</c> (typically <c>-exp(A_log)</c> in Mamba). / 連続時間の対角状態行列 <c>A</c> <c>[D × N]</c>（Mamba では通常 <c>-exp(A_log)</c>）。</param>
        /// <param name="b">The per-position input projection <c>B</c> of shape <c>[N × L]</c>. / 位置ごとの入力行列 <c>B</c> <c>[N × L]</c>。</param>
        /// <param name="c">The per-position output projection <c>C</c> of shape <c>[N × L]</c>. / 位置ごとの出力行列 <c>C</c> <c>[N × L]</c>。</param>
        /// <param name="d">The optional skip connection vector <c>D</c> of length <c>D</c>, or empty to omit. / スキップ接続ベクトル <c>D</c>（長さ <c>D</c>。空の場合は省略）。</param>
        /// <param name="approximate"><see langword="true"/> to use approximate exponential evaluation (as in <see cref="XMath.Exp{T}(ReadOnlyVectorSpan{T}, bool)"/>); <see langword="false"/> for exact evaluation. / 指数関数に近似式（<see cref="XMath.Exp{T}(ReadOnlyVectorSpan{T}, bool)"/> 相当）を使う場合は <see langword="true"/>、高精度計算を使う場合は <see langword="false"/>。</param>
        /// <returns>A new matrix of shape <c>[D × L]</c> with each column holding the output at that position. / 各位置の出力を列に格納した新しい行列 <c>[D × L]</c>。</returns>
        /// <remarks>
        /// <para>Updates <c>h_t(d, k) = exp(Δ_t(d) * A(d, k)) * h_{t-1}(d, k) + Δ_t(d) * x_t(d) * B_t(k)</c> and computes <c>y_t(d) = Σ_k C_t(k) * h_t(d, k) + D(d) * x_t(d)</c>, using zero-order hold discretization <c>Ā = exp(ΔA)</c> and first-order Euler discretization <c>B̄ = ΔB</c>.</para>
        /// <para>各チャネル <c>d</c>・状態 <c>k</c> について <c>h_t(d, k) = exp(Δ_t(d) * A(d, k)) * h_{t-1}(d, k) + Δ_t(d) * x_t(d) * B_t(k)</c>、<c>y_t(d) = Σ_k C_t(k) * h_t(d, k) + D(d) * x_t(d)</c> を計算します。離散化にはゼロ次ホールド <c>Ā = exp(ΔA)</c> と 1 次近似 <c>B̄ = ΔB</c> を用います。</para>
        /// </remarks>
        /// <exception cref="ArgumentException">The input shapes do not match. / 入力の形状が一致しない場合。</exception>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> is neither <see cref="float"/> nor <see cref="double"/>. / <typeparamref name="T"/> が <see cref="float"/> でも <see cref="double"/> でもない場合。</exception>
        /// <seealso href="https://arxiv.org/abs/2312.00752">Mamba: Linear-Time Sequence Modeling with Selective State Spaces (2023)</seealso>
        [InferenceOverloads]
        public static Matrix<T> SelectiveScan<T>(
            ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> delta, ReadOnlyMatrixSpan<T> a, ReadOnlyMatrixSpan<T> b, ReadOnlyMatrixSpan<T> c,
            ReadOnlyVectorSpan<T> d, bool approximate)
            where T : unmanaged, IFloatingPointIeee754<T>
        {
            var state = new Matrix<T>(a.Rows, a.Columns);
            var result = Matrix<T>.Uninitialized(x.Rows, x.Columns);
            ScanCore(x, delta, a, b, c, d, state, approximate, result);
            return result;
        }

        /// <summary>
        /// Runs the selective state-space scan from <paramref name="state"/>, updating <paramref name="state"/> in place and writing outputs to <paramref name="destination"/>.
        /// <para>指定した <paramref name="state"/> から選択的状態空間モデルの走査を実行し、<paramref name="state"/> を更新するとともに結果を <paramref name="destination"/> に書き込みます。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see cref="float"/> or <see cref="double"/>). / 浮動小数点の要素型（<see cref="float"/> または <see cref="double"/>）。</typeparam>
        /// <param name="x">The input matrix of shape <c>[D × L]</c>. / 入力行列 <c>[D × L]</c>。</param>
        /// <param name="delta">The step sizes <c>Δ</c> of shape <c>[D × L]</c>. / 刻み幅 <c>Δ</c> <c>[D × L]</c>。</param>
        /// <param name="a">The state matrix <c>A</c> of shape <c>[D × N]</c>. / 状態行列 <c>A</c> <c>[D × N]</c>。</param>
        /// <param name="b">The input projection <c>B</c> of shape <c>[N × L]</c>. / 入力行列 <c>B</c> <c>[N × L]</c>。</param>
        /// <param name="c">The output projection <c>C</c> of shape <c>[N × L]</c>. / 出力行列 <c>C</c> <c>[N × L]</c>。</param>
        /// <param name="d">The optional skip connection vector of length <c>D</c>, or empty to omit. / スキップ接続ベクトル（長さ <c>D</c>。空の場合は省略）。</param>
        /// <param name="state">The state matrix <c>h</c> of shape <c>[D × N]</c>, updated in place to the final state. / 状態行列 <c>h</c> <c>[D × N]</c>（最終位置後の状態にインプレースで更新されます）。</param>
        /// <param name="approximate"><see langword="true"/> to use approximate exponential evaluation; <see langword="false"/> for exact evaluation. / 指数関数に近似式を使う場合は <see langword="true"/>、高精度計算を使う場合は <see langword="false"/>。</param>
        /// <param name="destination">The output matrix span <c>y</c> of shape <c>[D × L]</c>. / 出力行列 <c>y</c> <c>[D × L]</c>。</param>
        /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
        /// <remarks>
        /// <para><paramref name="destination"/> and <paramref name="state"/> must not overlap each other or any input span. For chunked sequence processing, pass the updated <paramref name="state"/> to the next call.</para>
        /// <para><paramref name="destination"/> と <paramref name="state"/> は互いに重複せず、いずれの入力とも重複してはなりません。系列を分割して処理する場合は、更新された <paramref name="state"/> を次の呼び出しに渡してください。</para>
        /// </remarks>
        /// <exception cref="ArgumentException">The shapes do not match, or <paramref name="destination"/> or <paramref name="state"/> overlaps an input or each other. / 形状が一致しない場合、あるいは <paramref name="destination"/> または <paramref name="state"/> が入力や互いに重複する場合。</exception>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> is neither <see cref="float"/> nor <see cref="double"/>. / <typeparamref name="T"/> が <see cref="float"/> でも <see cref="double"/> でもない場合。</exception>
        [InferenceOverloads]
        public static MatrixSpan<T> SelectiveScan<T>(
            ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> delta, ReadOnlyMatrixSpan<T> a, ReadOnlyMatrixSpan<T> b, ReadOnlyMatrixSpan<T> c,
            ReadOnlyVectorSpan<T> d, MatrixSpan<T> state, bool approximate, MatrixSpan<T> destination)
            where T : unmanaged, IFloatingPointIeee754<T> =>
            ScanCore(x, delta, a, b, c, d, state, approximate, destination);

        /// <summary>
        /// Advances multiple independent sequences by a single step in the selective state-space model.
        /// <para>選択的状態空間モデルで複数の独立した系列を 1 ステップ進めます。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see cref="float"/> or <see cref="double"/>). / 浮動小数点の要素型（<see cref="float"/> または <see cref="double"/>）。</typeparam>
        /// <param name="x">The current-step inputs of shape <c>[D × S]</c>, with one column per sequence. / 各系列の現在ステップ入力 <c>[D × S]</c>（列が系列）。</param>
        /// <param name="delta">The step sizes <c>Δ</c> of shape <c>[D × S]</c>. / 刻み幅 <c>Δ</c> <c>[D × S]</c>。</param>
        /// <param name="a">The shared state matrix <c>A</c> of shape <c>[D × N]</c>. / 全系列で共通の状態行列 <c>A</c> <c>[D × N]</c>。</param>
        /// <param name="b">The input projections <c>B</c> of shape <c>[N × S]</c>. / 入力行列 <c>B</c> <c>[N × S]</c>。</param>
        /// <param name="c">The output projections <c>C</c> of shape <c>[N × S]</c>. / 出力行列 <c>C</c> <c>[N × S]</c>。</param>
        /// <param name="d">The optional skip connection vector of length <c>D</c>, or empty to omit. / スキップ接続ベクトル（長さ <c>D</c>。空の場合は省略）。</param>
        /// <param name="state">The per-sequence states of shape <c>[D, N, S]</c> (slice <c>s</c> holds <c>h</c> of shape <c>[D × N]</c>), updated in place. / 系列ごとの状態テンソル <c>[D, N, S]</c>（第 <c>s</c> スライスが系列 <c>s</c> の <c>h [D × N]</c>。インプレースで更新されます）。</param>
        /// <param name="approximate"><see langword="true"/> to use approximate exponential evaluation; <see langword="false"/> for exact evaluation. / 指数関数に近似式を使う場合は <see langword="true"/>、高精度計算を使う場合は <see langword="false"/>。</param>
        /// <param name="destination">The output matrix span <c>y</c> of shape <c>[D × S]</c>. / 出力行列 <c>y</c> <c>[D × S]</c>。</param>
        /// <returns><paramref name="destination"/>, with each column holding the output of the corresponding sequence. / 各系列の出力を列に格納した <paramref name="destination"/>。</returns>
        /// <remarks>
        /// <para><paramref name="destination"/> and <paramref name="state"/> must not overlap each other or any input span.</para>
        /// <para><paramref name="destination"/> と <paramref name="state"/> は互いに重複せず、いずれの入力とも重複してはなりません。</para>
        /// </remarks>
        /// <exception cref="ArgumentException">The shapes do not match, or <paramref name="destination"/> or <paramref name="state"/> overlaps an input or each other. / 形状が一致しない場合、あるいは <paramref name="destination"/> または <paramref name="state"/> が入力や互いに重複する場合。</exception>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> is neither <see cref="float"/> nor <see cref="double"/>. / <typeparamref name="T"/> が <see cref="float"/> でも <see cref="double"/> でもない場合。</exception>
        [InferenceOverloads]
        public static MatrixSpan<T> SelectiveScanStep<T>(
            ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> delta, ReadOnlyMatrixSpan<T> a, ReadOnlyMatrixSpan<T> b, ReadOnlyMatrixSpan<T> c,
            ReadOnlyVectorSpan<T> d, TensorSpan<T> state, bool approximate, MatrixSpan<T> destination)
            where T : unmanaged, IFloatingPointIeee754<T> =>
            ScanStepCore(x, delta, a, b, c, d, state, approximate, destination);
    }

    internal static unsafe MatrixSpan<T> CausalDepthwiseCore<T>(
        ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> kernel, ReadOnlyVectorSpan<T> bias, MatrixSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        Guard.RequireElementType<T>();
        int channels = x.Rows, length = x.Columns, taps = kernel.Columns;
        if (kernel.Rows != channels)
        {
            throw new ArgumentException($"The kernel must have {channels} rows (one per channel) but has {kernel.Rows}.", nameof(kernel));
        }

        Guard.RequirePositive(taps, nameof(kernel));
        Guard.RequireOptionalLength(channels, bias.Length, nameof(bias));
        Guard.RequireShape(channels, length, destination.Rows, destination.Columns, nameof(destination));
        Guard.RequireNoOverlap(x, destination, allowExact: false, nameof(destination));
        Guard.RequireNoOverlap(kernel, destination, allowExact: false, nameof(destination));
        Guard.RequireNoOverlap(bias, destination, nameof(destination));
        if (x.IsEmpty)
        {
            return destination;
        }

        T[]? rx = null, rk = null, rb = null, ry = null;
        try
        {
            var xd = Dense.Columns(x, ref rx);
            var kd = Dense.Columns(kernel, ref rk);
            var bd = Dense.Contiguous(bias, ref rb);
            var target = destination;
            bool direct = channels <= 1 || destination.AsReadOnly().RowStride == 1;
            if (!direct)
            {
                ry = ArrayPool<T>.Shared.Rent(channels * length);
                target = new MatrixSpan<T>(ry.AsSpan(0, channels * length), channels, length);
            }

            int ldx = xd.ColumnStride, ldk = kd.ColumnStride, ldy = target.AsReadOnly().ColumnStride;
            int threads = Split.Threads((double)channels * length * taps);
            fixed (T* px = &Dense.Ref(xd), pk = &Dense.Ref(kd), pb = &Dense.Ref(bd), py = &Dense.Ref(target))
            {
                var work = new DepthwiseWork<T>
                {
                    X = px,
                    Kernel = pk,
                    Bias = bias.Length == 0 ? null : pb,
                    Output = py,
                    Channels = channels,
                    Taps = taps,
                    InputStride = ldx,
                    KernelStride = ldk,
                    OutputStride = ldy,
                };
                Split.For(length, Split.Chunks(length, threads, 16), threads, ref work);
            }

            if (!direct)
            {
                target.CopyTo(destination);
            }
        }
        finally
        {
            Dense.Return(rx);
            Dense.Return(rk);
            Dense.Return(rb);
            Dense.Return(ry);
        }

        return destination;
    }

    private static int CheckScanShapes<T>(
        ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> delta, ReadOnlyMatrixSpan<T> a, ReadOnlyMatrixSpan<T> b, ReadOnlyMatrixSpan<T> c,
        ReadOnlyVectorSpan<T> d, int destinationRows, int destinationColumns)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        Guard.RequireElementType<T>();
        int channels = x.Rows, length = x.Columns, n = a.Columns;
        Guard.RequireShape(channels, length, delta.Rows, delta.Columns, nameof(delta));
        if (a.Rows != channels)
        {
            throw new ArgumentException($"A must have {channels} rows (one per channel) but has {a.Rows}.", nameof(a));
        }

        Guard.RequireShape(n, length, b.Rows, b.Columns, nameof(b));
        Guard.RequireShape(n, length, c.Rows, c.Columns, nameof(c));
        Guard.RequireOptionalLength(channels, d.Length, nameof(d));
        Guard.RequireShape(channels, length, destinationRows, destinationColumns, "destination");
        return n;
    }

    internal static unsafe MatrixSpan<T> ScanCore<T>(
        ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> delta, ReadOnlyMatrixSpan<T> a, ReadOnlyMatrixSpan<T> b, ReadOnlyMatrixSpan<T> c,
        ReadOnlyVectorSpan<T> d, MatrixSpan<T> state, bool approximate, MatrixSpan<T> destination,
        scoped Span<T> history = default, int checkpointInterval = 1)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        int n = CheckScanShapes(x, delta, a, b, c, d, destination.Rows, destination.Columns);
        int channels = x.Rows, length = x.Columns;
        Guard.RequireShape(channels, n, state.Rows, state.Columns, nameof(state));
        RequireScanDisjoint(x, state, destination);
        RequireScanDisjoint(delta, state, destination);
        RequireScanDisjoint(a, state, destination);
        RequireScanDisjoint(b, state, destination);
        RequireScanDisjoint(c, state, destination);

        Guard.RequireNoOverlap(state.AsReadOnly(), destination, allowExact: false, nameof(destination));
        Guard.RequireNoOverlap(d, destination, nameof(destination));
        Guard.RequireNoOverlap(d, state, nameof(state));
        if (channels == 0 || length == 0 || n == 0)
        {
            if (n == 0 && !destination.IsEmpty)
            {
                // 状態次元 N == 0 の場合は y = D·x のみ計算する
                for (int j = 0; j < length; j++)
                {
                    var column = destination.Column(j);
                    if (d.Length == 0)
                    {
                        column.Fill(T.Zero);
                    }
                    else
                    {
                        XMath.Hadamard(d, x.Column(j), column);
                    }
                }
            }

            return destination;
        }

        T[]? rx = null, rdt = null, ra = null, rb = null, rc = null, rd = null, ry = null, rh = null;
        try
        {
            var xd = Dense.Columns(x, ref rx);
            var dtd = Dense.Columns(delta, ref rdt);
            var ad = Dense.Columns(a, ref ra);
            var bd = Dense.Columns(b, ref rb);
            var cd = Dense.Columns(c, ref rc);
            var dd = Dense.Contiguous(d, ref rd);
            var target = destination;
            bool direct = channels <= 1 || destination.AsReadOnly().RowStride == 1;
            if (!direct)
            {
                ry = ArrayPool<T>.Shared.Rent(channels * length);
                target = new MatrixSpan<T>(ry.AsSpan(0, channels * length), channels, length);
            }

            var h = state;
            bool directState = channels <= 1 || state.AsReadOnly().RowStride == 1;
            if (!directState)
            {
                rh = ArrayPool<T>.Shared.Rent(channels * n);
                h = new MatrixSpan<T>(rh.AsSpan(0, channels * n), channels, n);
                state.CopyTo(h);
            }

            int ldx = xd.ColumnStride, ldt = dtd.ColumnStride, lda = ad.ColumnStride, ldb = bd.ColumnStride, ldc = cd.ColumnStride;
            int ldy = target.AsReadOnly().ColumnStride, ldh = h.AsReadOnly().ColumnStride;
            int threads = Split.Threads(4.0 * channels * n * length);
            int units = (channels + ChannelUnit - 1) / ChannelUnit;
            int groups = threads <= 1 ? 1 : Math.Min(units, threads * 2);
            if (!history.IsEmpty)
            {
                h.CopyTo(new MatrixSpan<T>(history[..(channels * n)], channels, n));
            }

            fixed (T* px = &Dense.Ref(xd), pt = &Dense.Ref(dtd), pa = &Dense.Ref(ad), pb = &Dense.Ref(bd), pc = &Dense.Ref(cd),
                pd = &Dense.Ref(dd), py = &Dense.Ref(target), ph = &Dense.Ref(h), saved = history)
            {
                var work = new ScanForwardWork<T>
                {
                    X = px,
                    Delta = pt,
                    A = pa,
                    B = pb,
                    C = pc,
                    Skip = d.Length == 0 ? null : pd,
                    Y = py,
                    H = ph,
                    History = saved,
                    InputStride = ldx,
                    DeltaStride = ldt,
                    AStride = lda,
                    BStride = ldb,
                    CStride = ldc,
                    OutputStride = ldy,
                    StateStride = ldh,
                    Channels = channels,
                    Length = length,
                    States = n,
                    Units = units,
                    Groups = groups,
                    CheckpointInterval = checkpointInterval,
                    Approximate = approximate,
                };
                Parallelism.For(groups, threads, ref work);
            }

            if (!direct)
            {
                target.CopyTo(destination);
            }

            if (!directState)
            {
                h.CopyTo(state);
            }
        }
        finally
        {
            Dense.Return(rx);
            Dense.Return(rdt);
            Dense.Return(ra);
            Dense.Return(rb);
            Dense.Return(rc);
            Dense.Return(rd);
            Dense.Return(ry);
            Dense.Return(rh);
        }

        return destination;
    }

    internal static unsafe MatrixSpan<T> ScanStepCore<T>(
        ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> delta, ReadOnlyMatrixSpan<T> a, ReadOnlyMatrixSpan<T> b, ReadOnlyMatrixSpan<T> c,
        ReadOnlyVectorSpan<T> d, TensorSpan<T> state, bool approximate, MatrixSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        int n = CheckScanShapes(x, delta, a, b, c, d, destination.Rows, destination.Columns);
        int channels = x.Rows, sequences = x.Columns;
        var stateView = state.AsReadOnly();
        if (stateView.Lengths[0] != channels || stateView.Lengths[1] != n || stateView.Lengths[2] != sequences)
        {
            throw new ArgumentException(
                $"The state must be [{channels}, {n}, {sequences}] but is [{stateView.Lengths[0]}, {stateView.Lengths[1]}, {stateView.Lengths[2]}].",
                nameof(state));
        }

        foreach (var input in new[] { 0, 1, 2, 3, 4 })
        {
            var view = input switch { 0 => x, 1 => delta, 2 => a, 3 => b, _ => c };
            Guard.RequireNoOverlap(view, destination, allowExact: false, nameof(destination));
            Guard.RequireNoOverlap(view, state, nameof(state));
        }

        Guard.RequireNoOverlap(destination.AsReadOnly(), state, nameof(destination));
        Guard.RequireNoOverlap(d, destination, nameof(destination));
        Guard.RequireNoOverlap(d, state, nameof(state));
        if (channels == 0 || sequences == 0)
        {
            return destination;
        }

        T[]? rx = null, rdt = null, ra = null, rb = null, rc = null, rd = null, ry = null, rh = null;
        try
        {
            var xd = Dense.Columns(x, ref rx);
            var dtd = Dense.Columns(delta, ref rdt);
            var ad = Dense.Columns(a, ref ra);
            var bd = Dense.Columns(b, ref rb);
            var cd = Dense.Columns(c, ref rc);
            var dd = Dense.Contiguous(d, ref rd);
            var target = destination;
            bool direct = channels <= 1 || destination.AsReadOnly().RowStride == 1;
            if (!direct)
            {
                ry = ArrayPool<T>.Shared.Rent(channels * sequences);
                target = new MatrixSpan<T>(ry.AsSpan(0, channels * sequences), channels, sequences);
            }

            var h = state;
            bool directState = channels <= 1 || stateView.RowStride == 1;
            if (!directState)
            {
                rh = ArrayPool<T>.Shared.Rent(channels * n * sequences);
                h = new TensorSpan<T>(rh.AsSpan(0, channels * n * sequences), channels, n, sequences);
                state.CopyTo(h);
            }

            var hv = h.AsReadOnly();
            int ldx = xd.ColumnStride, ldt = dtd.ColumnStride, lda = ad.ColumnStride, ldb = bd.ColumnStride, ldc = cd.ColumnStride;
            int ldy = target.AsReadOnly().ColumnStride, ldh = hv.ColumnStride, lds = hv.MatrixStride;
            int threads = Split.Threads(4.0 * channels * n * sequences);
            fixed (T* px = &Dense.Ref(xd), pt = &Dense.Ref(dtd), pa = &Dense.Ref(ad), pb = &Dense.Ref(bd), pc = &Dense.Ref(cd),
                pd = &Dense.Ref(dd), py = &Dense.Ref(target), ph = &System.Runtime.CompilerServices.Unsafe.AsRef(in hv.Reference))
            {
                T* xs = px, ts = pt, aa = pa, bs = pb, cs = pc, ds = d.Length == 0 ? null : pd, ys = py, hs = ph;
                if (n == 0)
                {
                    hs = null;
                }

                Split.For(sequences, Split.Chunks(sequences, threads, 4), threads, (s0, s1) =>
                {
                    var work = ArrayPool<T>.Shared.Rent(Math.Max(1, n * channels));
                    try
                    {
                        fixed (T* pw = work)
                        {
                            for (int s = s0; s < s1; s++)
                            {
                                Kernels.ScanStep(aa, lda, ts + (nint)s * ldt, xs + (nint)s * ldx, bs + (nint)s * ldb, cs + (nint)s * ldc, ds,
                                    hs + (nint)s * lds, ldh, ys + (nint)s * ldy, pw, 0, channels, n, approximate);
                            }
                        }
                    }
                    finally
                    {
                        ArrayPool<T>.Shared.Return(work);
                    }
                });
            }

            if (!direct)
            {
                target.CopyTo(destination);
            }

            if (!directState)
            {
                h.CopyTo(state);
            }
        }
        finally
        {
            Dense.Return(rx);
            Dense.Return(rdt);
            Dense.Return(ra);
            Dense.Return(rb);
            Dense.Return(rc);
            Dense.Return(rd);
            Dense.Return(ry);
            Dense.Return(rh);
        }

        return destination;
    }
}
