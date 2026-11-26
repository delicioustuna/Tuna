using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Tuna.NeuralNetwork;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む。
using Tuna.LinearAlgebra;

/// <summary>
/// Specifies the dimensions of a <see cref="MambaBlock{T}"/>.
/// <para><see cref="MambaBlock{T}"/> の各次元を表します。</para>
/// </summary>
/// <param name="Model">The input and output channel count (<c>d_model</c>). / 入出力チャネル数（<c>d_model</c>）。</param>
/// <param name="Inner">The expanded inner channel count (<c>d_inner</c>; defaults to <c>2 * d_model</c> in <c>mamba_ssm</c>). / 内部チャネル数（<c>d_inner</c>。<c>mamba_ssm</c> の既定値は <c>2 * d_model</c>）。</param>
/// <param name="State">The state-space dimension <c>N</c> (<c>d_state</c>). / 状態空間の次元 <c>N</c>（<c>d_state</c>）。</param>
/// <param name="ConvolutionKernel">The causal 1D convolution kernel length (<c>d_conv</c>). / 因果1次元畳み込みのカーネル長（<c>d_conv</c>）。</param>
/// <param name="DeltaRank">The low-rank projection dimension for step size <c>Δ</c> (<c>dt_rank</c>). / 刻み幅 <c>Δ</c> の低ランク射影次元（<c>dt_rank</c>）。</param>
public readonly record struct MambaDimensions(int Model, int Inner, int State, int ConvolutionKernel, int DeltaRank)
{
    /// <summary>
    /// Creates the default <c>mamba_ssm</c> dimensions for the specified model width.
    /// <para>指定したモデル次元に対する <c>mamba_ssm</c> の既定の次元構成を作成します。</para>
    /// </summary>
    /// <param name="model">The model feature dimension (<c>d_model</c>). / モデルの特徴次元（<c>d_model</c>）。</param>
    /// <returns>Dimensions with inner size <c>2 * model</c>, state size 16, convolution kernel length 4, and delta rank <c>⌈model / 16⌉</c>. / 内部次元 <c>2 * model</c>、状態次元 16、畳み込みカーネル長 4、刻み幅ランク <c>⌈model / 16⌉</c> の次元構成。</returns>
    public static MambaDimensions Default(int model) => new(model, 2 * model, 16, 4, (model + 15) / 16);

    internal int Projection => DeltaRank + 2 * State;
}

/// <summary>
/// Implements a Mamba selective state-space block for inference.
/// <para>推論用の選択的状態空間モデル（Mamba）ブロックを実装します。</para>
/// </summary>
/// <typeparam name="T">The floating-point element type (<see cref="float"/> or <see cref="double"/>). / 浮動小数点の要素型（<see cref="float"/> または <see cref="double"/>）。</typeparam>
/// <remarks>
/// <para>Weights follow the names and row-major layouts of <c>mamba_ssm.Mamba</c>. Sequence matrices have shape <c>[Model × length]</c>. To advance multiple sequences one step at a time, pass each sequence's next input as a column to <see cref="Step(ReadOnlyMatrixSpan{T}, MambaState{T}, bool, MatrixSpan{T})"/>. Concurrent calls using distinct <see cref="MambaState{T}"/> instances are thread-safe.</para>
/// <para>重みは <c>mamba_ssm.Mamba</c> と同じ名前および行優先レイアウトで受け取ります。系列行列の形状は <c>[Model × 系列長]</c> です。複数の系列を 1 ステップずつ進める場合は、各系列の次の入力を列として <see cref="Step(ReadOnlyMatrixSpan{T}, MambaState{T}, bool, MatrixSpan{T})"/> に渡します。異なる <see cref="MambaState{T}"/> を使用する並列呼び出しはスレッドセーフです。</para>
/// </remarks>
/// <seealso href="https://arxiv.org/abs/2312.00752">Mamba: Linear-Time Sequence Modeling with Selective State Spaces (2023)</seealso>
public sealed class MambaBlock<T> where T : unmanaged, IFloatingPointIeee754<T>
{
    private readonly T[] inWeight;     // [2·Inner × Model]
    private readonly T[] inBias;       // 2·Inner または空
    private readonly T[] convWeight;   // [Inner × K]（ピン留め）
    private readonly T[] convBias;     // Inner または空（ピン留め）
    private readonly T[] xWeight;      // [Projection × Inner]
    private readonly T[] deltaWeight;  // [Inner × DeltaRank]
    private readonly T[] deltaBias;    // Inner
    private readonly T[] a;            // [Inner × State] = −exp(A_log)
    private readonly T[] skip;         // D（Inner）
    private readonly T[] outWeight;    // [Model × Inner]
    private readonly T[] outBias;      // Model または空

    /// <summary>
    /// Initializes a new <see cref="MambaBlock{T}"/> from <c>mamba_ssm</c> weights.
    /// <para><c>mamba_ssm</c> の重みから <see cref="MambaBlock{T}"/> を初期化します。</para>
    /// </summary>
    /// <param name="dimensions">The block dimensions. / ブロックの次元構成。</param>
    /// <param name="inProjection"><c>in_proj.weight</c> in row-major <c>[2 * Inner, Model]</c> layout. / <c>in_proj.weight</c>（行優先 <c>[2 * Inner, Model]</c>）。</param>
    /// <param name="convolutionWeight"><c>conv1d.weight</c> in row-major <c>[Inner, 1, ConvolutionKernel]</c> layout. / <c>conv1d.weight</c>（行優先 <c>[Inner, 1, ConvolutionKernel]</c>）。</param>
    /// <param name="convolutionBias"><c>conv1d.bias</c> of length <c>Inner</c>, or empty for zero bias. / <c>conv1d.bias</c>（長さ <c>Inner</c>。空の場合は 0）。</param>
    /// <param name="xProjection"><c>x_proj.weight</c> in row-major <c>[DeltaRank + 2 * State, Inner]</c> layout. / <c>x_proj.weight</c>（行優先 <c>[DeltaRank + 2 * State, Inner]</c>）。</param>
    /// <param name="deltaProjection"><c>dt_proj.weight</c> in row-major <c>[Inner, DeltaRank]</c> layout. / <c>dt_proj.weight</c>（行優先 <c>[Inner, DeltaRank]</c>）。</param>
    /// <param name="deltaProjectionBias"><c>dt_proj.bias</c> of length <c>Inner</c>. / <c>dt_proj.bias</c>（長さ <c>Inner</c>）。</param>
    /// <param name="aLog"><c>A_log</c> in row-major <c>[Inner, State]</c> layout. / <c>A_log</c>（行優先 <c>[Inner, State]</c>）。</param>
    /// <param name="d">The skip connection vector <c>D</c> of length <c>Inner</c>, or empty to omit. / スキップ接続ベクトル <c>D</c>（長さ <c>Inner</c>。空の場合は省略）。</param>
    /// <param name="outProjection"><c>out_proj.weight</c> in row-major <c>[Model, Inner]</c> layout. / <c>out_proj.weight</c>（行優先 <c>[Model, Inner]</c>）。</param>
    /// <param name="inProjectionBias">Optional <c>in_proj.bias</c> of length <c>2 * Inner</c>, or empty for zero bias. / <c>in_proj.bias</c>（長さ <c>2 * Inner</c>。空の場合は 0）。</param>
    /// <param name="outProjectionBias">Optional <c>out_proj.bias</c> of length <c>Model</c>, or empty for zero bias. / <c>out_proj.bias</c>（長さ <c>Model</c>。空の場合は 0）。</param>
    /// <exception cref="ArgumentException">Any weight or bias span length does not match <paramref name="dimensions"/>. / 重みまたはバイアスの長さが <paramref name="dimensions"/> と一致しない場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">Any field of <paramref name="dimensions"/> is less than 1. / <paramref name="dimensions"/> のいずれかの次元が 1 未満の場合。</exception>
    /// <exception cref="NotSupportedException"><typeparamref name="T"/> is neither <see cref="float"/> nor <see cref="double"/>. / <typeparamref name="T"/> が <see cref="float"/> でも <see cref="double"/> でもない場合。</exception>
    public MambaBlock(MambaDimensions dimensions,
        ReadOnlySpan<T> inProjection, ReadOnlySpan<T> convolutionWeight, ReadOnlySpan<T> convolutionBias,
        ReadOnlySpan<T> xProjection, ReadOnlySpan<T> deltaProjection, ReadOnlySpan<T> deltaProjectionBias,
        ReadOnlySpan<T> aLog, ReadOnlySpan<T> d, ReadOnlySpan<T> outProjection,
        ReadOnlySpan<T> inProjectionBias = default, ReadOnlySpan<T> outProjectionBias = default)
    {
        Guard.RequireElementType<T>();
        var (model, inner, state, kernel, rank) = dimensions;
        Guard.RequirePositive(model, nameof(dimensions));
        Guard.RequirePositive(inner, nameof(dimensions));
        Guard.RequirePositive(state, nameof(dimensions));
        Guard.RequirePositive(kernel, nameof(dimensions));
        Guard.RequirePositive(rank, nameof(dimensions));
        int projection = checked(rank + 2 * state);
        Guard.RequireLength(checked(2 * inner * model), inProjection.Length, nameof(inProjection));
        Guard.RequireLength(checked(inner * kernel), convolutionWeight.Length, nameof(convolutionWeight));
        Guard.RequireOptionalLength(inner, convolutionBias.Length, nameof(convolutionBias));
        Guard.RequireLength(checked(projection * inner), xProjection.Length, nameof(xProjection));
        Guard.RequireLength(checked(inner * rank), deltaProjection.Length, nameof(deltaProjection));
        Guard.RequireLength(inner, deltaProjectionBias.Length, nameof(deltaProjectionBias));
        Guard.RequireLength(checked(inner * state), aLog.Length, nameof(aLog));
        Guard.RequireOptionalLength(inner, d.Length, nameof(d));
        Guard.RequireLength(checked(model * inner), outProjection.Length, nameof(outProjection));
        Guard.RequireOptionalLength(2 * inner, inProjectionBias.Length, nameof(inProjectionBias));
        Guard.RequireOptionalLength(model, outProjectionBias.Length, nameof(outProjectionBias));
        Dimensions = dimensions;
        inWeight = FromRowMajor(inProjection, 2 * inner, model);
        inBias = inProjectionBias.ToArray();
        convWeight = Dense.ToPinnedArray(new ReadOnlyMatrixSpan<T>(convolutionWeight, kernel, inner).T);
        convBias = GC.AllocateUninitializedArray<T>(convolutionBias.Length, pinned: true);
        convolutionBias.CopyTo(convBias);
        xWeight = FromRowMajor(xProjection, projection, inner);
        deltaWeight = FromRowMajor(deltaProjection, inner, rank);
        deltaBias = deltaProjectionBias.ToArray();
        a = FromRowMajor(aLog, inner, state);
        for (int i = 0; i < a.Length; i++)
        {
            a[i] = -T.Exp(a[i]);
        }

        skip = Pinned(d);
        outWeight = FromRowMajor(outProjection, model, inner);
        outBias = outProjectionBias.ToArray();
    }

    /// <summary>
    /// Gets the dimensions of this block.
    /// <para>このブロックの次元構成を取得します。</para>
    /// </summary>
    public MambaDimensions Dimensions { get; }

    /// <summary>
    /// Creates a zero-initialized state for the specified number of independent sequences.
    /// <para>指定した系列数に対応するゼロ初期化済みの状態オブジェクトを作成します。</para>
    /// </summary>
    /// <param name="sequences">The number of independent sequences. / 独立な系列数。</param>
    /// <returns>A new <see cref="MambaState{T}"/> with all state buffers initialized to 0. / すべての状態が 0 で初期化された新しい <see cref="MambaState{T}"/>。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="sequences"/> is negative. / <paramref name="sequences"/> が負の場合。</exception>
    public MambaState<T> CreateState(int sequences) => new(Dimensions, sequences);

    /// <summary>
    /// Processes a single sequence from a zero initial state and returns a new matrix.
    /// <para>初期状態 0 から 1 本の系列を処理し、出力を新しい行列として返します。</para>
    /// </summary>
    /// <param name="u">The input sequence matrix of shape <c>[Model × L]</c>. / 入力系列行列 <c>[Model × L]</c>。</param>
    /// <param name="approximate"><see langword="true"/> to use approximate SiLU, softplus, and exp (as in <see cref="XMath.Silu{T}(T, bool)"/>); <see langword="false"/> for exact evaluation. / SiLU、softplus、exp に近似式（<see cref="XMath.Silu{T}(T, bool)"/> 相当）を使う場合は <see langword="true"/>、高精度計算を使う場合は <see langword="false"/>。</param>
    /// <returns>A new matrix of shape <c>[Model × L]</c>. / 出力を格納した新しい行列 <c>[Model × L]</c>。</returns>
    /// <exception cref="ArgumentException"><c>u.Rows</c> does not match <see cref="MambaDimensions.Model"/>. / <c>u.Rows</c> が <see cref="MambaDimensions.Model"/> と一致しない場合。</exception>
    public Matrix<T> Forward(ReadOnlyMatrixSpan<T> u, bool approximate)
    {
        var result = Matrix<T>.Uninitialized(Dimensions.Model, u.Columns);
        Forward(u, null, approximate, result);
        return result;
    }

    /// <summary>
    /// Processes a single sequence, optionally continuing from and updating <paramref name="state"/>, and writes the result to <paramref name="destination"/>.
    /// <para>1 本の系列を処理し（必要に応じて <paramref name="state"/> を引き継いで更新）、結果を <paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="u">The input sequence matrix of shape <c>[Model × L]</c>. / 入力系列行列 <c>[Model × L]</c>。</param>
    /// <param name="state">Optional single-sequence state updated in place to the final position, or <see langword="null"/> to start from zero without retaining state. / 1 系列分の状態（最終位置後の状態に更新されます）。<see langword="null"/> の場合は初期状態 0 から開始し状態を保持しません。</param>
    /// <param name="approximate"><see langword="true"/> to use approximate SiLU, softplus, and exp; <see langword="false"/> for exact evaluation. / SiLU、softplus、exp に近似式を使う場合は <see langword="true"/>、高精度計算を使う場合は <see langword="false"/>。</param>
    /// <param name="destination">The output matrix span of shape <c>[Model × L]</c>. Must either alias <paramref name="u"/> identically or not overlap <paramref name="u"/>. / 出力先行列 <c>[Model × L]</c>。<paramref name="u"/> と完全一致するか重複しないビューである必要があります。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para>Chunked calls passing the updated <paramref name="state"/> produce bit-identical results to a single combined call.</para>
    /// <para>系列を分割して更新された <paramref name="state"/> を引き継ぎながら呼び出しても、まとめて処理した場合とビット単位で同一の結果になります。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">The matrix shapes, state dimensions or sequence count, or memory overlap rules are violated. / 行列形状、状態の次元や系列数、またはメモリ重複の条件を満たさない場合。</exception>
    public unsafe MatrixSpan<T> Forward(ReadOnlyMatrixSpan<T> u, MambaState<T>? state, bool approximate, MatrixSpan<T> destination)
    {
        var (model, inner, n, kernel, rank) = Dimensions;
        int length = u.Columns;
        Guard.RequireShape(model, length, u.Rows, u.Columns, nameof(u));
        Guard.RequireShape(model, length, destination.Rows, destination.Columns, nameof(destination));
        Guard.RequireNoOverlap(u, destination, allowExact: true, nameof(destination));
        if (state is not null)
        {
            RequireState(state, 1);
        }

        if (length == 0)
        {
            return destination;
        }

        int history = kernel - 1, projection = Dimensions.Projection;
        T[]? ru = null, ry = null, rh = null;
        var work = ArrayPool<T>.Shared.Rent(WorkLength(length));
        var past = ArrayPool<T>.Shared.Rent(Math.Max(1, inner * history));
        try
        {
            var ud = Dense.Columns(u, ref ru);
            var target = Target(destination, ref ry);

            // 畳み込み用の過去入力を古い順に history 列展開する（状態がなければ 0 クリア）
            for (int j = 0; j < history; j++)
            {
                var column = past.AsSpan(j * inner, inner);
                if (state is null)
                {
                    column.Clear();
                }
                else
                {
                    state.History.AsSpan(((state.Head + j) % history) * inner, inner).CopyTo(column);
                }
            }

            MatrixSpan<T> h;
            if (state is not null)
            {
                h = state.Ssm.GetMatrix(0);
            }
            else
            {
                rh = ArrayPool<T>.Shared.Rent(inner * n);
                h = new MatrixSpan<T>(rh.AsSpan(0, inner * n), inner, n);
                h.Fill(T.Zero);
            }

            int threads = Split.Threads(Work(length));
            int chunks = Split.Chunks(length, threads, 8);
            fixed (T* pu = &Dense.Ref(ud), pw = work, pp = past, py = &Dense.Ref(target))
            {
                T* us = pu, ps = pp, ys = py;
                int ldu = ud.ColumnStride, ldy = target.AsReadOnly().ColumnStride;
                var buffers = new Buffers(pw, inner, projection, length);

                Split.For(length, chunks, threads, (s, e) => InProject(us, ldu, buffers, s, e));
                Split.For(length, chunks, threads, (s, e) =>
                {
                    Convolve(buffers, ps, s, e);
                    Project(buffers, s, e, approximate);
                });
                XMathExtensions.ScanCore<T>(Views.Read(buffers.Xc, inner, length, inner), Views.Read(buffers.Delta, inner, length, inner),
                    new ReadOnlyMatrixSpan<T>(a, inner, n), Views.Read(buffers.Xdb + rank, n, length, projection),
                    Views.Read(buffers.Xdb + rank + n, n, length, projection), skip, h, approximate, Views.Write(buffers.Y, inner, length, inner));
                Split.For(length, chunks, threads, (s, e) => Finish(buffers, ys, ldy, s, e, approximate));

                // 過去 history 列と今回 length 列を連結した末尾 history 列を古い順に状態へ保存する
                if (state is not null && history > 0)
                {
                    var next = state.History;
                    for (int j = 0; j < history; j++)
                    {
                        int index = length + j;
                        var source = index < history
                            ? new ReadOnlySpan<T>(ps + (nint)index * inner, inner)
                            : new ReadOnlySpan<T>(buffers.Xz + (nint)(index - history) * 2 * inner, inner);
                        source.CopyTo(next.AsSpan(j * inner, inner));
                    }

                    state.Head = 0;
                }
            }

            if (ry is not null)
            {
                target.CopyTo(destination);
            }
        }
        finally
        {
            ArrayPool<T>.Shared.Return(work);
            ArrayPool<T>.Shared.Return(past);
            Dense.Return(ru);
            Dense.Return(ry);
            Dense.Return(rh);
        }

        return destination;
    }

    /// <summary>
    /// Advances multiple independent sequences by one step and returns a new matrix.
    /// <para>複数の独立した系列を 1 ステップ進め、出力を新しい行列として返します。</para>
    /// </summary>
    /// <param name="u">The next input for each sequence, of shape <c>[Model × S]</c> (one column per sequence). / 各系列の次の入力 <c>[Model × S]</c>（列が系列）。</param>
    /// <param name="state">The state for <c>S</c> sequences (created via <see cref="CreateState(int)"/>), updated in place by one step. / <c>S</c> 本の系列の状態（<see cref="CreateState(int)"/> で作成。1 ステップ分インプレースで更新されます）。</param>
    /// <param name="approximate"><see langword="true"/> to use approximate SiLU, softplus, and exp; <see langword="false"/> for exact evaluation. / SiLU、softplus、exp に近似式を使う場合は <see langword="true"/>、高精度計算を使う場合は <see langword="false"/>。</param>
    /// <returns>A new matrix of shape <c>[Model × S]</c>. / 出力を格納した新しい行列 <c>[Model × S]</c>。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="state"/> is <see langword="null"/>. / <paramref name="state"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException">The input shape or state dimensions/sequence count do not match. / 入力形状、または状態の次元や系列数が一致しない場合。</exception>
    public Matrix<T> Step(ReadOnlyMatrixSpan<T> u, MambaState<T> state, bool approximate)
    {
        var result = Matrix<T>.Uninitialized(Dimensions.Model, u.Columns);
        Step(u, state, approximate, result);
        return result;
    }

    /// <summary>
    /// Advances multiple independent sequences by one step, updating <paramref name="state"/> in place and writing outputs to <paramref name="destination"/>.
    /// <para>複数の独立した系列を 1 ステップ進め、<paramref name="state"/> を更新するとともに結果を <paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="u">The next input for each sequence, of shape <c>[Model × S]</c> (one column per sequence). / 各系列の次の入力 <c>[Model × S]</c>（列が系列）。</param>
    /// <param name="state">The state for <c>S</c> sequences, updated in place by one step. / <c>S</c> 本の系列の状態（1 ステップ分インプレースで更新されます）。</param>
    /// <param name="approximate"><see langword="true"/> to use approximate SiLU, softplus, and exp; <see langword="false"/> for exact evaluation. / SiLU、softplus、exp に近似式を使う場合は <see langword="true"/>、高精度計算を使う場合は <see langword="false"/>。</param>
    /// <param name="destination">The output matrix span of shape <c>[Model × S]</c>. Must either alias <paramref name="u"/> identically or not overlap <paramref name="u"/>. / 出力先行列 <c>[Model × S]</c>。<paramref name="u"/> と完全一致するか重複しないビューである必要があります。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para>Each output column matches the final column produced by passing that sequence's full prefix to <see cref="Forward(ReadOnlyMatrixSpan{T}, MambaState{T}?, bool, MatrixSpan{T})"/>.</para>
    /// <para>各列の出力は、その系列のこれまでの入力全体を <see cref="Forward(ReadOnlyMatrixSpan{T}, MambaState{T}?, bool, MatrixSpan{T})"/> で処理した際の最終列と一致します。</para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="state"/> is <see langword="null"/>. / <paramref name="state"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException">The matrix shapes, state dimensions or sequence count, or memory overlap rules are violated. / 行列形状、状態の次元や系列数、またはメモリ重複の条件を満たさない場合。</exception>
    public unsafe MatrixSpan<T> Step(ReadOnlyMatrixSpan<T> u, MambaState<T> state, bool approximate, MatrixSpan<T> destination)
    {
        ArgumentNullException.ThrowIfNull(state);
        var (model, inner, n, kernel, rank) = Dimensions;
        int sequences = u.Columns;
        Guard.RequireShape(model, sequences, u.Rows, u.Columns, nameof(u));
        Guard.RequireShape(model, sequences, destination.Rows, destination.Columns, nameof(destination));
        Guard.RequireNoOverlap(u, destination, allowExact: true, nameof(destination));
        RequireState(state, sequences);
        if (sequences == 0)
        {
            return destination;
        }

        int history = kernel - 1, projection = Dimensions.Projection;
        T[]? ru = null, ry = null;
        var work = ArrayPool<T>.Shared.Rent(WorkLength(sequences));
        try
        {
            var ud = Dense.Columns(u, ref ru);
            var target = Target(destination, ref ry);
            var ssm = state.Ssm;
            int ldh = (int)ssm.Strides[1], lds = (int)ssm.Strides[2], slotLength = inner * sequences, head = state.Head;
            int threads = Split.Threads(Work(sequences));
            fixed (T* pu = &Dense.Ref(ud), pw = work, pr = state.History, ph = ssm.AsSpan(), py = &Dense.Ref(target))
            {
                T* us = pu, ring = pr, hs = ph, ys = py;
                int ldu = ud.ColumnStride, ldy = target.AsReadOnly().ColumnStride;
                var buffers = new Buffers(pw, inner, projection, sequences);
                T* weights = Pointer(convWeight), biases = convBias.Length == 0 ? null : Pointer(convBias);
                T* aa = Pointer(a), ds = skip.Length == 0 ? null : Pointer(skip);
                Split.For(sequences, Split.Chunks(sequences, threads, 4), threads, (s0, s1) =>
                {
                    InProject(us, ldu, buffers, s0, s1);

                    // リングバッファの古い順にタップを組み、現在入力を最古スロットへ上書きする
                    T** taps = stackalloc T*[kernel];
                    for (int s = s0; s < s1; s++)
                    {
                        T* x = buffers.Xz + (nint)s * 2 * inner;
                        for (int j = 0; j < history; j++)
                        {
                            taps[j] = ring + (nint)((head + j) % history) * slotLength + (nint)s * inner;
                        }

                        taps[history] = x;
                        Kernels.DepthwiseTaps(taps, kernel, weights, inner, biases, buffers.Xc + (nint)s * inner, inner);
                        if (history > 0)
                        {
                            Buffer.MemoryCopy(x, ring + (nint)head * slotLength + (nint)s * inner, (long)inner * sizeof(T), (long)inner * sizeof(T));
                        }
                    }

                    Project(buffers, s0, s1, approximate);
                    var scanWork = ArrayPool<T>.Shared.Rent(n * inner);
                    try
                    {
                        fixed (T* pw2 = scanWork)
                        {
                            for (int s = s0; s < s1; s++)
                            {
                                T* xdb = buffers.Xdb + (nint)s * projection;
                                Kernels.ScanStep(aa, inner, buffers.Delta + (nint)s * inner, buffers.Xc + (nint)s * inner, xdb + rank, xdb + rank + n, ds,
                                    hs + (nint)s * lds, ldh, buffers.Y + (nint)s * inner, pw2, 0, inner, n, approximate);
                            }
                        }
                    }
                    finally
                    {
                        ArrayPool<T>.Shared.Return(scanWork);
                    }

                    Finish(buffers, ys, ldy, s0, s1, approximate);
                });
            }

            if (history > 0)
            {
                state.Head = (head + 1) % history;
            }

            if (ry is not null)
            {
                target.CopyTo(destination);
            }
        }
        finally
        {
            ArrayPool<T>.Shared.Return(work);
            Dense.Return(ru);
            Dense.Return(ry);
        }

        return destination;
    }

    private void RequireState(MambaState<T> state, int sequences)
    {
        if (state.Dimensions != Dimensions)
        {
            throw new ArgumentException($"The state was created for {state.Dimensions} but the block has {Dimensions}.", nameof(state));
        }

        if (state.Sequences != sequences)
        {
            throw new ArgumentException($"The state holds {state.Sequences} sequences but {sequences} are given.", nameof(state));
        }
    }

    private double Work(int columns)
    {
        var (model, inner, n, kernel, rank) = Dimensions;
        return (double)columns * (2.0 * inner * model + (double)Dimensions.Projection * inner + (double)inner * rank
            + (double)model * inner + 4.0 * inner * n + (double)inner * kernel);
    }

    // 列ごとの作業領域長: xz [2·Inner] + xc [Inner] + x_proj [Projection] + Δ [Inner] + y [Inner]
    private int WorkLength(int columns) => checked((5 * Dimensions.Inner + Dimensions.Projection) * columns);

    private MatrixSpan<T> Target(MatrixSpan<T> destination, ref T[]? rented)
    {
        if (Dimensions.Model <= 1 || destination.AsReadOnly().RowStride == 1)
        {
            return destination;
        }

        int length = Dimensions.Model * destination.Columns;
        rented = ArrayPool<T>.Shared.Rent(length);
        return new MatrixSpan<T>(rented.AsSpan(0, length), Dimensions.Model, destination.Columns);
    }

    private static unsafe T* Pointer(T[] array) => (T*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(array));

    private unsafe void InProject(T* u, int ldu, Buffers buffers, int s, int e)
    {
        int inner = Dimensions.Inner, model = Dimensions.Model, count = e - s;
        var xz = Views.Write(buffers.Xz + (nint)s * 2 * inner, 2 * inner, count, 2 * inner);
        XMath.Multiply(Views.Read(Pointer(inWeight), 2 * inner, model, 2 * inner), Views.Read(u + (nint)s * ldu, model, count, ldu), xz);
        if (inBias.Length != 0)
        {
            XMathExtensions.AddBias(xz, inBias);
        }
    }

    private unsafe void Convolve(Buffers buffers, T* past, int s, int e)
    {
        int inner = Dimensions.Inner, kernel = Dimensions.ConvolutionKernel, history = kernel - 1;
        T* weights = Pointer(convWeight), biases = convBias.Length == 0 ? null : Pointer(convBias);
        T** taps = stackalloc T*[kernel];
        for (int t = s; t < e; t++)
        {
            for (int j = 0; j < kernel; j++)
            {
                int source = t - history + j;
                taps[j] = source >= 0 ? buffers.Xz + (nint)source * 2 * inner : past + (nint)(history + source) * inner;
            }

            Kernels.DepthwiseTaps(taps, kernel, weights, inner, biases, buffers.Xc + (nint)t * inner, inner);
        }
    }

    private unsafe void Project(Buffers buffers, int s, int e, bool approximate)
    {
        int inner = Dimensions.Inner, rank = Dimensions.DeltaRank, projection = Dimensions.Projection, count = e - s;
        var xc = Views.Write(buffers.Xc + (nint)s * inner, inner, count, inner);
        XMath.InPlace.Silu(xc, approximate);
        var xdb = Views.Write(buffers.Xdb + (nint)s * projection, projection, count, projection);
        XMath.Multiply(Views.Read(Pointer(xWeight), projection, inner, projection), xc, xdb);
        var delta = Views.Write(buffers.Delta + (nint)s * inner, inner, count, inner);
        XMath.Multiply(Views.Read(Pointer(deltaWeight), inner, rank, inner), xdb[..rank, ..], delta);
        XMathExtensions.AddBias(delta, deltaBias);
        XMath.InPlace.Softplus(delta, approximate);
    }

    private unsafe void Finish(Buffers buffers, T* output, int ldo, int s, int e, bool approximate)
    {
        int inner = Dimensions.Inner, model = Dimensions.Model, count = e - s;
        var z = Views.Write(buffers.Xz + (nint)s * 2 * inner + inner, inner, count, 2 * inner);
        XMath.InPlace.Silu(z, approximate);
        var y = Views.Write(buffers.Y + (nint)s * inner, inner, count, inner);
        XMath.Hadamard(y, z, y);
        var o = Views.Write(output + (nint)s * ldo, model, count, ldo);
        XMath.Multiply(Views.Read(Pointer(outWeight), model, inner, model), y, o);
        if (outBias.Length != 0)
        {
            XMathExtensions.AddBias(o, outBias);
        }
    }

    private static T[] FromRowMajor(ReadOnlySpan<T> values, int rows, int columns) =>
        Dense.ToPinnedArray(new ReadOnlyMatrixSpan<T>(values, columns, rows).T);

    private static T[] Pinned(ReadOnlySpan<T> values)
    {
        var array = GC.AllocateUninitializedArray<T>(values.Length, pinned: true);
        values.CopyTo(array);
        return array;
    }

    // 1回の呼び出しで使う中間バッファ群（列優先、列ストライドは各領域の行数）
    private readonly unsafe struct Buffers
    {
        public Buffers(T* start, int inner, int projection, int columns)
        {
            Xz = start;
            Xc = Xz + (nint)2 * inner * columns;
            Xdb = Xc + (nint)inner * columns;
            Delta = Xdb + (nint)projection * columns;
            Y = Delta + (nint)inner * columns;
        }

        // in_proj 出力 [2·Inner × 列]（前半 x、後半 z）
        public T* Xz { get; }

        // 畳み込み・SiLU 後の x [Inner × 列]
        public T* Xc { get; }

        // x_proj 出力 [Projection × 列]（δ, B, C）
        public T* Xdb { get; }

        // Δ [Inner × 列]
        public T* Delta { get; }

        // 選択的走査出力 y [Inner × 列]
        public T* Y { get; }
    }
}

/// <summary>
/// Holds the recurrent inference state (causal convolution history and selective-scan state <c>h</c>) for one or more sequences in a <see cref="MambaBlock{T}"/>.
/// <para><see cref="MambaBlock{T}"/> における複数系列の推論状態（因果畳み込みの過去入力と選択的走査の状態 <c>h</c>）を保持します。</para>
/// </summary>
/// <typeparam name="T">The floating-point element type (<see cref="float"/> or <see cref="double"/>). / 浮動小数点の要素型（<see cref="float"/> または <see cref="double"/>）。</typeparam>
public sealed class MambaState<T> where T : unmanaged, IFloatingPointIeee754<T>
{
    internal MambaState(MambaDimensions dimensions, int sequences)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(sequences);
        Dimensions = dimensions;
        Sequences = sequences;
        History = GC.AllocateArray<T>(checked((dimensions.ConvolutionKernel - 1) * dimensions.Inner * sequences), pinned: true);
        Ssm = new Tensor<T>(dimensions.Inner, dimensions.State, sequences);
    }

    /// <summary>
    /// Gets the dimensions of the <see cref="MambaBlock{T}"/> for which this state was created.
    /// <para>この状態を作成した <see cref="MambaBlock{T}"/> の次元構成を取得します。</para>
    /// </summary>
    public MambaDimensions Dimensions { get; }

    /// <summary>
    /// Gets the number of independent sequences tracked by this state.
    /// <para>この状態が保持する独立な系列数を取得します。</para>
    /// </summary>
    public int Sequences { get; }

    // 過去入力のリングバッファ（ConvolutionKernel − 1 個の [Inner × Sequences]、最古スロットが Head）
    internal T[] History { get; }

    internal int Head { get; set; }

    // 選択的走査の状態テンソル [Inner, State, Sequences]
    internal Tensor<T> Ssm { get; }

    /// <summary>
    /// Resets all sequences to their initial zero state.
    /// <para>すべての系列の状態を初期状態（0）にリセットします。</para>
    /// </summary>
    public void Reset()
    {
        Array.Clear(History);
        Head = 0;
        Ssm.AsSpan().Clear();
    }
}
