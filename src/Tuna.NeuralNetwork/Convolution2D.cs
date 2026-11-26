using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Tuna.NeuralNetwork;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む。
using Tuna.LinearAlgebra;

/// <summary>
/// Specifies the number of zero-padded elements added to each edge of a 2D convolution input.
/// <para>2次元畳み込みの入力の各辺に追加するゼロパディング数を表します。</para>
/// </summary>
/// <param name="Top">The padding added to the start (top) of the height dimension. / 高さ方向の先頭（上）のパディング数。</param>
/// <param name="Bottom">The padding added to the end (bottom) of the height dimension. / 高さ方向の末尾（下）のパディング数。</param>
/// <param name="Left">The padding added to the start (left) of the width dimension. / 幅方向の先頭（左）のパディング数。</param>
/// <param name="Right">The padding added to the end (right) of the width dimension. / 幅方向の末尾（右）のパディング数。</param>
public readonly record struct Padding2D(int Top, int Bottom, int Left, int Right)
{
    /// <summary>
    /// Initializes a symmetric <see cref="Padding2D"/> with <paramref name="height"/> on top and bottom and <paramref name="width"/> on left and right, matching PyTorch <c>padding=(height, width)</c>.
    /// <para>上下に <paramref name="height"/>、左右に <paramref name="width"/> を設定する対称な <see cref="Padding2D"/> を初期化します（PyTorch の <c>padding=(height, width)</c> に対応）。</para>
    /// </summary>
    /// <param name="height">The top and bottom padding. / 上下のパディング数。</param>
    /// <param name="width">The left and right padding. / 左右のパディング数。</param>
    public Padding2D(int height, int width) : this(height, height, width, width)
    {
    }
}

/// <summary>
/// Performs 2D convolution (cross-correlation) over feature maps.
/// <para>特徴マップに対する 2 次元畳み込み（相互相関）を実行します。</para>
/// </summary>
/// <typeparam name="T">The floating-point element type (<see cref="float"/> or <see cref="double"/>). / 浮動小数点の要素型（<see cref="float"/> または <see cref="double"/>）。</typeparam>
/// <remarks>
/// <para>Feature maps use the layout <c>[channels, width, height]</c>. Permute row-major PyTorch <c>[channels, height, width]</c> tensors before passing them.</para>
/// <para>Computes <c>y(o, w, h) = b_o + Σ_{c, i, j} K(o, c, i, j) * x(c, w * sw + j * dw - left, h * sh + i * dh - top)</c> with out-of-bounds values treated as 0, equivalent to PyTorch <c>nn.Conv2d</c> with <c>groups = 1</c>. Weights are read-only, so concurrent calls on different inputs are thread-safe.</para>
/// <para>特徴マップのレイアウトは <c>[チャネル, 幅, 高さ]</c> です。PyTorch の行優先 <c>[チャネル, 高さ, 幅]</c> テンソルは軸を並べ替えてから渡してください。</para>
/// <para><c>y(o, w, h) = b_o + Σ_{c, i, j} K(o, c, i, j) * x(c, w * sw + j * dw - left, h * sh + i * dh - top)</c>（入力範囲外は 0）を計算し、PyTorch の <c>nn.Conv2d</c>（<c>groups = 1</c>）に対応します。重みは読み取り専用であるため、異なる入力に対する並列呼び出しはスレッドセーフです。</para>
/// </remarks>
public sealed partial class Convolution2D<T> where T : unmanaged, IFloatingPointIeee754<T>
{
    // 1回の行列積で処理する im2col 作業領域の要素数上限
    internal const int ColumnBufferElements = 1 << 18;

    // 列優先 [OutChannels × K]（K の添字は (i·KernelWidth + j)·InChannels + c）。ピン留め配列のためポインタで直接参照する。
    private readonly T[] weight;
    private readonly T[] bias;

    /// <summary>
    /// Initializes a new <see cref="Convolution2D{T}"/> layer from weights and optional bias.
    /// <para>重みと任意のバイアスから <see cref="Convolution2D{T}"/> を初期化します。</para>
    /// </summary>
    /// <param name="weight">The kernel weights in row-major <c>[OutChannels, InChannels, KernelHeight, KernelWidth]</c> order, matching PyTorch <c>nn.Conv2d.weight</c>. / 重み（PyTorch <c>nn.Conv2d.weight</c> と同じ行優先 <c>[OutChannels, InChannels, KernelHeight, KernelWidth]</c>）。</param>
    /// <param name="bias">Per-output-channel bias of length <paramref name="outChannels"/>, or empty for zero bias. / 出力チャネルごとのバイアス（長さ <paramref name="outChannels"/>。空の場合は 0）。</param>
    /// <param name="outChannels">The number of output channels. / 出力チャネル数。</param>
    /// <param name="inChannels">The number of input channels. / 入力チャネル数。</param>
    /// <param name="kernelHeight">The kernel height. / カーネルの高さ。</param>
    /// <param name="kernelWidth">The kernel width. / カーネルの幅。</param>
    /// <param name="strideHeight">The stride along the height dimension. / 高さ方向のストライド。</param>
    /// <param name="strideWidth">The stride along the width dimension. / 幅方向のストライド。</param>
    /// <param name="dilationHeight">The dilation along the height dimension. / 高さ方向のダイレーション（膨張率）。</param>
    /// <param name="dilationWidth">The dilation along the width dimension. / 幅方向のダイレーション（膨張率）。</param>
    /// <exception cref="ArgumentException">The length of <paramref name="weight"/> or <paramref name="bias"/> does not match the specified dimensions. / <paramref name="weight"/> または <paramref name="bias"/> の長さが指定した次元と一致しない場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">Any channel count, kernel dimension, stride, or dilation is less than 1. / チャネル数、カーネルサイズ、ストライド、またはダイレーションのいずれかが 1 未満の場合。</exception>
    /// <exception cref="NotSupportedException"><typeparamref name="T"/> is neither <see cref="float"/> nor <see cref="double"/>. / <typeparamref name="T"/> が <see cref="float"/> でも <see cref="double"/> でもない場合。</exception>
    public Convolution2D(ReadOnlySpan<T> weight, ReadOnlySpan<T> bias, int outChannels, int inChannels, int kernelHeight, int kernelWidth,
        int strideHeight = 1, int strideWidth = 1, int dilationHeight = 1, int dilationWidth = 1)
    {
        Guard.RequireElementType<T>();
        Guard.RequirePositive(outChannels, nameof(outChannels));
        Guard.RequirePositive(inChannels, nameof(inChannels));
        Guard.RequirePositive(kernelHeight, nameof(kernelHeight));
        Guard.RequirePositive(kernelWidth, nameof(kernelWidth));
        Guard.RequirePositive(strideHeight, nameof(strideHeight));
        Guard.RequirePositive(strideWidth, nameof(strideWidth));
        Guard.RequirePositive(dilationHeight, nameof(dilationHeight));
        Guard.RequirePositive(dilationWidth, nameof(dilationWidth));
        int k = checked(kernelHeight * kernelWidth * inChannels);
        Guard.RequireLength(checked(outChannels * k), weight.Length, nameof(weight));
        Guard.RequireOptionalLength(outChannels, bias.Length, nameof(bias));
        OutChannels = outChannels;
        InChannels = inChannels;
        KernelHeight = kernelHeight;
        KernelWidth = kernelWidth;
        StrideHeight = strideHeight;
        StrideWidth = strideWidth;
        DilationHeight = dilationHeight;
        DilationWidth = dilationWidth;
        this.weight = GC.AllocateUninitializedArray<T>(outChannels * k, pinned: true);
        for (int o = 0; o < outChannels; o++)
        {
            for (int c = 0; c < inChannels; c++)
            {
                for (int i = 0; i < kernelHeight; i++)
                {
                    for (int j = 0; j < kernelWidth; j++)
                    {
                        int source = ((o * inChannels + c) * kernelHeight + i) * kernelWidth + j;
                        int row = (i * kernelWidth + j) * inChannels + c;
                        this.weight[row * outChannels + o] = weight[source];
                    }
                }
            }
        }

        this.bias = GC.AllocateUninitializedArray<T>(bias.Length, pinned: true);
        bias.CopyTo(this.bias);
    }

    /// <summary>
    /// Gets the number of output channels.
    /// <para>出力チャネル数を取得します。</para>
    /// </summary>
    public int OutChannels { get; }

    /// <summary>
    /// Gets the number of input channels.
    /// <para>入力チャネル数を取得します。</para>
    /// </summary>
    public int InChannels { get; }

    /// <summary>
    /// Gets the kernel height.
    /// <para>カーネルの高さを取得します。</para>
    /// </summary>
    public int KernelHeight { get; }

    /// <summary>
    /// Gets the kernel width.
    /// <para>カーネルの幅を取得します。</para>
    /// </summary>
    public int KernelWidth { get; }

    /// <summary>
    /// Gets the stride along the height dimension.
    /// <para>高さ方向のストライドを取得します。</para>
    /// </summary>
    public int StrideHeight { get; }

    /// <summary>
    /// Gets the stride along the width dimension.
    /// <para>幅方向のストライドを取得します。</para>
    /// </summary>
    public int StrideWidth { get; }

    /// <summary>
    /// Gets the dilation along the height dimension.
    /// <para>高さ方向のダイレーションを取得します。</para>
    /// </summary>
    public int DilationHeight { get; }

    /// <summary>
    /// Gets the dilation along the width dimension.
    /// <para>幅方向のダイレーションを取得します。</para>
    /// </summary>
    public int DilationWidth { get; }

    // im2col の1列の長さ
    internal int ColumnLength => KernelHeight * KernelWidth * InChannels;

    /// <summary>
    /// Computes the output height for the specified input height and padding.
    /// <para>指定した入力の高さとパディングに対する出力の高さを計算します。</para>
    /// </summary>
    /// <param name="height">The input height. / 入力の高さ。</param>
    /// <param name="padding">The padding applied to the input. / 入力に適用するパディング。</param>
    /// <returns>The output height, or 0 if the kernel does not fit. / 出力の高さ（カーネルが収まらない場合は 0）。</returns>
    public int OutputHeight(int height, Padding2D padding) =>
        OutputLength(height, padding.Top, padding.Bottom, KernelHeight, StrideHeight, DilationHeight);

    /// <summary>
    /// Computes the output width for the specified input width and padding.
    /// <para>指定した入力の幅とパディングに対する出力の幅を計算します。</para>
    /// </summary>
    /// <param name="width">The input width. / 入力の幅。</param>
    /// <param name="padding">The padding applied to the input. / 入力に適用するパディング。</param>
    /// <returns>The output width, or 0 if the kernel does not fit. / 出力の幅（カーネルが収まらない場合は 0）。</returns>
    public int OutputWidth(int width, Padding2D padding) =>
        OutputLength(width, padding.Left, padding.Right, KernelWidth, StrideWidth, DilationWidth);

    internal static int OutputLength(int length, int before, int after, int kernel, int stride, int dilation)
    {
        long span = (long)length + before + after - (long)dilation * (kernel - 1) - 1;
        return span < 0 ? 0 : checked((int)(span / stride + 1));
    }

    /// <summary>
    /// Applies the 2D convolution to <paramref name="x"/> and returns a new tensor.
    /// <para>特徴マップ <paramref name="x"/> に 2 次元畳み込みを適用し、新しいテンソルとして返します。</para>
    /// </summary>
    /// <param name="x">The input feature map of shape <c>[InChannels, W, H]</c>. / 入力特徴マップ <c>[InChannels, W, H]</c>。</param>
    /// <param name="padding">The zero-padding applied to each edge. / 各辺に適用するゼロパディング。</param>
    /// <returns>A new tensor of shape <c>[OutChannels, OutputWidth, OutputHeight]</c>. / 出力を格納した新しいテンソル <c>[OutChannels, OutputWidth, OutputHeight]</c>。</returns>
    /// <exception cref="ArgumentException"><paramref name="x"/> does not have <see cref="InChannels"/> channels. / <paramref name="x"/> のチャネル数が <see cref="InChannels"/> と一致しない場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">Any padding value is negative. / パディングのいずれかが負の場合。</exception>
    public Tensor<T> Forward(ReadOnlyTensorSpan<T> x, Padding2D padding)
    {
        RequirePadding(padding);
        var result = new Tensor<T>(OutChannels, OutputWidth((int)x.Lengths[1], padding), OutputHeight((int)x.Lengths[2], padding));
        Forward(x, padding, result);
        return result;
    }

    /// <summary>
    /// Applies the 2D convolution to <paramref name="x"/> and writes the result into <paramref name="destination"/>.
    /// <para>特徴マップ <paramref name="x"/> に 2 次元畳み込みを適用し、結果を <paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input feature map of shape <c>[InChannels, W, H]</c>. / 入力特徴マップ <c>[InChannels, W, H]</c>。</param>
    /// <param name="padding">The zero-padding applied to each edge. / 各辺に適用するゼロパディング。</param>
    /// <param name="destination">The output tensor span of shape <c>[OutChannels, OutputWidth, OutputHeight]</c>. Must not overlap <paramref name="x"/>. / 出力先テンソルスパン <c>[OutChannels, OutputWidth, OutputHeight]</c>。<paramref name="x"/> と重複してはなりません。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException">The input or destination shape does not match, or <paramref name="destination"/> overlaps <paramref name="x"/>. / 入力または出力先の形状が一致しない場合、あるいは <paramref name="destination"/> が <paramref name="x"/> と重複する場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">Any padding value is negative. / パディングのいずれかが負の場合。</exception>
    public unsafe TensorSpan<T> Forward(ReadOnlyTensorSpan<T> x, Padding2D padding, TensorSpan<T> destination)
    {
        RequirePadding(padding);
        if (x.Lengths[0] != InChannels)
        {
            throw new ArgumentException($"The input must have {InChannels} channels (rows) but has {x.Lengths[0]}.", nameof(x));
        }

        int width = (int)x.Lengths[1], height = (int)x.Lengths[2];
        int outWidth = OutputWidth(width, padding), outHeight = OutputHeight(height, padding);
        var d = destination.AsReadOnly();
        if (d.Lengths[0] != OutChannels || d.Lengths[1] != outWidth || d.Lengths[2] != outHeight)
        {
            throw new ArgumentException(
                $"The destination must be [{OutChannels}, {outWidth}, {outHeight}] but is [{d.Lengths[0]}, {d.Lengths[1]}, {d.Lengths[2]}].",
                nameof(destination));
        }

        if (x.Data.Overlaps(d.Data))
        {
            throw new ArgumentException("The destination must not overlap the input.", nameof(destination));
        }

        ForwardCore(x, new ReadOnlyMatrixSpan<T>(weight, OutChannels, ColumnLength), new ReadOnlyVectorSpan<T>(bias),
            KernelShape, padding, destination);
        return destination;
    }

    /// <summary>
    /// Creates a causal streaming convolution that processes one height row at a time.
    /// <para>高さ方向に 1 行ずつ処理する因果的な 2 次元畳み込みストリームを作成します。</para>
    /// </summary>
    /// <param name="width">The width of each input row. / 入力の各行の幅。</param>
    /// <param name="left">The zero-padding on the left edge. / 左端のゼロパディング数。</param>
    /// <param name="right">The zero-padding on the right edge. / 右端のゼロパディング数。</param>
    /// <returns>A new <see cref="Convolution2DStream{T}"/> initialized with zero history. / 過去の入力履歴がすべて 0 で初期化された <see cref="Convolution2DStream{T}"/>。</returns>
    /// <exception cref="ArgumentException"><see cref="StrideHeight"/> is not 1. / <see cref="StrideHeight"/> が 1 でない場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="width"/>, <paramref name="left"/>, or <paramref name="right"/> is negative. / <paramref name="width"/>、<paramref name="left"/>、または <paramref name="right"/> が負の場合。</exception>
    public Convolution2DStream<T> CreateStream(int width, int left, int right) => new(this, width, left, right);

    private ConvolutionGeometry KernelShape => new(InChannels, KernelHeight, KernelWidth, StrideHeight, StrideWidth, DilationHeight, DilationWidth);

    internal static void RequirePadding(Padding2D padding)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(padding.Top, nameof(padding));
        ArgumentOutOfRangeException.ThrowIfNegative(padding.Bottom, nameof(padding));
        ArgumentOutOfRangeException.ThrowIfNegative(padding.Left, nameof(padding));
        ArgumentOutOfRangeException.ThrowIfNegative(padding.Right, nameof(padding));
    }

    internal unsafe void Row(T** rows, int* strides, int width, int left, int ow0, int ow1, T* cols, T* y, int ldy)
    {
        PackColumns(KernelShape, rows, strides, width, left, ow0, ow1, cols);
        EvaluateColumns(new ReadOnlyMatrixSpan<T>(weight, OutChannels, ColumnLength), new ReadOnlyVectorSpan<T>(bias),
            cols, ow1 - ow0, Views.Write(y, OutChannels, ow1 - ow0, ldy));
    }
}

/// <summary>
/// Performs causal 2D convolution incrementally, consuming one height row at a time.
/// <para>高さ方向に 1 行ずつ入力を受け取り、因果的な 2 次元畳み込みを逐次実行します。</para>
/// </summary>
/// <typeparam name="T">The floating-point element type (<see cref="float"/> or <see cref="double"/>). / 浮動小数点の要素型（<see cref="float"/> または <see cref="double"/>）。</typeparam>
/// <remarks>
/// <para>Passing row <c>t</c> to <see cref="Push(ReadOnlyMatrixSpan{T}, MatrixSpan{T})"/> produces output row <c>t</c>; rows before the first input are treated as 0. Requires a height stride of 1. This type is not thread-safe; do not call methods on the same instance concurrently.</para>
/// <para><see cref="Push(ReadOnlyMatrixSpan{T}, MatrixSpan{T})"/> に第 <c>t</c> 行を渡すと出力の第 <c>t</c> 行が得られ、最初の入力より前の行は 0 として扱われます。高さ方向のストライドは 1 である必要があります。この型はスレッドセーフではないため、同一インスタンスを複数スレッドから同時に操作しないでください。</para>
/// </remarks>
public sealed class Convolution2DStream<T> where T : unmanaged, IFloatingPointIeee754<T>
{
    private readonly int left, right, slots, slotLength;
    private readonly T[] ring; // slots 個の [InChannels × InputWidth] リングバッファ（ピン留め配列）
    private int head;          // 最新行のスロット番号

    /// <summary>
    /// Initializes a new <see cref="Convolution2DStream{T}"/> for the specified convolution and input row width.
    /// <para>指定した畳み込み層と入力行の幅から <see cref="Convolution2DStream{T}"/> を初期化します。</para>
    /// </summary>
    /// <param name="convolution">The 2D convolution layer (must have <see cref="Convolution2D{T}.StrideHeight"/> equal to 1). / 2 次元畳み込み層（<see cref="Convolution2D{T}.StrideHeight"/> が 1 であること）。</param>
    /// <param name="width">The width of each input row. / 入力の各行の幅。</param>
    /// <param name="left">The zero-padding on the left edge. / 幅方向先頭（左）のゼロパディング数。</param>
    /// <param name="right">The zero-padding on the right edge. / 幅方向末尾（右）のゼロパディング数。</param>
    /// <exception cref="ArgumentNullException"><paramref name="convolution"/> is <see langword="null"/>. / <paramref name="convolution"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="convolution"/> does not have a height stride of 1. / <paramref name="convolution"/> の高さ方向ストライドが 1 でない場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="width"/>, <paramref name="left"/>, or <paramref name="right"/> is negative. / <paramref name="width"/>、<paramref name="left"/>、または <paramref name="right"/> が負の場合。</exception>
    public Convolution2DStream(Convolution2D<T> convolution, int width, int left, int right)
    {
        ArgumentNullException.ThrowIfNull(convolution);
        if (convolution.StrideHeight != 1)
        {
            throw new ArgumentException("A stream needs a convolution with a height stride of 1.", nameof(convolution));
        }

        ArgumentOutOfRangeException.ThrowIfNegative(width);
        ArgumentOutOfRangeException.ThrowIfNegative(left);
        ArgumentOutOfRangeException.ThrowIfNegative(right);
        Convolution = convolution;
        InputWidth = width;
        this.left = left;
        this.right = right;
        OutputWidth = Convolution2D<T>.OutputLength(width, left, right, convolution.KernelWidth, convolution.StrideWidth, convolution.DilationWidth);
        slots = (convolution.KernelHeight - 1) * convolution.DilationHeight + 1;
        slotLength = checked(convolution.InChannels * width);
        ring = GC.AllocateArray<T>(checked(slots * slotLength), pinned: true);
    }

    /// <summary>
    /// Gets the underlying 2D convolution layer.
    /// <para>基になる 2 次元畳み込み層を取得します。</para>
    /// </summary>
    public Convolution2D<T> Convolution { get; }

    /// <summary>
    /// Gets the expected width of each input row.
    /// <para>入力の各行の幅を取得します。</para>
    /// </summary>
    public int InputWidth { get; }

    /// <summary>
    /// Gets the width of each output row.
    /// <para>出力の各行の幅を取得します。</para>
    /// </summary>
    public int OutputWidth { get; }

    /// <summary>
    /// Clears the buffered input history to zero, resetting the stream to its initial state.
    /// <para>保持している過去の入力行を 0 でクリアし、初期状態に戻します。</para>
    /// </summary>
    public void Reset()
    {
        Array.Clear(ring);
        head = 0;
    }

    /// <summary>
    /// Processes one input row and returns the corresponding output row as a new matrix.
    /// <para>入力の 1 行を処理し、対応する出力の 1 行を新しい行列として返します。</para>
    /// </summary>
    /// <param name="row">The input row of shape <c>[InChannels × InputWidth]</c>. / 入力行 <c>[InChannels × InputWidth]</c>。</param>
    /// <returns>A new matrix of shape <c>[OutChannels × OutputWidth]</c>. / 出力の 1 行を格納した新しい行列 <c>[OutChannels × OutputWidth]</c>。</returns>
    /// <exception cref="ArgumentException">The shape of <paramref name="row"/> does not match <c>[InChannels × InputWidth]</c>. / <paramref name="row"/> の形状が <c>[InChannels × InputWidth]</c> と一致しない場合。</exception>
    public Matrix<T> Push(ReadOnlyMatrixSpan<T> row)
    {
        var result = Matrix<T>.Uninitialized(Convolution.OutChannels, OutputWidth);
        Push(row, result);
        return result;
    }

    /// <summary>
    /// Processes one input row and writes the corresponding output row into <paramref name="destination"/>.
    /// <para>入力の 1 行を処理し、対応する出力の 1 行を <paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="row">The input row of shape <c>[InChannels × InputWidth]</c>. / 入力行 <c>[InChannels × InputWidth]</c>。</param>
    /// <param name="destination">The output row of shape <c>[OutChannels × OutputWidth]</c>. May overlap <paramref name="row"/>. / 出力行 <c>[OutChannels × OutputWidth]</c>。<paramref name="row"/> と重複していても構いません。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException">The shape of <paramref name="row"/> or <paramref name="destination"/> does not match. / <paramref name="row"/> または <paramref name="destination"/> の形状が一致しない場合。</exception>
    public unsafe MatrixSpan<T> Push(ReadOnlyMatrixSpan<T> row, MatrixSpan<T> destination)
    {
        var conv = Convolution;
        int channels = conv.InChannels, width = InputWidth, outWidth = OutputWidth;
        Guard.RequireShape(channels, width, row.Rows, row.Columns, nameof(row));
        Guard.RequireShape(conv.OutChannels, outWidth, destination.Rows, destination.Columns, nameof(destination));
        head = (head + 1) % slots;
        var slot = new MatrixSpan<T>(ring.AsSpan(head * slotLength, slotLength), channels, width);

        // 今回行は入力から直接読み、リングバッファへの退避は並列区間内で分散実行する。
        // 列内が非連続な入力や出力先と重複する入力は先にリングバッファへコピーしてそこから読む。
        var d = destination.AsReadOnly();
        bool fromInput = slotLength != 0 && (channels <= 1 || row.RowStride == 1)
            && !Guard.Overlaps(row, d, out _);
        if (!fromInput && slotLength != 0)
        {
            row.CopyTo(slot);
        }

        if (destination.IsEmpty)
        {
            if (fromInput)
            {
                row.CopyTo(slot);
            }

            return destination;
        }

        T[]? rented = null;
        var target = destination;
        bool direct = conv.OutChannels <= 1 || d.RowStride == 1;
        if (!direct)
        {
            rented = ArrayPool<T>.Shared.Rent(conv.OutChannels * outWidth);
            target = new MatrixSpan<T>(rented.AsSpan(0, conv.OutChannels * outWidth), conv.OutChannels, outWidth);
        }

        try
        {
            int ldy = target.AsReadOnly().ColumnStride, ldr = fromInput ? row.ColumnStride : channels;
            int threads = Split.Threads((double)conv.OutChannels * conv.ColumnLength * outWidth);

            // チャンクごとの GEMM で重みのパッキングが発生するため、1スレッドあたり最大1チャンクに抑える
            int chunks = Math.Max(1, Math.Min(threads, outWidth / 8));
            T* rs = (T*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(ring));
            fixed (T* py = &Dense.Ref(target), pr = &Dense.Ref(row))
            {
                T* ys = py, input = fromInput ? pr : rs + (nint)head * slotLength, current = rs + (nint)head * slotLength;
                int newest = head;
                Split.For(outWidth, chunks, threads, (w0, w1) =>
                {
                    T** rows = stackalloc T*[conv.KernelHeight];
                    int* strides = stackalloc int[conv.KernelHeight];
                    RingRows(rows, rs, newest);
                    new Span<int>(strides, conv.KernelHeight).Fill(channels);
                    rows[conv.KernelHeight - 1] = input;
                    strides[conv.KernelHeight - 1] = ldr;
                    int block = Math.Max(1, Math.Min(w1 - w0, Convolution2D<T>.ColumnBufferElements / conv.ColumnLength));
                    var buffer = ArrayPool<T>.Shared.Rent(conv.ColumnLength * block);
                    try
                    {
                        fixed (T* cols = buffer)
                        {
                            for (int w = w0; w < w1; w += block)
                            {
                                int end = Math.Min(w1, w + block);
                                conv.Row(rows, strides, width, left, w, end, cols, ys + (nint)w * ldy, ldy);
                            }
                        }
                    }
                    finally
                    {
                        ArrayPool<T>.Shared.Return(buffer);
                    }

                    // 入力列 [c0, c1) をリングバッファへ並列退避する（KernelHeight == 1 では過去行を参照しない）
                    if (fromInput && conv.KernelHeight > 1)
                    {
                        int c0 = w0 == 0 ? 0 : (int)((long)width * w0 / outWidth), c1 = w1 == outWidth ? width : (int)((long)width * w1 / outWidth);
                        nint bytes = (nint)channels * sizeof(T);
                        for (int c = c0; c < c1; c++)
                        {
                            Buffer.MemoryCopy(input + (nint)c * ldr, current + (nint)c * channels, bytes, bytes);
                        }
                    }
                });
            }

            if (!direct)
            {
                target.CopyTo(destination);
            }
        }
        finally
        {
            Dense.Return(rented);
        }

        return destination;
    }

    // カーネル各行に対応するリングバッファ上の過去行ポインタを埋める（最新スロットが current）
    private unsafe void RingRows(T** rows, T* ring, int current)
    {
        var conv = Convolution;
        for (int i = 0; i < conv.KernelHeight; i++)
        {
            int back = (conv.KernelHeight - 1 - i) * conv.DilationHeight;
            rows[i] = ring + (nint)((current - back + slots) % slots) * slotLength;
        }
    }
}
