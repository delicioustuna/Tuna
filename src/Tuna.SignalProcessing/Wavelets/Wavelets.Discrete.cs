using System.Buffers;
using System.Runtime.InteropServices;

namespace Tuna.SignalProcessing;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む。
using Tuna.LinearAlgebra;

/// <summary>
/// Provides discrete, stationary, packet, and continuous wavelet transforms, signal reconstruction, and wavelet denoising.
/// <para>離散・定常・パケット・連続ウェーブレット変換、信号の再構成、およびウェーブレットノイズ除去を提供します。</para>
/// </summary>
public static partial class Wavelets
{
    /// <summary>
    /// Computes the length of the approximation or detail coefficient vector produced by a single level of the discrete wavelet transform.
    /// <para>離散ウェーブレット変換の 1 段で生成される近似係数または詳細係数の長さを計算します。</para>
    /// </summary>
    /// <param name="length">The input signal length n, which must be at least 1. / 1 以上の入力信号の長さ n。</param>
    /// <param name="wavelet">The discrete wavelet to use. / 使用する離散ウェーブレット。</param>
    /// <param name="boundary">The boundary extension mode. / 境界拡張モード。</param>
    /// <returns>
    /// <c>ceil(n / 2)</c> for <see cref="WaveletBoundary.Periodization"/>; otherwise <c>floor((n + F - 1) / 2)</c>, where F is the filter length. /
    /// <see cref="WaveletBoundary.Periodization"/> の場合は <c>ceil(n / 2)</c>、それ以外の場合は <c>floor((n + F - 1) / 2)</c>（F はフィルタ長）。
    /// </returns>
    /// <exception cref="ArgumentNullException"><paramref name="wavelet"/> is <see langword="null"/>. / <paramref name="wavelet"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="length"/> is less than 1, or <paramref name="boundary"/> is not a defined value. / <paramref name="length"/> が 1 未満、または <paramref name="boundary"/> が定義外の値の場合。</exception>
    public static int CoefficientLength(int length, DiscreteWavelet wavelet, WaveletBoundary boundary)
    {
        ArgumentNullException.ThrowIfNull(wavelet);
        ArgumentOutOfRangeException.ThrowIfLessThan(length, 1);
        RequireBoundary(boundary);
        return boundary == WaveletBoundary.Periodization ? (length + 1) / 2 : (int)(((long)length + wavelet.FilterLength - 1) / 2);
    }

    /// <summary>
    /// Computes the signal length produced by a single level of the inverse discrete wavelet transform.
    /// <para>離散ウェーブレット逆変換の 1 段で生成される信号の長さを計算します。</para>
    /// </summary>
    /// <param name="coefficientLength">The length m of each approximation and detail vector, which must be at least 1. / 1 以上の近似係数・詳細係数の長さ m。</param>
    /// <param name="wavelet">The discrete wavelet to use. / 使用する離散ウェーブレット。</param>
    /// <param name="boundary">The boundary extension mode. / 境界拡張モード。</param>
    /// <returns>
    /// <c>2m</c> for <see cref="WaveletBoundary.Periodization"/>; otherwise <c>2m - F + 2</c>, where F is the filter length. When the original length was odd, this is 1 sample longer than the original signal. /
    /// <see cref="WaveletBoundary.Periodization"/> の場合は <c>2m</c>、それ以外の場合は <c>2m - F + 2</c>（F はフィルタ長）。元の長さが奇数の場合は元の信号より 1 要素長くなります。
    /// </returns>
    /// <exception cref="ArgumentException"><paramref name="coefficientLength"/> is too small for the wavelet filter length. / フィルタ長に対して係数の長さが不足している場合。</exception>
    /// <exception cref="OverflowException">The resulting length exceeds <see cref="int.MaxValue"/>. / 逆変換後の長さが <see cref="int.MaxValue"/> を超える場合。</exception>
    /// <exception cref="ArgumentNullException"><paramref name="wavelet"/> is <see langword="null"/>. / <paramref name="wavelet"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="coefficientLength"/> is less than 1, or <paramref name="boundary"/> is not a defined value. / <paramref name="coefficientLength"/> が 1 未満、または <paramref name="boundary"/> が定義外の値の場合。</exception>
    public static int InverseLength(int coefficientLength, DiscreteWavelet wavelet, WaveletBoundary boundary)
    {
        ArgumentNullException.ThrowIfNull(wavelet);
        ArgumentOutOfRangeException.ThrowIfLessThan(coefficientLength, 1);
        RequireBoundary(boundary);
        long length = boundary == WaveletBoundary.Periodization
            ? 2L * coefficientLength
            : 2L * coefficientLength - wavelet.FilterLength + 2;
        if (length < 1)
        {
            throw new ArgumentException(
                $"{coefficientLength} coefficients are too few for a filter of length {wavelet.FilterLength}.", nameof(coefficientLength));
        }

        return checked((int)length);
    }

    /// <summary>
    /// Computes the maximum useful decomposition level at which at least one coefficient remains free of boundary effects.
    /// <para>境界の影響を受けない係数が残る最大分解段数を計算します。</para>
    /// </summary>
    /// <param name="length">The input signal length n, which must be non-negative. / 0 以上の入力信号の長さ n。</param>
    /// <param name="wavelet">The discrete wavelet to use. / 使用する離散ウェーブレット。</param>
    /// <returns><c>floor(log2(n / (F - 1)))</c> for signal length n and filter length F, or 0 if <c>n &lt; F - 1</c>. / 信号長 n とフィルタ長 F に対する <c>floor(log2(n / (F - 1)))</c>（<c>n &lt; F - 1</c> の場合は 0）。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="wavelet"/> is <see langword="null"/>. / <paramref name="wavelet"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="length"/> is negative. / <paramref name="length"/> が負の場合。</exception>
    public static int MaxLevel(int length, DiscreteWavelet wavelet)
    {
        ArgumentNullException.ThrowIfNull(wavelet);
        ArgumentOutOfRangeException.ThrowIfNegative(length);
        long step = wavelet.FilterLength - 1;
        int level = 0;
        while (step * 2 <= length)
        {
            step *= 2;
            level++;
        }

        return step <= length ? level : 0;
    }

    /// <summary>
    /// Computes a single-level discrete wavelet transform using symmetric boundary extension.
    /// <para>対称境界拡張を用いて 1 段の離散ウェーブレット変換を計算します。</para>
    /// </summary>
    /// <param name="x">The input signal. / 入力信号。</param>
    /// <param name="wavelet">The discrete wavelet to use. / 使用する離散ウェーブレット。</param>
    /// <returns>
    /// A tuple of newly allocated approximation and detail coefficient vectors, each of length <see cref="CoefficientLength"/>. /
    /// <see cref="CoefficientLength"/> の長さを持つ近似係数ベクトルと詳細係数ベクトルのタプル。
    /// </returns>
    /// <remarks>
    /// <para>
    /// Uses <see cref="WaveletBoundary.Symmetric"/> at the signal edges.
    /// </para>
    /// <para>
    /// 境界条件には <see cref="WaveletBoundary.Symmetric"/> を使用します。
    /// </para>
    /// </remarks>
    public static (Vector<double> Approximation, Vector<double> Detail) Forward(ReadOnlyVectorSpan<double> x, DiscreteWavelet wavelet) =>
        Forward(x, wavelet, WaveletBoundary.Symmetric);

    /// <summary>
    /// Computes a single-level discrete wavelet transform using the specified boundary extension.
    /// <para>指定した境界拡張モードを用いて 1 段の離散ウェーブレット変換を計算します。</para>
    /// </summary>
    /// <param name="x">The input signal. / 入力信号。</param>
    /// <param name="wavelet">The discrete wavelet to use. / 使用する離散ウェーブレット。</param>
    /// <param name="boundary">The boundary extension mode applied at the signal edges. / 信号端に適用する境界拡張モード。</param>
    /// <returns>
    /// A tuple of newly allocated approximation and detail coefficient vectors, each of length <see cref="CoefficientLength"/>. /
    /// <see cref="CoefficientLength"/> の長さを持つ近似係数ベクトルと詳細係数ベクトルのタプル。
    /// </returns>
    public static (Vector<double> Approximation, Vector<double> Detail) Forward(
        ReadOnlyVectorSpan<double> x, DiscreteWavelet wavelet, WaveletBoundary boundary)
    {
        int length = CoefficientLength(RequireSignal(x, boundary), wavelet, boundary);
        var approximation = Vector<double>.Uninitialized(length);
        var detail = Vector<double>.Uninitialized(length);
        ForwardCore(x, wavelet, boundary, approximation.AsSpan(), detail.AsSpan());
        return (approximation, detail);
    }

    /// <summary>
    /// Computes a single-level discrete wavelet transform and writes the coefficients into the specified destination spans.
    /// <para>1 段の離散ウェーブレット変換を計算し、指定した出力先スパンへ係数を書き込みます。</para>
    /// </summary>
    /// <param name="x">The input signal. / 入力信号。</param>
    /// <param name="wavelet">The discrete wavelet to use. / 使用する離散ウェーブレット。</param>
    /// <param name="boundary">The boundary extension mode applied at the signal edges. / 信号端に適用する境界拡張モード。</param>
    /// <param name="approximation">The destination span for approximation coefficients, of length <see cref="CoefficientLength"/>. May alias <paramref name="x"/>. / 長さ <see cref="CoefficientLength"/> の近似係数の出力先。<paramref name="x"/> と重なっても構いません。</param>
    /// <param name="detail">The destination span for detail coefficients, of length <see cref="CoefficientLength"/>. Must not overlap <paramref name="approximation"/>. / 長さ <see cref="CoefficientLength"/> の詳細係数の出力先。<paramref name="approximation"/> と重なってはいけません。</param>
    public static void Forward(
        ReadOnlyVectorSpan<double> x, DiscreteWavelet wavelet, WaveletBoundary boundary, VectorSpan<double> approximation, VectorSpan<double> detail)
    {
        int length = CoefficientLength(RequireSignal(x, boundary), wavelet, boundary);
        RequireLength(length, approximation.Length, nameof(approximation));
        RequireLength(length, detail.Length, nameof(detail));
        RequireDisjoint(approximation, detail);
        double[] buffer = ArrayPool<double>.Shared.Rent(2 * length);
        var a = buffer.AsSpan(0, length);
        var d = buffer.AsSpan(length, length);
        ForwardCore(x, wavelet, boundary, a, d);
        Copy(a, approximation);
        Copy(d, detail);
        ArrayPool<double>.Shared.Return(buffer);
    }

    /// <summary>
    /// Computes a single-level inverse discrete wavelet transform using symmetric boundary extension.
    /// <para>対称境界拡張を用いて 1 段の離散ウェーブレット逆変換を計算します。</para>
    /// </summary>
    /// <param name="approximation">The approximation coefficients. / 近似係数。</param>
    /// <param name="detail">The detail coefficients. / 詳細係数。</param>
    /// <param name="wavelet">The discrete wavelet to use. / 使用する離散ウェーブレット。</param>
    /// <returns>A newly allocated vector containing the reconstructed signal of length <see cref="InverseLength"/>. / <see cref="InverseLength"/> の長さを持つ再構成信号のベクトル。</returns>
    /// <remarks>
    /// <para>
    /// Uses <see cref="WaveletBoundary.Symmetric"/> at the signal edges.
    /// </para>
    /// <para>
    /// 境界条件には <see cref="WaveletBoundary.Symmetric"/> を使用します。
    /// </para>
    /// </remarks>
    public static Vector<double> Inverse(ReadOnlyVectorSpan<double> approximation, ReadOnlyVectorSpan<double> detail, DiscreteWavelet wavelet) =>
        Inverse(approximation, detail, wavelet, WaveletBoundary.Symmetric);

    /// <summary>
    /// Computes a single-level inverse discrete wavelet transform using the specified boundary extension.
    /// <para>指定した境界拡張モードを用いて 1 段の離散ウェーブレット逆変換を計算します。</para>
    /// </summary>
    /// <param name="approximation">The approximation coefficients. / 近似係数。</param>
    /// <param name="detail">The detail coefficients. / 詳細係数。</param>
    /// <param name="wavelet">The discrete wavelet to use. / 使用する離散ウェーブレット。</param>
    /// <param name="boundary">The boundary extension mode applied at the signal edges. / 信号端に適用する境界拡張モード。</param>
    /// <returns>A newly allocated vector containing the reconstructed signal of length <see cref="InverseLength"/>. / <see cref="InverseLength"/> の長さを持つ再構成信号のベクトル。</returns>
    public static Vector<double> Inverse(
        ReadOnlyVectorSpan<double> approximation, ReadOnlyVectorSpan<double> detail, DiscreteWavelet wavelet, WaveletBoundary boundary)
    {
        var result = Vector<double>.Uninitialized(InverseLength(RequireCoefficients(approximation, detail), wavelet, boundary));
        InverseCore(approximation, detail, wavelet, boundary, result.AsSpan());
        return result;
    }

    /// <summary>
    /// Computes a single-level inverse discrete wavelet transform and writes the reconstructed signal into <paramref name="destination"/>.
    /// <para>1 段の離散ウェーブレット逆変換を計算し、再構成した信号を <paramref name="destination"/> へ書き込みます。</para>
    /// </summary>
    /// <param name="approximation">The approximation coefficients. / 近似係数。</param>
    /// <param name="detail">The detail coefficients. / 詳細係数。</param>
    /// <param name="wavelet">The discrete wavelet to use. / 使用する離散ウェーブレット。</param>
    /// <param name="boundary">The boundary extension mode applied at the signal edges. / 信号端に適用する境界拡張モード。</param>
    /// <param name="destination">The destination span of length <see cref="InverseLength"/>. May alias <paramref name="approximation"/> or <paramref name="detail"/>. / 長さ <see cref="InverseLength"/> の出力先スパン。入力係数と重なっても構いません。</param>
    /// <returns><paramref name="destination"/> containing the reconstructed signal. / 再構成した信号を書き込んだ <paramref name="destination"/>。</returns>
    public static VectorSpan<double> Inverse(
        ReadOnlyVectorSpan<double> approximation, ReadOnlyVectorSpan<double> detail, DiscreteWavelet wavelet, WaveletBoundary boundary,
        VectorSpan<double> destination)
    {
        int length = InverseLength(RequireCoefficients(approximation, detail), wavelet, boundary);
        RequireLength(length, destination.Length, nameof(destination));
        double[] buffer = ArrayPool<double>.Shared.Rent(length);
        InverseCore(approximation, detail, wavelet, boundary, buffer.AsSpan(0, length));
        Copy(buffer.AsSpan(0, length), destination);
        ArrayPool<double>.Shared.Return(buffer);
        return destination;
    }

    /// <summary>
    /// Computes a multi-level discrete wavelet decomposition using the maximum useful level and symmetric boundary extension.
    /// <para>最大有効段数と対称境界拡張を用いて、入力信号の多段離散ウェーブレット分解を計算します。</para>
    /// </summary>
    /// <param name="x">The input signal to decompose. / 分解対象の入力信号。</param>
    /// <param name="wavelet">The discrete wavelet to use. / 使用する離散ウェーブレット。</param>
    /// <returns>A decomposition containing the coarsest approximation and detail vectors ordered from coarsest to finest level. / 最も粗い近似係数と、粗い段から細かい段へ並べた詳細係数を保持する分解結果。</returns>
    /// <remarks>
    /// <para>
    /// The decomposition depth is determined by <see cref="MaxLevel(int, DiscreteWavelet)"/> and uses <see cref="WaveletBoundary.Symmetric"/>.
    /// </para>
    /// <para>
    /// 分解段数は <see cref="MaxLevel(int, DiscreteWavelet)"/> で決定され、境界条件には <see cref="WaveletBoundary.Symmetric"/> を使用します。
    /// </para>
    /// </remarks>
    public static WaveletDecomposition Decompose(ReadOnlyVectorSpan<double> x, DiscreteWavelet wavelet) =>
        Decompose(x, wavelet, WaveletBoundary.Symmetric);

    /// <summary>
    /// Computes a multi-level discrete wavelet decomposition using the maximum useful level and the specified boundary extension.
    /// <para>最大有効段数と指定した境界拡張モードを用いて、入力信号の多段離散ウェーブレット分解を計算します。</para>
    /// </summary>
    /// <param name="x">The input signal to decompose. / 分解対象の入力信号。</param>
    /// <param name="wavelet">The discrete wavelet to use. / 使用する離散ウェーブレット。</param>
    /// <param name="boundary">The boundary extension mode applied at the signal edges. / 信号端に適用する境界拡張モード。</param>
    /// <returns>A decomposition containing the coarsest approximation and detail vectors ordered from coarsest to finest level. / 最も粗い近似係数と、粗い段から細かい段へ並べた詳細係数を保持する分解結果。</returns>
    /// <remarks>
    /// <para>
    /// The decomposition depth is determined by <see cref="MaxLevel(int, DiscreteWavelet)"/>.
    /// </para>
    /// <para>
    /// 分解段数は <see cref="MaxLevel(int, DiscreteWavelet)"/> で決定されます。
    /// </para>
    /// </remarks>
    public static WaveletDecomposition Decompose(ReadOnlyVectorSpan<double> x, DiscreteWavelet wavelet, WaveletBoundary boundary)
    {
        ArgumentNullException.ThrowIfNull(wavelet);
        return Decompose(x, wavelet, MaxLevel(x.Length, wavelet), boundary);
    }

    /// <summary>
    /// Computes a multi-level discrete wavelet decomposition to the specified number of levels using symmetric boundary extension.
    /// <para>指定した段数と対称境界拡張を用いて、入力信号の多段離散ウェーブレット分解を計算します。</para>
    /// </summary>
    /// <param name="x">The input signal to decompose. / 分解対象の入力信号。</param>
    /// <param name="wavelet">The discrete wavelet to use. / 使用する離散ウェーブレット。</param>
    /// <param name="levels">The number of decomposition levels, which must be non-negative. / 0 以上の分解段数。</param>
    /// <returns>A decomposition containing the coarsest approximation and detail vectors ordered from coarsest to finest level. / 最も粗い近似係数と、粗い段から細かい段へ並べた詳細係数を保持する分解結果。</returns>
    /// <remarks>
    /// <para>
    /// Uses <see cref="WaveletBoundary.Symmetric"/> at the signal edges.
    /// </para>
    /// <para>
    /// 境界条件には <see cref="WaveletBoundary.Symmetric"/> を使用します。
    /// </para>
    /// </remarks>
    public static WaveletDecomposition Decompose(ReadOnlyVectorSpan<double> x, DiscreteWavelet wavelet, int levels) =>
        Decompose(x, wavelet, levels, WaveletBoundary.Symmetric);

    /// <summary>
    /// Computes a multi-level discrete wavelet decomposition to the specified number of levels and boundary extension.
    /// <para>指定した段数および境界拡張モードで、入力信号の多段離散ウェーブレット分解を計算します。</para>
    /// </summary>
    /// <param name="x">The input signal to decompose. / 分解対象の入力信号。</param>
    /// <param name="wavelet">The discrete wavelet to use. / 使用する離散ウェーブレット。</param>
    /// <param name="levels">The number of decomposition levels, which must be non-negative. / 0 以上の分解段数。</param>
    /// <param name="boundary">The boundary extension mode applied at the signal edges. / 信号端に適用する境界拡張モード。</param>
    /// <returns>A decomposition containing the coarsest approximation and detail vectors ordered from coarsest to finest level. / 最も粗い近似係数と、粗い段から細かい段へ並べた詳細係数を保持する分解結果。</returns>
    /// <remarks>
    /// <para>
    /// When <paramref name="levels"/> exceeds <see cref="MaxLevel(int, DiscreteWavelet)"/>, all coefficients at the coarsest levels are affected by boundary extension.
    /// When <paramref name="levels"/> is 0, a copy of the original signal is returned as the approximation with no detail levels.
    /// </para>
    /// <para>
    /// <paramref name="levels"/> が <see cref="MaxLevel(int, DiscreteWavelet)"/> を超える場合、粗い段の係数はすべて境界拡張の影響を受けます。
    /// <paramref name="levels"/> が 0 の場合は、元の信号のコピーを近似係数とし、詳細係数は空になります。
    /// </para>
    /// </remarks>
    public static WaveletDecomposition Decompose(ReadOnlyVectorSpan<double> x, DiscreteWavelet wavelet, int levels, WaveletBoundary boundary)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(levels);
        int length = RequireSignal(x, boundary);
        ArgumentNullException.ThrowIfNull(wavelet);
        var lengths = new int[levels + 1];
        lengths[0] = length;
        var details = new Vector<double>[levels];
        var approximation = x.ToVector();
        for (int level = 1; level <= levels; level++)
        {
            if (boundary == WaveletBoundary.Reflect && approximation.Length < 2)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(levels), $"Level {level} has a single approximation coefficient, which the reflect boundary cannot extend.");
            }

            var (a, d) = Forward(approximation, wavelet, boundary);
            lengths[level] = a.Length;
            details[levels - level] = d;
            approximation = a;
        }

        return new WaveletDecomposition(approximation, details, lengths, wavelet, boundary);
    }

    /// <summary>
    /// Reconstructs a signal from its multi-level discrete wavelet decomposition.
    /// <para>多段離散ウェーブレット分解から元の信号を再構成します。</para>
    /// </summary>
    /// <param name="decomposition">The wavelet decomposition containing approximation and detail coefficients. / 近似係数と詳細係数を保持するウェーブレット分解。</param>
    /// <returns>A newly allocated vector containing the reconstructed signal with the original signal length. / 元の信号と同じ長さの再構成信号を格納した新しいベクトル。</returns>
    /// <remarks>
    /// <para>
    /// When the original signal or any intermediate approximation had an odd length, the trailing extra sample produced by single-level inversion is trimmed at each level so the exact original length is restored.
    /// </para>
    /// <para>
    /// 元の信号や途中の近似係数の長さが奇数だった場合も、各段の逆変換で生じる末尾の余剰要素を切り詰めて元の長さに正確に復元します。
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="decomposition"/> is <see langword="null"/>. / <paramref name="decomposition"/> が <see langword="null"/> の場合。</exception>
    public static Vector<double> Reconstruct(WaveletDecomposition decomposition)
    {
        ArgumentNullException.ThrowIfNull(decomposition);
        var wavelet = decomposition.Wavelet;
        var boundary = decomposition.Boundary;
        int levels = decomposition.Levels;
        Vector<double> approximation = decomposition.Approximation.Clone();
        for (int i = 0; i < levels; i++)
        {
            var detail = decomposition.Details[i];
            int target = decomposition.ApproximationLength(levels - 1 - i);
            var full = Inverse(approximation, detail, wavelet, boundary);
            approximation = full.Length == target ? full : ((ReadOnlyVectorSpan<double>)full)[..target].ToVector();
        }

        return approximation;
    }

    // L1-cache tile size for inner multiply-accumulate sweeps.
    private const int Block = 1024;

    private static void ForwardCore(
        ReadOnlyVectorSpan<double> x, DiscreteWavelet wavelet, WaveletBoundary boundary, Span<double> approximation, Span<double> detail)
    {
        int n = x.Length;
        int f = wavelet.FilterLength;
        int m = approximation.Length;
        var low = wavelet.DecompositionLow;
        var high = wavelet.DecompositionHigh;

        // Split the extended signal into even and odd phases so each filter tap j accumulates contiguously over k.
        int offset = boundary == WaveletBoundary.Periodization ? f / 2 - 1 : 0;
        int first = 2 - f + offset;
        int half = m + f / 2 - 1;
        double[] buffer = ArrayPool<double>.Shared.Rent(2 * half);
        var even = buffer.AsSpan(0, half);
        var odd = buffer.AsSpan(half, half);
        for (int i = 0; i < half; i++)
        {
            even[i] = ExtendedValue(x, first + 2 * i, boundary);
            odd[i] = ExtendedValue(x, first + 2 * i + 1, boundary);
        }

        approximation.Clear();
        detail.Clear();
        for (int k0 = 0; k0 < m; k0 += Block)
        {
            int count = Math.Min(Block, m - k0);
            var a = approximation.Slice(k0, count);
            var d = detail.Slice(k0, count);
            for (int j = 0; j < f; j++)
            {
                int p = f - 1 - j;
                var source = ((p & 1) == 0 ? even : odd).Slice(k0 + (p >> 1), count);
                MultiplyAdd(source, low[j], high[j], a, d);
            }
        }

        ArrayPool<double>.Shared.Return(buffer);
    }

    private static void InverseCore(
        ReadOnlyVectorSpan<double> approximation, ReadOnlyVectorSpan<double> detail, DiscreteWavelet wavelet, WaveletBoundary boundary,
        Span<double> destination)
    {
        int m = approximation.Length;
        int f = wavelet.FilterLength;
        var low = wavelet.ReconstructionLow;
        var high = wavelet.ReconstructionHigh;

        // Copy into contiguous buffers (supporting aliasing and periodic wrap-around) and compute even/odd outputs separately.
        bool periodization = boundary == WaveletBoundary.Periodization;
        int pad = periodization ? f / 2 : 0;
        int padded = m + 2 * pad;
        int length = destination.Length;
        int outputs = (length + 1) / 2;
        double[] buffer = ArrayPool<double>.Shared.Rent(2 * padded + 2 * outputs);
        var a = buffer.AsSpan(0, padded);
        var d = buffer.AsSpan(padded, padded);
        var evenOutput = buffer.AsSpan(2 * padded, outputs);
        var oddOutput = buffer.AsSpan(2 * padded + outputs, outputs);
        for (int i = 0; i < padded; i++)
        {
            int k = periodization ? Modulo(i - pad, m) : i;
            a[i] = approximation[k];
            d[i] = detail[k];
        }

        evenOutput.Clear();
        oddOutput.Clear();
        for (int r = 0; r < 2; r++)
        {
            var output = r == 0 ? evenOutput : oddOutput;
            for (int q = 0; q < f / 2; q++)
            {
                int j, start;
                if (periodization)
                {
                    int parity = (r + f / 2 - 1) & 1;
                    j = parity + 2 * q;
                    start = pad + (r + f / 2 - 1 - parity) / 2 - q;
                }
                else
                {
                    j = f - 2 + r - 2 * q;
                    start = q;
                }

                for (int i0 = 0; i0 < outputs; i0 += Block)
                {
                    int count = Math.Min(Block, outputs - i0);
                    AddProducts(output.Slice(i0, count), a.Slice(start + i0, count), low[j], d.Slice(start + i0, count), high[j]);
                }
            }
        }

        for (int t = 0; t < length; t++)
        {
            destination[t] = (t & 1) == 0 ? evenOutput[t >> 1] : oddOutput[t >> 1];
        }

        ArrayPool<double>.Shared.Return(buffer);
    }

    // Separate multiply and add rounding guarantees bit-identical results across SIMD widths and scalar tail.
    private static void MultiplyAdd(ReadOnlySpan<double> source, double low, double high, Span<double> a, Span<double> d)
    {
        int i = 0;
        int width = System.Numerics.Vector<double>.Count;
        if (System.Numerics.Vector.IsHardwareAccelerated && source.Length >= width)
        {
            var vl = new System.Numerics.Vector<double>(low);
            var vh = new System.Numerics.Vector<double>(high);
            ref double s = ref MemoryMarshal.GetReference(source);
            ref double ra = ref MemoryMarshal.GetReference(a);
            ref double rd = ref MemoryMarshal.GetReference(d);
            for (; i <= source.Length - width; i += width)
            {
                var v = System.Numerics.Vector.LoadUnsafe(ref s, (nuint)i);
                System.Numerics.Vector.StoreUnsafe(System.Numerics.Vector.LoadUnsafe(ref ra, (nuint)i) + vl * v, ref ra, (nuint)i);
                System.Numerics.Vector.StoreUnsafe(System.Numerics.Vector.LoadUnsafe(ref rd, (nuint)i) + vh * v, ref rd, (nuint)i);
            }
        }

        for (; i < source.Length; i++)
        {
            a[i] += low * source[i];
            d[i] += high * source[i];
        }
    }

    // Separate multiply and add rounding guarantees bit-identical results across SIMD widths and scalar tail.
    private static void AddProducts(Span<double> output, ReadOnlySpan<double> a, double low, ReadOnlySpan<double> d, double high)
    {
        int i = 0;
        int width = System.Numerics.Vector<double>.Count;
        if (System.Numerics.Vector.IsHardwareAccelerated && output.Length >= width)
        {
            var vl = new System.Numerics.Vector<double>(low);
            var vh = new System.Numerics.Vector<double>(high);
            ref double ro = ref MemoryMarshal.GetReference(output);
            ref double ra = ref MemoryMarshal.GetReference(a);
            ref double rd = ref MemoryMarshal.GetReference(d);
            for (; i <= output.Length - width; i += width)
            {
                var sum = System.Numerics.Vector.LoadUnsafe(ref ra, (nuint)i) * vl + System.Numerics.Vector.LoadUnsafe(ref rd, (nuint)i) * vh;
                System.Numerics.Vector.StoreUnsafe(System.Numerics.Vector.LoadUnsafe(ref ro, (nuint)i) + sum, ref ro, (nuint)i);
            }
        }

        for (; i < output.Length; i++)
        {
            output[i] += a[i] * low + d[i] * high;
        }
    }

    private static double ExtendedValue(ReadOnlyVectorSpan<double> x, int position, WaveletBoundary boundary)
    {
        int n = x.Length;
        if ((uint)position < (uint)n)
        {
            return x[position];
        }

        int index = boundary == WaveletBoundary.Periodization ? PeriodizedIndex(position, n) : ExtendedIndex(position, n, boundary);
        return index < 0 ? 0.0 : x[index];
    }

    // Returns -1 for out-of-bounds zero-padded samples.
    private static int ExtendedIndex(int i, int n, WaveletBoundary boundary)
    {
        if ((uint)i < (uint)n)
        {
            return i;
        }

        switch (boundary)
        {
            case WaveletBoundary.Zero:
                return -1;
            case WaveletBoundary.Constant:
                return i < 0 ? 0 : n - 1;
            case WaveletBoundary.Periodic:
                return Modulo(i, n);
            case WaveletBoundary.Symmetric:
                {
                    int r = Modulo(i, 2 * n);
                    return r < n ? r : 2 * n - 1 - r;
                }

            case WaveletBoundary.Reflect:
                {
                    int r = Modulo(i, 2 * n - 2);
                    return r < n ? r : 2 * n - 2 - r;
                }

            default:
                throw new ArgumentOutOfRangeException(nameof(boundary));
        }
    }

    // Odd lengths duplicate the trailing sample to wrap with even period n + 1.
    private static int PeriodizedIndex(int i, int n)
    {
        int r = Modulo(i, n + (n & 1));
        return Math.Min(r, n - 1);
    }

    private static int Modulo(int value, int modulus)
    {
        int r = value % modulus;
        return r < 0 ? r + modulus : r;
    }

    private static int RequireSignal(ReadOnlyVectorSpan<double> x, WaveletBoundary boundary)
    {
        RequireBoundary(boundary);
        if (x.Length == 0)
        {
            throw new ArgumentException("The signal must not be empty.", nameof(x));
        }

        if (boundary == WaveletBoundary.Reflect && x.Length < 2)
        {
            throw new ArgumentException("The reflect boundary needs at least 2 samples.", nameof(x));
        }

        return x.Length;
    }

    private static int RequireCoefficients(ReadOnlyVectorSpan<double> approximation, ReadOnlyVectorSpan<double> detail)
    {
        if (approximation.Length != detail.Length)
        {
            throw new ArgumentException(
                $"The approximation ({approximation.Length}) and detail ({detail.Length}) coefficients must have the same length.", nameof(detail));
        }

        if (approximation.Length == 0)
        {
            throw new ArgumentException("The coefficients must not be empty.", nameof(approximation));
        }

        return approximation.Length;
    }

    private static void RequireBoundary(WaveletBoundary boundary)
    {
        if ((uint)boundary > (uint)WaveletBoundary.Periodic)
        {
            throw new ArgumentOutOfRangeException(nameof(boundary));
        }
    }

    private static void RequireLength(int expected, int actual, string name)
    {
        if (expected != actual)
        {
            throw new ArgumentException($"The destination must have {expected} elements, but it has {actual}.", name);
        }
    }

    private static void RequireDisjoint(VectorSpan<double> first, VectorSpan<double> second)
    {
        var a = first.AsReadOnly();
        var b = second.AsReadOnly();
        if (StridedLayout.Overlaps(in a.Reference, a.Length, 1, a.Stride, 0, in b.Reference, b.Length, 1, b.Stride, 0, out _))
        {
            throw new ArgumentException("The approximation and detail destinations must not overlap.", "detail");
        }
    }

    private static void Copy(ReadOnlySpan<double> source, VectorSpan<double> destination)
    {
        for (int i = 0; i < source.Length; i++)
        {
            destination[i] = source[i];
        }
    }
}
