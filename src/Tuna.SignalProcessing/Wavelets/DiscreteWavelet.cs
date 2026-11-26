using System.Globalization;

namespace Tuna.SignalProcessing;

/// <summary>
/// Represents a discrete wavelet filter bank consisting of decomposition and reconstruction low-pass and high-pass filters.
/// <para>分解および再構成の低域・高域フィルタ組から成る離散ウェーブレット基底を表します。</para>
/// </summary>
/// <remarks>
/// <para>High-pass filters are derived from the low-pass filters as <c>g[k] = (-1)^(k + 1) · h̃[k]</c> (decomposition) and <c>g̃[k] = (-1)^k · h[k]</c> (reconstruction). For orthogonal wavelets, <see cref="DecompositionLow"/> is the reverse of <see cref="ReconstructionLow"/>.</para>
/// <para>高域フィルタは低域フィルタから分解高域 <c>g[k] = (−1)^(k+1) · h̃[k]</c>、再構成高域 <c>g̃[k] = (−1)^k · h[k]</c> として導出されます。直交ウェーブレットでは <see cref="DecompositionLow"/> は <see cref="ReconstructionLow"/> の逆順です。</para>
/// </remarks>
public sealed class DiscreteWavelet
{
    private readonly double[] decompositionLow;
    private readonly double[] decompositionHigh;
    private readonly double[] reconstructionLow;
    private readonly double[] reconstructionHigh;

    private DiscreteWavelet(string name, double[] decompositionLow, double[] reconstructionLow, bool isOrthogonal, int vanishingMoments)
    {
        Name = name;
        IsOrthogonal = isOrthogonal;
        VanishingMoments = vanishingMoments;
        int length = decompositionLow.Length;
        this.decompositionLow = decompositionLow;
        this.reconstructionLow = reconstructionLow;
        decompositionHigh = new double[length];
        reconstructionHigh = new double[length];
        for (int k = 0; k < length; k++)
        {
            decompositionHigh[k] = (k % 2 == 0 ? -1.0 : 1.0) * reconstructionLow[k];
            reconstructionHigh[k] = (k % 2 == 0 ? 1.0 : -1.0) * decompositionLow[k];
        }
    }

    /// <summary>
    /// Gets the canonical wavelet name (such as <c>haar</c>, <c>db4</c>, <c>sym8</c>, <c>coif3</c>, <c>bior4.4</c>, or <c>rbio2.2</c>).
    /// <para>ウェーブレットの標準名（<c>haar</c>、<c>db4</c>、<c>sym8</c>、<c>coif3</c>、<c>bior4.4</c>、<c>rbio2.2</c> など）を取得します。</para>
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Gets the filter length shared by all four filters.
    /// <para>4 つのフィルタに共通のフィルタ長を取得します。</para>
    /// </summary>
    public int FilterLength => decompositionLow.Length;

    /// <summary>
    /// Gets a value indicating whether this wavelet is orthogonal (<see langword="true"/> for Haar, Daubechies, Symlet, and Coiflet; <see langword="false"/> for biorthogonal families).
    /// <para>直交ウェーブレットかどうかを取得します（Haar・Daubechies・Symlet・Coiflet は <see langword="true"/>、双直交系は <see langword="false"/>）。</para>
    /// </summary>
    public bool IsOrthogonal { get; }

    /// <summary>
    /// Gets the number of vanishing moments of the decomposition wavelet.
    /// <para>分解側ウェーブレットの消失モーメント数を取得します。</para>
    /// </summary>
    /// <remarks>
    /// <para>Determined directly from the multiplicity of zeros of the dual low-pass filter at <c>z = -1</c>. Polynomials of degree strictly less than this value produce zero interior detail coefficients.</para>
    /// <para>双対低域フィルタの <c>z = −1</c> における零点の重複度から直接算出された値です。この次数未満の多項式では、境界の影響を受けない内部の詳細係数が 0 になります。</para>
    /// </remarks>
    public int VanishingMoments { get; }

    /// <summary>
    /// Gets the decomposition low-pass filter coefficients.
    /// <para>分解用の低域フィルタ係数を取得します。</para>
    /// </summary>
    public ReadOnlySpan<double> DecompositionLow => decompositionLow;

    /// <summary>
    /// Gets the decomposition high-pass filter coefficients.
    /// <para>分解用の高域フィルタ係数を取得します。</para>
    /// </summary>
    public ReadOnlySpan<double> DecompositionHigh => decompositionHigh;

    /// <summary>
    /// Gets the reconstruction low-pass filter coefficients.
    /// <para>再構成用の低域フィルタ係数を取得します。</para>
    /// </summary>
    public ReadOnlySpan<double> ReconstructionLow => reconstructionLow;

    /// <summary>
    /// Gets the reconstruction high-pass filter coefficients.
    /// <para>再構成用の高域フィルタ係数を取得します。</para>
    /// </summary>
    public ReadOnlySpan<double> ReconstructionHigh => reconstructionHigh;

    /// <inheritdoc/>
    public override string ToString() => Name;

    /// <summary>
    /// Gets the Haar wavelet (identical filter coefficients to <c>db1</c>).
    /// <para>Haar ウェーブレット（<c>db1</c> と同一のフィルタ係数）を取得します。</para>
    /// </summary>
    public static DiscreteWavelet Haar { get; } = Orthogonal("haar", DiscreteWaveletFilters.Daubechies[0], 1);

    /// <summary>
    /// Returns the Daubechies extremal-phase wavelet of the specified order.
    /// <para>指定した次数の Daubechies ウェーブレットを返します。</para>
    /// </summary>
    /// <param name="order">The wavelet order, from 1 to 20. / 次数（1〜20）。</param>
    /// <returns>The Daubechies wavelet with <paramref name="order"/> vanishing moments and filter length <c>2 · order</c>. / 消失モーメント数が <paramref name="order"/>、フィルタ長が <c>2 · order</c> の最小位相ウェーブレット。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="order"/> is outside <c>[1, 20]</c>. / <paramref name="order"/> が 1〜20 の範囲外の場合。</exception>
    public static DiscreteWavelet Daubechies(int order)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(order, 1);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(order, 20);
        return Cache.Daubechies[order - 1];
    }

    /// <summary>
    /// Returns the Symlet (least-asymmetric Daubechies) wavelet of the specified order.
    /// <para>指定した次数の Symlet ウェーブレットを返します。</para>
    /// </summary>
    /// <param name="order">The wavelet order, from 2 to 20. / 次数（2〜20）。</param>
    /// <returns>The Symlet wavelet of order <paramref name="order"/>. / 指定した次数のほぼ対称な位相をもつ Symlet ウェーブレット。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="order"/> is outside <c>[2, 20]</c>. / <paramref name="order"/> が 2〜20 の範囲外の場合。</exception>
    public static DiscreteWavelet Symlet(int order)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(order, 2);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(order, 20);
        return Cache.Symlets[order - 2];
    }

    /// <summary>
    /// Returns the Coiflet wavelet of the specified order.
    /// <para>指定した次数の Coiflet ウェーブレットを返します。</para>
    /// </summary>
    /// <param name="order">The wavelet order, from 1 to 5. / 次数（1〜5）。</param>
    /// <returns>The Coiflet wavelet with filter length <c>6 · order</c> and <c>2 · order</c> vanishing moments. / フィルタ長 <c>6 · order</c>・消失モーメント数 <c>2 · order</c> の Coiflet ウェーブレット。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="order"/> is outside <c>[1, 5]</c>. / <paramref name="order"/> が 1〜5 の範囲外の場合。</exception>
    public static DiscreteWavelet Coiflet(int order)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(order, 1);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(order, 5);
        return Cache.Coiflets[order - 1];
    }

    /// <summary>
    /// Returns the Cohen–Daubechies–Feauveau biorthogonal wavelet for the specified order pair.
    /// <para>指定した次数の組に対応する双直交ウェーブレット（Cohen–Daubechies–Feauveau 系）を返します。</para>
    /// </summary>
    /// <param name="reconstructionOrder">The reconstruction order <c>Nr</c>. / 再構成側の次数 <c>Nr</c>。</param>
    /// <param name="decompositionOrder">The decomposition order <c>Nd</c>. / 分解側の次数 <c>Nd</c>。</param>
    /// <returns>The biorthogonal wavelet (for example, 2.2 is LeGall 5/3 and 4.4 is CDF 9/7). / 対応する双直交ウェーブレット（2.2 は LeGall 5/3、4.4 は CDF 9/7）。</returns>
    /// <remarks>
    /// <para>Supported order pairs are 1.1, 1.3, 1.5, 2.2, 2.4, 2.6, 2.8, 3.1, 3.3, 3.5, 3.7, 3.9, 4.4, 5.5, and 6.8.</para>
    /// <para>対応する次数の組は 1.1、1.3、1.5、2.2、2.4、2.6、2.8、3.1、3.3、3.5、3.7、3.9、4.4、5.5、6.8 です。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">The specified order pair is not supported. / 指定した次数の組が未対応の場合。</exception>
    public static DiscreteWavelet Biorthogonal(int reconstructionOrder, int decompositionOrder) =>
        FindBiorthogonal(reconstructionOrder, decompositionOrder, reverse: false);

    /// <summary>
    /// Returns the reverse biorthogonal wavelet obtained by swapping the decomposition and reconstruction filters of <see cref="Biorthogonal"/>.
    /// <para><see cref="Biorthogonal"/> の分解フィルタと再構成フィルタを入れ替えた逆双直交ウェーブレットを返します。</para>
    /// </summary>
    /// <param name="reconstructionOrder">The primary order <c>Nr</c>. / 再構成側の次数 <c>Nr</c>。</param>
    /// <param name="decompositionOrder">The dual order <c>Nd</c>. / 分解側の次数 <c>Nd</c>。</param>
    /// <returns>The reverse biorthogonal wavelet for the specified order pair. / 指定した次数の組に対応する逆双直交ウェーブレット。</returns>
    /// <remarks>
    /// <para>Supports the same order pairs as <see cref="Biorthogonal"/>.</para>
    /// <para>指定できる次数の組は <see cref="Biorthogonal"/> と同じです。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">The specified order pair is not supported. / 指定した次数の組が未対応の場合。</exception>
    public static DiscreteWavelet ReverseBiorthogonal(int reconstructionOrder, int decompositionOrder) =>
        FindBiorthogonal(reconstructionOrder, decompositionOrder, reverse: true);

    /// <summary>
    /// Looks up a built-in discrete wavelet by name (case-insensitive, ignoring surrounding whitespace).
    /// <para>名前から組み込みの離散ウェーブレットを取得します（前後の空白および大文字・小文字を区別しません）。</para>
    /// </summary>
    /// <param name="name">The wavelet name, such as <c>haar</c>, <c>db1</c>–<c>db20</c>, <c>sym2</c>–<c>sym20</c>, <c>coif1</c>–<c>coif5</c>, <c>biorNr.Nd</c>, or <c>rbioNr.Nd</c>. / ウェーブレット名（<c>haar</c>、<c>db1</c>〜<c>db20</c>、<c>sym2</c>〜<c>sym20</c>、<c>coif1</c>〜<c>coif5</c>、<c>biorNr.Nd</c>、<c>rbioNr.Nd</c>）。</param>
    /// <returns>The matching <see cref="DiscreteWavelet"/> instance. / 名前に対応する <see cref="DiscreteWavelet"/>。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="name"/> is <see langword="null"/>. / <paramref name="name"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="name"/> is not a supported wavelet name. / 未対応のウェーブレット名が指定された場合。</exception>
    public static DiscreteWavelet FromName(string name)
    {
        ArgumentNullException.ThrowIfNull(name);
        string lower = name.Trim().ToLowerInvariant();
        if (lower == "haar")
        {
            return Haar;
        }

        foreach (var (prefix, factory) in new (string, Func<int, DiscreteWavelet>)[] { ("db", Daubechies), ("sym", Symlet), ("coif", Coiflet) })
        {
            if (lower.StartsWith(prefix, StringComparison.Ordinal)
                && int.TryParse(lower.AsSpan(prefix.Length), NumberStyles.None, CultureInfo.InvariantCulture, out int order))
            {
                try
                {
                    return factory(order);
                }
                catch (ArgumentOutOfRangeException)
                {
                    break;
                }
            }
        }

        foreach (var (prefix, reverse) in new[] { ("bior", false), ("rbio", true) })
        {
            if (lower.StartsWith(prefix, StringComparison.Ordinal))
            {
                var orders = lower.AsSpan(prefix.Length);
                int dot = orders.IndexOf('.');
                if (dot > 0
                    && int.TryParse(orders[..dot], NumberStyles.None, CultureInfo.InvariantCulture, out int nr)
                    && int.TryParse(orders[(dot + 1)..], NumberStyles.None, CultureInfo.InvariantCulture, out int nd)
                    && TryFindBiorthogonal(nr, nd, reverse, out var wavelet))
                {
                    return wavelet;
                }
            }
        }

        throw new ArgumentException($"Unknown or unsupported wavelet '{name}'.", nameof(name));
    }

    private static DiscreteWavelet Orthogonal(string name, double[] reconstructionLow, int vanishingMoments)
    {
        var decompositionLow = reconstructionLow.ToArray();
        Array.Reverse(decompositionLow);
        return new DiscreteWavelet(name, decompositionLow, reconstructionLow, isOrthogonal: true, vanishingMoments);
    }

    private static DiscreteWavelet FindBiorthogonal(int reconstructionOrder, int decompositionOrder, bool reverse) =>
        TryFindBiorthogonal(reconstructionOrder, decompositionOrder, reverse, out var wavelet)
            ? wavelet
            : throw new ArgumentOutOfRangeException(
                nameof(reconstructionOrder),
                $"{(reverse ? "rbio" : "bior")}{reconstructionOrder}.{decompositionOrder} is not available. "
                + "Supported orders: 1.1, 1.3, 1.5, 2.2, 2.4, 2.6, 2.8, 3.1, 3.3, 3.5, 3.7, 3.9, 4.4, 5.5, 6.8.");

    private static bool TryFindBiorthogonal(int reconstructionOrder, int decompositionOrder, bool reverse, out DiscreteWavelet wavelet)
    {
        var table = DiscreteWaveletFilters.Biorthogonal;
        for (int i = 0; i < table.Length; i++)
        {
            if (table[i].Reconstruction == reconstructionOrder && table[i].Decomposition == decompositionOrder)
            {
                wavelet = reverse ? Cache.ReverseBiorthogonal[i] : Cache.Biorthogonal[i];
                return true;
            }
        }

        wavelet = null!;
        return false;
    }

    private static class Cache
    {
        internal static readonly DiscreteWavelet[] Daubechies = DiscreteWaveletFilters.Daubechies
            .Select((h, i) => Orthogonal($"db{i + 1}", h, i + 1)).ToArray();

        internal static readonly DiscreteWavelet[] Symlets = DiscreteWaveletFilters.Symlets
            .Select((h, i) => Orthogonal($"sym{i + 2}", h, i + 2)).ToArray();

        internal static readonly DiscreteWavelet[] Coiflets = DiscreteWaveletFilters.Coiflets
            .Select((h, i) => Orthogonal($"coif{i + 1}", h, 2 * (i + 1))).ToArray();

        // 分解側ウェーブレットの消失モーメント数は、再構成側低域フィルタの z = -1 での零点重複度に一致する。
        internal static readonly DiscreteWavelet[] Biorthogonal = DiscreteWaveletFilters.Biorthogonal
            .Select(f => new DiscreteWavelet(
                $"bior{f.Reconstruction}.{f.Decomposition}", f.DecompositionLow, f.ReconstructionLow, isOrthogonal: false,
                ZerosAtNyquist(f.ReconstructionLow)))
            .ToArray();

        internal static readonly DiscreteWavelet[] ReverseBiorthogonal = DiscreteWaveletFilters.Biorthogonal
            .Select(f => new DiscreteWavelet(
                $"rbio{f.Reconstruction}.{f.Decomposition}", Reversed(f.ReconstructionLow), Reversed(f.DecompositionLow), isOrthogonal: false,
                ZerosAtNyquist(f.DecompositionLow)))
            .ToArray();
    }

    private static double[] Reversed(double[] values)
    {
        var result = values.ToArray();
        Array.Reverse(result);
        return result;
    }

    // 交代モーメント和 Σ (-1)^k k^p f[k] = 0 が成り立つ最小の次数まで数え、z = -1 の零点重複度を求める。
    private static int ZerosAtNyquist(double[] filter)
    {
        int count = 0;
        for (int p = 0; p < filter.Length; p++)
        {
            double sum = 0, magnitude = 0;
            for (int k = 0; k < filter.Length; k++)
            {
                double term = (k % 2 == 0 ? 1.0 : -1.0) * Math.Pow(k, p) * filter[k];
                sum += term;
                magnitude += Math.Abs(term);
            }

            if (Math.Abs(sum) > 1e-9 * magnitude)
            {
                break;
            }

            count++;
        }

        return count;
    }
}
