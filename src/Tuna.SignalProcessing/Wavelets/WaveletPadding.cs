namespace Tuna.SignalProcessing;

/// <summary>
/// Specifies the boundary padding mode used by the continuous wavelet transform.
/// <para>連続ウェーブレット変換における信号境界のパディング方式を指定します。</para>
/// </summary>
public enum WaveletPadding
{
    /// <summary>
    /// Zero-pads the signal to <c>Fourier.NextFastLength(2n - 1)</c> so edge values do not wrap around (default).
    /// <para>長さ <c>Fourier.NextFastLength(2n − 1)</c> までゼロパディングして FFT を実行し、境界の回り込みを防ぎます（既定）。</para>
    /// </summary>
    Zero,

    /// <summary>
    /// Treats the signal as periodic with period <c>n</c> and transforms it directly without padding.
    /// <para>信号を周期 <c>n</c> の周期信号とみなし、パディングなしで長さ <c>n</c> の FFT を実行します。</para>
    /// </summary>
    Periodic,
}
