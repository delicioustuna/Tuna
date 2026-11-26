namespace Tuna.SignalProcessing;

/// <summary>
/// Specifies the threshold selection rule used for wavelet denoising.
/// <para>ウェーブレットノイズ除去におけるしきい値決定規則を指定します。</para>
/// </summary>
public enum WaveletThresholdRule
{
    /// <summary>
    /// Selects a subband-adaptive BayesShrink threshold from detail coefficient variance (default).
    /// <para>詳細係数の分散からサブバンドごとに BayesShrink しきい値を決定します（既定）。</para>
    /// </summary>
    /// <remarks>
    /// <para>Computes <c>σ² / √max(mean(d²) - σ², ε)</c> per detail subband, where <c>σ</c> is the estimated noise standard deviation.</para>
    /// <para>詳細係数のサブバンドごとに <c>σ² / √max(mean(d²) − σ², ε)</c> を計算します（<c>σ</c> は推定された雑音の標準偏差）。</para>
    /// </remarks>
    Bayes,

    /// <summary>
    /// Applies the VisuShrink universal threshold <c>σ · √(2 ln N)</c> (Donoho–Johnstone 1994) to all detail subbands, where <c>N</c> is the number of signal samples.
    /// <para>VisuShrink（Donoho–Johnstone 1994 の普遍しきい値）を適用し、すべての詳細係数に <c>σ · √(2 ln N)</c>（<c>N</c> は信号標本数）を使用します。</para>
    /// </summary>
    Universal,
}
