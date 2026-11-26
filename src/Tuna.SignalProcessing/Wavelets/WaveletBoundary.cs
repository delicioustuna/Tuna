namespace Tuna.SignalProcessing;

/// <summary>
/// Specifies the signal boundary extension mode used by discrete wavelet transforms.
/// <para>離散ウェーブレット変換における信号境界の拡張方式を指定します。</para>
/// </summary>
/// <remarks>
/// <para>For all modes except <see cref="Periodization"/>, a single-level decomposition of length <c>n</c> with filter length <c>F</c> produces coefficient length <c>⌊(n + F - 1) / 2⌋</c> and allows exact reconstruction.</para>
/// <para><see cref="Periodization"/> 以外のモードでは、信号長 <c>n</c>・フィルタ長 <c>F</c> に対する 1 段分解の係数長が <c>⌊(n + F − 1) / 2⌋</c> となり、逆変換で元の信号を完全再構成できます。</para>
/// </remarks>
public enum WaveletBoundary
{
    /// <summary>
    /// Half-point symmetric extension: <c>… x₂ x₁ | x₁ … xₙ | xₙ xₙ₋₁ …</c> (default).
    /// <para>半標本対称拡張 <c>… x₂ x₁ | x₁ … xₙ | xₙ xₙ₋₁ …</c>（既定）。</para>
    /// </summary>
    Symmetric,

    /// <summary>
    /// Periodization producing minimal coefficient length <c>⌈n / 2⌉</c> (duplicates the last sample when <c>n</c> is odd). Yields an orthogonal transform for orthogonal wavelets.
    /// <para>周期化。係数長が最小の <c>⌈n / 2⌉</c> となり（<c>n</c> が奇数のときは末尾を 1 点複製）、直交ウェーブレットでは変換が直交行列になります。</para>
    /// </summary>
    Periodization,

    /// <summary>
    /// Zero padding: <c>… 0 0 | x₁ … xₙ | 0 0 …</c>.
    /// <para>ゼロパディング <c>… 0 0 | x₁ … xₙ | 0 0 …</c>。</para>
    /// </summary>
    Zero,

    /// <summary>
    /// Constant edge replication: <c>… x₁ x₁ | x₁ … xₙ | xₙ xₙ …</c>.
    /// <para>端点の定数拡張 <c>… x₁ x₁ | x₁ … xₙ | xₙ xₙ …</c>。</para>
    /// </summary>
    Constant,

    /// <summary>
    /// Whole-point symmetric extension without repeating edge values: <c>… x₃ x₂ | x₁ … xₙ | xₙ₋₁ xₙ₋₂ …</c>. Requires <c>n ≥ 2</c>.
    /// <para>全標本対称拡張（端点を繰り返さない）<c>… x₃ x₂ | x₁ … xₙ | xₙ₋₁ xₙ₋₂ …</c>。<c>n ≥ 2</c> が必要です。</para>
    /// </summary>
    Reflect,

    /// <summary>
    /// Periodic extension: <c>… xₙ₋₁ xₙ | x₁ … xₙ | x₁ x₂ …</c>, producing coefficient length <c>⌊(n + F - 1) / 2⌋</c>.
    /// <para>周期拡張 <c>… xₙ₋₁ xₙ | x₁ … xₙ | x₁ x₂ …</c>（<see cref="Periodization"/> とは異なり係数長は <c>⌊(n + F − 1) / 2⌋</c> です）。</para>
    /// </summary>
    Periodic,
}
