namespace Tuna.SignalProcessing;

/// <summary>
/// Specifies how wavelet detail coefficients are shrunk by a threshold <c>T</c>.
/// <para>しきい値 <c>T</c> によるウェーブレット係数の収縮方式を指定します。</para>
/// </summary>
public enum WaveletShrinkage
{
    /// <summary>
    /// Soft thresholding: <c>sign(x) · max(|x| - T, 0)</c> (default, equivalent to <see cref="XMath.SoftThreshold{T}(LinearAlgebra.ReadOnlyVectorSpan{T}, T)"/>).
    /// <para>軟判定しきい値処理 <c>sign(x) · max(|x| − T, 0)</c>（既定。<see cref="XMath.SoftThreshold{T}(LinearAlgebra.ReadOnlyVectorSpan{T}, T)"/> と同一）。</para>
    /// </summary>
    Soft,

    /// <summary>
    /// Hard thresholding: sets coefficients with <c>|x| &lt; T</c> to zero and leaves all others unchanged.
    /// <para>硬判定しきい値処理。<c>|x| &lt; T</c> の係数を 0 にし、それ以外はそのまま残します。</para>
    /// </summary>
    Hard,
}
