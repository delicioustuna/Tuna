namespace Tuna.SignalProcessing;

/// <summary>
/// Specifies the scaling convention applied to discrete Fourier transforms.
/// <para>離散フーリエ変換に適用する正規化（スケーリング）方式を指定します。</para>
/// </summary>
public enum FourierNormalization
{
    /// <summary>
    /// No scaling on the forward transform and <c>1/n</c> on the inverse transform (default).
    /// <para>順変換ではスケーリングを行わず、逆変換で <c>1/n</c> を掛けます（既定）。</para>
    /// </summary>
    Backward,

    /// <summary>
    /// Scales both forward and inverse transforms by <c>1/√n</c>, making the transform unitary.
    /// <para>順変換・逆変換の両方に <c>1/√n</c> を掛け、ユニタリ変換にします。</para>
    /// </summary>
    Orthonormal,

    /// <summary>
    /// Scales the forward transform by <c>1/n</c> and applies no scaling on the inverse transform.
    /// <para>順変換で <c>1/n</c> を掛け、逆変換ではスケーリングを行いません。</para>
    /// </summary>
    Forward,
}

internal static class FourierScaling
{
    internal static double Forward(FourierNormalization normalization, int length) => normalization switch
    {
        FourierNormalization.Backward => 1,
        FourierNormalization.Orthonormal => length == 0 ? 1 : 1 / Math.Sqrt(length),
        FourierNormalization.Forward => length == 0 ? 1 : 1.0 / length,
        _ => throw new ArgumentOutOfRangeException(nameof(normalization)),
    };

    internal static double Inverse(FourierNormalization normalization, int length) => normalization switch
    {
        FourierNormalization.Backward => length == 0 ? 1 : 1.0 / length,
        FourierNormalization.Orthonormal => length == 0 ? 1 : 1 / Math.Sqrt(length),
        FourierNormalization.Forward => 1,
        _ => throw new ArgumentOutOfRangeException(nameof(normalization)),
    };
}
