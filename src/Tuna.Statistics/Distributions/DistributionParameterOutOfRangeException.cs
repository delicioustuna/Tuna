namespace Tuna.Statistics.Distributions;

/// <summary>
/// Indicates that a differentiable distribution parameter is outside its domain.
/// <para>自動微分に対応する分布のパラメータが定義域外であることを示す例外です。</para>
/// </summary>
public sealed class DistributionParameterOutOfRangeException : ArgumentOutOfRangeException
{
    /// <summary>
    /// Initializes an exception for an out-of-domain distribution parameter.
    /// <para>定義域外の分布パラメータを示す例外を初期化します。</para>
    /// </summary>
    /// <param name="paramName">The parameter name. / パラメータ名。</param>
    /// <param name="actualValue">The value outside the domain. / 定義域外の値。</param>
    /// <param name="message">A description of the valid domain. / 定義域を説明するメッセージ。</param>
    public DistributionParameterOutOfRangeException(string? paramName, object? actualValue, string? message)
        : base(paramName, actualValue, message)
    {
    }
}
