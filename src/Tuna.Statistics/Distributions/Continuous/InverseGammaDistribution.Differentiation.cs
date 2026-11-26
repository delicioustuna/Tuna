using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.Statistics.Distributions;

public readonly partial struct InverseGammaDistribution
{
    /// <summary>
    /// Records the inverse gamma log density and its derivatives.
    /// <para>逆ガンマ分布の対数密度とその微分を記録します。</para>
    /// </summary>
    /// <param name="shape">The positive finite shape. / 正で有限の形状。</param>
    /// <param name="scale">The positive finite scale. / 正で有限の尺度。</param>
    /// <param name="x">The variate. / 評価点。</param>
    /// <returns>The log density node. / 対数密度を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">The shape or scale is outside its domain. / 形状または尺度が範囲外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPdf(ScalarNode shape, ScalarNode scale, ScalarNode x) =>
        GammaLogProbability.Record(new InverseGammaDistribution(shape.Value, scale.Value, forDifferentiation: true), shape, scale, x, LogProbability.Density, inverse: true);

    /// <summary>
    /// Records the inverse gamma log cumulative probability and its derivatives.
    /// <para>逆ガンマ分布の対数累積確率とその微分を記録します。</para>
    /// </summary>
    /// <param name="shape">The positive finite shape. / 正で有限の形状。</param>
    /// <param name="scale">The positive finite scale. / 正で有限の尺度。</param>
    /// <param name="x">The variate. / 評価点。</param>
    /// <returns>The log cumulative probability node. / 対数累積確率を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">The shape or scale is outside its domain. / 形状または尺度が範囲外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogCdf(ScalarNode shape, ScalarNode scale, ScalarNode x) =>
        GammaLogProbability.Record(new InverseGammaDistribution(shape.Value, scale.Value, forDifferentiation: true), shape, scale, x, LogProbability.LowerTail, inverse: true);

    /// <summary>
    /// Records the inverse gamma log survival probability and its derivatives.
    /// <para>逆ガンマ分布の対数生存確率とその微分を記録します。</para>
    /// </summary>
    /// <param name="shape">The positive finite shape. / 正で有限の形状。</param>
    /// <param name="scale">The positive finite scale. / 正で有限の尺度。</param>
    /// <param name="x">The variate. / 評価点。</param>
    /// <returns>The log survival probability node. / 対数生存確率を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">The shape or scale is outside its domain. / 形状または尺度が範囲外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogSf(ScalarNode shape, ScalarNode scale, ScalarNode x) =>
        GammaLogProbability.Record(new InverseGammaDistribution(shape.Value, scale.Value, forDifferentiation: true), shape, scale, x, LogProbability.UpperTail, inverse: true);

    /// <summary>
    /// Records the inverse gamma log likelihood of independent observations as a single operation.
    /// <para>逆ガンマ分布に従う独立な観測値の対数尤度を 1 回の演算として記録します。</para>
    /// </summary>
    /// <param name="shape">The positive finite shape. / 正で有限の形状。</param>
    /// <param name="scale">The positive finite scale. / 正で有限の尺度。</param>
    /// <param name="observations">The observations, summed in input order. / 入力順に集計する観測値。</param>
    /// <returns>The log likelihood node. / 対数尤度を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">The shape or scale is outside its domain. / 形状または尺度が範囲外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPdf(ScalarNode shape, ScalarNode scale, ReadOnlyVectorSpan<double> observations) =>
        GammaLogProbability.RecordSum(new InverseGammaDistribution(shape.Value, scale.Value, forDifferentiation: true), shape, scale, observations, inverse: true);
}
