using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.Statistics.Distributions;

public readonly partial struct NormalDistribution
{
    private const double InverseSqrtTwoPi = 0.3989422804014327;

    /// <summary>
    /// Records the normal log density and its derivatives with respect to the mean, standard deviation, and variate.
    /// <para>正規分布の対数密度と、平均・標準偏差・評価点に関する微分を記録します。</para>
    /// </summary>
    /// <param name="mean">The finite mean <c>μ</c>. / 有限の平均 <c>μ</c>。</param>
    /// <param name="standardDeviation">The positive finite standard deviation <c>σ</c>. / 正で有限の標準偏差 <c>σ</c>。</param>
    /// <param name="x">The variate. / 評価点。</param>
    /// <returns>The log density node. / 対数密度を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">A distribution parameter is outside its domain. / 分布パラメータが範囲外の場合。</exception>
    /// <exception cref="InvalidOperationException">An input node is stale, the input nodes belong to different tapes, or the tape has already run backward. / 失効したノードがある場合、ノードのテープが異なる場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPdf(ScalarNode mean, ScalarNode standardDeviation, ScalarNode x)
    {
        var distribution = new NormalDistribution(mean.Value, standardDeviation.Value, forDifferentiation: true);
        double observation = x.Value;
        double z = distribution.Standardize(observation);
        double value = LogPdfValue(z, Math.Log(distribution._standardDeviation));
        bool requiresLocation = mean.RequiresGradient || x.RequiresGradient;
        bool requiresScale = standardDeviation.RequiresGradient;
        if (!requiresLocation && !requiresScale)
        {
            return value;
        }

        double dMean = requiresLocation ? distribution.LogPdfMeanDerivative(observation, z) : 0;
        double dScale = requiresScale ? distribution.LogPdfScaleDerivative(observation, z) : 0;
        return ScalarNode.Record(value, [(mean, dMean), (standardDeviation, dScale), (x, -dMean)]);
    }

    /// <summary>
    /// Records the normal log likelihood of independent observations as a single operation.
    /// <para>正規分布に従う独立な観測値の対数尤度を 1 回の演算として記録します。</para>
    /// </summary>
    /// <param name="mean">The finite mean <c>μ</c>. / 有限の平均 <c>μ</c>。</param>
    /// <param name="standardDeviation">The positive finite standard deviation <c>σ</c>. / 正で有限の標準偏差 <c>σ</c>。</param>
    /// <param name="observations">The observations. / 観測値のベクトル。</param>
    /// <returns>The log likelihood node for the observations. / 観測全体の対数尤度を表すノード。</returns>
    /// <remarks>
    /// <para>Values are summed in input order. An empty vector gives zero for the value and gradient. The tape stores only the aggregated partial derivatives.</para>
    /// <para>対数密度を入力順に合計します。空のベクトルでは値と勾配がともに 0 です。テープには各パラメータの偏微分の合計だけを保存します。</para>
    /// </remarks>
    /// <exception cref="DistributionParameterOutOfRangeException">A distribution parameter is outside its domain. / 分布パラメータが範囲外の場合。</exception>
    /// <exception cref="InvalidOperationException">An input node is stale, the input nodes belong to different tapes, or the tape has already run backward. / 失効したノードがある場合、ノードのテープが異なる場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPdf(ScalarNode mean, ScalarNode standardDeviation, ReadOnlyVectorSpan<double> observations)
    {
        var distribution = new NormalDistribution(mean.Value, standardDeviation.Value, forDifferentiation: true);
        bool requiresMean = mean.RequiresGradient;
        bool requiresScale = standardDeviation.RequiresGradient;
        var (value, dMean, dScale) = distribution.LogPdfSum(observations, requiresMean, requiresScale);
        return requiresMean || requiresScale ? ScalarNode.Record(value, [(mean, dMean), (standardDeviation, dScale)]) : value;
    }

    /// <summary>
    /// Records the normal log cumulative probability and its derivatives with respect to the mean, standard deviation, and variate.
    /// <para>正規分布の対数累積確率と、平均・標準偏差・評価点に関する微分を記録します。</para>
    /// </summary>
    /// <param name="mean">The finite mean <c>μ</c>. / 有限の平均 <c>μ</c>。</param>
    /// <param name="standardDeviation">The positive finite standard deviation <c>σ</c>. / 正で有限の標準偏差 <c>σ</c>。</param>
    /// <param name="x">The threshold in <c>P(X ≤ x)</c>. / 下側確率 <c>P(X ≤ x)</c> のしきい値。</param>
    /// <returns>The log cumulative probability node. / 対数累積確率を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">A distribution parameter is outside its domain. / 分布パラメータが範囲外の場合。</exception>
    /// <exception cref="InvalidOperationException">An input node is stale, the input nodes belong to different tapes, or the tape has already run backward. / 失効したノードがある場合、ノードのテープが異なる場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogCdf(ScalarNode mean, ScalarNode standardDeviation, ScalarNode x) =>
        LogTail(mean, standardDeviation, x, upper: false);

    /// <summary>
    /// Records the normal log survival probability and its derivatives with respect to the mean, standard deviation, and variate.
    /// <para>正規分布の対数上側確率と、平均・標準偏差・評価点に関する微分を記録します。</para>
    /// </summary>
    /// <param name="mean">The finite mean <c>μ</c>. / 有限の平均 <c>μ</c>。</param>
    /// <param name="standardDeviation">The positive finite standard deviation <c>σ</c>. / 正で有限の標準偏差 <c>σ</c>。</param>
    /// <param name="x">The threshold in <c>P(X &gt; x)</c>. / 上側確率 <c>P(X &gt; x)</c> のしきい値。</param>
    /// <returns>The log survival probability node. / 対数上側確率を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">A distribution parameter is outside its domain. / 分布パラメータが範囲外の場合。</exception>
    /// <exception cref="InvalidOperationException">An input node is stale, the input nodes belong to different tapes, or the tape has already run backward. / 失効したノードがある場合、ノードのテープが異なる場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogSf(ScalarNode mean, ScalarNode standardDeviation, ScalarNode x) =>
        LogTail(mean, standardDeviation, x, upper: true);

    private static ScalarNode LogTail(ScalarNode mean, ScalarNode standardDeviation, ScalarNode x, bool upper)
    {
        var distribution = new NormalDistribution(mean.Value, standardDeviation.Value, forDifferentiation: true);
        double observation = x.Value;
        double z = distribution.Standardize(observation);
        double value = upper ? XMath.Special.NormalLogSf(z) : XMath.Special.NormalLogCdf(z);
        bool requiresLocation = mean.RequiresGradient || x.RequiresGradient;
        bool requiresScale = standardDeviation.RequiresGradient;
        if (!requiresLocation && !requiresScale)
        {
            return value;
        }

        var (location, scale) = distribution.LogTailDerivatives(observation, z, upper, requiresScale);
        return upper
            ? ScalarNode.Record(value, [(mean, location), (standardDeviation, scale), (x, -location)])
            : ScalarNode.Record(value, [(mean, -location), (standardDeviation, -scale), (x, location)]);
    }

    internal double LogPdfMeanDerivative(double x, double z)
    {
        if (double.IsInfinity(z) && double.IsFinite(x) && _standardDeviation > 1)
        {
            // z 自体が範囲を超えていても z/σ が表現できる場合がある。
            double halfResidual = 0.5 * x - 0.5 * _mean;
            return (halfResidual / _standardDeviation) / (0.5 * _standardDeviation);
        }

        return Math.Abs(z) < 1e-300
            ? DistributionMath.ScaledRatio([x - _mean], [_standardDeviation, _standardDeviation])
            : z / _standardDeviation;
    }

    internal double LogPdfScaleDerivative(double x, double z)
    {
        double numerator = LogPdfScaleNumerator(x, z);
        return double.IsPositiveInfinity(numerator) && double.IsFinite(z)
            ? (z / _standardDeviation) * z
            : numerator / _standardDeviation;
    }

    private double LogPdfScaleNumerator(double x, double z)
    {
        double absolute = Math.Abs(z);
        if (absolute < 0.75 || absolute > 1.5 || double.IsNaN(z))
        {
            return Math.FusedMultiplyAdd(z, z, -1);
        }

        // ±1 の近くでは、丸めた z² から 1 を引くと分子が桁落ちする。差の下位桁を使って (|z| − 1)(|z| + 1) を求める。
        var (difference, error, scale) = RescaledDifference(x);
        double offset = (Math.Abs(difference) - scale) + (z < 0 ? -error : error);
        return (offset / scale) * (absolute + 1);
    }

    internal (double Location, double Scale) LogTailDerivatives(double x, double z, bool upper, bool requiresScale)
    {
        double hazardInput = upper ? z : -z;
        if (hazardInput < -2)
        {
            return LogNearOneDerivatives(x, z, requiresScale);
        }

        double hazard = XMath.Special.NormalHazard(hazardInput);
        double location = double.IsInfinity(hazard) ? Math.Abs(LogPdfMeanDerivative(x, z)) : hazard / _standardDeviation;
        if (!requiresScale)
        {
            return (location, 0);
        }

        double absolute = Math.Abs(z);
        double scale = absolute < 1e-300
            ? DistributionMath.ScaledRatio([x - _mean, hazard], [_standardDeviation, _standardDeviation])
            : absolute > 1e150 ? location * z : (z * hazard) / _standardDeviation;
        return (location, scale);
    }

    private (double Location, double Scale) LogNearOneDerivatives(double x, double z, bool requiresScale)
    {
        double absolute = Math.Abs(z);
        if (absolute >= 56)
        {
            // σ が最小の非正規化数でも φ(z)/σ と zφ(z)/σ が0に丸まる。
            return (0, 0);
        }

        var (difference, error, scale) = RescaledDifference(x);
        if (scale < 1e-280)
        {
            int power = -Math.ILogB(scale);
            (difference, error, scale) = (Math.ScaleB(difference, power), Math.ScaleB(error, power), Math.ScaleB(scale, power));
        }

        // z と z²/2 の下位桁を指数へ戻す。分母は大きい側の確率 Φ(|z|)。
        double zError = (Math.FusedMultiplyAdd(-z, scale, difference) + error) / scale;
        double halfSquare = (0.5 * z) * z;
        double exponentError = Math.FusedMultiplyAdd(0.5 * z, z, -halfSquare) + z * zError;
        double factor = InverseSqrtTwoPi * (1 - exponentError) / XMath.Special.NormalCdf(absolute);
        if (halfSquare < 700)
        {
            double hazard = factor * Math.Exp(-halfSquare);
            return (hazard / _standardDeviation, requiresScale ? (z * hazard) / _standardDeviation : 0);
        }

        // 密度が非正規化数や0に丸まる裾では、σ で割った後に丸める。
        double part = Math.Exp(-0.25 * halfSquare);
        double location = DistributionMath.ScaledRatio([factor, part, part, part, part], [_standardDeviation]);
        double derivative = requiresScale ? DistributionMath.ScaledRatio([z, factor, part, part, part, part], [_standardDeviation]) : 0;
        return (location, derivative);
    }

    private (double Difference, double Error, double Scale) RescaledDifference(double x)
    {
        double mean = _mean, scale = _standardDeviation;
        double difference = x - mean;
        if (double.IsInfinity(difference))
        {
            x *= 0.5;
            mean *= 0.5;
            scale *= 0.5;
            difference = x - mean;
        }

        return (difference, CompensatedSum.DifferenceError(x, mean, difference), scale);
    }

    // 観測数は最大 2³¹ − 1。大きな値だけを 2⁻³² 倍し、差の和が範囲内に収まるようにする。
    private const double LargeObservation = 1e298;
    private const double LargeObservationScale = 4294967296;

    internal (double Value, double Mean, double Scale) LogPdfSum(ReadOnlyVectorSpan<double> observations, bool requiresMean, bool requiresScale)
    {
        double value = 0, logScale = Math.Log(_standardDeviation);
        double nonfinite = 0, largestObservation = Math.Abs(_mean);
        bool hasNaN = false, needsScaledSquares = false;
        var residuals = new CompensatedSum();
        var largeResiduals = new CompensatedSum();
        var scaleNumerators = new CompensatedSum();
        for (int i = 0; i < observations.Length; i++)
        {
            double x = observations[i];
            double z = Standardize(x);
            value += LogPdfValue(z, logScale);
            if (!requiresMean && !requiresScale)
            {
                continue;
            }

            if (!double.IsFinite(x))
            {
                nonfinite += x;
                hasNaN |= double.IsNaN(x);
                continue;
            }

            if (requiresMean)
            {
                AddResidual(x, ref residuals, ref largeResiduals);
            }

            if (requiresScale)
            {
                largestObservation = Math.Max(largestObservation, Math.Abs(x));
                if (Math.Abs(z) <= 1e145)
                {
                    scaleNumerators.Add(LogPdfScaleNumerator(x, z));
                }
                else
                {
                    needsScaledSquares = true;
                }
            }
        }

        if (nonfinite != 0)
        {
            return (value, nonfinite, hasNaN ? double.NaN : double.PositiveInfinity);
        }

        double dMean = requiresMean ? MeanDerivative(residuals, largeResiduals) : 0;
        double dScale = needsScaledSquares
            ? LargeScaleDerivative(observations, largestObservation)
            : scaleNumerators.Value / _standardDeviation;
        // |Σz² − n| < n/4 では、標準化と二乗の丸め誤差が尺度の勾配で増幅される。
        if (requiresScale && !needsScaledSquares && observations.Length > 1
            && Math.Abs(scaleNumerators.Value) < 0.25 * observations.Length)
        {
            dScale = ScaleDerivativeNearCancellation(observations);
        }

        return (value, dMean, dScale);
    }

    private void AddResidual(double x, ref CompensatedSum residuals, ref CompensatedSum largeResiduals)
    {
        bool largeX = Math.Abs(x) > LargeObservation;
        bool largeMean = Math.Abs(_mean) > LargeObservation;
        if (!largeX && !largeMean)
        {
            residuals.AddDifference(x, _mean);
        }
        else if (largeX && largeMean)
        {
            largeResiduals.AddDifference(x / LargeObservationScale, _mean / LargeObservationScale);
        }
        else if (largeX)
        {
            largeResiduals.Add(x / LargeObservationScale);
            residuals.Add(-_mean);
        }
        else
        {
            largeResiduals.Add(-_mean / LargeObservationScale);
            residuals.Add(x);
        }
    }

    private double MeanDerivative(CompensatedSum residuals, CompensatedSum largeResiduals)
    {
        if (largeResiduals.Value == 0)
        {
            return DistributionMath.ScaledRatio([residuals.Value], [_standardDeviation, _standardDeviation]);
        }

        largeResiduals.Add(residuals.Sum / LargeObservationScale);
        if (largeResiduals.Value == 0)
        {
            return DistributionMath.ScaledRatio([residuals.Correction], [_standardDeviation, _standardDeviation]);
        }

        largeResiduals.Add(residuals.Correction / LargeObservationScale);
        return DistributionMath.ScaledRatio([largeResiduals.Value, LargeObservationScale], [_standardDeviation, _standardDeviation]);
    }

    private double LargeScaleDerivative(ReadOnlyVectorSpan<double> observations, double normalization)
    {
        // |z| > 10¹⁴⁵ の項があれば、Σz² に対し −n は無視できる。実際の差で先に割って二乗和の範囲を保つ。
        var squares = new CompensatedSum();
        for (int i = 0; i < observations.Length; i++)
        {
            double residual = DistributionMath.Standardize(observations[i], _mean, normalization);
            squares.Add(residual * residual);
        }

        return DistributionMath.ScaledRatio([squares.Value, normalization, normalization], [_standardDeviation, _standardDeviation, _standardDeviation]);
    }

    private double ScaleDerivativeNearCancellation(ReadOnlyVectorSpan<double> observations)
    {
        // (Σ(x − μ)² − nσ²)/σ³ の分子を、σ と同じ2の冪で正規化して計算する。
        int power = -Math.ILogB(_standardDeviation);
        double scale = Math.ScaleB(_standardDeviation, power);
        double scaleSquare = scale * scale;
        double scaleError = Math.FusedMultiplyAdd(scale, scale, -scaleSquare);
        double remainderScale = 0, remainderSquares = 0;
        var sum = new CompensatedSum();
        for (int i = 0; i < observations.Length; i++)
        {
            var (difference, error, localScale) = RescaledDifference(observations[i]);
            int localPower = localScale == _standardDeviation ? power : power + 1;
            double residual = Math.ScaleB(difference, localPower);
            double remainder = Math.ScaleB(error, localPower);
            double square = residual * residual;
            sum.AddDifference(square, scaleSquare);
            sum.Add(Math.FusedMultiplyAdd(residual, residual, -square));
            sum.Add(-scaleError);
            sum.Add((2 * residual) * remainder);

            // 残差の二乗だけが残る場合も、最後に σ で割るまで指数を保つ。
            double magnitude = Math.Abs(remainder);
            if (magnitude > remainderScale)
            {
                double ratio = remainderScale / magnitude;
                remainderSquares = 1 + remainderSquares * ratio * ratio;
                remainderScale = magnitude;
            }
            else if (magnitude > 0)
            {
                double ratio = magnitude / remainderScale;
                remainderSquares += ratio * ratio;
            }
        }

        return DistributionMath.ScaledRatio([sum.Value], [scale, scale, _standardDeviation])
            + DistributionMath.ScaledRatio([remainderSquares, remainderScale, remainderScale], [scale, scale, _standardDeviation]);
    }
}
