using Tuna.Integration;
using Tuna.LinearAlgebra;
using Tuna.RandomSource;

namespace Tuna.Statistics.Distributions;

// Genz (1992) の変換により超直方体確率 P(lower < X <= upper) を単位超立方体上の積分に帰着する。
// 有限自由度 ν の多変量 t 分布では s = √(V/ν)（V ~ χ²(ν)）の分位点変数を 1 次元追加する。
// インスタンスは作業バッファ _y を共有するため単一スレッドから評価する。
internal sealed class GenzProbability : IMultivariateIntegrand
{
    // 決定的な既定結果を得るための固定シード
    private const ulong Seed = 0x5851F42D4C957F2DUL;

    // 条件付き分散の丸め誤差による非正値のみを下限クリップする
    private const double MinimumVariance = double.Epsilon;

    // 正規化数の最小値 (2^-1022) および Φ⁻¹ が有限となる確率上限 (1 - 2^-53)
    private const double SmallestNormal = 2.2250738585072014e-308;
    private const double MaxProbability = 1 - 1.1102230246251565e-16;

    // 積分次元がこれ以下は直積求積 (Cubature)、それを超える次元は準モンテカルロ積分を使用する
    private const int ProductRuleMaxDimension = 2;

    private readonly int _dimension;
    private readonly double _degreesOfFreedom;
    private readonly bool _studentT;
    private readonly double[] _factor;
    private readonly double[] _lower;
    private readonly double[] _upper;
    private readonly double[] _y;

    private GenzProbability(double[] scale, double[] location, double[] lower, double[] upper, double degreesOfFreedom)
    {
        int d = location.Length;
        _dimension = d;
        _degreesOfFreedom = degreesOfFreedom;
        _studentT = !double.IsPositiveInfinity(degreesOfFreedom);
        _factor = new double[d * d];
        _lower = new double[d];
        _upper = new double[d];
        _y = new double[d];
        Pivot(scale, location, lower, upper);
    }

    internal int IntegrationDimension => _studentT ? _dimension : _dimension - 1;

    internal static IntegrationResult Probability(double[] scale, double[] location, double degreesOfFreedom,
        ReadOnlyVectorSpan<double> lower, ReadOnlyVectorSpan<double> upper, IntegrationOptions? options)
    {
        int d = location.Length;
        // 空区間の判定より先に、すべての成分の NaN を確かめる（成分の並び順で結果が変わらないように）
        for (int i = 0; i < d; i++)
        {
            if (double.IsNaN(lower[i]) || double.IsNaN(upper[i]))
            {
                return new IntegrationResult(double.NaN, double.NaN, 0, IntegrationTermination.NonFiniteValue);
            }
        }

        for (int i = 0; i < d; i++)
        {
            if (!(lower[i] < upper[i]))
            {
                return new IntegrationResult(0, 0, 0, IntegrationTermination.Tolerance);
            }
        }

        options ??= DefaultOptions;
        var integrand = new GenzProbability(scale, location, Copy(lower), Copy(upper), degreesOfFreedom);
        int m = integrand.IntegrationDimension;
        if (m == 0)
        {
            // 1次元の正規分布は積分しなくてよい
            return new IntegrationResult(integrand.Evaluate(default), 0, 1, IntegrationTermination.Tolerance);
        }

        if (m <= ProductRuleMaxDimension)
        {
            var zero = new double[m];
            var one = new double[m];
            Array.Fill(one, 1.0);
            return Cubature.Integrate(integrand, zero, one, options);
        }

        var quasiMonteCarlo = new QuasiMonteCarloOptions
        {
            AbsoluteTolerance = options.AbsoluteTolerance,
            RelativeTolerance = options.RelativeTolerance,
            MaxEvaluations = options.MaxEvaluations,
        };
        var result = QuasiMonteCarlo.Integrate(integrand, m, new Pcg64DxsmRandom(Seed), quasiMonteCarlo);
        return new IntegrationResult(result.Value, result.StandardError, result.Evaluations, result.Termination);
    }

    internal static IntegrationResult Cdf(double[] scale, double[] location, double degreesOfFreedom,
        ReadOnlyVectorSpan<double> upper, IntegrationOptions? options)
    {
        var lower = new double[location.Length];
        Array.Fill(lower, double.NegativeInfinity);
        return Probability(scale, location, degreesOfFreedom, lower, upper, options);
    }

    internal static readonly IntegrationOptions DefaultOptions = new()
    {
        AbsoluteTolerance = 1e-8,
        RelativeTolerance = 0,
        MaxEvaluations = 1L << 22,
    };

    public double Evaluate(ReadOnlyVectorSpan<double> x)
    {
        int d = _dimension;
        double scale = 1;
        if (_studentT)
        {
            // u = x[d - 1] のカイ二乗分位点 V から s = √(V/ν) を求める。
            // ν が小さいと s が 0 に丸まる測度が非無視となるため、有限端は 0 に縮み無限端は無限のまま保つ。
            scale = Math.Sqrt(2 * XMath.Special.InverseRegularizedGammaP(_degreesOfFreedom / 2, x[d - 1]) / _degreesOfFreedom);
        }

        double product = 1;
        for (int k = 0; k < d; k++)
        {
            double shift = 0;
            for (int m = 0; m < k; m++)
            {
                shift += _factor[k + m * d] * _y[m];
            }

            double diagonal = _factor[k + k * d];
            // 最後の段は区間の確率のみを乗算し、w を使用しない
            double w = k < d - 1 ? x[k] : 0.5;
            double y = Truncate((Scaled(scale, _lower[k]) - shift) / diagonal, (Scaled(scale, _upper[k]) - shift) / diagonal, w, out double probability);
            product *= probability;
            if (product == 0)
            {
                return 0;
            }

            _y[k] = y;
        }

        return product;
    }

    // ±∞ はそのまま返し、scale == 0 のときに 0 * ∞ を生じさせない
    private static double Scaled(double scale, double limit) => double.IsInfinity(limit) ? limit : scale * limit;

    // 上側区間は対称性により Φ(-a) - Φ(-b) で計算し、桁落ちを避ける（Truncate と同一式）
    private static double Interval(double a, double b) =>
        a > 0 ? XMath.Special.NormalCdf(-a) - XMath.Special.NormalCdf(-b) : XMath.Special.NormalCdf(b) - XMath.Special.NormalCdf(a);

    // 区間 [a, b] の標準正規確率とその w 分位点を求める（丸めによる区間外・±∞ を防ぐためクランプする）
    private static double Truncate(double a, double b, double w, out double probability)
    {
        if (a > 0)
        {
            // 上側の区間は符号を反転した下側の区間 [-b, -a] の分位点を使う
            double low = XMath.Special.NormalCdf(-b);
            probability = XMath.Special.NormalCdf(-a) - low;
            double p = Math.Clamp(low + w * probability, SmallestNormal, MaxProbability);
            return Math.Clamp(-XMath.Special.InverseNormalCdf(p), a, b);
        }

        double lower = XMath.Special.NormalCdf(a);
        probability = XMath.Special.NormalCdf(b) - lower;
        double q = Math.Clamp(lower + w * probability, SmallestNormal, MaxProbability);
        return Math.Clamp(XMath.Special.InverseNormalCdf(q), a, b);
    }

    // 各段で中点条件付き確率が最小の変数を先頭に出すピボット付きコレスキー分解を行う（Genz 1992）
    private void Pivot(double[] scale, double[] location, double[] lower, double[] upper)
    {
        int d = _dimension;
        var s = (double[])scale.Clone();
        var a = new double[d];
        var b = new double[d];
        for (int i = 0; i < d; i++)
        {
            a[i] = lower[i] - location[i];
            b[i] = upper[i] - location[i];
        }

        var mid = new double[d];
        var factor = _factor;
        for (int k = 0; k < d; k++)
        {
            int pivot = k;
            double smallest = double.PositiveInfinity;
            for (int j = k; j < d; j++)
            {
                double shift = 0;
                double variance = s[j + j * d];
                for (int m = 0; m < k; m++)
                {
                    shift += factor[j + m * d] * mid[m];
                    variance -= factor[j + m * d] * factor[j + m * d];
                }

                double sd = Math.Sqrt(Math.Max(variance, MinimumVariance));
                double probability = Interval((a[j] - shift) / sd, (b[j] - shift) / sd);
                if (probability < smallest)
                {
                    smallest = probability;
                    pivot = j;
                }
            }

            if (pivot != k)
            {
                SwapVariables(s, factor, a, b, k, pivot, d);
            }

            double rowShift = 0;
            double rowVariance = s[k + k * d];
            for (int m = 0; m < k; m++)
            {
                rowShift += factor[k + m * d] * mid[m];
                rowVariance -= factor[k + m * d] * factor[k + m * d];
            }

            double diagonal = Math.Sqrt(Math.Max(rowVariance, MinimumVariance));
            factor[k + k * d] = diagonal;
            for (int j = k + 1; j < d; j++)
            {
                double dot = s[j + k * d];
                for (int m = 0; m < k; m++)
                {
                    dot -= factor[j + m * d] * factor[k + m * d];
                }

                factor[j + k * d] = dot / diagonal;
            }

            mid[k] = Truncate((a[k] - rowShift) / diagonal, (b[k] - rowShift) / diagonal, 0.5, out _);
        }

        // 上三角は 0 のまま（下三角因子のみを使用する）
        Array.Copy(a, _lower, d);
        Array.Copy(b, _upper, d);
    }

    private static void SwapVariables(double[] s, double[] factor, double[] a, double[] b, int k, int p, int d)
    {
        for (int i = 0; i < d; i++)
        {
            (s[i + k * d], s[i + p * d]) = (s[i + p * d], s[i + k * d]);
        }

        for (int j = 0; j < d; j++)
        {
            (s[k + j * d], s[p + j * d]) = (s[p + j * d], s[k + j * d]);
        }

        for (int m = 0; m < k; m++)
        {
            (factor[k + m * d], factor[p + m * d]) = (factor[p + m * d], factor[k + m * d]);
        }

        (a[k], a[p]) = (a[p], a[k]);
        (b[k], b[p]) = (b[p], b[k]);
    }

    private static double[] Copy(ReadOnlyVectorSpan<double> values)
    {
        var copy = new double[values.Length];
        for (int i = 0; i < copy.Length; i++)
        {
            copy[i] = values[i];
        }

        return copy;
    }
}
