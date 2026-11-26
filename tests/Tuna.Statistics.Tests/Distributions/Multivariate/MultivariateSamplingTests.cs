using Tuna.LinearAlgebra;
using Tuna.RandomSource;
using Tuna.Statistics.Distributions;

namespace Tuna.Tests.Statistics;

// 多変量分布の乱数の検証。シードを固定するので結果は決定的で、偶然の失敗は起きない。
// 乱数から作る1次元の量のうち、分布が正確に分かるもの（線形な射影、2次形式、周辺分布）について
// Kolmogorov–Smirnov 検定を有意水準 1e-4 で行い、平均を標準誤差の5倍以内で確かめる。
// 離散分布は同時分布の度数と周辺分布（二項・ベータ二項・負の二項分布）を χ² 検定で同じ有意水準で確かめる。
public class MultivariateSamplingTests
{
    private const int Count = 100_000;
    private const double SignificanceLevel = 1e-4;

    private static Pcg64DxsmRandom Random(int stream) => new(20261028, (ulong)stream);

    // 標本は double に丸めた値なので、裾の確率が大きい分布（α の小さいディリクレ分布の周辺は 1% ほどが 1 から ε 以内）では
    // 丸めで同じ値に集まる。標本 s の丸める前の値は隣の double の間 [s⁻, s⁺] にあるとして、D を上から抑える。
    private static void AssertFollows(double[] samples, Func<double, double> cdf, string context)
    {
        Array.Sort(samples);
        double d = 0;
        for (int i = 0; i < samples.Length; i++)
        {
            double below = cdf(Math.BitDecrement(samples[i]));
            double above = cdf(Math.BitIncrement(samples[i]));
            d = Math.Max(d, Math.Max(below - (double)i / samples.Length, (i + 1.0) / samples.Length - above));
        }

        double pValue = XMath.Special.KolmogorovSf(Math.Sqrt(samples.Length) * d);
        Assert.True(pValue > SignificanceLevel, $"{context}: KS D = {d}, p = {pValue}");
    }

    private static void AssertMean(ReadOnlySpan<double> samples, double mean, double variance, string context)
    {
        if (!double.IsFinite(mean) || !double.IsFinite(variance))
        {
            return;
        }

        double sum = 0;
        foreach (double v in samples)
        {
            sum += v;
        }

        double sampleMean = sum / samples.Length;
        double tolerance = 5 * Math.Sqrt(variance / samples.Length);
        Assert.True(Math.Abs(sampleMean - mean) <= tolerance, $"{context}: mean {sampleMean}, expected {mean} ± {tolerance}");
    }

    private static double Dot(ReadOnlySpan<double> a, ReadOnlyVectorSpan<double> x)
    {
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            sum += a[i] * x[i];
        }

        return sum;
    }

    /// <summary>aᵀSa（S は列優先の対称行列）</summary>
    private static double Quadratic(ReadOnlySpan<double> a, ReadOnlyMatrixSpan<double> s)
    {
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            for (int j = 0; j < a.Length; j++)
            {
                sum += a[i] * s[i, j] * a[j];
            }
        }

        return sum;
    }

    /// <summary>‖L⁻¹(x - μ)‖²（L は下三角）</summary>
    private static double Mahalanobis(ReadOnlyVectorSpan<double> x, ReadOnlyVectorSpan<double> mean, ReadOnlyMatrixSpan<double> factor)
    {
        int d = mean.Length;
        Span<double> z = stackalloc double[d];
        double sum = 0;
        for (int i = 0; i < d; i++)
        {
            double r = x[i] - mean[i];
            for (int k = 0; k < i; k++)
            {
                r -= factor[i, k] * z[k];
            }

            z[i] = r / factor[i, i];
            sum += z[i] * z[i];
        }

        return sum;
    }

    private static readonly double[][] Directions = [[1.0, -0.5, 2.0], [0.0, 1.0, 0.0]];

    // ---------------------------------------------------------------- 多変量正規分布・多変量 t 分布

    [Theory]
    [InlineData("MultivariateNormal")]
    [InlineData("MultivariateNormalFromFactor")]
    public void NormalProjectionsAndRadiusFollowTheirDistributions(string name)
    {
        var d = (MultivariateNormalDistribution)MultivariateContractTests.Vector(name);
        var samples = d.Sample(Random(1), Count);
        foreach (double[] a in Directions)
        {
            var projection = new NormalDistribution(Dot(a, d.Mean), Math.Sqrt(Quadratic(a, d.Covariance)));
            double[] values = Enumerable.Range(0, Count).Select(j => Dot(a, samples.Column(j))).ToArray();
            AssertMean(values, projection.Mean, projection.Variance, name);
            AssertFollows(values, projection.Cdf, $"{name} aᵀX");
        }

        // ‖L⁻¹(X - μ)‖² ~ χ²(d)
        var radius = new ChiSquaredDistribution(d.Dimension);
        double[] squared = Enumerable.Range(0, Count).Select(j => Mahalanobis(samples.Column(j), d.Mean, d.CholeskyFactor)).ToArray();
        AssertFollows(squared, radius.Cdf, $"{name} radius");
    }

    [Theory]
    [InlineData(3.5)]
    [InlineData(1.0)]
    [InlineData(0.4)]
    [InlineData(40.0)]
    public void StudentTProjectionsAndRadiusFollowTheirDistributions(double nu)
    {
        var d = new MultivariateStudentTDistribution(nu, MultivariateContractTests.Mean3, MultivariateContractTests.Square(MultivariateContractTests.Covariance3));
        var samples = d.Sample(Random(2), Count);
        foreach (double[] a in Directions)
        {
            var projection = new StudentTDistribution(nu, Dot(a, d.Location), Math.Sqrt(Quadratic(a, d.Scale)));
            double[] values = Enumerable.Range(0, Count).Select(j => Dot(a, samples.Column(j))).ToArray();
            AssertMean(values, projection.Mean, projection.Variance, $"ν = {nu}");
            AssertFollows(values, projection.Cdf, $"ν = {nu} aᵀX");
        }

        // ‖L⁻¹(X - μ)‖²/d ~ F(d, ν)
        var radius = new FDistribution(d.Dimension, nu);
        double[] squared = Enumerable.Range(0, Count)
            .Select(j => Mahalanobis(samples.Column(j), d.Location, d.CholeskyFactor) / d.Dimension).ToArray();
        AssertFollows(squared, radius.Cdf, $"ν = {nu} radius");
    }

    // ---------------------------------------------------------------- ウィシャート分布・逆ウィシャート分布

    private static double[] QuadraticForms(IMatrixContinuousDistribution d, ReadOnlySpan<double> a, int stream)
    {
        var random = Random(stream);
        var x = new Matrix<double>(d.Rows, d.Columns);
        double[] values = new double[Count];
        for (int k = 0; k < Count; k++)
        {
            d.Sample(random, x);
            values[k] = Quadratic(a, x);
        }

        return values;
    }

    [Theory]
    [InlineData(4.5)]
    [InlineData(2.2)]
    [InlineData(30.0)]
    public void WishartQuadraticFormsFollowChiSquared(double nu)
    {
        // aᵀXa/(aᵀΣa) ~ χ²(ν)（実数の ν でも成り立つ）
        var d = new WishartDistribution(nu, MultivariateContractTests.Square(MultivariateContractTests.Covariance3));
        var chiSquared = new ChiSquaredDistribution(nu);
        foreach (double[] a in Directions)
        {
            double scale = Quadratic(a, d.Scale);
            double[] values = QuadraticForms(d, a, 3).Select(v => v / scale).ToArray();
            AssertMean(values, chiSquared.Mean, chiSquared.Variance, $"ν = {nu}");
            AssertFollows(values, chiSquared.Cdf, $"ν = {nu}");
        }
    }

    [Theory]
    [InlineData(6.5)]
    [InlineData(2.3)]
    [InlineData(40.0)]
    public void InverseWishartQuadraticFormsFollowInverseGamma(double nu)
    {
        // aᵀXa ~ InverseGamma((ν - p + 1)/2, aᵀΨa/2)（AXAᵀ ~ W⁻¹(ν, AΨAᵀ) の (1, 1) 成分）
        var d = new InverseWishartDistribution(nu, MultivariateContractTests.Square(MultivariateContractTests.Covariance3));
        foreach (double[] a in Directions)
        {
            var inverseGamma = new InverseGammaDistribution(0.5 * (nu - d.Dimension + 1), 0.5 * Quadratic(a, d.Scale));
            double[] values = QuadraticForms(d, a, 4);
            AssertMean(values, inverseGamma.Mean, inverseGamma.Variance, $"ν = {nu}");
            AssertFollows(values, inverseGamma.Cdf, $"ν = {nu}");
        }
    }

    [Fact]
    public void WishartAndInverseWishartSampleMeansMatchTheMoments()
    {
        var scale = MultivariateContractTests.Square(MultivariateContractTests.Covariance3);
        foreach (var (d, mean, variance) in new (IMatrixContinuousDistribution, Matrix<double>, Matrix<double>)[]
        {
            (new WishartDistribution(4.5, scale), new WishartDistribution(4.5, scale).GetMean(), new WishartDistribution(4.5, scale).GetVariance()),
            (new InverseWishartDistribution(9.5, scale), new InverseWishartDistribution(9.5, scale).GetMean(),
                new InverseWishartDistribution(9.5, scale).GetVariance()),
        })
        {
            var random = Random(5);
            var x = new Matrix<double>(3, 3);
            var entries = new double[9][];
            for (int e = 0; e < 9; e++)
            {
                entries[e] = new double[Count];
            }

            for (int k = 0; k < Count; k++)
            {
                d.Sample(random, x);
                for (int e = 0; e < 9; e++)
                {
                    entries[e][k] = x[e % 3, e / 3];
                }
            }

            for (int e = 0; e < 9; e++)
            {
                AssertMean(entries[e], mean[e % 3, e / 3], variance[e % 3, e / 3], $"{d.GetType().Name}[{e % 3}, {e / 3}]");
            }
        }
    }

    // ---------------------------------------------------------------- ディリクレ分布

    [Theory]
    [InlineData(new double[] { 2, 3, 0.5 })]
    [InlineData(new double[] { 0.01, 0.02, 0.05 })]
    [InlineData(new double[] { 1e5, 2e5, 3e5, 4e5 })]
    [InlineData(new double[] { 1, 1 })]
    public void DirichletMarginalsFollowBeta(double[] alpha)
    {
        // Xᵢ ~ Beta(αᵢ, α₀ - αᵢ)、X₁ + X₂ ~ Beta(α₁ + α₂, α₀ - α₁ - α₂)
        var d = new DirichletDistribution(alpha);
        var samples = d.Sample(Random(6), Count);
        var covariance = d.GetCovariance();
        var mean = d.GetMean();
        double a0 = d.Concentration;
        for (int i = 0; i < alpha.Length; i++)
        {
            var beta = new BetaDistribution(alpha[i], a0 - alpha[i]);
            double[] values = samples.Row(i).ToArray();
            AssertMean(values, mean[i], covariance[i, i], $"α = [{string.Join(", ", alpha)}] X{i}");
            AssertFollows(values, beta.Cdf, $"α = [{string.Join(", ", alpha)}] X{i}");
        }

        if (alpha.Length > 2)
        {
            var aggregated = new BetaDistribution(alpha[0] + alpha[1], a0 - alpha[0] - alpha[1]);
            double[] values = Enumerable.Range(0, Count).Select(j => samples[0, j] + samples[1, j]).ToArray();
            AssertFollows(values, aggregated.Cdf, "X₁ + X₂");
        }
    }

    // ---------------------------------------------------------------- 行列正規分布

    [Fact]
    public void MatrixNormalProjectionsFollowNormal()
    {
        // aᵀvec(X) ~ N(aᵀvec(M), aᵀ(V ⊗ U)a)
        var d = (MatrixNormalDistribution)MultivariateContractTests.Matrix("MatrixNormal");
        int n = d.Rows;
        int p = d.Columns;
        double[][] directions = [[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0], [1.0, -0.5, 2.0, 0.3, 0.0, -1.0]];
        foreach (double[] a in directions)
        {
            double mean = 0;
            double variance = 0;
            for (int k = 0; k < n * p; k++)
            {
                mean += a[k] * d.Mean[k % n, k / n];
                for (int l = 0; l < n * p; l++)
                {
                    variance += a[k] * d.ColumnCovariance[k / n, l / n] * d.RowCovariance[k % n, l % n] * a[l];
                }
            }

            var random = Random(7);
            var x = new Matrix<double>(n, p);
            double[] values = new double[Count];
            for (int s = 0; s < Count; s++)
            {
                d.Sample(random, x);
                values[s] = Dot(a, x.AsVector());
            }

            var normal = new NormalDistribution(mean, Math.Sqrt(variance));
            AssertMean(values, mean, variance, $"a = [{string.Join(", ", a)}]");
            AssertFollows(values, normal.Cdf, $"a = [{string.Join(", ", a)}]");
        }
    }

    // ---------------------------------------------------------------- LKJ 分布

    [Theory]
    [InlineData(4, 1.5)]
    [InlineData(3, 0.3)]
    [InlineData(2, 1.0)]
    [InlineData(5, 1e3)]
    public void LkjOffDiagonalEntriesFollowBeta(int dimension, double eta)
    {
        // (rᵢⱼ + 1)/2 ~ Beta(η - 1 + d/2, η - 1 + d/2)。onion 法は行ごとに作るので、最初の行・最後の行・離れた組を見る
        var d = new LkjCorrelationDistribution(dimension, eta);
        double shape = eta - 1 + 0.5 * dimension;
        var beta = new BetaDistribution(shape, shape);
        (int Row, int Column)[] entries = [(1, 0), (dimension - 1, dimension - 2), (dimension - 1, 0)];
        var random = Random(8);
        var x = new Matrix<double>(dimension, dimension);
        var values = entries.Select(_ => new double[Count]).ToArray();
        for (int k = 0; k < Count; k++)
        {
            d.Sample(random, x);
            for (int e = 0; e < entries.Length; e++)
            {
                values[e][k] = 0.5 * (x[entries[e].Row, entries[e].Column] + 1);
            }
        }

        for (int e = 0; e < entries.Length; e++)
        {
            AssertMean(values[e], 0.5, beta.Variance, $"r{entries[e]}");
            AssertFollows(values[e], beta.Cdf, $"d = {dimension}, η = {eta}, r{entries[e]}");
        }
    }

    // ---------------------------------------------------------------- 離散の多変量分布

    // 期待度数が5以上の値はそれぞれ1つの区間、残り（範囲の外を含む）はまとめて1つの区間にして χ² 検定を有意水準 1e-4 で行う。
    // まとめた区間の期待度数が5未満なら、期待度数の最も小さい区間と合わせる
    private static void AssertChiSquared(List<(double Probability, long Observed)> bins, long restObserved, string context)
    {
        bins.Sort((a, b) => b.Probability.CompareTo(a.Probability));
        double rest = 1 - bins.Sum(b => b.Probability);
        while (bins.Count > 0 && Math.Max(rest, 0) * Count < 5)
        {
            rest += bins[^1].Probability;
            restObserved += bins[^1].Observed;
            bins.RemoveAt(bins.Count - 1);
        }

        double chiSquared = 0;
        foreach (var (probability, observed) in bins)
        {
            double expected = probability * Count;
            chiSquared += (observed - expected) * (observed - expected) / expected;
        }

        double restExpected = rest * Count;
        chiSquared += (restObserved - restExpected) * (restObserved - restExpected) / restExpected;
        int degreesOfFreedom = bins.Count;
        Assert.True(degreesOfFreedom > 0, $"{context}: no bins");
        double pValue = XMath.Special.RegularizedGammaQ(0.5 * degreesOfFreedom, 0.5 * chiSquared);
        Assert.True(pValue > SignificanceLevel, $"{context}: χ² = {chiSquared}, df = {degreesOfFreedom}, p = {pValue}");
    }

    /// <summary>0 以上の整数値の標本が確率関数 <paramref name="pmf"/> にしたがうかを、隣り合う値を期待度数が5以上になるまでまとめて χ² 検定する</summary>
    private static void AssertFollowsPmf(long[] samples, Func<long, double> pmf, long upper, string context)
    {
        var bins = new List<(double Probability, long Observed)>();
        var counts = samples.GroupBy(v => v).ToDictionary(g => g.Key, g => (long)g.Count());
        double probability = 0;
        long observed = 0;
        for (long k = 0; k <= upper; k++)
        {
            probability += pmf(k);
            observed += counts.GetValueOrDefault(k);
            if (probability * Count >= 5)
            {
                bins.Add((probability, observed));
                probability = 0;
                observed = 0;
            }
        }

        AssertChiSquared(bins, Count - bins.Sum(b => b.Observed), context);
    }

    private static double LogBetaBinomial(long n, double a, double b, long k) =>
        XMath.Special.LogBinomial(n, k) + XMath.Special.LogBeta(k + a, n - k + b) - XMath.Special.LogBeta(a, b);

    private static readonly string[] JointNames =
    [
        "Multinomial", "MultinomialZeroProbability", "DirichletMultinomial", "DirichletMultinomialSmall",
        "NegativeMultinomial", "NegativeMultinomialZeroProbability",
    ];

    public static TheoryData<string> DiscreteNames() => [.. JointNames];

    [Theory]
    [MemberData(nameof(DiscreteNames))]
    public void DiscreteJointFrequenciesFollowThePmf(string name)
    {
        // 台の点を成分の和の小さい順に、確率の和が 1 - 1e-12 に届くまで列挙する
        var d = MultivariateContractTests.Discrete(name);
        int k = d.Dimension;
        var samples = d.Sample(Random(20 + Array.IndexOf(JointNames, name)), Count);
        var counts = new Dictionary<string, long>();
        for (int j = 0; j < Count; j++)
        {
            string key = string.Join(",", samples.Column(j).ToArray());
            counts[key] = counts.GetValueOrDefault(key) + 1;
        }

        var bins = new List<(double Probability, long Observed)>();
        double total = 0;
        for (long sum = 0; total < 1 - 1e-12 && sum <= 10000; sum++)
        {
            foreach (long[] x in MultinomialTests.Compositions(sum, k))
            {
                double probability = d.Pmf(x);
                total += probability;
                if (probability * Count >= 5)
                {
                    string key = string.Join(",", x);
                    bins.Add((probability, counts.GetValueOrDefault(key)));
                }
            }
        }

        AssertChiSquared(bins, Count - bins.Sum(b => b.Observed), name);

        var mean = d.GetMean();
        var covariance = d.GetCovariance();
        for (int i = 0; i < k; i++)
        {
            AssertMean(samples.Row(i).ToArray().Select(v => (double)v).ToArray(), mean[i], covariance[i, i], $"{name} X{i}");
        }
    }

    [Fact]
    public void LargeMultinomialSamplesMatchTheMoments()
    {
        // n = 2⁴⁰ では同時分布を数えられないので、総和と各成分の平均を確かめる
        var d = MultivariateContractTests.Discrete("MultinomialLarge");
        var samples = d.Sample(Random(30), Count);
        for (int j = 0; j < Count; j++)
        {
            Assert.Equal(1L << 40, samples[0, j] + samples[1, j] + samples[2, j]);
        }

        var mean = d.GetMean();
        var covariance = d.GetCovariance();
        for (int i = 0; i < d.Dimension; i++)
        {
            AssertMean(samples.Row(i).ToArray().Select(v => (double)v).ToArray(), mean[i], covariance[i, i], $"X{i}");
        }
    }

    [Fact]
    public void MultinomialMarginalsFollowBinomial()
    {
        // Xᵢ ~ B(n, pᵢ)、X₁ + X₂ ~ B(n, p₁ + p₂)
        var d = new MultinomialDistribution(1000, 0.1, 0.6, 0.3);
        var samples = d.Sample(Random(31), Count);
        for (int i = 0; i < 3; i++)
        {
            var binomial = new BinomialDistribution(1000, d.Probabilities[i]);
            AssertFollowsPmf(samples.Row(i).ToArray(), binomial.Pmf, 1000, $"X{i}");
        }

        var aggregated = new BinomialDistribution(1000, 0.7);
        AssertFollowsPmf(Enumerable.Range(0, Count).Select(j => samples[0, j] + samples[1, j]).ToArray(), aggregated.Pmf, 1000, "X₁ + X₂");
    }

    [Fact]
    public void DirichletMultinomialMarginalsFollowBetaBinomial()
    {
        // Xᵢ ~ BetaBin(n, αᵢ, α₀ - αᵢ)、X₁ + X₂ ~ BetaBin(n, α₁ + α₂, α₀ - α₁ - α₂)
        double[] alpha = [2, 3, 0.5];
        var d = new DirichletMultinomialDistribution(200, alpha);
        var samples = d.Sample(Random(32), Count);
        for (int i = 0; i < 3; i++)
        {
            double a = alpha[i];
            double b = d.Concentration - a;
            AssertFollowsPmf(samples.Row(i).ToArray(), k => Math.Exp(LogBetaBinomial(200, a, b, k)), 200, $"X{i}");
        }

        AssertFollowsPmf(Enumerable.Range(0, Count).Select(j => samples[0, j] + samples[1, j]).ToArray(),
            k => Math.Exp(LogBetaBinomial(200, 5, 0.5, k)), 200, "X₁ + X₂");
    }

    [Theory]
    [InlineData(2.5, 0.4, new double[] { 0.25, 0.35 })]
    [InlineData(50.0, 0.7, new double[] { 0.1, 0.1, 0.1 })]
    [InlineData(0.3, 0.05, new double[] { 0.9, 0.05 })]
    public void NegativeMultinomialMarginalsFollowNegativeBinomial(double r, double p0, double[] p)
    {
        // Xᵢ ~ NB(r, p₀/(p₀ + pᵢ))、ΣXᵢ ~ NB(r, p₀)
        var d = new NegativeMultinomialDistribution(r, p0, p);
        var samples = d.Sample(Random(33), Count);
        for (int i = 0; i < p.Length; i++)
        {
            var marginal = new NegativeBinomialDistribution(r, p0 / (p0 + p[i]));
            AssertFollowsPmf(samples.Row(i).ToArray(), marginal.Pmf, marginal.InverseSf(1e-12), $"X{i}");
        }

        var total = new NegativeBinomialDistribution(r, p0);
        long[] sums = Enumerable.Range(0, Count).Select(j => Enumerable.Range(0, p.Length).Sum(i => samples[i, j])).ToArray();
        AssertFollowsPmf(sums, total.Pmf, total.InverseSf(1e-12), "ΣXᵢ");
    }
}
