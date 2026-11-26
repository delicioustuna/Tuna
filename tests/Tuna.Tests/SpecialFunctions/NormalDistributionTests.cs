using System.Numerics;
using Tuna;

namespace Tuna.Tests;

// mpmath 1.4.1による独立参照値。binary64入力を正確に変換し、
// 90/160桁（正規分布は490/560桁）でbinary64への丸め一致を確認。
// 生成式と選定理由: tools/SpecialFunctionReferences/generate_remaining_accuracy_references.py
// 通常テストはC#の固定値と対象APIの直接呼び出しだけで完結する。
[Trait("SpecialFunction", "NormalDistribution")]
public class NormalDistributionTests
{

    // 中央・erfc切替の隣接値・正規化数と非正規数の裾。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(-40.0, 0.0, 0.0)]
    [InlineData(-10.0, 7.694598626706419e-23, 0.0)]
    [InlineData(-2.0, 0.05399096651318805, 0.0)]
    [InlineData(0.0, 0.3989422804014327, 0.0)]
    [InlineData(0.5, 0.35206532676429947, 0.0)]
    [InlineData(2.0, 0.05399096651318805, 0.0)]
    [InlineData(10.0, 7.694598626706419e-23, 0.0)]
    [InlineData(38.0, 1.097221052e-314, 1e-323)]
    [InlineData(1.7677669529663687, 0.08362284478816655, 0.0)]
    [InlineData(1.7677669529663689, 0.08362284478816652, 0.0)]
    [InlineData(1.767766952966369, 0.08362284478816649, 0.0)]
    [InlineData(37.5, 1.7282337322841054e-306, 0.0)]
    [InlineData(38.5, 5.4e-323, 1e-323)]
    public void NormalPdf_MatchesIndependentReferences(double x, double expected, double absoluteTolerance)
    {
        double actual = XMath.Special.NormalPdf(x);
        if (expected == 0 || double.IsInfinity(expected))
        {
            Assert.Equal(expected, actual);
        }
        else
        {
            SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 2e-12, absoluteTolerance: absoluteTolerance);
            if (Math.Abs(expected) < 2.2250738585072014e-308)
            {
                Assert.Equal(Math.Sign(expected), Math.Sign(actual));
            }
        }
    }

    // 各引数のNaNを独立に検査。
    [Theory]
    [InlineData(double.NaN)]
    public void NormalPdf_RejectsNaN(double x)
    {
        double actual = XMath.Special.NormalPdf(x);
        Assert.True(double.IsNaN(actual));
    }

    // 中央・erfc切替の隣接値・正規化数と非正規数の裾。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(-40.0, 0.0)]
    [InlineData(-10.0, 7.619853024160525e-24)]
    [InlineData(-2.0, 0.02275013194817921)]
    [InlineData(0.0, 0.5)]
    [InlineData(0.5, 0.6914624612740131)]
    [InlineData(2.0, 0.9772498680518208)]
    [InlineData(10.0, 1.0)]
    [InlineData(38.0, 1.0)]
    [InlineData(1.7677669529663687, 0.9614500641282291)]
    [InlineData(1.7677669529663689, 0.9614500641282291)]
    [InlineData(1.767766952966369, 0.9614500641282291)]
    [InlineData(37.5, 1.0)]
    [InlineData(38.5, 1.0)]
    public void NormalCdf_MatchesIndependentReferences(double x, double expected)
    {
        double actual = XMath.Special.NormalCdf(x);
        if (expected == 0 || double.IsInfinity(expected))
        {
            Assert.Equal(expected, actual);
        }
        else
        {
            SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 2e-12);
            if (Math.Abs(expected) < 2.2250738585072014e-308)
            {
                Assert.Equal(Math.Sign(expected), Math.Sign(actual));
            }
        }
    }

    // 各引数のNaNを独立に検査。
    [Theory]
    [InlineData(double.NaN)]
    public void NormalCdf_RejectsNaN(double x)
    {
        double actual = XMath.Special.NormalCdf(x);
        Assert.True(double.IsNaN(actual));
    }

    // 中央・erfc切替の隣接値・正規化数と非正規数の裾。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(-40.0, 1.0, 0.0)]
    [InlineData(-10.0, 1.0, 0.0)]
    [InlineData(-2.0, 0.9772498680518208, 0.0)]
    [InlineData(0.0, 0.5, 0.0)]
    [InlineData(0.5, 0.3085375387259869, 0.0)]
    [InlineData(2.0, 0.02275013194817921, 0.0)]
    [InlineData(10.0, 7.619853024160525e-24, 0.0)]
    [InlineData(38.0, 2.88542835e-316, 1e-323)]
    [InlineData(1.7677669529663687, 0.0385499358717709, 0.0)]
    [InlineData(1.7677669529663689, 0.03854993587177088, 0.0)]
    [InlineData(1.767766952966369, 0.038549935871770864, 0.0)]
    [InlineData(37.5, 4.605353009581955e-308, 0.0)]
    [InlineData(38.5, 0.0, 0.0)]
    public void NormalSf_MatchesIndependentReferences(double x, double expected, double absoluteTolerance)
    {
        double actual = XMath.Special.NormalSf(x);
        if (expected == 0 || double.IsInfinity(expected))
        {
            Assert.Equal(expected, actual);
        }
        else
        {
            SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 2e-12, absoluteTolerance: absoluteTolerance);
            if (Math.Abs(expected) < 2.2250738585072014e-308)
            {
                Assert.Equal(Math.Sign(expected), Math.Sign(actual));
            }
        }
    }

    // 各引数のNaNを独立に検査。
    [Theory]
    [InlineData(double.NaN)]
    public void NormalSf_RejectsNaN(double x)
    {
        double actual = XMath.Special.NormalSf(x);
        Assert.True(double.IsNaN(actual));
    }

    // 中央・erfc切替の隣接値・正規化数と非正規数の裾。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(-40.0, -800.9189385332047, 2e-12)]
    [InlineData(-10.0, -50.918938533204674, 2e-12)]
    [InlineData(-2.0, -2.9189385332046727, 2e-12)]
    [InlineData(0.0, -0.9189385332046728, 0.0)]
    [InlineData(0.5, -1.0439385332046727, 2e-12)]
    [InlineData(2.0, -2.9189385332046727, 2e-12)]
    [InlineData(10.0, -50.918938533204674, 2e-12)]
    [InlineData(38.0, -722.9189385332047, 2e-12)]
    [InlineData(1.7677669529663687, -2.4814385332046727, 2e-12)]
    [InlineData(1.7677669529663689, -2.4814385332046727, 2e-12)]
    [InlineData(1.767766952966369, -2.481438533204673, 2e-12)]
    [InlineData(37.5, -704.0439385332047, 2e-12)]
    [InlineData(38.5, -742.0439385332047, 2e-12)]
    public void NormalLogPdf_MatchesIndependentReferences(double x, double expected, double absoluteTolerance)
    {
        double actual = XMath.Special.NormalLogPdf(x);
        if (expected == 0 || double.IsInfinity(expected))
        {
            Assert.Equal(expected, actual);
        }
        else
        {
            SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 2e-12, absoluteTolerance: absoluteTolerance);
            if (Math.Abs(expected) < 2.2250738585072014e-308)
            {
                Assert.Equal(Math.Sign(expected), Math.Sign(actual));
            }
        }
    }

    // 各引数のNaNを独立に検査。
    [Theory]
    [InlineData(double.NaN)]
    public void NormalLogPdf_RejectsNaN(double x)
    {
        double actual = XMath.Special.NormalLogPdf(x);
        Assert.True(double.IsNaN(actual));
    }

    // 中央・erfc切替の隣接値・正規化数と非正規数の裾。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(-40.0, -804.6084420137538, 2e-12)]
    [InlineData(-10.0, -53.23128515051247, 2e-12)]
    [InlineData(-2.0, -3.783184333682032, 2e-12)]
    [InlineData(0.0, -0.6931471805599453, 0.0)]
    [InlineData(0.5, -0.3689464152886564, 0.0)]
    [InlineData(2.0, -0.02301290932896349, 0.0)]
    [InlineData(10.0, -7.619853024160525e-24, 0.0)]
    [InlineData(38.0, -2.88542835e-316, 1e-323)]
    [InlineData(1.7677669529663687, -0.03931265068604848, 0.0)]
    [InlineData(1.7677669529663689, -0.039312650686048466, 0.0)]
    [InlineData(1.767766952966369, -0.039312650686048445, 0.0)]
    [InlineData(37.5, -4.605353009581955e-308, 0.0)]
    [InlineData(38.5, -0.0, 0.0)]
    public void NormalLogCdf_MatchesIndependentReferences(double x, double expected, double absoluteTolerance)
    {
        double actual = XMath.Special.NormalLogCdf(x);
        if (expected == 0 || double.IsInfinity(expected))
        {
            Assert.Equal(expected, actual);
        }
        else
        {
            SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 2e-12, absoluteTolerance: absoluteTolerance);
            if (Math.Abs(expected) < 2.2250738585072014e-308)
            {
                Assert.Equal(Math.Sign(expected), Math.Sign(actual));
            }
        }
    }

    // 各引数のNaNを独立に検査。
    [Theory]
    [InlineData(double.NaN)]
    public void NormalLogCdf_RejectsNaN(double x)
    {
        double actual = XMath.Special.NormalLogCdf(x);
        Assert.True(double.IsNaN(actual));
    }

    // 中央・erfc切替の隣接値・正規化数と非正規数の裾。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(-40.0, -0.0, 0.0)]
    [InlineData(-10.0, -7.619853024160525e-24, 0.0)]
    [InlineData(-2.0, -0.02301290932896349, 0.0)]
    [InlineData(0.0, -0.6931471805599453, 0.0)]
    [InlineData(0.5, -1.1759117615936185, 2e-12)]
    [InlineData(2.0, -3.783184333682032, 2e-12)]
    [InlineData(10.0, -53.23128515051247, 2e-12)]
    [InlineData(38.0, -726.5572160188201, 2e-12)]
    [InlineData(1.7677669529663687, -3.2558008424820706, 2e-12)]
    [InlineData(1.7677669529663689, -3.255800842482071, 2e-12)]
    [InlineData(1.767766952966369, -3.2558008424820715, 2e-12)]
    [InlineData(37.5, -707.6689893175072, 2e-12)]
    [InlineData(38.5, -745.695270290411, 2e-12)]
    public void NormalLogSf_MatchesIndependentReferences(double x, double expected, double absoluteTolerance)
    {
        double actual = XMath.Special.NormalLogSf(x);
        if (expected == 0 || double.IsInfinity(expected))
        {
            Assert.Equal(expected, actual);
        }
        else
        {
            SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 2e-12, absoluteTolerance: absoluteTolerance);
            if (Math.Abs(expected) < 2.2250738585072014e-308)
            {
                Assert.Equal(Math.Sign(expected), Math.Sign(actual));
            }
        }
    }

    // 各引数のNaNを独立に検査。
    [Theory]
    [InlineData(double.NaN)]
    public void NormalLogSf_RejectsNaN(double x)
    {
        double actual = XMath.Special.NormalLogSf(x);
        Assert.True(double.IsNaN(actual));
    }

    // 独立した高精度求根。最小非正規確率を含む。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(5e-324, -38.467405617144344)]
    [InlineData(1e-300, -37.0470962993612)]
    [InlineData(1e-30, -11.464024688443615)]
    [InlineData(1e-10, -6.361340902404057)]
    [InlineData(0.01, -2.326347874040841)]
    [InlineData(0.25, -0.6744897501960817)]
    [InlineData(0.5, 0.0)]
    [InlineData(0.9, 1.2815515655446006)]
    [InlineData(0.9999999999999999, 8.209536151601387)]
    public void InverseNormalCdf_MatchesIndependentReferences(double p, double expected)
    {
        double actual = XMath.Special.InverseNormalCdf(p);
        if (expected == 0 || double.IsInfinity(expected))
        {
            Assert.Equal(expected, actual);
        }
        else
        {
            SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 2e-12);
            if (Math.Abs(expected) < 2.2250738585072014e-308)
            {
                Assert.Equal(Math.Sign(expected), Math.Sign(actual));
            }
        }
    }

    // 各引数のNaNを独立に検査。
    [Theory]
    [InlineData(double.NaN)]
    public void InverseNormalCdf_RejectsNaN(double p)
    {
        double actual = XMath.Special.InverseNormalCdf(p);
        Assert.True(double.IsNaN(actual));
    }

    // 独立した高精度求根。最小非正規確率を含む。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(5e-324, 38.467405617144344)]
    [InlineData(1e-300, 37.0470962993612)]
    [InlineData(1e-30, 11.464024688443615)]
    [InlineData(1e-10, 6.361340902404057)]
    [InlineData(0.01, 2.326347874040841)]
    [InlineData(0.25, 0.6744897501960817)]
    [InlineData(0.5, 0.0)]
    [InlineData(0.9, -1.2815515655446006)]
    [InlineData(0.9999999999999999, -8.209536151601387)]
    public void InverseNormalSf_MatchesIndependentReferences(double p, double expected)
    {
        double actual = XMath.Special.InverseNormalSf(p);
        if (expected == 0 || double.IsInfinity(expected))
        {
            Assert.Equal(expected, actual);
        }
        else
        {
            SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 2e-12);
            if (Math.Abs(expected) < 2.2250738585072014e-308)
            {
                Assert.Equal(Math.Sign(expected), Math.Sign(actual));
            }
        }
    }

    // 各引数のNaNを独立に検査。
    [Theory]
    [InlineData(double.NaN)]
    public void InverseNormalSf_RejectsNaN(double p)
    {
        double actual = XMath.Special.InverseNormalSf(p);
        Assert.True(double.IsNaN(actual));
    }

    // 対数入力の独立求根・線形確率のアンダーフロー境界。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(-1e-300, 37.0470962993612, 0.0)]
    [InlineData(-0.1, 1.3096177994584932, 0.0)]
    [InlineData(-0.6931471805599453, 2.9064941568900345e-17, 2e-15)]
    [InlineData(-2.0, -1.1015196284987503, 0.0)]
    [InlineData(-100.0, -13.888476033003887, 0.0)]
    [InlineData(-708.3964185322642, -37.5193793471445, 0.0)]
    [InlineData(-708.3964185322641, -37.5193793471445, 0.0)]
    [InlineData(-708.396418532264, -37.51937934714449, 0.0)]
    [InlineData(-1000.0, -44.6157477319694, 0.0)]
    [InlineData(-1000000.0, -1414.2077829910174, 0.0)]
    [InlineData(-1e+100, -1.414213562373095e+50, 0.0)]
    public void InverseNormalLogCdf_MatchesIndependentReferences(double logP, double expected, double absoluteTolerance)
    {
        double actual = XMath.Special.InverseNormalLogCdf(logP);
        if (expected == 0 || double.IsInfinity(expected))
        {
            Assert.Equal(expected, actual);
        }
        else
        {
            SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 2e-12, absoluteTolerance: absoluteTolerance);
            if (Math.Abs(expected) < 2.2250738585072014e-308)
            {
                Assert.Equal(Math.Sign(expected), Math.Sign(actual));
            }
        }
    }

    // 各引数のNaNを独立に検査。
    [Theory]
    [InlineData(double.NaN)]
    public void InverseNormalLogCdf_RejectsNaN(double logP)
    {
        double actual = XMath.Special.InverseNormalLogCdf(logP);
        Assert.True(double.IsNaN(actual));
    }

    // 対数入力の独立求根・線形確率のアンダーフロー境界。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(-1e-300, -37.0470962993612, 0.0)]
    [InlineData(-0.1, -1.3096177994584932, 0.0)]
    [InlineData(-0.6931471805599453, -2.9064941568900345e-17, 2e-15)]
    [InlineData(-2.0, 1.1015196284987503, 0.0)]
    [InlineData(-100.0, 13.888476033003887, 0.0)]
    [InlineData(-708.3964185322642, 37.5193793471445, 0.0)]
    [InlineData(-708.3964185322641, 37.5193793471445, 0.0)]
    [InlineData(-708.396418532264, 37.51937934714449, 0.0)]
    [InlineData(-1000.0, 44.6157477319694, 0.0)]
    [InlineData(-1000000.0, 1414.2077829910174, 0.0)]
    [InlineData(-1e+100, 1.414213562373095e+50, 0.0)]
    public void InverseNormalLogSf_MatchesIndependentReferences(double logP, double expected, double absoluteTolerance)
    {
        double actual = XMath.Special.InverseNormalLogSf(logP);
        if (expected == 0 || double.IsInfinity(expected))
        {
            Assert.Equal(expected, actual);
        }
        else
        {
            SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 2e-12, absoluteTolerance: absoluteTolerance);
            if (Math.Abs(expected) < 2.2250738585072014e-308)
            {
                Assert.Equal(Math.Sign(expected), Math.Sign(actual));
            }
        }
    }

    // 各引数のNaNを独立に検査。
    [Theory]
    [InlineData(double.NaN)]
    public void InverseNormalLogSf_RejectsNaN(double logP)
    {
        double actual = XMath.Special.InverseNormalLogSf(logP);
        Assert.True(double.IsNaN(actual));
    }

    [Fact]
    public void EndpointsAndInverseDomains()
    {
        Assert.Equal(0, XMath.Special.NormalCdf(double.NegativeInfinity));
        Assert.Equal(1, XMath.Special.NormalCdf(double.PositiveInfinity));
        Assert.Equal(1, XMath.Special.NormalSf(double.NegativeInfinity));
        Assert.Equal(0, XMath.Special.NormalSf(double.PositiveInfinity));
        Assert.Equal(0, XMath.Special.NormalPdf(double.PositiveInfinity));
        Assert.Equal(double.NegativeInfinity, XMath.Special.NormalLogPdf(double.PositiveInfinity));
        Assert.Equal(double.NegativeInfinity, XMath.Special.NormalLogCdf(double.NegativeInfinity));
        Assert.Equal(0, XMath.Special.NormalLogCdf(double.PositiveInfinity));
        Assert.Equal(0, XMath.Special.NormalLogSf(double.NegativeInfinity));
        Assert.Equal(double.NegativeInfinity, XMath.Special.NormalLogSf(double.PositiveInfinity));
        Assert.Equal(double.NegativeInfinity, XMath.Special.InverseNormalCdf(0));
        Assert.Equal(double.PositiveInfinity, XMath.Special.InverseNormalCdf(1));
        Assert.Equal(double.PositiveInfinity, XMath.Special.InverseNormalSf(0));
        Assert.Equal(double.NegativeInfinity, XMath.Special.InverseNormalSf(1));
        Assert.Equal(double.PositiveInfinity, XMath.Special.InverseNormalLogCdf(0));
        Assert.Equal(double.NegativeInfinity, XMath.Special.InverseNormalLogCdf(double.NegativeInfinity));
        Assert.Equal(double.NegativeInfinity, XMath.Special.InverseNormalLogSf(0));
        Assert.Equal(double.PositiveInfinity, XMath.Special.InverseNormalLogSf(double.NegativeInfinity));
        Assert.True(double.IsNaN(XMath.Special.InverseNormalCdf(-.1)));
        Assert.True(double.IsNaN(XMath.Special.InverseNormalSf(1.1)));
        Assert.True(double.IsNaN(XMath.Special.InverseNormalLogCdf(.1)));
        Assert.True(double.IsNaN(XMath.Special.InverseNormalLogSf(.1)));
    }

    [Fact]
    public void ProbabilitiesAreBoundedMonotoneAndComplementary()
    {
        double previous = 0;
        foreach (double x in new[] { -38.0, -10.0, -2.0, 0.0, 2.0, 10.0, 38.0 })
        {
            double cdf = XMath.Special.NormalCdf(x), sf = XMath.Special.NormalSf(x);
            Assert.InRange(cdf, 0, 1);
            Assert.InRange(sf, 0, 1);
            Assert.True(cdf >= previous);
            SpecialFunctionAssert.Close(1, cdf + sf, relativeTolerance: 2e-15);
            Assert.Equal(sf, XMath.Special.NormalCdf(-x));
            previous = cdf;
        }
    }

    // 区分多項式近似の区間境界および誤差が大きくなりやすい点を ULP 単位で検証する。
    // 参照値は mpmath 1.4.1（45 桁・70 桁で binary64 丸め一致）。許容誤差は CDF が 4 ULP（Math.Exp の実装差を考慮）、分位点が 3 ULP。
    [Trait("Category", "AuditRegression")]
    [Theory]
    [InlineData(-1.6096653226677529, 0.053735469617235)]
    [InlineData(-0.66, 0.25462691467133614)]
    [InlineData(-0.6599999999999999, 0.2546269146713362)]
    [InlineData(0.3, 0.6179114221889527)]
    [InlineData(-2.8981070611724427, 0.0018771120574024877)]
    [InlineData(-7.963489042505749, 8.362741496987539e-16)]
    [InlineData(-23.563514409759556, 4.562769255132382e-123)]
    [InlineData(-37.48382460596923, 8.449409897635463e-308)]
    [InlineData(1.5, 0.9331927987311419)]
    public void NormalCdf_PiecewiseApproximation_WithinUlps(double x, double expected)
    {
        SpecialFunctionAssert.WithinUlps(expected, XMath.Special.NormalCdf(x), 4);
    }

    [Trait("Category", "AuditRegression")]
    [Theory]
    [InlineData(1.6782393420959316, 0.046650192137587525)]
    [InlineData(0.66, 0.25462691467133614)]
    [InlineData(0.6599999999999999, 0.2546269146713362)]
    [InlineData(2.44823142263017, 0.007177970732395515)]
    [InlineData(23.6417667934584, 7.1724573505707844e-124)]
    [InlineData(30.450976433453874, 5.813855512289695e-204)]
    public void NormalSf_PiecewiseApproximation_WithinUlps(double x, double expected)
    {
        SpecialFunctionAssert.WithinUlps(expected, XMath.Special.NormalSf(x), 4);
    }

    [Trait("Category", "AuditRegression")]
    [Theory]
    [InlineData(0.054430634577080084, -1.6033324804517912)]
    [InlineData(0.9371246600820449, 1.5310757197016398)]
    [InlineData(0.24999999999999997, -0.6744897501960818)]
    [InlineData(0.25, -0.6744897501960817)]
    [InlineData(0.75, 0.6744897501960817)]
    [InlineData(0.7500000000000001, 0.674489750196082)]
    [InlineData(0.105, -1.2535654384704507)]
    [InlineData(0.125, -1.150349380376008)]
    [InlineData(0.5000000000000001, 2.782916424671767e-16)]
    [InlineData(0.9, 1.2815515655446006)]
    [InlineData(1e-10, -6.361340902404057)]
    [InlineData(1e-300, -37.0470962993612)]
    [InlineData(2.2250738585072014e-308, -37.5193793471445)]
    [InlineData(1e-310, -37.663060331949524)]
    [InlineData(0.9999999999999999, 8.209536151601387)]
    public void InverseNormalCdf_HalleyRefinement_WithinUlps(double p, double expected)
    {
        SpecialFunctionAssert.WithinUlps(expected, XMath.Special.InverseNormalCdf(p), 3);
        SpecialFunctionAssert.WithinUlps(-expected, XMath.Special.InverseNormalSf(p), 3);
    }

    // 1/4 < p < 3/4 は p を経由せず 1/2 - p = -expm1(ln p + ln 2)/2 から求める（経由すると p の丸め誤差が十数倍に拡大していた）
    [Trait("Category", "AuditRegression")]
    [Theory]
    [InlineData(-0.6456754017809202, 0.060969634087244866)]
    [InlineData(-0.6774095109651682, 0.01988157772998456)]
    [InlineData(-1e-10, 6.361340902411735)]
    [InlineData(-2.0, -1.1015196284987503)]
    [InlineData(-700.0, -37.295079632647415)]
    [InlineData(-800.0, -39.88469483825668)]
    [InlineData(-1e5, -447.1978936785251)]
    public void InverseNormalLogCdf_WithinUlps(double logP, double expected)
    {
        SpecialFunctionAssert.WithinUlps(expected, XMath.Special.InverseNormalLogCdf(logP), 3);
        SpecialFunctionAssert.WithinUlps(-expected, XMath.Special.InverseNormalLogSf(logP), 3);
    }

    // 小さい上側確率と 1 の直前における分位点検証（erfinv を 500/560 桁で評価し binary64 丸め一致を確認）。
    [Trait("Category", "AuditRegression")]
    [Trait("Category", "CoreAudit20261102")]
    [Theory]
    [InlineData(1e-320, 38.26912534303265)]
    [InlineData(1e-20, 9.262340089798407)]
    [InlineData(0.9999999999999999, -8.209536151601387)]
    public void InverseNormalSf_SmallTailAndEndpointNeighbor_FixedReferences(double p, double expected)
    {
        double actual = XMath.Special.InverseNormalSf(p);
        if (double.IsInfinity(expected) || expected == 0)
        {
            Assert.Equal(expected, actual);
        }
        else
        {
            SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 2e-12,
                absoluteTolerance: Math.Abs(expected) < 2.2250738585072014e-308 ? 2 * double.Epsilon : 0);
        }
    }
}
