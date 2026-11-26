using System.Numerics;
using Tuna;

namespace Tuna.Tests;

// mpmath 1.4.1による独立参照値。binary64入力を正確に変換し、
// 90/160桁（正規分布は490/560桁）でbinary64への丸め一致を確認。
// 生成式と選定理由: tools/SpecialFunctionReferences/generate_remaining_accuracy_references.py
// 通常テストはC#の固定値と対象APIの直接呼び出しだけで完結する。
[Trait("SpecialFunction", "MarcumQ")]
public class MarcumQTests
{

    // 独立した高精度Poisson混合・級数/積分の切替。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(1.0, 0.0, 0.1, 0.24817036595415073)]
    [InlineData(2.0, 0.0, 4.0, 0.8646647167633873)]
    [InlineData(0.5, 0.2, 0.01, 0.26529645963100434)]
    [InlineData(4.0, 2.0, 3.0, 0.24627270146198138)]
    [InlineData(10.0, 20.0, 30.0, 0.5374020277651606)]
    [InlineData(3.0, 80.0, 85.0, 0.5655017570235729)]
    [InlineData(5.0, 100.0, 10.0, 3.5594767402899395e-13)]
    [InlineData(5.0, 100.0, 300.0, 0.999999999999626)]
    [InlineData(2.0, 59.99999999999999, 70.0, 0.7117075530468674)]
    [InlineData(2.0, 60.0, 70.0, 0.7117075530468673)]
    [InlineData(2.0, 60.00000000000001, 70.0, 0.7117075530468672)]
    public void NoncentralChiSquaredCdf_MatchesIndependentReferences(double k, double lambda, double x, double expected)
    {
        double actual = XMath.Special.NoncentralChiSquaredCdf(k, lambda, x);
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
    [InlineData(double.NaN, 0.0, 0.1)]
    [InlineData(1.0, double.NaN, 0.1)]
    [InlineData(1.0, 0.0, double.NaN)]
    public void NoncentralChiSquaredCdf_RejectsNaN(double k, double lambda, double x)
    {
        double actual = XMath.Special.NoncentralChiSquaredCdf(k, lambda, x);
        Assert.True(double.IsNaN(actual));
    }

    // 独立した高精度Poisson混合・級数/積分の切替。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(1.0, 0.0, 0.1, 0.7518296340458492)]
    [InlineData(2.0, 0.0, 4.0, 0.1353352832366127)]
    [InlineData(0.5, 0.2, 0.01, 0.7347035403689957)]
    [InlineData(4.0, 2.0, 3.0, 0.7537272985380186)]
    [InlineData(10.0, 20.0, 30.0, 0.4625979722348395)]
    [InlineData(3.0, 80.0, 85.0, 0.43449824297642714)]
    [InlineData(5.0, 100.0, 10.0, 0.9999999999996441)]
    [InlineData(5.0, 100.0, 300.0, 3.7408210485244424e-13)]
    [InlineData(2.0, 59.99999999999999, 70.0, 0.28829244695313255)]
    [InlineData(2.0, 60.0, 70.0, 0.2882924469531327)]
    [InlineData(2.0, 60.00000000000001, 70.0, 0.2882924469531329)]
    public void NoncentralChiSquaredSf_MatchesIndependentReferences(double k, double lambda, double x, double expected)
    {
        double actual = XMath.Special.NoncentralChiSquaredSf(k, lambda, x);
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
    [InlineData(double.NaN, 0.0, 0.1)]
    [InlineData(1.0, double.NaN, 0.1)]
    [InlineData(1.0, 0.0, double.NaN)]
    public void NoncentralChiSquaredSf_RejectsNaN(double k, double lambda, double x)
    {
        double actual = XMath.Special.NoncentralChiSquaredSf(k, lambda, x);
        Assert.True(double.IsNaN(actual));
    }

    // Marcumの定義と非心χ²の対応・級数切替。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(1.0, 0.0, 2.0, 0.1353352832366127)]
    [InlineData(0.5, 1.0, 2.0, 0.16000515196308715)]
    [InlineData(2.0, 3.0, 4.0, 0.2866542093655808)]
    [InlineData(5.0, 10.0, 11.0, 0.2841518664985058)]
    [InlineData(2.0, 20.0, 10.0, 1.0)]
    [InlineData(2.0, 10.0, 30.0, 1.4325955221119185e-88)]
    [InlineData(2.0, 7.745966692414833, 8.0, 0.4748890790764176)]
    [InlineData(2.0, 7.745966692414834, 8.0, 0.4748890790764179)]
    [InlineData(2.0, 7.745966692414835, 8.0, 0.47488907907641825)]
    public void MarcumQ_MatchesIndependentReferences(double mu, double x, double y, double expected)
    {
        double actual = XMath.Special.MarcumQ(mu, x, y);
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
    [InlineData(double.NaN, 0.0, 2.0)]
    [InlineData(1.0, double.NaN, 2.0)]
    [InlineData(1.0, 0.0, double.NaN)]
    public void MarcumQ_RejectsNaN(double mu, double x, double y)
    {
        double actual = XMath.Special.MarcumQ(mu, x, y);
        Assert.True(double.IsNaN(actual));
    }

    [Fact]
    public void EndpointsAndDomains()
    {
        Assert.Equal(1, XMath.Special.MarcumQ(2, 3, 0));
        Assert.Equal(0, XMath.Special.MarcumQ(2, 3, double.PositiveInfinity));
        Assert.Equal(1, XMath.Special.MarcumQ(2, double.PositiveInfinity, 3));
        Assert.True(double.IsNaN(XMath.Special.MarcumQ(2, double.PositiveInfinity, double.PositiveInfinity)));
        Assert.True(double.IsNaN(XMath.Special.MarcumQ(0, 1, 1)));
        Assert.True(double.IsNaN(XMath.Special.MarcumQ(1, -1, 1)));
        Assert.True(double.IsNaN(XMath.Special.MarcumQ(1, 1, -1)));
        Assert.Equal(0, XMath.Special.NoncentralChiSquaredCdf(2, 3, -1));
        Assert.Equal(1, XMath.Special.NoncentralChiSquaredSf(2, 3, -1));
        Assert.Equal(1, XMath.Special.NoncentralChiSquaredCdf(2, 3, double.PositiveInfinity));
        Assert.Equal(0, XMath.Special.NoncentralChiSquaredSf(2, 3, double.PositiveInfinity));
        Assert.True(double.IsNaN(XMath.Special.NoncentralChiSquaredCdf(-1, 3, 1)));
        Assert.True(double.IsNaN(XMath.Special.NoncentralChiSquaredSf(2, -1, 1)));
    }

    [Theory]
    [InlineData(.5, .1)]
    [InlineData(4.0, 2.0)]
    [InlineData(10.0, 100.0)]
    public void ProbabilitiesAreBoundedMonotoneAndComplementary(double k, double lambda)
    {
        double previous = 0;
        foreach (double x in new[] { .01, 1.0, 10.0, 50.0, 200.0, 500.0 })
        {
            double p = XMath.Special.NoncentralChiSquaredCdf(k, lambda, x);
            double q = XMath.Special.NoncentralChiSquaredSf(k, lambda, x);
            Assert.InRange(p, 0, 1);
            Assert.InRange(q, 0, 1);
            Assert.True(p >= previous);
            SpecialFunctionAssert.Close(1, p + q, relativeTolerance: 2e-15);
            previous = p;
        }
    }
}
