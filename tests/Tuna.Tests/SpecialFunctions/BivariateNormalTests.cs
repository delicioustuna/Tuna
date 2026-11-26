using System.Numerics;
using Tuna;

namespace Tuna.Tests;

// mpmath 1.4.1による独立参照値。binary64入力を正確に変換し、
// 90/160桁（正規分布は490/560桁）でbinary64への丸め一致を確認。
// 生成式と選定理由: tools/SpecialFunctionReferences/generate_remaining_accuracy_references.py
// 通常テストはC#の固定値と対象APIの直接呼び出しだけで完結する。
[Trait("SpecialFunction", "BivariateNormal")]
public class BivariateNormalTests
{

    // 独立した条件付き密度の積分・高相関切替。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0.0, 0.0, 0.5, 0.3333333333333333)]
    [InlineData(0.0, 0.0, -0.5, 0.16666666666666666)]
    [InlineData(0.3, -0.7, 0.0, 0.14951210446319219)]
    [InlineData(1.0, 2.0, 0.5, 0.8318608311308805)]
    [InlineData(-2.0, -1.0, -0.5, 0.0001468597660142571)]
    [InlineData(-8.0, -7.0, 0.5, 2.2164301924363994e-19)]
    [InlineData(1.0, -0.5, 0.99, 0.3085375387259869)]
    [InlineData(0.3, -0.7, 0.9249999999999999, 0.2417252673631033)]
    [InlineData(0.3, -0.7, 0.925, 0.2417252673631033)]
    [InlineData(0.3, -0.7, 0.9250000000000002, 0.2417252673631033)]
    [InlineData(0.3, -0.7, -0.9250000000000002, 0.010766113192357846)]
    [InlineData(0.3, -0.7, -0.925, 0.01076611319235787)]
    [InlineData(0.3, -0.7, -0.9249999999999999, 0.010766113192357894)]
    public void BivariateNormalCdf_MatchesIndependentReferences(double x, double y, double rho, double expected)
    {
        double actual = XMath.Special.BivariateNormalCdf(x, y, rho);
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
    [InlineData(double.NaN, 0.0, 0.5)]
    [InlineData(0.0, double.NaN, 0.5)]
    [InlineData(0.0, 0.0, double.NaN)]
    public void BivariateNormalCdf_RejectsNaN(double x, double y, double rho)
    {
        double actual = XMath.Special.BivariateNormalCdf(x, y, rho);
        Assert.True(double.IsNaN(actual));
    }

    [Theory]
    [InlineData(-2.0, -.5)]
    [InlineData(.3, .7)]
    [InlineData(2.0, -1.0)]
    public void DegenerateCorrelationsAndIndependentLimit(double x, double y)
    {
        SpecialFunctionAssert.Close(XMath.Special.NormalCdf(x) * XMath.Special.NormalCdf(y),
            XMath.Special.BivariateNormalCdf(x, y, 0), relativeTolerance: 2e-12);
        Assert.Equal(XMath.Special.NormalCdf(Math.Min(x, y)), XMath.Special.BivariateNormalCdf(x, y, 1));
        SpecialFunctionAssert.Close(Math.Max(0, XMath.Special.NormalCdf(x) - XMath.Special.NormalCdf(-y)),
            XMath.Special.BivariateNormalCdf(x, y, -1), relativeTolerance: 2e-12, absoluteTolerance: 2e-15);
        Assert.Equal(0, XMath.Special.BivariateNormalCdf(double.NegativeInfinity, y, .5));
        Assert.Equal(XMath.Special.NormalCdf(y), XMath.Special.BivariateNormalCdf(double.PositiveInfinity, y, .5));
        Assert.True(double.IsNaN(XMath.Special.BivariateNormalCdf(x, y, 1.1)));
    }

    [Theory]
    [InlineData(-.9)]
    [InlineData(.3)]
    [InlineData(.99)]
    public void OriginHasArcsineClosedForm(double rho)
    {
        SpecialFunctionAssert.Close(.25 + Math.Asin(rho) / (2 * Math.PI),
            XMath.Special.BivariateNormalCdf(0, 0, rho), relativeTolerance: 2e-12);
        double actual = XMath.Special.BivariateNormalCdf(.3, -.7, rho);
        SpecialFunctionAssert.Close(actual, XMath.Special.BivariateNormalCdf(-.7, .3, rho), relativeTolerance: 2e-12);
        Assert.InRange(actual, 0, Math.Min(XMath.Special.NormalCdf(.3), XMath.Special.NormalCdf(-.7)));
    }
}
