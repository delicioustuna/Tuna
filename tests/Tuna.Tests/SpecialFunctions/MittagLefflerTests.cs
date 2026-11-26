using System.Numerics;
using Tuna;

namespace Tuna.Tests;

// mpmath 1.4.1による独立参照値。binary64入力を正確に変換し、
// 90/160桁（正規分布は490/560桁）でbinary64への丸め一致を確認。
// 生成式と選定理由: tools/SpecialFunctionReferences/generate_remaining_accuracy_references.py
// 通常テストはC#の固定値と対象APIの直接呼び出しだけで完結する。
[Trait("SpecialFunction", "MittagLeffler")]
public class MittagLefflerTests
{

    // 独立した高精度級数・指数/erfc/cosの閉形式・負beta。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(1.0, 1.0, -20.0, 2.061153622438558e-09)]
    [InlineData(0.5, 1.0, -10.0, 0.05614099274382259)]
    [InlineData(2.0, 1.0, -10.0, -0.9997860728793259)]
    [InlineData(0.7, 1.3, 0.2, 1.3430753193024672)]
    [InlineData(0.7, 1.3, -5.0, 0.13592283992138554)]
    [InlineData(1.5, 0.5, -2.0, -0.5158078020855824)]
    [InlineData(2.5, 2.0, 20.0, 3.346202568669092)]
    [InlineData(0.8, -0.5, 0.5, 0.3297476120426755)]
    [InlineData(1.0, 2.0, 0.001, 1.0005001667083417)]
    [InlineData(1.0, 0.0, -3.0, -0.14936120510359183)]
    public void MittagLeffler_Real3_MatchesIndependentReferences(double alpha, double beta, double x, double expected)
    {
        double actual = XMath.Special.MittagLeffler(alpha, beta, x);
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
    [InlineData(double.NaN, 1.0, -20.0)]
    [InlineData(1.0, double.NaN, -20.0)]
    [InlineData(1.0, 1.0, double.NaN)]
    public void MittagLeffler_Real3_RejectsNaN(double alpha, double beta, double x)
    {
        double actual = XMath.Special.MittagLeffler(alpha, beta, x);
        Assert.True(double.IsNaN(actual));
    }

    // 1パラメータ版の独立参照。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0.5, -2.0, 0.25539567631050575)]
    [InlineData(0.7, -1.0, 0.3996119781155994)]
    [InlineData(1.0, 2.0, 7.38905609893065)]
    [InlineData(1.5, -2.0, 0.02943068560282647)]
    [InlineData(2.0, 5.0, 4.731673471130766)]
    public void MittagLeffler_Real2_MatchesIndependentReferences(double alpha, double x, double expected)
    {
        double actual = XMath.Special.MittagLeffler(alpha, x);
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
    [InlineData(double.NaN, -2.0)]
    [InlineData(0.5, double.NaN)]
    public void MittagLeffler_Real2_RejectsNaN(double alpha, double x)
    {
        double actual = XMath.Special.MittagLeffler(alpha, x);
        Assert.True(double.IsNaN(actual));
    }

    [Theory]
    [InlineData(-2.0)]
    [InlineData(.1)]
    [InlineData(5.0)]
    public void ElementaryCasesAndDomains(double x)
    {
        SpecialFunctionAssert.Close(Math.Exp(x), XMath.Special.MittagLeffler(1, x), relativeTolerance: 2e-12);
        double expected = x > 0 ? Math.Cosh(Math.Sqrt(x)) : Math.Cos(Math.Sqrt(-x));
        SpecialFunctionAssert.Close(expected, XMath.Special.MittagLeffler(2, x), relativeTolerance: 2e-12);
        Assert.Equal(1, XMath.Special.MittagLeffler(.3, 1, 0));
        Assert.Equal(0, XMath.Special.MittagLeffler(.3, -1, 0));
        Assert.True(double.IsNaN(XMath.Special.MittagLeffler(0, x)));
        Assert.True(double.IsNaN(XMath.Special.MittagLeffler(-1, 1, x)));
    }
}
