using System.Numerics;
using Tuna;

namespace Tuna.Tests;

// mpmath 1.4.1による独立参照値。binary64入力を正確に変換し、
// 90/160桁（正規分布は490/560桁）でbinary64への丸め一致を確認。
// 生成式と選定理由: tools/SpecialFunctionReferences/generate_remaining_accuracy_references.py
// 通常テストはC#の固定値と対象APIの直接呼び出しだけで完結する。
[Trait("SpecialFunction", "BarnesG")]
public class BarnesGTests
{

    // 非整数引数・整数の閉形式・漸化式/漸近展開の切替。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0.1, 0.10880645561709085)]
    [InlineData(0.5, 0.6032442812094462)]
    [InlineData(1.0, 1.0)]
    [InlineData(1.5, 1.069222649266413)]
    [InlineData(2.0, 1.0)]
    [InlineData(3.0, 1.0)]
    [InlineData(5.0, 12.0)]
    [InlineData(10.0, 5056584744960000.0)]
    [InlineData(39.0, double.PositiveInfinity)]
    [InlineData(40.0, double.PositiveInfinity)]
    [InlineData(41.0, double.PositiveInfinity)]
    public void BarnesG_MatchesIndependentReferences(double z, double expected)
    {
        double actual = XMath.Special.BarnesG(z);
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
    public void BarnesG_RejectsNaN(double z)
    {
        double actual = XMath.Special.BarnesG(z);
        Assert.True(double.IsNaN(actual));
    }

    // 非整数引数・整数の閉形式・漸化式/漸近展開の切替。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0.1, -2.2181846116046207, 2e-12)]
    [InlineData(0.5, -0.5054330544896953, 0.0)]
    [InlineData(1.0, 0.0, 0.0)]
    [InlineData(1.5, 0.06693188843500471, 0.0)]
    [InlineData(2.0, 0.0, 0.0)]
    [InlineData(3.0, 0.0, 0.0)]
    [InlineData(5.0, 2.4849066497880004, 2e-12)]
    [InlineData(10.0, 36.15946769873876, 2e-12)]
    [InlineData(39.0, 1577.7883153755226, 2e-12)]
    [InlineData(40.0, 1680.7565139900366, 2e-12)]
    [InlineData(41.0, 1787.38827425068, 2e-12)]
    public void LogBarnesG_MatchesIndependentReferences(double z, double expected, double absoluteTolerance)
    {
        double actual = XMath.Special.LogBarnesG(z);
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
    public void LogBarnesG_RejectsNaN(double z)
    {
        double actual = XMath.Special.LogBarnesG(z);
        Assert.True(double.IsNaN(actual));
    }

    // 非整数引数・整数の閉形式・漸化式/漸近展開の切替。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0.1, 0.9612552618072304)]
    [InlineData(0.5, 0.8804492351734234)]
    [InlineData(1.0, 1.0)]
    [InlineData(1.5, 1.6174885279489462)]
    [InlineData(2.0, 4.0)]
    [InlineData(3.0, 108.0)]
    [InlineData(5.0, 86400000.0)]
    [InlineData(10.0, 2.1577941222941857e+44)]
    [InlineData(39.0, double.PositiveInfinity)]
    [InlineData(40.0, double.PositiveInfinity)]
    [InlineData(41.0, double.PositiveInfinity)]
    public void Hyperfactorial_MatchesIndependentReferences(double z, double expected)
    {
        double actual = XMath.Special.Hyperfactorial(z);
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
    public void Hyperfactorial_RejectsNaN(double z)
    {
        double actual = XMath.Special.Hyperfactorial(z);
        Assert.True(double.IsNaN(actual));
    }

    // 非整数引数・整数の閉形式・漸化式/漸近展開の切替。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0.1, 0.9847727243540465)]
    [InlineData(0.5, 0.9475739010838258)]
    [InlineData(1.0, 1.0)]
    [InlineData(1.5, 1.2596482574951922)]
    [InlineData(2.0, 2.0)]
    [InlineData(3.0, 12.0)]
    [InlineData(5.0, 34560.0)]
    [InlineData(10.0, 6.658606584104737e+27)]
    [InlineData(39.0, double.PositiveInfinity)]
    [InlineData(40.0, double.PositiveInfinity)]
    [InlineData(41.0, double.PositiveInfinity)]
    public void Superfactorial_MatchesIndependentReferences(double z, double expected)
    {
        double actual = XMath.Special.Superfactorial(z);
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
    public void Superfactorial_RejectsNaN(double z)
    {
        double actual = XMath.Special.Superfactorial(z);
        Assert.True(double.IsNaN(actual));
    }

    [Theory]
    [InlineData(.3)]
    [InlineData(1.5)]
    [InlineData(10.0)]
    public void FunctionalEquation(double z)
    {
        SpecialFunctionAssert.Close(XMath.Special.Gamma(z) * XMath.Special.BarnesG(z),
            XMath.Special.BarnesG(z + 1), relativeTolerance: 2e-12);
        Assert.Equal(1, XMath.Special.BarnesG(1));
        Assert.Equal(1, XMath.Special.BarnesG(2));
        Assert.Equal(0, XMath.Special.BarnesG(0));
        Assert.Equal(0, XMath.Special.BarnesG(-2));
        Assert.Equal(double.NegativeInfinity, XMath.Special.LogBarnesG(0));
        SpecialFunctionAssert.Close(108, XMath.Special.Hyperfactorial(3), relativeTolerance: 2e-12);
        SpecialFunctionAssert.Close(12, XMath.Special.Superfactorial(3), relativeTolerance: 2e-12);
    }
}
