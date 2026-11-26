using System.Numerics;
using Tuna;

namespace Tuna.Tests;

// mpmath 1.4.1による独立参照値。binary64入力を正確に変換し、
// 90/160桁（正規分布は490/560桁）でbinary64への丸め一致を確認。
// 生成式と選定理由: tools/SpecialFunctionReferences/generate_remaining_accuracy_references.py
// 通常テストはC#の固定値と対象APIの直接呼び出しだけで完結する。
[Trait("SpecialFunction", "BesselIntegral")]
public class BesselIntegralTests
{

    // 定義積分を独立に計算・原点の対数特異点・級数/漸近切替。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0.001, 0.0009999999166666698)]
    [InlineData(0.5, 0.48968050664604507)]
    [InlineData(5.0, 0.7153119177847678)]
    [InlineData(51.0, 0.9929546949961533)]
    [InlineData(100.0, 0.9226625569601661)]
    [InlineData(1.4999999999999998, 1.2414495143899675)]
    [InlineData(1.5, 1.2414495143899678)]
    [InlineData(1.5000000000000002, 1.2414495143899678)]
    [InlineData(34.99999999999999, 1.0475706824474624)]
    [InlineData(35.0, 1.0475706824474613)]
    [InlineData(35.00000000000001, 1.0475706824474604)]
    [InlineData(49.99999999999999, 0.9014121225818342)]
    [InlineData(50.0, 0.9014121225818346)]
    [InlineData(50.00000000000001, 0.901412122581835)]
    [InlineData(-2.0, -1.4257702931970266)]
    public void IntegralBesselJ0_MatchesIndependentReferences(double x, double expected)
    {
        double actual = XMath.Special.IntegralBesselJ0(x);
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
    public void IntegralBesselJ0_RejectsNaN(double x)
    {
        double actual = XMath.Special.IntegralBesselJ0(x);
        Assert.True(double.IsNaN(actual));
    }

    // 定義積分を独立に計算・原点の対数特異点・級数/漸近切替。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0.001, -0.005108037217399165)]
    [InlineData(0.5, -0.5617954559146403)]
    [InlineData(5.0, 0.1997193876223364)]
    [InlineData(51.0, -0.11146898718306773)]
    [InlineData(100.0, -0.01959806485340497)]
    [InlineData(1.4999999999999998, -0.5117590340286501)]
    [InlineData(1.5, -0.51175903402865)]
    [InlineData(1.5000000000000002, -0.5117590340286499)]
    [InlineData(34.99999999999999, 0.12610404368787084)]
    [InlineData(35.0, 0.12610404368787115)]
    [InlineData(35.00000000000001, 0.12610404368787148)]
    [InlineData(49.99999999999999, -0.05481407100006279)]
    [InlineData(50.0, -0.05481407100006349)]
    [InlineData(50.00000000000001, -0.054814071000064184)]
    public void IntegralBesselY0_MatchesIndependentReferences(double x, double expected)
    {
        double actual = XMath.Special.IntegralBesselY0(x);
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
    public void IntegralBesselY0_RejectsNaN(double x)
    {
        double actual = XMath.Special.IntegralBesselY0(x);
        Assert.True(double.IsNaN(actual));
    }

    // 積分した高精度収束級数・原点の対数特異点・級数/漸近切替。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0.001, 0.0010000000833333366)]
    [InlineData(0.5, 0.5105148087974031)]
    [InlineData(5.0, 31.848667776169886)]
    [InlineData(51.0, 7.972759688963115e+20)]
    [InlineData(100.0, 1.079217066847346e+42)]
    [InlineData(1.4999999999999998, 1.8060693729267818)]
    [InlineData(1.5, 1.8060693729267823)]
    [InlineData(1.5000000000000002, 1.8060693729267825)]
    [InlineData(34.99999999999999, 108955565324344.12)]
    [InlineData(35.0, 108955565324344.89)]
    [InlineData(35.00000000000001, 108955565324345.66)]
    [InlineData(49.99999999999999, 2.9629659299471937e+20)]
    [InlineData(50.0, 2.9629659299472146e+20)]
    [InlineData(50.00000000000001, 2.9629659299472356e+20)]
    public void IntegralBesselI0_MatchesIndependentReferences(double x, double expected)
    {
        double actual = XMath.Special.IntegralBesselI0(x);
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
    public void IntegralBesselI0_RejectsNaN(double x)
    {
        double actual = XMath.Special.IntegralBesselI0(x);
        Assert.True(double.IsNaN(actual));
    }

    // 積分した高精度収束級数・原点の対数特異点・級数/漸近切替。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0.001, 0.00802368749105892)]
    [InlineData(0.5, 0.9271025209311491)]
    [InlineData(5.0, 1.567387390728366)]
    [InlineData(51.0, 1.5707963267948966)]
    [InlineData(100.0, 1.5707963267948966)]
    [InlineData(1.4999999999999998, 1.3945824637327209)]
    [InlineData(1.5, 1.3945824637327209)]
    [InlineData(1.5000000000000002, 1.3945824637327209)]
    [InlineData(34.99999999999999, 1.5707963267948966)]
    [InlineData(35.0, 1.5707963267948966)]
    [InlineData(35.00000000000001, 1.5707963267948966)]
    [InlineData(49.99999999999999, 1.5707963267948966)]
    [InlineData(50.0, 1.5707963267948966)]
    [InlineData(50.00000000000001, 1.5707963267948966)]
    public void IntegralBesselK0_MatchesIndependentReferences(double x, double expected)
    {
        double actual = XMath.Special.IntegralBesselK0(x);
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
    public void IntegralBesselK0_RejectsNaN(double x)
    {
        double actual = XMath.Special.IntegralBesselK0(x);
        Assert.True(double.IsNaN(actual));
    }

    [Fact]
    public void OriginsAndDomains()
    {
        Assert.Equal(0, XMath.Special.IntegralBesselJ0(0));
        Assert.Equal(0, XMath.Special.IntegralBesselY0(0));
        Assert.Equal(0, XMath.Special.IntegralBesselI0(0));
        Assert.Equal(0, XMath.Special.IntegralBesselK0(0));
        Assert.Equal(1, XMath.Special.IntegralBesselJ0(double.PositiveInfinity));
        Assert.Equal(0, XMath.Special.IntegralBesselY0(double.PositiveInfinity));
        Assert.Equal(double.PositiveInfinity, XMath.Special.IntegralBesselI0(double.PositiveInfinity));
        SpecialFunctionAssert.Close(Math.PI / 2, XMath.Special.IntegralBesselK0(double.PositiveInfinity), relativeTolerance: 2e-12);
        Assert.True(double.IsNaN(XMath.Special.IntegralBesselY0(-1)));
        Assert.True(double.IsNaN(XMath.Special.IntegralBesselK0(-1)));
    }
}
