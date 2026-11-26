using System.Numerics;
using Tuna;

namespace Tuna.Tests;

// mpmath 1.4.1による独立参照値。binary64入力を正確に変換し、
// 90/160桁（正規分布は490/560桁）でbinary64への丸め一致を確認。
// 生成式と選定理由: tools/SpecialFunctionReferences/generate_remaining_accuracy_references.py
// 通常テストはC#の固定値と対象APIの直接呼び出しだけで完結する。
[Trait("SpecialFunction", "LegendreFunctions")]
public class LegendreFunctionsTests
{

    // 整数次数・Condon–Shortley位相・m>n・区間端付近。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0, 0.2, 0.2027325540540822)]
    [InlineData(1, 0.5, -0.7253469278329726)]
    [InlineData(5, -0.3, 0.06103298206386362)]
    [InlineData(20, 0.9, -0.3471484373422187)]
    [InlineData(2, 0.999, 2.2903062640501997)]
    public void LegendreQ_Integer2_MatchesIndependentReferences(int n, double x, double expected)
    {
        double actual = XMath.Special.LegendreQ(n, x);
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
    [InlineData(0, double.NaN)]
    public void LegendreQ_Integer2_RejectsNaN(int n, double x)
    {
        double actual = XMath.Special.LegendreQ(n, x);
        Assert.True(double.IsNaN(actual));
    }

    // 整数次数・Condon–Shortley位相・m>n・区間端付近。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(2, 1, 0.3, 1.5477961718781537)]
    [InlineData(5, -2, 0.4, -0.010325112530866062)]
    [InlineData(20, 3, 0.8, 3067.472933711759)]
    [InlineData(3, 5, 0.9, -21962.902389381215)]
    public void AssociatedLegendreQ_Integer3_MatchesIndependentReferences(int n, int m, double x, double expected)
    {
        double actual = XMath.Special.AssociatedLegendreQ(n, m, x);
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
    [InlineData(2, 1, double.NaN)]
    public void AssociatedLegendreQ_Integer3_RejectsNaN(int n, int m, double x)
    {
        double actual = XMath.Special.AssociatedLegendreQ(n, m, x);
        Assert.True(double.IsNaN(actual));
    }

    // 非整数次数/位数・負引数・整数位数の極限・大次数。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0.3, 0.2, 0.809948069025646)]
    [InlineData(2.5, -0.4, 0.21467698917325698)]
    [InlineData(10.5, 0.8, 0.3101386972847781)]
    [InlineData(-0.3, 0.5, 1.0613022485533188)]
    [InlineData(20.5, -0.7, -0.04176738595057368)]
    public void LegendreP_MatchesIndependentReferences(double nu, double x, double expected)
    {
        double actual = XMath.Special.LegendreP(nu, x);
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
    [InlineData(double.NaN, 0.2)]
    [InlineData(0.3, double.NaN)]
    public void LegendreP_RejectsNaN(double nu, double x)
    {
        double actual = XMath.Special.LegendreP(nu, x);
        Assert.True(double.IsNaN(actual));
    }

    // 非整数次数/位数・負引数・整数位数の極限・大次数。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0.3, 0.2, -0.37724017719222674)]
    [InlineData(2.5, -0.4, 0.6700381703387136)]
    [InlineData(10.5, 0.8, 0.0024586965923700035)]
    [InlineData(-0.3, 0.5, 1.3315033967066319)]
    [InlineData(20.5, -0.7, 0.31682546492128355)]
    public void LegendreQ_Real2_MatchesIndependentReferences(double nu, double x, double expected)
    {
        double actual = XMath.Special.LegendreQ(nu, x);
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
    [InlineData(double.NaN, 0.2)]
    [InlineData(0.3, double.NaN)]
    public void LegendreQ_Real2_RejectsNaN(double nu, double x)
    {
        double actual = XMath.Special.LegendreQ(nu, x);
        Assert.True(double.IsNaN(actual));
    }

    // 非整数次数/位数・負引数・整数位数の極限・大次数。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0.3, 0.4, 0.2, 0.4844135444994591)]
    [InlineData(2.5, -0.3, -0.4, -0.0038235746874972472)]
    [InlineData(10.5, 2.0, 0.8, -36.946913394532665)]
    [InlineData(-0.3, 0.2, 0.5, 1.0334877226365715)]
    [InlineData(20.5, -2.0, -0.7, 0.00013795189287686403)]
    public void AssociatedLegendreP_MatchesIndependentReferences(double nu, double mu, double x, double expected)
    {
        double actual = XMath.Special.AssociatedLegendreP(nu, mu, x);
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
    [InlineData(double.NaN, 0.4, 0.2)]
    [InlineData(0.3, double.NaN, 0.2)]
    [InlineData(0.3, 0.4, double.NaN)]
    public void AssociatedLegendreP_RejectsNaN(double nu, double mu, double x)
    {
        double actual = XMath.Special.AssociatedLegendreP(nu, mu, x);
        Assert.True(double.IsNaN(actual));
    }

    // 非整数次数/位数・負引数・整数位数の極限・大次数。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0.3, 0.4, 0.2, -1.0212192992642672)]
    [InlineData(2.5, -0.3, -0.4, 0.5404568865907488)]
    [InlineData(10.5, 2.0, 0.8, 14.037600700905497)]
    [InlineData(-0.3, 0.2, 0.5, 0.5830427148564704)]
    [InlineData(20.5, -2.0, -0.7, -0.0007065619887570443)]
    public void AssociatedLegendreQ_Real3_MatchesIndependentReferences(double nu, double mu, double x, double expected)
    {
        double actual = XMath.Special.AssociatedLegendreQ(nu, mu, x);
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
    [InlineData(double.NaN, 0.4, 0.2)]
    [InlineData(0.3, double.NaN, 0.2)]
    [InlineData(0.3, 0.4, double.NaN)]
    public void AssociatedLegendreQ_Real3_RejectsNaN(double nu, double mu, double x)
    {
        double actual = XMath.Special.AssociatedLegendreQ(nu, mu, x);
        Assert.True(double.IsNaN(actual));
    }

    [Theory]
    [InlineData(-.8)]
    [InlineData(.2)]
    [InlineData(.9)]
    public void LowestIntegerDegreesAndDomains(double x)
    {
        SpecialFunctionAssert.Close(Math.Atanh(x), XMath.Special.LegendreQ(0, x), relativeTolerance: 2e-12);
        SpecialFunctionAssert.Close(x * Math.Atanh(x) - 1, XMath.Special.LegendreQ(1, x), relativeTolerance: 2e-12);
        SpecialFunctionAssert.Close(-1 / Math.Sqrt((1 - x) * (1 + x)), XMath.Special.AssociatedLegendreQ(0, 1, x), relativeTolerance: 2e-12);
        Assert.True(double.IsNaN(XMath.Special.LegendreQ(0, 1)));
        Assert.True(double.IsNaN(XMath.Special.AssociatedLegendreQ(2, -3, x)));
        Assert.True(double.IsNaN(XMath.Special.LegendreP(.3, -1)));
        Assert.True(double.IsNaN(XMath.Special.LegendreQ(.3, 1)));
        Assert.True(double.IsNaN(XMath.Special.AssociatedLegendreP(.3, .2, 1.1)));
        Assert.True(double.IsNaN(XMath.Special.AssociatedLegendreQ(.3, .2, -1)));
    }
}
