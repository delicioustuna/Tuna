using System.Numerics;
using Tuna;

namespace Tuna.Tests;

// mpmath 1.4.1による独立参照値。binary64入力を正確に変換し、
// 90/160桁（正規分布は490/560桁）でbinary64への丸め一致を確認。
// 生成式と選定理由: tools/SpecialFunctionReferences/generate_remaining_accuracy_references.py
// 通常テストはC#の固定値と対象APIの直接呼び出しだけで完結する。
[Trait("SpecialFunction", "ModifiedSphericalBessel")]
public class ModifiedSphericalBesselTests
{

    // 原点の近く・符号/次数反射・最終結果が有限な指数上下限。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0, 0.1, 1.0016675001984403)]
    [InlineData(1, 0.1, 0.03336667857363341)]
    [InlineData(5, 0.1, 9.623710240437372e-10)]
    [InlineData(0, 10.0, 1101.3232874703394)]
    [InlineData(10, 20.0, 771563.8929174455)]
    [InlineData(30, 100.0, 1.3014723273934537e+39)]
    [InlineData(0, 710.0, 1.5732357508181063e+305)]
    [InlineData(1, 710.0, 1.571019925816954e+305)]
    [InlineData(1, -2.0, -0.974382743580061)]
    [InlineData(2, -2.0, 0.3518560885534178)]
    [InlineData(0, 1e-300, 1.0)]
    [InlineData(0, 711.0, 4.270483387995776e+305)]
    [InlineData(1, 711.0, 4.2644770822461336e+305)]
    [InlineData(0, 715.0, 2.318560983669101e+307)]
    [InlineData(1, 715.0, 2.3153182410345986e+307)]
    public void ModifiedSphericalBesselI_MatchesIndependentReferences(int n, double x, double expected)
    {
        double actual = XMath.Special.ModifiedSphericalBesselI(n, x);
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

    // 大きい次数で e⁻ˣI が、極小・非正規化数の x で I や √(π/(2x)) が倍精度の範囲を外れても、iₙ は範囲内となる点の回帰テスト。
    // 期待値は級数 DLMF 10.53.3 と mpmath 1.4.1 の besseli（30桁・50桁）で丸め一致を確認。
    [Trait("Category", "Accuracy")]
    [Theory]
    // e⁻ˣ I_(n+1/2) がアンダーフローする。
    [InlineData(1000, 475.9279105617155, 1.8384845540464597e-169)]
    [InlineData(2000, 1500.0, 4.645485950742281e+127)]
    [InlineData(1000, -586.3209876640166, 6.013342794842248e-67)]
    // x ≤ 1 の級数。I_(n+1/2) がアンダーフローする。
    [InlineData(1, 1e-300, 3.3333333333333334e-301)]
    [InlineData(10, 1e-30, 7.2730919455576e-311)]
    [InlineData(150, 1.0, 8.8662479989517e-310)]
    // 非正規化数の x で √(π/(2x)) がオーバーフローする。
    [InlineData(1, 1e-310, 3.333333333333e-311)]
    [InlineData(2, 1e-310, 0.0)]
    [InlineData(300, 1.0, 0.0)]
    // 級数と I_(n+1/2) の切替の両側。
    [InlineData(50, 0.999, 3.472385649339559e-81)]
    [InlineData(50, 1.001, 3.837654273859105e-81)]
    public void ModifiedSphericalBesselI_LargeOrderAndTinyArgument(int n, double x, double expected)
    {
        double actual = XMath.Special.ModifiedSphericalBesselI(n, x);
        if (expected == 0)
        {
            Assert.Equal(expected, actual);
        }
        else if (Math.Abs(expected) < 2.2250738585072014e-308)
        {
            SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 0, absoluteTolerance: 2 * double.Epsilon);
            Assert.Equal(Math.Sign(expected), Math.Sign(actual));
        }
        else
        {
            SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 2e-12);
        }
    }

    // 各引数のNaNを独立に検査。
    [Theory]
    [InlineData(0, double.NaN)]
    public void ModifiedSphericalBesselI_RejectsNaN(int n, double x)
    {
        double actual = XMath.Special.ModifiedSphericalBesselI(n, x);
        Assert.True(double.IsNaN(actual));
    }

    // 原点の近く・符号/次数反射・最終結果が有限な指数上下限。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0, 0.1, 14.213152925974635, 0.0)]
    [InlineData(1, 0.1, 156.34468218572098, 0.0)]
    [InlineData(5, 0.1, 1483578155.175787, 0.0)]
    [InlineData(0, 10.0, 7.131404290765751e-06, 0.0)]
    [InlineData(10, 20.0, 2.253146801524262e-09, 0.0)]
    [InlineData(30, 100.0, 5.772215084463502e-44, 0.0)]
    [InlineData(0, 710.0, 9.903287268974e-312, 1e-323)]
    [InlineData(1, 710.0, 9.9172355609e-312, 1e-323)]
    [InlineData(-2, 2.0, 0.1594381243453636, 0.0)]
    [InlineData(100, 700.0, 2.9542560725186455e-304, 0.0)]
    [InlineData(1300, 700.0, 1.4995155148883681e+134, 0.0)]
    [InlineData(0, 738.0, 5e-324, 1e-323)]
    public void ModifiedSphericalBesselK_MatchesIndependentReferences(int n, double x, double expected, double absoluteTolerance)
    {
        double actual = XMath.Special.ModifiedSphericalBesselK(n, x);
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
    [InlineData(0, double.NaN)]
    public void ModifiedSphericalBesselK_RejectsNaN(int n, double x)
    {
        double actual = XMath.Special.ModifiedSphericalBesselK(n, x);
        Assert.True(double.IsNaN(actual));
    }

    [Fact]
    public void OriginInfinityAndInvalidDomains()
    {
        Assert.Equal(1, XMath.Special.ModifiedSphericalBesselI(0, 0));
        Assert.Equal(0, XMath.Special.ModifiedSphericalBesselI(3, 0));
        Assert.Equal(double.NegativeInfinity, XMath.Special.ModifiedSphericalBesselI(1, double.NegativeInfinity));
        Assert.Equal(double.PositiveInfinity, XMath.Special.ModifiedSphericalBesselK(0, 0));
        Assert.Equal(0, XMath.Special.ModifiedSphericalBesselK(1, double.PositiveInfinity));
        Assert.True(double.IsNaN(XMath.Special.ModifiedSphericalBesselI(-1, 1)));
        Assert.True(double.IsNaN(XMath.Special.ModifiedSphericalBesselK(1, -1)));
    }

    [Theory]
    [InlineData(0, double.PositiveInfinity, double.PositiveInfinity)]
    [InlineData(0, double.NegativeInfinity, double.PositiveInfinity)]
    [InlineData(1, double.PositiveInfinity, double.PositiveInfinity)]
    [InlineData(1, double.NegativeInfinity, double.NegativeInfinity)]
    public void InfiniteArgumentLimits(int n, double x, double expected)
    {
        Assert.Equal(expected, XMath.Special.ModifiedSphericalBesselI(n, x));
    }

    [Theory]
    [InlineData(.1)]
    [InlineData(2.0)]
    public void LowestOrderClosedForms(double x)
    {
        SpecialFunctionAssert.Close(Math.Sinh(x) / x, XMath.Special.ModifiedSphericalBesselI(0, x), relativeTolerance: 2e-12);
        SpecialFunctionAssert.Close(Math.PI * Math.Exp(-x) / (2 * x), XMath.Special.ModifiedSphericalBesselK(0, x), relativeTolerance: 2e-12);
        Assert.Equal(XMath.Special.ModifiedSphericalBesselK(1, x), XMath.Special.ModifiedSphericalBesselK(-2, x));
    }
}
