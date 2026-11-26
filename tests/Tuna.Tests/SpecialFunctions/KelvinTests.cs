using System.Numerics;
using Tuna;

namespace Tuna.Tests;

// mpmath 1.4.1による独立参照値。binary64入力を正確に変換し、
// 90/160桁（正規分布は490/560桁）でbinary64への丸め一致を確認。
// 生成式と選定理由: tools/SpecialFunctionReferences/generate_remaining_accuracy_references.py
// 通常テストはC#の固定値と対象APIの直接呼び出しだけで完結する。
[Trait("SpecialFunction", "Kelvin")]
public class KelvinTests
{

    // 整数次数の位相・級数/積分/漸近展開の切替。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0, 0.1, 0.9999984375000678)]
    [InlineData(1, 0.1, -0.035399514815076624)]
    [InlineData(2, 1.0, 0.01041124172310777)]
    [InlineData(-1, 2.0, 0.9970776519264285)]
    [InlineData(0, 10.0, 138.84046594163266)]
    [InlineData(3, 20.0, 103195.17856924573)]
    [InlineData(0, 30.0, -46117602.577984996)]
    [InlineData(1, 31.0, -103303836.04110874)]
    [InlineData(5, 100.0, -1.7986165198005525e+29)]
    public void KelvinBer_MatchesIndependentReferences(int n, double x, double expected)
    {
        double actual = XMath.Special.KelvinBer(n, x);
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
    public void KelvinBer_RejectsNaN(int n, double x)
    {
        double actual = XMath.Special.KelvinBer(n, x);
        Assert.True(double.IsNaN(actual));
    }

    // 整数次数の位相・級数/積分/漸近展開の切替。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0, 0.1, 0.0024999995659722293)]
    [InlineData(1, 0.1, 0.03531112647510091)]
    [InlineData(2, 1.0, -0.1246745356785978)]
    [InlineData(-1, 2.0, -0.2997754370020335)]
    [InlineData(0, 10.0, 56.37045855390664)]
    [InlineData(3, 20.0, -23908.749860930606)]
    [InlineData(0, 30.0, 109955713.1825061)]
    [InlineData(1, 31.0, -211268250.35612)]
    [InlineData(5, 100.0, 5.16705620611736e+28)]
    public void KelvinBei_MatchesIndependentReferences(int n, double x, double expected)
    {
        double actual = XMath.Special.KelvinBei(n, x);
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
    public void KelvinBei_RejectsNaN(int n, double x)
    {
        double actual = XMath.Special.KelvinBei(n, x);
        Assert.True(double.IsNaN(actual));
    }

    // 整数次数の位相・級数/積分/漸近展開の切替。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0, 0.1, 2.4204739810381715)]
    [InlineData(1, 0.1, -7.1466817140519705)]
    [InlineData(2, 1.0, 0.41803361794402505)]
    [InlineData(-1, 2.0, 0.23080592951812295)]
    [InlineData(0, 10.0, 0.0001294663302148061)]
    [InlineData(3, 20.0, 2.0158137759227683e-07)]
    [InlineData(0, 30.0, -1.2938269376020803e-10)]
    [InlineData(1, 31.0, 2.223863616470757e-11)]
    [InlineData(5, 100.0, -2.3390853933046874e-32)]
    public void KelvinKer_MatchesIndependentReferences(int n, double x, double expected)
    {
        double actual = XMath.Special.KelvinKer(n, x);
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
    public void KelvinKer_RejectsNaN(int n, double x)
    {
        double actual = XMath.Special.KelvinKer(n, x);
        Assert.True(double.IsNaN(actual));
    }

    // 整数次数の位相・級数/積分/漸近展開の切替。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0, 0.1, -0.776850646536661)]
    [InlineData(1, 0.1, -6.940242155964795)]
    [InlineData(2, 1.0, 1.884202418720101)]
    [InlineData(-1, 2.0, -0.08004939780706674)]
    [InlineData(0, 10.0, -0.0003075245690881442)]
    [InlineData(3, 20.0, -1.2270271052561885e-07)]
    [InlineData(0, 30.0, -5.289996606628324e-11)]
    [InlineData(1, 31.0, 6.4878367801617e-11)]
    [InlineData(5, 100.0, 1.2912884062819384e-32)]
    public void KelvinKei_MatchesIndependentReferences(int n, double x, double expected)
    {
        double actual = XMath.Special.KelvinKei(n, x);
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
    public void KelvinKei_RejectsNaN(int n, double x)
    {
        double actual = XMath.Special.KelvinKei(n, x);
        Assert.True(double.IsNaN(actual));
    }

    [Fact]
    public void OriginsAndDomains()
    {
        Assert.Equal(1, XMath.Special.KelvinBer(0, 0));
        Assert.Equal(0, XMath.Special.KelvinBei(0, 0));
        Assert.Equal(0, XMath.Special.KelvinBer(1, 0));
        Assert.Equal(0, XMath.Special.KelvinBei(1, 0));
        Assert.Equal(double.PositiveInfinity, XMath.Special.KelvinKer(0, 0));
        SpecialFunctionAssert.Close(-Math.PI / 4, XMath.Special.KelvinKei(0, 0), relativeTolerance: 2e-12);
        Assert.Equal(0, XMath.Special.KelvinKer(2, double.PositiveInfinity));
        Assert.Equal(0, XMath.Special.KelvinKei(2, double.PositiveInfinity));
        Assert.True(double.IsNaN(XMath.Special.KelvinKer(0, -1)));
        Assert.True(double.IsNaN(XMath.Special.KelvinKei(0, -1)));
        Assert.True(double.IsNaN(XMath.Special.KelvinBer(0, double.PositiveInfinity)));
    }

    // 冪級数の倍精度・double-double 切替（x = 2）の両側と、double-double 精度が必要な x ≈ 29.5 の検証。
    // 参照値は DLMF 10.61.1 の級数を binary64 入力のまま 300/500 bit で評価して一致を確認。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0, 29.476723011936446, -2248760.684010596, 83060683.825226)]
    [InlineData(0, 1.9999999999999998, 0.7517341827138083, 0.972291627306661)]
    [InlineData(0, 2.0, 0.7517341827138082, 0.9722916273066612)]
    [InlineData(0, 2.0000000000000004, 0.751734182713808, 0.9722916273066616)]
    [InlineData(0, 1e-100, 1.0, 2.5e-201)]
    [InlineData(1, 1.9999999999999998, -0.9970776519264284, 0.2997754370020336)]
    [InlineData(1, 2.0000000000000004, -0.9970776519264288, 0.29977543700203335)]
    [InlineData(1, 1e-100, -3.5355339059327375e-101, 3.5355339059327375e-101)]
    [InlineData(10, 1.0, 6.1158290022627825e-12, -2.690507365637099e-10)]
    [InlineData(10, 1.9999999999999998, 2.5025349684458952e-08, -2.745298322718207e-07)]
    [InlineData(10, 2.0000000000000004, 2.5025349684459054e-08, -2.745298322718216e-07)]
    public void KelvinBerBei_SeriesPrecisionSwitch_MatchesReference(int n, double x, double expectedBer, double expectedBei)
    {
        SpecialFunctionAssert.Close(expectedBer, XMath.Special.KelvinBer(n, x), relativeTolerance: 2e-12);
        SpecialFunctionAssert.Close(expectedBei, XMath.Special.KelvinBei(n, x), relativeTolerance: 2e-12);
    }
}
