using System.Numerics;
using Tuna;

namespace Tuna.Tests;

// mpmath 1.4.1による独立参照値。binary64入力を正確に変換し、
// 90/160桁（正規分布は490/560桁）でbinary64への丸め一致を確認。
// 生成式と選定理由: tools/SpecialFunctionReferences/generate_remaining_accuracy_references.py
// 通常テストはC#の固定値と対象APIの直接呼び出しだけで完結する。
[Trait("SpecialFunction", "WignerSymbols")]
public class WignerSymbolsTests
{

    // 半整数/整数・符号・高精度Racah有限和。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(1.0, 1.0, 0.0, 0.0, 0.0, 0.0, -0.5773502691896257)]
    [InlineData(0.5, 0.5, 1.0, 0.5, -0.5, 0.0, 0.408248290463863)]
    [InlineData(2.0, 3.0, 4.0, 1.0, -2.0, 1.0, 0.19720265943665385)]
    [InlineData(10.0, 10.0, 10.0, 0.0, 0.0, 0.0, -0.05768853868256272)]
    [InlineData(20.0, 20.0, 20.0, 1.0, -1.0, 0.0, -0.014783442067309836)]
    public void Wigner3j_MatchesIndependentReferences(double j1, double j2, double j3, double m1, double m2, double m3, double expected)
    {
        double actual = XMath.Special.Wigner3j(j1, j2, j3, m1, m2, m3);
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
    [InlineData(double.NaN, 1.0, 0.0, 0.0, 0.0, 0.0)]
    [InlineData(1.0, double.NaN, 0.0, 0.0, 0.0, 0.0)]
    [InlineData(1.0, 1.0, double.NaN, 0.0, 0.0, 0.0)]
    [InlineData(1.0, 1.0, 0.0, double.NaN, 0.0, 0.0)]
    [InlineData(1.0, 1.0, 0.0, 0.0, double.NaN, 0.0)]
    [InlineData(1.0, 1.0, 0.0, 0.0, 0.0, double.NaN)]
    public void Wigner3j_RejectsNaN(double j1, double j2, double j3, double m1, double m2, double m3)
    {
        double actual = XMath.Special.Wigner3j(j1, j2, j3, m1, m2, m3);
        Assert.True(double.IsNaN(actual));
    }

    // 高精度Racah有限和・半整数の三角条件。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.16666666666666666)]
    [InlineData(0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 0.16666666666666666)]
    [InlineData(2.0, 3.0, 4.0, 3.0, 2.0, 2.0, 0.08748177652797065)]
    [InlineData(10.0, 10.0, 10.0, 10.0, 10.0, 10.0, -0.0029191867806092103)]
    public void Wigner6j_MatchesIndependentReferences(double j1, double j2, double j3, double j4, double j5, double j6, double expected)
    {
        double actual = XMath.Special.Wigner6j(j1, j2, j3, j4, j5, j6);
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
    [InlineData(double.NaN, 1.0, 1.0, 1.0, 1.0, 1.0)]
    [InlineData(1.0, double.NaN, 1.0, 1.0, 1.0, 1.0)]
    [InlineData(1.0, 1.0, double.NaN, 1.0, 1.0, 1.0)]
    [InlineData(1.0, 1.0, 1.0, double.NaN, 1.0, 1.0)]
    [InlineData(1.0, 1.0, 1.0, 1.0, double.NaN, 1.0)]
    [InlineData(1.0, 1.0, 1.0, 1.0, 1.0, double.NaN)]
    public void Wigner6j_RejectsNaN(double j1, double j2, double j3, double j4, double j5, double j6)
    {
        double actual = XMath.Special.Wigner6j(j1, j2, j3, j4, j5, j6);
        Assert.True(double.IsNaN(actual));
    }

    // Wigner3jとの位相・正規化。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0.5, 0.5, 0.5, -0.5, 1.0, 0.0, 0.7071067811865476)]
    [InlineData(0.5, 0.5, 0.5, -0.5, 0.0, 0.0, 0.7071067811865476)]
    [InlineData(2.0, 1.0, 3.0, -2.0, 4.0, -1.0, 0.5916079783099616)]
    public void ClebschGordan_MatchesIndependentReferences(double j1, double m1, double j2, double m2, double j, double m, double expected)
    {
        double actual = XMath.Special.ClebschGordan(j1, m1, j2, m2, j, m);
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
    [InlineData(double.NaN, 0.5, 0.5, -0.5, 1.0, 0.0)]
    [InlineData(0.5, double.NaN, 0.5, -0.5, 1.0, 0.0)]
    [InlineData(0.5, 0.5, double.NaN, -0.5, 1.0, 0.0)]
    [InlineData(0.5, 0.5, 0.5, double.NaN, 1.0, 0.0)]
    [InlineData(0.5, 0.5, 0.5, -0.5, double.NaN, 0.0)]
    [InlineData(0.5, 0.5, 0.5, -0.5, 1.0, double.NaN)]
    public void ClebschGordan_RejectsNaN(double j1, double m1, double j2, double m2, double j, double m)
    {
        double actual = XMath.Special.ClebschGordan(j1, m1, j2, m2, j, m);
        Assert.True(double.IsNaN(actual));
    }

    [Fact]
    public void SelectionRulesAndInvalidQuantumNumbers()
    {
        Assert.Equal(0, XMath.Special.Wigner3j(1, 1, 1, 1, 1, 1));
        Assert.Equal(0, XMath.Special.Wigner3j(1, 1, 3, 0, 0, 0));
        Assert.Equal(0, XMath.Special.Wigner6j(1, 1, 3, 1, 1, 1));
        Assert.Equal(0, XMath.Special.ClebschGordan(1, 1, 1, 1, 1, 0));
        Assert.True(double.IsNaN(XMath.Special.Wigner3j(.3, 1, 1, 0, 0, 0)));
        Assert.True(double.IsNaN(XMath.Special.Wigner6j(-1, 1, 1, 1, 1, 1)));
        Assert.True(double.IsNaN(XMath.Special.ClebschGordan(1, .3, 1, 0, 1, 0)));
    }

    [Fact]
    public void ClebschGordanNormalization()
    {
        double sum = 0;
        for (int m = -2; m <= 2; m++)
        {
            double c = XMath.Special.ClebschGordan(2, m, 2, -m, 2, 0);
            sum += c * c;
        }

        SpecialFunctionAssert.Close(1, sum, relativeTolerance: 2e-12);
    }
}
