using System.Numerics;
using Tuna;

namespace Tuna.Tests;

// mpmath 1.4.1による独立参照値。binary64入力を正確に変換し、
// 90/160桁（正規分布は490/560桁）でbinary64への丸め一致を確認。
// 生成式と選定理由: tools/SpecialFunctionReferences/generate_remaining_accuracy_references.py
// 通常テストはC#の固定値と対象APIの直接呼び出しだけで完結する。
[Trait("SpecialFunction", "OwensT")]
public class OwensTTests
{

    // 符号・a>1の反転・ah=9の隣接値・極小の裾。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0.0, 0.5, 0.07379180882521663)]
    [InlineData(0.2, 0.1, 0.01554762128247184)]
    [InlineData(1.0, 1.0, 0.06674188216570097)]
    [InlineData(-1.0, -0.5, -0.04306469112078536)]
    [InlineData(0.1, 10.0, 0.22679969643349898)]
    [InlineData(2.0, 3.0, 0.01137506597154504)]
    [InlineData(8.0, 0.5, 3.1103239107887044e-16)]
    [InlineData(37.5, 0.3, 2.3026765047909776e-308)]
    [InlineData(8.999999999999998, 1.0, 5.642942029769294e-20)]
    [InlineData(9.0, 1.0, 5.642942029769203e-20)]
    [InlineData(9.000000000000002, 1.0, 5.642942029769112e-20)]
    public void OwensT_MatchesIndependentReferences(double h, double a, double expected)
    {
        double actual = XMath.Special.OwensT(h, a);
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
    [InlineData(double.NaN, 0.5)]
    [InlineData(0.0, double.NaN)]
    public void OwensT_RejectsNaN(double h, double a)
    {
        double actual = XMath.Special.OwensT(h, a);
        Assert.True(double.IsNaN(actual));
    }

    [Theory]
    [InlineData(.3, .5)]
    [InlineData(2.0, 3.0)]
    [InlineData(10.0, .2)]
    public void SymmetriesAndClosedForms(double h, double a)
    {
        Assert.Equal(XMath.Special.OwensT(h, a), XMath.Special.OwensT(-h, a));
        Assert.Equal(-XMath.Special.OwensT(h, a), XMath.Special.OwensT(h, -a));
        SpecialFunctionAssert.Close(Math.Atan(a) / (2 * Math.PI), XMath.Special.OwensT(0, a), relativeTolerance: 2e-12);
        SpecialFunctionAssert.Close(.5 * XMath.Special.NormalCdf(h) * XMath.Special.NormalSf(h),
            XMath.Special.OwensT(h, 1), relativeTolerance: 2e-12);
        Assert.Equal(0, XMath.Special.OwensT(h, 0));
        Assert.Equal(0, XMath.Special.OwensT(double.PositiveInfinity, a));
        SpecialFunctionAssert.Close(.5 * XMath.Special.NormalSf(h), XMath.Special.OwensT(h, double.PositiveInfinity), relativeTolerance: 2e-12);
    }
}
