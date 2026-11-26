using System.Numerics;
using Tuna;

namespace Tuna.Tests;

// mpmath 1.4.1による独立参照値。binary64入力を正確に変換し、
// 90/160桁（正規分布は490/560桁）でbinary64への丸め一致を確認。
// 生成式と選定理由: tools/SpecialFunctionReferences/generate_remaining_accuracy_references.py
// 通常テストはC#の固定値と対象APIの直接呼び出しだけで完結する。
[Trait("SpecialFunction", "Kolmogorov")]
public class KolmogorovTests
{

    // theta変換の切替と両側の裾。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0.05, 2.4231674791576992e-213)]
    [InlineData(0.1, 6.6093052422455605e-53)]
    [InlineData(0.25, 2.682381008482983e-08)]
    [InlineData(0.5, 0.03605475633512491)]
    [InlineData(0.8199999999999998, 0.48802829470150255)]
    [InlineData(0.82, 0.4880282947015027)]
    [InlineData(0.8200000000000001, 0.4880282947015029)]
    [InlineData(1.0, 0.7300003283226455)]
    [InlineData(2.0, 0.9993290747442203)]
    [InlineData(6.0, 1.0)]
    [InlineData(19.0, 1.0)]
    public void KolmogorovCdf_MatchesIndependentReferences(double x, double expected)
    {
        double actual = XMath.Special.KolmogorovCdf(x);
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
    public void KolmogorovCdf_RejectsNaN(double x)
    {
        double actual = XMath.Special.KolmogorovCdf(x);
        Assert.True(double.IsNaN(actual));
    }

    // theta変換の切替と両側の裾。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0.05, 1.0, 0.0)]
    [InlineData(0.1, 1.0, 0.0)]
    [InlineData(0.25, 0.9999999731761899, 0.0)]
    [InlineData(0.5, 0.9639452436648751, 0.0)]
    [InlineData(0.8199999999999998, 0.5119717052984974, 0.0)]
    [InlineData(0.82, 0.5119717052984973, 0.0)]
    [InlineData(0.8200000000000001, 0.5119717052984971, 0.0)]
    [InlineData(1.0, 0.2699996716773545, 0.0)]
    [InlineData(2.0, 0.0006709252557796953, 0.0)]
    [InlineData(6.0, 1.0760372320042276e-31, 0.0)]
    [InlineData(19.0, 5.5006506247e-314, 1e-323)]
    public void KolmogorovSf_MatchesIndependentReferences(double x, double expected, double absoluteTolerance)
    {
        double actual = XMath.Special.KolmogorovSf(x);
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
    public void KolmogorovSf_RejectsNaN(double x)
    {
        double actual = XMath.Special.KolmogorovSf(x);
        Assert.True(double.IsNaN(actual));
    }

    // theta変換の切替と両側の裾。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0.05, 4.778294548415971e-209, 0.0)]
    [InlineData(0.1, 1.624171397432998e-49, 0.0)]
    [InlineData(0.25, 4.128551064736226e-06, 0.0)]
    [InlineData(0.5, 0.6395828509404566, 0.0)]
    [InlineData(0.8199999999999998, 1.5888034698267486, 0.0)]
    [InlineData(0.82, 1.5888034698267484, 0.0)]
    [InlineData(0.8200000000000001, 1.5888034698267481, 0.0)]
    [InlineData(1.0, 1.0719485583569417, 0.0)]
    [InlineData(2.0, 0.005367402045629683, 0.0)]
    [InlineData(6.0, 2.5824893568101465e-30, 0.0)]
    [InlineData(19.0, 4.18049447497e-312, 1e-323)]
    public void KolmogorovPdf_MatchesIndependentReferences(double x, double expected, double absoluteTolerance)
    {
        double actual = XMath.Special.KolmogorovPdf(x);
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
    public void KolmogorovPdf_RejectsNaN(double x)
    {
        double actual = XMath.Special.KolmogorovPdf(x);
        Assert.True(double.IsNaN(actual));
    }

    // 独立した二分求根・裾確率。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(1e-30, 0.13087241009861583)]
    [InlineData(1e-10, 0.22013554252928297)]
    [InlineData(0.01, 0.4410276985179294)]
    [InlineData(0.5, 0.8275735551899077)]
    [InlineData(0.9, 1.2238478702170825)]
    [InlineData(0.9999999999999999, 4.326080659802649)]
    public void InverseKolmogorovCdf_MatchesIndependentReferences(double p, double expected)
    {
        double actual = XMath.Special.InverseKolmogorovCdf(p);
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
    public void InverseKolmogorovCdf_RejectsNaN(double p)
    {
        double actual = XMath.Special.InverseKolmogorovCdf(p);
        Assert.True(double.IsNaN(actual));
    }

    // 独立した二分求根・裾確率。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(1e-30, 5.9063821401252605)]
    [InlineData(1e-10, 3.4437623401231106)]
    [InlineData(0.01, 1.6276236115189504)]
    [InlineData(0.5, 0.8275735551899077)]
    [InlineData(0.9, 0.5711732651063401)]
    [InlineData(0.9999999999999999, 0.17698070738282087)]
    public void InverseKolmogorovSf_MatchesIndependentReferences(double p, double expected)
    {
        double actual = XMath.Special.InverseKolmogorovSf(p);
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
    public void InverseKolmogorovSf_RejectsNaN(double p)
    {
        double actual = XMath.Special.InverseKolmogorovSf(p);
        Assert.True(double.IsNaN(actual));
    }

    [Fact]
    public void EndpointsAndInverseDomains()
    {
        Assert.Equal(0, XMath.Special.KolmogorovCdf(0));
        Assert.Equal(1, XMath.Special.KolmogorovSf(0));
        Assert.Equal(0, XMath.Special.KolmogorovPdf(0));
        Assert.Equal(1, XMath.Special.KolmogorovCdf(double.PositiveInfinity));
        Assert.Equal(0, XMath.Special.KolmogorovSf(double.PositiveInfinity));
        Assert.Equal(0, XMath.Special.KolmogorovPdf(double.PositiveInfinity));
        Assert.Equal(0, XMath.Special.InverseKolmogorovCdf(0));
        Assert.Equal(double.PositiveInfinity, XMath.Special.InverseKolmogorovCdf(1));
        Assert.Equal(double.PositiveInfinity, XMath.Special.InverseKolmogorovSf(0));
        Assert.Equal(0, XMath.Special.InverseKolmogorovSf(1));
        Assert.True(double.IsNaN(XMath.Special.InverseKolmogorovCdf(-.1)));
        Assert.True(double.IsNaN(XMath.Special.InverseKolmogorovSf(1.1)));
    }

    [Fact]
    public void ProbabilitiesAreBoundedMonotoneAndComplementary()
    {
        double previous = 0;
        foreach (double x in new[] { .1, .3, .5, .82, 1.0, 2.0, 10.0 })
        {
            double cdf = XMath.Special.KolmogorovCdf(x), sf = XMath.Special.KolmogorovSf(x);
            Assert.InRange(cdf, 0, 1);
            Assert.InRange(sf, 0, 1);
            Assert.True(cdf >= previous);
            Assert.True(XMath.Special.KolmogorovPdf(x) >= 0);
            SpecialFunctionAssert.Close(1, cdf + sf, relativeTolerance: 2e-15);
            previous = cdf;
        }
    }
}
