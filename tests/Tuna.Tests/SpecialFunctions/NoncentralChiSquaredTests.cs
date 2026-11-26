using System.Numerics;
using Tuna;

namespace Tuna.Tests;

// mpmath 1.4.1による独立参照値。binary64入力を正確に変換し、
// 90/160桁（正規分布は490/560桁）でbinary64への丸め一致を確認。
// 生成式と選定理由: tools/SpecialFunctionReferences/generate_remaining_accuracy_references.py
// 通常テストはC#の固定値と対象APIの直接呼び出しだけで完結する。
[Trait("SpecialFunction", "NoncentralChiSquared")]
public class NoncentralChiSquaredTests
{

    // 非心度0・非整数自由度・大きな非心度・両裾。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(1.0, 0.0, 0.1, 1.200038948430136)]
    [InlineData(2.0, 0.0, 4.0, 0.06766764161830635)]
    [InlineData(0.5, 0.2, 0.01, 6.61650127245761)]
    [InlineData(4.0, 2.0, 3.0, 0.12083649092711131)]
    [InlineData(10.0, 20.0, 30.0, 0.03943738764205098)]
    [InlineData(3.0, 80.0, 85.0, 0.021472407959948182)]
    [InlineData(5.0, 100.0, 10.0, 4.298584933590631e-13)]
    [InlineData(5.0, 100.0, 300.0, 7.925751665062347e-14)]
    public void NoncentralChiSquaredPdf_MatchesIndependentReferences(double k, double lambda, double x, double expected)
    {
        double actual = XMath.Special.NoncentralChiSquaredPdf(k, lambda, x);
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
    [InlineData(double.NaN, 0.0, 0.1)]
    [InlineData(1.0, double.NaN, 0.1)]
    [InlineData(1.0, 0.0, double.NaN)]
    public void NoncentralChiSquaredPdf_RejectsNaN(double k, double lambda, double x)
    {
        double actual = XMath.Special.NoncentralChiSquaredPdf(k, lambda, x);
        Assert.True(double.IsNaN(actual));
    }

    // 非心度0・非整数自由度・大きな非心度・両裾。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(1.0, 0.0, 0.1, 0.18235401329235007, 0.0)]
    [InlineData(2.0, 0.0, 4.0, -2.6931471805599454, 2e-12)]
    [InlineData(0.5, 0.2, 0.01, 1.8895667215467664, 2e-12)]
    [InlineData(4.0, 2.0, 3.0, -2.1133169618863707, 2e-12)]
    [InlineData(10.0, 20.0, 30.0, -3.233040987697734, 2e-12)]
    [InlineData(3.0, 80.0, 85.0, -3.840986518748348, 2e-12)]
    [InlineData(5.0, 100.0, 10.0, -28.47532032559498, 2e-12)]
    [InlineData(5.0, 100.0, 300.0, -30.16607413932454, 2e-12)]
    public void NoncentralChiSquaredLogPdf_MatchesIndependentReferences(double k, double lambda, double x, double expected, double absoluteTolerance)
    {
        double actual = XMath.Special.NoncentralChiSquaredLogPdf(k, lambda, x);
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
    [InlineData(double.NaN, 0.0, 0.1)]
    [InlineData(1.0, double.NaN, 0.1)]
    [InlineData(1.0, 0.0, double.NaN)]
    public void NoncentralChiSquaredLogPdf_RejectsNaN(double k, double lambda, double x)
    {
        double actual = XMath.Special.NoncentralChiSquaredLogPdf(k, lambda, x);
        Assert.True(double.IsNaN(actual));
    }

    [Fact]
    public void OriginLimitsAndDomains()
    {
        Assert.Equal(double.PositiveInfinity, XMath.Special.NoncentralChiSquaredPdf(1, 2, 0));
        SpecialFunctionAssert.Close(Math.Exp(-1) / 2, XMath.Special.NoncentralChiSquaredPdf(2, 2, 0), relativeTolerance: 2e-12);
        Assert.Equal(0, XMath.Special.NoncentralChiSquaredPdf(3, 2, 0));
        Assert.Equal(0, XMath.Special.NoncentralChiSquaredPdf(3, 2, -1));
        Assert.Equal(0, XMath.Special.NoncentralChiSquaredPdf(3, 2, double.PositiveInfinity));
        Assert.Equal(double.NegativeInfinity, XMath.Special.NoncentralChiSquaredLogPdf(3, 2, double.PositiveInfinity));
        Assert.True(double.IsNaN(XMath.Special.NoncentralChiSquaredPdf(0, 2, 1)));
        Assert.True(double.IsNaN(XMath.Special.NoncentralChiSquaredLogPdf(3, -1, 1)));
    }

    // ポアソン混合の正項和で生成した参照値（100/180 桁で丸め一致）を用い、k/2-1 や (a+j)-1 の桁落ち、k/2 のアンダーフロー、t=1 の両隣を検証する。
    [Trait("Category", "AuditRegression")]
    [Trait("Category", "CoreAudit20261102")]
    [Theory]
    [InlineData(5e-324, 1.0, 1.0, 0.10395520767485422)]
    [InlineData(5e-324, 10.0, 1.0, 0.02956885714516434)]
    [InlineData(1e-320, 1.0, 1.0, 0.10395520767485422)]
    [InlineData(1e-320, 10.0, 1.0, 0.02956885714516434)]
    [InlineData(1e-20, 1.0, 1.0, 0.10395520767485422)]
    [InlineData(1e-20, 10.0, 1.0, 0.02956885714516434)]
    [InlineData(1e-16, 1.0, 1.0, 0.10395520767485424)]
    [InlineData(1e-16, 10.0, 1.0, 0.02956885714516434)]
    [InlineData(1e-15, 1.0, 1.0, 0.1039552076748544)]
    [InlineData(1e-15, 10.0, 1.0, 0.02956885714516433)]
    [InlineData(1e-14, 1.0, 1.0, 0.10395520767485594)]
    [InlineData(1e-14, 10.0, 1.0, 0.02956885714516423)]
    [InlineData(1e-10, 1.0, 1.0, 0.10395520769203395)]
    [InlineData(1e-10, 10.0, 1.0, 0.02956885714404216)]
    [InlineData(1e-08, 1.0, 1.0, 0.10395520939282647)]
    [InlineData(1e-08, 10.0, 1.0, 0.029568857032946295)]
    [InlineData(1e-06, 1.0, 1.0, 0.10395537947205158)]
    [InlineData(1e-06, 10.0, 1.0, 0.02956884592336058)]
    [InlineData(0.1, 1.0, 1.0, 0.1208203265704189)]
    [InlineData(0.1, 10.0, 1.0, 0.028454872098971344)]
    [InlineData(1e-20, 3.9999999999999996, 1.0, 0.13056742402402788)]
    [InlineData(1e-10, 3.9999999999999996, 1.0, 0.1305674240247548)]
    [InlineData(1e-20, 4.0, 1.0, 0.13056742402402785)]
    [InlineData(1e-10, 4.0, 1.0, 0.13056742402475477)]
    [InlineData(1e-20, 4.000000000000001, 1.0, 0.13056742402402785)]
    [InlineData(1e-10, 4.000000000000001, 1.0, 0.13056742402475477)]
    public void NoncentralChiSquaredPdf_TinyShapeAndSeriesBoundary_FixedReferences(double k, double lambda, double x, double expected)
    {
        double actual = XMath.Special.NoncentralChiSquaredPdf(k, lambda, x);
        if (double.IsInfinity(expected) || expected == 0)
        {
            Assert.Equal(expected, actual);
        }
        else
        {
            SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 2e-12,
                absoluteTolerance: Math.Abs(expected) < 2.2250738585072014e-308 ? 2 * double.Epsilon : 0);
        }
    }

    // 微小な自由度でも対数密度が ∞ や NaN にならず有限値として得られることを検証する（参照値は 100/180 桁で一致を確認）。
    [Trait("Category", "AuditRegression")]
    [Trait("Category", "CoreAudit20261102")]
    [Theory]
    [InlineData(5e-324, 1.0, 1.0, -2.2637951680507764)]
    [InlineData(5e-324, 10.0, 1.0, -3.52103359499673)]
    [InlineData(1e-320, 1.0, 1.0, -2.2637951680507764)]
    [InlineData(1e-320, 10.0, 1.0, -3.52103359499673)]
    [InlineData(1e-20, 1.0, 1.0, -2.2637951680507764)]
    [InlineData(1e-20, 10.0, 1.0, -3.52103359499673)]
    [InlineData(1e-16, 1.0, 1.0, -2.2637951680507764)]
    [InlineData(1e-16, 10.0, 1.0, -3.52103359499673)]
    [InlineData(1e-15, 1.0, 1.0, -2.263795168050775)]
    [InlineData(1e-15, 10.0, 1.0, -3.52103359499673)]
    [InlineData(1e-14, 1.0, 1.0, -2.26379516805076)]
    [InlineData(1e-14, 10.0, 1.0, -3.5210335949967337)]
    [InlineData(1e-10, 1.0, 1.0, -2.2637951678855157)]
    [InlineData(1e-10, 10.0, 1.0, -3.5210335950346816)]
    [InlineData(1e-08, 1.0, 1.0, -2.263795151524695)]
    [InlineData(1e-08, 10.0, 1.0, -3.521033598791873)]
    [InlineData(1e-06, 1.0, 1.0, -2.263793515444243)]
    [InlineData(1e-06, 10.0, 1.0, -3.5210339745110897)]
    [InlineData(0.1, 1.0, 1.0, -2.1134507413251957)]
    [InlineData(0.1, 10.0, 1.0, -3.5594358817896787)]
    [InlineData(1e-20, 3.9999999999999996, 1.0, -2.0358655264538403)]
    [InlineData(1e-10, 3.9999999999999996, 1.0, -2.035865526448273)]
    [InlineData(1e-20, 4.0, 1.0, -2.0358655264538403)]
    [InlineData(1e-10, 4.0, 1.0, -2.035865526448273)]
    [InlineData(1e-20, 4.000000000000001, 1.0, -2.0358655264538403)]
    [InlineData(1e-10, 4.000000000000001, 1.0, -2.0358655264482732)]
    public void NoncentralChiSquaredLogPdf_TinyShapeAndSeriesBoundary_FixedReferences(double k, double lambda, double x, double expected)
    {
        double actual = XMath.Special.NoncentralChiSquaredLogPdf(k, lambda, x);
        if (double.IsInfinity(expected) || expected == 0)
        {
            Assert.Equal(expected, actual);
        }
        else
        {
            SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 2e-12,
                absoluteTolerance: Math.Abs(expected) < 2.2250738585072014e-308 ? 2 * double.Epsilon : 0);
        }
    }
}
