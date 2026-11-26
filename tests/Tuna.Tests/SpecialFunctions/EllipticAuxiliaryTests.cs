using System.Numerics;
using Tuna;

namespace Tuna.Tests;

// mpmath 1.4.1による独立参照値。binary64入力を正確に変換し、
// 90/160桁（正規分布は490/560桁）でbinary64への丸め一致を確認。
// 生成式と選定理由: tools/SpecialFunctionReferences/generate_remaining_accuracy_references.py
// 通常テストはC#の固定値と対象APIの直接呼び出しだけで完結する。
[Trait("SpecialFunction", "EllipticAuxiliary")]
public class EllipticAuxiliaryTests
{

    // 等引数・極端な比・中間演算の上下限。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(1.0, 1.0, 1.0)]
    [InlineData(1.0, 2.0, 1.4567910310469068)]
    [InlineData(0.01, 100.0, 14.823539385881489)]
    [InlineData(1e-200, 1e+200, 1.702907311399468e+197)]
    [InlineData(1e+308, 1e+308, 1e+308)]
    [InlineData(1e-300, 2e-300, 1.4567910310469069e-300)]
    public void ArithmeticGeometricMean_MatchesIndependentReferences(double a, double b, double expected)
    {
        double actual = XMath.Special.ArithmeticGeometricMean(a, b);
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
    [InlineData(double.NaN, 1.0)]
    [InlineData(1.0, double.NaN)]
    public void ArithmeticGeometricMean_RejectsNaN(double a, double b)
    {
        double actual = XMath.Special.ArithmeticGeometricMean(a, b);
        Assert.True(double.IsNaN(actual));
    }

    // 完全楕円D・m≈0の相殺。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(-10.0, 0.28482661481790295)]
    [InlineData(-0.5, 0.6720681345377233)]
    [InlineData(1e-20, 0.7853981633974483)]
    [InlineData(0.1, 0.816837118224562)]
    [InlineData(0.5, 1.006861592507393)]
    [InlineData(0.99, 2.7067109272370207)]
    public void EllipticD_Real1_MatchesIndependentReferences(double m, double expected)
    {
        double actual = XMath.Special.EllipticD(m);
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
    public void EllipticD_Real1_RejectsNaN(double m)
    {
        double actual = XMath.Special.EllipticD(m);
        Assert.True(double.IsNaN(actual));
    }

    // 不完全楕円Dの独立積分・周期延長。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0.1, 0.5, 0.0003331662590469842)]
    [InlineData(1.0, 0.5, 0.31177377844145737)]
    [InlineData(-1.0, 0.5, -0.31177377844145737)]
    [InlineData(2.0, 0.5, 1.5629750662150341)]
    [InlineData(0.3, -2.0, 0.008413064758561174)]
    public void EllipticD_Real2_MatchesIndependentReferences(double phi, double m, double expected)
    {
        double actual = XMath.Special.EllipticD(phi, m);
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
    [InlineData(0.1, double.NaN)]
    public void EllipticD_Real2_RejectsNaN(double phi, double m)
    {
        double actual = XMath.Special.EllipticD(phi, m);
        Assert.True(double.IsNaN(actual));
    }

    // Jacobi Zetaの奇対称性・周期延長。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0.1, 0.1, 0.005033407116600616)]
    [InlineData(1.0, 0.5, 0.13823539676104707)]
    [InlineData(-1.0, 0.5, -0.13823539676104707)]
    [InlineData(2.0, 0.5, -0.11777235923324854)]
    [InlineData(0.3, 0.99, 0.21183799399652486)]
    [InlineData(0.3, -2.0, -0.23551074096408342)]
    public void JacobiZeta_MatchesIndependentReferences(double phi, double m, double expected)
    {
        double actual = XMath.Special.JacobiZeta(phi, m);
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
    [InlineData(double.NaN, 0.1)]
    [InlineData(0.1, double.NaN)]
    public void JacobiZeta_RejectsNaN(double phi, double m)
    {
        double actual = XMath.Special.JacobiZeta(phi, m);
        Assert.True(double.IsNaN(actual));
    }

    // Heumanの定義・奇対称性・周期延長。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0.1, 0.1, 0.09728936489790664)]
    [InlineData(1.0, 0.5, 0.7474006351030856)]
    [InlineData(-1.0, 0.5, -0.7474006351030856)]
    [InlineData(0.3, 0.99, 0.19386105140993554)]
    [InlineData(2.0, 0.5, 1.1793726360143606)]
    public void HeumanLambda_MatchesIndependentReferences(double phi, double m, double expected)
    {
        double actual = XMath.Special.HeumanLambda(phi, m);
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
    [InlineData(double.NaN, 0.1)]
    [InlineData(0.1, double.NaN)]
    public void HeumanLambda_RejectsNaN(double phi, double m)
    {
        double actual = XMath.Special.HeumanLambda(phi, m);
        Assert.True(double.IsNaN(actual));
    }

    // 主枝の独立逆関数参照・端点。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0.1, 0.5, 0.10025119457863096)]
    [InlineData(-0.3, 0.5, -0.307054930495754)]
    [InlineData(1.0, 0.5, 1.8540746773013719)]
    [InlineData(0.8, 0.99, 1.0958242843457442)]
    [InlineData(0.5, -2.0, 0.48633426751333386)]
    public void InverseJacobiSN_MatchesIndependentReferences(double x, double m, double expected)
    {
        double actual = XMath.Special.InverseJacobiSN(x, m);
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
    [InlineData(0.1, double.NaN)]
    public void InverseJacobiSN_RejectsNaN(double x, double m)
    {
        double actual = XMath.Special.InverseJacobiSN(x, m);
        Assert.True(double.IsNaN(actual));
    }

    // 主枝の独立逆関数参照・端点。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0.1, 0.5, 1.712651906791571)]
    [InlineData(-0.3, 0.5, 2.2786835651903887)]
    [InlineData(1.0, 0.5, 0.0)]
    [InlineData(0.8, 0.99, 0.6925378011864756)]
    [InlineData(0.5, -2.0, 0.8596637463629874)]
    public void InverseJacobiCN_MatchesIndependentReferences(double x, double m, double expected)
    {
        double actual = XMath.Special.InverseJacobiCN(x, m);
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
    [InlineData(0.1, double.NaN)]
    public void InverseJacobiCN_RejectsNaN(double x, double m)
    {
        double actual = XMath.Special.InverseJacobiCN(x, m);
        Assert.True(double.IsNaN(actual));
    }

    // 主枝の独立逆関数参照・端点。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0.9, 0.5, 0.688792792395509)]
    [InlineData(0.8, 0.5, 1.0996754774930042)]
    [InlineData(0.2, 0.99, 2.3665253125475454)]
    [InlineData(1.0, 0.5, 0.0)]
    public void InverseJacobiDN_MatchesIndependentReferences(double x, double m, double expected)
    {
        double actual = XMath.Special.InverseJacobiDN(x, m);
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
    [InlineData(0.9, double.NaN)]
    public void InverseJacobiDN_RejectsNaN(double x, double m)
    {
        double actual = XMath.Special.InverseJacobiDN(x, m);
        Assert.True(double.IsNaN(actual));
    }

    // mの両端近く。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(1e-20, 6.25e-22)]
    [InlineData(0.01, 0.0006281456603830156)]
    [InlineData(0.5, 0.04321391826377225)]
    [InlineData(0.9, 0.14017312695426157)]
    [InlineData(0.9999999999999999, 0.7789534248779896)]
    public void EllipticNome_MatchesIndependentReferences(double m, double expected)
    {
        double actual = XMath.Special.EllipticNome(m);
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
    public void EllipticNome_RejectsNaN(double m)
    {
        double actual = XMath.Special.EllipticNome(m);
        Assert.True(double.IsNaN(actual));
    }

    // nome逆変換の独立参照。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(1e-20, 1.6e-19)]
    [InlineData(0.001, 0.015872700939449716)]
    [InlineData(0.1, 0.8024032982175763)]
    [InlineData(0.5, 0.9999895221373104)]
    [InlineData(0.9, 1.0)]
    public void EllipticNomeInverse_MatchesIndependentReferences(double q, double expected)
    {
        double actual = XMath.Special.EllipticNomeInverse(q);
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
    public void EllipticNomeInverse_RejectsNaN(double q)
    {
        double actual = XMath.Special.EllipticNomeInverse(q);
        Assert.True(double.IsNaN(actual));
    }

    [Fact]
    public void EndpointsAndInvalidDomains()
    {
        SpecialFunctionAssert.Close(Math.PI / 4, XMath.Special.EllipticD(0), relativeTolerance: 2e-12);
        Assert.Equal(0, XMath.Special.EllipticD(0, .5));
        Assert.Equal(0, XMath.Special.JacobiZeta(0, .5));
        Assert.Equal(0, XMath.Special.HeumanLambda(0, .5));
        Assert.Equal(0, XMath.Special.InverseJacobiSN(0, .5));
        Assert.Equal(0, XMath.Special.InverseJacobiCN(1, .5));
        Assert.Equal(0, XMath.Special.InverseJacobiDN(1, .5));
        Assert.True(double.IsNaN(XMath.Special.InverseJacobiSN(1.1, .5)));
        Assert.True(double.IsNaN(XMath.Special.InverseJacobiCN(-1.1, .5)));
        Assert.True(double.IsNaN(XMath.Special.InverseJacobiDN(.1, .5)));
        Assert.True(double.IsNaN(XMath.Special.ArithmeticGeometricMean(-1, 1)));
        Assert.Equal(0, XMath.Special.ArithmeticGeometricMean(0, 1));
        Assert.Equal(0, XMath.Special.EllipticNome(0));
        Assert.Equal(1, XMath.Special.EllipticNome(1));
        Assert.Equal(0, XMath.Special.EllipticNomeInverse(0));
        Assert.Equal(1, XMath.Special.EllipticNomeInverse(1));
        Assert.True(double.IsNaN(XMath.Special.EllipticNome(-.1)));
        Assert.True(double.IsNaN(XMath.Special.EllipticNomeInverse(1.1)));
    }
    // m=1 の極限を独立参照値で検証する。
    [Trait("Category", "AuditRegression")]
    [Trait("Category", "CoreAuditIdentities20261102")]
    [Theory]
    [InlineData(1e-100, 1.0, 230.95165647996453)]
    [InlineData(1e-160, 1.0, 369.10676205960726)]
    [InlineData(1e-200, 1.0, 461.2101657793691)]
    [InlineData(1e-308, 1.0, 709.889355822726)]
    [InlineData(1e-310, 1.0, 714.4945260087142)]
    [InlineData(0.0, 1.0, double.PositiveInfinity)]
    public void InverseJacobiCN_DegenerateParameterMatchesFixedReferences(double x, double m, double expected)
    {
        double actual = XMath.Special.InverseJacobiCN(x, m);
        if (double.IsInfinity(expected))
        {
            Assert.Equal(expected, actual);
        }
        else
        {
            SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 2e-12);
        }
    }

    // m=1 の極限を独立参照値で検証する。
    [Trait("Category", "AuditRegression")]
    [Trait("Category", "CoreAuditIdentities20261102")]
    [Theory]
    [InlineData(1e-100, 1.0, 230.95165647996453)]
    [InlineData(1e-160, 1.0, 369.10676205960726)]
    [InlineData(1e-200, 1.0, 461.2101657793691)]
    [InlineData(1e-308, 1.0, 709.889355822726)]
    [InlineData(1e-310, 1.0, 714.4945260087142)]
    [InlineData(0.0, 1.0, double.PositiveInfinity)]
    public void InverseJacobiDN_DegenerateParameterMatchesFixedReferences(double x, double m, double expected)
    {
        double actual = XMath.Special.InverseJacobiDN(x, m);
        if (double.IsInfinity(expected))
        {
            Assert.Equal(expected, actual);
        }
        else
        {
            SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 2e-12);
        }
    }

    // m=1 の極限を独立参照値で検証する。
    [Trait("Category", "AuditRegression")]
    [Trait("Category", "CoreAuditIdentities20261102")]
    [Theory]
    [InlineData(-1e+16, 1.0, 0.7796880066069788)]
    [InlineData(1e+16, 1.0, -0.7796880066069788)]
    [InlineData(-1e+17, 1.0, -0.4645301048353727)]
    [InlineData(1e+17, 1.0, 0.4645301048353727)]
    [InlineData(-1e+18, 1.0, 0.9929693207404051)]
    [InlineData(1e+18, 1.0, -0.9929693207404051)]
    [InlineData(-2e+18, 1.0, -0.23507950944432615)]
    [InlineData(2e+18, 1.0, 0.23507950944432615)]
    [InlineData(-1e+20, 1.0, 0.6452512852657808)]
    [InlineData(1e+20, 1.0, -0.6452512852657808)]
    public void JacobiZeta_DegenerateParameterMatchesFixedReferences(double x, double m, double expected)
    {
        double actual = XMath.Special.JacobiZeta(x, m);
        if (double.IsInfinity(expected))
        {
            Assert.Equal(expected, actual);
        }
        else
        {
            SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 2e-12);
        }
    }
}
