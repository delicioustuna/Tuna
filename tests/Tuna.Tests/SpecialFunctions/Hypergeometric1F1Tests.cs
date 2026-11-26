using System.Numerics;
using Tuna;

namespace Tuna.Tests;

// mpmath 1.4.1による独立参照値。binary64入力を正確に変換し、
// 90/160桁（正規分布は490/560桁）でbinary64への丸め一致を確認。
// 生成式と選定理由: tools/SpecialFunctionReferences/generate_remaining_accuracy_references.py
// 通常テストはC#の固定値と対象APIの直接呼び出しだけで完結する。
[Trait("SpecialFunction", "Hypergeometric1F1")]
public class Hypergeometric1F1Tests
{

    // 終端多項式・Kummer変換・漸近展開・大きな指数。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(1.0, 2.0, 0.5, 1.2974425414002564)]
    [InlineData(0.5, 1.5, -20.0, 0.19816636482997366)]
    [InlineData(-3.0, 2.0, 4.0, 0.3333333333333333)]
    [InlineData(-2.0, -3.0, 0.5, 1.375)]
    [InlineData(10.0, 3.0, 100.0, 2.6666800256732336e+52)]
    [InlineData(2.0, 2.0, -10.0, 4.5399929762484854e-05)]
    [InlineData(0.3, 1.7, 8.0, 58.4611084798955)]
    [InlineData(0.3, 1.7, 50.0, 6.721270085099629e+18)]
    [InlineData(1.0, 2.0, 700.0, 1.4489029353357207e+301)]
    public void Hypergeometric1F1_MatchesIndependentReferences(double a, double b, double x, double expected)
    {
        double actual = XMath.Special.Hypergeometric1F1(a, b, x);
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
    [InlineData(double.NaN, 2.0, 0.5)]
    [InlineData(1.0, double.NaN, 0.5)]
    [InlineData(1.0, 2.0, double.NaN)]
    public void Hypergeometric1F1_RejectsNaN(double a, double b, double x)
    {
        double actual = XMath.Special.Hypergeometric1F1(a, b, x);
        Assert.True(double.IsNaN(actual));
    }

    // 分母の極でも有限な正則化版。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(1.0, 2.0, 0.5, 1.2974425414002564)]
    [InlineData(0.5, 1.5, -20.0, 0.22360679769319114)]
    [InlineData(-3.0, 2.0, 4.0, 0.3333333333333333)]
    [InlineData(-2.0, -3.0, 0.5, 0.0)]
    [InlineData(10.0, 3.0, 100.0, 1.3333400128366168e+52)]
    [InlineData(2.0, 2.0, -10.0, 4.5399929762484854e-05)]
    [InlineData(0.3, 1.7, 8.0, 64.33922126158656)]
    [InlineData(0.3, 1.7, 50.0, 7.397076353980223e+18)]
    [InlineData(1.0, 2.0, 700.0, 1.4489029353357207e+301)]
    [InlineData(1.0, 0.0, 0.5, 0.8243606353500641)]
    [InlineData(0.5, -2.0, 1.0, 0.7572437837620767)]
    public void Hypergeometric1F1Regularized_MatchesIndependentReferences(double a, double b, double x, double expected)
    {
        double actual = XMath.Special.Hypergeometric1F1Regularized(a, b, x);
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
    [InlineData(double.NaN, 2.0, 0.5)]
    [InlineData(1.0, double.NaN, 0.5)]
    [InlineData(1.0, 2.0, double.NaN)]
    public void Hypergeometric1F1Regularized_RejectsNaN(double a, double b, double x)
    {
        double actual = XMath.Special.Hypergeometric1F1Regularized(a, b, x);
        Assert.True(double.IsNaN(actual));
    }

    [Theory]
    [InlineData(.3, 1.7, -10.0)]
    [InlineData(2.0, 3.0, .5)]
    [InlineData(-2.0, 1.0, 4.0)]
    public void KummerTransformation(double a, double b, double x)
    {
        SpecialFunctionAssert.Close(Math.Exp(x) * XMath.Special.Hypergeometric1F1(b - a, b, -x), XMath.Special.Hypergeometric1F1(a, b, x), relativeTolerance: 2e-12);
        SpecialFunctionAssert.Close(XMath.Special.Hypergeometric1F1(a, b, x) / XMath.Special.Gamma(b), XMath.Special.Hypergeometric1F1Regularized(a, b, x), relativeTolerance: 2e-12);
    }

    [Fact]
    public void PolesAndTerminatingCases()
    {
        Assert.True(double.IsNaN(XMath.Special.Hypergeometric1F1(1, 0, .5)));
        Assert.True(double.IsNaN(XMath.Special.Hypergeometric1F1(-4, -3, .5)));
        Assert.Equal(1, XMath.Special.Hypergeometric1F1(0, -3, 2));
        Assert.Equal(0, XMath.Special.Hypergeometric1F1Regularized(-2, -3, 1));
    }
}
