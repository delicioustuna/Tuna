using System.Numerics;
using Tuna;

namespace Tuna.Tests;

// mpmath 1.4.1による独立参照値。binary64入力を正確に変換し、
// 90/160桁（正規分布は490/560桁）でbinary64への丸め一致を確認。
// 生成式と選定理由: tools/SpecialFunctionReferences/generate_remaining_accuracy_references.py
// 通常テストはC#の固定値と対象APIの直接呼び出しだけで完結する。
[Trait("SpecialFunction", "Hypergeometric0F1")]
public class Hypergeometric0F1Tests
{

    // 正負引数・負の非整数パラメータ・分母の極の両隣・大引数。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(1.0, 0.0, 1.0)]
    [InlineData(0.5, 1.0, 3.7621956910836314)]
    [InlineData(1.5, -10.0, 0.006539307734329603)]
    [InlineData(-0.5, 0.2, 0.516372514583539)]
    [InlineData(10.0, 100.0, 2052.95425683801)]
    [InlineData(100.0, -1000.0, 2.554798380348188e-05)]
    [InlineData(-2.0000000000000004, 1.0, -239523900289852.84)]
    [InlineData(-1.9999999999999998, 1.0, 479047800579708.4)]
    [InlineData(1.0, 10000.0, 2.0396871734097245e+85)]
    public void Hypergeometric0F1_MatchesIndependentReferences(double b, double x, double expected)
    {
        double actual = XMath.Special.Hypergeometric0F1(b, x);
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
    [InlineData(double.NaN, 0.0)]
    [InlineData(1.0, double.NaN)]
    public void Hypergeometric0F1_RejectsNaN(double b, double x)
    {
        double actual = XMath.Special.Hypergeometric0F1(b, x);
        Assert.True(double.IsNaN(actual));
    }

    [Theory]
    [InlineData(-2.0)]
    [InlineData(0.0)]
    [InlineData(1.0)]
    public void ClosedFormsAndPoles(double x)
    {
        double expected = x >= 0 ? Math.Cosh(2 * Math.Sqrt(x)) : Math.Cos(2 * Math.Sqrt(-x));
        SpecialFunctionAssert.Close(expected, XMath.Special.Hypergeometric0F1(.5, x), relativeTolerance: 2e-12);
        Assert.Equal(1, XMath.Special.Hypergeometric0F1(2, 0));
        Assert.True(double.IsNaN(XMath.Special.Hypergeometric0F1(0, x)));
        Assert.True(double.IsNaN(XMath.Special.Hypergeometric0F1(-1, x)));
    }

    // x/b を固定した巨大な正の b において、各項の分母 (b+k)(k+1) の中間オーバーフローを回避できることを検証する（参照値は 100/180 桁で一致を確認）。
    [Trait("Category", "AuditRegression")]
    [Trait("Category", "CoreAudit20261102")]
    [Theory]
    [InlineData(1e+100, -1e+100, 0.36787944117144233)]
    [InlineData(1e+100, 1.0000000000000001e+99, 1.1051709180756477)]
    [InlineData(1e+100, 1e+100, 2.718281828459045)]
    [InlineData(1e+307, -1e+307, 0.36787944117144233)]
    [InlineData(1e+307, 1e+306, 1.1051709180756477)]
    [InlineData(1e+307, 1e+307, 2.718281828459045)]
    [InlineData(8.988465674311578e+307, -8.988465674311578e+307, 0.36787944117144233)]
    [InlineData(8.988465674311578e+307, 8.988465674311578e+306, 1.1051709180756477)]
    [InlineData(8.988465674311578e+307, 8.988465674311578e+307, 2.718281828459045)]
    [InlineData(8.988465674311579e+307, -8.988465674311579e+307, 0.36787944117144233)]
    [InlineData(8.988465674311579e+307, 8.988465674311579e+306, 1.1051709180756477)]
    [InlineData(8.988465674311579e+307, 8.988465674311579e+307, 2.718281828459045)]
    [InlineData(8.98846567431158e+307, -8.98846567431158e+307, 0.36787944117144233)]
    [InlineData(8.98846567431158e+307, 8.98846567431158e+306, 1.1051709180756477)]
    [InlineData(8.98846567431158e+307, 8.98846567431158e+307, 2.718281828459045)]
    [InlineData(1e+308, -1e+308, 0.36787944117144233)]
    [InlineData(1e+308, 1.0000000000000001e+307, 1.1051709180756477)]
    [InlineData(1e+308, 1e+308, 2.718281828459045)]
    public void Hypergeometric0F1_LargeBalancedParameters_FixedReferences(double b, double x, double expected)
    {
        double actual = XMath.Special.Hypergeometric0F1(b, x);
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
