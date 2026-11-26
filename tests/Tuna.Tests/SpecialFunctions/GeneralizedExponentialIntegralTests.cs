using System.Numerics;
using Tuna;

namespace Tuna.Tests;

// mpmath 1.4.1による独立参照値。binary64入力を正確に変換し、
// 90/160桁（正規分布は490/560桁）でbinary64への丸め一致を確認。
// 生成式と選定理由: tools/SpecialFunctionReferences/generate_remaining_accuracy_references.py
// 通常テストはC#の固定値と対象APIの直接呼び出しだけで完結する。
[Trait("SpecialFunction", "GeneralizedExponentialIntegral")]
public class GeneralizedExponentialIntegralTests
{

    // 実数次数・整数次数の両隣・負次数・指数の小さい裾。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(-3.0, 0.1, 59999.76918996446)]
    [InlineData(-0.5, 1.0, 0.5072822338117733)]
    [InlineData(0.5, 0.1, 3.6697045911267447)]
    [InlineData(1.0000000000000002, 0.1, 1.8229239584193901)]
    [InlineData(1.9999999999999998, 0.5, 0.32664386232455306)]
    [InlineData(3.0, 2.0, 0.03013337979781589)]
    [InlineData(20.5, 50.0, 2.7469221335293936e-24)]
    [InlineData(100.0, 1.0, 0.003678422930396813)]
    [InlineData(0.5, 700.0, 1.4075212793384078e-307)]
    public void ExpIntegralE_MatchesIndependentReferences(double p, double x, double expected)
    {
        double actual = XMath.Special.ExpIntegralE(p, x);
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
    [InlineData(-3.0, double.NaN)]
    public void ExpIntegralE_RejectsNaN(double p, double x)
    {
        double actual = XMath.Special.ExpIntegralE(p, x);
        Assert.True(double.IsNaN(actual));
    }

    [Theory]
    [InlineData(.5, .2)]
    [InlineData(2.3, 2.0)]
    [InlineData(-2.3, 10.0)]
    public void RecurrenceAndEndpoints(double p, double x)
    {
        SpecialFunctionAssert.Close(Math.Exp(-x), p * XMath.Special.ExpIntegralE(p + 1, x) + x * XMath.Special.ExpIntegralE(p, x), relativeTolerance: 2e-12);
        Assert.Equal(.5, XMath.Special.ExpIntegralE(3.0, 0));
        Assert.Equal(double.PositiveInfinity, XMath.Special.ExpIntegralE(.5, 0));
        Assert.Equal(0, XMath.Special.ExpIntegralE(p, double.PositiveInfinity));
        Assert.True(double.IsNaN(XMath.Special.ExpIntegralE(p, -1)));
    }

    // E_p(x) = exp(-x)/p · (1 + O((1+x)/p))（省略項は相対 1e-99 未満）。x = 1 の隣接値で計算経路の切替を検証する（参照値は 100/180 桁で一致を確認）。
    [Trait("Category", "AuditRegression")]
    [Trait("Category", "CoreAudit20261102")]
    [Theory]
    [InlineData(1e+100, 0.1, 9.048374180359596e-101)]
    [InlineData(1e+100, 0.9999999999999999, 3.6787944117144236e-101)]
    [InlineData(1e+100, 1.0, 3.678794411714423e-101)]
    [InlineData(1e+100, 1.0000000000000002, 3.678794411714422e-101)]
    [InlineData(1e+100, 2.0, 1.3533528323661269e-101)]
    [InlineData(1e+307, 0.1, 9.048374180359596e-308)]
    [InlineData(1e+307, 0.9999999999999999, 3.6787944117144235e-308)]
    [InlineData(1e+307, 1.0, 3.678794411714423e-308)]
    [InlineData(1e+307, 1.0000000000000002, 3.6787944117144225e-308)]
    [InlineData(1e+307, 2.0, 1.353352832366127e-308)]
    [InlineData(1e+308, 0.1, 9.048374180359596e-309)]
    [InlineData(1e+308, 0.9999999999999999, 3.678794411714426e-309)]
    [InlineData(1e+308, 1.0, 3.67879441171442e-309)]
    [InlineData(1e+308, 1.0000000000000002, 3.67879441171442e-309)]
    [InlineData(1e+308, 2.0, 1.353352832366126e-309)]
    public void ExpIntegralE_HugeOrderAndUnitArgumentNeighbors_FixedReferences(double p, double x, double expected)
    {
        double actual = XMath.Special.ExpIntegralE(p, x);
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
