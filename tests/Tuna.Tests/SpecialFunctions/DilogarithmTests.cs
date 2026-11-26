using Tuna;

namespace Tuna.Tests;

// 有限参照値は mpmath 1.4.1、100 桁以上で生成し binary64 に丸めた。
// 入力も binary64 の値を正確に高精度へ変換（十進文字列の値とは区別）。
// 参照値生成: https://mpmath.org/doc/current/functions/index.html
[Trait("SpecialFunction", "Dilogarithm")]
public class DilogarithmTests
{

    // https://dlmf.nist.gov/25.12#i
    [Theory]
    // 原点。
    [InlineData(0.0, 0.0)]
    // Li2(x)≈xで原点の相対精度。
    [InlineData(1e-20, 1e-20)]
    // 原点の負側。
    [InlineData(-1e-20, -1e-20)]
    // 既知値-π²/12。
    [InlineData(-1.0, -0.8224670334241132)]
    // 反転公式へ切替直後。
    [InlineData(-1.0000000000000002, -0.8224670334241134)]
    // Bernoulli級数の負端。
    [InlineData(-0.9999999999999999, -0.8224670334241131)]
    // 既知値π²/12-ln²2/2と級数の端。
    [InlineData(0.5, 0.5822405264650125)]
    // 補引数変換の開始。
    [InlineData(0.5000000000000001, 0.5822405264650127)]
    // 対数項が特異になる端点の内側。
    [InlineData(0.9999999999999999, 1.6449340668482222)]
    // 既知値π²/6。
    [InlineData(1.0, 1.6449340668482264)]
    // 反転公式の大きい負の引数。
    [InlineData(-1e+308, -251481.2761102782)]
    public void Dilogarithm_MatchesReference(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.Dilogarithm(x), 5e-13, 0);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/25.12#i
    [Theory]
    // x>1で値が複素数。
    [InlineData(1.0000000000000002, double.NaN)]
    // 正の無限大は定義域外。
    [InlineData(double.PositiveInfinity, double.NaN)]
    // 負の無限大で-ln²(-x)/2。
    [InlineData(double.NegativeInfinity, double.NegativeInfinity)]
    // NaN。
    [InlineData(double.NaN, double.NaN)]
    public void Dilogarithm_SpecialValues(double x, double expected)
    {
        double actual = XMath.Special.Dilogarithm(x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // 0の符号は解析的な線形項の向きを保持。https://dlmf.nist.gov/25.12.E1
    [Fact]
    public void Dilogarithm_PreservesNegativeZero()
    {
        Assert.Equal(long.MinValue, BitConverter.DoubleToInt64Bits(XMath.Special.Dilogarithm(-0.0)));
    }

    // 探索で得た補強点。mpmath 1.4.1で独立に計算し、複数精度でbinary64丸めを確認。
    // 入力・期待値・許容誤差をここに固定する。出典と選定方針は README.md を参照。

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(-5249483887192480, -656.7529418259602)]
    [InlineData(-2.1937479035260472e+83, -18414.483357512217)]
    [InlineData(-9.138610590589905e+90, -21935.28386718143)]
    // 前回の探索で最大相対誤差となった入力。
    [InlineData(-322634711434747300, -814.3065263666557)]
    [InlineData(0.3686484976931238, 0.4097134229823484)]
    [InlineData(0.9652237088109507, 1.4909639655799969)]
    [InlineData(0.056853450764297375, 0.05768262590526616)]
    public void Dilogarithm_ExploredRelativeValues(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.Dilogarithm(x), relativeTolerance: 2e-12);
    }
}
