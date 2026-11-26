using Tuna;

namespace Tuna.Tests;

// 有限参照値は mpmath 1.4.1、100 桁以上で生成し binary64 に丸めた。
// 入力も binary64 の値を正確に高精度へ変換（十進文字列の値とは区別）。
// 参照値生成: https://mpmath.org/doc/current/functions/index.html
[Trait("SpecialFunction", "ExponentialIntegral")]
public class ExponentialIntegralTests
{

    // https://dlmf.nist.gov/6.2; https://dlmf.nist.gov/6.12; https://dlmf.nist.gov/8.19
    [Theory]
    // 0次はexp(-x)/x。
    [InlineData(0, 0.5, 1.2130613194252668)]
    // E1の対数特異点の近く。
    [InlineData(1, 1e-20, 45.474486194979384)]
    // 級数にDigammaを含む項。
    [InlineData(2, 0.25, 0.5177301244604703)]
    // 高次数の級数。
    [InlineData(10, 0.25, 0.0839219939386743)]
    // 級数の末端。
    [InlineData(1, 1.0, 0.21938393439552029)]
    // 連分数へ切替直後。
    [InlineData(1, 1.0000000000000002, 0.2193839343955202)]
    // 高次数の連分数。
    [InlineData(10, 1.0000000000000002, 0.03639399403141639)]
    // 極小の有限値。
    [InlineData(1, 700.0, 1.406518766234033e-307)]
    // subnormalの尾部。
    [InlineData(1, 730.0, 1.2623e-320)]
    public void ExpIntegralE_MatchesReference(int n, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.ExpIntegralE(n, x), 5e-13, double.Epsilon);
    }

    // https://dlmf.nist.gov/6.2; https://dlmf.nist.gov/6.12; https://dlmf.nist.gov/8.19
    [Theory]
    // 負側は-E1(-x)。
    [InlineData(-10.0, -4.156968929685325e-06)]
    // 原点の負側で対数発散。
    [InlineData(-1e-20, -45.474486194979384)]
    // 原点の正側で対数発散。
    [InlineData(1e-20, -45.474486194979384)]
    // 零点の左隣で専用Taylor展開の相対精度。
    [InlineData(0.37250741078136657, -2.6748041020008385e-16)]
    // 丸めた零点。
    [InlineData(0.3725074107813666, -5.1196989365556847e-17)]
    // 零点の右隣。
    [InlineData(0.3725074107813667, 1.6508643146897012e-16)]
    // Taylor領域の外側。
    [InlineData(0.29250641078136663, -0.3366862645707971)]
    // 正項級数の末端。
    [InlineData(50.0, 1.058563689713169e+20)]
    // 漸近展開の切替直後。
    [InlineData(50.00000000000001, 1.0585636897131764e+20)]
    // 指数分割の切替。
    [InlineData(700.0, 1.4509787360525608e+301)]
    // exp(x)単体が溢れてもEiは有限。
    [InlineData(710.0, 3.1509156882062014e+305)]
    public void ExpIntegralEi_MatchesReference(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.ExpIntegralEi(x), 2e-12, 0);
    }

    // https://dlmf.nist.gov/6.2; https://dlmf.nist.gov/6.12; https://dlmf.nist.gov/8.19
    [Theory]
    // x<1で負の値。
    [InlineData(0.5, -0.37867104306108795)]
    // x=1の特異点の左隣。
    [InlineData(0.9999999999999999, -36.15958490477557)]
    // 特異点の右隣。
    [InlineData(1.0000000000000002, -35.46643772421562)]
    // オフセット付きLiでなくliの値。
    [InlineData(2.0, 1.045163780117493)]
    // 大引数。
    [InlineData(1e+100, 4.361971987140703e+97)]
    public void LogIntegral_MatchesReference(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.LogIntegral(x), 5e-13, 0);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/6.2; https://dlmf.nist.gov/6.12; https://dlmf.nist.gov/8.19
    [Theory]
    // 負の次数。
    [InlineData(-1, 1.0, double.NaN)]
    // 0次の原点発散。
    [InlineData(0, 0.0, double.PositiveInfinity)]
    // 1次の原点発散。
    [InlineData(1, 0.0, double.PositiveInfinity)]
    // 2次で有限な原点値。
    [InlineData(2, 0.0, 1.0)]
    // 3次で有限な原点値。
    [InlineData(3, 0.0, 0.5)]
    // 負の引数。
    [InlineData(1, -1.0, double.NaN)]
    // 正の無限大。
    [InlineData(1, double.PositiveInfinity, 0.0)]
    // NaN。
    [InlineData(1, double.NaN, double.NaN)]
    public void ExpIntegralE_SpecialValues(int n, double x, double expected)
    {
        double actual = XMath.Special.ExpIntegralE(n, x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/6.2; https://dlmf.nist.gov/6.12; https://dlmf.nist.gov/8.19
    [Theory]
    // 原点の対数極。
    [InlineData(0.0, double.NegativeInfinity)]
    // 正の無限大。
    [InlineData(double.PositiveInfinity, double.PositiveInfinity)]
    // 負の無限大の減衰。
    [InlineData(double.NegativeInfinity, 0.0)]
    // NaN。
    [InlineData(double.NaN, double.NaN)]
    public void ExpIntegralEi_SpecialValues(double x, double expected)
    {
        double actual = XMath.Special.ExpIntegralEi(x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/6.2; https://dlmf.nist.gov/6.12; https://dlmf.nist.gov/8.19
    [Theory]
    // 原点の連続拡張。
    [InlineData(0.0, 0.0)]
    // x=1の対数極。
    [InlineData(1.0, double.NegativeInfinity)]
    // 負の引数。
    [InlineData(-1.0, double.NaN)]
    // 正の無限大。
    [InlineData(double.PositiveInfinity, double.PositiveInfinity)]
    // NaN。
    [InlineData(double.NaN, double.NaN)]
    public void LogIntegral_SpecialValues(double x, double expected)
    {
        double actual = XMath.Special.LogIntegral(x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // 探索で得た補強点。mpmath 1.4.1で独立に計算し、複数精度でbinary64丸めを確認。
    // 入力・期待値・許容誤差をここに固定する。出典と選定方針は README.md を参照。

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(56, 0.00478329143693907, 0.018093454208314156)]
    [InlineData(51, 7.851625492219366e-8, 0.019999998397627516)]
    [InlineData(56, 313.3897491524596, 2.134292214637163e-139)]
    [InlineData(76, 0.1613575711337037, 0.011321816468212372)]
    [InlineData(46, 0.00001223548901418967, 0.02222194414466722)]
    [InlineData(10, 2.669366734680934e-7, 0.11111107774403202)]
    public void ExpIntegralE_ExploredRelativeValues(int n, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.ExpIntegralE(n, x), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(-213.64394413119447, -7.654285763866641e-96)]
    [InlineData(-242.90975335866608, -1.313447112292439e-108)]
    [InlineData(276.43110141917873, 4.0972716711952505e+117)]
    [InlineData(618.7088424789354, 8.148046840681401e+265)]
    [InlineData(-332.9445965066742, -7.591514193549094e-148)]
    [InlineData(-508.9133698893304, -1.880342467901008e-224)]
    public void ExpIntegralEi_ExploredRelativeValues(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.ExpIntegralEi(x), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(2.78374003282533e+179, 6.753632920101687e+176)]
    [InlineData(1.981757234182293e-34, -2.521592475072812e-36)]
    [InlineData(9.103249182419019e+113, 3.482539186011758e+111)]
    [InlineData(5.979852073913731e-126, -2.0667651501714276e-128)]
    [InlineData(4.917577864698098e-28, -7.700033026471436e-30)]
    // 前回の探索で最大相対誤差となった入力。
    [InlineData(1.0731768523654114e+187, 2.4977833421434652e+184)]
    public void LogIntegral_ExploredRelativeValues(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.LogIntegral(x), relativeTolerance: 2e-12);
    }
}
