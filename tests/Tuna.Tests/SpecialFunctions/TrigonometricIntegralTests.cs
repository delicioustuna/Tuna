using Tuna;

namespace Tuna.Tests;

// 有限参照値は mpmath 1.4.1、100 桁以上で生成し binary64 に丸めた。
// 入力も binary64 の値を正確に高精度へ変換（十進文字列の値とは区別）。
// 参照値生成: https://mpmath.org/doc/current/functions/index.html
[Trait("SpecialFunction", "TrigonometricIntegral")]
public class TrigonometricIntegralTests
{

    // https://dlmf.nist.gov/6.2; https://dlmf.nist.gov/6.6; https://dlmf.nist.gov/6.12
    [Theory]
    // 原点近傍で線形項または対数項。
    [InlineData(1e-20, 1e-20)]
    // 既知の通常値。
    [InlineData(1.0, 0.946083070367183)]
    // Si/Ci級数の末端。
    [InlineData(2.0, 1.6054129768026948)]
    // Si/Ci連分数の切替直後。
    [InlineData(2.0000000000000004, 1.605412976802695)]
    // 振動または増大領域。
    [InlineData(10.0, 1.6583475942188741)]
    // 奇関数の負側。
    [InlineData(-1.0, -0.946083070367183)]
    // 原点。
    [InlineData(0.0, 0.0)]
    public void SinIntegral_MatchesReference(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.SinIntegral(x), 2e-12, 0);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/6.2; https://dlmf.nist.gov/6.6; https://dlmf.nist.gov/6.12
    [Theory]
    // NaN。
    [InlineData(double.NaN, double.NaN)]
    // 正の無限大。
    [InlineData(double.PositiveInfinity, 1.5707963267948966)]
    // 負の無限大。
    [InlineData(double.NegativeInfinity, -1.5707963267948966)]
    public void SinIntegral_SpecialValues(double x, double expected)
    {
        double actual = XMath.Special.SinIntegral(x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // https://dlmf.nist.gov/6.2; https://dlmf.nist.gov/6.6; https://dlmf.nist.gov/6.12
    [Theory]
    // 原点近傍で線形項または対数項。
    [InlineData(1e-20, -45.474486194979384)]
    // 既知の通常値。
    [InlineData(1.0, 0.33740392290096816)]
    // Si/Ci級数の末端。
    [InlineData(2.0, 0.422980828774865)]
    // Si/Ci連分数の切替直後。
    [InlineData(2.0000000000000004, 0.4229808287748649)]
    // 振動または増大領域。
    [InlineData(10.0, -0.04545643300445537)]
    public void CosIntegral_MatchesReference(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.CosIntegral(x), 2e-12, 0);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/6.2; https://dlmf.nist.gov/6.6; https://dlmf.nist.gov/6.12
    [Theory]
    // NaN。
    [InlineData(double.NaN, double.NaN)]
    // 正の無限大。
    [InlineData(double.PositiveInfinity, 0.0)]
    // 原点の対数極。
    [InlineData(0.0, double.NegativeInfinity)]
    // 負側は複素数になるためNaN。
    [InlineData(-1.0, double.NaN)]
    public void CosIntegral_SpecialValues(double x, double expected)
    {
        double actual = XMath.Special.CosIntegral(x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // https://dlmf.nist.gov/6.2; https://dlmf.nist.gov/6.6; https://dlmf.nist.gov/6.12
    [Theory]
    // 原点近傍で線形項または対数項。
    [InlineData(1e-20, 1e-20)]
    // 既知の通常値。
    [InlineData(1.0, 1.0572508753757286)]
    // Si/Ci級数の末端。
    [InlineData(2.0, 2.5015674333549756)]
    // Si/Ci連分数の切替直後。
    [InlineData(2.0000000000000004, 2.5015674333549764)]
    // 振動または増大領域。
    [InlineData(10.0, 1246.1144901994232)]
    // 級数の末端。
    [InlineData(50.0, 5.292818448565845e+19)]
    // Ei表現へ切替直後。
    [InlineData(50.00000000000001, 5.292818448565882e+19)]
    // 中間expが溢れる付近の有限値。
    [InlineData(710.0, 1.5754578441031007e+305)]
    // 奇関数の負側。
    [InlineData(-1.0, -1.0572508753757286)]
    // 原点。
    [InlineData(0.0, 0.0)]
    public void SinhIntegral_MatchesReference(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.SinhIntegral(x), 2e-12, 0);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/6.2; https://dlmf.nist.gov/6.6; https://dlmf.nist.gov/6.12
    [Theory]
    // NaN。
    [InlineData(double.NaN, double.NaN)]
    // 正の無限大。
    [InlineData(double.PositiveInfinity, double.PositiveInfinity)]
    // 負の無限大。
    [InlineData(double.NegativeInfinity, double.NegativeInfinity)]
    public void SinhIntegral_SpecialValues(double x, double expected)
    {
        double actual = XMath.Special.SinhIntegral(x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // https://dlmf.nist.gov/6.2; https://dlmf.nist.gov/6.6; https://dlmf.nist.gov/6.12
    [Theory]
    // 原点近傍で線形項または対数項。
    [InlineData(1e-20, -45.474486194979384)]
    // 既知の通常値。
    [InlineData(1.0, 0.8378669409802082)]
    // Si/Ci級数の末端。
    [InlineData(2.0, 2.4526669226469147)]
    // Si/Ci連分数の切替直後。
    [InlineData(2.0000000000000004, 2.4526669226469155)]
    // 振動または増大領域。
    [InlineData(10.0, 1246.1144860424545)]
    // 級数の末端。
    [InlineData(50.0, 5.292818448565845e+19)]
    // Ei表現へ切替直後。
    [InlineData(50.00000000000001, 5.292818448565882e+19)]
    // 中間expが溢れる付近の有限値。
    [InlineData(710.0, 1.5754578441031007e+305)]
    public void CoshIntegral_MatchesReference(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.CoshIntegral(x), 2e-12, 0);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/6.2; https://dlmf.nist.gov/6.6; https://dlmf.nist.gov/6.12
    [Theory]
    // NaN。
    [InlineData(double.NaN, double.NaN)]
    // 正の無限大。
    [InlineData(double.PositiveInfinity, double.PositiveInfinity)]
    // 原点の対数極。
    [InlineData(0.0, double.NegativeInfinity)]
    // 負側は複素数になるためNaN。
    [InlineData(-1.0, double.NaN)]
    public void CoshIntegral_SpecialValues(double x, double expected)
    {
        double actual = XMath.Special.CoshIntegral(x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // https://dlmf.nist.gov/6.2; https://dlmf.nist.gov/6.6; https://dlmf.nist.gov/6.12
    [Theory]
    // Ciの最初の零点付近は絶対誤差で確認。
    [InlineData(0.6165054856207162, -9.121512798937272e-17)]
    // Ciの最初の零点付近は絶対誤差で確認。
    [InlineData(0.6165054856207163, 5.571548945612863e-17)]
    // Ciの最初の零点付近は絶対誤差で確認。
    [InlineData(0.6165054856207164, 2.0264610690162993e-16)]
    public void CosIntegral_NearZero_MatchesReference(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.CosIntegral(x), 5e-13, 2e-14);
    }

    // R11: Ei の正項級数を scale=10^80 のBigInteger固定小数点で2500項まで評価。
    // Ei(717)/2との差(E1/2、定数項)は結果の半ulpを大幅に下回る。
    // 正しく丸めた参照値は .tmp の ReviewProbe で生成。https://dlmf.nist.gov/6.6; https://dlmf.nist.gov/6.5
    [Trait("Category", "ReviewRegression")]
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void HyperbolicIntegrals_NearOverflow_RemainFinite(bool cosh)
    {
        double actual = cosh ? XMath.Special.CoshIntegral(717) : XMath.Special.SinhIntegral(717);
        SpecialFunctionAssert.Close(1.7108083299905491e308, actual);
    }

    // 探索で得た補強点。mpmath 1.4.1で独立に計算し、複数精度でbinary64丸めを確認。
    // 入力・期待値・許容誤差をここに固定する。出典と選定方針は README.md を参照。

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(12.464412527640365, 1.4925767379256643)]
    [InlineData(4.2827040201810313e-10, 4.2827040201810313e-10)]
    [InlineData(170.9896481846892, 1.5694460429533847)]
    [InlineData(0.03734165003956854, 0.037338757429849906)]
    [InlineData(2.087976013128887e-10, 2.087976013128887e-10)]
    [InlineData(3.5739818592496713, 1.8250728600345365)]
    public void SinIntegral_ExploredRelativeValues(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.SinIntegral(x), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(12.464412527640365, -0.014249163877563477)]
    [InlineData(4.2827040201810313e-10, -20.99405067451832)]
    [InlineData(170.9896481846892, 0.005689984483352104)]
    [InlineData(0.03734165003956854, -2.7107788667998367)]
    [InlineData(2.087976013128887e-10, -21.71244008304961)]
    // 前回の探索で最大相対誤差となった入力。
    [InlineData(3.5739818592496713, -0.05142864089688742)]
    [InlineData(37.62099816524341, -0.002772758953451138)]
    public void CosIntegral_ExploredRelativeValues(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.CosIntegral(x), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(12.464412527640365, 11406.765839373007)]
    [InlineData(4.2827040201810313e-10, 4.2827040201810313e-10)]
    [InlineData(170.9896481846892, 5.350857793673607e+71)]
    [InlineData(2.275976544422687, 3.0422353109015297)]
    [InlineData(0.03734165003956854, 0.03734454289130439)]
    [InlineData(2.087976013128887e-10, 2.087976013128887e-10)]
    public void SinhIntegral_ExploredRelativeValues(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.SinhIntegral(x), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(12.464412527640365, 11406.765839084796)]
    [InlineData(4.2827040201810313e-10, -20.99405067451832)]
    [InlineData(170.9896481846892, 5.350857793673607e+71)]
    [InlineData(0.03734165003956854, -2.7100816673847428)]
    [InlineData(2.087976013128887e-10, -21.71244008304961)]
    [InlineData(0.6206170920866825, 0.19802470666486338)]
    public void CoshIntegral_ExploredRelativeValues(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.CoshIntegral(x), relativeTolerance: 2e-12);
    }
}
