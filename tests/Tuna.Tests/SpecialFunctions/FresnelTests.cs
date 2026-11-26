using Tuna;

namespace Tuna.Tests;

// 有限参照値は mpmath 1.4.1、100 桁以上で生成し binary64 に丸めた。
// 入力も binary64 の値を正確に高精度へ変換（十進文字列の値とは区別）。
// 参照値生成: https://mpmath.org/doc/current/functions/index.html
[Trait("SpecialFunction", "Fresnel")]
public class FresnelTests
{

    // https://dlmf.nist.gov/7.2#iii; https://dlmf.nist.gov/7.12#ii
    [Theory]
    // 原点の既知値。
    [InlineData(0.0, 0.0)]
    // Sの先頭はπx³/6、Cの先頭はx。
    [InlineData(1e-10, 5.235987755982989e-31)]
    // 奇関数の符号。
    [InlineData(-1.0, -0.43825914739035476)]
    // πt²/2 の正規化を確認。
    [InlineData(1.0, 0.43825914739035476)]
    // 級数の切替直前。
    [InlineData(1.4999999999999998, 0.6975049600820931)]
    // 複素連分数へ切替。
    [InlineData(1.5, 0.6975049600820931)]
    // 切替直後。
    [InlineData(1.5000000000000002, 0.697504960082093)]
    // 振動領域。
    [InlineData(10.0, 0.46816997858488224)]
    // x²の丸め誤差が位相に効く大引数。
    [InlineData(1000000.25, 0.49999968322294214)]
    // 専用大引数枝の境界。
    [InlineData(9007199254740992.0, 0.49999999999999994)]
    public void FresnelS_MatchesReference(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.FresnelS(x), 5e-13, 0);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/7.2#iii; https://dlmf.nist.gov/7.12#ii
    [Theory]
    // 正の無限大の極限。
    [InlineData(double.PositiveInfinity, 0.5)]
    // 負の無限大の極限。
    [InlineData(double.NegativeInfinity, -0.5)]
    // NaN。
    [InlineData(double.NaN, double.NaN)]
    public void FresnelS_SpecialValues(double x, double expected)
    {
        double actual = XMath.Special.FresnelS(x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // https://dlmf.nist.gov/7.2#iii; https://dlmf.nist.gov/7.12#ii
    [Theory]
    // 原点の既知値。
    [InlineData(0.0, 0.0)]
    // Sの先頭はπx³/6、Cの先頭はx。
    [InlineData(1e-10, 1e-10)]
    // 奇関数の符号。
    [InlineData(-1.0, -0.7798934003768229)]
    // πt²/2 の正規化を確認。
    [InlineData(1.0, 0.7798934003768229)]
    // 級数の切替直前。
    [InlineData(1.4999999999999998, 0.44526117603982174)]
    // 複素連分数へ切替。
    [InlineData(1.5, 0.4452611760398215)]
    // 切替直後。
    [InlineData(1.5000000000000002, 0.44526117603982135)]
    // 振動領域。
    [InlineData(10.0, 0.49989869420551575)]
    // x²の丸め誤差が位相に効く大引数。
    [InlineData(1000000.25, 0.500000031199817)]
    // 専用大引数枝の境界。
    [InlineData(9007199254740992.0, 0.5)]
    public void FresnelC_MatchesReference(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.FresnelC(x), 5e-13, 0);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/7.2#iii; https://dlmf.nist.gov/7.12#ii
    [Theory]
    // 正の無限大の極限。
    [InlineData(double.PositiveInfinity, 0.5)]
    // 負の無限大の極限。
    [InlineData(double.NegativeInfinity, -0.5)]
    // NaN。
    [InlineData(double.NaN, double.NaN)]
    public void FresnelC_SpecialValues(double x, double expected)
    {
        double actual = XMath.Special.FresnelC(x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // 同時計算APIでもπ/2による正規化を独立参照値で確認。https://dlmf.nist.gov/7.2#iii
    [Fact]
    public void Fresnel_CombinedAtOne()
    {
        var value = XMath.Special.Fresnel(1);
        SpecialFunctionAssert.Close(0.43825914739035476, value.S);
        SpecialFunctionAssert.Close(0.7798934003768228, value.C);
        var nan = XMath.Special.Fresnel(double.NaN);
        Assert.True(double.IsNaN(nan.S) && double.IsNaN(nan.C));
    }

    // 探索で得た補強点。mpmath 1.4.1で独立に計算し、複数精度でbinary64丸めを確認。
    // 入力・期待値・許容誤差をここに固定する。出典と選定方針は README.md を参照。

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(4600.092003818454, 0.5000530668687418)]
    [InlineData(-1436.4335857573533, -0.5001429319834361)]
    [InlineData(-9913.518757577498, -0.5000311391569507)]
    [InlineData(9053.984462427845, 0.5000185628054508)]
    [InlineData(0.00019411922127006147, 3.830048236396408e-12)]
    [InlineData(1.3153285289087743e-12, 1.191520014931351e-36)]
    public void FresnelS_ExploredRelativeValues(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.FresnelS(x), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(4600.092003818454, 0.4999555922122012)]
    // 前回の探索で最大相対誤差となった入力。
    [InlineData(-7601.290177545381, -0.5000226182500072)]
    [InlineData(-1436.4335857573533, -0.5001693394129056)]
    [InlineData(-9913.518757577498, -0.4999921693238479)]
    [InlineData(9053.984462427845, 0.4999701431953671)]
    [InlineData(0.00019411922127006147, 0.00019411922127006139)]
    [InlineData(1.3153285289087743e-12, 1.3153285289087743e-12)]
    public void FresnelC_ExploredRelativeValues(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.FresnelC(x), relativeTolerance: 2e-12);
    }
}
