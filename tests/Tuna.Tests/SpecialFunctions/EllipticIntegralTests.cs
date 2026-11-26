using Tuna;

namespace Tuna.Tests;

// 有限参照値は mpmath 1.4.1、100 桁以上で生成し binary64 に丸めた。
// 入力も binary64 の値を正確に高精度へ変換（十進文字列の値とは区別）。
// 参照値生成: https://mpmath.org/doc/current/functions/index.html
[Trait("SpecialFunction", "EllipticIntegral")]
public class EllipticIntegralTests
{
    // 負の最大有限引数、巨大な負の値、0と1の近傍。mpmath 1.3.0 の100桁と180桁で一致。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(-1.7976931348623157e+308, 2.6572401146362276e-152)]
    [InlineData(-1e+308, 3.559843986822029e-152)]
    [InlineData(-1e+100, 1.1651554901082217e-48)]
    [InlineData(1e-16, 1.5707963267948966)]
    [InlineData(0.9999999999999999, 19.75469464595844)]
    public void EllipticK_WideParameterRange_PreservesDoublePrecision(double m, double expected)
    {
        SpecialFunctionAssert.WithinUlps(expected, XMath.Special.EllipticK(m), 8);
    }

    // R_F - m R_D/3 の差では桁落ちする、m → 1 の値。参照値は100桁以上で計算した値。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0.9999999999991549, 1.000000000006248)]
    [InlineData(0.9999392226818927, 1.0001744478857217)]
    [InlineData(0.9999999718600495, 1.0000001347809673)]
    [InlineData(0.9999999999881201, 1.000000000079978)]
    [InlineData(0.9999999999988732, 1.0000000000082492)]
    [InlineData(0.9999999999999999, 1.000000000000001)]
    public void EllipticE_NearOne_PreservesDoublePrecision(double m, double expected)
    {
        SpecialFunctionAssert.WithinUlps(expected, XMath.Special.EllipticE(m), 8);
    }

    // https://dlmf.nist.gov/19.16; https://dlmf.nist.gov/19.18; https://dlmf.nist.gov/19.20
    [Theory]
    // 引数が一致する退化形の既知値。
    [InlineData(1.0, 1.0, 1.0, 1.0)]
    // 1つだけ0は有限。
    [InlineData(0.0, 1.0, 2.0, 1.3110287771460598)]
    // 非対称な通常値。
    [InlineData(1.0, 2.0, 3.0, 0.7269459354689082)]
    // 順序を入れ替えた通常値。
    [InlineData(3.0, 1.0, 2.0, 0.7269459354689082)]
    // 引数の大きさが大きく異なる。
    [InlineData(1e-20, 1.0, 2.0, 1.3110287770753493)]
    // RF(t,t,t)=1/√t。途中の和だけが溢れる大引数。
    [InlineData(1e+308, 1e+308, 1e+308, 1e-154)]
    // RFの同次性と小さい引数。
    [InlineData(1e-200, 1e-200, 1e-200, 1e+100)]
    public void CarlsonRF_MatchesReference(double x, double y, double z, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.CarlsonRF(x, y, z), 2e-12, 0);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/19.16; https://dlmf.nist.gov/19.18; https://dlmf.nist.gov/19.20
    [Theory]
    // 負のx。
    [InlineData(-1.0, 1.0, 1.0, double.NaN)]
    // 負のy。
    [InlineData(1.0, -1.0, 1.0, double.NaN)]
    // 負のz。
    [InlineData(1.0, 1.0, -1.0, double.NaN)]
    // x NaN。
    [InlineData(double.NaN, 1.0, 1.0, double.NaN)]
    // y NaN。
    [InlineData(1.0, double.NaN, 1.0, double.NaN)]
    // z NaN。
    [InlineData(1.0, 1.0, double.NaN, double.NaN)]
    // 無限大引数の規約。
    [InlineData(double.PositiveInfinity, 1.0, 1.0, 0.0)]
    // 二つの0で発散。
    [InlineData(0.0, 0.0, 1.0, double.PositiveInfinity)]
    public void CarlsonRF_SpecialValues(double x, double y, double z, double expected)
    {
        double actual = XMath.Special.CarlsonRF(x, y, z);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // https://dlmf.nist.gov/19.16; https://dlmf.nist.gov/19.18; https://dlmf.nist.gov/19.20
    [Theory]
    // 引数が一致する退化形の既知値。
    [InlineData(1.0, 1.0, 1.0, 1.0)]
    // 1つだけ0は有限。
    [InlineData(0.0, 1.0, 2.0, 1.0679379896673957)]
    // 非対称な通常値。
    [InlineData(1.0, 2.0, 3.0, 0.29046028102899063)]
    // 順序を入れ替えた通常値。
    [InlineData(3.0, 1.0, 2.0, 0.3751723729571543)]
    // 引数の大きさが大きく異なる。
    [InlineData(1e-20, 1.0, 2.0, 1.0679379895613297)]
    // RD(t,t,t)=t^(-3/2) の極小有限値。
    [InlineData(1e+200, 1e+200, 1e+200, 1e-300)]
    // 同次性による巨大有限値。
    [InlineData(1e-100, 1e-100, 1e-100, 1e+150)]
    public void CarlsonRD_MatchesReference(double x, double y, double z, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.CarlsonRD(x, y, z), 2e-12, 0);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/19.16; https://dlmf.nist.gov/19.18; https://dlmf.nist.gov/19.20
    [Theory]
    // 負のx。
    [InlineData(-1.0, 1.0, 1.0, double.NaN)]
    // 負のy。
    [InlineData(1.0, -1.0, 1.0, double.NaN)]
    // 負のz。
    [InlineData(1.0, 1.0, -1.0, double.NaN)]
    // x NaN。
    [InlineData(double.NaN, 1.0, 1.0, double.NaN)]
    // y NaN。
    [InlineData(1.0, double.NaN, 1.0, double.NaN)]
    // z NaN。
    [InlineData(1.0, 1.0, double.NaN, double.NaN)]
    // 無限大引数の規約。
    [InlineData(double.PositiveInfinity, 1.0, 1.0, 0.0)]
    // 二つの0で発散。
    [InlineData(0.0, 0.0, 1.0, double.PositiveInfinity)]
    // 特別な引数z=0で発散。
    [InlineData(1.0, 1.0, 0.0, double.PositiveInfinity)]
    public void CarlsonRD_SpecialValues(double x, double y, double z, double expected)
    {
        double actual = XMath.Special.CarlsonRD(x, y, z);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // https://dlmf.nist.gov/19.16; https://dlmf.nist.gov/19.18; https://dlmf.nist.gov/19.20
    [Theory]
    // 引数が一致する退化形の既知値。
    [InlineData(1.0, 1.0, 1.0, 1.0)]
    // 1つだけ0は有限。
    [InlineData(0.0, 1.0, 2.0, 0.955049447256928)]
    // 非対称な通常値。
    [InlineData(1.0, 2.0, 3.0, 1.4018470999908952)]
    // 順序を入れ替えた通常値。
    [InlineData(3.0, 1.0, 2.0, 1.4018470999908952)]
    // 引数の大きさが大きく異なる。
    [InlineData(1e-20, 1.0, 2.0, 0.955049447256928)]
    // 二つの0でもRGは√z/2で有限。
    [InlineData(0.0, 0.0, 4.0, 1.0)]
    // RG(t,t,t)=√t。xyだけが溢れる条件。
    [InlineData(1e+200, 1e+200, 1e+200, 1e+100)]
    public void CarlsonRG_MatchesReference(double x, double y, double z, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.CarlsonRG(x, y, z), 2e-12, 0);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/19.16; https://dlmf.nist.gov/19.18; https://dlmf.nist.gov/19.20
    [Theory]
    // 負のx。
    [InlineData(-1.0, 1.0, 1.0, double.NaN)]
    // 負のy。
    [InlineData(1.0, -1.0, 1.0, double.NaN)]
    // 負のz。
    [InlineData(1.0, 1.0, -1.0, double.NaN)]
    // x NaN。
    [InlineData(double.NaN, 1.0, 1.0, double.NaN)]
    // y NaN。
    [InlineData(1.0, double.NaN, 1.0, double.NaN)]
    // z NaN。
    [InlineData(1.0, 1.0, double.NaN, double.NaN)]
    // 無限大引数の規約。
    [InlineData(double.PositiveInfinity, 1.0, 1.0, double.PositiveInfinity)]
    // 全零の連続拡張。
    [InlineData(0.0, 0.0, 0.0, 0.0)]
    public void CarlsonRG_SpecialValues(double x, double y, double z, double expected)
    {
        double actual = XMath.Special.CarlsonRG(x, y, z);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // https://dlmf.nist.gov/19.16; https://dlmf.nist.gov/19.18; https://dlmf.nist.gov/19.20
    [Theory]
    // 退化形π/2。
    [InlineData(0.0, 1.0, 1.5707963267948966)]
    // 同じ引数では1。
    [InlineData(1.0, 1.0, 1.0)]
    // 逆双曲線関数に帰着。
    [InlineData(2.0, 1.0, 0.881373587019543)]
    // 負のyはCauchy主値。
    [InlineData(1.0, -1.0, 0.6232252401402305)]
    // 主値が0になる退化形。
    [InlineData(0.0, -1.0, 0.0)]
    // 同次性で大引数。
    [InlineData(1e+200, 1e+200, 1e-100)]
    public void CarlsonRC_MatchesReference(double x, double y, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.CarlsonRC(x, y), 5e-13, 0);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/19.16; https://dlmf.nist.gov/19.18; https://dlmf.nist.gov/19.20
    [Theory]
    // y=0で発散。
    [InlineData(1.0, 0.0, double.PositiveInfinity)]
    // x<0。
    [InlineData(-1.0, 1.0, double.NaN)]
    // x NaN。
    [InlineData(double.NaN, 1.0, double.NaN)]
    // y NaN。
    [InlineData(1.0, double.NaN, double.NaN)]
    // 無限大引数。
    [InlineData(double.PositiveInfinity, 1.0, 0.0)]
    public void CarlsonRC_SpecialValues(double x, double y, double expected)
    {
        double actual = XMath.Special.CarlsonRC(x, y);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // https://dlmf.nist.gov/19.16; https://dlmf.nist.gov/19.18; https://dlmf.nist.gov/19.20
    [Theory]
    // 全引数一致で1。
    [InlineData(1.0, 1.0, 1.0, 1.0, 1.0)]
    // 1つだけ0は有限。
    [InlineData(0.0, 1.0, 2.0, 3.0, 0.7768862377858233)]
    // p=z でRDに一致。
    [InlineData(1.0, 2.0, 3.0, 3.0, 0.29046028102899063)]
    // pが小さい正の値。
    [InlineData(1.0, 2.0, 3.0, 0.5, 0.7939649105046136)]
    // 負のpはCauchy主値。
    [InlineData(1.0, 2.0, 3.0, -0.5, 0.2072200111587186)]
    // 主値変換時の並べ替え。
    [InlineData(3.0, 1.0, 2.0, -0.5, 0.2072200111587186)]
    // p=0極の近傍。
    [InlineData(1.0, 2.0, 3.0, 1e-12, 16.87227136487185)]
    // 三次の中間量が溢れても最終結果はt^(-3/2)で有限。
    [InlineData(1e+150, 1e+150, 1e+150, 1e+150, 1e-225)]
    public void CarlsonRJ_MatchesReference(double x, double y, double z, double p, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.CarlsonRJ(x, y, z, p), 2e-12, 0);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/19.16; https://dlmf.nist.gov/19.18; https://dlmf.nist.gov/19.20
    [Theory]
    // p=0極。
    [InlineData(1.0, 1.0, 1.0, 0.0, double.PositiveInfinity)]
    // 二つの0で発散。
    [InlineData(0.0, 0.0, 1.0, 1.0, double.PositiveInfinity)]
    // 負のx。
    [InlineData(-1.0, 1.0, 1.0, 1.0, double.NaN)]
    // 負のy。
    [InlineData(1.0, -1.0, 1.0, 1.0, double.NaN)]
    // 負のz。
    [InlineData(1.0, 1.0, -1.0, 1.0, double.NaN)]
    // x NaN。
    [InlineData(double.NaN, 1.0, 1.0, 1.0, double.NaN)]
    // y NaN。
    [InlineData(1.0, double.NaN, 1.0, 1.0, double.NaN)]
    // z NaN。
    [InlineData(1.0, 1.0, double.NaN, 1.0, double.NaN)]
    // p NaN。
    [InlineData(1.0, 1.0, 1.0, double.NaN, double.NaN)]
    // p無限大。
    [InlineData(1.0, 1.0, 1.0, double.PositiveInfinity, 0.0)]
    public void CarlsonRJ_SpecialValues(double x, double y, double z, double p, double expected)
    {
        double actual = XMath.Special.CarlsonRJ(x, y, z, p);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // https://dlmf.nist.gov/19.2; https://dlmf.nist.gov/19.6; https://dlmf.nist.gov/19.12
    [Theory]
    // m=0でπ/2。m=k²の規約。
    [InlineData(0.0, 1.5707963267948966)]
    // 代表値。
    [InlineData(0.5, 1.8540746773013719)]
    // 負のパラメータも実数。
    [InlineData(-1.0, 1.3110287771460598)]
    // 大きい負のパラメータ。
    [InlineData(-1000000.0, 0.00829404781659062)]
    // m=1の対数特異点のすぐ内側。
    [InlineData(0.9999999999999999, 19.75469464595844)]
    public void EllipticK_MatchesReference(double m, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.EllipticK(m), 2e-12, 0);
    }

    // https://dlmf.nist.gov/19.2; https://dlmf.nist.gov/19.6; https://dlmf.nist.gov/19.12
    [Theory]
    // m=0でπ/2。m=k²の規約。
    [InlineData(0.0, 1.5707963267948966)]
    // 代表値。
    [InlineData(0.5, 1.3506438810476755)]
    // 負のパラメータも実数。
    [InlineData(-1.0, 1.910098894513856)]
    // 大きい負のパラメータ。
    [InlineData(-1000000.0, 1000.0043970243486)]
    // m=1の対数特異点のすぐ内側。
    [InlineData(0.9999999999999999, 1.000000000000001)]
    // m=1で有限な値1。
    [InlineData(1.0, 1.0)]
    public void EllipticE_MatchesReference(double m, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.EllipticE(m), 2e-12, 0);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/19.2; https://dlmf.nist.gov/19.6; https://dlmf.nist.gov/19.12
    [Theory]
    // m=1の端点。
    [InlineData(1.0, double.PositiveInfinity)]
    // m>1で複素数。
    [InlineData(1.0000000000000002, double.NaN)]
    // 負の無限大の極限。
    [InlineData(double.NegativeInfinity, 0.0)]
    // NaN。
    [InlineData(double.NaN, double.NaN)]
    public void EllipticK_SpecialValues(double m, double expected)
    {
        double actual = XMath.Special.EllipticK(m);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/19.2; https://dlmf.nist.gov/19.6; https://dlmf.nist.gov/19.12
    [Theory]
    // m=1の端点。
    [InlineData(1.0, 1.0)]
    // m>1で複素数。
    [InlineData(1.0000000000000002, double.NaN)]
    // 負の無限大の極限。
    [InlineData(double.NegativeInfinity, double.PositiveInfinity)]
    // NaN。
    [InlineData(double.NaN, double.NaN)]
    public void EllipticE_SpecialValues(double m, double expected)
    {
        double actual = XMath.Special.EllipticE(m);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // https://dlmf.nist.gov/19.2; https://dlmf.nist.gov/19.6; https://dlmf.nist.gov/19.12
    [Theory]
    // n=0はKに一致。
    [InlineData(0.0, 0.5, 1.8540746773013719)]
    // 特性とパラメータの順序。
    [InlineData(0.25, 0.5, 2.167619360766556)]
    // 負の特性。
    [InlineData(-2.0, 0.5, 1.0226975086065924)]
    // n>1のCauchy主値。
    [InlineData(2.0, 0.5, -0.31354468346518405)]
    // 特性n=1の極に接近。
    [InlineData(0.9999999999999999, 0.5, 210828713.28594348)]
    // m=1の特異点近傍。
    [InlineData(0.25, 0.9999999999999999, 25.97338876505522)]
    public void EllipticPi_MatchesReference(double n, double m, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.EllipticPi(n, m), 2e-12, 0);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/19.2; https://dlmf.nist.gov/19.6; https://dlmf.nist.gov/19.12
    [Theory]
    // m=1で正の発散。
    [InlineData(0.25, 1.0, double.PositiveInfinity)]
    // n=1,m<1の被積分関数は正で、符号の決まる極。
    [InlineData(1.0, 0.5, double.PositiveInfinity)]
    // m>1。
    [InlineData(0.25, 1.0000000000000002, double.NaN)]
    // n NaN。
    [InlineData(double.NaN, 0.5, double.NaN)]
    // m NaN。
    [InlineData(0.25, double.NaN, double.NaN)]
    public void EllipticPi_SpecialValues(double n, double m, double expected)
    {
        double actual = XMath.Special.EllipticPi(n, m);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // https://dlmf.nist.gov/19.2; https://dlmf.nist.gov/19.6; https://dlmf.nist.gov/19.12
    [Theory]
    // 零振幅。
    [InlineData(0.0, 0.5, 0.0)]
    // m=0の初等関数。
    [InlineData(0.5, 0.0, 0.5)]
    // 通常の不完全積分。
    [InlineData(0.5, 0.5, 0.5104671356280047)]
    // 振幅に関する奇関数。
    [InlineData(-0.5, 0.5, -0.5104671356280047)]
    // 完全積分への接続。
    [InlineData(1.5707963267948966, 0.5, 1.8540746773013719)]
    // 振幅周期還元の切替直前。
    [InlineData(1.5707963267948963, 0.5, 1.8540746773013714)]
    // 周期還元の切替直後。
    [InlineData(1.5707963267948968, 0.5, 1.854074677301372)]
    // 複数周期をまたぐ振幅。
    [InlineData(9.67477796076938, 0.5, 11.375752072462724)]
    // m>1でも分岐点を越えない振幅は実数。
    [InlineData(0.25, 2.0, 0.25545065914979276)]
    // 負のパラメータ。
    [InlineData(0.5, -2.0, 0.46693412001953666)]
    // m=1の有限な不完全積分。
    [InlineData(0.5, 1.0, 0.5222381032784403)]
    public void EllipticF_Incomplete_MatchesReference(double phi, double m, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.EllipticF(phi, m), 2e-12, 0);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/19.2; https://dlmf.nist.gov/19.6; https://dlmf.nist.gov/19.12
    [Theory]
    // m>1で実数の分岐点を越える振幅。
    [InlineData(1.0, 2.0, double.NaN)]
    // m>1の周期積分は複素数。
    [InlineData(3.141592653589793, 2.0, double.NaN)]
    // 無限大振幅は未対応。
    [InlineData(double.PositiveInfinity, 0.5, double.NaN)]
    // 振幅 NaN。
    [InlineData(double.NaN, 0.5, double.NaN)]
    // m NaN。
    [InlineData(0.5, double.NaN, double.NaN)]
    public void EllipticF_Incomplete_SpecialValues(double phi, double m, double expected)
    {
        double actual = XMath.Special.EllipticF(phi, m);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // https://dlmf.nist.gov/19.2; https://dlmf.nist.gov/19.6; https://dlmf.nist.gov/19.12
    [Theory]
    // 零振幅。
    [InlineData(0.0, 0.5, 0.0)]
    // m=0の初等関数。
    [InlineData(0.5, 0.0, 0.5)]
    // 通常の不完全積分。
    [InlineData(0.5, 0.5, 0.48991095979251714)]
    // 振幅に関する奇関数。
    [InlineData(-0.5, 0.5, -0.48991095979251714)]
    // 完全積分への接続。
    [InlineData(1.5707963267948966, 0.5, 1.3506438810476755)]
    // 振幅周期還元の切替直前。
    [InlineData(1.5707963267948963, 0.5, 1.3506438810476753)]
    // 周期還元の切替直後。
    [InlineData(1.5707963267948968, 0.5, 1.3506438810476755)]
    // 複数周期をまたぐ振幅。
    [InlineData(9.67477796076938, 0.5, 8.352571392090658)]
    // m>1でも分岐点を越えない振幅は実数。
    [InlineData(0.25, 2.0, 0.24475717385517395)]
    // 負のパラメータ。
    [InlineData(0.5, -2.0, 0.5372356189385809)]
    // m=1の有限な不完全積分。
    [InlineData(0.5, 1.0, 0.479425538604203)]
    public void EllipticE_Incomplete_MatchesReference(double phi, double m, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.EllipticE(phi, m), 2e-12, 0);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/19.2; https://dlmf.nist.gov/19.6; https://dlmf.nist.gov/19.12
    [Theory]
    // m>1で実数の分岐点を越える振幅。
    [InlineData(1.0, 2.0, double.NaN)]
    // m>1の周期積分は複素数。
    [InlineData(3.141592653589793, 2.0, double.NaN)]
    // 無限大振幅は未対応。
    [InlineData(double.PositiveInfinity, 0.5, double.NaN)]
    // 振幅 NaN。
    [InlineData(double.NaN, 0.5, double.NaN)]
    // m NaN。
    [InlineData(0.5, double.NaN, double.NaN)]
    public void EllipticE_Incomplete_SpecialValues(double phi, double m, double expected)
    {
        double actual = XMath.Special.EllipticE(phi, m);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // https://dlmf.nist.gov/19.2; https://dlmf.nist.gov/19.6; https://dlmf.nist.gov/19.12
    [Theory]
    // n=0でFへ接続。
    [InlineData(0.0, 0.5, 0.5, 0.5104671356280047)]
    // 通常値。
    [InlineData(0.25, 0.5, 0.5, 0.5211231226642417)]
    // 負の振幅。
    [InlineData(0.25, -0.5, 0.5, -0.5211231226642417)]
    // 完全積分へ接続。
    [InlineData(0.25, 1.5707963267948966, 0.5, 2.167619360766556)]
    // 周期補正。
    [InlineData(0.25, 3.391592653589793, 0.5, 4.587852795664929)]
    // 積分路内の極はCauchy主値。
    [InlineData(2.0, 1.0, 0.5, 0.7045837467687983)]
    public void EllipticPi_Incomplete_MatchesReference(double n, double phi, double m, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.EllipticPi(n, phi, m), 2e-12, 0);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/19.2; https://dlmf.nist.gov/19.6; https://dlmf.nist.gov/19.12
    [Theory]
    // n NaN。
    [InlineData(double.NaN, 0.5, 0.5, double.NaN)]
    // 振幅 NaN。
    [InlineData(0.25, double.NaN, 0.5, double.NaN)]
    // m NaN。
    [InlineData(0.25, 0.5, double.NaN, double.NaN)]
    // 実数の分岐点を越える振幅。
    [InlineData(0.25, 1.0, 2.0, double.NaN)]
    public void EllipticPi_Incomplete_SpecialValues(double n, double phi, double m, double expected)
    {
        double actual = XMath.Special.EllipticPi(n, phi, m);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // 対角値RF/RC=v^(-1/2)、RD/RJ=v^(-3/2)、RG=√vは独立な厳密恒等式。
    // 正規化の開始点と両隣、最小非正規数、最大有限値、最終overflow/underflow。
    // https://dlmf.nist.gov/19.16; https://dlmf.nist.gov/19.20
    public static IEnumerable<object[]> CarlsonDiagonalBoundaryInputs()
    {
        foreach (var (function, limit) in new[] { ("RF", 1020), ("RC", 1020), ("RD", 600), ("RJ", 300), ("RG", 500) })
        {
            foreach (int exponent in new[] { -limit, limit + 1 })
            {
                double boundary = Math.ScaleB(1, exponent);
                yield return new object[] { function, Math.BitDecrement(boundary) };
                yield return new object[] { function, boundary };
                yield return new object[] { function, Math.BitIncrement(boundary) };
            }

            yield return new object[] { function, double.Epsilon };
            yield return new object[] { function, double.MaxValue };
            if (function is "RD" or "RJ")
            {
                foreach (int exponent in new[] { -684, -682, 714, 716 })
                {
                    yield return new object[] { function, Math.ScaleB(1, exponent) };
                }
            }
        }
    }

    [Theory]
    [MemberData(nameof(CarlsonDiagonalBoundaryInputs))]
    public void Carlson_DiagonalBoundaryValues_MatchIdentity(string function, double v)
    {
        double inverseRoot = 1 / Math.Sqrt(v);
        double expected = function switch
        {
            "RF" or "RC" => inverseRoot,
            "RD" or "RJ" => inverseRoot / v,
            "RG" => Math.Sqrt(v),
            _ => throw new ArgumentException(function)
        };
        double actual = function switch
        {
            "RF" => XMath.Special.CarlsonRF(v, v, v),
            "RC" => XMath.Special.CarlsonRC(v, v),
            "RD" => XMath.Special.CarlsonRD(v, v, v),
            "RJ" => XMath.Special.CarlsonRJ(v, v, v, v),
            "RG" => XMath.Special.CarlsonRG(v, v, v),
            _ => throw new ArgumentException(function)
        };
        if (expected == 0 || double.IsInfinity(expected))
        {
            Assert.Equal(expected, actual);
        }
        else
        {
            Assert.True(actual > 0, $"{function}({v:R}) must remain nonzero");
            SpecialFunctionAssert.Close(expected, actual, 2e-12, double.Epsilon);
        }
    }

    // 対角値だけでは、引数ごとの正規化漏れや負引数の主値変換を検出できない。
    // 既存の独立参照値を同次性で拡大・縮小する。https://dlmf.nist.gov/19.18
    [Theory]
    [InlineData(-1022)]
    [InlineData(1022)]
    public void CarlsonRF_RC_ExtremeScaling_MatchesHomogeneity(int exponent)
    {
        double v = Math.ScaleB(1, exponent);
        SpecialFunctionAssert.Close(Math.ScaleB(0.7269459354689082, -exponent / 2), XMath.Special.CarlsonRF(v, 2 * v, 3 * v), 2e-12);
        SpecialFunctionAssert.Close(Math.ScaleB(0.881373587019543, -exponent / 2), XMath.Special.CarlsonRC(2 * v, v), 2e-12);
        SpecialFunctionAssert.Close(Math.ScaleB(0.6232252401402305, -exponent / 2), XMath.Special.CarlsonRC(v, -v), 2e-12);
    }

    [Theory]
    [InlineData(-602)]
    [InlineData(-502)]
    [InlineData(-302)]
    [InlineData(302)]
    [InlineData(502)]
    [InlineData(602)]
    public void CarlsonRD_RJ_RG_Scaling_MatchesHomogeneity(int exponent)
    {
        double v = Math.ScaleB(1, exponent);
        SpecialFunctionAssert.Close(Math.ScaleB(0.29046028102899063, -3 * exponent / 2), XMath.Special.CarlsonRD(v, 2 * v, 3 * v), 2e-12);
        SpecialFunctionAssert.Close(Math.ScaleB(0.7939649105046136, -3 * exponent / 2), XMath.Special.CarlsonRJ(v, 2 * v, 3 * v, 0.5 * v), 2e-12);
        SpecialFunctionAssert.Close(Math.ScaleB(0.2072200111587186, -3 * exponent / 2), XMath.Special.CarlsonRJ(v, 2 * v, 3 * v, -0.5 * v), 2e-12);
        SpecialFunctionAssert.Close(Math.ScaleB(1.4018470999908952, exponent / 2), XMath.Special.CarlsonRG(v, 2 * v, 3 * v), 2e-12);
    }

    // 微小な2引数と巨大な1引数。正規化によるsubnormalの丸めを検出。
    // RF(t,t,z)=atanh(√(1-t/z))/√(z-t)、750桁で評価してbinary64へ丸めた。
    // https://dlmf.nist.gov/19.20
    [Fact]
    public void CarlsonRF_DisparateSubnormalInputs_MatchesReference()
    {
        SpecialFunctionAssert.Close(7.2370487194712994e-152, XMath.Special.CarlsonRF(1e-320, 1e-320, 1e308), 2e-12);
    }

    [Theory]
    [InlineData(1.0, -1.0, double.PositiveInfinity)]
    [InlineData(1.0, 0.0, double.PositiveInfinity)]
    [InlineData(1.0, 1.0, double.PositiveInfinity)]
    [InlineData(1.0000000000000002, 1.0, double.NaN)]
    [InlineData(1.0, 1.0000000000000002, double.NaN)]
    public void EllipticPi_SingularCorner_HasExpectedValue(double n, double m, double expected)
    {
        double actual = XMath.Special.EllipticPi(n, m);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    private const double RelativeTolerance = 2e-12;

    // R12: Π(n|0)=π/(2sqrt(1-n)) (n<1)。積分のtan置換から得られる閉形式。
    // https://dlmf.nist.gov/19.2
    [Trait("Category", "ReviewRegression")]
    [Theory]
    [InlineData(-1e4)]
    [InlineData(-1e8)]
    [InlineData(-1e16)]
    [InlineData(-1e308)]
    public void EllipticPi_ZeroParameterAndNegativeCharacteristic_MatchesClosedForm(double n)
    {
        SpecialFunctionAssert.Close(Math.PI / (2 * Math.Sqrt(1 - n)), XMath.Special.EllipticPi(n, 0));
    }

    // 正規数の入力でもα,βは先に0へ落ちる。負のpは主値の別経路。
    // 参照はRJ(x,y,y,p)=3(RC(x,y)-RC(x,p))/(p-y)とRCの初等関数表示から生成。
    [Trait("Category", "AuditRegression")]
    [Theory]
    [InlineData(1e200, 1, 1, 1e-180, 6.216979751083923e-98)]
    [InlineData(1e160, 1e160, 1e160, 1e-160, 1.1043202861788218e-237)]
    [InlineData(1e-100, 1e150, 1e150, -1e150, -2.356194490192345e-225)]
    public void CarlsonRJ_DisparateNormalArguments_MatchesReference(double x, double y, double z,
        double p, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.CarlsonRJ(x, y, z, p), RelativeTolerance);
    }

    // RGの対数的に大きい項同士の引き算。1ulp保証は要求せず、普通の誤差基準で検査。
    [Trait("Category", "AuditRegression")]
    [Theory]
    [InlineData(1e-12, 1, 1e12, 500000.00000367546)]
    [InlineData(1e-200, 1, 1e200, 5e99)]
    public void CarlsonRG_DisparateNormalArguments_MatchesReferenceInEveryOrder(double x, double y, double z,
        double expected)
    {
        foreach (var (a, b, c) in new[] { (x, y, z), (x, z, y), (y, x, z), (y, z, x), (z, x, y), (z, y, x) })
        {
            SpecialFunctionAssert.Close(expected, XMath.Special.CarlsonRG(a, b, c), RelativeTolerance);
        }
    }

    // 不完全 Π の極端な負の n における相対精度検証（参照値は mpmath 1.4.1 の 160/240 桁、m = 0 は閉形式でも確認）。
    [Trait("Category", "AuditRegression")]
    [Theory]
    [InlineData(-1e8, 0, 0.0001570559796705687)]
    [InlineData(-1e16, 0, 1.5707963031426724e-8)]
    [InlineData(-1e308, 0, 1.5707963267948967e-154)]
    [InlineData(-1e8, 0.5, 0.00015705699942617104)]
    [InlineData(-1e16, 0.5, 1.5707963041628204e-8)]
    [InlineData(-1e308, 0.5, 1.5707963267948967e-154)]
    public void IncompleteEllipticPi_NegativeCharacteristic_MatchesReference(double n,
        double m, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.EllipticPi(n, 0.4, m), relativeTolerance: 1e-15);
    }

    // 探索で得た補強点。mpmath 1.4.1で独立に計算し、複数精度でbinary64丸めを確認。
    // 入力・期待値・許容誤差をここに固定する。出典と選定方針は README.md を参照。

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(3.5845935131778256e+98, 3.7223090430178707e+50, 1.0004668221595188e+30, 2.9910430954571516e-48)]
    [InlineData(4.798952980659008e+104, 1.7341799679444636e-19, 3.0935251355372316e+147, 9.110538652627154e-73)]
    [InlineData(7.530221874249904e-127, 8.003048850009654e+48, 1.1509608202105073e+107, 2.0144820375144257e-52)]
    [InlineData(3.5371908585791505e+23, 5.804969959606028e-79, 2.7670204884700994e-118, 1.9936421024074692e-10)]
    [InlineData(1.8888242385548885e-109, 1.3192956969849538e-120, 1.4422014355183406e-109, 3.8620782058246794e+54)]
    [InlineData(1.2779408265213438e-102, 1.4622274611763025e+134, 1.1405957772460317e+73, 5.932666533362968e-66)]
    public void CarlsonRF_ExploredRelativeValues(double x, double y, double z, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.CarlsonRF(x, y, z), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(3.5845935131778256e+98, 3.7223090430178707e+50, 1.0004668221595188e+30, 8.210956252504912e-90)]
    [InlineData(4.798952980659008e+104, 1.7341799679444636e-19, 3.0935251355372316e+147, 8.660746330388951e-220)]
    [InlineData(7.530221874249904e-127, 8.003048850009654e+48, 1.1509608202105073e+107, 5.173953593626115e-159)]
    [InlineData(3.5371908585791505e+23, 5.804969959606028e-79, 2.7670204884700994e-118, 3.9800252925947304e+86)]
    [InlineData(1.8888242385548885e-109, 1.3192956969849538e-120, 1.4422014355183406e-109, 3.881517570619811e+163)]
    [InlineData(1.2779408265213438e-102, 1.4622274611763025e+134, 1.1405957772460317e+73, 2.175114018451803e-140)]
    public void CarlsonRD_ExploredRelativeValues(double x, double y, double z, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.CarlsonRD(x, y, z), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(3.5845935131778256e+98, 3.7223090430178707e+50, 1.0004668221595188e+30, 9.466511386431943e+48)]
    [InlineData(4.798952980659008e+104, 1.7341799679444636e-19, 3.0935251355372316e+147, 2.7809733617643802e+73)]
    [InlineData(7.530221874249904e-127, 8.003048850009654e+48, 1.1509608202105073e+107, 1.696290673948975e+53)]
    [InlineData(3.5371908585791505e+23, 5.804969959606028e-79, 2.7670204884700994e-118, 297371436867.2263)]
    [InlineData(1.8888242385548885e-109, 1.3192956969849538e-120, 1.4422014355183406e-109, 3.2016478190964624e-55)]
    [InlineData(1.2779408265213438e-102, 1.4622274611763025e+134, 1.1405957772460317e+73, 6.0461298802959536e+66)]
    public void CarlsonRG_ExploredRelativeValues(double x, double y, double z, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.CarlsonRG(x, y, z), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(4.648454609202243e-79, 1.5159975601732864e+146, 1.275764831203344e-73)]
    [InlineData(18283643400.008976, 3.093511405869363e+82, 8.930875110384563e-42)]
    [InlineData(1.3355742780806962e-65, 5.719290095644529e+28, 6.568236989965711e-15)]
    [InlineData(6.805587238267745e-147, 1.6666061289682213e+66, 1.2167557008696607e-33)]
    [InlineData(1.0215579881019258e+134, 6.07585376152572e+120, 1.5750872749348348e-66)]
    [InlineData(8.851513940972044e-94, 3.203109096853441e-104, 4.273501674922193e+47)]
    public void CarlsonRC_ExploredRelativeValues(double x, double y, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.CarlsonRC(x, y), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(3.8766972131104506e-35, 3.5254128929243905e-83, 1.9492539654636447e-98, 9.431138813446317e-83, 6.940102379278275e+99)]
    [InlineData(4.519277481402449e+71, 50061324.91929615, 9.444019091406818e-63, 1.0142316113127564e-34, 9.837548512281052e-23)]
    [InlineData(9.62107943970624e-69, 7.071936257075444e+60, 9.305634860223202e+24, 2.133426571074008e+66, 2.257502697161858e-95)]
    [InlineData(5.629141040633539e+72, 1.866571626033672e-33, 1.8223996148757833e-51, 1.8620698897334555e+27, 4.7377242174739935e-62)]
    [InlineData(6.264316029596907e+40, 9.807202790814348e+81, 4.756608390704074e+99, 7.714897847607599e-68, 2.192426397650467e-109)]
    [InlineData(6.467531937406445e-77, 2.201996547911279e-93, 3.544314351554672e+82, 9.56500392843282e-62, 3.0251201426666144e+21)]
    public void CarlsonRJ_ExploredRelativeValues(double x, double y, double z, double p, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.CarlsonRJ(x, y, z, p), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(0.2761237405809186, 1.7001832076348586)]
    [InlineData(0.9999999718600495, 10.079332197494052)]
    [InlineData(0.9999392226818927, 6.240520936768468)]
    [InlineData(0.9999999999881201, 13.964379225334824)]
    [InlineData(0.9999999999988732, 15.142129417238015)]
    [InlineData(-9952.28600631738, 0.06003286984519493)]
    public void EllipticK_ExploredRelativeValues(double m, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.EllipticK(m), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(0.2761237405809186, 1.4559843976379718)]
    [InlineData(0.9999999718600495, 1.0000001347809673)]
    [InlineData(0.9999392226818927, 1.0001744478857217)]
    // 前回の探索で最大相対誤差となった入力。
    [InlineData(0.9999999999881201, 1.000000000079978)]
    [InlineData(0.9999999999988732, 1.0000000000082492)]
    [InlineData(-9952.28600631738, 99.79366749139092)]
    public void EllipticE_ExploredRelativeValues(double m, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.EllipticE(m), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(-68.24543603050284, -9.706464374325535, 0.15822572248274858)]
    [InlineData(-0.023967797837784587, -0.5876639224061133, 1.3792260516689347)]
    [InlineData(-0.0209805829193535, -8.164881101676945, 0.8316643345506414)]
    [InlineData(-0.00000789621988723653, -2.3118342907124783, 1.1388311137915277)]
    [InlineData(-1765886.168746879, -9.974748459210725, 0.001180449712508)]
    [InlineData(-9.225115932675251, 0.7260467402428645, 0.5637827624607106)]
    public void EllipticPi_ExploredRelativeValues(double n, double m, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.EllipticPi(n, m), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(-13.031802946381159, -8.355025651917703, -7.045692889142019)]
    [InlineData(-17.71599363082374, -7.062662181244523, -9.755108768985771)]
    [InlineData(15.989756868487284, -1.1483960616927735, 13.134687850709616)]
    [InlineData(-7.999669966893862, -4.473681053947958, -4.974521980209801)]
    [InlineData(-4.060224565575066, -9.956175306865058, -2.164685915393273)]
    [InlineData(0.848438500930353, -7.123784406278997, 0.59543176563277)]
    public void EllipticF_ExploredRelativeValues(double phi, double m, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.EllipticF(phi, m), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(-13.031802946381159, -8.355025651917703, -27.746718712695806)]
    [InlineData(-17.71599363082374, -7.062662181244523, -36.310555851395385)]
    [InlineData(15.989756868487284, -1.1483960616927735, 19.824957077200576)]
    [InlineData(-7.999669966893862, -4.473681053947958, -13.987473154430624)]
    [InlineData(-4.060224565575066, -9.956175306865058, -8.875238468010258)]
    [InlineData(0.848438500930353, -7.123784406278997, 1.293615353148036)]
    public void EllipticE_ExploredRelativeValues_2Arguments(double phi, double m, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.EllipticE(phi, m), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(-0.00002698541330064537, -5.580140446115934, 0.06698292478232837, -5.685871686144077)]
    [InlineData(-2232.8671766098996, -7.371773013092675, -3.7414762786800546, -0.16271189638625214)]
    [InlineData(-200.22115513409364, -0.6622037361811106, -2.1272291222500304, -0.1019381563753629)]
    [InlineData(-0.00006819458010140409, 8.023646585403988, -2.1107005750169616, 5.893396729960171)]
    [InlineData(-0.00002336213641415442, -3.118418683719206, -2.0764560036409376, -2.3029515882107376)]
    [InlineData(-0.00003515944791655217, -9.59092332819548, -9.660637339876887, -4.952482824156832)]
    public void EllipticPi_ExploredRelativeValues_3Arguments(double n, double phi, double m, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.EllipticPi(n, phi, m), relativeTolerance: 2e-12);
    }
}
