using System.Numerics;
using Tuna;

namespace Tuna.Tests;

// 有限参照値は mpmath 1.4.1、100 桁以上で生成し binary64 に丸めた。
// 入力も binary64 の値を正確に高精度へ変換（十進文字列の値とは区別）。
// 参照値生成: https://mpmath.org/doc/current/functions/index.html
[Trait("SpecialFunction", "Bessel")]
public class BesselTests
{

    // https://dlmf.nist.gov/10.7; https://dlmf.nist.gov/10.17; https://dlmf.nist.gov/10.27; https://dlmf.nist.gov/10.30; https://dlmf.nist.gov/10.40
    [Theory]
    // 整数0次でTemme補助量のμ=0。
    [InlineData(0.0, 1.0, 0.7651976865579666)]
    // 整数1次の正規化。
    [InlineData(1.0, 1.0, 0.4400505857449335)]
    // 半整数は初等関数に帰着。
    [InlineData(0.5, 1.0, 0.6713967071418031)]
    // 次数を最近接整数へ減らす切替直前。
    [InlineData(0.49999999999999994, 0.25, 0.3947995987413701)]
    // 次数切替直後。
    [InlineData(0.5000000000000001, 0.25, 0.39479959874136994)]
    // 原点近くの冪の振舞い。
    [InlineData(0.25, 1e-10, 0.0029337386158861294)]
    // Temme級数の末端直前。
    [InlineData(0.25, 1.9999999999999998, 0.3978110643381785)]
    // Steed法への切替。
    [InlineData(0.25, 2.0, 0.39781106433817837)]
    // 漸近展開の直前。
    [InlineData(0.25, 24.999999999999996, 0.040436476712673176)]
    // 漸近展開の開始。
    [InlineData(0.25, 25.0, 0.04043647671267372)]
    // x=ν²の切替直前。
    [InlineData(10.0, 99.99999999999999, -0.054732176935471194)]
    // 次数依存の漸近展開の境界。
    [InlineData(10.0, 100.0, -0.05473217693547201)]
    // ν≫x でJ/Iが極小、Y/Kが巨大。
    [InlineData(50.0, 1.0, 2.9060049481732392e-80)]
    // x≈ν の転移域。
    [InlineData(50.0, 50.0, 0.12140902189761506)]
    // 負の整数次数で零係数による反射。
    [InlineData(-1.0, 1.0, -0.4400505857449335)]
    // 負の実数次数の反射。
    [InlineData(-0.25, 1.0, 0.6693848172615745)]
    // 負の次数の符号。
    [InlineData(-1.25, 0.25, -2.9146219660160684)]
    // 大引数で位相還元を保持。
    [InlineData(0.5, 10000000000.0, -3.889735307156991e-06)]
    // 整数次数だけは負のxも実数で、奇数次の符号を確認。
    [InlineData(1.0, -1.0, -0.4400505857449335)]
    // 負のxで偶数次。
    [InlineData(2.0, -1.0, 0.11490348493190047)]
    // 次数0の単独経路の切替点。mpmath 1.3.0 の100桁と180桁の参照値が binary64 で一致。
    [InlineData(0.0, 1.4901161193847653e-08, 1.0)]
    [InlineData(0.0, 1.490116119384766e-08, 0.9999999999999999)]
    [InlineData(0.0, 1.9999999999999998, 0.22389077914123579)]
    [InlineData(0.0, 2.0, 0.22389077914123567)]
    [InlineData(0.0, 24.999999999999996, 0.09626678327595767)]
    [InlineData(0.0, 25.0, 0.09626678327595811)]
    public void BesselJ_MatchesReference(double nu, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.BesselJ(nu, x), 2e-12, double.Epsilon);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/10.7; https://dlmf.nist.gov/10.17; https://dlmf.nist.gov/10.27; https://dlmf.nist.gov/10.30; https://dlmf.nist.gov/10.40
    [Theory]
    // 次数NaN。
    [InlineData(double.NaN, 1.0, double.NaN)]
    // 引数NaN。
    [InlineData(0.0, double.NaN, double.NaN)]
    // 無限大次数。
    [InlineData(double.PositiveInfinity, 1.0, double.NaN)]
    // 負の非整数引数は複素数。
    [InlineData(0.25, -1.0, double.NaN)]
    // 正の無限大引数。
    [InlineData(0.0, double.PositiveInfinity, 0.0)]
    // 0次の原点値。
    [InlineData(0.0, 0.0, 1.0)]
    // 正の次数の原点値。
    [InlineData(1.0, 0.0, 0.0)]
    // 負の整数次数の原点値。
    [InlineData(-1.0, 0.0, 0.0)]
    // 負の半整数の正の片側極。
    [InlineData(-0.5, 0.0, double.PositiveInfinity)]
    // 隣の負半整数の負の片側極。
    [InlineData(-1.5, 0.0, double.NegativeInfinity)]
    public void BesselJ_SpecialValues(double nu, double x, double expected)
    {
        double actual = XMath.Special.BesselJ(nu, x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // https://dlmf.nist.gov/10.7; https://dlmf.nist.gov/10.17; https://dlmf.nist.gov/10.27; https://dlmf.nist.gov/10.30; https://dlmf.nist.gov/10.40
    [Theory]
    // 整数0次でTemme補助量のμ=0。
    [InlineData(0.0, 1.0, 0.08825696421567696)]
    // 整数1次の正規化。
    [InlineData(1.0, 1.0, -0.7812128213002887)]
    // 半整数は初等関数に帰着。
    [InlineData(0.5, 1.0, -0.4310988680183761)]
    // 次数を最近接整数へ減らす切替直前。
    [InlineData(0.49999999999999994, 0.25, -1.546160524106077)]
    // 次数切替直後。
    [InlineData(0.5000000000000001, 0.25, -1.5461605241060772)]
    // 原点近くの冪の振舞い。
    [InlineData(0.25, 1e-10, -433.9960387126444)]
    // Temme級数の末端直前。
    [InlineData(0.25, 1.9999999999999998, 0.392738399615385)]
    // Steed法への切替。
    [InlineData(0.25, 2.0, 0.39273839961538504)]
    // 漸近展開の直前。
    [InlineData(0.25, 24.999999999999996, -0.15435631659425936)]
    // 漸近展開の開始。
    [InlineData(0.25, 25.0, -0.15435631659425922)]
    // x=ν²の切替直前。
    [InlineData(10.0, 99.99999999999999, 0.058331574236415704)]
    // 次数依存の漸近展開の境界。
    [InlineData(10.0, 100.0, 0.058331574236414926)]
    // ν≫x でJ/Iが極小、Y/Kが巨大。
    [InlineData(50.0, 1.0, -2.191142812605339e+77)]
    // x≈ν の転移域。
    [InlineData(50.0, 50.0, -0.21031655464397742)]
    // 負の整数次数で零係数による反射。
    [InlineData(-1.0, 1.0, 0.7812128213002887)]
    // 負の実数次数の反射。
    [InlineData(-0.25, 1.0, 0.3944309363909674)]
    // 負の次数の符号。
    [InlineData(-1.25, 0.25, 2.822491715033366)]
    // 大引数で位相還元を保持。
    [InlineData(0.5, 10000000000.0, -6.966486666678868e-06)]
    // 次数0の単独経路の切替点。mpmath 1.3.0 の100桁と180桁の参照値が binary64 で一致。
    [InlineData(0.0, 1.4901161193847653e-08, -11.546855503046569)]
    [InlineData(0.0, 1.490116119384766e-08, -11.546855503046569)]
    [InlineData(0.0, 1.9999999999999998, 0.5103756726497451)]
    [InlineData(0.0, 2.0, 0.5103756726497451)]
    [InlineData(0.0, 24.999999999999996, -0.1272494322680065)]
    [InlineData(0.0, 25.0, -0.12724943226800614)]
    public void BesselY_MatchesReference(double nu, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.BesselY(nu, x), 2e-12, double.Epsilon);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/10.7; https://dlmf.nist.gov/10.17; https://dlmf.nist.gov/10.27; https://dlmf.nist.gov/10.30; https://dlmf.nist.gov/10.40
    [Theory]
    // 次数NaN。
    [InlineData(double.NaN, 1.0, double.NaN)]
    // 引数NaN。
    [InlineData(0.0, double.NaN, double.NaN)]
    // 無限大次数。
    [InlineData(double.PositiveInfinity, 1.0, double.NaN)]
    // 負の非整数引数は複素数。
    [InlineData(0.25, -1.0, double.NaN)]
    // 正の無限大引数。
    [InlineData(0.0, double.PositiveInfinity, 0.0)]
    // 0次原点の対数極。
    [InlineData(0.0, 0.0, double.NegativeInfinity)]
    // 正次数の原点極。
    [InlineData(1.0, 0.0, double.NegativeInfinity)]
    // 負整数次数の符号。
    [InlineData(-1.0, 0.0, double.PositiveInfinity)]
    // 負半整数は原点で有限な零。
    [InlineData(-0.5, 0.0, 0.0)]
    public void BesselY_SpecialValues(double nu, double x, double expected)
    {
        double actual = XMath.Special.BesselY(nu, x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // https://dlmf.nist.gov/10.7; https://dlmf.nist.gov/10.17; https://dlmf.nist.gov/10.27; https://dlmf.nist.gov/10.30; https://dlmf.nist.gov/10.40
    [Theory]
    // 整数0次でTemme補助量のμ=0。
    [InlineData(0.0, 1.0, 1.2660658777520084)]
    // 整数1次の正規化。
    [InlineData(1.0, 1.0, 0.565159103992485)]
    // 半整数は初等関数に帰着。
    [InlineData(0.5, 1.0, 0.9376748882454876)]
    // 次数を最近接整数へ減らす切替直前。
    [InlineData(0.49999999999999994, 0.25, 0.40311093489975935)]
    // 次数切替直後。
    [InlineData(0.5000000000000001, 0.25, 0.4031109348997592)]
    // 原点近くの冪の振舞い。
    [InlineData(0.25, 1e-10, 0.0029337386158861294)]
    // Temme級数の末端直前。
    [InlineData(0.25, 1.9999999999999998, 2.2033544516736296)]
    // Steed法への切替。
    [InlineData(0.25, 2.0, 2.20335445167363)]
    // 漸近展開の直前。
    [InlineData(0.25, 24.999999999999996, 5767196111.738611)]
    // 漸近展開の開始。
    [InlineData(0.25, 25.0, 5767196111.738631)]
    // x=ν²の切替直前。
    [InlineData(10.0, 99.99999999999999, 6.498975524720056e+41)]
    // 次数依存の漸近展開の境界。
    [InlineData(10.0, 100.0, 6.498975524720148e+41)]
    // ν≫x でJ/Iが極小、Y/Kが巨大。
    [InlineData(50.0, 1.0, 2.934635308511838e-80)]
    // x≈ν の転移域。
    [InlineData(50.0, 50.0, 17650802430.016712)]
    // 負の整数次数で零係数による反射。
    [InlineData(-1.0, 1.0, 0.565159103992485)]
    // 負の実数次数の反射。
    [InlineData(-0.25, 1.0, 1.3177528951478925)]
    // 負の次数の符号。
    [InlineData(-1.25, 0.25, -2.571505015247321)]
    // 整数次数だけは負のxも実数で、奇数次の符号を確認。
    [InlineData(1.0, -1.0, -0.565159103992485)]
    // 負のxで偶数次。
    [InlineData(2.0, -1.0, 0.13574766976703828)]
    // exp(x)自体は溢れるがIは有限。
    [InlineData(0.0, 710.0, 3.345334558619656e+306)]
    public void BesselI_MatchesReference(double nu, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.BesselI(nu, x), 2e-12, double.Epsilon);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/10.7; https://dlmf.nist.gov/10.17; https://dlmf.nist.gov/10.27; https://dlmf.nist.gov/10.30; https://dlmf.nist.gov/10.40
    [Theory]
    // 次数NaN。
    [InlineData(double.NaN, 1.0, double.NaN)]
    // 引数NaN。
    [InlineData(0.0, double.NaN, double.NaN)]
    // 無限大次数。
    [InlineData(double.PositiveInfinity, 1.0, double.NaN)]
    // 負の非整数引数は複素数。
    [InlineData(0.25, -1.0, double.NaN)]
    // 正の無限大引数。
    [InlineData(0.0, double.PositiveInfinity, double.PositiveInfinity)]
    // 0次の原点値。
    [InlineData(0.0, 0.0, 1.0)]
    // 正の次数の原点値。
    [InlineData(1.0, 0.0, 0.0)]
    // 負の整数次数の原点値。
    [InlineData(-1.0, 0.0, 0.0)]
    // 負の半整数の正の片側極。
    [InlineData(-0.5, 0.0, double.PositiveInfinity)]
    // 隣の負半整数の負の片側極。
    [InlineData(-1.5, 0.0, double.NegativeInfinity)]
    public void BesselI_SpecialValues(double nu, double x, double expected)
    {
        double actual = XMath.Special.BesselI(nu, x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // https://dlmf.nist.gov/10.7; https://dlmf.nist.gov/10.17; https://dlmf.nist.gov/10.27; https://dlmf.nist.gov/10.30; https://dlmf.nist.gov/10.40
    [Theory]
    // 整数0次でTemme補助量のμ=0。
    [InlineData(0.0, 1.0, 0.42102443824070834)]
    // 整数1次の正規化。
    [InlineData(1.0, 1.0, 0.6019072301972346)]
    // 半整数は初等関数に帰着。
    [InlineData(0.5, 1.0, 0.46106850444789454)]
    // 次数を最近接整数へ減らす切替直前。
    [InlineData(0.49999999999999994, 0.25, 1.9521640631515476)]
    // 次数切替直後。
    [InlineData(0.5000000000000001, 0.25, 1.9521640631515478)]
    // 原点近くの冪の振舞い。
    [InlineData(0.25, 1e-10, 681.7174746305784)]
    // Temme級数の末端直前。
    [InlineData(0.25, 1.9999999999999998, 0.11537827684085679)]
    // Steed法への切替。
    [InlineData(0.25, 2.0, 0.11537827684085676)]
    // 漸近展開の直前。
    [InlineData(0.25, 24.999999999999996, 3.468411261478815e-12)]
    // 漸近展開の開始。
    [InlineData(0.25, 25.0, 3.4684112614788024e-12)]
    // x=ν²の切替直前。
    [InlineData(10.0, 99.99999999999999, 7.65542797738821e-45)]
    // 次数依存の漸近展開の境界。
    [InlineData(10.0, 100.0, 7.6554279773881e-45)]
    // ν≫x でJ/Iが極小、Y/Kが巨大。
    [InlineData(50.0, 1.0, 3.406896854161702e+77)]
    // x≈ν の転移域。
    [InlineData(50.0, 50.0, 4.0060134766400893e-13)]
    // 負の整数次数で零係数による反射。
    [InlineData(-1.0, 1.0, 0.6019072301972346)]
    // 負の実数次数の反射。
    [InlineData(-0.25, 1.0, 0.4307397744485855)]
    // 負の次数の符号。
    [InlineData(-1.25, 0.25, 5.859189736638019)]
    // subnormalの減衰値。
    [InlineData(0.0, 730.0, 4.2791e-319)]
    // 次数0の単独経路の切替点。mpmath 1.3.0 の100桁と180桁の参照値が binary64 で一致。
    [InlineData(0.0, 1.4901161193847653e-08, 18.13775821021699)]
    [InlineData(0.0, 1.490116119384766e-08, 18.13775821021699)]
    [InlineData(0.0, 1.9999999999999998, 0.11389387274953347)]
    [InlineData(0.0, 2.0, 0.11389387274953344)]
    [InlineData(0.0, 24.999999999999996, 3.464161562213127e-12)]
    [InlineData(0.0, 25.0, 3.4641615622131143e-12)]
    public void BesselK_MatchesReference(double nu, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.BesselK(nu, x), 2e-12, double.Epsilon);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/10.7; https://dlmf.nist.gov/10.17; https://dlmf.nist.gov/10.27; https://dlmf.nist.gov/10.30; https://dlmf.nist.gov/10.40
    [Theory]
    // 次数NaN。
    [InlineData(double.NaN, 1.0, double.NaN)]
    // 引数NaN。
    [InlineData(0.0, double.NaN, double.NaN)]
    // 無限大次数。
    [InlineData(double.PositiveInfinity, 1.0, double.NaN)]
    // 負の非整数引数は複素数。
    [InlineData(0.25, -1.0, double.NaN)]
    // 正の無限大引数。
    [InlineData(0.0, double.PositiveInfinity, 0.0)]
    // 原点の正の極。
    [InlineData(0.0, 0.0, double.PositiveInfinity)]
    // Kは次数の偶関数。
    [InlineData(-1.0, 0.0, double.PositiveInfinity)]
    // 整数次数でも負のxは複素数。
    [InlineData(1.0, -1.0, double.NaN)]
    public void BesselK_SpecialValues(double nu, double x, double expected)
    {
        double actual = XMath.Special.BesselK(nu, x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // https://dlmf.nist.gov/10.7; https://dlmf.nist.gov/10.17; https://dlmf.nist.gov/10.27; https://dlmf.nist.gov/10.30; https://dlmf.nist.gov/10.40
    [Theory]
    // 整数0次でTemme補助量のμ=0。
    [InlineData(0.0, 1.0, 0.46575960759364043)]
    // 整数1次の正規化。
    [InlineData(1.0, 1.0, 0.20791041534970844)]
    // 半整数は初等関数に帰着。
    [InlineData(0.5, 1.0, 0.3449513138882446)]
    // 次数を最近接整数へ減らす切替直前。
    [InlineData(0.49999999999999994, 0.25, 0.3139431117645787)]
    // 次数切替直後。
    [InlineData(0.5000000000000001, 0.25, 0.31394311176457856)]
    // 原点近くの冪の振舞い。
    [InlineData(0.25, 1e-10, 0.0029337386155927556)]
    // Temme級数の末端直前。
    [InlineData(0.25, 1.9999999999999998, 0.2981915987879022)]
    // Steed法への切替。
    [InlineData(0.25, 2.0, 0.29819159878790213)]
    // 漸近展開の直前。
    [InlineData(0.25, 24.999999999999996, 0.08009449585806488)]
    // 漸近展開の開始。
    [InlineData(0.25, 25.0, 0.08009449585806487)]
    // x=ν²の切替直前。
    [InlineData(10.0, 99.99999999999999, 0.02417668271825883)]
    // 次数依存の漸近展開の境界。
    [InlineData(10.0, 100.0, 0.02417668271825883)]
    // ν≫x でJ/Iが極小、Y/Kが巨大。
    [InlineData(50.0, 1.0, 1.0795919973373183e-80)]
    // x≈ν の転移域。
    [InlineData(50.0, 50.0, 3.4043982503335886e-12)]
    // 負の整数次数で零係数による反射。
    [InlineData(-1.0, 1.0, 0.20791041534970844)]
    // 負の実数次数の反射。
    [InlineData(-0.25, 1.0, 0.48477419866905697)]
    // 負の次数の符号。
    [InlineData(-1.25, 0.25, -2.002690119546658)]
    // 大引数で位相還元を保持。
    [InlineData(0.5, 10000000000.0, 3.989422804014327e-06)]
    // 整数次数だけは負のxも実数で、奇数次の符号を確認。
    [InlineData(1.0, -1.0, -0.20791041534970844)]
    // 負のxで偶数次。
    [InlineData(2.0, -1.0, 0.04993877689422354)]
    // 非スケール版がオーバーフローまたはアンダーフローする領域。
    [InlineData(0.0, 1000.0, 0.012617240455891257)]
    // 大きい負の次数の反射で、e⁻ˣ を掛ける前の K の項がオーバーフローする領域（期待値は BesselIK_LargeOrder_MatchesIntegralReference と同じ積分表示）。
    [InlineData(-1000.5, 500.0, 8.93017160403059e-78)]
    public void BesselIScaled_MatchesReference(double nu, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.BesselIScaled(nu, x), 2e-12, double.Epsilon);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/10.7; https://dlmf.nist.gov/10.17; https://dlmf.nist.gov/10.27; https://dlmf.nist.gov/10.30; https://dlmf.nist.gov/10.40
    [Theory]
    // 次数NaN。
    [InlineData(double.NaN, 1.0, double.NaN)]
    // 引数NaN。
    [InlineData(0.0, double.NaN, double.NaN)]
    // 無限大次数。
    [InlineData(double.PositiveInfinity, 1.0, double.NaN)]
    // 負の非整数引数は複素数。
    [InlineData(0.25, -1.0, double.NaN)]
    // 正の無限大引数。
    [InlineData(0.0, double.PositiveInfinity, 0.0)]
    // 0次の原点値。
    [InlineData(0.0, 0.0, 1.0)]
    // 正の次数の原点値。
    [InlineData(1.0, 0.0, 0.0)]
    // 負の整数次数の原点値。
    [InlineData(-1.0, 0.0, 0.0)]
    // 負の半整数の正の片側極。
    [InlineData(-0.5, 0.0, double.PositiveInfinity)]
    // 隣の負半整数の負の片側極。
    [InlineData(-1.5, 0.0, double.NegativeInfinity)]
    public void BesselIScaled_SpecialValues(double nu, double x, double expected)
    {
        double actual = XMath.Special.BesselIScaled(nu, x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // https://dlmf.nist.gov/10.7; https://dlmf.nist.gov/10.17; https://dlmf.nist.gov/10.27; https://dlmf.nist.gov/10.30; https://dlmf.nist.gov/10.40
    [Theory]
    // 整数0次でTemme補助量のμ=0。
    [InlineData(0.0, 1.0, 1.144463079806895)]
    // 整数1次の正規化。
    [InlineData(1.0, 1.0, 1.6361534862632583)]
    // 半整数は初等関数に帰着。
    [InlineData(0.5, 1.0, 1.2533141373155003)]
    // 次数を最近接整数へ減らす切替直前。
    [InlineData(0.49999999999999994, 0.25, 2.5066282746310002)]
    // 次数切替直後。
    [InlineData(0.5000000000000001, 0.25, 2.5066282746310007)]
    // 原点近くの冪の振舞い。
    [InlineData(0.25, 1e-10, 681.7174746987502)]
    // Temme級数の末端直前。
    [InlineData(0.25, 1.9999999999999998, 0.8525365601750416)]
    // Steed法への切替。
    [InlineData(0.25, 2.0, 0.8525365601750416)]
    // 漸近展開の直前。
    [InlineData(0.25, 24.999999999999996, 0.24974260374343674)]
    // 漸近展開の開始。
    [InlineData(0.25, 25.0, 0.2497426037434367)]
    // x=ν²の切替直前。
    [InlineData(10.0, 99.99999999999999, 0.20578687173955781)]
    // 次数依存の漸近展開の境界。
    [InlineData(10.0, 100.0, 0.2057868717395578)]
    // ν≫x でJ/Iが極小、Y/Kが巨大。
    [InlineData(50.0, 1.0, 9.26090581010204e+77)]
    // x≈ν の転移域。
    [InlineData(50.0, 50.0, 2077000021.993019)]
    // 負の整数次数で零係数による反射。
    [InlineData(-1.0, 1.0, 1.6361534862632583)]
    // 負の実数次数の反射。
    [InlineData(-0.25, 1.0, 1.1708721016781378)]
    // 負の次数の符号。
    [InlineData(-1.25, 0.25, 7.52334854303917)]
    // 大引数で位相還元を保持。
    [InlineData(0.5, 10000000000.0, 1.2533141373155002e-05)]
    // 非スケール版が溢れる/underflowする領域。
    [InlineData(0.0, 1000.0, 0.03962832160075422)]
    // 次数0の単独経路の切替点。mpmath 1.3.0 の100桁と180桁の参照値が binary64 で一致。
    [InlineData(0.0, 1.4901161193847653e-08, 18.137758480490653)]
    [InlineData(0.0, 1.490116119384766e-08, 18.137758480490653)]
    [InlineData(0.0, 1.9999999999999998, 0.8415682150707715)]
    [InlineData(0.0, 2.0, 0.8415682150707714)]
    [InlineData(0.0, 24.999999999999996, 0.24943660457559672)]
    [InlineData(0.0, 25.0, 0.2494366045755967)]
    public void BesselKScaled_MatchesReference(double nu, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.BesselKScaled(nu, x), 2e-12, double.Epsilon);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/10.7; https://dlmf.nist.gov/10.17; https://dlmf.nist.gov/10.27; https://dlmf.nist.gov/10.30; https://dlmf.nist.gov/10.40
    [Theory]
    // 次数NaN。
    [InlineData(double.NaN, 1.0, double.NaN)]
    // 引数NaN。
    [InlineData(0.0, double.NaN, double.NaN)]
    // 無限大次数。
    [InlineData(double.PositiveInfinity, 1.0, double.NaN)]
    // 負の非整数引数は複素数。
    [InlineData(0.25, -1.0, double.NaN)]
    // 正の無限大引数。
    [InlineData(0.0, double.PositiveInfinity, 0.0)]
    // 原点の正の極。
    [InlineData(0.0, 0.0, double.PositiveInfinity)]
    // Kは次数の偶関数。
    [InlineData(-1.0, 0.0, double.PositiveInfinity)]
    // 整数次数でも負のxは複素数。
    [InlineData(1.0, -1.0, double.NaN)]
    public void BesselKScaled_SpecialValues(double nu, double x, double expected)
    {
        double actual = XMath.Special.BesselKScaled(nu, x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // https://dlmf.nist.gov/10.21
    [Theory]
    // 最初の零点の丸め位置と両隣。絶対誤差で評価。
    [InlineData(0.0, 2.4048255576957724, 1.6946014926954953e-16)]
    // 最初の零点の丸め位置と両隣。絶対誤差で評価。
    [InlineData(0.0, 2.404825557695773, -6.10876525973673e-17)]
    // 最初の零点の丸め位置と両隣。絶対誤差で評価。
    [InlineData(0.0, 2.4048255576957733, -2.916354544642841e-16)]
    public void BesselJ_NearZero_MatchesReference(double nu, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.BesselJ(nu, x), 5e-13, 2e-14);
    }

    // https://dlmf.nist.gov/10.21
    [Theory]
    // 最初の零点の丸め位置と両隣。絶対誤差で評価。
    [InlineData(0.0, 0.8935769662791674, -1.2102460161073392e-16)]
    // 最初の零点の丸め位置と両隣。絶対誤差で評価。
    [InlineData(0.0, 0.8935769662791675, -2.3389279284062102e-17)]
    // 最初の零点の丸め位置と両隣。絶対誤差で評価。
    [InlineData(0.0, 0.8935769662791676, 7.42460430426097e-17)]
    public void BesselY_NearZero_MatchesReference(double nu, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.BesselY(nu, x), 5e-13, 2e-14);
    }

    // https://dlmf.nist.gov/10.4
    [Theory]
    // 実数次数と複素数の実部・虚部を独立の参照値で確認。
    [InlineData(0.25, 1.0, 0.7522313333407901)]
    // 負次数の反射と虚部の符号。
    [InlineData(-1.25, 10.0, 0.06211047687532449)]
    public void HankelH1Real_MatchesReference(double nu, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.HankelH1(nu, x).Real, 5e-13, 0);
    }

    // https://dlmf.nist.gov/10.4
    [Theory]
    // 実数次数と複素数の実部・虚部を独立の参照値で確認。
    [InlineData(0.25, 1.0, -0.19442175367716438)]
    // 負次数の反射と虚部の符号。
    [InlineData(-1.25, 10.0, -0.2453975769674069)]
    public void HankelH1Imaginary_MatchesReference(double nu, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.HankelH1(nu, x).Imaginary, 5e-13, 0);
    }

    // https://dlmf.nist.gov/10.4
    [Theory]
    // 実数次数と複素数の実部・虚部を独立の参照値で確認。
    [InlineData(0.25, 1.0, 0.7522313333407901)]
    // 負次数の反射と虚部の符号。
    [InlineData(-1.25, 10.0, 0.06211047687532449)]
    public void HankelH2Real_MatchesReference(double nu, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.HankelH2(nu, x).Real, 5e-13, 0);
    }

    // https://dlmf.nist.gov/10.4
    [Theory]
    // 実数次数と複素数の実部・虚部を独立の参照値で確認。
    [InlineData(0.25, 1.0, 0.19442175367716438)]
    // 負次数の反射と虚部の符号。
    [InlineData(-1.25, 10.0, 0.2453975769674069)]
    public void HankelH2Imaginary_MatchesReference(double nu, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.HankelH2(nu, x).Imaginary, 5e-13, 0);
    }

    // ロンスキアンの交差積は独立に次数の漸化関係と正規化を確認。https://dlmf.nist.gov/10.5; https://dlmf.nist.gov/10.28
    [Theory]
    [InlineData(0.25, 1.0)]
    [InlineData(1.0, 2.0)]
    [InlineData(10.0, 100.0)]
    public void Bessel_CrossProducts(double nu, double x)
    {
        double jy = XMath.Special.BesselJ(nu + 1, x) * XMath.Special.BesselY(nu, x)
            - XMath.Special.BesselJ(nu, x) * XMath.Special.BesselY(nu + 1, x);
        SpecialFunctionAssert.Close(2 / (Math.PI * x), jy, 3e-12);
        double ik = XMath.Special.BesselIScaled(nu, x) * XMath.Special.BesselKScaled(nu + 1, x)
            + XMath.Special.BesselIScaled(nu + 1, x) * XMath.Special.BesselKScaled(nu, x);
        SpecialFunctionAssert.Close(1 / x, ik, 3e-12);
    }

    // Complex戻り値でもNaNの成分を失わない。https://dlmf.nist.gov/10.4
    [Fact]
    public void Hankel_NaNInputs()
    {
        var h1 = XMath.Special.HankelH1(double.NaN, 1);
        var h2 = XMath.Special.HankelH2(0, double.NaN);
        Assert.True(double.IsNaN(h1.Real) && double.IsNaN(h1.Imaginary));
        Assert.True(double.IsNaN(h2.Real) && double.IsNaN(h2.Imaginary));
    }

    private const double RelativeTolerance = 2e-12;

    // J0(x), I0(x), exp(-x)I0(x) は、このxではいずれも1に丸められる。
    [Trait("Category", "ReviewRegression")]
    [Theory]
    [InlineData("J")]
    [InlineData("I")]
    [InlineData("IScaled")]
    public void Bessel_OrderZero_TinyArgument_RoundsToOne(string function)
    {
        double value = function switch
        {
            "J" => XMath.Special.BesselJ(0, 1e-310),
            "I" => XMath.Special.BesselI(0, 1e-310),
            _ => XMath.Special.BesselIScaled(0, 1e-310),
        };
        Assert.Equal(1, value);
    }

    // 半整数の閉形式。大きな中間積を作らず期待値を求める。https://dlmf.nist.gov/10.39
    [Trait("Category", "ReviewRegression")]
    [Theory]
    [InlineData("J")]
    [InlineData("IScaled")]
    [InlineData("KScaled")]
    [InlineData("I")]
    public void Bessel_HalfOrder_LargestScale_PreservesCorrectRange(string function)
    {
        double x = 1e308;
        double rootReciprocal = Math.Sqrt(1 / x);
        (double expected, double actual) = function switch
        {
            "J" => (Math.Sqrt(2 / Math.PI) * rootReciprocal * Math.Sin(x), XMath.Special.BesselJ(0.5, x)),
            "IScaled" => (rootReciprocal / Math.Sqrt(2 * Math.PI), XMath.Special.BesselIScaled(0.5, x)),
            "KScaled" => (Math.Sqrt(Math.PI / 2) * rootReciprocal, XMath.Special.BesselKScaled(0.5, x)),
            _ => (double.PositiveInfinity, XMath.Special.BesselI(0.5, x)),
        };
        if (double.IsInfinity(expected))
        {
            Assert.Equal(expected, actual);
        }
        else
        {
            SpecialFunctionAssert.Close(expected, actual);
        }
    }

    // Kの両和の収束判定と、Iのロンスキアンの打ち消し。
    // 半整数だけに限定せず、次数を丸めてμへ移す両側も確認する。
    [Trait("Category", "AuditRegression")]
    [Theory]
    [InlineData(false, 0.49, 1e-9, 32635.13754958369)]
    [InlineData(false, 0.5, 1e-9, 39633.27293642684)]
    [InlineData(false, 0.51, 1e-9, 48155.819683988826)]
    [InlineData(true, 0.49, 1e-7, 0.00029859908746708406)]
    [InlineData(true, 0.5, 1e-7, 0.0002523132522020164)]
    [InlineData(true, 0.51, 1e-7, 0.0002131822533166071)]
    public void BesselIK_SmallNormalArgument_MatchesReference(bool firstKind, double nu, double x, double expected)
    {
        double actual = firstKind ? XMath.Special.BesselI(nu, x) : XMath.Special.BesselK(nu, x);
        SpecialFunctionAssert.Close(expected, actual, RelativeTolerance);
    }

    // A5: 入力xの条件数は約ν。μ<0側の内部の打ち消しを入力の条件数で許容しない。
    [Trait("Category", "AuditRegression")]
    [Theory]
    [InlineData(0.49, 0.0002985990874670831)]
    [InlineData(0.51, 0.0002131822533166064)]
    [InlineData(1.51, 7.05901501048366e-12)]
    public void BesselJ_SmallNormalArgument_MatchesReference(double nu, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.BesselJ(nu, 1e-7), RelativeTolerance);
    }

    // 探索で得た補強点。mpmath 1.4.1で独立に計算し、複数精度でbinary64丸めを確認。
    // 入力・期待値・許容誤差をここに固定する。出典と選定方針は README.md を参照。

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(40.33793204684984, 15.941338686872186, 1.693472050786213e-13)]
    [InlineData(83.56804371845921, 0.6445572391264699, 1.63979419317895e-167)]
    [InlineData(90.49038154204656, 2.806384014555157, 1.4836042887821482e-126)]
    [InlineData(-4.212676076884946, 4.5531187976322963e-7, 1.4977800476761987e+28)]
    [InlineData(11.739636134752391, 2.4706491222581886e-8, 5.814213885578462e-102)]
    [InlineData(59.195869400213866, 0.0010060592865777528, 1.8122838298493573e-276)]
    // 高次数の回帰点
    [InlineData(170.5, 3, 1.0987506058052299e-278)]
    [InlineData(1000.5, 10005, 0.007367167626236061)]
    // 振動域で誤差の大きかった点
    // 前回の探索で最大相対誤差となった入力。
    [InlineData(100.7, 1007, 0.003319830617310794)]
    public void BesselJ_ExploredRelativeValues(double nu, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.BesselJ(nu, x), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(40.33793204684984, 15.941338686872186, -50730530736.632385)]
    [InlineData(27.539005524023956, 8.451689650577462, -8653226917.627478)]
    [InlineData(83.56804371845921, 0.6445572391264699, -2.322915798701006e+164)]
    [InlineData(90.49038154204656, 2.806384014555157, -2.3721306681475818e+123)]
    [InlineData(-4.212676076884946, 4.5531187976322963e-7, -1.8977647897552118e+28)]
    [InlineData(59.195869400213866, 0.0010060592865777528, -2.9671022202929417e+273)]
    public void BesselY_ExploredRelativeValues(double nu, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.BesselY(nu, x), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(40.33793204684984, 15.941338686872186, 3.671547729827491e-12)]
    [InlineData(83.56804371845921, 0.6445572391264699, 1.6438270196905396e-167)]
    [InlineData(90.49038154204656, 2.806384014555157, 1.548855225821051e-126)]
    [InlineData(-4.212676076884946, 4.5531187976322963e-7, 1.4977800476761506e+28)]
    [InlineData(11.739636134752391, 2.4706491222581886e-8, 5.814213885578462e-102)]
    [InlineData(59.195869400213866, 0.0010060592865777528, 1.812283845085558e-276)]
    public void BesselI_ExploredRelativeValues(double nu, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.BesselI(nu, x), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(40.33793204684984, 15.941338686872186, 3139653501.978189)]
    [InlineData(83.56804371845921, 0.6445572391264699, 3.6396593182241277e+164)]
    [InlineData(90.49038154204656, 2.806384014555157, 3.565726101234772e+123)]
    [InlineData(-4.212676076884946, 4.5531187976322963e-7, 3.79757880599075e+28)]
    [InlineData(11.739636134752391, 2.4706491222581886e-8, 7.325282265134269e+99)]
    [InlineData(59.195869400213866, 0.0010060592865777528, 4.660713228331063e+273)]
    public void BesselK_ExploredRelativeValues(double nu, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.BesselK(nu, x), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(40.33793204684984, 15.941338686872186, 4.381408561181146e-19)]
    [InlineData(83.56804371845921, 0.6445572391264699, 8.628364085747847e-168)]
    [InlineData(90.49038154204656, 2.806384014555157, 9.358661381874577e-128)]
    [InlineData(-4.212676076884946, 4.5531187976322963e-7, 1.4977793657192567e+28)]
    [InlineData(11.739636134752391, 2.4706491222581886e-8, 5.81421374192964e-102)]
    [InlineData(59.195869400213866, 0.0010060592865777528, 1.8104614969421396e-276)]
    public void BesselIScaled_ExploredRelativeValues(double nu, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.BesselIScaled(nu, x), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(40.33793204684984, 15.941338686872186, 2.630977578709387e+16)]
    [InlineData(27.539005524023956, 8.451689650577462, 16560559921129.92)]
    [InlineData(83.56804371845921, 0.6445572391264699, 6.934072635678201e+164)]
    [InlineData(90.49038154204656, 2.806384014555157, 5.901264380010896e+124)]
    [InlineData(-4.212676076884946, 4.5531187976322963e-7, 3.797580535073888e+28)]
    [InlineData(59.195869400213866, 0.0010060592865777528, 4.665404541630462e+273)]
    public void BesselKScaled_ExploredRelativeValues(double nu, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.BesselKScaled(nu, x), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(45.948148420655464, 2.1006985916764234e-7, 0.0)]
    [InlineData(89.41383086989352, 0.005765958943163182, 0.0)]
    [InlineData(84.78718843992887, 2.379249919122132e-8, 0.0)]
    [InlineData(71.5785208771056, 0.0002511799290365942, 0.0)]
    [InlineData(83.97334824883536, 0.0040248785270589026, 0.0)]
    public void BesselJ_ExploredExactValues(double nu, double x, double expected)
    {
        Assert.Equal(expected, XMath.Special.BesselJ(nu, x));
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(45.948148420655464, 2.1006985916764234e-7, double.NegativeInfinity)]
    [InlineData(89.41383086989352, 0.005765958943163182, double.NegativeInfinity)]
    [InlineData(84.78718843992887, 2.379249919122132e-8, double.NegativeInfinity)]
    [InlineData(71.5785208771056, 0.0002511799290365942, double.NegativeInfinity)]
    [InlineData(83.97334824883536, 0.0040248785270589026, double.NegativeInfinity)]
    // オーバーフロー時の符号
    [InlineData(170.5, 1.5, double.NegativeInfinity)]
    public void BesselY_ExploredExactValues(double nu, double x, double expected)
    {
        Assert.Equal(expected, XMath.Special.BesselY(nu, x));
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(45.948148420655464, 2.1006985916764234e-7, 0.0)]
    [InlineData(89.41383086989352, 0.005765958943163182, 0.0)]
    [InlineData(84.78718843992887, 2.379249919122132e-8, 0.0)]
    [InlineData(71.5785208771056, 0.0002511799290365942, 0.0)]
    [InlineData(83.97334824883536, 0.0040248785270589026, 0.0)]
    public void BesselI_ExploredExactValues(double nu, double x, double expected)
    {
        Assert.Equal(expected, XMath.Special.BesselI(nu, x));
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(45.948148420655464, 2.1006985916764234e-7, double.PositiveInfinity)]
    [InlineData(89.41383086989352, 0.005765958943163182, double.PositiveInfinity)]
    [InlineData(84.78718843992887, 2.379249919122132e-8, double.PositiveInfinity)]
    [InlineData(71.5785208771056, 0.0002511799290365942, double.PositiveInfinity)]
    [InlineData(83.97334824883536, 0.0040248785270589026, double.PositiveInfinity)]
    public void BesselK_ExploredExactValues(double nu, double x, double expected)
    {
        Assert.Equal(expected, XMath.Special.BesselK(nu, x));
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(45.948148420655464, 2.1006985916764234e-7, 0.0)]
    [InlineData(89.41383086989352, 0.005765958943163182, 0.0)]
    [InlineData(84.78718843992887, 2.379249919122132e-8, 0.0)]
    [InlineData(71.5785208771056, 0.0002511799290365942, 0.0)]
    [InlineData(83.97334824883536, 0.0040248785270589026, 0.0)]
    public void BesselIScaled_ExploredExactValues(double nu, double x, double expected)
    {
        Assert.Equal(expected, XMath.Special.BesselIScaled(nu, x));
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(45.948148420655464, 2.1006985916764234e-7, double.PositiveInfinity)]
    [InlineData(89.41383086989352, 0.005765958943163182, double.PositiveInfinity)]
    [InlineData(84.78718843992887, 2.379249919122132e-8, double.PositiveInfinity)]
    [InlineData(71.5785208771056, 0.0002511799290365942, double.PositiveInfinity)]
    [InlineData(83.97334824883536, 0.0040248785270589026, double.PositiveInfinity)]
    public void BesselKScaled_ExploredExactValues(double nu, double x, double expected)
    {
        Assert.Equal(expected, XMath.Special.BesselKScaled(nu, x));
    }

    // 大きい次数で e⁻ˣI・eˣK や漸化式・Debye 展開の中間値が倍精度の範囲を外れても、I・K 自体は範囲内となる点の回帰テスト。
    // 期待値は積分表示 DLMF 10.32.2・10.32.9 を mpmath 1.4.1（30桁・45桁）で数値積分し（ν=10⁵ の I は超幾何級数）、binary64 丸め一致を確認。
    // mpmath の besselk は大きい整数次数で精度が落ちる（K₁₀₀₀(800) で相対 3.7e-14）ため使わない。
    [Trait("Category", "Accuracy")]
    [Theory]
    // Steed 法（ν < 10⁴）。e⁻ˣI がアンダーフローし、eˣK がオーバーフローする。
    [InlineData(true, 1000.0, 500.0, 4.674525236594426e-144)]
    [InlineData(false, 1000.0, 500.0, 9.567037207675176e+139)]
    // eˣ・e⁻ˣ 自体が倍精度の範囲を外れる。
    [InlineData(true, 1000.0, 800.0, 1.7850006513554681e+99)]
    [InlineData(false, 1000.0, 800.0, 2.187306658024166e-103)]
    // I は正規化数の下限、K は上限の近く。
    [InlineData(true, 500.5, 92.05551630115369, 5.397894230214888e-302)]
    [InlineData(false, 500.5, 92.05551630115369, 1.8201917126843798e+298)]
    // e⁻ˣI が非正規化数になって桁落ちしやすい点。
    [InlineData(true, 1000.0, 588.2732774933864, 1.0529094620444241e-63)]
    [InlineData(true, 300.0, 20.70758284100671, 1.5798039144119e-310)]
    // 結果は 1 前後でも、漸化式の中間値がオーバーフローしやすい点。
    [InlineData(true, 3000.0, 1988.1, 0.005253087252489232)]
    [InlineData(false, 3000.0, 1988.1, 0.026447089730740336)]
    // Iは上限、Kは下限の近く。
    [InlineData(true, 3000.0, 2400.0, 2.6421543097409523e+301)]
    [InlineData(false, 3000.0, 2400.0, 4.925707791282933e-306)]
    // Debye展開（ν ≥ 10⁴）。
    [InlineData(true, 10000.0, 6627.0, 0.0016597149549938172)]
    [InlineData(false, 10000.0, 6627.0, 0.025111939330274554)]
    // 指数 νη の約1.2×10⁵同士の打ち消しを倍精度2つ分で計算する。
    [InlineData(true, 100000.0, 66270.0, 4.4453366719669904e-07)]
    [InlineData(false, 100000.0, 66270.0, 9.375816582182658)]
    // 負の次数の反射 I₋ᵥ = Iᵥ + (2/π) sin(νπ) Kᵥ で、Kの項が支配的。
    [InlineData(true, -1000.5, 500.0, 1.2534319367507726e+140)]
    public void BesselIK_LargeOrder_MatchesIntegralReference(bool firstKind, double nu, double x, double expected)
    {
        double actual = firstKind ? XMath.Special.BesselI(nu, x) : XMath.Special.BesselK(nu, x);
        SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 2e-12, absoluteTolerance: 2 * double.Epsilon);
    }

    // 同じ領域で、真値が倍精度の範囲外なら 0 または ∞ に丸める。
    // 範囲外であることは上の積分表示、ν=10⁴, x=8000 は log I（超幾何級数）と K < 1/(x Iᵥ₊₁) で確認した。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(true, 3000.0, 1500.0, 0.0)]
    [InlineData(false, 3000.0, 1500.0, double.PositiveInfinity)]
    [InlineData(false, 3000.0, 900.0, double.PositiveInfinity)]
    // Debye展開（ν ≥ 10⁴）。
    [InlineData(true, 10000.0, 2000.0, 0.0)]
    [InlineData(false, 10000.0, 1000.0, double.PositiveInfinity)]
    [InlineData(true, 10000.0, 8000.0, double.PositiveInfinity)]
    [InlineData(false, 10000.0, 8000.0, 0.0)]
    public void BesselIK_LargeOrder_RoundsOutOfRangeValues(bool firstKind, double nu, double x, double expected)
    {
        double actual = firstKind ? XMath.Special.BesselI(nu, x) : XMath.Special.BesselK(nu, x);
        Assert.Equal(expected, actual);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(40.33793204684984, 15.941338686872186, 1.693472050786213e-13)]
    [InlineData(27.539005524023956, 8.451689650577462, 1.403594800664746e-12)]
    [InlineData(36.733920846012076, 0.00032062456481369325, 7.4683260701291796e-183)]
    [InlineData(41.984401463370986, 0.00004071140789290403, 8.235995481466556e-249)]
    [InlineData(18.667031972440487, 0.0001977063227297061, 3.8157584454018046e-92)]
    [InlineData(83.56804371845921, 0.6445572391264699, 1.63979419317895e-167)]
    public void HankelH1_Real_ExploredRelativeValues(double nu, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.HankelH1(nu, x).Real, relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(40.33793204684984, 15.941338686872186, -50730530736.632385)]
    [InlineData(27.539005524023956, 8.451689650577462, -8653226917.627478)]
    [InlineData(36.733920846012076, 0.00032062456481369325, -1.1602713546903394e+180)]
    [InlineData(41.984401463370986, 0.00004071140789290403, -9.20547200159163e+245)]
    [InlineData(18.667031972440487, 0.0001977063227297061, -4.468831523210653e+89)]
    [InlineData(83.56804371845921, 0.6445572391264699, -2.322915798701006e+164)]
    public void HankelH1_Imaginary_ExploredRelativeValues(double nu, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.HankelH1(nu, x).Imaginary, relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(45.948148420655464, 2.1006985916764234e-7, 0.0)]
    [InlineData(89.41383086989352, 0.005765958943163182, 0.0)]
    public void HankelH1_Real_ExploredExactValues(double nu, double x, double expected)
    {
        Assert.Equal(expected, XMath.Special.HankelH1(nu, x).Real);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(45.948148420655464, 2.1006985916764234e-7, double.NegativeInfinity)]
    [InlineData(89.41383086989352, 0.005765958943163182, double.NegativeInfinity)]
    public void HankelH1_Imaginary_ExploredExactValues(double nu, double x, double expected)
    {
        Assert.Equal(expected, XMath.Special.HankelH1(nu, x).Imaginary);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(40.33793204684984, 15.941338686872186, 1.693472050786213e-13)]
    [InlineData(27.539005524023956, 8.451689650577462, 1.403594800664746e-12)]
    [InlineData(36.733920846012076, 0.00032062456481369325, 7.4683260701291796e-183)]
    [InlineData(41.984401463370986, 0.00004071140789290403, 8.235995481466556e-249)]
    [InlineData(18.667031972440487, 0.0001977063227297061, 3.8157584454018046e-92)]
    [InlineData(83.56804371845921, 0.6445572391264699, 1.63979419317895e-167)]
    public void HankelH2_Real_ExploredRelativeValues(double nu, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.HankelH2(nu, x).Real, relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(40.33793204684984, 15.941338686872186, 50730530736.632385)]
    [InlineData(27.539005524023956, 8.451689650577462, 8653226917.627478)]
    [InlineData(36.733920846012076, 0.00032062456481369325, 1.1602713546903394e+180)]
    [InlineData(41.984401463370986, 0.00004071140789290403, 9.20547200159163e+245)]
    [InlineData(18.667031972440487, 0.0001977063227297061, 4.468831523210653e+89)]
    [InlineData(83.56804371845921, 0.6445572391264699, 2.322915798701006e+164)]
    public void HankelH2_Imaginary_ExploredRelativeValues(double nu, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.HankelH2(nu, x).Imaginary, relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(45.948148420655464, 2.1006985916764234e-7, 0.0)]
    [InlineData(89.41383086989352, 0.005765958943163182, 0.0)]
    public void HankelH2_Real_ExploredExactValues(double nu, double x, double expected)
    {
        Assert.Equal(expected, XMath.Special.HankelH2(nu, x).Real);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(45.948148420655464, 2.1006985916764234e-7, double.PositiveInfinity)]
    [InlineData(89.41383086989352, 0.005765958943163182, double.PositiveInfinity)]
    public void HankelH2_Imaginary_ExploredExactValues(double nu, double x, double expected)
    {
        Assert.Equal(expected, XMath.Special.HankelH2(nu, x).Imaginary);
    }
}
