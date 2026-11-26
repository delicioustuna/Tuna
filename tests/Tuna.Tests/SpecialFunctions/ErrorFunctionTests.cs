using Tuna;

namespace Tuna.Tests;

// 有限参照値は mpmath 1.4.1、100 桁以上で生成し binary64 に丸めた。
// 入力も binary64 の値を正確に高精度へ変換（十進文字列の値とは区別）。
// 参照値生成: https://mpmath.org/doc/current/functions/index.html
[Trait("SpecialFunction", "ErrorFunction")]
public class ErrorFunctionTests
{

    // https://dlmf.nist.gov/7.2; https://dlmf.nist.gov/7.12; https://dlmf.nist.gov/7.17
    [Theory]
    // 原点の既知値。
    [InlineData(0.0, 0.0)]
    // 原点付近の線形項。
    [InlineData(1e-20, 1.1283791670955125e-20)]
    // 奇偶性と負の引数。
    [InlineData(-0.5, -0.5204998778130465)]
    // erf級数の切替直前。
    [InlineData(0.49999999999999994, 0.5204998778130465)]
    // 正項級数の開始。
    [InlineData(0.5, 0.5204998778130465)]
    // erfc連分数の切替直前。
    [InlineData(1.2499999999999998, 0.9229001282564582)]
    // 連分数の開始点。
    [InlineData(1.25, 0.9229001282564583)]
    // 連分数の切替直後。
    [InlineData(1.2500000000000002, 0.9229001282564583)]
    // erfが1に近づく領域。
    [InlineData(5.0, 0.9999999999984626)]
    public void Erf_MatchesReference(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.Erf(x), 5e-13, 0);
    }

    // https://dlmf.nist.gov/7.2; https://dlmf.nist.gov/7.12; https://dlmf.nist.gov/7.17
    [Theory]
    // 原点の既知値。
    [InlineData(0.0, 1.0)]
    // 原点付近の線形項。
    [InlineData(1e-20, 1.0)]
    // 奇偶性と負の引数。
    [InlineData(-0.5, 1.5204998778130465)]
    // erf級数の切替直前。
    [InlineData(0.49999999999999994, 0.47950012218695354)]
    // 正項級数の開始。
    [InlineData(0.5, 0.4795001221869535)]
    // erfc連分数の切替直前。
    [InlineData(1.2499999999999998, 0.07709987174354183)]
    // 連分数の開始点。
    [InlineData(1.25, 0.07709987174354177)]
    // 連分数の切替直後。
    [InlineData(1.2500000000000002, 0.07709987174354171)]
    // erfが1に近づく領域。
    [InlineData(5.0, 1.537459794428035e-12)]
    // erfが1に丸められてもerfcは非零。
    [InlineData(10.0, 2.088487583762545e-45)]
    // 通常数の小さい尾部。
    [InlineData(26.0, 5.663192408856143e-296)]
    // subnormal の尾部を零へ早く丸めない。
    [InlineData(27.0, 5.23705e-319)]
    public void Erfc_MatchesReference(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.Erfc(x), 5e-13, double.Epsilon);
    }

    // https://dlmf.nist.gov/7.2; https://dlmf.nist.gov/7.12; https://dlmf.nist.gov/7.17
    [Theory]
    // 原点の既知値。
    [InlineData(0.0, 1.0)]
    // 原点付近の線形項。
    [InlineData(1e-20, 1.0)]
    // 奇偶性と負の引数。
    [InlineData(-0.5, 1.952360489182557)]
    // erf級数の切替直前。
    [InlineData(0.49999999999999994, 0.6156903441929259)]
    // 正項級数の開始。
    [InlineData(0.5, 0.6156903441929259)]
    // erfc連分数の切替直前。
    [InlineData(1.2499999999999998, 0.36782291645236115)]
    // 連分数の開始点。
    [InlineData(1.25, 0.3678229164523611)]
    // 連分数の切替直後。
    [InlineData(1.2500000000000002, 0.36782291645236104)]
    // erfが1に近づく領域。
    [InlineData(5.0, 0.11070463773306863)]
    // 負側で指数増大しても有限な値。
    [InlineData(-26.6, 3.894337719605585e+307)]
    // erfcはunderflowしてもスケーリング版は有限。
    [InlineData(100.0, 0.005641613782989433)]
    // 漸近近似の境界。
    [InlineData(100000000.0, 5.641895835477562e-09)]
    // 漸近近似へ切替直後。
    [InlineData(100000000.00000001, 5.6418958354775614e-09)]
    // 大引数の1/(√πx)挙動。
    [InlineData(10000000000.0, 5.641895835477563e-11)]
    public void Erfcx_MatchesReference(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.Erfcx(x), 2e-12, 0);
    }

    // https://dlmf.nist.gov/7.2; https://dlmf.nist.gov/7.12; https://dlmf.nist.gov/7.17
    [Theory]
    // 原点。
    [InlineData(0.0, 0.0)]
    // 奇関数の負側。
    [InlineData(-1.0, -1.6504257587975428)]
    // 級数の切替直前。
    [InlineData(2.9999999999999996, 1629.9946226015616)]
    // Dawson表現へ切替。
    [InlineData(3.0, 1629.9946226015657)]
    // 指数増大する領域。
    [InlineData(10.0, 1.5243074227086696e+42)]
    // exp(x²)単体は溢れるがerfiの最終結果は有限。
    [InlineData(26.65, 5.912168581618709e+306)]
    public void Erfi_MatchesReference(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.Erfi(x), 2e-12, 0);
    }

    // https://dlmf.nist.gov/7.2; https://dlmf.nist.gov/7.12; https://dlmf.nist.gov/7.17
    [Theory]
    // 原点。
    [InlineData(0.0, 0.0)]
    // 原点でF(x)≈x。
    [InlineData(1e-20, 1e-20)]
    // 級数の切替直前。
    [InlineData(0.19999999999999998, 0.19475103336802804)]
    // Rybicki法へ切替。
    [InlineData(0.2, 0.19475103336802807)]
    // 中央域の既知値。
    [InlineData(1.0, 0.5380795069127684)]
    // 奇関数の符号。
    [InlineData(-1.0, -0.5380795069127684)]
    // 漸近展開の切替直前。
    [InlineData(9.999999999999998, 0.05025384718759854)]
    // 漸近展開の開始。
    [InlineData(10.0, 0.05025384718759853)]
    // 漸近項1/(2x)で2xだけが溢れる条件。次項は丸めに影響しない。
    [InlineData(1e+308, 5e-309)]
    public void Dawson_MatchesReference(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.Dawson(x), 2e-12, double.Epsilon);
    }

    // https://dlmf.nist.gov/7.2; https://dlmf.nist.gov/7.12; https://dlmf.nist.gov/7.17
    [Theory]
    // 原点。
    [InlineData(0.0, 0.0)]
    // 小入力の相対精度。
    [InlineData(1e-20, 8.86226925452758e-21)]
    // 奇関数。
    [InlineData(-0.25, -0.2253120550121781)]
    // Newton直接反復側。
    [InlineData(0.49999999999999994, 0.4769362762044698)]
    // InverseErfcへ切替。
    [InlineData(0.5, 0.4769362762044699)]
    // Giles初期値のw=5切替直前。
    [InlineData(0.9966253323094463, 2.0727940839413184)]
    // Giles初期値の切替直後。
    [InlineData(0.9966253323094465, 2.0727940839413326)]
    // 正端点のすぐ内側。
    [InlineData(0.9999999999999999, 5.8635847487551676)]
    // 負端点のすぐ内側。
    [InlineData(-0.9999999999999999, -5.8635847487551676)]
    public void InverseErf_MatchesReference(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.InverseErf(x), 5e-13, 0);
    }

    // https://dlmf.nist.gov/7.2; https://dlmf.nist.gov/7.12; https://dlmf.nist.gov/7.17
    [Theory]
    // 通常の正分位点。
    [InlineData(0.5, 0.4769362762044699)]
    // 負分位点への対称変換。
    [InlineData(1.5, -0.4769362762044699)]
    // 零点q=1のすぐ左で相対精度。
    [InlineData(0.9999999999999999, 9.839095376804142e-17)]
    // 零点q=1のすぐ右。
    [InlineData(1.0000000000000002, -1.9678190753608283e-16)]
    // 旧実装の裾条件の境界。初期値の適用範囲外への外挿を再発させない。
    [InlineData(1e-30, 8.148616223169865)]
    // 旧裾条件の直内側も回帰確認。
    [InlineData(9.999999999999999e-31, 8.148616223169865)]
    // 1-qが1へ丸められる尾部。
    [InlineData(1e-300, 26.209469960516124)]
    // 最小subnormalの確率でも有限な逆関数。
    [InlineData(5e-324, 27.21329321081295)]
    // 負の無限端点のすぐ内側。
    [InlineData(1.9999999999999998, -5.805018683193453)]
    public void InverseErfc_MatchesReference(double q, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.InverseErfc(q), 2e-12, 0);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/7.2; https://dlmf.nist.gov/7.12; https://dlmf.nist.gov/7.17
    [Theory]
    // 正の無限大の極限。
    [InlineData(double.PositiveInfinity, 1.0)]
    // 負の無限大の極限。
    [InlineData(double.NegativeInfinity, -1.0)]
    // NaN。
    [InlineData(double.NaN, double.NaN)]
    public void Erf_SpecialValues(double x, double expected)
    {
        double actual = XMath.Special.Erf(x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/7.2; https://dlmf.nist.gov/7.12; https://dlmf.nist.gov/7.17
    [Theory]
    // 正の無限大の極限。
    [InlineData(double.PositiveInfinity, 0.0)]
    // 負の無限大の極限。
    [InlineData(double.NegativeInfinity, 2.0)]
    // NaN。
    [InlineData(double.NaN, double.NaN)]
    public void Erfc_SpecialValues(double x, double expected)
    {
        double actual = XMath.Special.Erfc(x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/7.2; https://dlmf.nist.gov/7.12; https://dlmf.nist.gov/7.17
    [Theory]
    // 正の無限大の極限。
    [InlineData(double.PositiveInfinity, 0.0)]
    // 負の無限大の極限。
    [InlineData(double.NegativeInfinity, double.PositiveInfinity)]
    // NaN。
    [InlineData(double.NaN, double.NaN)]
    public void Erfcx_SpecialValues(double x, double expected)
    {
        double actual = XMath.Special.Erfcx(x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/7.2; https://dlmf.nist.gov/7.12; https://dlmf.nist.gov/7.17
    [Theory]
    // 正の無限大の極限。
    [InlineData(double.PositiveInfinity, double.PositiveInfinity)]
    // 負の無限大の極限。
    [InlineData(double.NegativeInfinity, double.NegativeInfinity)]
    // NaN。
    [InlineData(double.NaN, double.NaN)]
    public void Erfi_SpecialValues(double x, double expected)
    {
        double actual = XMath.Special.Erfi(x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/7.2; https://dlmf.nist.gov/7.12; https://dlmf.nist.gov/7.17
    [Theory]
    // 正の無限大の極限。
    [InlineData(double.PositiveInfinity, 0.0)]
    // 負の無限大の極限。
    [InlineData(double.NegativeInfinity, -0.0)]
    // NaN。
    [InlineData(double.NaN, double.NaN)]
    public void Dawson_SpecialValues(double x, double expected)
    {
        double actual = XMath.Special.Dawson(x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/7.2; https://dlmf.nist.gov/7.12; https://dlmf.nist.gov/7.17
    [Theory]
    // 正端点。
    [InlineData(1.0, double.PositiveInfinity)]
    // 負端点。
    [InlineData(-1.0, double.NegativeInfinity)]
    // 正端点の外。
    [InlineData(1.0000000000000002, double.NaN)]
    // 負端点の外。
    [InlineData(-1.0000000000000002, double.NaN)]
    // NaN。
    [InlineData(double.NaN, double.NaN)]
    public void InverseErf_SpecialValues(double x, double expected)
    {
        double actual = XMath.Special.InverseErf(x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/7.2; https://dlmf.nist.gov/7.12; https://dlmf.nist.gov/7.17
    [Theory]
    // q=0端点。
    [InlineData(0.0, double.PositiveInfinity)]
    // q=2端点。
    [InlineData(2.0, double.NegativeInfinity)]
    // 中央値。
    [InlineData(1.0, 0.0)]
    // q<0。
    [InlineData(-1e-10, double.NaN)]
    // q>2。
    [InlineData(2.0000000000000004, double.NaN)]
    // NaN。
    [InlineData(double.NaN, double.NaN)]
    public void InverseErfc_SpecialValues(double q, double expected)
    {
        double actual = XMath.Special.InverseErfc(q);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // erf と逆関数の原点の向きを符号ビットで確認。https://dlmf.nist.gov/7.2
    [Fact]
    public void Erf_SignedZero()
    {
        Assert.Equal(0L, BitConverter.DoubleToInt64Bits(XMath.Special.Erf(0.0)));
        Assert.Equal(0L, BitConverter.DoubleToInt64Bits(XMath.Special.InverseErf(0.0)));
        Assert.Equal(long.MinValue, BitConverter.DoubleToInt64Bits(XMath.Special.Erf(-0.0)));
        Assert.Equal(long.MinValue, BitConverter.DoubleToInt64Bits(XMath.Special.InverseErf(-0.0)));
    }

    // 中央域q=0.5（対称側1.5）とw=16の理論境界の両隣。
    // wの丸めを考慮し、裾の両方式に入る1e-7/1e-8も収録する。
    // 参照値はerfinv(1-q)を160桁・220桁で計算。https://dlmf.nist.gov/7.17
    [Theory]
    [InlineData(0.49999999999999994, 0.47693627620446993)]
    [InlineData(0.5, 0.4769362762044699)]
    [InlineData(0.5000000000000001, 0.47693627620446977)]
    [InlineData(1.4999999999999998, -0.4769362762044696)]
    [InlineData(1.5, -0.4769362762044699)]
    [InlineData(1.5000000000000002, -0.4769362762044701)]
    [InlineData(5.6267588942650333e-8, 3.839783581356177)]
    [InlineData(5.626758894265034e-8, 3.839783581356177)]
    [InlineData(5.6267588942650347e-8, 3.839783581356177)]
    [InlineData(1e-7, 3.766562581570838)]
    [InlineData(1e-8, 4.05223724387139)]
    public void InverseErfc_NewBranchBoundaries_MatchReference(double q, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.InverseErfc(q), 2e-12, 0);
    }

    // 中間exp(x²)が溢れる入力の両隣と、最終結果がまだ有限の両符号。
    // 参照値はerfiを160桁・220桁で計算。https://dlmf.nist.gov/7.2
    [Theory]
    [InlineData(26.649999999999995, 5.91216858161759e+306)]
    [InlineData(26.65, 5.912168581618709e+306)]
    [InlineData(26.650000000000002, 5.912168581619828e+306)]
    [InlineData(-26.65, -5.912168581618709e+306)]
    [InlineData(26.7, 8.499867261268985e+307)]
    [InlineData(-26.7, -8.499867261268985e+307)]
    [InlineData(26.714033109640933, 1.797693134861981e308)]
    [InlineData(-26.714033109640933, -1.797693134861981e308)]
    public void Erfi_NearOverflow_RemainsFinite(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.Erfi(x), 2e-12, 0);
    }

    [Theory]
    [InlineData(26.8, double.PositiveInfinity)]
    [InlineData(-26.8, double.NegativeInfinity)]
    [InlineData(27.000000000000004, double.PositiveInfinity)]
    [InlineData(-27.000000000000004, double.NegativeInfinity)]
    [InlineData(26.714033109640937, double.PositiveInfinity)]
    [InlineData(26.71403310964094, double.PositiveInfinity)]
    [InlineData(-26.714033109640937, double.NegativeInfinity)]
    [InlineData(-26.71403310964094, double.NegativeInfinity)]
    public void Erfi_FinalOverflowHasCorrectSign(double x, double expected)
    {
        Assert.Equal(expected, XMath.Special.Erfi(x));
    }

    // 2*xのoverflow境界の両隣と最大有限値。次の漸近項は1ulp未満。
    [Fact]
    public void Dawson_LargestFiniteInputsRemainNonzero()
    {
        double boundary = double.MaxValue / 2;
        foreach (double x in new[] { Math.BitDecrement(boundary), boundary, Math.BitIncrement(boundary), double.MaxValue })
        {
            double expected = 0.5 / x;
            double positive = XMath.Special.Dawson(x);
            double negative = XMath.Special.Dawson(-x);
            Assert.True(positive > 0 && negative < 0);
            SpecialFunctionAssert.Close(expected, positive, 2e-12, double.Epsilon);
            SpecialFunctionAssert.Close(-expected, negative, 2e-12, double.Epsilon);
        }

        Assert.Equal(0L, BitConverter.DoubleToInt64Bits(XMath.Special.Dawson(double.PositiveInfinity)));
        Assert.Equal(long.MinValue, BitConverter.DoubleToInt64Bits(XMath.Special.Dawson(double.NegativeInfinity)));
    }

    // 外側の最隣接値もNaNとする。端点への丸め込みを検出。
    [Fact]
    public void InverseErfc_ImmediatelyOutsideDomainIsNaN()
    {
        Assert.True(double.IsNaN(XMath.Special.InverseErfc(-double.Epsilon)));
        Assert.True(double.IsNaN(XMath.Special.InverseErfc(Math.BitIncrement(2))));
    }

    // 探索で得た補強点。mpmath 1.4.1で独立に計算し、複数精度でbinary64丸めを確認。
    // 入力・期待値・許容誤差をここに固定する。出典と選定方針は README.md を参照。

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(-15.753572722854534, -1.0)]
    [InlineData(-0.37454350547972837, -0.4036693071617985)]
    [InlineData(2.071486750312669, 0.996605191212457)]
    [InlineData(25.32559363502599, 1.0)]
    [InlineData(-24.419787139069545, -1.0)]
    [InlineData(-6.109768973550345, -1.0)]
    public void Erf_ExploredRelativeValues(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.Erf(x), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(-15.753572722854534, 2.0)]
    [InlineData(11.354661642076088, 5.0317097856115475e-58)]
    [InlineData(2.071486750312669, 0.0033948087875429497)]
    [InlineData(25.32559363502599, 6.269895468080859e-281)]
    // 前回の探索で最大相対誤差となった入力。
    [InlineData(2.0021049998795775, 0.004634413682087666)]
    [InlineData(-24.419787139069545, 2.0)]
    [InlineData(-6.109768973550345, 2.0)]
    // 非正規数になる出力
    [InlineData(26.5, 2.2109076642637343e-307)]
    public void Erfc_ExploredRelativeValues(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.Erfc(x), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(-15.753572722854534, 1.2080538441237682e+108)]
    [InlineData(11.354661642076088, 0.04949742932168799)]
    [InlineData(-7.533606737201406, 8.902529229977647e+24)]
    [InlineData(2.071486750312669, 0.24796926185776544)]
    [InlineData(25.32559363502599, 0.022260121340151404)]
    [InlineData(-24.419787139069545, 1.9147977674062233e+259)]
    public void Erfcx_ExploredRelativeValues(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.Erfcx(x), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(-15.753572722854534, -2.167612890862872e+106)]
    [InlineData(11.354661642076088, 4.907031259149604e+54)]
    [InlineData(-0.37454350547972837, -0.44324978541829785)]
    [InlineData(2.071486750312669, 23.67247881010309)]
    [InlineData(25.32559363502599, 7.915381992069133e+276)]
    [InlineData(-24.419787139069545, -2.2138134146712771e+257)]
    public void Erfi_ExploredRelativeValues(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.Erfi(x), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(-15.753572722854534, -0.03180316700592953)]
    [InlineData(11.354661642076088, 0.04420757632208743)]
    [InlineData(2.071486750312669, 0.28721435725705396)]
    [InlineData(25.32559363502599, 0.019758300769845503)]
    [InlineData(-0.5755250151012206, -0.4637929988960788)]
    [InlineData(-24.419787139069545, -0.0204924101062415)]
    public void Dawson_ExploredRelativeValues(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.Dawson(x), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(0.8170428756369879, 0.9416576817251934)]
    [InlineData(-0.09427406005807804, -0.0837435633157814)]
    [InlineData(0.16586947509363248, 0.14807311315895025)]
    [InlineData(-0.871720733726745, -1.0754546714974458)]
    [InlineData(0.8496267949982936, 1.016971219009872)]
    [InlineData(0.2509584454267486, 0.2262058688834534)]
    public void InverseErf_ExploredRelativeValues(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.InverseErf(x), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(6.454004699985505e-125, 16.80959129381953)]
    [InlineData(4.7782299787393255e-57, 11.255494224404888)]
    [InlineData(1.0751828235362803e-102, 15.214939948986276)]
    [InlineData(2.447418699723476e-77, 13.162500369886668)]
    [InlineData(4.886326906188534e-7, 3.5572586222769536)]
    [InlineData(1.3419190245304232e-259, 24.337424181939312)]
    public void InverseErfc_ExploredRelativeValues(double q, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.InverseErfc(q), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 非正規数になる出力
    [InlineData(27, 5.23705e-319)]
    [InlineData(27.2, 1e-323)]
    public void Erfc_ExploredSubnormalValues(double x, double expected)
    {
        double actual = XMath.Special.Erfc(x);
        Assert.True(actual == 0 || Math.Sign(actual) == Math.Sign(expected));
        SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 0, absoluteTolerance: 2 * double.Epsilon);
    }

    // 区分多項式近似の区間境界および誤差が大きくなりやすい点を ULP 単位で検証する。
    // 参照値は mpmath 1.4.1（45 桁・70 桁で binary64 丸め一致）。許容誤差は e^(-x²) を使う関数が 4 ULP（Math.Exp の実装差を考慮）、その他が 3 ULP。
    [Trait("Category", "AuditRegression")]
    [Theory]
    [InlineData(1.178243929952832, 0.9043440429275471)]
    [InlineData(-0.9993814961370013, -0.8424438885281319)]
    [InlineData(0.4699999999999999, 0.49374505088608206)]
    [InlineData(0.47, 0.4937450508860821)]
    [InlineData(0.9999999999999999, 0.8427007929497148)]
    [InlineData(1.0, 0.8427007929497149)]
    [InlineData(2.5, 0.999593047982555)]
    [InlineData(5.5, 0.9999999999999927)]
    public void Erf_PiecewiseApproximation_WithinUlps(double x, double expected)
    {
        SpecialFunctionAssert.WithinUlps(expected, XMath.Special.Erf(x), 3);
    }

    [Trait("Category", "AuditRegression")]
    [Theory]
    [InlineData(1.1288050992199286, 0.11040587959722334)]
    [InlineData(0.4699999999999999, 0.5062549491139179)]
    [InlineData(0.47, 0.5062549491139179)]
    [InlineData(0.25, 0.7236736098317631)]
    [InlineData(0.5, 0.4795001221869535)]
    [InlineData(1.0, 0.15729920705028513)]
    [InlineData(1.5495057209760255, 0.02842777558806937)]
    [InlineData(3.9999999999999996, 1.5417257900280076e-08)]
    [InlineData(4.0, 1.541725790028002e-08)]
    [InlineData(4.176655708752589, 3.490641915725997e-09)]
    [InlineData(20.031729419055885, 1.5126577954257267e-176)]
    [InlineData(26.5, 2.2109076642637343e-307)]
    [InlineData(-0.7, 1.6778011938374184)]
    [InlineData(-3.0, 1.9999779095030015)]
    public void Erfc_PiecewiseApproximation_WithinUlps(double x, double expected)
    {
        SpecialFunctionAssert.WithinUlps(expected, XMath.Special.Erfc(x), 4);
    }

    [Trait("Category", "AuditRegression")]
    [Theory]
    [InlineData(1.2661176086124684, 0.36448474146408966)]
    [InlineData(0.24999999999999997, 0.7703465477309968)]
    [InlineData(0.25, 0.7703465477309968)]
    [InlineData(0.3125, 0.7260859551237695)]
    [InlineData(2.0, 0.25539567631050575)]
    [InlineData(3.9999999999999996, 0.1369994576250614)]
    [InlineData(4.0, 0.13699945762506138)]
    [InlineData(36.91467587405916, 0.01527801030261964)]
    [InlineData(1e10, 5.641895835477563e-11)]
    [InlineData(1e300, 5.641895835477562e-301)]
    [InlineData(-0.24367433662771365, 1.3472795346021902)]
    [InlineData(-3.0, 16205.988853999586)]
    public void Erfcx_PiecewiseApproximation_WithinUlps(double x, double expected)
    {
        SpecialFunctionAssert.WithinUlps(expected, XMath.Special.Erfcx(x), 4);
    }

    // 2e^(x²) が溢れる直前の負の x で、e^(x²) の補正が ∞ - ∞ = NaN になっていた
    [Trait("Category", "AuditRegression")]
    [Theory]
    [InlineData(-26.7)]
    [InlineData(-26.65)]
    [InlineData(-26.63)]
    public void Erfcx_NegativeArgumentPastOverflow_IsPositiveInfinity(double x)
    {
        Assert.Equal(double.PositiveInfinity, XMath.Special.Erfcx(x));
    }

    [Trait("Category", "AuditRegression")]
    [Theory]
    [InlineData(0.9117253834057351, 1.2053131342487555)]
    [InlineData(0.49999999999999994, 0.4769362762044698)]
    [InlineData(0.5, 0.4769362762044699)]
    [InlineData(0.1, 0.08885599049425769)]
    [InlineData(-0.3, -0.2724627147267543)]
    [InlineData(0.9999999999999999, 5.8635847487551676)]
    [InlineData(1e-300, 8.86226925452758e-301)]
    public void InverseErf_HalleyRefinement_WithinUlps(double x, double expected)
    {
        SpecialFunctionAssert.WithinUlps(expected, XMath.Special.InverseErf(x), 3);
    }

    [Trait("Category", "AuditRegression")]
    [Theory]
    [InlineData(0.15810566703353612, 0.9980609883623405)]
    [InlineData(0.49999999999999994, 0.47693627620446993)]
    [InlineData(0.5, 0.4769362762044699)]
    [InlineData(1.5, -0.4769362762044699)]
    [InlineData(1.9, -1.1630871536766738)]
    [InlineData(1e-20, 6.601580622355143)]
    [InlineData(1e-300, 26.209469960516124)]
    [InlineData(1e-320, 27.073153719853043)]
    public void InverseErfc_HalleyRefinement_WithinUlps(double q, double expected)
    {
        SpecialFunctionAssert.WithinUlps(expected, XMath.Special.InverseErfc(q), 3);
    }
}
