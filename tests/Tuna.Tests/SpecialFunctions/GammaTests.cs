using Tuna;

namespace Tuna.Tests;

// 有限参照値は mpmath 1.4.1、100 桁以上で生成し binary64 に丸めた。
// 入力も binary64 の値を正確に高精度へ変換（十進文字列の値とは区別）。
// 参照値生成: https://mpmath.org/doc/current/functions/index.html
[Trait("SpecialFunction", "Gamma")]
public class GammaTests
{

    // https://dlmf.nist.gov/5.2; https://dlmf.nist.gov/5.5; https://dlmf.nist.gov/5.11
    [Theory]
    // 半整数の既知値 √π。
    [InlineData(0.5, 1.772453850905516)]
    // 反射公式で負の符号。
    [InlineData(-0.5, -3.544907701811032)]
    // 負の区間ごとの符号交代。
    [InlineData(-1.5, 2.363271801207355)]
    // 小引数近似 1/x の使用領域。
    [InlineData(1e-21, 1.0000000000000001e+21)]
    // 小引数で負側の符号を保持。
    [InlineData(-1e-21, -1.0000000000000001e+21)]
    // 負の整数極のすぐ右。
    [InlineData(-1.9999999999999998, 2251799813685248.5)]
    // 負の整数極のすぐ左。
    [InlineData(-2.0000000000000004, -1125899906842623.5)]
    // Lanczos 有理式評価の切替直前。
    [InlineData(4.999999999999999, 23.999999999999968)]
    // Lanczos 有理式評価の切替直後。
    [InlineData(5.000000000000001, 24.000000000000032)]
    // 累乗を二分する境界の手前。
    [InlineData(139.5, 8.14861879182775e+237)]
    // 累乗を二分する境界の後。
    [InlineData(140.5, 1.1367323214599712e+240)]
    // double のオーバーフロー直前の大きな有限値。
    [InlineData(171.5, 9.4833675668248e+307)]
    public void Gamma_MatchesReference(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.Gamma(x), 5e-13, 0);
    }

    // https://dlmf.nist.gov/5.2; https://dlmf.nist.gov/5.5; https://dlmf.nist.gov/5.11
    [Theory]
    // ln Γ の零点 1 の右隣で相対精度を確認。
    [InlineData(1.0000000000000002, -1.2816762426960008e-16)]
    // 零点 1 の左隣。
    [InlineData(0.9999999999999999, 6.408381213480008e-17)]
    // 零点 2 の右隣。
    [InlineData(2.0000000000000004, 1.8775396131086244e-16)]
    // 零点 2 の左隣。
    [InlineData(1.9999999999999998, -9.387698065543117e-17)]
    // 小引数級数の切替。
    [InlineData(0.25, 1.2880225246980774)]
    // 1 のまわりの級数の端。
    [InlineData(0.75, 0.20328095143129538)]
    // 1 のまわりの級数の端。
    [InlineData(1.25, -0.09827183642181316)]
    // 2 のまわりの級数の端。
    [InlineData(1.75, -0.08440112102048555)]
    // 2 のまわりの級数の端。
    [InlineData(2.25, 0.1248717148923966)]
    // Stirling 展開へ切り替わる直前。
    [InlineData(9.999999999999998, 12.801827480081466)]
    // Stirling 展開へ切り替わる点。
    [InlineData(10.0, 12.801827480081469)]
    // 対数は負の Γ の絶対値に対して定義。
    [InlineData(-3.5, -1.309006684993042)]
    // Γ 自体は溢れるが対数は有限。
    [InlineData(1e+100, 2.2925850929940456e+102)]
    public void LogGamma_MatchesReference(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.LogGamma(x), 5e-13, 0);
    }

    // https://dlmf.nist.gov/5.2; https://dlmf.nist.gov/5.5; https://dlmf.nist.gov/5.11
    [Theory]
    // 半整数の既知値。
    [InlineData(0.5, 0.5641895835477563)]
    // 負の Γ の逆数の符号。
    [InlineData(-0.5, -0.28209479177387814)]
    // 整関数の零点の直右。
    [InlineData(-1.9999999999999998, 4.440892098500625e-16)]
    // 零点の直左。
    [InlineData(-2.0000000000000004, -8.881784197001256e-16)]
    // 逆数の直接評価側。
    [InlineData(170.5, 1.797884546798036e-306)]
    // Γ が大きい領域の対数評価側。
    [InlineData(171.5, 1.054477740057499e-308)]
    // Γ は溢れるが逆数は subnormal で有限。
    [InlineData(180.0, 0.0)]
    public void ReciprocalGamma_MatchesReference(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.ReciprocalGamma(x), 5e-13, double.Epsilon);
    }

    // https://dlmf.nist.gov/5.2; https://dlmf.nist.gov/5.5; https://dlmf.nist.gov/5.11
    [Theory]
    // 双方の Γ が溢れても比は有限。
    [InlineData(200.0, 199.0, 199.0)]
    // 近い大引数の対数差で桁落ちを避ける。
    [InlineData(10000000000.5, 10000000000.0, 99999.99999875)]
    // 非整数差の通常値。
    [InlineData(5.5, 4.0, 8.723796297425586)]
    // 負の分子の符号。
    [InlineData(-0.5, 0.5, -2.0)]
    // 積を直接取る差20の境界。
    [InlineData(21.25, 1.25, 5.720210205895441e+18)]
    // 差21はGamma経由。
    [InlineData(22.25, 1.25, 1.2155446687527813e+20)]
    public void GammaRatio_MatchesReference(double a, double b, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.GammaRatio(a, b), 5e-13, 0);
    }

    // https://dlmf.nist.gov/5.2; https://dlmf.nist.gov/5.5; https://dlmf.nist.gov/5.11
    [Theory]
    // 積による整数次数。
    [InlineData(2.5, 3.0, 39.375)]
    // 負の整数次数の逆積。
    [InlineData(2.5, -2.0, 1.3333333333333333)]
    // 実数次数はGamma比で定義。
    [InlineData(3.5, 0.5, 1.8054066673528202)]
    // 負の整数引数でも有限な積。
    [InlineData(-3.0, 2.0, 6.0)]
    // 直接積の上限20。
    [InlineData(2.5, 20.0, 1.792469119010237e+20)]
    // 上限を超えてGamma比へ切替。
    [InlineData(2.5, 21.0, 4.0330555177730336e+21)]
    // 負の大きい次数。
    [InlineData(2.5, -21.0, -8.524182186062014e-17)]
    public void Pochhammer_MatchesReference(double x, double n, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.Pochhammer(x, n), 5e-13, 0);
    }

    // https://dlmf.nist.gov/5.2; https://dlmf.nist.gov/5.5; https://dlmf.nist.gov/5.11
    [Theory]
    // 積による整数次数。
    [InlineData(2.5, 3.0, 1.875)]
    // 負の整数次数の逆積。
    [InlineData(2.5, -2.0, 0.06349206349206349)]
    // 実数次数はGamma比で定義。
    [InlineData(3.5, 0.5, 1.9386213994279082)]
    // 負の整数引数でも有限な積。
    [InlineData(-3.0, 2.0, 12.0)]
    // 直接積の上限20。
    [InlineData(2.5, 20.0, -90589426099245.8)]
    // 上限を超えてGamma比へ切替。
    [InlineData(2.5, 21.0, 1585314956736801.5)]
    // 負の大きい次数。
    [InlineData(2.5, -21.0, 2.6377762035407498e-23)]
    public void FallingFactorial_MatchesReference(double x, double n, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.FallingFactorial(x, n), 5e-13, 0);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/5.2; https://dlmf.nist.gov/5.5; https://dlmf.nist.gov/5.11
    [Theory]
    // +0 の片側極。
    [InlineData(0.0, double.PositiveInfinity)]
    // -0 の片側極。
    [InlineData(-0.0, double.NegativeInfinity)]
    // 負の整数では二側の符号が異なる。
    [InlineData(-1.0, double.NaN)]
    // 階乗テーブル直後でオーバーフロー。
    [InlineData(172.0, double.PositiveInfinity)]
    // 正の無限大。
    [InlineData(double.PositiveInfinity, double.PositiveInfinity)]
    // 負の無限大。
    [InlineData(double.NegativeInfinity, double.NaN)]
    // NaN を伝播。
    [InlineData(double.NaN, double.NaN)]
    public void Gamma_SpecialValues(double x, double expected)
    {
        double actual = XMath.Special.Gamma(x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/5.2; https://dlmf.nist.gov/5.5; https://dlmf.nist.gov/5.11
    [Theory]
    // 原点の対数極。
    [InlineData(0.0, double.PositiveInfinity)]
    // 絶対値の対数は整数極で +∞。
    [InlineData(-2.0, double.PositiveInfinity)]
    // 既知の零点。
    [InlineData(1.0, 0.0)]
    // 既知の零点。
    [InlineData(2.0, 0.0)]
    // 正の無限大。
    [InlineData(double.PositiveInfinity, double.PositiveInfinity)]
    // 公開 API の無限大規約。
    [InlineData(double.NegativeInfinity, double.PositiveInfinity)]
    // NaN を伝播。
    [InlineData(double.NaN, double.NaN)]
    public void LogGamma_SpecialValues(double x, double expected)
    {
        double actual = XMath.Special.LogGamma(x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/5.2; https://dlmf.nist.gov/5.5; https://dlmf.nist.gov/5.11
    [Theory]
    // +0 の符号。
    [InlineData(0.0, 1.0)]
    // -0 の符号。
    [InlineData(-0.0, -1.0)]
    // 負の半整数。
    [InlineData(-0.5, -1.0)]
    // 隣の区間で符号交代。
    [InlineData(-1.5, 1.0)]
    // 整数極には符号なし。
    [InlineData(-2.0, double.NaN)]
    // 正の無限大。
    [InlineData(double.PositiveInfinity, 1.0)]
    // 負の無限大。
    [InlineData(double.NegativeInfinity, double.NaN)]
    // NaN を伝播。
    [InlineData(double.NaN, double.NaN)]
    public void SignGamma_SpecialValues(double x, double expected)
    {
        double actual = XMath.Special.SignGamma(x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/5.2; https://dlmf.nist.gov/5.5; https://dlmf.nist.gov/5.11
    [Theory]
    // 原点は零点。
    [InlineData(0.0, 0.0)]
    // 負整数は零点。
    [InlineData(-20.0, 0.0)]
    // 正の無限大で0。
    [InlineData(double.PositiveInfinity, 0.0)]
    // 負の無限大。
    [InlineData(double.NegativeInfinity, double.NaN)]
    // NaN を伝播。
    [InlineData(double.NaN, double.NaN)]
    public void ReciprocalGamma_SpecialValues(double x, double expected)
    {
        double actual = XMath.Special.ReciprocalGamma(x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/5.2; https://dlmf.nist.gov/5.5; https://dlmf.nist.gov/5.11
    [Theory]
    // 共に極: 残差の比 (-1)^(2-3) 3!/2!。
    [InlineData(-2.0, -3.0, -3.0)]
    // 分母だけ極なら0。
    [InlineData(1.0, -2.0, 0.0)]
    // 分子だけ極には符号が定まらない。
    [InlineData(-2.0, 1.0, double.NaN)]
    // 同じ有限引数。
    [InlineData(3.0, 3.0, 1.0)]
    // 第1引数 NaN。
    [InlineData(double.NaN, 1.0, double.NaN)]
    // 第2引数 NaN。
    [InlineData(1.0, double.NaN, double.NaN)]
    public void GammaRatio_SpecialValues(double a, double b, double expected)
    {
        double actual = XMath.Special.GammaRatio(a, b);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/5.2; https://dlmf.nist.gov/5.5; https://dlmf.nist.gov/5.11
    [Theory]
    // 0次の空積。
    [InlineData(0.0, 0.0, 1.0)]
    // 負の整数で多項式としての値。
    [InlineData(-3.0, 4.0, 0.0)]
    // 第1引数 NaN。
    [InlineData(double.NaN, 1.0, double.NaN)]
    // 次数 NaN。
    [InlineData(1.0, double.NaN, double.NaN)]
    public void Pochhammer_SpecialValues(double x, double n, double expected)
    {
        double actual = XMath.Special.Pochhammer(x, n);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/5.2; https://dlmf.nist.gov/5.5; https://dlmf.nist.gov/5.11
    [Theory]
    // 0次の空積。
    [InlineData(0.0, 0.0, 1.0)]
    // 負の整数で多項式としての値。
    [InlineData(-3.0, 4.0, 360.0)]
    // 第1引数 NaN。
    [InlineData(double.NaN, 1.0, double.NaN)]
    // 次数 NaN。
    [InlineData(1.0, double.NaN, double.NaN)]
    public void FallingFactorial_SpecialValues(double x, double n, double expected)
    {
        double actual = XMath.Special.FallingFactorial(x, n);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // 負側で underflow しても Γ の区間の符号を失わない。https://dlmf.nist.gov/5.2
    [Fact]
    public void Gamma_NegativeUnderflowPreservesSign()
    {
        Assert.Equal(long.MinValue, BitConverter.DoubleToInt64Bits(XMath.Special.Gamma(-200.5)));
        Assert.Equal(0L, BitConverter.DoubleToInt64Bits(XMath.Special.Gamma(-201.5)));
    }

    // R03: Γ(x+1/2)/Γ(x)=sqrt(x)(1-1/(8x)+O(x^-2))。
    // x=1e16 で補正は比較の許容誤差より十分小さい。https://dlmf.nist.gov/5.11
    [Trait("Category", "ReviewRegression")]
    [Fact]
    public void Pochhammer_LargeBaseAndHalfShift_MatchesAsymptotic()
    {
        SpecialFunctionAssert.Close(1e8, XMath.Special.Pochhammer(1e16, 0.5));
    }

    [Trait("Category", "ReviewRegression")]
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void RisingAndFallingFactorial_LargeBase_HaveTrueOverflow(bool falling)
    {
        double value = falling ? XMath.Special.FallingFactorial(1e20, 21) : XMath.Special.Pochhammer(1e20, 21);
        Assert.Equal(double.PositiveInfinity, value);
    }

    // R04: 真の逆積は約1e-320で非零。積を先に作るとオーバーフローする。
    [Trait("Category", "ReviewRegression")]
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ReciprocalGammaProducts_PreserveSubnormalResult(bool rising)
    {
        double value = rising ? XMath.Special.Pochhammer(1e16, -20) : XMath.Special.GammaRatio(1e16, 1e16 + 20);
        // 1e-320 からの補正は最小subnormalの半分より小さく、期待値は同じ丸め値になる。
        Assert.Equal(1e-320, value);
    }

    // 探索で得た補強点。mpmath 1.4.1で独立に計算し、複数精度でbinary64丸めを確認。
    // 入力・期待値・許容誤差をここに固定する。出典と選定方針は README.md を参照。

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(1.3496341385957334e-8, 74094153.63856904)]
    [InlineData(0.1606125499406396, 5.787898447961044)]
    [InlineData(1.2592061358030995, 0.9045522240415143)]
    [InlineData(3.518198326901962e-9, 284236391.7698227)]
    [InlineData(-82.06834027183578, -2.2943245550215998e-122)]
    [InlineData(-93.90155104762762, 1.485307527210871e-145)]
    public void Gamma_ExploredRelativeValues(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.Gamma(x), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(1.3496341385957334e-8, 18.120847188888757)]
    [InlineData(0.1606125499406396, 1.755769263342519)]
    [InlineData(0.000006205439268791562, 11.990080767254115)]
    [InlineData(1.2592061358030995, -0.10031523779088243)]
    [InlineData(-82.06834027183578, -280.08494285661664)]
    [InlineData(-93.90155104762762, -333.47921664429106)]
    public void LogGamma_ExploredRelativeValues(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.LogGamma(x), relativeTolerance: 2e-12);
    }

    // |x| < 1/4 の負の x では ln |Γ(x)| = ln Γ(1 + x) - ln |x| を使い、非正規化数での sin(π|x|) の桁落ちを避ける（参照値は mpmath）。
    [Trait("Category", "AuditRegression")]
    [Theory]
    [InlineData(-4.9406564584124654e-324, 744.44007192138126231)]
    [InlineData(-1e-310, 713.8013788281541651)]
    [InlineData(-1e-200, 460.51701859880913682)]
    [InlineData(-1e-20, 46.051701859880913735)]
    [InlineData(-0.1, 2.3689613327287886039)]
    [InlineData(-0.2, 1.7614975908339379186)]
    public void LogGamma_SmallNegativeArguments_MatchReference(double x, double expected)
    {
        SpecialFunctionAssert.WithinUlps(expected, XMath.Special.LogGamma(x), 2);
    }

    // 負の x の 1/Γ(x) = Γ(1 - x) sin(πx)/π で、1 - x が 2 の冪をまたいで丸まる誤差を ψ(1 - x) で補正する（参照値は mpmath 300 桁）。
    [Trait("Category", "AuditRegression")]
    [Theory]
    [InlineData(-127.3, 3.3232609522447942086e+213)]
    [InlineData(-63.73, 9.8413814882638637597e+87)]
    public void ReciprocalGamma_OneMinusXCrossingPowerOfTwo_MatchesReference(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.ReciprocalGamma(x), 1e-15);
    }

    // 1/Γ(x) = x + γx² + O(x³)。|x| < 1e-20 では γx² が x の半 ULP 未満のため、正しく丸めた値は x そのもの（Γ(x) のオーバーフローを回避）。
    [Trait("Category", "AuditRegression")]
    [Theory]
    [InlineData(double.Epsilon)]
    [InlineData(1e-320)]
    [InlineData(1e-310)]
    [InlineData(2.2250738585072014e-308)]
    [InlineData(1e-300)]
    [InlineData(9.9e-21)]
    [InlineData(-double.Epsilon)]
    [InlineData(-1e-310)]
    [InlineData(-9.9e-21)]
    public void ReciprocalGamma_TinyArgument_IsTheArgument(double x)
    {
        Assert.Equal(x, XMath.Special.ReciprocalGamma(x));
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(1.3496341385957334e-8, 1.0)]
    [InlineData(0.1606125499406396, 1.0)]
    [InlineData(1.2592061358030995, 1.0)]
    [InlineData(-82.06834027183578, -1.0)]
    [InlineData(-93.90155104762762, 1.0)]
    [InlineData(-4.6101880019901245, -1.0)]
    public void SignGamma_ExploredRelativeValues(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.SignGamma(x), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(1.3496341385957334e-8, 1.3496341491097876e-08)]
    [InlineData(0.1606125499406396, 0.1727742822357706)]
    [InlineData(1.2592061358030995, 1.1055193646332853)]
    [InlineData(3.518198326901962e-9, 3.5181983340465757e-09)]
    [InlineData(-82.06834027183578, -4.358581255695908e+121)]
    // 前回の探索で最大相対誤差となった入力。
    [InlineData(-63.06770706967952, 1.7646865181389668e+86)]
    [InlineData(-93.90155104762762, 6.732612483812107e+144)]
    public void ReciprocalGamma_ExploredRelativeValues(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.ReciprocalGamma(x), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(0.016344850629496345, 0.0033591846257550666, 0.20402730858300847)]
    [InlineData(1.1539970509379556, 0.00013520428500724343, 0.00012598377698040434)]
    [InlineData(0.00012755719073028734, 0.01002759530413384, 79.05655272222307)]
    [InlineData(6.783268155271497, 14.530625807565752, 1.9233293104146167e-08)]
    [InlineData(0.0005100828619827718, 0.0006119196206666883, 1.1997178881475858)]
    [InlineData(95.60908474492403, 1.536617613531539, 1.9590185426352334e+147)]
    public void GammaRatio_ExploredRelativeValues(double a, double b, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.GammaRatio(a, b), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(38.12494750947028, 1.6572741147026804, 423.29131608735184)]
    [InlineData(35.5995754391552, 3.8711296478741595, 1179216.0131476982)]
    [InlineData(20.12179351682665, -3.699088413216545, 2.3976597667051028e-05)]
    [InlineData(38.15100625629641, -8.973847849255822, 2.3203961410079413e-14)]
    [InlineData(35.06607806326825, 4.636471204950128, 18337961.92830241)]
    [InlineData(15.312323712376319, -7.090354624872772, 3.922302039431666e-08)]
    public void Pochhammer_ExploredRelativeValues(double x, double n, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.Pochhammer(x, n), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(38.12494750947028, 1.6572741147026804, 411.3669574703187)]
    [InlineData(35.5995754391552, 3.8711296478741595, 862588.7261781995)]
    [InlineData(20.12179351682665, -3.699088413216545, 1.0043595679748676e-05)]
    [InlineData(38.15100625629641, -8.973847849255822, 2.167621293437745e-15)]
    [InlineData(35.06607806326825, 4.636471204950128, 11325432.35933194)]
    [InlineData(15.312323712376319, -7.090354624872772, 7.813002520045043e-10)]
    public void FallingFactorial_ExploredRelativeValues(double x, double n, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.FallingFactorial(x, n), relativeTolerance: 2e-12);
    }
}
