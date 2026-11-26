using Tuna;

namespace Tuna.Tests;

// 有限参照値は mpmath 1.4.1、100 桁以上で生成し binary64 に丸めた。
// 入力も binary64 の値を正確に高精度へ変換（十進文字列の値とは区別）。
// 参照値生成: https://mpmath.org/doc/current/functions/index.html
[Trait("SpecialFunction", "JacobiElliptic")]
public class JacobiEllipticTests
{

    // https://dlmf.nist.gov/22.5; https://dlmf.nist.gov/22.6; https://dlmf.nist.gov/22.17
    [Theory]
    // 原点sn=0,cn=dn=1。
    [InlineData(0.0, 0.5, 0.0)]
    // m=0ではsin/cos/1。
    [InlineData(0.5, 0.0, 0.479425538604203)]
    // m=1ではtanh/sech。
    [InlineData(0.5, 1.0, 0.46211715726000974)]
    // m=0のすぐ内側。
    [InlineData(0.5, 1e-16, 0.479425538604203)]
    // m=1のすぐ内側。
    [InlineData(0.5, 0.9999999999999999, 0.46211715726000974)]
    // snの奇性とcn/dnの偶性。
    [InlineData(-0.5, 0.5, -0.4707504736556573)]
    // 負のmの虚数モジュラス変換。
    [InlineData(0.5, -1.0, 0.49689119041931196)]
    // m>1の逆数モジュラス変換。
    [InlineData(0.5, 2.0, 0.4449849631683637)]
    // 複数の四分周期を越える。
    [InlineData(5.562224031904115, 0.5, -1.0)]
    // 双曲線関数への退化で極小cn/dn。
    [InlineData(10.0, 1.0, 0.9999999958776927)]
    // m=1のtanhは大引数で1に丸められる。
    [InlineData(711.0, 1.0, 1.0)]
    public void JacobiSN_MatchesReference(double u, double m, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.JacobiSN(u, m), 2e-12, double.Epsilon);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/22.5; https://dlmf.nist.gov/22.6; https://dlmf.nist.gov/22.17
    [Theory]
    // u NaN。
    [InlineData(double.NaN, 0.5, double.NaN)]
    // m NaN。
    [InlineData(0.5, double.NaN, double.NaN)]
    // 無限大uは未対応。
    [InlineData(double.PositiveInfinity, 0.5, double.NaN)]
    // 無限大m。
    [InlineData(0.5, double.PositiveInfinity, double.NaN)]
    public void JacobiSN_SpecialValues(double u, double m, double expected)
    {
        double actual = XMath.Special.JacobiSN(u, m);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // https://dlmf.nist.gov/22.5; https://dlmf.nist.gov/22.6; https://dlmf.nist.gov/22.17
    [Theory]
    // 原点sn=0,cn=dn=1。
    [InlineData(0.0, 0.5, 1.0)]
    // m=0ではsin/cos/1。
    [InlineData(0.5, 0.0, 0.8775825618903728)]
    // m=1ではtanh/sech。
    [InlineData(0.5, 1.0, 0.8868188839700739)]
    // m=0のすぐ内側。
    [InlineData(0.5, 1e-16, 0.8775825618903728)]
    // m=1のすぐ内側。
    [InlineData(0.5, 0.9999999999999999, 0.8868188839700739)]
    // snの奇性とcn/dnの偶性。
    [InlineData(-0.5, 0.5, 0.8822663948904403)]
    // 負のmの虚数モジュラス変換。
    [InlineData(0.5, -1.0, 0.8678128513012924)]
    // m>1の逆数モジュラス変換。
    [InlineData(0.5, 2.0, 0.8955380408190653)]
    // 双曲線関数への退化で極小cn/dn。
    [InlineData(10.0, 1.0, 9.079985933781725e-05)]
    // sechはsubnormalでも非零。coshの中間overflowを検出。
    [InlineData(711.0, 1.0, 3.293467350449585e-309)]
    public void JacobiCN_MatchesReference(double u, double m, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.JacobiCN(u, m), 2e-12, double.Epsilon);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/22.5; https://dlmf.nist.gov/22.6; https://dlmf.nist.gov/22.17
    [Theory]
    // u NaN。
    [InlineData(double.NaN, 0.5, double.NaN)]
    // m NaN。
    [InlineData(0.5, double.NaN, double.NaN)]
    // 無限大uは未対応。
    [InlineData(double.PositiveInfinity, 0.5, double.NaN)]
    // 無限大m。
    [InlineData(0.5, double.PositiveInfinity, double.NaN)]
    public void JacobiCN_SpecialValues(double u, double m, double expected)
    {
        double actual = XMath.Special.JacobiCN(u, m);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // https://dlmf.nist.gov/22.5; https://dlmf.nist.gov/22.6; https://dlmf.nist.gov/22.17
    [Theory]
    // 原点sn=0,cn=dn=1。
    [InlineData(0.0, 0.5, 1.0)]
    // m=0ではsin/cos/1。
    [InlineData(0.5, 0.0, 1.0)]
    // m=1ではtanh/sech。
    [InlineData(0.5, 1.0, 0.8868188839700739)]
    // m=0のすぐ内側。
    [InlineData(0.5, 1e-16, 1.0)]
    // m=1のすぐ内側。
    [InlineData(0.5, 0.9999999999999999, 0.8868188839700739)]
    // snの奇性とcn/dnの偶性。
    [InlineData(-0.5, 0.5, 0.9429724257773857)]
    // 負のmの虚数モジュラス変換。
    [InlineData(0.5, -1.0, 1.1166471488864873)]
    // m>1の逆数モジュラス変換。
    [InlineData(0.5, 2.0, 0.7771594206519663)]
    // 複数の四分周期を越える。
    [InlineData(5.562224031904115, 0.5, 0.7071067811865476)]
    // 双曲線関数への退化で極小cn/dn。
    [InlineData(10.0, 1.0, 9.079985933781725e-05)]
    // sechはsubnormalでも非零。coshの中間overflowを検出。
    [InlineData(711.0, 1.0, 3.293467350449585e-309)]
    public void JacobiDN_MatchesReference(double u, double m, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.JacobiDN(u, m), 2e-12, double.Epsilon);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/22.5; https://dlmf.nist.gov/22.6; https://dlmf.nist.gov/22.17
    [Theory]
    // u NaN。
    [InlineData(double.NaN, 0.5, double.NaN)]
    // m NaN。
    [InlineData(0.5, double.NaN, double.NaN)]
    // 無限大uは未対応。
    [InlineData(double.PositiveInfinity, 0.5, double.NaN)]
    // 無限大m。
    [InlineData(0.5, double.PositiveInfinity, double.NaN)]
    public void JacobiDN_SpecialValues(double u, double m, double expected)
    {
        double actual = XMath.Special.JacobiDN(u, m);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // https://dlmf.nist.gov/22.16
    [Theory]
    // 原点。
    [InlineData(0.0, 0.5, 0.0)]
    // m=0でam=u。
    [InlineData(0.5, 0.0, 0.5)]
    // m=1のGudermannian。
    [InlineData(0.5, 1.0, 0.48038107913372946)]
    // m=1のすぐ内側。
    [InlineData(0.5, 0.9999999999999999, 0.48038107913372946)]
    // 負のu。
    [InlineData(-0.5, 0.5, -0.49014120541425493)]
    // arcsin(sn)の主値に戻さず振幅を連続にする。
    [InlineData(5.562224031904115, 0.5, 4.71238898038469)]
    // 複数周期にまたがる振幅。
    [InlineData(10.0, 0.5, 8.39183082303414)]
    public void JacobiAmplitude_MatchesReference(double u, double m, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.JacobiAmplitude(u, m), 2e-12, 0);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/22.16
    [Theory]
    // 振幅のみm<0未対応。
    [InlineData(0.5, -1.0, double.NaN)]
    // 振幅のみm>1未対応。
    [InlineData(0.5, 1.0000000000000002, double.NaN)]
    // u NaN。
    [InlineData(double.NaN, 0.5, double.NaN)]
    // m NaN。
    [InlineData(0.5, double.NaN, double.NaN)]
    // m=0の振幅はu。
    [InlineData(double.PositiveInfinity, 0.0, double.PositiveInfinity)]
    // m=1の振幅の極限はπ/2（uではない）。
    [InlineData(double.PositiveInfinity, 1.0, 1.5707963267948966)]
    // 負側Gudermannianの極限。
    [InlineData(double.NegativeInfinity, 1.0, -1.5707963267948966)]
    public void JacobiAmplitude_SpecialValues(double u, double m, double expected)
    {
        double actual = XMath.Special.JacobiAmplitude(u, m);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // https://dlmf.nist.gov/22.5; https://dlmf.nist.gov/22.6; https://dlmf.nist.gov/22.17
    [Theory]
    // 同時計算APIの各成分を独立参照値で確認。
    [InlineData(0.5, 0.5, 0.4707504736556573)]
    // 同時計算APIの負パラメータ変換。
    [InlineData(0.5, -1.0, 0.49689119041931196)]
    // 同時計算APIの逆数変換。
    [InlineData(0.5, 2.0, 0.4449849631683637)]
    public void JacobiEllipticSn_MatchesReference(double u, double m, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.JacobiElliptic(u, m).Sn, 5e-13, 0);
    }

    // https://dlmf.nist.gov/22.5; https://dlmf.nist.gov/22.6; https://dlmf.nist.gov/22.17
    [Theory]
    // 同時計算APIの各成分を独立参照値で確認。
    [InlineData(0.5, 0.5, 0.8822663948904403)]
    // 同時計算APIの負パラメータ変換。
    [InlineData(0.5, -1.0, 0.8678128513012924)]
    // 同時計算APIの逆数変換。
    [InlineData(0.5, 2.0, 0.8955380408190653)]
    public void JacobiEllipticCn_MatchesReference(double u, double m, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.JacobiElliptic(u, m).Cn, 5e-13, 0);
    }

    // https://dlmf.nist.gov/22.5; https://dlmf.nist.gov/22.6; https://dlmf.nist.gov/22.17
    [Theory]
    // 同時計算APIの各成分を独立参照値で確認。
    [InlineData(0.5, 0.5, 0.9429724257773857)]
    // 同時計算APIの負パラメータ変換。
    [InlineData(0.5, -1.0, 1.1166471488864873)]
    // 同時計算APIの逆数変換。
    [InlineData(0.5, 2.0, 0.7771594206519663)]
    public void JacobiEllipticDn_MatchesReference(double u, double m, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.JacobiElliptic(u, m).Dn, 5e-13, 0);
    }

    // 二つの代数恒等式は変換後の成分の取違えを検出。https://dlmf.nist.gov/22.6
    [Theory]
    [InlineData(0.5, -1.0)]
    [InlineData(0.5, 0.5)]
    [InlineData(0.5, 2.0)]
    public void JacobiElliptic_Identities(double u, double m)
    {
        var v = XMath.Special.JacobiElliptic(u, m);
        SpecialFunctionAssert.Close(1, v.Sn * v.Sn + v.Cn * v.Cn);
        SpecialFunctionAssert.Close(1, v.Dn * v.Dn + m * v.Sn * v.Sn);
    }

    // 同時計算でもNaNを各成分へ伝播する。https://dlmf.nist.gov/22.2
    [Fact]
    public void JacobiElliptic_NaN()
    {
        var v = XMath.Special.JacobiElliptic(double.NaN, 0.5);
        Assert.True(double.IsNaN(v.Sn) && double.IsNaN(v.Cn) && double.IsNaN(v.Dn));
    }

    // 3Kの丸め値と両隣。零点近傍では絶対誤差で比較する。
    // 他のCN参照値、特に非零subnormalにはこの許容誤差を適用しない。
    // https://dlmf.nist.gov/22.4
    [Theory]
    [InlineData(5.5622240319041145, -8.745831439620406e-16)]
    [InlineData(5.562224031904115, -2.465461604885306e-16)]
    [InlineData(5.562224031904116, 3.814908229849794e-16)]
    [InlineData(-5.562224031904116, 3.814908229849794e-16)]
    [InlineData(-5.562224031904115, -2.465461604885306e-16)]
    [InlineData(-5.5622240319041145, -8.745831439620406e-16)]
    public void JacobiCN_NearZero_MatchesReference(double u, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.JacobiCN(u, 0.5), 0, 1e-15);
        SpecialFunctionAssert.Close(expected, XMath.Special.JacobiElliptic(u, 0.5).Cn, 0, 1e-15);
    }

    // 原点は高精度計算の残差ではなく厳密値で確認。変換と退化経路を含む。
    // https://dlmf.nist.gov/22.5
    [Theory]
    [InlineData(-1.0)]
    [InlineData(0.0)]
    [InlineData(0.5)]
    [InlineData(1.0)]
    [InlineData(2.0)]
    public void JacobiElliptic_OriginHasExactValues(double m)
    {
        var value = XMath.Special.JacobiElliptic(0, m);
        Assert.Equal(0, value.Sn);
        Assert.Equal(1, value.Cn);
        Assert.Equal(1, value.Dn);
        Assert.Equal(0, XMath.Special.JacobiSN(0, m));
        Assert.Equal(1, XMath.Special.JacobiCN(0, m));
        Assert.Equal(1, XMath.Special.JacobiDN(0, m));
        if (m >= 0 && m <= 1)
        {
            Assert.Equal(0, XMath.Special.JacobiAmplitude(0, m));
        }
    }

    // m=1のsech。通常数/非正規数とcoshが溢れる入力の両隣を検証する。
    // 参照値は1/cosh(u)を160桁・220桁で独立計算し、丸めの一致を確認。
    // https://dlmf.nist.gov/22.5
    [Theory]
    [InlineData(707.9999999999999, 6.615106007277568e-308)]
    [InlineData(708.0, 6.615106007276816e-308)]
    [InlineData(708.0000000000001, 6.615106007276064e-308)]
    [InlineData(710.9999999999999, 3.29346735044996e-309)]
    [InlineData(711.0, 3.293467350449585e-309)]
    [InlineData(711.0000000000001, 3.29346735044921e-309)]
    [InlineData(709.089565712824, 2.2250738585073847e-308)]
    [InlineData(709.0895657128241, 2.225073858507132e-308)]
    [InlineData(709.0895657128242, 2.2250738585068793e-308)]
    [InlineData(745.2, double.Epsilon)]
    public void JacobiElliptic_SechRemainsNonzero(double u, double expected)
    {
        foreach (double signedU in new[] { u, -u })
        {
            var value = XMath.Special.JacobiElliptic(signedU, 1);
            Assert.True(value.Cn > 0 && value.Dn > 0,
                $"u={signedU:R}: cn={value.Cn:R}, dn={value.Dn:R}; expected nonzero {expected:R}");
            SpecialFunctionAssert.Close(expected, value.Cn, 2e-12, double.Epsilon);
            SpecialFunctionAssert.Close(expected, value.Dn, 2e-12, double.Epsilon);
            SpecialFunctionAssert.Close(expected, XMath.Special.JacobiCN(signedU, 1), 2e-12, double.Epsilon);
            SpecialFunctionAssert.Close(expected, XMath.Special.JacobiDN(signedU, 1), 2e-12, double.Epsilon);
        }
    }

    [Theory]
    [InlineData(746.0, 1.0)]
    [InlineData(-746.0, -1.0)]
    [InlineData(1000.0, 1.0)]
    [InlineData(-1000.0, -1.0)]
    public void JacobiElliptic_HyperbolicLimitUnderflowsToZero(double u, double sn)
    {
        var value = XMath.Special.JacobiElliptic(u, 1);
        Assert.Equal(sn, value.Sn);
        Assert.Equal(0, value.Cn);
        Assert.Equal(0, value.Dn);
        Assert.Equal(Math.CopySign(Math.PI / 2, u), XMath.Special.JacobiAmplitude(u, 1));
    }

    // R13: 位相の参照値なしでも、実引数のsn,cnは[-1,1]内の有限値。
    // https://dlmf.nist.gov/22.16; https://dlmf.nist.gov/22.17
    [Trait("Category", "ReviewRegression")]
    [Theory]
    [InlineData(1e308, 0.5)]
    [InlineData(1.0, -1e20)]
    public void JacobiElliptic_FiniteInputs_HaveBoundedSnAndCn(double u, double m)
    {
        var (sn, cn, dn) = XMath.Special.JacobiElliptic(u, m);
        Assert.True(double.IsFinite(sn), $"sn={sn:R}");
        Assert.True(double.IsFinite(cn), $"cn={cn:R}");
        Assert.True(double.IsFinite(dn), $"dn={dn:R}");
        Assert.InRange(sn, -1, 1);
        Assert.InRange(cn, -1, 1);
    }

    // double-double 周期還元と負・逆数モジュラス変換の精度検証（参照値は mpmath 1.4.1 の 160/240 桁、巨大引数全域の保証ではない）。
    [Trait("Category", "AuditRegression")]
    [Theory]
    [InlineData(1e9, 0.5, 0.9690463386374097, -0.24687890467480306, 0.7283368704018184)]
    [InlineData(1000000000.1, 0.5, 0.9483057502542496, -0.31735816364909364, 0.741861241754388)]
    [InlineData(1, -1e20, -1.3491076018493962e-10, 1, 1.6793127526961227)]
    [InlineData(1e10, 0.5, -0.21298647760001463, 0.9770551470411168, 0.9885941433064275)]
    [InlineData(1e15, 0.5, -0.8851134754125868, 0.46537526324789996, 0.7799276042181903)]
    [InlineData(1e10, 2, -0.7069069669187547, 0.7073065390067639, 0.023771416521811937)]
    public void JacobiElliptic_LargePhaseRegression_MatchesReference(double u, double m,
        double expectedSn, double expectedCn, double expectedDn)
    {
        foreach (double sign in new[] { 1.0, -1.0 })
        {
            var (sn, cn, dn) = XMath.Special.JacobiElliptic(sign * u, m);
            SpecialFunctionAssert.Close(sign * expectedSn, sn, relativeTolerance: 5e-14);
            SpecialFunctionAssert.Close(expectedCn, cn, relativeTolerance: 5e-14);
            SpecialFunctionAssert.Close(expectedDn, dn, relativeTolerance: 5e-14);
            SpecialFunctionAssert.Close(sign * expectedSn, XMath.Special.JacobiSN(sign * u, m), relativeTolerance: 5e-14);
            SpecialFunctionAssert.Close(expectedCn, XMath.Special.JacobiCN(sign * u, m), relativeTolerance: 5e-14);
            SpecialFunctionAssert.Close(expectedDn, XMath.Special.JacobiDN(sign * u, m), relativeTolerance: 5e-14);
        }
    }

    // K(1-1e-13)のbinary64丸めと両隣、およびK±1e-4。
    // uは実装のEllipticKで生成せず独立参照から固定。cnの1e-22を絶対誤差で通さない。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(16.352942017154653, 9.366964097923162e-22, 3.1627692694650775e-7)]
    [InlineData(16.352942017154657, -1.869449548595287e-22, 3.1627692694650775e-7)]
    [InlineData(16.35294201715466, -1.3105863195113736e-21, 3.1627692694650775e-7)]
    [InlineData(16.352842017154657, 3.162769274710294e-11, 3.1627692852789237e-7)]
    [InlineData(16.353042017154657, -3.1627692747476833e-11, 3.1627692852789237e-7)]
    public void JacobiElliptic_NearQuarterPeriod_PreservesSmallComponents(double u,
        double expectedCn, double expectedDn)
    {
        const double m = 0.9999999999999;
        var value = XMath.Special.JacobiElliptic(u, m);
        SpecialFunctionAssert.Close(expectedCn, value.Cn, relativeTolerance: 5e-13);
        SpecialFunctionAssert.Close(expectedDn, value.Dn, relativeTolerance: 5e-13);
        SpecialFunctionAssert.Close(expectedCn, XMath.Special.JacobiCN(u, m), relativeTolerance: 5e-13);
        SpecialFunctionAssert.Close(expectedDn, XMath.Special.JacobiDN(u, m), relativeTolerance: 5e-13);
    }

    // A4: 変換引数u*sqrt(...)の中間overflow。巨大uの高精度な位相還元は要求しない。
    [Trait("Category", "AuditRegression")]
    [Theory]
    [InlineData(-1e20)]
    [InlineData(1e20)]
    public void JacobiElliptic_HugeFiniteTransformedArgument_RemainsFiniteAndBounded(double m)
    {
        foreach (double u in new[] { -1e308, 1e308 })
        {
            var (sn, cn, dn) = XMath.Special.JacobiElliptic(u, m);
            Assert.True(double.IsFinite(sn) && double.IsFinite(cn) && double.IsFinite(dn),
                $"u={u:R}, m={m:R}, sn={sn:R}, cn={cn:R}, dn={dn:R}");
            Assert.InRange(sn, -1, 1);
            Assert.InRange(cn, -1, 1);
            Assert.InRange(dn, m > 1 ? -1 : 1, m > 1 ? 1 : Math.Sqrt(1 - m));
        }
    }

    // 探索で得た補強点。mpmath 1.4.1で独立に計算し、複数精度でbinary64丸めを確認。
    // 入力・期待値・許容誤差をここに固定する。出典と選定方針は README.md を参照。

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(-11.685708262741493, -1.534308688702743, -0.6048148650887591)]
    [InlineData(1.1240288493645636, -2.6097731565619053, 0.9996902691158583)]
    [InlineData(25.701681099975346, -3.3052113192642003, 0.3937466655368654)]
    [InlineData(-48.20172547861424, 0.08144629840782525, 0.07164146651536288)]
    [InlineData(47.70320105833642, -2.1601289624403623, 0.7993805977069884)]
    [InlineData(15.023061289780074, -4.860783720184287, -0.4048993413289348)]
    public void JacobiSN_ExploredRelativeValues(double u, double m, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.JacobiSN(u, m), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(-11.685708262741493, -1.534308688702743, -0.7963661086257162)]
    [InlineData(1.1240288493645636, -2.6097731565619053, -0.024887061599608993)]
    [InlineData(25.701681099975346, -3.3052113192642003, 0.9192189964195692)]
    [InlineData(-48.20172547861424, 0.08144629840782525, -0.9974304488411851)]
    [InlineData(47.70320105833642, -2.1601289624403623, -0.6008249828441041)]
    [InlineData(15.023061289780074, -4.860783720184287, 0.9143612652509919)]
    // 前回の探索で最大相対誤差となった入力。
    [InlineData(41.913296763222064, -3.8206510906080613, -0.15794445875167554)]
    public void JacobiCN_ExploredRelativeValues(double u, double m, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.JacobiCN(u, m), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(-11.685708262741493, -1.534308688702743, 1.2495005741920429)]
    [InlineData(25.701681099975346, -3.3052113192642003, 1.2298081904180986)]
    [InlineData(-48.20172547861424, 0.08144629840782525, 0.9997909666004502)]
    [InlineData(47.70320105833642, -2.1601289624403623, 1.5428358897119048)]
    [InlineData(41.913296763222064, -3.8206510906080613, 2.173784580259749)]
    [InlineData(12.400984648565682, 0.6201563740645728, 0.9109074652067101)]
    public void JacobiDN_ExploredRelativeValues(double u, double m, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.JacobiDN(u, m), relativeTolerance: 2e-12);
    }

    // 振幅降下（atan2 形式が必要な m≈1, m≪0）と double-double 周期還元の精度検証。
    // 参照値は mpmath 1.4.1 の ellipfun を binary64 入力のまま 300/500 bit で評価して一致を確認。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(26.77344838986187, 0.9999999999999977, 0.9999999923586369, -0.0001236233239394769, 0.00012362333336920323)]
    [InlineData(2.0306241764702904, -6.620225567571701e18, -1.3596725288355352e-05, 0.9999999999075645, 34984.10089172285)]
    [InlineData(-58.16045036126646, 0.2042985093488431, 0.9999976007566453, -0.00219054352913874, 0.8920215641864622)]
    public void JacobiElliptic_AmplitudeDescentAndPeriodReduction_MatchReference(double u, double m,
        double expectedSn, double expectedCn, double expectedDn)
    {
        var (sn, cn, dn) = XMath.Special.JacobiElliptic(u, m);
        SpecialFunctionAssert.Close(expectedSn, sn, relativeTolerance: 2e-12);
        SpecialFunctionAssert.Close(expectedCn, cn, relativeTolerance: 2e-12);
        SpecialFunctionAssert.Close(expectedDn, dn, relativeTolerance: 2e-12);
    }

    // 2K の丸め値の近く（sn の零点）を周期の還元を経て求める。倍精度の K で還元すると相対誤差が 16〜48 になる。
    // u は零点の真上なので、既存の K 近傍の検査（JacobiElliptic_NearQuarterPeriod_PreservesSmallComponents）と同じく相対 5e-13 とする
    // （この許容誤差は入力の条件数を超える前方精度を確かめる回帰検出用で、通常域の要件とは別）。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(7.391274725979748, 0.99, 3.8345806527215675e-17)]
    [InlineData(5.156184226696347, 0.9, -9.872816161820983e-17)]
    public void JacobiSN_NearTwiceQuarterPeriod_PreservesSmallValue(double u, double m, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.JacobiSN(u, m), relativeTolerance: 5e-13);
        SpecialFunctionAssert.Close(expected, XMath.Special.JacobiElliptic(u, m).Sn, relativeTolerance: 5e-13);
    }
}
