using Tuna;

namespace Tuna.Tests;

// 有限参照値は mpmath 1.4.1、100 桁以上で生成し binary64 に丸めた。
// 入力も binary64 の値を正確に高精度へ変換（十進文字列の値とは区別）。
// 参照値生成: https://mpmath.org/doc/current/functions/index.html
[Trait("SpecialFunction", "SphericalBessel")]
public class SphericalBesselTests
{

    // https://dlmf.nist.gov/10.47; https://dlmf.nist.gov/10.49; https://dlmf.nist.gov/10.52
    [Theory]
    // 0次の初等関数。
    [InlineData(0, 1.0, 0.8414709848078965)]
    // 1次の初等関数。
    [InlineData(1, 1.0, 0.3011686789397568)]
    // 原点に近いときの冪。
    [InlineData(2, 1e-05, 6.6666666666190484e-12)]
    // n≫xでの小さいjと大きいy。
    [InlineData(10, 1.0, 7.116552640047314e-11)]
    // Besselコア切替直前。
    [InlineData(1, 1.9999999999999998, 0.4353977749799916)]
    // Besselコア切替。
    [InlineData(1, 2.0, 0.4353977749799916)]
    // 高次数の振動領域。
    [InlineData(10, 25.0, -0.03625328560112857)]
    // j0は偶、y0は奇。
    [InlineData(0, -1.0, 0.8414709848078965)]
    // j1は奇、y1は偶。
    [InlineData(1, -1.0, -0.3011686789397568)]
    // 負のxで次数の偶奇を確認。
    [InlineData(2, -1.0, 0.06203505201137386)]
    public void SphericalBesselJ_MatchesReference(int n, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.SphericalBesselJ(n, x), 2e-12, 0);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/10.47; https://dlmf.nist.gov/10.49; https://dlmf.nist.gov/10.52
    [Theory]
    // 負の次数は未対応。
    [InlineData(-1, 1.0, double.NaN)]
    // 0次の原点。
    [InlineData(0, 0.0, 1.0)]
    // 正次数の原点。
    [InlineData(1, 0.0, 0.0)]
    // 正の無限大。
    [InlineData(1, double.PositiveInfinity, 0.0)]
    // NaN。
    [InlineData(1, double.NaN, double.NaN)]
    public void SphericalBesselJ_SpecialValues(int n, double x, double expected)
    {
        double actual = XMath.Special.SphericalBesselJ(n, x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // https://dlmf.nist.gov/10.47; https://dlmf.nist.gov/10.49; https://dlmf.nist.gov/10.52
    [Theory]
    // 0次の初等関数。
    [InlineData(0, 1.0, -0.5403023058681398)]
    // 1次の初等関数。
    [InlineData(1, 1.0, -1.3817732906760363)]
    // 原点に近いときの冪。
    [InlineData(2, 1e-05, -3000000000049999.5)]
    // n≫xでの小さいjと大きいy。
    [InlineData(10, 1.0, -672215008.2562084)]
    // Besselコア切替直前。
    [InlineData(1, 1.9999999999999998, -0.3506120042760554)]
    // Besselコア切替。
    [InlineData(1, 2.0, -0.35061200427605527)]
    // 高次数の振動領域。
    [InlineData(10, 25.0, -0.02115833930109747)]
    // j0は偶、y0は奇。
    [InlineData(0, -1.0, 0.5403023058681398)]
    // j1は奇、y1は偶。
    [InlineData(1, -1.0, -1.3817732906760363)]
    // 負のxで次数の偶奇を確認。
    [InlineData(2, -1.0, 3.605017566159969)]
    public void SphericalBesselY_MatchesReference(int n, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.SphericalBesselY(n, x), 2e-12, 0);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/10.47; https://dlmf.nist.gov/10.49; https://dlmf.nist.gov/10.52
    [Theory]
    // 負の次数は未対応。
    [InlineData(-1, 1.0, double.NaN)]
    // 0次の原点。
    [InlineData(0, 0.0, double.NegativeInfinity)]
    // 正次数の原点。
    [InlineData(1, 0.0, double.NegativeInfinity)]
    // 正の無限大。
    [InlineData(1, double.PositiveInfinity, 0.0)]
    // NaN。
    [InlineData(1, double.NaN, double.NaN)]
    public void SphericalBesselY_SpecialValues(int n, double x, double expected)
    {
        double actual = XMath.Special.SphericalBesselY(n, x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // https://dlmf.nist.gov/10.47; https://dlmf.nist.gov/10.49; https://dlmf.nist.gov/10.52
    [Theory]
    // j0の最初の零点πの両側。絶対誤差で評価。
    [InlineData(0, 3.1415926535897927, 1.8033970416801674e-16)]
    // j0の最初の零点πの両側。絶対誤差で評価。
    [InlineData(0, 3.141592653589793, 3.8981718325193755e-17)]
    // j0の最初の零点πの両側。絶対誤差で評価。
    [InlineData(0, 3.1415926535897936, -1.023762675176292e-16)]
    public void SphericalBesselJ_NearZero_MatchesReference(int n, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.SphericalBesselJ(n, x), 5e-13, 1e-15);
    }

    // R09: j1(x)=x/3+O(x^3)、j2(x)=x^2/15+O(x^4)。補正は半ulpより小さい。
    // https://dlmf.nist.gov/10.52
    [Trait("Category", "ReviewRegression")]
    [Theory]
    [InlineData(1, 1e-310, 3.333333333333e-311)]
    [InlineData(2, 1e-155, 6.666666666666e-312)]
    public void SphericalBesselJ_TinyArgument_PreservesFiniteSubnormal(int n, double x, double expected)
    {
        Assert.Equal(expected, XMath.Special.SphericalBesselJ(n, x));
    }

    // 探索で得た補強点。mpmath 1.4.1で独立に計算し、複数精度でbinary64丸めを確認。
    // 入力・期待値・許容誤差をここに固定する。出典と選定方針は README.md を参照。

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(40, 0.0000028653869664235593, 2.9989097620067052e-283)]
    [InlineData(54, 0.0001047912577339817, 3.605501536893681e-304)]
    [InlineData(61, 1.5892023784710243, 1.7841614247068463e-91)]
    [InlineData(47, 46.19196072907965, 0.015548086119261022)]
    [InlineData(8, 0.0005118146967734697, 1.3664507730267975e-34)]
    [InlineData(30, 8.679455384627579e-7, 8.013634172447977e-225)]
    public void SphericalBesselJ_ExploredRelativeValues(int n, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.SphericalBesselJ(n, x), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(40, 0.0000028653869664235593, -1.4367073120477485e+286)]
    [InlineData(54, 0.0001047912577339817, -2.428190547371842e+305)]
    // 前回の探索で最大相対誤差となった入力。
    [InlineData(61, 1.5892023784710243, -2.8683135328912466e+88)]
    [InlineData(12, 64.02394504535825, 0.011779131698337119)]
    [InlineData(47, 46.19196072907965, -0.053691984975340584)]
    [InlineData(8, 0.0005118146967734697, -8.410936152327486e+35)]
    public void SphericalBesselY_ExploredRelativeValues(int n, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.SphericalBesselY(n, x), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(77, 0.00020193778039688314, 0.0)]
    [InlineData(87, 0.0003668914095240756, 0.0)]
    [InlineData(60, 9.936941895863955e-8, 0.0)]
    public void SphericalBesselJ_ExploredExactValues(int n, double x, double expected)
    {
        Assert.Equal(expected, XMath.Special.SphericalBesselJ(n, x));
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(77, 0.00020193778039688314, double.NegativeInfinity)]
    [InlineData(87, 0.0003668914095240756, double.NegativeInfinity)]
    [InlineData(60, 9.936941895863955e-8, double.NegativeInfinity)]
    public void SphericalBesselY_ExploredExactValues(int n, double x, double expected)
    {
        Assert.Equal(expected, XMath.Special.SphericalBesselY(n, x));
    }
}
