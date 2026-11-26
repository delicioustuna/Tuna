using Tuna;

namespace Tuna.Tests;

// 有限参照値は mpmath 1.4.1、100 桁以上で生成し binary64 に丸めた。
// 入力も binary64 の値を正確に高精度へ変換（十進文字列の値とは区別）。
// 参照値生成: https://mpmath.org/doc/current/functions/index.html
[Trait("SpecialFunction", "IncompleteGamma")]
public class IncompleteGammaTests
{
    // 前因子を共有する経路と、級数・漸近展開の経路。mpmath 1.3.0 の100桁と180桁で一致。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(false, 2.5, 0.03743422675270363, 0.5)]
    [InlineData(true, 2.5, 0.0012497305630313753, 10.0)]
    [InlineData(true, 0.001, 0.0003108630710340944, 0.8)]
    [InlineData(false, 10000.0, 0.5013298083399552, 10000.0)]
    public void InverseGamma_TailAndTransitionReferences(bool upper, double a, double probability, double expected)
    {
        double actual = upper ? XMath.Special.InverseRegularizedGammaQ(a, probability)
            : XMath.Special.InverseRegularizedGammaP(a, probability);
        SpecialFunctionAssert.Close(expected, actual, 5e-13);
    }

    // https://dlmf.nist.gov/8.2; https://dlmf.nist.gov/8.4; https://dlmf.nist.gov/8.12
    [Theory]
    // P=1-exp(-x) の微小値で桁落ちを確認。
    [InlineData(1.0, 1e-20, 1e-20)]
    // Q=exp(-x) の極小尾確率。
    [InlineData(1.0, 700.0, 1.0)]
    // 誤差関数に一致する半整数。
    [InlineData(0.5, 0.25, 0.5204998778130465)]
    // 小さいaでQを直接計算する枝。
    [InlineData(1e-12, 0.25, 0.9999999999989557)]
    // 小引数枝内部の境界直前。
    [InlineData(0.3, 0.49999999999999994, 0.8138118046743926)]
    // 小引数枝内部の境界。
    [InlineData(0.3, 0.5, 0.8138118046743926)]
    // 小a級数の切替直前。
    [InlineData(0.5, 1.0999999999999999, 0.8619892624313404)]
    // 連分数へ切替。
    [InlineData(0.5, 1.1, 0.8619892624313404)]
    // x=a+1 の級数側。
    [InlineData(3.5, 4.499999999999999, 0.7473439535034361)]
    // x=a+1 の連分数側。
    [InlineData(3.5, 4.5, 0.7473439535034362)]
    // 前置因子のStirling切替直前。
    [InlineData(9.999999999999998, 10.0, 0.5420702855281481)]
    // 前置因子のStirling切替点。
    [InlineData(10.0, 10.0, 0.5420702855281478)]
    // 大パラメータの遷移領域 x≈a。
    [InlineData(10000.0, 10000.0, 0.5013298083399552)]
    public void RegularizedGammaP_MatchesReference(double a, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.RegularizedGammaP(a, x), 2e-12, double.Epsilon);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/8.2; https://dlmf.nist.gov/8.4; https://dlmf.nist.gov/8.12
    [Theory]
    // x=0端点。
    [InlineData(2.0, 0.0, 0.0)]
    // x=+∞端点。
    [InlineData(2.0, double.PositiveInfinity, 1.0)]
    // 有限xでa→∞。
    [InlineData(double.PositiveInfinity, 2.0, 0.0)]
    // 同時∞は経路依存。
    [InlineData(double.PositiveInfinity, double.PositiveInfinity, double.NaN)]
    // 正則化のaは正。
    [InlineData(0.0, 1.0, double.NaN)]
    // 負のa。
    [InlineData(-1.0, 1.0, double.NaN)]
    // 負のx。
    [InlineData(2.0, -1.0, double.NaN)]
    // a NaN。
    [InlineData(double.NaN, 1.0, double.NaN)]
    // x NaN。
    [InlineData(2.0, double.NaN, double.NaN)]
    public void RegularizedGammaP_SpecialValues(double a, double x, double expected)
    {
        double actual = XMath.Special.RegularizedGammaP(a, x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // https://dlmf.nist.gov/8.2; https://dlmf.nist.gov/8.4; https://dlmf.nist.gov/8.12
    [Theory]
    // P=1-exp(-x) の微小値で桁落ちを確認。
    [InlineData(1.0, 1e-20, 1.0)]
    // Q=exp(-x) の極小尾確率。
    [InlineData(1.0, 700.0, 9.85967654375977e-305)]
    // 誤差関数に一致する半整数。
    [InlineData(0.5, 0.25, 0.4795001221869535)]
    // 小さいaでQを直接計算する枝。
    [InlineData(1e-12, 0.25, 1.0442826344438007e-12)]
    // 小引数枝内部の境界直前。
    [InlineData(0.3, 0.49999999999999994, 0.18618819532560735)]
    // 小引数枝内部の境界。
    [InlineData(0.3, 0.5, 0.18618819532560735)]
    // 小a級数の切替直前。
    [InlineData(0.5, 1.0999999999999999, 0.13801073756865959)]
    // 連分数へ切替。
    [InlineData(0.5, 1.1, 0.13801073756865953)]
    // x=a+1 の級数側。
    [InlineData(3.5, 4.499999999999999, 0.2526560464965639)]
    // x=a+1 の連分数側。
    [InlineData(3.5, 4.5, 0.2526560464965638)]
    // 前置因子のStirling切替直前。
    [InlineData(9.999999999999998, 10.0, 0.457929714471852)]
    // 前置因子のStirling切替点。
    [InlineData(10.0, 10.0, 0.4579297144718522)]
    // 大パラメータの遷移領域 x≈a。
    [InlineData(10000.0, 10000.0, 0.4986701916600448)]
    public void RegularizedGammaQ_MatchesReference(double a, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.RegularizedGammaQ(a, x), 2e-12, double.Epsilon);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/8.2; https://dlmf.nist.gov/8.4; https://dlmf.nist.gov/8.12
    [Theory]
    // x=0端点。
    [InlineData(2.0, 0.0, 1.0)]
    // x=+∞端点。
    [InlineData(2.0, double.PositiveInfinity, 0.0)]
    // 有限xでa→∞。
    [InlineData(double.PositiveInfinity, 2.0, 1.0)]
    // 同時∞は経路依存。
    [InlineData(double.PositiveInfinity, double.PositiveInfinity, double.NaN)]
    // 正則化のaは正。
    [InlineData(0.0, 1.0, double.NaN)]
    // 負のa。
    [InlineData(-1.0, 1.0, double.NaN)]
    // 負のx。
    [InlineData(2.0, -1.0, double.NaN)]
    // a NaN。
    [InlineData(double.NaN, 1.0, double.NaN)]
    // x NaN。
    [InlineData(2.0, double.NaN, double.NaN)]
    public void RegularizedGammaQ_SpecialValues(double a, double x, double expected)
    {
        double actual = XMath.Special.RegularizedGammaQ(a, x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // https://dlmf.nist.gov/8.2; https://dlmf.nist.gov/8.4; https://dlmf.nist.gov/8.12
    [Theory]
    // 半整数の正規化。
    [InlineData(0.5, 0.25, 0.9225620128255849)]
    // 下側級数と補関数の切替。
    [InlineData(2.5, 3.5, 1.0360343155785856)]
    // Γ(a) は巨大でも下側積分は有限。
    [InlineData(171.5, 1.0, 0.0021575768996978696)]
    // Γ(a) が溢れても下側積分は有限。
    [InlineData(200.0, 1.0, 0.0018485939631227091)]
    public void LowerIncompleteGamma_MatchesReference(double a, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.LowerIncompleteGamma(a, x), 5e-13, 0);
    }

    // https://dlmf.nist.gov/8.2; https://dlmf.nist.gov/8.4; https://dlmf.nist.gov/8.12
    [Theory]
    // a=0 はE1で有限。
    [InlineData(0.0, 1.0, 0.21938393439552029)]
    // 負整数aは一般化指数積分へ委譲。
    [InlineData(-1.0, 0.25, 2.070920497841881)]
    // 負非整数aの級数側。
    [InlineData(-2.5, 0.9999999999999999, 0.09655664863127521)]
    // 負非整数aの連分数側。
    [InlineData(-2.5, 1.0, 0.09655664863127517)]
    // 負整数aの直右でも上側関数は整関数。
    [InlineData(-0.9999999999999999, 0.25, 2.070920497841881)]
    // 負整数aの直左。極同士の差の桁落ちを検出。
    [InlineData(-1.0000000000000002, 0.25, 2.0709204978418816)]
    // Γ(a) は溢れるが上側尾部は有限。
    [InlineData(200.0, 1000.0, 6.335068535476873e+162)]
    public void UpperIncompleteGamma_MatchesReference(double a, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.UpperIncompleteGamma(a, x), 2e-12, 0);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/8.2; https://dlmf.nist.gov/8.4; https://dlmf.nist.gov/8.12
    [Theory]
    // 下側積分の零端点。
    [InlineData(2.0, 0.0, 0.0)]
    // 上端∞ではΓ(2)。
    [InlineData(2.0, double.PositiveInfinity, 1.0)]
    // 下側はa>0のみ（上側との違い）。
    [InlineData(0.0, 1.0, double.NaN)]
    // a無限大。
    [InlineData(double.PositiveInfinity, 1.0, double.NaN)]
    // 負のx。
    [InlineData(2.0, -1.0, double.NaN)]
    // a NaN。
    [InlineData(double.NaN, 1.0, double.NaN)]
    // x NaN。
    [InlineData(2.0, double.NaN, double.NaN)]
    public void LowerIncompleteGamma_SpecialValues(double a, double x, double expected)
    {
        double actual = XMath.Special.LowerIncompleteGamma(a, x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/8.2; https://dlmf.nist.gov/8.4; https://dlmf.nist.gov/8.12
    [Theory]
    // x=0でΓ(2)。
    [InlineData(2.0, 0.0, 1.0)]
    // a=0で下限の積分が発散。
    [InlineData(0.0, 0.0, double.PositiveInfinity)]
    // a<0の下限発散は正。
    [InlineData(-1.0, 0.0, double.PositiveInfinity)]
    // 上端∞で0。
    [InlineData(-2.0, double.PositiveInfinity, 0.0)]
    // 無限大a。
    [InlineData(double.PositiveInfinity, 1.0, double.NaN)]
    // 負のx。
    [InlineData(2.0, -1.0, double.NaN)]
    // a NaN。
    [InlineData(double.NaN, 1.0, double.NaN)]
    // x NaN。
    [InlineData(2.0, double.NaN, double.NaN)]
    public void UpperIncompleteGamma_SpecialValues(double a, double x, double expected)
    {
        double actual = XMath.Special.UpperIncompleteGamma(a, x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // https://dlmf.nist.gov/8.2; https://dlmf.nist.gov/8.4; https://dlmf.nist.gov/8.12
    [Theory]
    // a<1 と微小確率。補関数側では大きい分位点。
    [InlineData(0.5, 1e-20, 7.853981633974482e-41)]
    // 指数分布の厳密逆関数で極小尾確率。
    [InlineData(1.0, 1e-100, 1e-100)]
    // 初期値のa=1境界。
    [InlineData(1.0, 0.5, 0.6931471805599453)]
    // a>1 の初期値。
    [InlineData(2.0, 0.25, 0.9612787631147771)]
    // 確率1の直前で端点へ丸めない。
    [InlineData(10.0, 0.9999999999999999, 61.10439016905224)]
    // 大きいaの中央分位点。
    [InlineData(100.0, 0.5, 99.66686491931549)]
    public void InverseRegularizedGammaP_MatchesReference(double a, double probability, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.InverseRegularizedGammaP(a, probability), 2e-12, 0);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/8.2; https://dlmf.nist.gov/8.4; https://dlmf.nist.gov/8.12
    [Theory]
    // 確率0端点。
    [InlineData(2.0, 0.0, 0.0)]
    // 確率1端点。
    [InlineData(2.0, 1.0, double.PositiveInfinity)]
    // a=0。
    [InlineData(0.0, 0.5, double.NaN)]
    // a無限大。
    [InlineData(double.PositiveInfinity, 0.5, double.NaN)]
    // 負の確率。
    [InlineData(2.0, -1e-10, double.NaN)]
    // 1を超える確率。
    [InlineData(2.0, 1.0000000000000002, double.NaN)]
    // a NaN。
    [InlineData(double.NaN, 0.5, double.NaN)]
    // 確率 NaN。
    [InlineData(2.0, double.NaN, double.NaN)]
    public void InverseRegularizedGammaP_SpecialValues(double a, double probability, double expected)
    {
        double actual = XMath.Special.InverseRegularizedGammaP(a, probability);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // https://dlmf.nist.gov/8.2; https://dlmf.nist.gov/8.4; https://dlmf.nist.gov/8.12
    [Theory]
    // a<1 と微小確率。補関数側では大きい分位点。
    [InlineData(0.5, 1e-20, 43.58086671345491)]
    // 指数分布の厳密逆関数で極小尾確率。
    [InlineData(1.0, 1e-100, 230.25850929940458)]
    // 初期値のa=1境界。
    [InlineData(1.0, 0.5, 0.6931471805599453)]
    // a>1 の初期値。
    [InlineData(2.0, 0.25, 2.6926345288896956)]
    // 確率1の直前で端点へ丸めない。
    [InlineData(10.0, 0.9999999999999999, 0.11617212311617982)]
    // 大きいaの中央分位点。
    [InlineData(100.0, 0.5, 99.66686491931549)]
    public void InverseRegularizedGammaQ_MatchesReference(double a, double probability, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.InverseRegularizedGammaQ(a, probability), 2e-12, 0);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/8.2; https://dlmf.nist.gov/8.4; https://dlmf.nist.gov/8.12
    [Theory]
    // 確率0端点。
    [InlineData(2.0, 0.0, double.PositiveInfinity)]
    // 確率1端点。
    [InlineData(2.0, 1.0, 0.0)]
    // a=0。
    [InlineData(0.0, 0.5, double.NaN)]
    // a無限大。
    [InlineData(double.PositiveInfinity, 0.5, double.NaN)]
    // 負の確率。
    [InlineData(2.0, -1e-10, double.NaN)]
    // 1を超える確率。
    [InlineData(2.0, 1.0000000000000002, double.NaN)]
    // a NaN。
    [InlineData(double.NaN, 0.5, double.NaN)]
    // 確率 NaN。
    [InlineData(2.0, double.NaN, double.NaN)]
    public void InverseRegularizedGammaQ_SpecialValues(double a, double probability, double expected)
    {
        double actual = XMath.Special.InverseRegularizedGammaQ(a, probability);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // 負整数とその両隣、|a+m|=1/4の両側、m=170/171、x=1の両側。
    // 参照値はgammainc(a,x,+∞)を160桁・220桁で計算。
    // https://dlmf.nist.gov/8.2; https://dlmf.nist.gov/8.8
    [Theory]
    [InlineData(-1.2500000000000002, 0.25, 2.5623553550613822)]
    [InlineData(-1.25, 0.25, 2.562355355061382)]
    [InlineData(-1.2499999999999998, 0.25, 2.5623553550613813)]
    [InlineData(-0.7500000000000001, 0.25, 1.6979433017804222)]
    [InlineData(-0.75, 0.25, 1.697943301780422)]
    [InlineData(-0.7499999999999999, 0.25, 1.697943301780422)]
    [InlineData(-2.2500000000000004, 0.25, 6.693286543040224)]
    [InlineData(-2.25, 0.25, 6.693286543040221)]
    [InlineData(-2.2499999999999996, 0.25, 6.693286543040219)]
    [InlineData(-1.7500000000000002, 0.25, 4.06467527809744)]
    [InlineData(-1.75, 0.25, 4.064675278097439)]
    [InlineData(-1.7499999999999998, 0.25, 4.064675278097438)]
    [InlineData(-2.0000000000000004, 0.25, 5.1949460156503005)]
    [InlineData(-2.0, 0.25, 5.194946015650299)]
    [InlineData(-1.9999999999999998, 0.25, 5.194946015650297)]
    [InlineData(-5.000000000000001, 0.25, 150.2738887513919)]
    [InlineData(-5.0, 0.25, 150.27388875139172)]
    [InlineData(-4.999999999999999, 0.25, 150.27388875139158)]
    [InlineData(-0.125, 0.75, 0.33093875775564324)]
    [InlineData(-170.125, 0.75, 4974987066664110000.0)]
    [InlineData(-170.875, 0.75, 6146021624647105000.0)]
    [InlineData(-171.125, 0.75, 6594724232311448000.0)]
    [InlineData(-1.0000000000000002, 0.9999999999999999, 0.14849550677592208)]
    [InlineData(-1.0000000000000002, 1.0, 0.14849550677592205)]
    [InlineData(-1.0000000000000002, 1.0000000000000002, 0.14849550677592196)]
    [InlineData(-2.0000000000000004, 0.9999999999999999, 0.10969196719776017)]
    [InlineData(-2.0000000000000004, 1.0, 0.10969196719776013)]
    [InlineData(-2.0000000000000004, 1.0000000000000002, 0.10969196719776005)]
    public void UpperIncompleteGamma_PoleAndBranchBoundaries_MatchReference(double a, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.UpperIncompleteGamma(a, x), 2e-12, 0);
    }

    // 定義域端点の直外。負の最小非正規数を0と同一扱いにしない。
    [Fact]
    public void IncompleteGamma_ImmediatelyOutsideDomainIsNaN()
    {
        Assert.True(double.IsNaN(XMath.Special.RegularizedGammaP(2, -double.Epsilon)));
        Assert.True(double.IsNaN(XMath.Special.RegularizedGammaQ(2, -double.Epsilon)));
        Assert.True(double.IsNaN(XMath.Special.LowerIncompleteGamma(2, -double.Epsilon)));
        Assert.True(double.IsNaN(XMath.Special.UpperIncompleteGamma(2, -double.Epsilon)));
        foreach (double probability in new[] { -double.Epsilon, Math.BitIncrement(1) })
        {
            Assert.True(double.IsNaN(XMath.Special.InverseRegularizedGammaP(2, probability)));
            Assert.True(double.IsNaN(XMath.Special.InverseRegularizedGammaQ(2, probability)));
        }
    }

    private const double RelativeTolerance = 2e-12;

    // R08: DLMF 8.12.15,17: P(a,a)=1/2+(1/3+1/(540a)+O(a^-2))/sqrt(2πa)。
    // a>=1e12で省略項は許容誤差より十分小さい。https://dlmf.nist.gov/8.12
    [Trait("Category", "ReviewRegression")]
    [Theory]
    [InlineData(1e12, false)]
    [InlineData(1e12, true)]
    [InlineData(1e16, false)]
    [InlineData(1e16, true)]
    public void IncompleteGamma_AtLargeShape_MatchesCentralAsymptotic(double a, bool upper)
    {
        double correction = (1.0 / 3 + 1 / (540 * a)) / Math.Sqrt(2 * Math.PI * a);
        double expected = upper ? 0.5 - correction : 0.5 + correction;
        double actual = upper ? XMath.Special.RegularizedGammaQ(a, a) : XMath.Special.RegularizedGammaP(a, a);
        SpecialFunctionAssert.Close(expected, actual);
    }

    [Trait("Category", "AuditRegression")]
    [Theory]
    [InlineData(false, 2, 5e-201, 1e-100)]
    [InlineData(false, 0.5, 0.93480358092187, 1.7000000000000004)]
    [InlineData(true, 2, 0.49324551494238356, 1.7)]
    [InlineData(true, 0.5, 0.4385780260809999, 0.3)]
    public void InverseGamma_TailAndInterior_MatchesReference(bool upper, double a,
        double probability, double expected)
    {
        double actual = upper ? XMath.Special.InverseRegularizedGammaQ(a, probability)
            : XMath.Special.InverseRegularizedGammaP(a, probability);
        SpecialFunctionAssert.Close(expected, actual, RelativeTolerance);
    }

    // Q と前因子がアンダーフローする上側の裾（q が非正規化数の点を含む）。
    // 参照値は mpmath で Q(a, eᵗ) = q を 40/55 桁で解いた値（a = 1 は x = -ln q）。
    [Trait("Category", "AuditRegression")]
    [Theory]
    [InlineData(0.01, 5e-324, 733.3076550567743)]
    [InlineData(0.5, 1e-320, 732.9556523387926)]
    [InlineData(1, 5e-324, 744.4400719213812)]
    [InlineData(3, 1e-320, 749.3752422953545)]
    [InlineData(30, 1e-320, 861.6096655703569)]
    [InlineData(100, 1e-307, 1035.147448255348)]
    [InlineData(150, 1e-300, 1139.6373659465191)]
    [InlineData(200, 1e-305, 1266.125181018081)]
    [InlineData(300, 1e-307, 1480.6971204733104)]
    [InlineData(400, 1e-320, 1713.7005671201275)]
    [InlineData(1000, 1e-320, 2739.775758654535)]
    public void InverseRegularizedGammaQ_TailBeyondUnderflow_MatchesReference(double a, double q, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.InverseRegularizedGammaQ(a, q), RelativeTolerance);
    }

    // Temmeの公開APIの適用域で、小さい裾を直接参照する。P+Q=1だけでは精度を判定しない。
    // 参照は整数aのポアソン和を小さい側から90/150桁で直接足したもの。
    [Trait("Category", "AuditRegression")]
    [Theory]
    [InlineData(9200, 1.0130928844019716e-16, 0.9999999999999999)]
    [InlineData(10900, 1, 1.0988648788660544e-18)]
    public void RegularizedGamma_LargeShapeTails_MatchIndependentReferences(double x, double p, double q)
    {
        SpecialFunctionAssert.Close(p, XMath.Special.RegularizedGammaP(10000, x), RelativeTolerance);
        SpecialFunctionAssert.Close(q, XMath.Special.RegularizedGammaQ(10000, x), RelativeTolerance);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0.1)]
    [InlineData(1)]
    [InlineData(100)]
    [InlineData(10000)]
    public void ProbabilitiesAreBoundedAndMonotone(double a)
    {
        double previousP = 0, previousQ = 1;
        // Logarithmic x grid reaches both tails; the independent references above check accuracy.
        for (int i = 0; i <= 120; i++)
        {
            double x = a * Math.Pow(10, -6 + i * 0.1);
            double p = XMath.Special.RegularizedGammaP(a, x), q = XMath.Special.RegularizedGammaQ(a, x);
            Assert.True(double.IsFinite(p) && double.IsFinite(q));
            Assert.InRange(p, previousP, 1);
            Assert.InRange(q, 0, previousQ);
            SpecialFunctionAssert.Close(1, p + q, 5e-15);
            previousP = p;
            previousQ = q;
        }
    }

    // 探索で得た補強点。mpmath 1.4.1で独立に計算し、複数精度でbinary64丸めを確認。
    // 入力・期待値・許容誤差をここに固定する。出典と選定方針は README.md を参照。

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(0.0009468820043832526, 4.050867474936521e-7, 0.9866977727332953)]
    // 前回の探索で最大相対誤差となった入力。
    [InlineData(296.4174406665836, 69.89982153197833, 7.514392439506457e-90)]
    [InlineData(0.000004894502833172574, 2.0618465058862404e-9, 0.9999049412707719)]
    [InlineData(0.000004726450390207151, 4.4037435890894605e-8, 0.9999226734687724)]
    [InlineData(0.03398419677443966, 37.725356120342134, 1.0)]
    [InlineData(9999, 9199.04000100005, 1.0129950231825189e-16)]
    [InlineData(9999, 9899.005000125006, 0.158651191785893)]
    [InlineData(9999, 9999, 0.5013298748354332)]
    [InlineData(9999, 10098.994999874994, 0.8413487508462459)]
    [InlineData(9999, 10798.95999899995, 0.9999999999999969)]
    [InlineData(10000, 9200, 1.0130928844019716e-16)]
    [InlineData(10000, 9900, 0.15865119219356466)]
    [InlineData(10000, 10000, 0.5013298083399552)]
    [InlineData(10000, 10100, 0.8413487504471796)]
    [InlineData(10000, 10800, 0.9999999999999969)]
    [InlineData(10001, 9200.96000099995, 1.0131907397619251e-16)]
    [InlineData(10001, 9900.995000124994, 0.1586511926011548)]
    [InlineData(10001, 10001, 0.5013297418544508)]
    [InlineData(10001, 10101.004999875006, 0.8413487500481927)]
    [InlineData(10001, 10801.03999900005, 0.9999999999999969)]
    public void RegularizedGammaP_ExploredRelativeValues(double a, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.RegularizedGammaP(a, x), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(0.0009468820043832526, 4.050867474936521e-7, 0.013302227266704719)]
    [InlineData(0.0052615558954497085, 160.022843618791, 1.071830041174554e-74)]
    [InlineData(0.000004894502833172574, 2.0618465058862404e-9, 9.505872922815209e-05)]
    [InlineData(78.05095594828293, 3.567586315362677e-8, 1.0)]
    [InlineData(0.000004726450390207151, 4.4037435890894605e-8, 7.732653122759121e-05)]
    [InlineData(9999, 9199.04000100005, 0.9999999999999999)]
    [InlineData(9999, 9899.005000125006, 0.841348808214107)]
    [InlineData(9999, 9999, 0.49867012516456677)]
    [InlineData(9999, 10098.994999874994, 0.15865124915375403)]
    [InlineData(9999, 10798.95999899995, 3.1100561363860525e-15)]
    [InlineData(10000, 9200, 0.9999999999999999)]
    [InlineData(10000, 9900, 0.8413488078064354)]
    [InlineData(10000, 10000, 0.4986701916600448)]
    [InlineData(10000, 10100, 0.15865124955282037)]
    [InlineData(10000, 10800, 3.1098199621688446e-15)]
    [InlineData(10001, 9200.96000099995, 0.9999999999999999)]
    [InlineData(10001, 9900.995000124994, 0.8413488073988452)]
    [InlineData(10001, 10001, 0.49867025814554927)]
    [InlineData(10001, 10101.004999875006, 0.15865124995180735)]
    [InlineData(10001, 10801.03999900005, 3.1095838399830762e-15)]
    public void RegularizedGammaQ_ExploredRelativeValues(double a, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.RegularizedGammaQ(a, x), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(0.0009468820043832526, 4.050867474936521e-7, 1041.4807311373766)]
    [InlineData(0.000004894502833172574, 2.0618465058862404e-9, 204290.84433304673)]
    [InlineData(0.003902338560480752, 1.400590720727541e-7, 240.95150344881685)]
    [InlineData(0.42061041046996556, 0.000002535757077537614, 0.010530017100399849)]
    [InlineData(2.4328061911290417, 104.96076808889882, 1.2694125307858672)]
    [InlineData(0.000004726450390207151, 4.4037435890894605e-8, 211558.328760285)]
    public void LowerIncompleteGamma_ExploredRelativeValues(double a, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.LowerIncompleteGamma(a, x), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(0.0009468820043832526, 4.050867474936521e-7, 14.040787120767042)]
    [InlineData(0.0052615558954497085, 160.022843618791, 2.0309658686843846e-72)]
    [InlineData(0.000004894502833172574, 2.0618465058862404e-9, 19.42147423590626)]
    [InlineData(0.003902338560480752, 1.400590720727541e-7, 14.731723794694151)]
    [InlineData(78.05095594828293, 3.567586315362677e-8, 1.8121520235830414e+113)]
    [InlineData(0.000004726450390207151, 4.4037435890894605e-8, 16.36033680343391)]
    public void UpperIncompleteGamma_ExploredRelativeValues(double a, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.UpperIncompleteGamma(a, x), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(296.4174406665836, 69.89982153197833, double.PositiveInfinity)]
    [InlineData(558.7984689345237, 0.012654525181154477, 0.0)]
    [InlineData(66.04712842351381, 0.000006371322572708826, 0.0)]
    [InlineData(78.05095594828293, 3.567586315362677e-8, 0.0)]
    [InlineData(9999, 9199.04000100005, double.PositiveInfinity)]
    [InlineData(9999, 9899.005000125006, double.PositiveInfinity)]
    [InlineData(9999, 9999, double.PositiveInfinity)]
    [InlineData(9999, 10098.994999874994, double.PositiveInfinity)]
    [InlineData(9999, 10798.95999899995, double.PositiveInfinity)]
    [InlineData(10000, 9200, double.PositiveInfinity)]
    [InlineData(10000, 9900, double.PositiveInfinity)]
    [InlineData(10000, 10000, double.PositiveInfinity)]
    [InlineData(10000, 10100, double.PositiveInfinity)]
    [InlineData(10000, 10800, double.PositiveInfinity)]
    [InlineData(10001, 9200.96000099995, double.PositiveInfinity)]
    [InlineData(10001, 9900.995000124994, double.PositiveInfinity)]
    [InlineData(10001, 10001, double.PositiveInfinity)]
    [InlineData(10001, 10101.004999875006, double.PositiveInfinity)]
    [InlineData(10001, 10801.03999900005, double.PositiveInfinity)]
    public void LowerIncompleteGamma_ExploredExactValues(double a, double x, double expected)
    {
        Assert.Equal(expected, XMath.Special.LowerIncompleteGamma(a, x));
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(296.4174406665836, 69.89982153197833, double.PositiveInfinity)]
    [InlineData(558.7984689345237, 0.012654525181154477, double.PositiveInfinity)]
    [InlineData(0.08352971822137889, 878.368563369448, 0.0)]
    [InlineData(9999, 9199.04000100005, double.PositiveInfinity)]
    [InlineData(9999, 9899.005000125006, double.PositiveInfinity)]
    [InlineData(9999, 9999, double.PositiveInfinity)]
    [InlineData(9999, 10098.994999874994, double.PositiveInfinity)]
    [InlineData(9999, 10798.95999899995, double.PositiveInfinity)]
    [InlineData(10000, 9200, double.PositiveInfinity)]
    [InlineData(10000, 9900, double.PositiveInfinity)]
    [InlineData(10000, 10000, double.PositiveInfinity)]
    [InlineData(10000, 10100, double.PositiveInfinity)]
    [InlineData(10000, 10800, double.PositiveInfinity)]
    [InlineData(10001, 9200.96000099995, double.PositiveInfinity)]
    [InlineData(10001, 9900.995000124994, double.PositiveInfinity)]
    [InlineData(10001, 10001, double.PositiveInfinity)]
    [InlineData(10001, 10101.004999875006, double.PositiveInfinity)]
    [InlineData(10001, 10801.03999900005, double.PositiveInfinity)]
    public void UpperIncompleteGamma_ExploredExactValues(double a, double x, double expected)
    {
        Assert.Equal(expected, XMath.Special.UpperIncompleteGamma(a, x));
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(558.7984689345237, 0.012654525181154477, 0.0)]
    [InlineData(66.04712842351381, 0.000006371322572708826, 0.0)]
    [InlineData(78.05095594828293, 3.567586315362677e-8, 0.0)]
    public void RegularizedGammaP_ExploredExactValues(double a, double x, double expected)
    {
        Assert.Equal(expected, XMath.Special.RegularizedGammaP(a, x));
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(0.08352971822137889, 878.368563369448, 0.0)]
    public void RegularizedGammaQ_ExploredExactValues(double a, double x, double expected)
    {
        Assert.Equal(expected, XMath.Special.RegularizedGammaQ(a, x));
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(2.1583229421601002, 2.303580557653755e-46, 1.0590427841929565e-21)]
    [InlineData(4.102495449595471, 5.638296889977964e-20, 4.5805091570775715e-05)]
    [InlineData(14.623946037079545, 3.0775217426259214e-20, 0.29652758353512493)]
    [InlineData(4.8325215711390515, 6.046341980142484e-37, 8.129363485595323e-08)]
    [InlineData(8.029862495676905, 1.4204084307054493e-51, 1.756703165954705e-06)]
    [InlineData(15.50379261398889, 8.97706583286133e-52, 0.003372682830178973)]
    public void InverseRegularizedGammaP_ExploredRelativeValues(double a, double p, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.InverseRegularizedGammaP(a, p), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(2.1583229421601002, 2.303580557653755e-46, 110.46974097110407)]
    [InlineData(4.102495449595471, 5.638296889977964e-20, 54.88336149017041)]
    [InlineData(14.623946037079545, 3.0775217426259214e-20, 80.74595805952127)]
    [InlineData(4.8325215711390515, 6.046341980142484e-37, 98.0818639067498)]
    [InlineData(8.029862495676905, 1.4204084307054493e-51, 143.45584464207812)]
    [InlineData(15.50379261398889, 8.97706583286133e-52, 165.1528586873532)]
    public void InverseRegularizedGammaQ_ExploredRelativeValues(double a, double q, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.InverseRegularizedGammaQ(a, q), relativeTolerance: 2e-12);
    }

    // a ≥ 171 では Γ(a) がオーバーフローするため対数で評価する。中心付近の小さい側の裾がアンダーフローしても、0 ではなく +∞ を返すことを確認する。
    [Trait("Category", "AuditRegression")]
    [Fact]
    public void IncompleteGamma_LargeShapeCentralTails_OverflowInsteadOfZero()
    {
        Assert.Equal(double.PositiveInfinity, XMath.Special.LowerIncompleteGamma(61034.26167712388, 51602.628860369434));
        Assert.Equal(double.PositiveInfinity, XMath.Special.UpperIncompleteGamma(1e6, 1.25e6));
        Assert.Equal(double.PositiveInfinity, XMath.Special.LowerIncompleteGamma(1e6, 0.75e6));
    }

    // 負の整数とその隣接 double。x^a が先にオーバーフローしても Γ(a,x) 自体は有限になる点と、オーバーフローしない対照点を検証する（参照値は 100/180 桁で一致を確認）。
    [Trait("Category", "AuditRegression")]
    [Trait("Category", "CoreAudit20261102")]
    [Theory]
    [InlineData(-171.00000000000003, 0.0156, 5.443556388370106e+306)]
    [InlineData(-171.0, 0.0156, 5.443556388369464e+306)]
    [InlineData(-170.99999999999997, 0.0156, 5.4435563883688207e+306)]
    [InlineData(-310.00000000000006, 0.1, 2.9178860820504403e+307)]
    [InlineData(-310.0, 0.1, 2.917886082050059e+307)]
    [InlineData(-309.99999999999994, 0.1, 2.9178860820496774e+307)]
    [InlineData(-170.00000000000003, 0.0156, 8.541895963250281e+304)]
    [InlineData(-170.0, 0.0156, 8.541895963249272e+304)]
    [InlineData(-169.99999999999997, 0.0156, 8.541895963248264e+304)]
    [InlineData(-171.00000000000003, 0.02, 1.91486275061726e+288)]
    [InlineData(-171.0, 0.02, 1.9148627506170474e+288)]
    [InlineData(-170.99999999999997, 0.02, 1.9148627506168348e+288)]
    public void UpperIncompleteGamma_NegativeShapeAndIntegerNeighbors_FixedReferences(double a, double x, double expected)
    {
        double actual = XMath.Special.UpperIncompleteGamma(a, x);
        if (double.IsInfinity(expected) || expected == 0)
        {
            Assert.Equal(expected, actual);
        }
        else
        {
            SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 2e-12,
                absoluteTolerance: Math.Abs(expected) < 2.2250738585072014e-308 ? 2 * double.Epsilon : 0);
        }
    }

    // 上側を独立に評価し、1 - P の引き算による桁落ちを避ける。非正規化数になる確率には絶対 2 ULP を使う（参照値は 100/180 桁で一致を確認）。
    [Trait("Category", "AuditRegression")]
    [Trait("Category", "CoreAudit20261102")]
    [Theory]
    [InlineData(1e-320, 0.1, 1.823e-320)]
    [InlineData(1e-320, 1.1, 1.86e-321)]
    [InlineData(1e-320, 2.0, 4.9e-322)]
    [InlineData(1e-20, 0.1, 1.8229239584193904e-20)]
    [InlineData(1e-20, 1.1, 1.8599090453604014e-21)]
    [InlineData(1e-20, 2.0, 4.890051070806112e-22)]
    [InlineData(1e-10, 0.1, 1.8229239583260838e-10)]
    [InlineData(1e-10, 1.1, 1.8599090455640366e-11)]
    [InlineData(1e-10, 2.0, 4.890051071569974e-12)]
    [InlineData(0.1, 0.1, 0.17244824041414947)]
    [InlineData(0.1, 1.1, 0.020600778209340292)]
    [InlineData(0.1, 2.0, 0.005673823979811528)]
    public void RegularizedGammaQ_TinyShape_FixedReferences(double a, double x, double expected)
    {
        double actual = XMath.Special.RegularizedGammaQ(a, x);
        if (double.IsInfinity(expected) || expected == 0)
        {
            Assert.Equal(expected, actual);
        }
        else
        {
            SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 2e-12,
                absoluteTolerance: Math.Abs(expected) < 2.2250738585072014e-308 ? 2 * double.Epsilon : 0);
        }
    }
    // 正則化した値が非正規数または 0 でも、非正則化した積分は通常の有限値となる。
    [Trait("Category", "AuditRegression")]
    [Trait("Category", "CoreAuditIdentities20261102")]
    [Theory]
    [InlineData(1e-300, 0.5, 0.5597735947761608)]
    [InlineData(1e-300, 1.0, 0.21938393439552029)]
    [InlineData(1e-300, 1.1, 0.18599090453604014)]
    [InlineData(1e-300, 2.0, 0.04890051070806112)]
    [InlineData(1e-308, 0.5, 0.5597735947761608)]
    [InlineData(1e-308, 1.0, 0.21938393439552029)]
    [InlineData(1e-308, 1.1, 0.18599090453604014)]
    [InlineData(1e-308, 2.0, 0.04890051070806112)]
    [InlineData(1e-310, 0.5, 0.5597735947761608)]
    [InlineData(1e-310, 1.0, 0.21938393439552029)]
    [InlineData(1e-310, 1.1, 0.18599090453604014)]
    [InlineData(1e-310, 2.0, 0.04890051070806112)]
    [InlineData(5e-324, 0.5, 0.5597735947761608)]
    [InlineData(5e-324, 1.0, 0.21938393439552029)]
    [InlineData(5e-324, 1.1, 0.18599090453604014)]
    [InlineData(5e-324, 2.0, 0.04890051070806112)]
    public void UpperIncompleteGamma_PreservesFiniteValuesWhenNormalizationLosesRange(double a, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.UpperIncompleteGamma(a, x), relativeTolerance: 2e-12);
    }
}
