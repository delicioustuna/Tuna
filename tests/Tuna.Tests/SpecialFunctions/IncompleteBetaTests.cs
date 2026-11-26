using Tuna;

namespace Tuna.Tests;

// 有限参照値は mpmath 1.4.1、100 桁以上で生成し binary64 に丸めた。
// 入力も binary64 の値を正確に高精度へ変換（十進文字列の値とは区別）。
// 参照値生成: https://mpmath.org/doc/current/functions/index.html
[Trait("SpecialFunction", "IncompleteBeta")]
public class IncompleteBetaTests
{
    // 前因子を共有する経路と、級数・漸近展開の経路。mpmath 1.3.0 の100桁と180桁で一致。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(false, 2.5, 3.5, 0.4869041915261174, 0.4)]
    [InlineData(true, 0.001, 0.2, 0.004147512325232789, 0.7000000000000001)]
    [InlineData(false, 500.0, 700.0, 0.5012979220663781, 0.4166666666666667)]
    public void InverseBeta_TailAndTransitionReferences(bool upper, double a, double b, double probability, double expected)
    {
        double actual = upper ? XMath.Special.InverseRegularizedIncompleteBetaComplement(a, b, probability)
            : XMath.Special.InverseRegularizedIncompleteBeta(a, b, probability);
        SpecialFunctionAssert.Close(expected, actual, 5e-13);
    }

    // https://dlmf.nist.gov/8.17; https://dlmf.nist.gov/8.18
    [Theory]
    // 一様分布でI=x。
    [InlineData(1.0, 1.0, 0.25, 0.25)]
    // 半整数の逆正弦分布。
    [InlineData(0.5, 0.5, 0.25, 0.3333333333333333)]
    // a<1、xが0に近い尾部。
    [InlineData(0.1, 5.0, 1e-20, 0.012233374999999998)]
    // b<1、xが1の直前の補尾部。
    [InlineData(5.0, 0.1, 0.9999999999999999, 0.9689481620022673)]
    // (a+1)/(a+b+2) の切替直前。
    [InlineData(2.0, 3.0, 0.4285714285714285, 0.5735110370678882)]
    // 切替直後。
    [InlineData(2.0, 3.0, 0.4285714285714286, 0.5735110370678884)]
    // Stirling前置因子の切替直前。
    [InlineData(9.999999999999998, 12.0, 0.45, 0.48831354307835306)]
    // 切替点。
    [InlineData(10.0, 12.0, 0.45, 0.4883135430783529)]
    // Betaはunderflowしても正則化値は1/2。
    [InlineData(1000.0, 1000.0, 0.5, 0.5)]
    // 大パラメータの遷移領域。
    [InlineData(100.0, 200.0, 0.3, 0.10884306564490975)]
    public void RegularizedIncompleteBeta_MatchesReference(double a, double b, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.RegularizedIncompleteBeta(a, b, x), 2e-12, 0);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/8.17; https://dlmf.nist.gov/8.18
    [Theory]
    // x=0端点。
    [InlineData(2.0, 3.0, 0.0, 0.0)]
    // x=1端点。
    [InlineData(2.0, 3.0, 1.0, 1.0)]
    // a=0。
    [InlineData(0.0, 3.0, 0.5, double.NaN)]
    // b=0。
    [InlineData(2.0, 0.0, 0.5, double.NaN)]
    // 無限大パラメータ。
    [InlineData(double.PositiveInfinity, 3.0, 0.5, double.NaN)]
    // x<0。
    [InlineData(2.0, 3.0, -1e-10, double.NaN)]
    // x>1。
    [InlineData(2.0, 3.0, 1.0000000000000002, double.NaN)]
    // a NaN。
    [InlineData(double.NaN, 3.0, 0.5, double.NaN)]
    // b NaN。
    [InlineData(2.0, double.NaN, 0.5, double.NaN)]
    // x NaN。
    [InlineData(2.0, 3.0, double.NaN, double.NaN)]
    public void RegularizedIncompleteBeta_SpecialValues(double a, double b, double x, double expected)
    {
        double actual = XMath.Special.RegularizedIncompleteBeta(a, b, x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // https://dlmf.nist.gov/8.17; https://dlmf.nist.gov/8.18
    [Theory]
    // 一様分布でI=x。
    [InlineData(1.0, 1.0, 0.25, 0.75)]
    // 半整数の逆正弦分布。
    [InlineData(0.5, 0.5, 0.25, 0.6666666666666666)]
    // a<1、xが0に近い尾部。
    [InlineData(0.1, 5.0, 1e-20, 0.987766625)]
    // b<1、xが1の直前の補尾部。
    [InlineData(5.0, 0.1, 0.9999999999999999, 0.03105183799773269)]
    // (a+1)/(a+b+2) の切替直前。
    [InlineData(2.0, 3.0, 0.4285714285714285, 0.42648896293211175)]
    // 切替直後。
    [InlineData(2.0, 3.0, 0.4285714285714286, 0.4264889629321116)]
    // Stirling前置因子の切替直前。
    [InlineData(9.999999999999998, 12.0, 0.45, 0.5116864569216469)]
    // 切替点。
    [InlineData(10.0, 12.0, 0.45, 0.5116864569216472)]
    // Betaはunderflowしても正則化値は1/2。
    [InlineData(1000.0, 1000.0, 0.5, 0.5)]
    // 大パラメータの遷移領域。
    [InlineData(100.0, 200.0, 0.3, 0.8911569343550902)]
    public void RegularizedIncompleteBetaComplement_MatchesReference(double a, double b, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.RegularizedIncompleteBetaComplement(a, b, x), 2e-12, 0);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/8.17; https://dlmf.nist.gov/8.18
    [Theory]
    // x=0端点。
    [InlineData(2.0, 3.0, 0.0, 1.0)]
    // x=1端点。
    [InlineData(2.0, 3.0, 1.0, 0.0)]
    // a=0。
    [InlineData(0.0, 3.0, 0.5, double.NaN)]
    // b=0。
    [InlineData(2.0, 0.0, 0.5, double.NaN)]
    // 無限大パラメータ。
    [InlineData(double.PositiveInfinity, 3.0, 0.5, double.NaN)]
    // x<0。
    [InlineData(2.0, 3.0, -1e-10, double.NaN)]
    // x>1。
    [InlineData(2.0, 3.0, 1.0000000000000002, double.NaN)]
    // a NaN。
    [InlineData(double.NaN, 3.0, 0.5, double.NaN)]
    // b NaN。
    [InlineData(2.0, double.NaN, 0.5, double.NaN)]
    // x NaN。
    [InlineData(2.0, 3.0, double.NaN, double.NaN)]
    public void RegularizedIncompleteBetaComplement_SpecialValues(double a, double b, double x, double expected)
    {
        double actual = XMath.Special.RegularizedIncompleteBetaComplement(a, b, x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // https://dlmf.nist.gov/8.17; https://dlmf.nist.gov/8.18
    [Theory]
    // 非正則化の既知の多項式値。
    [InlineData(2.0, 3.0, 0.25, 0.021809895833333332)]
    // 半整数の既知値。
    [InlineData(0.5, 0.5, 0.25, 1.0471975511965979)]
    // 非常に小さいBetaの半分は非零。
    [InlineData(500.0, 500.0, 0.5, 7.399507995628054e-303)]
    public void IncompleteBeta_MatchesReference(double a, double b, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.IncompleteBeta(a, b, x), 2e-12, 0);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/8.17; https://dlmf.nist.gov/8.18
    [Theory]
    // 零端点。
    [InlineData(2.0, 3.0, 0.0, 0.0)]
    // 完全Betaへの接続。
    [InlineData(2.0, 3.0, 1.0, 0.08333333333333333)]
    // a定義域外。
    [InlineData(0.0, 3.0, 0.5, double.NaN)]
    // b定義域外。
    [InlineData(2.0, 0.0, 0.5, double.NaN)]
    // x定義域外。
    [InlineData(2.0, 3.0, -1.0, double.NaN)]
    // x NaN。
    [InlineData(2.0, 3.0, double.NaN, double.NaN)]
    public void IncompleteBeta_SpecialValues(double a, double b, double x, double expected)
    {
        double actual = XMath.Special.IncompleteBeta(a, b, x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // https://dlmf.nist.gov/8.17; https://dlmf.nist.gov/8.18
    [Theory]
    // 一様分布で正確な逆関数。
    [InlineData(1.0, 1.0, 1e-20, 1e-20)]
    // a,b<1 で端の密度が発散する分布。
    [InlineData(0.5, 0.5, 1e-20, 2.4674011002723395e-40)]
    // 小パラメータ初期値。
    [InlineData(0.5, 2.0, 0.25, 0.028309543718691405)]
    // 双方1以上の初期値。
    [InlineData(2.0, 3.0, 0.25, 0.2430220837560763)]
    // 非対称分布の中央分位点。
    [InlineData(10.0, 20.0, 0.5, 0.32958487942447046)]
    // 確率1の直前。
    [InlineData(2.0, 3.0, 0.9999999999999999, 0.999996972270482)]
    public void InverseRegularizedIncompleteBeta_MatchesReference(double a, double b, double probability, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.InverseRegularizedIncompleteBeta(a, b, probability), 2e-12, double.Epsilon);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/8.17; https://dlmf.nist.gov/8.18
    [Theory]
    // 確率0端点。
    [InlineData(2.0, 3.0, 0.0, 0.0)]
    // 確率1端点。
    [InlineData(2.0, 3.0, 1.0, 1.0)]
    // a=0。
    [InlineData(0.0, 3.0, 0.5, double.NaN)]
    // b=0。
    [InlineData(2.0, 0.0, 0.5, double.NaN)]
    // a無限大。
    [InlineData(double.PositiveInfinity, 3.0, 0.5, double.NaN)]
    // 負の確率。
    [InlineData(2.0, 3.0, -0.1, double.NaN)]
    // 確率>1。
    [InlineData(2.0, 3.0, 1.0000000000000002, double.NaN)]
    // a NaN。
    [InlineData(double.NaN, 3.0, 0.5, double.NaN)]
    // b NaN。
    [InlineData(2.0, double.NaN, 0.5, double.NaN)]
    // 確率 NaN。
    [InlineData(2.0, 3.0, double.NaN, double.NaN)]
    public void InverseRegularizedIncompleteBeta_SpecialValues(double a, double b, double probability, double expected)
    {
        double actual = XMath.Special.InverseRegularizedIncompleteBeta(a, b, probability);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // https://dlmf.nist.gov/8.17; https://dlmf.nist.gov/8.18
    [Theory]
    // 一様分布で正確な逆関数。
    [InlineData(1.0, 1.0, 1e-20, 1.0)]
    // a,b<1 で端の密度が発散する分布。
    [InlineData(0.5, 0.5, 1e-20, 1.0)]
    // 小パラメータ初期値。
    [InlineData(0.5, 2.0, 0.25, 0.3112241790384896)]
    // 双方1以上の初期値。
    [InlineData(2.0, 3.0, 0.25, 0.5436782854190803)]
    // 非対称分布の中央分位点。
    [InlineData(10.0, 20.0, 0.5, 0.32958487942447046)]
    // 確率1の直前。
    [InlineData(2.0, 3.0, 0.9999999999999999, 4.301594725588786e-09)]
    public void InverseRegularizedIncompleteBetaComplement_MatchesReference(double a, double b, double probability, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.InverseRegularizedIncompleteBetaComplement(a, b, probability), 2e-12, double.Epsilon);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/8.17; https://dlmf.nist.gov/8.18
    [Theory]
    // 確率0端点。
    [InlineData(2.0, 3.0, 0.0, 1.0)]
    // 確率1端点。
    [InlineData(2.0, 3.0, 1.0, 0.0)]
    // a=0。
    [InlineData(0.0, 3.0, 0.5, double.NaN)]
    // b=0。
    [InlineData(2.0, 0.0, 0.5, double.NaN)]
    // a無限大。
    [InlineData(double.PositiveInfinity, 3.0, 0.5, double.NaN)]
    // 負の確率。
    [InlineData(2.0, 3.0, -0.1, double.NaN)]
    // 確率>1。
    [InlineData(2.0, 3.0, 1.0000000000000002, double.NaN)]
    // a NaN。
    [InlineData(double.NaN, 3.0, 0.5, double.NaN)]
    // b NaN。
    [InlineData(2.0, double.NaN, 0.5, double.NaN)]
    // 確率 NaN。
    [InlineData(2.0, 3.0, double.NaN, double.NaN)]
    public void InverseRegularizedIncompleteBetaComplement_SpecialValues(double a, double b, double probability, double expected)
    {
        double actual = XMath.Special.InverseRegularizedIncompleteBetaComplement(a, b, probability);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // 区間[0,1]の直外を順・逆・補関数の全APIで拒否する。
    [Fact]
    public void IncompleteBeta_ImmediatelyOutsideDomainIsNaN()
    {
        foreach (double value in new[] { -double.Epsilon, Math.BitIncrement(1) })
        {
            Assert.True(double.IsNaN(XMath.Special.RegularizedIncompleteBeta(2, 3, value)));
            Assert.True(double.IsNaN(XMath.Special.RegularizedIncompleteBetaComplement(2, 3, value)));
            Assert.True(double.IsNaN(XMath.Special.IncompleteBeta(2, 3, value)));
            Assert.True(double.IsNaN(XMath.Special.InverseRegularizedIncompleteBeta(2, 3, value)));
            Assert.True(double.IsNaN(XMath.Special.InverseRegularizedIncompleteBetaComplement(2, 3, value)));
        }
    }

    private const double RelativeTolerance = 2e-12;

    // R06: I_(1/2)(a,a)=1/2。補関数も個別に検査する。https://dlmf.nist.gov/8.17
    [Trait("Category", "ReviewRegression")]
    [Theory]
    [InlineData(2.0, false)]
    [InlineData(2.0, true)]
    [InlineData(1e160, false)]
    [InlineData(1e160, true)]
    [InlineData(1e308, false)]
    [InlineData(1e308, true)]
    public void IncompleteBeta_SymmetricMidpoint_IsOneHalf(double a, bool complement)
    {
        double value = complement ? XMath.Special.RegularizedIncompleteBetaComplement(a, a, 0.5)
            : XMath.Special.RegularizedIncompleteBeta(a, a, 0.5);
        SpecialFunctionAssert.Close(0.5, value);
    }

    // R07: I_x(1/2,1)=sqrt(x)なので逆関数はp^2。丸め後の0も検査する。
    [Trait("Category", "ReviewRegression")]
    [Theory]
    [InlineData(1e-150)]
    [InlineData(1e-200)]
    [InlineData(1e-300)]
    public void InverseIncompleteBeta_UnitSecondParameter_MatchesSquaredProbability(double p)
    {
        double expected = p * p;
        double actual = XMath.Special.InverseRegularizedIncompleteBeta(0.5, 1, p);
        if (expected == 0)
        {
            Assert.Equal(0, actual);
        }
        else
        {
            SpecialFunctionAssert.Close(expected, actual);
        }
    }

    // 初期値の桁違いと、根に到達した直後の中点戻り。普通の確率と補関数も含む。
    [Trait("Category", "AuditRegression")]
    [Theory]
    [InlineData(false, 2, 3, 1e-200, 4.08248290463863e-101)]
    [InlineData(false, 2, 3, 0.18080000000000002, 0.2)]
    [InlineData(true, 2, 3, 0.08370000000000004, 0.7)]
    [InlineData(false, 3, 20, 1.5400000000000002e-297, 1e-100)]
    public void InverseBeta_TailAndInterior_MatchesReference(bool complement, double a, double b,
        double probability, double expected)
    {
        double actual = complement
            ? XMath.Special.InverseRegularizedIncompleteBetaComplement(a, b, probability)
            : XMath.Special.InverseRegularizedIncompleteBeta(a, b, probability);
        SpecialFunctionAssert.Close(expected, actual, RelativeTolerance);
    }

    // A1: 丸めた中心を引く前に、入力xと厳密なa/(a+b)の差を保持する必要がある。
    // 最大値の0は、整数パラメータの二項確率のChernoff上界からも確認。
    [Trait("Category", "AuditRegression")]
    [Theory]
    [InlineData(2e30, 1e30, 0.6666666666666665, 0.293256526968985)]
    [InlineData(2e30, 1e30, 0.6666666666666666, 0.4459209192756595)]
    [InlineData(2e30, 1e30, 0.6666666666666667, 0.6071689951452612)]
    [InlineData(1e308, 5e307, 0.6666666666666666, 0)]
    public void IncompleteBeta_AsymmetricRoundedCenter_MatchesReference(double a, double b, double x,
        double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.RegularizedIncompleteBeta(a, b, x), RelativeTolerance);
        SpecialFunctionAssert.Close(1 - expected, XMath.Special.RegularizedIncompleteBetaComplement(a, b, x),
            RelativeTolerance);
    }

    // A2: I_x(a,1)=x^a、I_x(1,b)=1-(1-x)^b。真の小さい方は0へ丸められる。
    [Trait("Category", "AuditRegression")]
    [Theory]
    [InlineData(1e308, 1, 0)]
    [InlineData(1, 1e308, 1)]
    public void IncompleteBeta_OneHugeParameter_HasCorrectRoundedLimit(double a, double b, double expected)
    {
        Assert.Equal(expected, XMath.Special.RegularizedIncompleteBeta(a, b, 0.5));
        Assert.Equal(1 - expected, XMath.Special.RegularizedIncompleteBetaComplement(a, b, 0.5));
    }

    // By symmetry, I_(1/2)(a, a) = 1/2 even for subnormal shapes (DLMF 8.17.E4).
    [Trait("Category", "AuditRegression")]
    [Theory]
    [InlineData(double.Epsilon)]
    [InlineData(1e-310)]
    [InlineData(1e-300)]
    public void SymmetricSubnormalShapes_PreserveHalfProbability(double a)
    {
        SpecialFunctionAssert.Close(0.5, XMath.Special.RegularizedIncompleteBeta(a, a, 0.5), RelativeTolerance);
        SpecialFunctionAssert.Close(0.5, XMath.Special.RegularizedIncompleteBetaComplement(a, a, 0.5), RelativeTolerance);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0.1, 0.5)]
    [InlineData(1, 1)]
    [InlineData(2, 20)]
    [InlineData(10000, 20000)]
    public void ProbabilitiesAreBoundedAndMonotone(double a, double b)
    {
        double previousP = 0, previousQ = 1;
        // A logit grid resolves neighborhoods of both endpoints.
        for (int i = -60; i <= 60; i++)
        {
            double x = 1 / (1 + Math.Exp(-i * 0.5));
            double p = XMath.Special.RegularizedIncompleteBeta(a, b, x);
            double q = XMath.Special.RegularizedIncompleteBetaComplement(a, b, x);
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
    [InlineData(0.06826292824563261, 0.08730731210457515, 0.23809463925359542, 0.5214660986492881)]
    [InlineData(0.0001757684640244297, 69.8397682651148, 0.22263734239249633, 0.9999999999997519)]
    [InlineData(0.03732232331842256, 0.00938897367380737, 0.7492261524737802, 0.2088339101107264)]
    [InlineData(1.1934273169091016, 0.0002674067759876832, 0.1951493695996659, 3.579735771597746e-05)]
    [InlineData(0.10134291559458954, 0.00013437691686024475, 0.15256774143232144, 0.001111232663432416)]
    [InlineData(54.280891055516896, 66.63821536384923, 0.4513691605811504, 0.5241763547837013)]
    // 同じ分岐をパラメータ交換後の入力で確認
    [InlineData(0.01, 1e-10, 0.7, 1.0083486045267544e-08)]
    [InlineData(0.5, 1e-10, 0.7, 2.4198702422320155e-10)]
    [InlineData(1, 1e-10, 0.7, 1.2039728042534583e-10)]
    [InlineData(1.0000000000000002, 1e-10, 0.7, 1.203972804253458e-10)]
    [InlineData(9.999999999999998, 1e-10, 0.999999999995, 2.319261484001295e-09)]
    [InlineData(10, 1e-10, 0.999999999995, 2.319261484001295e-09)]
    [InlineData(10.000000000000002, 1e-10, 0.999999999995, 2.319261484001295e-09)]
    [InlineData(1000000, 1e-10, 1, 1.0)]
    [InlineData(0.01, 1e-20, 0.7, 1.008348614610192e-18)]
    [InlineData(0.5, 1e-20, 0.7, 2.4198702426718915e-20)]
    [InlineData(1, 1e-20, 0.7, 1.2039728043259358e-20)]
    [InlineData(1.0000000000000002, 1e-20, 0.7, 1.2039728043259355e-20)]
    [InlineData(9.999999999999998, 1e-20, 1, 1.0)]
    [InlineData(10, 1e-20, 1, 1.0)]
    [InlineData(10.000000000000002, 1e-20, 1, 1.0)]
    [InlineData(1000000, 1e-20, 1, 1.0)]
    [InlineData(1, 0.08751071060680152, 0.7, 0.09999999999999995)]
    [InlineData(1, 0.08751071060680153, 0.7, 0.09999999999999996)]
    [InlineData(1, 0.08751071060680155, 0.7, 0.09999999999999998)]
    [InlineData(1, 0.5757166424934448, 0.7, 0.49999999999999983)]
    [InlineData(1, 0.5757166424934449, 0.7, 0.4999999999999999)]
    [InlineData(1, 0.575716642493445, 0.7, 0.49999999999999994)]
    [InlineData(0.5, 0.24999999999999997, 0.7, 0.41598165027659967)]
    [InlineData(0.5, 0.25, 0.7, 0.41598165027659967)]
    [InlineData(0.5, 0.25000000000000006, 0.7, 0.4159816502765997)]
    [InlineData(10, 0.009999999999999998, 0.97, 0.009331218705722634)]
    [InlineData(10, 0.01, 0.97, 0.009331218705722635)]
    [InlineData(10, 0.010000000000000002, 0.97, 0.009331218705722637)]
    [InlineData(10, 0.01, 0.9, 0.002207487088514673)]
    [InlineData(7, 0.5, 0.924, 0.3013812159220373)]
    [InlineData(7, 0.5, 0.9239999999999999, 0.3013812159220369)]
    // 中心の漸近展開への切替近傍
    [InlineData(599.9999999999999, 299.99999999999994, 0.6509619050856152, 0.15876674457324838)]
    [InlineData(599.9999999999999, 299.99999999999994, 0.6666666666666667, 0.49686508226412474)]
    [InlineData(599.9999999999999, 299.99999999999994, 0.6823714282477182, 0.8412323191123845)]
    [InlineData(600, 300, 0.6509619050856151, 0.15876674457324666)]
    [InlineData(600, 300, 0.6666666666666666, 0.4968650822641219)]
    [InlineData(600, 300, 0.6823714282477181, 0.8412323191123827)]
    [InlineData(600.0000000000001, 300.00000000000006, 0.6509619050856151, 0.15876674457324666)]
    [InlineData(600.0000000000001, 300.00000000000006, 0.6666666666666666, 0.4968650822641219)]
    [InlineData(600.0000000000001, 300.00000000000006, 0.6823714282477181, 0.8412323191123827)]
    [InlineData(300, 300, 0.5499999999999999, 0.9929484138910368)]
    [InlineData(300, 300, 0.55, 0.9929484138910369)]
    [InlineData(300, 300, 0.5500000000000002, 0.992948413891037)]
    [InlineData(900, 300, 0.7874999999999999, 0.9990352484028682)]
    [InlineData(900, 300, 0.7875, 0.9990352484028682)]
    [InlineData(900, 300, 0.7875000000000001, 0.9990352484028683)]
    [InlineData(1000, 1000, 0.5473276378744922, 0.9999889286334245)]
    [InlineData(1000, 1000, 0.5473276378744923, 0.9999889286334245)]
    [InlineData(1000, 1000, 0.5473276378744925, 0.9999889286334245)]
    public void RegularizedIncompleteBeta_ExploredRelativeValues(double a, double b, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.RegularizedIncompleteBeta(a, b, x), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(0.06826292824563261, 0.08730731210457515, 0.23809463925359542, 0.478533901350712)]
    [InlineData(0.0001757684640244297, 69.8397682651148, 0.22263734239249633, 2.4811108999709793e-13)]
    [InlineData(0.03732232331842256, 0.00938897367380737, 0.7492261524737802, 0.7911660898892736)]
    [InlineData(1.1934273169091016, 0.0002674067759876832, 0.1951493695996659, 0.999964202642284)]
    [InlineData(0.10134291559458954, 0.00013437691686024475, 0.15256774143232144, 0.9988887673365676)]
    [InlineData(54.280891055516896, 66.63821536384923, 0.4513691605811504, 0.4758236452162987)]
    // 小さい形状パラメータの補確率
    [InlineData(0.00001, 1, 0.3, 1.2039655566024552e-05)]
    [InlineData(1e-10, 1, 0.3, 1.2039728042534586e-10)]
    [InlineData(1e-20, 1, 0.3, 1.203972804325936e-20)]
    // 小さい形状パラメータの分岐近傍
    [InlineData(1e-10, 0.01, 0.3, 1.0083486045267544e-08)]
    [InlineData(1e-10, 0.5, 0.3, 2.419870242232016e-10)]
    [InlineData(1e-10, 1.0000000000000002, 0.3, 1.2039728042534583e-10)]
    [InlineData(1e-10, 9.999999999999998, 5.000000000000001e-12, 2.3192614922753314e-09)]
    [InlineData(1e-10, 10, 5.0000000000000005e-12, 2.3192614922753314e-09)]
    [InlineData(1e-10, 10.000000000000002, 5e-12, 2.3192614922753314e-09)]
    [InlineData(1e-10, 1000000, 5e-17, 2.3141782918954095e-09)]
    [InlineData(1e-20, 0.01, 0.3, 1.008348614610192e-18)]
    [InlineData(1e-20, 0.5, 0.3, 2.4198702426718918e-20)]
    [InlineData(1e-20, 1.0000000000000002, 0.3, 1.2039728043259357e-20)]
    [InlineData(1e-20, 9.999999999999998, 5.0000000000000005e-22, 4.621846587946665e-19)]
    [InlineData(1e-20, 10, 5e-22, 4.621846587946665e-19)]
    [InlineData(1e-20, 10.000000000000002, 4.999999999999999e-22, 4.621846587946665e-19)]
    [InlineData(1e-20, 1000000, 4.9999999999999995e-27, 4.616763387553941e-19)]
    [InlineData(0.08751071060680152, 1, 0.3, 0.09999999999999996)]
    [InlineData(0.08751071060680153, 1, 0.3, 0.09999999999999998)]
    [InlineData(0.08751071060680155, 1, 0.3, 0.09999999999999999)]
    [InlineData(0.5757166424934448, 1, 0.3, 0.4999999999999999)]
    [InlineData(0.5757166424934449, 1, 0.3, 0.49999999999999994)]
    [InlineData(0.575716642493445, 1, 0.3, 0.5)]
    [InlineData(0.24999999999999997, 0.5, 0.3, 0.41598165027659967)]
    [InlineData(0.25, 0.5, 0.3, 0.4159816502765997)]
    [InlineData(0.25000000000000006, 0.5, 0.3, 0.4159816502765998)]
    [InlineData(0.009999999999999998, 10, 0.03, 0.00933121870572264)]
    [InlineData(0.01, 10, 0.03, 0.009331218705722642)]
    [InlineData(0.010000000000000002, 10, 0.03, 0.009331218705722644)]
    [InlineData(0.01, 10, 0.09999999999999999, 0.0022074870885146728)]
    [InlineData(0.01, 10, 0.1, 0.0022074870885146723)]
    [InlineData(0.01, 10, 0.10000000000000002, 0.002207487088514672)]
    [InlineData(0.5, 7, 0.07599999999999998, 0.3013812159220372)]
    [InlineData(0.5, 7, 0.076, 0.30138121592203715)]
    [InlineData(0.5, 7, 0.07600000000000001, 0.3013812159220371)]
    // 中心の漸近展開への切替近傍
    [InlineData(599.9999999999999, 299.99999999999994, 0.6509619050856152, 0.8412332554267516)]
    [InlineData(599.9999999999999, 299.99999999999994, 0.6666666666666667, 0.5031349177358753)]
    [InlineData(599.9999999999999, 299.99999999999994, 0.6823714282477182, 0.15876768088761553)]
    [InlineData(600, 300, 0.6509619050856151, 0.8412332554267533)]
    [InlineData(600, 300, 0.6666666666666666, 0.503134917735878)]
    [InlineData(600, 300, 0.6823714282477181, 0.15876768088761725)]
    [InlineData(600.0000000000001, 300.00000000000006, 0.6509619050856151, 0.8412332554267533)]
    [InlineData(600.0000000000001, 300.00000000000006, 0.6666666666666666, 0.503134917735878)]
    [InlineData(600.0000000000001, 300.00000000000006, 0.6823714282477181, 0.15876768088761722)]
    [InlineData(300, 300, 0.5499999999999999, 0.007051586108963213)]
    [InlineData(300, 300, 0.55, 0.007051586108963106)]
    [InlineData(300, 300, 0.5500000000000002, 0.007051586108962999)]
    [InlineData(900, 300, 0.7874999999999999, 0.0009647515971318253)]
    [InlineData(900, 300, 0.7875, 0.0009647515971317941)]
    [InlineData(900, 300, 0.7875000000000001, 0.0009647515971317629)]
    [InlineData(1000, 1000, 0.5473276378744922, 1.1071366575490894e-05)]
    [InlineData(1000, 1000, 0.5473276378744923, 1.10713665754904e-05)]
    [InlineData(1000, 1000, 0.5473276378744925, 1.1071366575489908e-05)]
    public void RegularizedIncompleteBetaComplement_ExploredRelativeValues(double a, double b, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.RegularizedIncompleteBetaComplement(a, b, x), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(0.06826292824563261, 0.08730731210457515, 0.23809463925359542, 13.492447746970448)]
    [InlineData(0.0001757684640244297, 69.8397682651148, 0.22263734239249633, 5684.4886555949515)]
    [InlineData(20.925559859068503, 94.24404585449102, 0.4205476635455462, 1.1973088942169598e-24)]
    [InlineData(0.03732232331842256, 0.00938897367380737, 0.7492261524737802, 27.822365255395408)]
    [InlineData(0.10134291559458954, 0.00013437691686024475, 0.15256774143232144, 8.280312656967624)]
    [InlineData(54.280891055516896, 66.63821536384923, 0.4513691605811504, 1.8028816464347314e-37)]
    public void IncompleteBeta_ExploredRelativeValues(double a, double b, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.IncompleteBeta(a, b, x), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(15.514022552822228, 12.831336855837954, 1.0203877560407046e-35, 0.001879851826670889)]
    [InlineData(16.17569854275672, 10.256456943570505, 0.0000013718305405852216, 0.1943991111275482)]
    [InlineData(5.956998515395676, 2.472131912172618, 2.8943266654464105e-15, 0.0023298203544637993)]
    [InlineData(9.816529143299778, 1.3541260905690329, 3.080972222358962e-17, 0.018894129193959715)]
    [InlineData(8.68628534762183, 16.780980820235197, 0.018177836846738796, 0.16510637117504798)]
    [InlineData(6.514082424883701, 14.12669444119964, 3.022270899407371e-8, 0.013657599057099004)]
    // 極端な裾の逆関数
    [InlineData(2, 100000000000000000000.0, 1e-300, 1.4142135623730951e-170)]
    // 条件数の大きい逆関数と通常域の回帰点
    [InlineData(0.5, 7, 0.7, 0.07641840544305054)]
    // 前回の探索で最大相対誤差となった入力。
    [InlineData(0.01, 50, 0.3, 5.8930766585448114e-55)]
    [InlineData(10, 200, 1e-300, 2.2147557388903529e-32)]
    [InlineData(12.5, 40, 1e-300, 1.2011833015673963e-25)]
    public void InverseRegularizedIncompleteBeta_ExploredRelativeValues(double a, double b, double p, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.InverseRegularizedIncompleteBeta(a, b, p), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(15.514022552822228, 12.831336855837954, 1.0203877560407046e-35, 0.9995019075891413)]
    [InlineData(16.17569854275672, 10.256456943570505, 0.0000013718305405852216, 0.9339865388204726)]
    [InlineData(5.956998515395676, 2.472131912172618, 2.8943266654464105e-15, 0.9999996829555945)]
    [InlineData(9.816529143299778, 1.3541260905690329, 3.080972222358962e-17, 0.9999999999999264)]
    [InlineData(8.68628534762183, 16.780980820235197, 0.018177836846738796, 0.5449349180020708)]
    [InlineData(6.514082424883701, 14.12669444119964, 3.022270899407371e-8, 0.8467805631404101)]
    // 極端に大きい第2パラメータの逆補関数
    [InlineData(0.5, 100000000000000000000.0, 1e-30, 6.639994635250711e-19)]
    public void InverseRegularizedIncompleteBetaComplement_ExploredRelativeValues(double a, double b, double q, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.InverseRegularizedIncompleteBetaComplement(a, b, q), relativeTolerance: 2e-12);
    }
    // 正則化した値が非正規数または 0 でも、非正則化した積分は通常の有限値となる。
    [Trait("Category", "AuditRegression")]
    [Trait("Category", "CoreAuditIdentities20261102")]
    [Theory]
    [InlineData(1.0, 1e-300, 0.5, 0.6931471805599453)]
    [InlineData(1.0, 1e-310, 0.5, 0.6931471805599453)]
    [InlineData(1.0, 1e-320, 0.5, 0.6931471805599453)]
    [InlineData(1.0, 5e-324, 0.5, 0.6931471805599453)]
    [InlineData(2.0, 1e-300, 0.5, 0.19314718055994531)]
    [InlineData(2.0, 1e-310, 0.5, 0.19314718055994531)]
    [InlineData(2.0, 1e-320, 0.5, 0.19314718055994531)]
    [InlineData(2.0, 5e-324, 0.5, 0.19314718055994531)]
    [InlineData(5.0, 1e-300, 0.5, 0.010855513893278643)]
    [InlineData(5.0, 1e-310, 0.5, 0.010855513893278643)]
    [InlineData(5.0, 1e-320, 0.5, 0.010855513893278643)]
    [InlineData(5.0, 5e-324, 0.5, 0.010855513893278643)]
    public void IncompleteBeta_PreservesFiniteValuesWhenNormalizationLosesRange(double a, double b, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.IncompleteBeta(a, b, x), relativeTolerance: 2e-12);
    }

    // 正則化値が非正規数・0 に丸まっても、B(a, b) を掛けた値は通常の数になる（参照は mpmath 120桁、同じ binary64 入力）
    [Theory]
    [InlineData(1000.0, 5e-324, 0.5, 1.8646662852252765e-304)]
    [InlineData(975.0, 5e-324, 0.5, 6.417048396036065e-297)]
    [InlineData(1000.0, 1e-310, 0.5, 1.8646662852252765e-304)]
    [InlineData(1000.0, 1e-5, 0.5, 1.86465337897916e-304)]
    [InlineData(1100.0, 0.5, 0.5, 0.0)]
    public void IncompleteBeta_RecoversValueWhenRegularizedValueUnderflows(double a, double b, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.IncompleteBeta(a, b, x),
            relativeTolerance: 2e-12, absoluteTolerance: 2 * double.Epsilon);
    }

    // 非正規数も相対2e-12と絶対2*double.Epsilonの大きい方を許容する。
    // 参照はmpmath 140桁、同じbinary64入力。一律の絶対2 ulpや正しい丸めは要求しない。
    [Theory]
    [InlineData(975.0, 0.5, 0.48539901939551267, 1.263201808253116e-309)]
    [InlineData(1020.0, 5e-324, 0.5, 1.7434500411435e-310)]
    [InlineData(1020.0, 1e-310, 0.5, 1.7434500411435e-310)]
    [InlineData(9.942266394822983, 1.6821784523232741, 1.4095132671818168e-31, 1.8807962280437834e-308)]
    [InlineData(50.818174447779626, 22.070972194462534, 9.510960596883324e-07, 1.8981415057696396e-308)]
    [InlineData(44.47438865503035, 2.4598218813351456, 1.3091903833542776e-07, 1.717108106301804e-308)]
    // min(a, b) が一様漸近展開の下限 300 以上でも、中心から遠い裾は連分数の経路になる
    [InlineData(300.0, 300.0, 0.10700129376955325, 5.000000000000034e-309)]
    [InlineData(300.00000000000006, 301.0, 0.10704695542972707, 5.00000000000007e-309)]
    [InlineData(299.99999999999994, 301.0, 0.10704695542972698, 5.000000000000143e-309)]
    [InlineData(300.00000000000006, 300.0, 0.10700129376955331, 5.000000000000084e-309)]
    // 完全ベータ関数の値が非正規数になる中心・端点。参照は120/200桁で丸めが一致し、x=1と0.5は有理数からも確認。
    [InlineData(510.0, 510.0, 0.5, 6.98716044090679e-309)]
    [InlineData(509.0, 509.0, 0.49, 1.4644215179530206e-308)]
    [InlineData(510.0, 510.0, 1.0, 1.3974320881813577e-308)]
    public void IncompleteBeta_SubnormalResultsMatchReference(double a, double b, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.IncompleteBeta(a, b, x),
            relativeTolerance: 2e-12, absoluteTolerance: 2 * double.Epsilon);
    }
}
