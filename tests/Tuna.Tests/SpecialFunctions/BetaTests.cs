using Tuna;

namespace Tuna.Tests;

// 有限参照値は mpmath 1.4.1、100 桁以上で生成し binary64 に丸めた。
// 入力も binary64 の値を正確に高精度へ変換（十進文字列の値とは区別）。
// 参照値生成: https://mpmath.org/doc/current/functions/index.html
[Trait("SpecialFunction", "Beta")]
public class BetaTests
{

    // https://dlmf.nist.gov/5.12; https://dlmf.nist.gov/5.11
    [Theory]
    // 半整数で B=π。
    [InlineData(0.5, 0.5, 3.141592653589793)]
    // 整数の既知値1/12。
    [InlineData(2.0, 3.0, 0.08333333333333333)]
    // Stirling の切替直前。
    [InlineData(9.999999999999998, 12.0, 2.835142154027607e-07)]
    // 双方Stirling展開の境界。
    [InlineData(10.0, 12.0, 2.835142154027603e-07)]
    // 直接 Gamma の積は溢れるが Beta は有限。
    [InlineData(170.0, 170.0, 1.21479019883015e-103)]
    // 大きく異なるパラメータで対数差の桁落ちを確認。
    [InlineData(10000000000.0, 0.5, 1.7724538509276716e-05)]
    // 解析接続の負の値。
    [InlineData(-0.5, 1.25, -2.6220575542921196)]
    // 解析接続の正の値。
    [InlineData(-1.5, 0.25, 2.1850479619101)]
    // 極めて小さいが非零の有限値。
    [InlineData(500.0, 500.0, 1.479901599125611e-302)]
    public void Beta_MatchesReference(double a, double b, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.Beta(a, b), 2e-12, 0);
    }

    // https://dlmf.nist.gov/5.12; https://dlmf.nist.gov/5.11
    [Theory]
    // 半整数で B=π。
    [InlineData(0.5, 0.5, 1.1447298858494002)]
    // 整数の既知値1/12。
    [InlineData(2.0, 3.0, -2.4849066497880004)]
    // Stirling の切替直前。
    [InlineData(9.999999999999998, 12.0, -15.076003572521552)]
    // 双方Stirling展開の境界。
    [InlineData(10.0, 12.0, -15.076003572521552)]
    // 直接 Gamma の積は溢れるが Beta は有限。
    [InlineData(170.0, 170.0, -236.97169319236434)]
    // 大きく異なるパラメータで対数差の桁落ちを確認。
    [InlineData(10000000000.0, 0.5, -10.940560522033028)]
    // 解析接続の負の値。
    [InlineData(-0.5, 1.25, 0.9639593356315369)]
    // 解析接続の正の値。
    [InlineData(-1.5, 0.25, 0.7816377788375822)]
    // 極めて小さいが非零の有限値。
    [InlineData(500.0, 500.0, -694.9887224857134)]
    // Beta が underflow するが LogBeta は有限。
    [InlineData(1000.0, 1000.0, -1388.4826016359023)]
    public void LogBeta_MatchesReference(double a, double b, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.LogBeta(a, b), 5e-13, 0);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/5.12; https://dlmf.nist.gov/5.11
    [Theory]
    // a+b が極でBetaは零点。
    [InlineData(-0.5, 0.5, 0.0)]
    // パラメータ自体が極。
    [InlineData(0.0, 2.0, double.NaN)]
    // 第2パラメータが極。
    [InlineData(2.0, -1.0, double.NaN)]
    // 正の無限大の極限。
    [InlineData(double.PositiveInfinity, 2.0, 0.0)]
    // 第1引数 NaN。
    [InlineData(double.NaN, 2.0, double.NaN)]
    // 第2引数 NaN。
    [InlineData(2.0, double.NaN, double.NaN)]
    public void Beta_SpecialValues(double a, double b, double expected)
    {
        double actual = XMath.Special.Beta(a, b);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/5.12; https://dlmf.nist.gov/5.11
    [Theory]
    // a+b が極でBetaは零点。
    [InlineData(-0.5, 0.5, double.NegativeInfinity)]
    // パラメータ自体が極。
    [InlineData(0.0, 2.0, double.NaN)]
    // 第2パラメータが極。
    [InlineData(2.0, -1.0, double.NaN)]
    // 正の無限大の極限。
    [InlineData(double.PositiveInfinity, 2.0, double.NegativeInfinity)]
    // 第1引数 NaN。
    [InlineData(double.NaN, 2.0, double.NaN)]
    // 第2引数 NaN。
    [InlineData(2.0, double.NaN, double.NaN)]
    public void LogBeta_SpecialValues(double a, double b, double expected)
    {
        double actual = XMath.Special.LogBeta(a, b);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // 引数の順番で枝や符号が変化しない。https://dlmf.nist.gov/5.12.E1
    [Theory]
    [InlineData(0.5, 2.0)]
    [InlineData(170.0, 12.0)]
    [InlineData(-0.5, 1.25)]
    public void Beta_IsSymmetric(double a, double b)
    {
        SpecialFunctionAssert.Close(XMath.Special.Beta(a, b), XMath.Special.Beta(b, a));
        SpecialFunctionAssert.Close(XMath.Special.LogBeta(a, b), XMath.Special.LogBeta(b, a));
    }

    // R01: B(a,1)=1/a。a+1 が a に丸められる場合も成立する。
    // https://dlmf.nist.gov/5.12
    [Trait("Category", "ReviewRegression")]
    [Theory]
    [InlineData(1e10)]
    [InlineData(1e16)]
    [InlineData(1e20)]
    public void Beta_WithUnitParameter_MatchesReciprocal(double a)
    {
        SpecialFunctionAssert.Close(1 / a, XMath.Special.Beta(a, 1));
    }

    [Trait("Category", "ReviewRegression")]
    [Fact]
    public void Beta_WithParameterTwo_PreservesSmallShift()
    {
        double a = 1e20;
        SpecialFunctionAssert.Close((1 / a) / (a + 1), XMath.Special.Beta(a, 2));
    }

    // R02: log B(a,a)=-2a log 2+O(log a)。補正は binary64 の1 ulpより小さい。
    // https://dlmf.nist.gov/5.11; https://dlmf.nist.gov/5.12
    [Trait("Category", "ReviewRegression")]
    [Fact]
    public void LogBeta_LargeEqualFiniteParameters_RemainsFinite()
    {
        SpecialFunctionAssert.Close(-2 * Math.Log(2) * 1e308, XMath.Special.LogBeta(1e308, 1e308));
    }

    // 探索で得た補強点。mpmath 1.4.1で独立に計算し、複数精度でbinary64丸めを確認。
    // 入力・期待値・許容誤差をここに固定する。出典と選定方針は README.md を参照。

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(0.2762424649805344, 0.000121900421621928, 8206.656217893795)]
    [InlineData(0.15206769102040615, 24.335836573676676, 3.783550480086214)]
    [InlineData(4611.511768958046, 2.396482848707028e-7, 4172772.780292099)]
    // 前回の探索で最大相対誤差となった入力。
    [InlineData(8.064290160283274, 0.04503086337367807, 19.781822634244346)]
    [InlineData(0.000019016849498304964, 2068.3469631335747, 52576.734969675555)]
    [InlineData(4.057476424837525e-8, 1521.8649461568698, 24645853.101011463)]
    [InlineData(17.369501225402125, 1.0305063483729555, 0.05184208305746484)]
    public void Beta_ExploredRelativeValues(double a, double b, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.Beta(a, b), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(0.2762424649805344, 0.000121900421621928, 9.012700837864056)]
    [InlineData(0.15206769102040615, 24.335836573676676, 1.330662849229738)]
    [InlineData(0.00004262328290821077, 0.00020106569048822462, 10.25537093271885)]
    [InlineData(4611.511768958046, 2.396482848707028e-7, 15.244091308125288)]
    [InlineData(0.000019016849498304964, 2068.3469631335747, 10.870028999925143)]
    [InlineData(4.057476424837525e-8, 1521.8649461568698, 17.020119213104227)]
    public void LogBeta_ExploredRelativeValues(double a, double b, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.LogBeta(a, b), relativeTolerance: 2e-12);
    }

    // Γ(large)/Γ(large + small) が非正規化数へアンダーフローする領域でも B(a, b) の精度を保つことを確認する（参照値は mpmath 60/120 桁）。
    [Trait("Category", "AuditRegression")]
    [Theory]
    [InlineData(291.4225659551792, 158.6317113926334, 3.5450689397250515e-128)]
    [InlineData(1053892.5018589264, 54.99290264035022, 1.3782602051667102e-260)]
    [InlineData(2973691.037143808, 48.560535263593515, 1.0164099180325543e-254)]
    [InlineData(200000, 60, 1.1922867952152679e-238)]
    [InlineData(5151.126848786688, 142.20002420932477, 1.0895321275476134e-285)]
    public void Beta_GammaRatioUnderflow_MatchesReference(double a, double b, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.Beta(a, b), 2e-12);
        SpecialFunctionAssert.Close(expected, XMath.Special.Beta(b, a), 2e-12);
    }

    // 正則化しない不完全ベータが、アンダーフローに近い B(a, b) でも桁落ちしないことを確認する。
    [Trait("Category", "AuditRegression")]
    [Fact]
    public void IncompleteBeta_WithUnderflowingGammaRatio_MatchesReference()
    {
        SpecialFunctionAssert.Close(1.190033693740643e-238, XMath.Special.IncompleteBeta(200000, 60, 0.9998), 2e-12);
    }

    // 一方が負の非整数で他方が大きい場合でも、a + b の丸めで負の引数の寄与が失われないことを確認する（参照値は mpmath）。
    [Trait("Category", "AuditRegression")]
    [Theory]
    [InlineData(-0.02553886106533554, 1.847369247220685e+249, -92346492.06380011)]
    [InlineData(-19.776128575732276, 20727176102104.32, 8.77084158455441e+245)]
    [InlineData(-150.3, 200.25, -4.471313619392425e+47)]
    public void Beta_NegativeAndLargeParameters_MatchesReference(double a, double b, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.Beta(a, b), 2e-12);
        SpecialFunctionAssert.Close(expected, XMath.Special.Beta(b, a), 2e-12);
    }

    [Trait("Category", "AuditRegression")]
    [Theory]
    [InlineData(-0.02553886106533554, 1.847369247220685e+249, 18.341058278655797)]
    [InlineData(-19.776128575732276, 20727176102104.32, 566.304780547072)]
    [InlineData(-5.920031184728952, 2.000129923737712e+87, 1186.1401911368794)]
    public void LogBeta_NegativeAndLargeParameters_MatchesReference(double a, double b, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.LogBeta(a, b), 2e-12);
        SpecialFunctionAssert.Close(expected, XMath.Special.LogBeta(b, a), 2e-12);
    }

    // B(a, b) ~ Γ(a) b⁻ᵃ は a < 0 で b → ∞ のとき発散し、符号は Γ(a) と一致する。
    [Trait("Category", "AuditRegression")]
    [Theory]
    [InlineData(-5.920031184728952, 2.000129923737712e+87, double.PositiveInfinity)]
    [InlineData(-0.5, double.PositiveInfinity, double.NegativeInfinity)]
    [InlineData(-1.5, double.PositiveInfinity, double.PositiveInfinity)]
    public void Beta_NegativeParameterWithDivergingPartner_Overflows(double a, double b, double expected)
    {
        Assert.Equal(expected, XMath.Special.Beta(a, b));
        Assert.Equal(expected, XMath.Special.Beta(b, a));
        if (double.IsPositiveInfinity(b))
        {
            Assert.Equal(double.PositiveInfinity, XMath.Special.LogBeta(a, b));
        }
    }

    // B(a, 1) = 1/a より ln|B(-ε, 1)| = -ln ε（非正規化数の引数でも精度を保つ）。
    [Trait("Category", "ReviewRegression")]
    [Fact]
    public void LogBeta_SubnormalArgument_KeepsAccuracy()
    {
        SpecialFunctionAssert.WithinUlps(-Math.Log(double.Epsilon), XMath.Special.LogBeta(-double.Epsilon, 1), 8);
    }

    // 多変数のベータ関数。参照値は mpmath 1.4.1 で Σ ln Γ(αᵢ) - ln Γ(Σ αᵢ) を 100 桁と 140 桁で求めて丸めが一致した値
    // （tools/DistributionReferences/generate_multivariate_references.py --special）。許容誤差は 64ε(1 + κ)。
    // κ は各成分についての相対条件数と、実装が足す連鎖の項 ln B(Sₖ₋₁, α₍ₖ₎) の大きさ Σ|項|/|ln B(α)| の和。
    // https://dlmf.nist.gov/5.12
    [Theory]
    [InlineData(new double[] { 2.0, 3.0 }, -2.4849066497880004, 5.1e-14)]
    [InlineData(new double[] { 0.5, 0.5, 0.5 }, 1.8378770664093456, 5.2e-14)]
    [InlineData(new double[] { 1.0, 1.0, 1.0, 1.0 }, -1.791759469228055, 8.7e-14)]
    [InlineData(new double[] { 3.0, 4.0, 5.0, 6.0, 7.0 }, -37.755025963184224, 4.4e-14)]
    [InlineData(new double[] { 7.0, 0.5, 2.0, 10000.0 }, -80.35065366813203, 4.3e-14)]
    [InlineData(new double[] { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 }, 22.527126517342058, 3.5e-14)]
    // 小さな成分（ln Γ の極の近く）。
    [InlineData(new double[] { 0.001, 2.0, 0.05 }, 9.853666146206795, 3.1e-14)]
    [InlineData(new double[] { 1e-300, 1e-300 }, 691.4686750787737, 2.8e-14)]
    // 大きな成分（ln Γ どうしの打ち消し）。
    [InlineData(new double[] { 1000000.0, 2000000.0, 3000000.0 }, -6068437.565877463, 4.3e-14)]
    [InlineData(new double[] { 1e+300, 1e+300, 1.0 }, -1.3862943611198907e+300, 4.3e-14)]
    public void LogBetaVector_MatchesReference(double[] alpha, double expected, double tolerance)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.LogBeta(alpha), tolerance);
    }

    [Theory]
    [InlineData(2.0, 3.0)]
    [InlineData(0.5, 1e10)]
    [InlineData(1e-300, 7.0)]
    public void LogBetaVector_TwoComponentsMatchTheScalarFunction(double a, double b)
    {
        Assert.Equal(XMath.Special.LogBeta(Math.Min(a, b), Math.Max(a, b)), XMath.Special.LogBeta(new[] { a, b }));
        Assert.Equal(XMath.Special.LogBeta(new[] { a, b }), XMath.Special.LogBeta(new[] { b, a }));
    }

    [Fact]
    public void LogBetaVector_IsSymmetricAndUsesTheHeapForManyComponents()
    {
        double[] alpha = [7.0, 0.5, 2.0, 1e4, 0.25];
        double expected = XMath.Special.LogBeta(alpha);
        Assert.Equal(expected, XMath.Special.LogBeta(new[] { 0.25, 1e4, 2.0, 7.0, 0.5 }));

        // 100 成分の 1/2: 50 ln π - ln Γ(50)
        double[] halves = Enumerable.Repeat(0.5, 100).ToArray();
        SpecialFunctionAssert.Close(50 * Math.Log(Math.PI) - XMath.Special.LogGamma(50), XMath.Special.LogBeta(halves), 1e-14);
    }

    [Theory]
    // 空、0以下・NaN の成分は NaN。1成分は0。+∞ の成分、和が溢れる場合は -∞。
    [InlineData(new double[] { }, double.NaN)]
    [InlineData(new double[] { 2.0, 0.0 }, double.NaN)]
    [InlineData(new double[] { 2.0, -0.5 }, double.NaN)]
    [InlineData(new double[] { 2.0, double.NaN, 3.0 }, double.NaN)]
    [InlineData(new double[] { 3.5 }, 0.0)]
    [InlineData(new double[] { double.PositiveInfinity }, 0.0)]
    [InlineData(new double[] { 2.0, double.PositiveInfinity }, double.NegativeInfinity)]
    [InlineData(new double[] { 1e308, 1e308, 1e308 }, double.NegativeInfinity)]
    // 部分和は有限で、連鎖の項の和が -1.98e308 になる。
    [InlineData(new double[] { 6e307, 6e307, 6e307 }, double.NegativeInfinity)]
    public void LogBetaVector_SpecialValues(double[] alpha, double expected)
    {
        Assert.Equal(expected, XMath.Special.LogBeta(alpha));
    }
}
