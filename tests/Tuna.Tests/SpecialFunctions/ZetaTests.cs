using Tuna;

namespace Tuna.Tests;

// 有限参照値は mpmath 1.4.1、100 桁以上で生成し binary64 に丸めた。
// 入力も binary64 の値を正確に高精度へ変換（十進文字列の値とは区別）。
// 参照値生成: https://mpmath.org/doc/current/functions/index.html
[Trait("SpecialFunction", "Zeta")]
public class ZetaTests
{
    // 1-sの丸め補正、Gamma倍数公式への切替両隣と内部。
    // mpmath 1.4.1、正確なbinary64入力、160/240桁の丸め一致。
    // この大きさではzeta(s)-1も同じbinary64へ丸められる。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(-63.45641986239672, 7.09988402174031e36)]
    [InlineData(-169.99999999999997, -2.102928822341658e157)]
    [InlineData(-170.00000000000003, 2.1029288223420528e157)]
    [InlineData(-171, 1.2819489863482243e172)]
    [InlineData(-200.5, -2.320000663352899e215)]
    // 反射公式の経路で 1 - s が丸められる帯（-256 < s ≤ -255）において Γ の ψ 補正が効いていることを確認する（3e-15 は補正検出用）。
    [InlineData(-255.3265589882968, 9.076071717675488e300)]
    public void Zeta_NegativeReflectionRegression_MatchesReference(double s, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.Zeta(s), relativeTolerance: 3e-15);
        SpecialFunctionAssert.Close(expected, XMath.Special.ZetaMinusOne(s), relativeTolerance: 3e-15);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(-63.45641986239672, -1.7970784908531994e56)]
    [InlineData(-183.57, -5.151764608361216e245)]
    [InlineData(-200.5, 1.0544651608041441e276)]
    public void DirichletEta_NegativeReflectionRegression_MatchesReference(double s, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.DirichletEta(s), relativeTolerance: 3e-15);
    }

    // https://dlmf.nist.gov/25.2; https://dlmf.nist.gov/25.4; https://dlmf.nist.gov/25.11
    [Theory]
    // 解析接続の既知値-1/2。
    [InlineData(0.0, -0.5)]
    // 負の奇整数の既知値。
    [InlineData(-1.0, -0.08333333333333333)]
    // 負の偶整数の自明な零点。
    [InlineData(-2.0, 0.0)]
    // 関数等式の非整数。
    [InlineData(-2.5, 0.008516928777850331)]
    // 唯一の実数極の左隣。
    [InlineData(0.9999999999999999, -9007199254740991.0)]
    // 極の右隣。
    [InlineData(1.0000000000000002, 4503599627370497.0)]
    // Laurent級数とEuler-Maclaurinの切替直前。
    [InlineData(1.9999999999999998, 1.6449340668482266)]
    // 切替直後。
    [InlineData(2.0000000000000004, 1.644934066848226)]
    // 解析接続の非整数。
    [InlineData(0.5, -1.4603545088095868)]
    // 整数定数表の末尾。
    [InlineData(40.0, 1.0000000000009095)]
    // 整数表の直後。
    [InlineData(41.0, 1.0000000000004547)]
    // 負の大引数で対数Gammaを使う枝。
    [InlineData(-171.5, 4.739302330550545e+172)]
    public void Zeta_MatchesReference(double s, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.Zeta(s), 2e-12, 0);
    }

    // https://dlmf.nist.gov/25.2; https://dlmf.nist.gov/25.4; https://dlmf.nist.gov/25.11
    [Theory]
    // s<1 の解析接続。
    [InlineData(0.5, -2.4603545088095866)]
    // 極の直右。
    [InlineData(1.0000000000000002, 4503599627370495.5)]
    // ζ(2)-1。
    [InlineData(2.0, 0.6449340668482264)]
    // 直接和へ切替点。
    [InlineData(64.0, 5.421010862456646e-20)]
    // 直接和の開始。
    [InlineData(64.00000000000001, 5.421010862456592e-20)]
    // ζ自体が1に丸められても差は非零。
    [InlineData(100.0, 7.888609052210118e-31)]
    // 極小の非零値。
    [InlineData(1000.0, 9.332636185032189e-302)]
    public void ZetaMinusOne_MatchesReference(double s, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.ZetaMinusOne(s), 5e-13, 0);
    }

    // https://dlmf.nist.gov/25.2; https://dlmf.nist.gov/25.4; https://dlmf.nist.gov/25.11
    [Theory]
    // 既知値1/2。
    [InlineData(0.0, 0.5)]
    // 解析接続の既知値1/4。
    [InlineData(-1.0, 0.25)]
    // ζの極を打ち消す左側。
    [InlineData(0.9999999999999999, 0.6931471805599453)]
    // 可除特異点の値ln2。
    [InlineData(1.0, 0.6931471805599453)]
    // ζの極を打ち消す右側。
    [InlineData(1.0000000000000002, 0.6931471805599454)]
    // 非整数値。
    [InlineData(0.5, 0.6048986434216304)]
    // 大引数近似の開始領域。
    [InlineData(65.0, 1.0)]
    public void DirichletEta_MatchesReference(double s, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.DirichletEta(s), 5e-13, 0);
    }

    // https://dlmf.nist.gov/25.2; https://dlmf.nist.gov/25.4; https://dlmf.nist.gov/25.11
    [Theory]
    // s=0 は1/2-a。
    [InlineData(0.0, 2.5, -2.0)]
    // 0<s<1 の解析接続。
    [InlineData(0.5, 0.5, -0.6048986434216304)]
    // 半整数の既知値。
    [InlineData(2.0, 0.5, 4.934802200544679)]
    // Riemannへの委譲。
    [InlineData(2.0, 1.0, 1.6449340668482264)]
    // ζ-1への委譲で差を保持。
    [InlineData(100.0, 2.0, 7.888609052210118e-31)]
    // 高次数直接和の切替。
    [InlineData(30.0, 3.25, 4.401880695798085e-16)]
    // 直接和を試す領域。
    [InlineData(31.0, 3.25, 1.3543226631079842e-16)]
    // s=1極の直右。
    [InlineData(1.0000000000000002, 10.0, 4503599627370493.5)]
    // 大きい第2引数。
    [InlineData(2.0, 10000000000.0, 1.00000000005e-10)]
    public void HurwitzZeta_MatchesReference(double s, double a, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.HurwitzZeta(s, a), 5e-13, 0);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/25.2; https://dlmf.nist.gov/25.4; https://dlmf.nist.gov/25.11
    [Theory]
    // 極または可除特異点。
    [InlineData(1.0, double.PositiveInfinity)]
    // 正の無限大。
    [InlineData(double.PositiveInfinity, 1.0)]
    // 負の無限大。
    [InlineData(double.NegativeInfinity, double.NaN)]
    // NaN。
    [InlineData(double.NaN, double.NaN)]
    public void Zeta_SpecialValues(double s, double expected)
    {
        double actual = XMath.Special.Zeta(s);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/25.2; https://dlmf.nist.gov/25.4; https://dlmf.nist.gov/25.11
    [Theory]
    // 極または可除特異点。
    [InlineData(1.0, double.PositiveInfinity)]
    // 正の無限大。
    [InlineData(double.PositiveInfinity, 0.0)]
    // 負の無限大。
    [InlineData(double.NegativeInfinity, double.NaN)]
    // NaN。
    [InlineData(double.NaN, double.NaN)]
    public void ZetaMinusOne_SpecialValues(double s, double expected)
    {
        double actual = XMath.Special.ZetaMinusOne(s);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/25.2; https://dlmf.nist.gov/25.4; https://dlmf.nist.gov/25.11
    [Theory]
    // 極または可除特異点。
    [InlineData(1.0, 0.6931471805599453)]
    // 正の無限大。
    [InlineData(double.PositiveInfinity, 1.0)]
    // 負の無限大。
    [InlineData(double.NegativeInfinity, double.NaN)]
    // NaN。
    [InlineData(double.NaN, double.NaN)]
    public void DirichletEta_SpecialValues(double s, double expected)
    {
        double actual = XMath.Special.DirichletEta(s);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/25.2; https://dlmf.nist.gov/25.4; https://dlmf.nist.gov/25.11
    [Theory]
    // 未対応 s<0 を明示。
    [InlineData(-1.0, 1.0, double.NaN)]
    // a=0 の定義域境界。
    [InlineData(2.0, 0.0, double.NaN)]
    // a<0。
    [InlineData(2.0, -0.5, double.NaN)]
    // s=1極。
    [InlineData(1.0, 2.0, double.PositiveInfinity)]
    // a<1 の極限。
    [InlineData(double.PositiveInfinity, 0.5, double.PositiveInfinity)]
    // a=1 の極限。
    [InlineData(double.PositiveInfinity, 1.0, 1.0)]
    // a>1 の極限。
    [InlineData(double.PositiveInfinity, 2.0, 0.0)]
    // a→∞で減衰。
    [InlineData(2.0, double.PositiveInfinity, 0.0)]
    // 解析接続のa→∞。
    [InlineData(0.5, double.PositiveInfinity, double.NegativeInfinity)]
    // s NaN。
    [InlineData(double.NaN, 1.0, double.NaN)]
    // a NaN。
    [InlineData(2.0, double.NaN, double.NaN)]
    public void HurwitzZeta_SpecialValues(double s, double a, double expected)
    {
        double actual = XMath.Special.HurwitzZeta(s, a);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // R15: etaの自明零点。因子のoverflowとzetaの0の積でNaNにしない。
    // https://dlmf.nist.gov/25.2; https://dlmf.nist.gov/25.4
    [Trait("Category", "ReviewRegression")]
    [Theory]
    [InlineData(-100.0)]
    [InlineData(-2000.0)]
    public void DirichletEta_NegativeEvenInteger_IsZero(double s)
    {
        Assert.Equal(0, XMath.Special.DirichletEta(s));
    }

    // 探索で得た補強点。mpmath 1.4.1で独立に計算し、複数精度でbinary64丸めを確認。
    // 入力・期待値・許容誤差をここに固定する。出典と選定方針は README.md を参照。

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(22.530020965824576, 1.0000001651333448)]
    [InlineData(-81.91975597716194, -5.480136857111086e+55)]
    [InlineData(-16.1165465146253, -0.230059357326459)]
    [InlineData(-47.10565624194395, 3.075582346440579e+21)]
    [InlineData(24.60475396534774, 1.0000000391968553)]
    [InlineData(-83.45601079413424, 1.739432401237902e+58)]
    public void Zeta_ExploredRelativeValues(double s, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.Zeta(s), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(22.530020965824576, 0.999999834902258)]
    [InlineData(-81.91975597716194, 5.013317231320659e+80)]
    [InlineData(-16.1165465146253, 32691.19296061963)]
    [InlineData(-47.10565624194395, -9.314785304799167e+35)]
    [InlineData(24.60475396534774, 0.9999999608067888)]
    // 前回の探索で最大相対誤差となった入力。
    [InlineData(-81.85354779044653, 7.324727611771805e+80)]
    [InlineData(-83.45601079413424, -4.615301943081661e+83)]
    public void DirichletEta_ExploredRelativeValues(double s, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.DirichletEta(s), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(119.59672097233785, 9.9494621515536e-37)]
    [InlineData(181.15997673316687, 2.9202026418689824e-55)]
    [InlineData(58.769775676345375, 2.0348606944554213e-18)]
    [InlineData(110.3598455680161, 6.003108145178272e-34)]
    [InlineData(5.136395223700564, 0.03324222659660321)]
    [InlineData(195.17648707075455, 1.7620691324510422e-59)]
    public void ZetaMinusOne_ExploredRelativeValues(double s, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.ZetaMinusOne(s), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(34.03268967055598, 598.3117386091092, 5.802400373431173e-94)]
    [InlineData(24.512644401739433, 7094.481623499802, 1.2136395662317968e-92)]
    [InlineData(22.21197509099312, 0.00015606946201718414, 3.5803645147426436e+84)]
    [InlineData(29.85000593900812, 0.000016528555472456726, 5.441475064519606e+142)]
    [InlineData(10.463430850175655, 6.029058461868533, 8.726410369766122e-09)]
    [InlineData(57.51675733613475, 1.5338828629503565, 2.0598489893785343e-11)]
    public void HurwitzZeta_ExploredRelativeValues(double s, double a, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.HurwitzZeta(s, a), relativeTolerance: 2e-12);
    }
}
