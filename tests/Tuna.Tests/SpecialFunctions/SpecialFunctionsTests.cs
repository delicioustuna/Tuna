using Tuna;

namespace Tuna.Tests;

// 有限参照値は mpmath 1.4.1、100 桁以上で生成し binary64 に丸めた。
// 入力も binary64 の値を正確に高精度へ変換（十進文字列の値とは区別）。
// 参照値生成: https://mpmath.org/doc/current/functions/index.html
[Trait("SpecialFunction", "SpecialFunctions")]
public class SpecialFunctionsTests
{

    // 公開される定数の丸め値を確認。https://dlmf.nist.gov/5.2.E3
    [Fact]
    public void EulerGamma_IsCorrectlyRounded()
    {
        Assert.Equal(0.5772156649015329, XMath.Special.EulerGamma);
    }
}
