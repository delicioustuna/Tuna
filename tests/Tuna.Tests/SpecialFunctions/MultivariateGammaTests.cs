using Tuna;

namespace Tuna.Tests;

// 有限参照値は mpmath 1.4.1 で、定義 (p(p - 1)/4) ln π + Σⱼ ln Γ(a - j/2) を 100 桁と 140 桁で求めて binary64 への丸めが一致した値
// （tools/DistributionReferences/generate_multivariate_references.py --special）。入力は binary64 の値を正確に変換した。
// 許容誤差は 64ε(1 + κ)。κ は a についての相対条件数と、和の項の大きさ Σ|項|/|ln Γₚ(a)| の和。
[Trait("SpecialFunction", "MultivariateGamma")]
public class MultivariateGammaTests
{
    // https://dlmf.nist.gov/35.3.E4
    [Theory]
    // p = 1 は ln Γ。
    [InlineData(2.5, 1, 0.2846828704729192, 1.2e-13)]
    // 定義域の端 a > 1/2 の近くと、ln Γ が負の領域。
    [InlineData(0.75, 2, 2.063668419054073, 5.6e-14)]
    [InlineData(1.25, 2, 0.6773740579341823, 6.7e-14)]
    [InlineData(3.0, 3, 2.694924879806965, 6.1e-14)]
    [InlineData(10.5, 5, 64.30930570957575, 5.4e-14)]
    [InlineData(10.25, 20, 206.91690195046777, 4.7e-14)]
    [InlineData(30.0, 50, 2420.863212380659, 5.3e-14)]
    // 大きな引数と次元。
    [InlineData(1000.0, 10, 58922.58309774964, 4.5e-14)]
    [InlineData(100000000.0, 4, 6968272212.587486, 4.3e-14)]
    [InlineData(1e+300, 3, 2.069326583694641e+303, 4.3e-14)]
    // 最小の引数 a - (p - 1)/2 が極に近い（a - (p - 1)/2 は Sterbenz の補題により正確）。
    [InlineData(0.5000001, 2, 17.262825283261538, 4.1e-09)]
    [InlineData(1.0000001, 3, 18.40755511027916, 7.7e-09)]
    public void LogMultivariateGamma_MatchesReference(double a, int dimension, double expected, double tolerance)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.LogMultivariateGamma(a, dimension), tolerance);
    }

    [Theory]
    [InlineData(0.5)]
    [InlineData(2.5)]
    [InlineData(1e-300)]
    [InlineData(1e300)]
    public void LogMultivariateGamma_OneDimensionIsLogGamma(double a)
    {
        Assert.Equal(XMath.Special.LogGamma(a), XMath.Special.LogMultivariateGamma(a, 1));
    }

    // https://dlmf.nist.gov/35.3.E4 から ln Γₚ(a) = ln Γₚ₋₁(a) + ln Γ(a - (p - 1)/2) + ((p - 1)/2) ln π
    [Theory]
    [InlineData(3.7, 4)]
    [InlineData(12.25, 9)]
    [InlineData(1.0000001, 3)]
    public void LogMultivariateGamma_SatisfiesTheDimensionRecurrence(double a, int dimension)
    {
        double expected = XMath.Special.LogMultivariateGamma(a, dimension - 1) + XMath.Special.LogGamma(a - 0.5 * (dimension - 1))
            + 0.5 * (dimension - 1) * Math.Log(Math.PI);
        SpecialFunctionAssert.Close(expected, XMath.Special.LogMultivariateGamma(a, dimension), 1e-14);
    }

    [Theory]
    // 定義域 a > (p - 1)/2 の外と、次元が1未満。
    [InlineData(0.5, 2, double.NaN)]
    [InlineData(0.49, 2, double.NaN)]
    [InlineData(1.0, 3, double.NaN)]
    [InlineData(-1.5, 1, double.NaN)]
    [InlineData(0.0, 1, double.NaN)]
    [InlineData(2.0, 0, double.NaN)]
    [InlineData(2.0, -1, double.NaN)]
    [InlineData(double.NaN, 2, double.NaN)]
    [InlineData(double.NegativeInfinity, 2, double.NaN)]
    [InlineData(double.PositiveInfinity, 3, double.PositiveInfinity)]
    // 値が double の最大値を超える（補償付きの和で ∞ - ∞ にしない）。
    [InlineData(1e306, 2, double.PositiveInfinity)]
    [InlineData(1e306, 50, double.PositiveInfinity)]
    public void LogMultivariateGamma_SpecialValues(double a, int dimension, double expected)
    {
        Assert.Equal(expected, XMath.Special.LogMultivariateGamma(a, dimension));
    }
}
