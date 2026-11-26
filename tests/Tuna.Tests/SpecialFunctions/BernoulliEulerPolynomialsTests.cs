using System.Numerics;
using Tuna;

namespace Tuna.Tests;

// mpmath 1.4.1による独立参照値。binary64入力を正確に変換し、
// 90/160桁（正規分布は490/560桁）でbinary64への丸め一致を確認。
// 生成式と選定理由: tools/SpecialFunctionReferences/generate_remaining_accuracy_references.py
// 通常テストはC#の固定値と対象APIの直接呼び出しだけで完結する。
[Trait("SpecialFunction", "BernoulliEulerPolynomials")]
public class BernoulliEulerPolynomialsTests
{

    // 次数12の切替・反射/平行移動・高次・零点の隣接値。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0, 0.25, 1.0)]
    [InlineData(1, 0.25, -0.25)]
    [InlineData(2, 0.25, -0.020833333333333332)]
    [InlineData(5, -0.2, 0.01568)]
    [InlineData(8, 0.7, 0.010359476666666676)]
    [InlineData(11, 0.25, 0.13249659538269043)]
    [InlineData(12, 0.25, 6.176512756627122e-05)]
    [InlineData(13, 0.5, 0.0)]
    [InlineData(20, -0.2, -163.50781870214743)]
    [InlineData(20, 2.3, 3087.3492159339717)]
    [InlineData(50, 0.125, 5.30391374092769e+24)]
    [InlineData(100, 0.75, 2.2389647088546112e+48)]
    [InlineData(13, 0.49999999999999994, -1.825689328965213e-16)]
    [InlineData(13, 0.5000000000000001, 3.651378657930426e-16)]
    public void BernoulliPolynomial_MatchesIndependentReferences(int n, double x, double expected)
    {
        double actual = XMath.Special.BernoulliPolynomial(n, x);
        if (expected == 0 || double.IsInfinity(expected))
        {
            Assert.Equal(expected, actual);
        }
        else
        {
            SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 2e-12);
            if (Math.Abs(expected) < 2.2250738585072014e-308)
            {
                Assert.Equal(Math.Sign(expected), Math.Sign(actual));
            }
        }
    }

    // 各引数のNaNを独立に検査。
    [Theory]
    [InlineData(0, double.NaN)]
    public void BernoulliPolynomial_RejectsNaN(int n, double x)
    {
        double actual = XMath.Special.BernoulliPolynomial(n, x);
        Assert.True(double.IsNaN(actual));
    }

    // 次数12の切替・反射/平行移動・高次・零点の隣接値。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0, 0.25, 1.0)]
    [InlineData(1, 0.25, -0.25)]
    [InlineData(2, 0.25, -0.1875)]
    [InlineData(5, -0.2, -0.40432)]
    [InlineData(8, 0.7, 4.37721081)]
    [InlineData(11, 0.25, 122.15223574638367)]
    [InlineData(12, 0.25, 466.5883432030678)]
    [InlineData(13, 0.5, 0.0)]
    [InlineData(20, -0.2, -207613680.24305555)]
    [InlineData(20, 2.3, 285756095.9945327)]
    [InlineData(50, 0.125, -2.057458182368942e+39)]
    [InlineData(100, 0.75, 1.619613940088683e+108)]
    [InlineData(13, 0.49999999999999994, -4.7618084876664e-13)]
    [InlineData(13, 0.5000000000000001, 9.5236169753328e-13)]
    public void EulerPolynomial_MatchesIndependentReferences(int n, double x, double expected)
    {
        double actual = XMath.Special.EulerPolynomial(n, x);
        if (expected == 0 || double.IsInfinity(expected))
        {
            Assert.Equal(expected, actual);
        }
        else
        {
            SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 2e-12);
            if (Math.Abs(expected) < 2.2250738585072014e-308)
            {
                Assert.Equal(Math.Sign(expected), Math.Sign(actual));
            }
        }
    }

    // 各引数のNaNを独立に検査。
    [Theory]
    [InlineData(0, double.NaN)]
    public void EulerPolynomial_RejectsNaN(int n, double x)
    {
        double actual = XMath.Special.EulerPolynomial(n, x);
        Assert.True(double.IsNaN(actual));
    }

    [Theory]
    [InlineData(3, .25)]
    [InlineData(12, .25)]
    [InlineData(20, .3)]
    public void ReflectionAndTranslation(int n, double x)
    {
        double sign = n % 2 == 0 ? 1 : -1;
        SpecialFunctionAssert.Close(sign * XMath.Special.BernoulliPolynomial(n, x), XMath.Special.BernoulliPolynomial(n, 1 - x), relativeTolerance: 2e-12);
        SpecialFunctionAssert.Close(sign * XMath.Special.EulerPolynomial(n, x), XMath.Special.EulerPolynomial(n, 1 - x), relativeTolerance: 2e-12);
        // 高次の大きな値どうしを引いて小さい差を作らず、恒等式の両辺を値の尺度で比較する。
        SpecialFunctionAssert.Close(XMath.Special.BernoulliPolynomial(n, x) + n * Math.Pow(x, n - 1), XMath.Special.BernoulliPolynomial(n, x + 1), relativeTolerance: 2e-12);
        SpecialFunctionAssert.Close(2 * Math.Pow(x, n) - XMath.Special.EulerPolynomial(n, x), XMath.Special.EulerPolynomial(n, x + 1), relativeTolerance: 2e-12);
    }

    [Fact]
    public void DegreesAndExactZeros()
    {
        Assert.True(double.IsNaN(XMath.Special.BernoulliPolynomial(-1, .5)));
        Assert.True(double.IsNaN(XMath.Special.EulerPolynomial(-1, .5)));
        Assert.Equal(0, XMath.Special.BernoulliPolynomial(13, .5));
        Assert.Equal(0, XMath.Special.EulerPolynomial(13, .5));
        Assert.Equal(0, XMath.Special.EulerPolynomial(12, 0));
    }
}
