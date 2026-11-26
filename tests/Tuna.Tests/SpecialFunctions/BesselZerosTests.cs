using System.Numerics;
using Tuna;

namespace Tuna.Tests;

// mpmath 1.4.1による独立参照値。binary64入力を正確に変換し、
// 90/160桁（正規分布は490/560桁）でbinary64への丸め一致を確認。
// 生成式と選定理由: tools/SpecialFunctionReferences/generate_remaining_accuracy_references.py
// 通常テストはC#の固定値と対象APIの直接呼び出しだけで完結する。
[Trait("SpecialFunction", "BesselZeros")]
public class BesselZerosTests
{

    // 非整数次数・大次数の最初の零点・導関数の番号規約。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0.0, 1, 2.404825557695773)]
    [InlineData(0.0, 3, 8.653727912911013)]
    [InlineData(0.5, 1, 3.141592653589793)]
    [InlineData(1.0, 1, 3.8317059702075125)]
    [InlineData(5.0, 2, 12.338604197466944)]
    [InlineData(20.0, 5, 41.41306551389263)]
    [InlineData(100.0, 1, 108.83616589840977)]
    [InlineData(1e-10, 1, 2.404825557850062)]
    public void BesselJZero_MatchesIndependentReferences(double nu, int k, double expected)
    {
        double actual = XMath.Special.BesselJZero(nu, k);
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
    [InlineData(double.NaN, 1)]
    public void BesselJZero_RejectsNaN(double nu, int k)
    {
        double actual = XMath.Special.BesselJZero(nu, k);
        Assert.True(double.IsNaN(actual));
    }

    // 非整数次数・大次数の最初の零点・導関数の番号規約。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0.0, 1, 0.8935769662791675)]
    [InlineData(0.0, 3, 7.086051060301773)]
    [InlineData(0.5, 1, 1.5707963267948966)]
    [InlineData(1.0, 1, 2.197141326031017)]
    [InlineData(5.0, 2, 10.597176726782031)]
    [InlineData(20.0, 5, 39.60724990429065)]
    [InlineData(100.0, 1, 104.38020425686611)]
    [InlineData(1e-10, 1, 0.8935769664238695)]
    public void BesselYZero_MatchesIndependentReferences(double nu, int k, double expected)
    {
        double actual = XMath.Special.BesselYZero(nu, k);
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
    [InlineData(double.NaN, 1)]
    public void BesselYZero_RejectsNaN(double nu, int k)
    {
        double actual = XMath.Special.BesselYZero(nu, k);
        Assert.True(double.IsNaN(actual));
    }

    // 非整数次数・大次数の最初の零点・導関数の番号規約。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0.0, 1, 0.0)]
    [InlineData(0.0, 3, 7.015586669815619)]
    [InlineData(0.5, 1, 1.1655611852072114)]
    [InlineData(1.0, 1, 1.8411837813406593)]
    [InlineData(5.0, 2, 10.519860873772307)]
    [InlineData(20.0, 5, 39.58453089075877)]
    [InlineData(100.0, 1, 103.76837768254227)]
    [InlineData(1e-10, 1, 1.414213562426128e-05)]
    public void BesselJPrimeZero_MatchesIndependentReferences(double nu, int k, double expected)
    {
        double actual = XMath.Special.BesselJPrimeZero(nu, k);
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
    [InlineData(double.NaN, 1)]
    public void BesselJPrimeZero_RejectsNaN(double nu, int k)
    {
        double actual = XMath.Special.BesselJPrimeZero(nu, k);
        Assert.True(double.IsNaN(actual));
    }

    // 非整数次数・大次数の最初の零点・導関数の番号規約。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0.0, 1, 2.197141326031017)]
    [InlineData(0.0, 3, 8.596005868331169)]
    [InlineData(0.5, 1, 2.9750863216882792)]
    [InlineData(1.0, 1, 3.6830228565851777)]
    [InlineData(5.0, 2, 12.280868725807847)]
    [InlineData(20.0, 5, 41.39254740647775)]
    [InlineData(100.0, 1, 108.65470181072872)]
    [InlineData(1e-10, 1, 2.197141326198829)]
    public void BesselYPrimeZero_MatchesIndependentReferences(double nu, int k, double expected)
    {
        double actual = XMath.Special.BesselYPrimeZero(nu, k);
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
    [InlineData(double.NaN, 1)]
    public void BesselYPrimeZero_RejectsNaN(double nu, int k)
    {
        double actual = XMath.Special.BesselYPrimeZero(nu, k);
        Assert.True(double.IsNaN(actual));
    }

    [Fact]
    public void InvalidIndicesAndDerivativeZeroConvention()
    {
        Assert.True(double.IsNaN(XMath.Special.BesselJZero(0, 0)));
        Assert.True(double.IsNaN(XMath.Special.BesselYZero(-1, 1)));
        Assert.True(double.IsNaN(XMath.Special.BesselJPrimeZero(0, -1)));
        Assert.True(double.IsNaN(XMath.Special.BesselYPrimeZero(0, 0)));
        Assert.Equal(0, XMath.Special.BesselJPrimeZero(0, 1));
        Assert.True(XMath.Special.BesselJZero(0, 1) < XMath.Special.BesselYZero(0, 2));
        SpecialFunctionAssert.Close(0, XMath.Special.BesselJ(0, XMath.Special.BesselJZero(0, 1)), relativeTolerance: 0, absoluteTolerance: 2e-14);
    }
}
