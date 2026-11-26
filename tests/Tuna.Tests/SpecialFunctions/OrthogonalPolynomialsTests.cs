using System.Numerics;
using Tuna;

namespace Tuna.Tests;

// 有限参照値は mpmath 1.4.1、100 桁以上で生成し binary64 に丸めた。
// 入力も binary64 の値を正確に高精度へ変換（十進文字列の値とは区別）。
// 参照値生成: https://mpmath.org/doc/current/functions/index.html
[Trait("SpecialFunction", "OrthogonalPolynomials")]
public class OrthogonalPolynomialsTests
{

    // https://dlmf.nist.gov/18.3; https://dlmf.nist.gov/18.5; https://dlmf.nist.gov/18.7
    [Theory]
    // 0次の単位多項式。
    [InlineData(0, 0.25, 1.0)]
    // 1次の正規化。
    [InlineData(1, 0.25, 0.25)]
    // 二つのHermiteの流儀と係数の区別。
    [InlineData(2, 0.25, -0.40625)]
    // 奇数次の原点零点。
    [InlineData(3, 0.0, 0.0)]
    // 負のxと高次の漸化式。
    [InlineData(10, -0.25, 0.22120021656155586)]
    // 直交区間外も多項式として定義。
    [InlineData(20, 2.0, 35789689195.11653)]
    // 高次数の正端点。
    [InlineData(50, 1.0, 1.0)]
    // 負端点で奇数次数の符号。
    [InlineData(51, -1.0, -1.0)]
    // 端点のすぐ内側。
    [InlineData(50, 0.9999999999999999, 0.9999999999998584)]
    // 端点のすぐ外側。
    [InlineData(50, 1.0000000000000002, 1.000000000000283)]
    public void LegendreP_MatchesReference(int n, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.LegendreP(n, x), 2e-12, 0);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/18.3; https://dlmf.nist.gov/18.5; https://dlmf.nist.gov/18.7
    [Theory]
    // 負の次数。
    [InlineData(-1, 0.5, double.NaN)]
    // 0次でもNaNを伝播する規約。
    [InlineData(0, double.NaN, double.NaN)]
    public void LegendreP_SpecialValues(int n, double x, double expected)
    {
        double actual = XMath.Special.LegendreP(n, x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // https://dlmf.nist.gov/18.3; https://dlmf.nist.gov/18.5; https://dlmf.nist.gov/18.7
    [Theory]
    // 0次の単位多項式。
    [InlineData(0, 0.25, 1.0)]
    // 1次の正規化。
    [InlineData(1, 0.25, 0.5)]
    // 二つのHermiteの流儀と係数の区別。
    [InlineData(2, 0.25, -1.75)]
    // 奇数次の原点零点。
    [InlineData(3, 0.0, 0.0)]
    // 負のxと高次の漸化式。
    [InlineData(10, -0.25, -12875.9755859375)]
    // 直交区間外も多項式として定義。
    [InlineData(20, 2.0, 5080118660096.0)]
    public void HermiteH_MatchesReference(int n, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.HermiteH(n, x), 2e-12, 0);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/18.3; https://dlmf.nist.gov/18.5; https://dlmf.nist.gov/18.7
    [Theory]
    // 負の次数。
    [InlineData(-1, 0.5, double.NaN)]
    // 0次でもNaNを伝播する規約。
    [InlineData(0, double.NaN, double.NaN)]
    public void HermiteH_SpecialValues(int n, double x, double expected)
    {
        double actual = XMath.Special.HermiteH(n, x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // https://dlmf.nist.gov/18.3; https://dlmf.nist.gov/18.5; https://dlmf.nist.gov/18.7
    [Theory]
    // 0次の単位多項式。
    [InlineData(0, 0.25, 1.0)]
    // 1次の正規化。
    [InlineData(1, 0.25, 0.25)]
    // 二つのHermiteの流儀と係数の区別。
    [InlineData(2, 0.25, -0.9375)]
    // 奇数次の原点零点。
    [InlineData(3, 0.0, 0.0)]
    // 負のxと高次の漸化式。
    [InlineData(10, -0.25, -661.8390645980835)]
    // 直交区間外も多項式として定義。
    [InlineData(20, 2.0, -1628815029.0)]
    public void HermiteHe_MatchesReference(int n, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.HermiteHe(n, x), 2e-12, 0);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/18.3; https://dlmf.nist.gov/18.5; https://dlmf.nist.gov/18.7
    [Theory]
    // 負の次数。
    [InlineData(-1, 0.5, double.NaN)]
    // 0次でもNaNを伝播する規約。
    [InlineData(0, double.NaN, double.NaN)]
    public void HermiteHe_SpecialValues(int n, double x, double expected)
    {
        double actual = XMath.Special.HermiteHe(n, x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // https://dlmf.nist.gov/18.3; https://dlmf.nist.gov/18.5; https://dlmf.nist.gov/18.7
    [Theory]
    // 0次の単位多項式。
    [InlineData(0, 0.25, 1.0)]
    // 1次の正規化。
    [InlineData(1, 0.25, 0.25)]
    // 二つのHermiteの流儀と係数の区別。
    [InlineData(2, 0.25, -0.875)]
    // 奇数次の原点零点。
    [InlineData(3, 0.0, 0.0)]
    // 負のxと高次の漸化式。
    [InlineData(10, -0.25, 0.81689453125)]
    // 直交区間外も多項式として定義。
    [InlineData(20, 2.0, 137379191137.0)]
    // 高次数の正端点。
    [InlineData(50, 1.0, 1.0)]
    // 負端点で奇数次数の符号。
    [InlineData(51, -1.0, -1.0)]
    // 端点のすぐ内側。
    [InlineData(50, 0.9999999999999999, 0.9999999999997224)]
    // 端点のすぐ外側。
    [InlineData(50, 1.0000000000000002, 1.0000000000005551)]
    public void ChebyshevT_MatchesReference(int n, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.ChebyshevT(n, x), 2e-12, 0);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/18.3; https://dlmf.nist.gov/18.5; https://dlmf.nist.gov/18.7
    [Theory]
    // 負の次数。
    [InlineData(-1, 0.5, double.NaN)]
    // 0次でもNaNを伝播する規約。
    [InlineData(0, double.NaN, double.NaN)]
    public void ChebyshevT_SpecialValues(int n, double x, double expected)
    {
        double actual = XMath.Special.ChebyshevT(n, x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // https://dlmf.nist.gov/18.3; https://dlmf.nist.gov/18.5; https://dlmf.nist.gov/18.7
    [Theory]
    // 0次の単位多項式。
    [InlineData(0, 0.25, 1.0)]
    // 1次の正規化。
    [InlineData(1, 0.25, 0.5)]
    // 二つのHermiteの流儀と係数の区別。
    [InlineData(2, 0.25, -0.75)]
    // 奇数次の原点零点。
    [InlineData(3, 0.0, 0.0)]
    // 負のxと高次の漸化式。
    [InlineData(10, -0.25, 0.9658203125)]
    // 直交区間外も多項式として定義。
    [InlineData(20, 2.0, 296011017105.0)]
    // 高次数の正端点。
    [InlineData(50, 1.0, 51.0)]
    // 負端点で奇数次数の符号。
    [InlineData(51, -1.0, -52.0)]
    // 端点のすぐ内側。
    [InlineData(50, 0.9999999999999999, 50.99999999999509)]
    // 端点のすぐ外側。
    [InlineData(50, 1.0000000000000002, 51.00000000000981)]
    public void ChebyshevU_MatchesReference(int n, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.ChebyshevU(n, x), 2e-12, 0);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/18.3; https://dlmf.nist.gov/18.5; https://dlmf.nist.gov/18.7
    [Theory]
    // 負の次数。
    [InlineData(-1, 0.5, double.NaN)]
    // 0次でもNaNを伝播する規約。
    [InlineData(0, double.NaN, double.NaN)]
    public void ChebyshevU_SpecialValues(int n, double x, double expected)
    {
        double actual = XMath.Special.ChebyshevU(n, x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // https://dlmf.nist.gov/14.7; https://dlmf.nist.gov/14.9
    [Theory]
    // 0次0位。
    [InlineData(0, 0, 0.25, 1.0)]
    // Condon–Shortley位相でP11(0)=-1。
    [InlineData(1, 1, 0.0, -1.0)]
    // m=0でLegendreに一致。
    [InlineData(3, 0, 0.25, -0.3359375)]
    // 高位の正の次数。
    [InlineData(3, 2, 0.25, 3.515625)]
    // 負位数の階乗比。
    [InlineData(3, -2, 0.25, 0.029296875)]
    // 負位数の奇数位相。
    [InlineData(3, -1, 0.25, -0.08320862657867498)]
    // n=mの初期値。
    [InlineData(10, 10, 0.25, 474152461.3648653)]
    // 端点のすぐ内側で平方根の桁落ちを避ける。
    [InlineData(10, 1, 0.9999999999999999, -8.195638656616187e-07)]
    public void AssociatedLegendreP_MatchesReference(int n, int m, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.AssociatedLegendreP(n, m, x), 2e-12, 0);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/14.7; https://dlmf.nist.gov/14.9
    [Theory]
    // m>0の正端点。
    [InlineData(3, 1, 1.0, 0.0)]
    // m>0の負端点。
    [InlineData(3, 1, -1.0, 0.0)]
    // |m|>nの規約。
    [InlineData(2, 3, 0.5, 0.0)]
    // 最小intでも|m|>n。整数absのoverflowで例外を投げない。
    [InlineData(2, -2147483648, 0.5, 0.0)]
    // 負のn。
    [InlineData(-1, 0, 0.5, double.NaN)]
    // 正端点の外。
    [InlineData(2, 1, 1.0000000000000002, double.NaN)]
    // 負端点の外。
    [InlineData(2, 1, -1.0000000000000002, double.NaN)]
    // NaN。
    [InlineData(2, 1, double.NaN, double.NaN)]
    public void AssociatedLegendreP_SpecialValues(int n, int m, double x, double expected)
    {
        double actual = XMath.Special.AssociatedLegendreP(n, m, x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // https://dlmf.nist.gov/18.3; https://dlmf.nist.gov/18.5; https://dlmf.nist.gov/18.7
    [Theory]
    // 0次。
    [InlineData(0, 0.25, 1.0)]
    // 1次。
    [InlineData(1, 0.25, 0.75)]
    // 高次の普通Laguerre。
    [InlineData(10, 2.0, -0.3090652557319224)]
    // 原点では1。
    [InlineData(10, 0.0, 1.0)]
    public void LaguerreL_MatchesReference(int n, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.LaguerreL(n, x), 5e-13, 0);
    }

    // https://dlmf.nist.gov/18.3; https://dlmf.nist.gov/18.5; https://dlmf.nist.gov/18.7
    [Theory]
    // 一般化の0次。
    [InlineData(0, 0.5, 0.25, 1.0)]
    // 一般化の1次。
    [InlineData(1, 0.5, 0.25, 1.25)]
    // 一般化の高次。
    [InlineData(10, 0.5, 2.0, 0.2418683493747072)]
    // α=-1でも多項式として有限。
    [InlineData(3, -1.0, 0.25, -0.19010416666666666)]
    // 負整数αでΓの極を含む表現を避ける。
    [InlineData(5, -3.0, 0.25, -0.0022867838541666668)]
    // 原点は一般化二項係数。
    [InlineData(10, 0.5, 0.0, 3.7001380920410156)]
    public void LaguerreL_Generalized_MatchesReference(int n, double alpha, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.LaguerreL(n, alpha, x), 2e-12, 0);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/18.3; https://dlmf.nist.gov/18.5; https://dlmf.nist.gov/18.7
    [Theory]
    // 負の次数。
    [InlineData(-1, 0.5, double.NaN)]
    // NaN。
    [InlineData(0, double.NaN, double.NaN)]
    public void LaguerreL_SpecialValues(int n, double x, double expected)
    {
        double actual = XMath.Special.LaguerreL(n, x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/18.3; https://dlmf.nist.gov/18.5; https://dlmf.nist.gov/18.7
    [Theory]
    // 負の次数。
    [InlineData(-1, 0.5, 0.5, double.NaN)]
    // α NaN。
    [InlineData(0, double.NaN, 0.5, double.NaN)]
    // x NaN。
    [InlineData(0, 0.5, double.NaN, double.NaN)]
    public void LaguerreL_Generalized_SpecialValues(int n, double alpha, double x, double expected)
    {
        double actual = XMath.Special.LaguerreL(n, alpha, x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // https://dlmf.nist.gov/18.3; https://dlmf.nist.gov/18.5; https://dlmf.nist.gov/18.7
    [Theory]
    // λ=0でも0次は1。
    [InlineData(0, 0.0, 0.25, 1.0)]
    // λ=0の高次は0。
    [InlineData(3, 0.0, 0.25, 0.0)]
    // λ=1/2でLegendre。
    [InlineData(10, 0.5, 0.25, 0.22120021656155586)]
    // λ=1でChebyshevU。
    [InlineData(10, 1.0, 0.25, 0.9658203125)]
    // 通常の高次。
    [InlineData(10, 2.5, 0.25, 12.718109715729952)]
    // 負整数λで次数が退化。
    [InlineData(3, -1.0, 0.25, 0.0)]
    // 負半整数λの拡張。
    [InlineData(5, -0.5, 0.25, -0.0750732421875)]
    public void GegenbauerC_MatchesReference(int n, double lambda, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.GegenbauerC(n, lambda, x), 2e-12, 0);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/18.3; https://dlmf.nist.gov/18.5; https://dlmf.nist.gov/18.7
    [Theory]
    // 負の次数。
    [InlineData(-1, 0.5, 0.5, double.NaN)]
    // λ NaN。
    [InlineData(0, double.NaN, 0.5, double.NaN)]
    // x NaN。
    [InlineData(0, 0.5, double.NaN, double.NaN)]
    public void GegenbauerC_SpecialValues(int n, double lambda, double x, double expected)
    {
        double actual = XMath.Special.GegenbauerC(n, lambda, x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // https://dlmf.nist.gov/18.3; https://dlmf.nist.gov/18.5; https://dlmf.nist.gov/18.7
    [Theory]
    // 0次。
    [InlineData(0, 0.5, 1.5, 0.25, 1.0)]
    // 1次のα,βの向き。
    [InlineData(1, 0.5, 1.5, 0.25, 0.0)]
    // α=β=0でLegendre。
    [InlineData(10, 0.0, 0.0, 0.25, 0.22120021656155586)]
    // 正端点はC(n+α,n)。
    [InlineData(10, 0.5, 1.5, 1.0, 3.7001380920410156)]
    // 負端点は(-1)^n C(n+β,n)。
    [InlineData(11, 0.5, 1.5, -1.0, -32.23605155944824)]
    // 高次の非対称パラメータ。
    [InlineData(10, 0.5, 1.5, 0.25, 0.23207986168563366)]
    // α+β=-2で漸化式分母が0でも多項式は有限。
    [InlineData(3, -1.0, -1.0, 0.25, -0.1171875)]
    // 同じ退化境界の非対称パラメータ。
    [InlineData(3, -1.5, -0.5, 0.25, 0.078125)]
    public void JacobiP_MatchesReference(int n, double alpha, double beta, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.JacobiP(n, alpha, beta, x), 2e-12, 0);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/18.3; https://dlmf.nist.gov/18.5; https://dlmf.nist.gov/18.7
    [Theory]
    // 負の次数。
    [InlineData(-1, 0.5, 0.5, 0.5, double.NaN)]
    // α NaN。
    [InlineData(0, double.NaN, 0.5, 0.5, double.NaN)]
    // β NaN。
    [InlineData(0, 0.5, double.NaN, 0.5, double.NaN)]
    // x NaN。
    [InlineData(0, 0.5, 0.5, double.NaN, double.NaN)]
    public void JacobiP_SpecialValues(int n, double alpha, double beta, double x, double expected)
    {
        double actual = XMath.Special.JacobiP(n, alpha, beta, x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // https://dlmf.nist.gov/14.30
    [Theory]
    // Y00=1/√(4π)。
    [InlineData(0, 0, 0.0, 0.0, 0.28209479177387814)]
    // Condon–Shortleyの負符号と複素位相。
    [InlineData(1, 1, 1.0, 0.25, -0.28168541878329745)]
    // 通常の正位数。
    [InlineData(3, 2, 1.0, 0.25, 0.3431214123396539)]
    // 負位数は(-1)^m共役。
    [InlineData(3, -2, 1.0, 0.25, 0.3431214123396539)]
    // 負の奇位数の位相。
    [InlineData(3, -1, 1.0, 0.25, 0.12110986186041864)]
    // 階乗比を作らず正規化する高位数。
    [InlineData(50, 40, 1.0, 0.25, -0.4846832245188539)]
    // 天頂角が極小: cosθが1でもsinθは非零。
    [InlineData(10, 1, 1e-10, 0.25, -6.568339405674932e-10)]
    public void SphericalHarmonicYReal_MatchesReference(int l, int m, double theta, double phi, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.SphericalHarmonicY(l, m, theta, phi).Real, 2e-12, 1e-30);
    }

    // https://dlmf.nist.gov/14.30
    [Theory]
    // Y00=1/√(4π)。
    [InlineData(0, 0, 0.0, 0.0, 0.0)]
    // Condon–Shortleyの負符号と複素位相。
    [InlineData(1, 1, 1.0, 0.25, -0.07192609601207935)]
    // 通常の正位数。
    [InlineData(3, 2, 1.0, 0.25, 0.1874480818798708)]
    // 負位数は(-1)^m共役。
    [InlineData(3, -2, 1.0, 0.25, -0.1874480818798708)]
    // 負の奇位数の位相。
    [InlineData(3, -1, 1.0, 0.25, -0.0309244248062536)]
    // 階乗比を作らず正規化する高位数。
    [InlineData(50, 40, 1.0, 0.25, -0.3142496165045824)]
    // 天頂角が極小: cosθが1でもsinθは非零。
    [InlineData(10, 1, 1e-10, 0.25, -1.6771724030768766e-10)]
    public void SphericalHarmonicYImaginary_MatchesReference(int l, int m, double theta, double phi, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.SphericalHarmonicY(l, m, theta, phi).Imaginary, 2e-12, 1e-30);
    }

    // |m|>lの0規約とNaN成分。最小intはMath.Absの例外を起こしやすい境界。https://dlmf.nist.gov/14.30
    [Fact]
    public void SphericalHarmonicY_Domain()
    {
        Assert.Equal(System.Numerics.Complex.Zero, XMath.Special.SphericalHarmonicY(2, 3, 1, 0));
        Assert.Equal(System.Numerics.Complex.Zero, XMath.Special.SphericalHarmonicY(2, int.MinValue, 1, 0));
        var badDegree = XMath.Special.SphericalHarmonicY(-1, 0, 1, 0);
        Assert.True(double.IsNaN(badDegree.Real) && double.IsNaN(badDegree.Imaginary));
        var badTheta = XMath.Special.SphericalHarmonicY(1, 0, double.NaN, 0);
        Assert.True(double.IsNaN(badTheta.Real) && double.IsNaN(badTheta.Imaginary));
        var badPhi = XMath.Special.SphericalHarmonicY(1, 0, 1, double.NaN);
        Assert.True(double.IsNaN(badPhi.Real) && double.IsNaN(badPhi.Imaginary));
    }

    // 退化点と有限和への切替|係数|=1/4の両隣。後続の漸化式も含める。
    // 参照値は一般化二項係数の有限和を160桁・220桁で独立計算。
    // https://dlmf.nist.gov/18.5.E7
    [Theory]
    [InlineData(3, -1.5, -0.7500000000000001, 0.25, 0.040359497070312486)]
    [InlineData(3, -1.5, -0.75, 0.25, 0.0403594970703125)]
    [InlineData(3, -1.5, -0.7499999999999999, 0.25, 0.040359497070312514)]
    [InlineData(3, -1.5, -0.25000000000000006, 0.25, 0.11918640136718749)]
    [InlineData(3, -1.5, -0.25, 0.25, 0.1191864013671875)]
    [InlineData(3, -1.5, -0.24999999999999997, 0.25, 0.1191864013671875)]
    [InlineData(6, -2.5, -2.7500000000000004, 0.25, -0.014405705907847718)]
    [InlineData(6, -2.5, -2.75, 0.25, -0.014405705907847732)]
    [InlineData(6, -2.5, -2.7499999999999996, 0.25, -0.014405705907847746)]
    [InlineData(6, -2.5, -2.2500000000000004, 0.25, -0.030848210153635577)]
    [InlineData(6, -2.5, -2.25, 0.25, -0.03084821015363559)]
    [InlineData(6, -2.5, -2.2499999999999996, 0.25, -0.030848210153635605)]
    [InlineData(3, -1.0, -1.0000000000000002, 0.25, -0.11718750000000003)]
    [InlineData(3, -1.0, -1.0, 0.25, -0.1171875)]
    [InlineData(3, -1.0, -0.9999999999999999, 0.25, -0.11718749999999999)]
    [InlineData(3, -1.5, -0.5000000000000001, 0.25, 0.07812499999999999)]
    [InlineData(3, -1.5, -0.5, 0.25, 0.078125)]
    [InlineData(3, -1.5, -0.49999999999999994, 0.25, 0.07812500000000001)]
    [InlineData(6, -2.5, -2.5000000000000004, 0.25, -0.02240371704101561)]
    [InlineData(6, -2.5, -2.5, 0.25, -0.022403717041015625)]
    [InlineData(6, -2.5, -2.4999999999999996, 0.25, -0.02240371704101564)]
    [InlineData(3, -1.5, -0.75000001, 0.25, 0.04035949563659667)]
    [InlineData(3, -1.5, -0.74999999, 0.25, 0.040359498504028334)]
    [InlineData(3, -1.5, -0.25000001, 0.25, 0.11918639966979981)]
    [InlineData(3, -1.5, -0.24999999, 0.25, 0.1191864030645752)]
    [InlineData(6, -2.5, -2.75000001, 0.25, -0.01440570560189323)]
    [InlineData(6, -2.5, -2.74999999, 0.25, -0.014405706213802237)]
    [InlineData(6, -2.5, -2.25000001, 0.25, -0.030848209812224345)]
    [InlineData(6, -2.5, -2.24999999, 0.25, -0.030848210495046838)]
    public void JacobiP_DegenerateAndBranchBoundaries_MatchReference(int n, double alpha, double beta, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.JacobiP(n, alpha, beta, x), 2e-12, 0);
    }

    // |m|>nの両側、最小・最大intと0次。absのoverflowや片側だけの判定を検出。
    [Theory]
    [InlineData(0, -1)]
    [InlineData(0, 1)]
    [InlineData(0, int.MinValue)]
    [InlineData(0, int.MaxValue)]
    [InlineData(2, -3)]
    [InlineData(2, 3)]
    [InlineData(2, int.MinValue)]
    [InlineData(2, int.MaxValue)]
    public void PolynomialOrders_OutsideDegreeAreExactlyZero(int degree, int order)
    {
        Assert.Equal(0, XMath.Special.AssociatedLegendreP(degree, order, 0.5));
        Assert.Equal(System.Numerics.Complex.Zero, XMath.Special.SphericalHarmonicY(degree, order, 1, 0.25));
    }

    // 有効な位数の端を誤って除外しない。P_2^2(1/2)=9/4、P_2^-2=3/32。
    [Theory]
    [InlineData(-2, 0.09375)]
    [InlineData(2, 2.25)]
    public void AssociatedLegendreP_OrderEndpointsRemainValid(int order, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.AssociatedLegendreP(2, order, 0.5));
        var y = XMath.Special.SphericalHarmonicY(2, order, Math.PI / 2, 0);
        SpecialFunctionAssert.Close(Math.Sqrt(15 / (32 * Math.PI)), y.Real);
        Assert.Equal(0, y.Imaginary);
    }

    // R14: P_n^-n(0)=1/(2^n n!)。ガンマ比を参照計算に使わない。
    // https://dlmf.nist.gov/14.7; https://dlmf.nist.gov/14.9.E3
    [Trait("Category", "ReviewRegression")]
    [Fact]
    public void AssociatedLegendre_NegativeDiagonalOrder_PreservesFiniteResult()
    {
        double expected = 1;
        for (int j = 1; j <= 100; j++)
        {
            expected /= 2.0 * j;
        }

        SpecialFunctionAssert.Close(expected, XMath.Special.AssociatedLegendreP(100, -100, 0));
    }

    // Y_l^l(θ,0)=(-1)^l sqrt((2l+1)/(4π) Π_(j=1)^l((2j-1)/(2j))) sin(θ)^l。
    // 正規化定数とsin^lを別々に評価し、sin^(2l)の先行アンダーフローを避ける。
    // https://dlmf.nist.gov/14.30
    [Trait("Category", "ReviewRegression")]
    [Fact]
    public void SphericalHarmonic_HighOrderNearPole_PreservesFiniteResult()
    {
        double norm = 1;
        for (int j = 1; j <= 200; j++)
        {
            norm *= (2.0 * j - 1) / (2 * j);
        }

        double expected = Math.Sqrt(401 * norm / (4 * Math.PI)) * Math.Pow(Math.Sin(0.1), 200);
        var actual = XMath.Special.SphericalHarmonicY(200, 200, 0.1, 0);
        SpecialFunctionAssert.Close(expected, actual.Real);
        Assert.Equal(0, actual.Imaginary);
    }

    // 探索で得た補強点。mpmath 1.4.1で独立に計算し、複数精度でbinary64丸めを確認。
    // 入力・期待値・許容誤差をここに固定する。出典と選定方針は README.md を参照。

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(51, 0.2228160922511102, 0.09436494696008023)]
    [InlineData(7, -0.6793871976816175, -0.209734803611832)]
    [InlineData(55, -0.7881996111150615, 0.016970193408538243)]
    [InlineData(17, 0.8507337629233569, -0.22590637004965897)]
    [InlineData(41, -0.20465917221890595, -0.09570962945699851)]
    [InlineData(10, 0.43305838112516004, 0.00101898567847434)]
    public void LegendreP_ExploredRelativeValues(int n, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.LegendreP(n, x), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(51, 0.2228160922511102, -1.557682289714118e+40)]
    [InlineData(42, -0.35705229736263, 2.899380772318155e+31)]
    [InlineData(7, -0.6793871976816175, 273.71114385443343)]
    [InlineData(55, -0.7881996111150615, 2.7325027533035053e+44)]
    [InlineData(17, 0.8507337629233569, -4106365716.601898)]
    [InlineData(41, -0.20465917221890595, -2.950731323720417e+30)]
    public void HermiteH_ExploredRelativeValues(int n, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.HermiteH(n, x), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(51, 0.2228160922511102, -4.203196059061322e+32)]
    [InlineData(42, -0.35705229736263, 9.295643749893729e+24)]
    [InlineData(7, -0.6793871976816175, 41.38219433005654)]
    [InlineData(55, -0.7881996111150615, -5.481057375109317e+35)]
    [InlineData(17, 0.8507337629233569, -3961163.4290336384)]
    [InlineData(41, -0.20465917221890595, -1.9918607679716805e+24)]
    public void HermiteHe_ExploredRelativeValues(int n, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.HermiteHe(n, x), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(51, 0.2228160922511102, 0.8941622542376552)]
    [InlineData(7, -0.6793871976816175, -0.8697497741824176)]
    [InlineData(55, -0.7881996111150615, -0.32611501578229446)]
    [InlineData(17, 0.8507337629233569, -0.9998606147694546)]
    [InlineData(41, -0.20465917221890595, -0.8271642468536337)]
    [InlineData(47, 0.2025636749801616, 0.16134631195042778)]
    public void ChebyshevT_ExploredRelativeValues(int n, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.ChebyshevT(n, x), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(51, 0.2228160922511102, 0.791825188515807)]
    [InlineData(7, -0.6793871976816175, -0.4128378450160124)]
    [InlineData(55, -0.7881996111150615, 0.8846178387708202)]
    [InlineData(41, -0.20465917221890595, -0.7096668983059005)]
    [InlineData(35, 0.6749394014232073, -1.3545627996814436)]
    [InlineData(10, 0.43305838112516004, -0.23596162823628553)]
    public void ChebyshevU_ExploredRelativeValues(int n, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.ChebyshevU(n, x), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(51, 0.2228160922511102, 0.32566929173386283)]
    [InlineData(7, -0.6793871976816175, 12.768383783231778)]
    [InlineData(55, -0.7881996111150615, 41800.42080884499)]
    [InlineData(17, 0.8507337629233569, 0.3591362348536796)]
    [InlineData(41, -0.20465917221890595, 51.972988516220035)]
    [InlineData(53, 0.38136092837409774, -0.11855857616582388)]
    public void LaguerreL_ExploredRelativeValues(int n, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.LaguerreL(n, x), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(22, 8, 0.17545184273633718, 8172715176.235273)]
    [InlineData(26, 9, -0.47496907923529186, -273825157207.23154)]
    [InlineData(29, 20, -0.5499285893754893, 5.292176041150267e+27)]
    [InlineData(13, -12, -0.9433563158846874, -8.554653418663902e-19)]
    [InlineData(11, 8, -0.639054163639426, -47635128.66812471)]
    [InlineData(13, -1, 0.4242930095420736, 0.015724015505962505)]
    public void AssociatedLegendreP_ExploredRelativeValues(int n, int m, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.AssociatedLegendreP(n, m, x), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(26, 3.5616072247018358, 0.6315614510039376, -324.3000362762869)]
    [InlineData(34, 3.250411864929416, -0.5403229275095731, 354526.42592971004)]
    [InlineData(39, 1.059127418748678, 0.9901377700295313, -1.939880188853183)]
    [InlineData(34, 0.10105605840481897, 0.20658576513297588, -0.1634418116285695)]
    [InlineData(39, 0.32672641754084225, -0.5479026870077093, 2267.8313357826205)]
    [InlineData(1, 3.7097619584747683, 0.5228182130976489, 4.186943745377119)]
    public void LaguerreL_ExploredRelativeValues_3Arguments(int n, double alpha, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.LaguerreL(n, alpha, x), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(26, 3.5616072247018358, 0.6315614510039376, -229.1038957561052)]
    [InlineData(34, 3.250411864929416, -0.5403229275095731, 348.4597327576406)]
    [InlineData(39, 1.059127418748678, 0.9901377700295313, -6.662382761839358)]
    [InlineData(34, 0.10105605840481897, 0.20658576513297588, -0.005718283577626771)]
    [InlineData(39, 0.32672641754084225, -0.5479026870077093, -0.03793371366608297)]
    [InlineData(1, 3.7097619584747683, 0.5228182130976489, 3.8790622362948253)]
    public void GegenbauerC_ExploredRelativeValues(int n, double lambda, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.GegenbauerC(n, lambda, x), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    // 前回の探索で最大相対誤差となった入力。
    [InlineData(32, 3.461988037941982, 0.6242892305130581, 0.9562494652312001, -20.644014204761326)]
    [InlineData(36, 3.8222307607303385, 0.40103999517440636, 0.7634014003862377, -1.5116477983138754)]
    [InlineData(0, 3.631280997588723, 1.4832676885568938, 0.06522193090239137, 1.0)]
    [InlineData(5, 2.121026277341117, 2.1882267421956927, -0.43699990727254323, 0.28646262256128413)]
    [InlineData(27, 3.3587419991608884, 0.9189228179333475, -0.5231245024148401, 0.439184403563564)]
    [InlineData(13, 0.13274325280235066, 3.93684114047198, -0.4027999343916875, -0.8584840541232803)]
    public void JacobiP_ExploredRelativeValues(int n, double alpha, double beta, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.JacobiP(n, alpha, beta, x), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(20, -11, 0.969525213403641, 2.2817570122473017, -0.2537209805247381)]
    [InlineData(21, 15, 2.863699932616508, -0.03719568844149235, -2.3754557833082177e-06)]
    [InlineData(28, 4, 0.34754265253833955, -1.7611052721653149, -0.3688667199461423)]
    [InlineData(16, 3, 2.347768615228666, -1.2836277193874863, 0.0025783065527587817)]
    [InlineData(2, 0, 0.48100085896298633, 0.7337600397444355, 0.42824452461458523)]
    [InlineData(29, 28, 1.3794564924140322, -2.055161176278025, 0.3291000555089361)]
    public void SphericalHarmonicY_Real_ExploredRelativeValues(int l, int m, double theta, double phi, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.SphericalHarmonicY(l, m, theta, phi).Real, relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(20, -11, 0.969525213403641, 2.2817570122473017, -0.008481013282674999)]
    [InlineData(3, -1, 1.2010049956594409, 0.5791693712238986, 0.05720966432206318)]
    [InlineData(21, 15, 2.863699932616508, -0.03719568844149235, 1.4824673748873812e-06)]
    [InlineData(28, 4, 0.34754265253833955, -1.7611052721653149, 0.3514585576059713)]
    [InlineData(7, 5, 2.9012706900530216, 1.4184137944041888, -0.0022841778817249223)]
    [InlineData(29, 28, 1.3794564924140322, -2.055161176278025, -0.507889157333853)]
    public void SphericalHarmonicY_Imaginary_ExploredRelativeValues(int l, int m, double theta, double phi, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.SphericalHarmonicY(l, m, theta, phi).Imaginary, relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(2, 0, 0.48100085896298633, 0.7337600397444355, 0.0)]
    public void SphericalHarmonicY_Imaginary_ExploredExactValues(int l, int m, double theta, double phi, double expected)
    {
        Assert.Equal(expected, XMath.Special.SphericalHarmonicY(l, m, theta, phi).Imaginary);
    }

    // 奇数次の原点は厳密に 0。偶数次の参照値は階乗比で生成し、最初のオーバーフローをまたぐ境界を確認する。
    [Trait("Category", "AuditRegression")]
    [Trait("Category", "CoreAuditFollowup20261102")]
    [Theory]
    [InlineData(268, 0.0, 4.602578455835625e+307)]
    [InlineData(269, 0.0, 0.0)]
    [InlineData(270, 0.0, double.NegativeInfinity)]
    [InlineData(271, 0.0, 0.0)]
    [InlineData(272, 0.0, double.PositiveInfinity)]
    [InlineData(273, 0.0, 0.0)]
    public void HermiteH_ExactOriginParity_FixedReferences(int n, double x, double expected)
    {
        double actual = XMath.Special.HermiteH(n, x);
        if (expected == 0 || double.IsInfinity(expected))
        {
            Assert.Equal(expected, actual);
        }
        else
        {
            SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 2e-12,
                absoluteTolerance: Math.Abs(expected) < 2.2250738585072014e-308 ? 2 * double.Epsilon : 0);
        }
    }

    // 奇数次の原点は厳密に 0。偶数次の参照値は階乗比で生成し、最初のオーバーフローをまたぐ境界を確認する。
    [Trait("Category", "AuditRegression")]
    [Trait("Category", "CoreAuditFollowup20261102")]
    [Theory]
    [InlineData(300, 0.0, 3.753274111571926e+306)]
    [InlineData(301, 0.0, 0.0)]
    [InlineData(302, 0.0, double.NegativeInfinity)]
    [InlineData(303, 0.0, 0.0)]
    [InlineData(304, 0.0, double.PositiveInfinity)]
    [InlineData(305, 0.0, 0.0)]
    public void HermiteHe_ExactOriginParity_FixedReferences(int n, double x, double expected)
    {
        double actual = XMath.Special.HermiteHe(n, x);
        if (expected == 0 || double.IsInfinity(expected))
        {
            Assert.Equal(expected, actual);
        }
        else
        {
            SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 2e-12,
                absoluteTolerance: Math.Abs(expected) < 2.2250738585072014e-308 ? 2 * double.Epsilon : 0);
        }
    }
}
