using System.Numerics;
using Tuna;

namespace Tuna.Tests;

// mpmath 1.4.1による独立参照値。binary64入力を正確に変換し、
// 90/160桁（正規分布は490/560桁）でbinary64への丸め一致を確認。
// 生成式と選定理由: tools/SpecialFunctionReferences/generate_remaining_accuracy_references.py
// 通常テストはC#の固定値と対象APIの直接呼び出しだけで完結する。
[Trait("SpecialFunction", "Faddeeva")]
public class FaddeevaTests
{

    // 上下半平面・実軸・連分数/積分の切替。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0.0, 0.0, 1.0, 0.0)]
    [InlineData(1.0, 0.0, 0.36787944117144233, 0.6071577058413937)]
    [InlineData(0.0, 1.0, 0.427583576155807, 0.0)]
    [InlineData(0.5, 0.2, 0.6632226250660399, 0.3507512817259095)]
    [InlineData(1.0, -1.0, -1.1370378783511974, 2.026813791854195)]
    [InlineData(5.0, 0.1, 0.002406911716942712, 0.11519442455072769)]
    [InlineData(10.0, 2.0, 0.011001556705733516, 0.05447181709865651)]
    [InlineData(-2.0, 3.0, 0.13075746966984858, -0.08111265047745665)]
    [InlineData(6.999999999999999, 0.3, 0.0035587272611619938, 0.08128995781449772)]
    [InlineData(7.0, 0.3, 0.003558727261161993, 0.0812899578144977)]
    [InlineData(7.000000000000001, 0.3, 0.003558727261161992, 0.08128995781449769)]
    [InlineData(0.3, 1.3999999999999997, 0.33105362259582194, 0.052936024476872186)]
    [InlineData(0.3, 1.4, 0.3310536225958219, 0.05293602447687218)]
    [InlineData(0.3, 1.4000000000000001, 0.33105362259582183, 0.052936024476872165)]
    public void Faddeeva_MatchesIndependentReferences(double real, double imaginary, double expectedReal, double expectedImaginary)
    {
        Complex actual = XMath.Special.Faddeeva(new Complex(real, imaginary));
        if (expectedReal == 0 || double.IsInfinity(expectedReal))
        {
            Assert.Equal(expectedReal, actual.Real);
        }
        else
        {
            SpecialFunctionAssert.Close(expectedReal, actual.Real, relativeTolerance: 2e-12);
            if (Math.Abs(expectedReal) < 2.2250738585072014e-308)
            {
                Assert.Equal(Math.Sign(expectedReal), Math.Sign(actual.Real));
            }
        }

        if (expectedImaginary == 0 || double.IsInfinity(expectedImaginary))
        {
            Assert.Equal(expectedImaginary, actual.Imaginary);
        }
        else
        {
            SpecialFunctionAssert.Close(expectedImaginary, actual.Imaginary, relativeTolerance: 2e-12);
            if (Math.Abs(expectedImaginary) < 2.2250738585072014e-308)
            {
                Assert.Equal(Math.Sign(expectedImaginary), Math.Sign(actual.Imaginary));
            }
        }
    }

    // 各引数のNaNを独立に検査。
    [Theory]
    [InlineData(double.NaN, 0.0)]
    [InlineData(0.0, double.NaN)]
    public void Faddeeva_RejectsNaN(double real, double imaginary)
    {
        Complex actual = XMath.Special.Faddeeva(new Complex(real, imaginary));
        Assert.True(double.IsNaN(actual.Real));
        Assert.True(double.IsNaN(actual.Imaginary));
    }

    // Voigtの中心・裾・Gaussian極限。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0.0, 1.0, 1.0, 0.2087092805203677)]
    [InlineData(1.0, 1.0, 0.5, 0.20017963759083915)]
    [InlineData(-3.0, 0.2, 2.0, 0.049237807489497826)]
    [InlineData(10.0, 1.0, 0.01, 3.2837310592837986e-05)]
    [InlineData(0.0, 2.0, 0.0, 0.19947114020071635)]
    public void VoigtProfile_MatchesIndependentReferences(double x, double sigma, double gamma, double expected)
    {
        double actual = XMath.Special.VoigtProfile(x, sigma, gamma);
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
    [InlineData(double.NaN, 1.0, 1.0)]
    [InlineData(0.0, double.NaN, 1.0)]
    [InlineData(0.0, 1.0, double.NaN)]
    public void VoigtProfile_RejectsNaN(double x, double sigma, double gamma)
    {
        double actual = XMath.Special.VoigtProfile(x, sigma, gamma);
        Assert.True(double.IsNaN(actual));
    }

    [Theory]
    [InlineData(.3, .7)]
    [InlineData(2.0, .1)]
    public void ReflectionAndConjugation(double real, double imaginary)
    {
        Complex z = new(real, imaginary);
        Complex sum = XMath.Special.Faddeeva(z) + XMath.Special.Faddeeva(-z);
        Complex expected = 2 * Complex.Exp(-z * z);
        SpecialFunctionAssert.Close(expected.Real, sum.Real, relativeTolerance: 2e-12);
        SpecialFunctionAssert.Close(expected.Imaginary, sum.Imaginary, relativeTolerance: 2e-12);
        Complex actual = XMath.Special.Faddeeva(-Complex.Conjugate(z));
        expected = Complex.Conjugate(XMath.Special.Faddeeva(z));
        SpecialFunctionAssert.Close(expected.Real, actual.Real, relativeTolerance: 2e-12);
        SpecialFunctionAssert.Close(expected.Imaginary, actual.Imaginary, relativeTolerance: 2e-12);
    }

    [Fact]
    public void VoigtLimitsAndDomain()
    {
        SpecialFunctionAssert.Close(2 / (5 * Math.PI), XMath.Special.VoigtProfile(1, 0, 2), relativeTolerance: 2e-12);
        SpecialFunctionAssert.Close(XMath.Special.NormalPdf(1), XMath.Special.VoigtProfile(1, 1, 0), relativeTolerance: 2e-12);
        Assert.Equal(0, XMath.Special.VoigtProfile(double.PositiveInfinity, 1, 1));
        Assert.True(double.IsNaN(XMath.Special.VoigtProfile(1, -1, 1)));
        Assert.True(double.IsNaN(XMath.Special.VoigtProfile(1, 1, -1)));
        Assert.True(double.IsNaN(XMath.Special.VoigtProfile(1, 0, 0)));
    }

    // x = -y では反射項の絶対値が 2。位相の丸め残りが大きい領域でも振幅が崩れないことを検証する。
    [Trait("Category", "AuditRegression")]
    [Trait("Category", "CoreAuditFollowup20261102")]
    [Theory]
    [InlineData(-1000.1, -1000.1, -1.8971077869375617, 0.6337937861062585)]
    [InlineData(1000.1, -1000.1, -1.8971077869375617, -0.6337937861062585)]
    [InlineData(-10000000.1, -10000000.1, 1.0751829426480446, 1.6864108289596718)]
    [InlineData(10000000.1, -10000000.1, 1.0751829426480446, -1.6864108289596718)]
    [InlineData(-100000000.1, -100000000.1, 1.8464018118286545, -0.7686353772727456)]
    [InlineData(100000000.1, -100000000.1, 1.8464018118286545, 0.7686353772727456)]
    [InlineData(-1000000000.1, -1000000000.1, 1.923735737790767, 0.5470290757835763)]
    [InlineData(1000000000.1, -1000000000.1, 1.923735737790767, -0.5470290757835763)]
    public void Faddeeva_LargeReflectionPhase_FixedReferences(double real, double imaginary, double expectedReal, double expectedImaginary)
    {
        Complex actual = XMath.Special.Faddeeva(new Complex(real, imaginary));
        SpecialFunctionAssert.ComplexClose(new Complex(expectedReal, expectedImaginary), actual,
            relativeTolerance: 2e-12, absoluteTolerance: 2 * double.Epsilon);
    }

    // 大引数上半平面 w(z) ≈ i/(√π z)（複素除算の分母の和のみオーバーフローする点を含む）。
    [Trait("Category", "AuditRegression")]
    [Trait("Category", "CoreAuditFollowup20261102")]
    [Theory]
    [InlineData(-1e+307, 1e+307, 2.8209479177387815e-308, -2.8209479177387815e-308)]
    [InlineData(1e+307, 1e+307, 2.8209479177387815e-308, 2.8209479177387815e-308)]
    [InlineData(-1e+308, 1e+308, 2.82094791773878e-309, -2.82094791773878e-309)]
    [InlineData(1e+308, 1e+308, 2.82094791773878e-309, 2.82094791773878e-309)]
    [InlineData(-1.7e+308, 1.7e+308, 1.659381128081635e-309, -1.659381128081635e-309)]
    [InlineData(1.7e+308, 1.7e+308, 1.659381128081635e-309, 1.659381128081635e-309)]
    [InlineData(-1e+308, 2e+307, 1.08497996836107e-309, -5.42489984180535e-309)]
    [InlineData(1e+308, 2e+307, 1.08497996836107e-309, 5.42489984180535e-309)]
    public void Faddeeva_LargeUpperHalfPlane_FixedReferences(double real, double imaginary, double expectedReal, double expectedImaginary)
    {
        Complex actual = XMath.Special.Faddeeva(new Complex(real, imaginary));
        SpecialFunctionAssert.ComplexClose(new Complex(expectedReal, expectedImaginary), actual,
            relativeTolerance: 2e-12, absoluteTolerance: 2 * double.Epsilon);
    }

    // γ = 0 の閉形式（標準密度がアンダーフローしても小さい σ で割った密度が表現可能な領域を含む）。
    [Trait("Category", "AuditRegression")]
    [Trait("Category", "CoreAuditFollowup20261102")]
    [Theory]
    [InlineData(8.0, 1.0, 0.0, 5.052271083536892e-15)]
    [InlineData(38.0, 1.0, 0.0, 1.097221052e-314)]
    [InlineData(40.0, 1.0, 0.0, 0.0)]
    [InlineData(8e-100, 1e-100, 0.0, 5.052271083536892e+85)]
    [InlineData(3.8e-99, 1e-100, 0.0, 1.0972210520077622e-214)]
    [InlineData(4e-99, 1e-100, 0.0, 1.4632702508383032e-248)]
    [InlineData(8e-200, 1e-200, 0.0, 5.052271083536892e+185)]
    [InlineData(3.8e-199, 1e-200, 0.0, 1.0972210520076413e-114)]
    [InlineData(4e-199, 1e-200, 0.0, 1.4632702508383033e-148)]
    public void VoigtProfile_ScaledGaussianLimit_FixedReferences(double x, double sigma, double gamma, double expected)
    {
        double actual = XMath.Special.VoigtProfile(x, sigma, gamma);
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

    // σ = 0 の閉形式 γ/(π(x² + γ²))（大きな幅でも非零の非正規化数を保持する）。
    [Trait("Category", "AuditRegression")]
    [Trait("Category", "CoreAuditFollowup20261102")]
    [Theory]
    [InlineData(0.0, 0.0, 1e+100, 3.183098861837907e-101)]
    [InlineData(1e+100, 0.0, 1e+100, 1.5915494309189535e-101)]
    [InlineData(0.0, 0.0, 1e+308, 3.183098861837906e-309)]
    [InlineData(1e+308, 0.0, 1e+308, 1.591549430918953e-309)]
    [InlineData(0.0, 0.0, 1.7e+308, 1.87241109519877e-309)]
    [InlineData(1.7e+308, 0.0, 1.7e+308, 9.36205547599384e-310)]
    public void VoigtProfile_ScaledLorentzianLimit_FixedReferences(double x, double sigma, double gamma, double expected)
    {
        double actual = XMath.Special.VoigtProfile(x, sigma, gamma);
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

    // スケール則 V(sx; sσ, sγ) = V(x; σ, γ)/s（σ√2 の中間オーバーフロー域を含む）。
    [Trait("Category", "AuditRegression")]
    [Trait("Category", "CoreAuditFollowup20261102")]
    [Theory]
    [InlineData(0.0, 1e+308, 1.0, 3.989422804014326e-309)]
    [InlineData(1e+308, 1e+308, 1.0, 2.41970724519143e-309)]
    [InlineData(0.0, 1e+308, 1e+308, 2.087092805203676e-309)]
    [InlineData(1e+308, 1e+308, 1e+308, 1.657956626891665e-309)]
    [InlineData(0.0, 1.7e+308, 1.0, 2.34671929647902e-309)]
    [InlineData(1.7e+308, 1.7e+308, 1.0, 1.423357203053783e-309)]
    [InlineData(0.0, 1.7e+308, 1e+308, 1.55226445908657e-309)]
    [InlineData(1.7e+308, 1.7e+308, 1e+308, 1.13830203953398e-309)]
    public void VoigtProfile_ScaledMixedWidths_FixedReferences(double x, double sigma, double gamma, double expected)
    {
        double actual = XMath.Special.VoigtProfile(x, sigma, gamma);
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
