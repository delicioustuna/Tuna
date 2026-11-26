using System.Numerics;
using Tuna;

namespace Tuna.Tests;

// mpmath 1.4.1による独立参照値。binary64入力を正確に変換し、
// 90/160桁（正規分布は490/560桁）でbinary64への丸め一致を確認。
// 生成式と選定理由: tools/SpecialFunctionReferences/generate_remaining_accuracy_references.py
// 通常テストはC#の固定値と対象APIの直接呼び出しだけで完結する。
[Trait("SpecialFunction", "ComplexGamma")]
public class ComplexGammaTests
{

    // 反射・漸近領域・複素対数の枝・極の近く。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(1.0, 0.0, 1.0, 0.0, 0.0, 0.0)]
    [InlineData(0.5, 1.0, 0.3006946172606558, -0.4249678794331238, 0.0, 0.0)]
    [InlineData(-0.5, 0.1, -3.3926661426873252, -0.01284755039222288, 0.0, 0.0)]
    [InlineData(-3.0, 2.0, -0.0017020848023722272, -0.002895711894544827, 0.0, 0.0)]
    [InlineData(0.1, -4.0, 0.0016565414245271815, -0.0021188384833955186, 0.0, 0.0)]
    [InlineData(5.0, 4.0, 4.810416066354929, 0.8365033943755388, 0.0, 0.0)]
    [InlineData(20.0, 30.0, -1453876687.5534809, 1163777777.8031573, 0.0, 0.0)]
    [InlineData(-10.0, -0.1, 6.319850257556926e-07, 2.63755023684929e-06, 0.0, 0.0)]
    [InlineData(-2.0, 1e-08, 0.4613921675492335, -49999999.99999999, 5.960464477539063e-08, 0.0)]
    public void Gamma_MatchesIndependentReferences(double real, double imaginary, double expectedReal, double expectedImaginary, double absoluteReal, double absoluteImaginary)
    {
        Complex actual = XMath.Special.Gamma(new Complex(real, imaginary));
        if (expectedReal == 0 || double.IsInfinity(expectedReal))
        {
            Assert.Equal(expectedReal, actual.Real);
        }
        else
        {
            SpecialFunctionAssert.Close(expectedReal, actual.Real, relativeTolerance: 2e-12, absoluteTolerance: absoluteReal);
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
            SpecialFunctionAssert.Close(expectedImaginary, actual.Imaginary, relativeTolerance: 2e-12, absoluteTolerance: absoluteImaginary);
            if (Math.Abs(expectedImaginary) < 2.2250738585072014e-308)
            {
                Assert.Equal(Math.Sign(expectedImaginary), Math.Sign(actual.Imaginary));
            }
        }
    }

    // 各引数のNaNを独立に検査。
    [Theory]
    [InlineData(double.NaN, 0.0)]
    [InlineData(1.0, double.NaN)]
    public void Gamma_RejectsNaN(double real, double imaginary)
    {
        Complex actual = XMath.Special.Gamma(new Complex(real, imaginary));
        Assert.True(double.IsNaN(actual.Real));
        Assert.True(double.IsNaN(actual.Imaginary));
    }

    // 反射・漸近領域・複素対数の枝・極の近く。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(1.0, 0.0, 1.0, 0.0, 0.0, 0.0)]
    [InlineData(0.5, 1.0, 1.1095130202521408, 1.56806064476794, 0.0, 0.0)]
    [InlineData(-0.5, 0.1, -0.29474920839332414, 0.0011161738728885102, 0.0, 0.0)]
    [InlineData(-3.0, 2.0, -150.86408326070048, 256.6610780783385, 0.0, 0.0)]
    [InlineData(0.1, -4.0, 229.00631058171643, 292.91593715470367, 0.0, 0.0)]
    [InlineData(5.0, 4.0, 0.201780546057769, -0.035088464151121435, 0.0, 0.0)]
    [InlineData(20.0, 30.0, -4.192098087851003e-10, -3.355628877454329e-10, 0.0, 0.0)]
    [InlineData(-10.0, -0.1, 85913.33176289636, -358553.95192122215, 0.0, 0.0)]
    [InlineData(-2.0, 1e-08, 1.8455686701969347e-16, 2.0000000000000004e-08, 2.6469779601696886e-23, 0.0)]
    public void ReciprocalGamma_MatchesIndependentReferences(double real, double imaginary, double expectedReal, double expectedImaginary, double absoluteReal, double absoluteImaginary)
    {
        Complex actual = XMath.Special.ReciprocalGamma(new Complex(real, imaginary));
        if (expectedReal == 0 || double.IsInfinity(expectedReal))
        {
            Assert.Equal(expectedReal, actual.Real);
        }
        else
        {
            SpecialFunctionAssert.Close(expectedReal, actual.Real, relativeTolerance: 2e-12, absoluteTolerance: absoluteReal);
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
            SpecialFunctionAssert.Close(expectedImaginary, actual.Imaginary, relativeTolerance: 2e-12, absoluteTolerance: absoluteImaginary);
            if (Math.Abs(expectedImaginary) < 2.2250738585072014e-308)
            {
                Assert.Equal(Math.Sign(expectedImaginary), Math.Sign(actual.Imaginary));
            }
        }
    }

    // 各引数のNaNを独立に検査。
    [Theory]
    [InlineData(double.NaN, 0.0)]
    [InlineData(1.0, double.NaN)]
    public void ReciprocalGamma_RejectsNaN(double real, double imaginary)
    {
        Complex actual = XMath.Special.ReciprocalGamma(new Complex(real, imaginary));
        Assert.True(double.IsNaN(actual.Real));
        Assert.True(double.IsNaN(actual.Imaginary));
    }

    // 反射・漸近領域・複素対数の枝・極の近く。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(1.0, 0.0, 0.0, 0.0)]
    [InlineData(0.5, 1.0, -0.6527906442043729, -0.9550077243425691)]
    [InlineData(-0.5, 0.1, 1.2216232551552817, -3.1378058120793657)]
    [InlineData(-3.0, 2.0, -5.696140372778611, -8.385378486515382)]
    [InlineData(0.1, -4.0, -5.918386444788175, -0.9072420726485169)]
    [InlineData(5.0, 4.0, 1.5856790792301796, 6.4553578194337184)]
    [InlineData(20.0, 30.0, 21.345074493863446, 96.71434768953618)]
    [InlineData(-10.0, -0.1, -12.817747216699871, 32.75154609552116)]
    [InlineData(-2.0, 1e-08, 17.72753356339242, -7.853981624746639)]
    public void LogGamma_MatchesIndependentReferences(double real, double imaginary, double expectedReal, double expectedImaginary)
    {
        Complex actual = XMath.Special.LogGamma(new Complex(real, imaginary));
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
    [InlineData(1.0, double.NaN)]
    public void LogGamma_RejectsNaN(double real, double imaginary)
    {
        Complex actual = XMath.Special.LogGamma(new Complex(real, imaginary));
        Assert.True(double.IsNaN(actual.Real));
        Assert.True(double.IsNaN(actual.Imaginary));
    }

    // 反射・漸近領域・複素対数の枝・極の近く。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(1.0, 0.0, -0.5772156649015329, 0.0, 0.0, 0.0)]
    [InlineData(0.5, 1.0, -0.051761650994412545, 1.5649405178158793, 0.0, 0.0)]
    [InlineData(-0.5, 0.1, 0.040619541255299124, 0.8624770629534108, 0.0, 0.0)]
    [InlineData(-3.0, 2.0, 1.3953607461432083, 2.624653436488384, 0.0, 0.0)]
    [InlineData(0.1, -4.0, 1.3887156692844709, -1.6709869529239543, 0.0, 0.0)]
    [InlineData(5.0, 4.0, 1.7953598173334109, 0.7255022091860445, 0.0, 0.0)]
    [InlineData(20.0, 30.0, 3.57739211539108, 0.9943913598840195, 0.0, 0.0)]
    [InlineData(-10.0, -0.1, 2.351797836200318, -10.317325967996728, 0.0, 0.0)]
    [InlineData(-2.0, 1e-08, 0.9227843350984671, 100000000.00000003, 1.1920928955078125e-07, 0.0)]
    public void Digamma_MatchesIndependentReferences(double real, double imaginary, double expectedReal, double expectedImaginary, double absoluteReal, double absoluteImaginary)
    {
        Complex actual = XMath.Special.Digamma(new Complex(real, imaginary));
        if (expectedReal == 0 || double.IsInfinity(expectedReal))
        {
            Assert.Equal(expectedReal, actual.Real);
        }
        else
        {
            SpecialFunctionAssert.Close(expectedReal, actual.Real, relativeTolerance: 2e-12, absoluteTolerance: absoluteReal);
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
            SpecialFunctionAssert.Close(expectedImaginary, actual.Imaginary, relativeTolerance: 2e-12, absoluteTolerance: absoluteImaginary);
            if (Math.Abs(expectedImaginary) < 2.2250738585072014e-308)
            {
                Assert.Equal(Math.Sign(expectedImaginary), Math.Sign(actual.Imaginary));
            }
        }
    }

    // 各引数のNaNを独立に検査。
    [Theory]
    [InlineData(double.NaN, 0.0)]
    [InlineData(1.0, double.NaN)]
    public void Digamma_RejectsNaN(double real, double imaginary)
    {
        Complex actual = XMath.Special.Digamma(new Complex(real, imaginary));
        Assert.True(double.IsNaN(actual.Real));
        Assert.True(double.IsNaN(actual.Imaginary));
    }

    [Theory]
    [InlineData(.3, .7)]
    [InlineData(-2.3, 1.0)]
    [InlineData(10.0, 5.0)]
    public void RecurrenceAndConjugation(double real, double imaginary)
    {
        Complex z = new(real, imaginary);
        Complex expected = z * XMath.Special.Gamma(z);
        Complex actual = XMath.Special.Gamma(z + 1);
        SpecialFunctionAssert.Close(expected.Real, actual.Real, relativeTolerance: 2e-12);
        SpecialFunctionAssert.Close(expected.Imaginary, actual.Imaginary, relativeTolerance: 2e-12);
        expected = Complex.Conjugate(XMath.Special.Gamma(z));
        actual = XMath.Special.Gamma(Complex.Conjugate(z));
        SpecialFunctionAssert.Close(expected.Real, actual.Real, relativeTolerance: 2e-12);
        SpecialFunctionAssert.Close(expected.Imaginary, actual.Imaginary, relativeTolerance: 2e-12);
    }

    [Fact]
    public void ReciprocalGammaVanishesAtPoles()
    {
        Assert.Equal(Complex.Zero, XMath.Special.ReciprocalGamma(Complex.Zero));
        Assert.Equal(Complex.Zero, XMath.Special.ReciprocalGamma(new Complex(-2, 0)));
    }

    // 負の整数極の上下近傍および原点近傍での参照値検証（小さい成分単独の相対精度は要求しない）。
    [Trait("Category", "AuditRegression")]
    [Trait("Category", "CoreAuditFollowup20261102")]
    [Theory]
    [InlineData(-1.0, -1e-150, 345.38776394910684, 4.71238898038469)]
    [InlineData(-1.0, 1e-150, 345.38776394910684, -4.71238898038469)]
    [InlineData(-1.0, -1e-160, 368.4136148790473, 4.71238898038469)]
    [InlineData(-1.0, 1e-160, 368.4136148790473, -4.71238898038469)]
    [InlineData(-1.0, -1e-200, 460.51701859880916, 4.71238898038469)]
    [InlineData(-1.0, 1e-200, 460.51701859880916, -4.71238898038469)]
    [InlineData(-2.0, -1e-150, 344.6946167685469, 7.853981633974483)]
    [InlineData(-2.0, 1e-150, 344.6946167685469, -7.853981633974483)]
    [InlineData(-2.0, -1e-160, 367.72046769848737, 7.853981633974483)]
    [InlineData(-2.0, 1e-160, 367.72046769848737, -7.853981633974483)]
    [InlineData(-2.0, -1e-200, 459.8238714182492, 7.853981633974483)]
    [InlineData(-2.0, 1e-200, 459.8238714182492, -7.853981633974483)]
    [InlineData(0.0, -1e-200, 460.51701859880916, 1.5707963267948966)]
    [InlineData(0.0, 1e-200, 460.51701859880916, -1.5707963267948966)]
    public void LogGamma_NearComplexPoles_FixedReferences(double real, double imaginary, double expectedReal, double expectedImaginary)
    {
        Complex actual = XMath.Special.LogGamma(new Complex(real, imaginary));
        SpecialFunctionAssert.ComplexClose(new Complex(expectedReal, expectedImaginary), actual,
            relativeTolerance: 2e-12, absoluteTolerance: 2 * double.Epsilon);
    }

    // 負の整数極の上下近傍および原点近傍での参照値検証。
    [Trait("Category", "AuditRegression")]
    [Trait("Category", "CoreAuditFollowup20261102")]
    [Theory]
    [InlineData(-1.0, -1e-150, -0.42278433509846713, -1e+150)]
    [InlineData(-1.0, 1e-150, -0.42278433509846713, 1e+150)]
    [InlineData(-1.0, -1e-160, -0.42278433509846713, -1e+160)]
    [InlineData(-1.0, 1e-160, -0.42278433509846713, 1e+160)]
    [InlineData(-1.0, -1e-200, -0.42278433509846713, -1e+200)]
    [InlineData(-1.0, 1e-200, -0.42278433509846713, 1e+200)]
    [InlineData(-2.0, -1e-150, 0.46139216754923357, 5e+149)]
    [InlineData(-2.0, 1e-150, 0.46139216754923357, -5e+149)]
    [InlineData(-2.0, -1e-160, 0.46139216754923357, 5e+159)]
    [InlineData(-2.0, 1e-160, 0.46139216754923357, -5e+159)]
    [InlineData(-2.0, -1e-200, 0.46139216754923357, 5e+199)]
    [InlineData(-2.0, 1e-200, 0.46139216754923357, -5e+199)]
    [InlineData(0.0, -1e-200, -0.5772156649015329, 1e+200)]
    [InlineData(0.0, 1e-200, -0.5772156649015329, -1e+200)]
    public void Gamma_NearComplexPoles_FixedReferences(double real, double imaginary, double expectedReal, double expectedImaginary)
    {
        Complex actual = XMath.Special.Gamma(new Complex(real, imaginary));
        SpecialFunctionAssert.ComplexClose(new Complex(expectedReal, expectedImaginary), actual,
            relativeTolerance: 2e-12, absoluteTolerance: 2 * double.Epsilon);
    }

    // 負の整数極の上下近傍および原点近傍での参照値検証。
    [Trait("Category", "AuditRegression")]
    [Trait("Category", "CoreAuditFollowup20261102")]
    [Theory]
    [InlineData(-1.0, -1e-150, -4.227843350984671e-301, 1e-150)]
    [InlineData(-1.0, 1e-150, -4.227843350984671e-301, -1e-150)]
    [InlineData(-1.0, -1e-160, -4.23e-321, 1e-160)]
    [InlineData(-1.0, 1e-160, -4.23e-321, -1e-160)]
    [InlineData(-1.0, -1e-200, -0.0, 1e-200)]
    [InlineData(-1.0, 1e-200, -0.0, -1e-200)]
    [InlineData(-2.0, -1e-150, 1.8455686701969343e-300, -2e-150)]
    [InlineData(-2.0, 1e-150, 1.8455686701969343e-300, 2e-150)]
    [InlineData(-2.0, -1e-160, 1.8453e-320, -2e-160)]
    [InlineData(-2.0, 1e-160, 1.8453e-320, 2e-160)]
    [InlineData(-2.0, -1e-200, 0.0, -2e-200)]
    [InlineData(-2.0, 1e-200, 0.0, 2e-200)]
    [InlineData(0.0, -1e-200, -0.0, -1e-200)]
    [InlineData(0.0, 1e-200, -0.0, 1e-200)]
    public void ReciprocalGamma_NearComplexPoles_FixedReferences(double real, double imaginary, double expectedReal, double expectedImaginary)
    {
        Complex actual = XMath.Special.ReciprocalGamma(new Complex(real, imaginary));
        SpecialFunctionAssert.ComplexClose(new Complex(expectedReal, expectedImaginary), actual,
            relativeTolerance: 2e-12, absoluteTolerance: 2 * double.Epsilon);
    }

    // 負の整数極の上下近傍および原点近傍での参照値検証。
    [Trait("Category", "AuditRegression")]
    [Trait("Category", "CoreAuditFollowup20261102")]
    [Theory]
    [InlineData(-1.0, -1e-150, 0.42278433509846713, -1e+150)]
    [InlineData(-1.0, 1e-150, 0.42278433509846713, 1e+150)]
    [InlineData(-1.0, -1e-160, 0.42278433509846713, -1e+160)]
    [InlineData(-1.0, 1e-160, 0.42278433509846713, 1e+160)]
    [InlineData(-1.0, -1e-200, 0.42278433509846713, -1e+200)]
    [InlineData(-1.0, 1e-200, 0.42278433509846713, 1e+200)]
    [InlineData(-2.0, -1e-150, 0.9227843350984671, -1e+150)]
    [InlineData(-2.0, 1e-150, 0.9227843350984671, 1e+150)]
    [InlineData(-2.0, -1e-160, 0.9227843350984671, -1e+160)]
    [InlineData(-2.0, 1e-160, 0.9227843350984671, 1e+160)]
    [InlineData(-2.0, -1e-200, 0.9227843350984671, -1e+200)]
    [InlineData(-2.0, 1e-200, 0.9227843350984671, 1e+200)]
    [InlineData(0.0, -1e-200, -0.5772156649015329, -1e+200)]
    [InlineData(0.0, 1e-200, -0.5772156649015329, 1e+200)]
    public void Digamma_NearComplexPoles_FixedReferences(double real, double imaginary, double expectedReal, double expectedImaginary)
    {
        Complex actual = XMath.Special.Digamma(new Complex(real, imaginary));
        SpecialFunctionAssert.ComplexClose(new Complex(expectedReal, expectedImaginary), actual,
            relativeTolerance: 2e-12, absoluteTolerance: 2 * double.Epsilon);
    }

    // 絶対値が double 上限を超えても、実部・虚部が個別に有限なら有限の 2 成分を返すことを検証する。
    [Trait("Category", "AuditRegression")]
    [Trait("Category", "CoreAuditFollowup20261102")]
    [Theory]
    [InlineData(171.6, -0.15, 1.1369799807040395e+308, -1.1054477905265567e+308)]
    [InlineData(171.6, 0.15, 1.1369799807040395e+308, 1.1054477905265567e+308)]
    [InlineData(171.65, -0.1, 1.7855683205540461e+308, -1.0087698838255427e+308)]
    [InlineData(171.65, 0.1, 1.7855683205540461e+308, 1.0087698838255427e+308)]
    [InlineData(171.65, -0.15, 1.4702799105179313e+308, -1.4296295763255581e+308)]
    [InlineData(171.65, 0.15, 1.4702799105179313e+308, 1.4296295763255581e+308)]
    [InlineData(171.65, -0.2, 1.0583303607512985e+308, -1.7564369031954672e+308)]
    [InlineData(171.65, 0.2, 1.0583303607512985e+308, 1.7564369031954672e+308)]
    public void Gamma_FiniteCartesianComponents_FixedReferences(double real, double imaginary, double expectedReal, double expectedImaginary)
    {
        Complex actual = XMath.Special.Gamma(new Complex(real, imaginary));
        SpecialFunctionAssert.ComplexClose(new Complex(expectedReal, expectedImaginary), actual,
            relativeTolerance: 2e-12, absoluteTolerance: 2 * double.Epsilon);
    }

    // 絶対値が double 上限を超えても、実部・虚部が個別に有限なら有限の 2 成分を返すことを検証する。
    [Trait("Category", "AuditRegression")]
    [Trait("Category", "CoreAuditFollowup20261102")]
    [Theory]
    [InlineData(0.5, -452.5, -1.196188116102179e+308, 1.545844219856864e+308)]
    [InlineData(0.5, 452.5, -1.196188116102179e+308, -1.545844219856864e+308)]
    public void ReciprocalGamma_FiniteCartesianComponents_FixedReferences(double real, double imaginary, double expectedReal, double expectedImaginary)
    {
        Complex actual = XMath.Special.ReciprocalGamma(new Complex(real, imaginary));
        SpecialFunctionAssert.ComplexClose(new Complex(expectedReal, expectedImaginary), actual,
            relativeTolerance: 2e-12, absoluteTolerance: 2 * double.Epsilon);
    }

    // 負の実軸上の分岐切断における ±0 規約を検証する（-0.0 は実行時に明示生成する）。
    [Trait("Category", "AuditRegression")]
    [Trait("Category", "CoreAuditFollowup20261102")]
    [Theory]
    [InlineData(-0.5, true, 1.2655121234846454, 3.141592653589793)]
    [InlineData(-0.5, false, 1.2655121234846454, -3.141592653589793)]
    [InlineData(-1.5, true, 0.860047015376481, 6.283185307179586)]
    [InlineData(-1.5, false, 0.860047015376481, -6.283185307179586)]
    [InlineData(-10.25, true, -14.20399790093109, 34.55751918948773)]
    [InlineData(-10.25, false, -14.20399790093109, -34.55751918948773)]
    public void LogGamma_SignedZeroBranchCut_FixedReferences(double real, bool below, double expectedReal, double expectedImaginary)
    {
        Complex actual = XMath.Special.LogGamma(new Complex(real, Math.CopySign(0.0, below ? -1.0 : 1.0)));
        SpecialFunctionAssert.ComplexClose(new Complex(expectedReal, expectedImaginary), actual,
            relativeTolerance: 2e-12, absoluteTolerance: 2 * double.Epsilon);
    }
}
