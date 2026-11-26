using System.Buffers.Binary;
using System.Security.Cryptography;
using Tuna;
using Tuna.Differentiation;
using Xunit.Abstractions;
using static Tuna.Tests.Differentiation.DifferentiationAssert;

namespace Tuna.Tests.Differentiation;

/// <summary>高精度で生成した初等関数の局所微分との照合</summary>
public class ElementaryReferenceTests(ITestOutputHelper output)
{
    // 実装を測る前に相対16εか4 ULPと決め、アンダーフロー付近を大きい絶対誤差で通さない。
    private const double RelativeTolerance = SixteenEpsilon;
    private const ulong UlpTolerance = 4;

    [Theory]
    [InlineData("Exp")]
    [InlineData("Log")]
    [InlineData("Sqrt")]
    [InlineData("Square")]
    [InlineData("Abs")]
    [InlineData("Expm1")]
    [InlineData("Log1p")]
    [InlineData("Logistic")]
    [InlineData("LogLogistic")]
    [InlineData("Logit")]
    [InlineData("Softplus")]
    [InlineData("Log1mExp")]
    [InlineData("Sin")]
    [InlineData("Cos")]
    [InlineData("Tan")]
    [InlineData("Tanh")]
    [InlineData("Atan")]
    public void LocalDerivativesMatchHighPrecisionReferences(string function)
    {
        var tape = new Tape();
        using var fingerprint = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        Span<byte> bits = stackalloc byte[16];
        int count = 0;
        int subnormalCount = 0;
        double maximumRelativeError = 0;
        double maximumRelativeInput = double.NaN;
        ulong maximumUlps = 0;
        foreach (var reference in ElementaryReferences.Rows.Where(row => row.Function == function))
        {
            tape.Reset();
            var x = tape.Leaf(reference.X);
            var y = Evaluate(function, x);
            tape.Backward(y);
            double actual = tape.GetGradient(x);
            BinaryPrimitives.WriteInt64LittleEndian(bits, BitConverter.DoubleToInt64Bits(y.Value));
            BinaryPrimitives.WriteInt64LittleEndian(bits[8..], BitConverter.DoubleToInt64Bits(actual));
            fingerprint.AppendData(bits);
            double expected = reference.Derivative;
            count++;
            if (double.IsInfinity(expected))
            {
                Assert.True(actual == expected, $"{function}({reference.X:R}): expected {expected:R}, actual {actual:R}");
                continue;
            }

            Assert.True(double.IsFinite(actual), $"{function}({reference.X:R}): expected {expected:R}, actual {actual:R}");
            double relativeError = expected == 0 ? (actual == 0 ? 0 : double.PositiveInfinity) : Math.Abs((actual - expected) / expected);
            ulong ulps = UlpDistance(actual, expected);
            maximumUlps = Math.Max(maximumUlps, ulps);
            if (double.IsNormal(expected) && (double.IsNaN(maximumRelativeInput) || relativeError > maximumRelativeError))
            {
                maximumRelativeError = relativeError;
                maximumRelativeInput = reference.X;
            }
            else if (double.IsSubnormal(expected))
            {
                subnormalCount++;
            }

            Assert.True(relativeError <= RelativeTolerance || ulps <= UlpTolerance,
                $"{function}({reference.X:R}): expected {expected:R}, actual {actual:R}, relative error {relativeError:E17}, ULP distance {ulps}");
        }

        Assert.True(count > 0);
        output.WriteLine($"{function}: value/gradient fingerprint={Convert.ToHexString(fingerprint.GetHashAndReset())}");
        output.WriteLine(FormattableString.Invariant(
            $"{function}: samples={count}, max relative error (normal references)={maximumRelativeError:E17} at x={maximumRelativeInput:R}, max ULP distance (all finite references)={maximumUlps}, subnormal references={subnormalCount}"));
    }

    private static ScalarNode Evaluate(string function, ScalarNode x) => function switch
    {
        "Exp" => XMath.Exp(x),
        "Log" => XMath.Log(x),
        "Sqrt" => XMath.Sqrt(x),
        "Square" => XMath.Square(x),
        "Abs" => XMath.Abs(x),
        "Expm1" => XMath.Expm1(x),
        "Log1p" => XMath.Log1p(x),
        "Logistic" => XMath.Logistic(x),
        "LogLogistic" => XMath.LogLogistic(x),
        "Logit" => XMath.Logit(x),
        "Softplus" => XMath.Softplus(x),
        "Log1mExp" => XMath.Log1mExp(x),
        "Sin" => XMath.Sin(x),
        "Cos" => XMath.Cos(x),
        "Tan" => XMath.Tan(x),
        "Tanh" => XMath.Tanh(x),
        "Atan" => XMath.Atan(x),
        _ => throw new ArgumentOutOfRangeException(nameof(function))
    };
}
