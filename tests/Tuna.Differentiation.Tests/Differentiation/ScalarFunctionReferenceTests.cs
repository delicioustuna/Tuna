using System.Buffers.Binary;
using System.Security.Cryptography;
using Tuna.Differentiation;
using Tuna.Tests.ReferenceData;
using Xunit.Abstractions;

namespace Tuna.Tests.Differentiation;

/// <summary>多入力の初等関数と特殊関数の局所微分を高精度参照表で照合する</summary>
public class ScalarFunctionReferenceTests(ITestOutputHelper output)
{
    [Theory]
    [InlineData("Pow")]
    [InlineData("LogAddExp")]
    [InlineData("LogSubExp")]
    [InlineData("LogSumExp")]
    [InlineData("LogGamma")]
    [InlineData("Digamma")]
    [InlineData("LogBeta")]
    [InlineData("Erf")]
    [InlineData("Erfc")]
    [InlineData("NormalPdf")]
    [InlineData("NormalLogPdf")]
    [InlineData("NormalCdf")]
    [InlineData("NormalSf")]
    [InlineData("NormalLogCdf")]
    [InlineData("NormalLogSf")]
    public void LocalDerivativesMatchHighPrecisionReferences(string function)
    {
        var tape = new Tape();
        var statistics = new ScalarFunctionReferenceStatistics();
        using var fingerprint = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        Span<byte> bits = stackalloc byte[8];
        int cases = 0;
        foreach (var reference in ScalarFunctionReferences.Rows.Where(row => row.Function == function))
        {
            tape.Reset();
            ScalarNode[] inputs = new ScalarNode[reference.Inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
            {
                inputs[i] = tape.Leaf(reference.Inputs[i]);
            }

            ScalarNode result = Evaluate(function, inputs);
            tape.Backward(result);
            BinaryPrimitives.WriteInt64LittleEndian(bits, BitConverter.DoubleToInt64Bits(result.Value));
            fingerprint.AppendData(bits);
            cases++;
            for (int i = 0; i < inputs.Length; i++)
            {
                double actual = tape.GetGradient(inputs[i]);
                string context = $"{function}([{string.Join(", ", reference.Inputs.Select(x => x.ToString("R")))}]), input {i}";
                BinaryPrimitives.WriteInt64LittleEndian(bits, BitConverter.DoubleToInt64Bits(actual));
                fingerprint.AppendData(bits);
                var comparison = ScalarFunctionReferenceAssert.Matches(reference, i, actual, context);
                statistics.Observe(reference, i, comparison, context);
            }
        }

        Assert.True(cases > 0);
        output.WriteLine($"{function}: value/gradient fingerprint={Convert.ToHexString(fingerprint.GetHashAndReset())}");
        statistics.WriteTo(output, function, cases);
    }

    private static ScalarNode Evaluate(string function, ScalarNode[] inputs) => function switch
    {
        "Pow" => XMath.Pow(inputs[0], inputs[1]),
        "LogAddExp" => XMath.LogAddExp(inputs[0], inputs[1]),
        "LogSubExp" => XMath.LogSubExp(inputs[0], inputs[1]),
        "LogSumExp" => XMath.LogSumExp((ReadOnlySpan<ScalarNode>)inputs),
        "LogGamma" => XMath.Special.LogGamma(inputs[0]),
        "Digamma" => XMath.Special.Digamma(inputs[0]),
        "LogBeta" => XMath.Special.LogBeta(inputs[0], inputs[1]),
        "Erf" => XMath.Special.Erf(inputs[0]),
        "Erfc" => XMath.Special.Erfc(inputs[0]),
        "NormalPdf" => XMath.Special.NormalPdf(inputs[0]),
        "NormalLogPdf" => XMath.Special.NormalLogPdf(inputs[0]),
        "NormalCdf" => XMath.Special.NormalCdf(inputs[0]),
        "NormalSf" => XMath.Special.NormalSf(inputs[0]),
        "NormalLogCdf" => XMath.Special.NormalLogCdf(inputs[0]),
        "NormalLogSf" => XMath.Special.NormalLogSf(inputs[0]),
        _ => throw new ArgumentOutOfRangeException(nameof(function))
    };
}
