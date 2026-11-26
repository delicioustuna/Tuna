using Tuna.Tests.ReferenceData;

namespace Tuna.Tests;

/// <summary>コアの導関数値を、拡張に依存せず高精度参照表で照合する</summary>
public class DerivativeValueReferenceTests
{
    [Theory]
    [InlineData("Erf")]
    [InlineData("NormalPdf")]
    [InlineData("NormalLogCdf")]
    [InlineData("NormalLogSf")]
    [InlineData("LogBeta")]
    public void CoreDerivativeValuesMatchHighPrecisionReferences(string function)
    {
        foreach (var reference in ScalarFunctionReferences.Rows.Where(row => row.Function == function))
        {
            for (int i = 0; i < reference.Inputs.Length; i++)
            {
                double expected = function == "NormalLogSf" ? -reference.Derivatives[i] : reference.Derivatives[i];
                double actual = CoreDerivative(function, reference.Inputs, i);
                string context = $"{function} core, input case [{string.Join(", ", reference.Inputs)}], input {i}";
                ScalarFunctionReferenceAssert.Matches(reference, i, actual, context, expectedValue: expected);
            }
        }
    }

    private static double CoreDerivative(string function, double[] inputs, int index) => function switch
    {
        "Erf" => XMath.Special.ErfDerivative(inputs[0]),
        "NormalPdf" => XMath.Special.NormalPdfDerivative(inputs[0]),
        "NormalLogCdf" => XMath.Special.NormalReverseHazard(inputs[0]),
        "NormalLogSf" => XMath.Special.NormalHazard(inputs[0]),
        "LogBeta" => XMath.Special.LogBetaDerivative(inputs[index], inputs[1 - index]),
        _ => throw new ArgumentOutOfRangeException(nameof(function))
    };
}
