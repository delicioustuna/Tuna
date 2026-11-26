using Xunit;
using Xunit.Abstractions;

namespace Tuna.Tests.ReferenceData;

internal readonly record struct ScalarFunctionReferenceError(double Relative, ulong Ulps, double Scaled,
    bool Conditioned, bool StrictAccepted, bool ExtendedAccepted);

/// <summary>コアの導関数値と AD が共有する、独立参照表との比較</summary>
internal static class ScalarFunctionReferenceAssert
{
    private const ulong UlpTolerance = 4;
    // 拡張精度の ψ の差が強く相殺する点だけの丸め幅。通常精度の絶対誤差の床は置かない。
    private const double ExtendedPrecisionScaledTolerance = 3.1554436208840472e-30; // 64 × 2⁻¹⁰⁴
    private const double ConditioningThreshold = 1e12;

    internal static ScalarFunctionReferenceError? Matches(ScalarFunctionReference reference, int index,
        double actual, string context, double? expectedValue = null)
    {
        double expected = expectedValue ?? reference.Derivatives[index];
        if (!double.IsFinite(expected))
        {
            Assert.True(double.IsNaN(expected) ? double.IsNaN(actual) : actual == expected,
                $"{context}: expected {expected:R}, actual {actual:R}");
            return null;
        }

        Assert.True(double.IsFinite(actual), $"{context}: expected {expected:R}, actual {actual:R}");
        double relative = expected == 0 ? (actual == 0 ? 0 : double.PositiveInfinity) : Math.Abs((actual - expected) / expected);
        ulong ulps = ReferencePrecision.UlpDistance(actual, expected);
        double scale = reference.ConditionScales?[index] ?? 0;
        double scaled = scale > 0 && double.IsNormal(expected) ? Math.Abs(actual - expected) / scale : double.PositiveInfinity;
        bool strict = relative <= ReferencePrecision.SixteenEpsilon || ulps <= UlpTolerance;
        bool conditioned = IsConditionedReference(reference, index);
        bool extended = conditioned && actual != 0 && Math.Sign(actual) == Math.Sign(expected)
            && scaled <= ExtendedPrecisionScaledTolerance;
        Assert.True(strict || extended,
            $"{context}: expected {expected:R}, actual {actual:R}, relative error {relative:E17}, ULP distance {ulps}, scaled error {scaled:E17}");
        return new(relative, ulps, scaled, conditioned, strict, extended);
    }

    private static bool IsConditionedReference(ScalarFunctionReference reference, int index)
    {
        if (reference.Function != "LogBeta")
        {
            return false;
        }

        double expected = reference.Derivatives[index];
        double scale = reference.ConditionScales?[index] ?? 0;
        // 有限な binary64 の和の符号は、厳密な和の符号と一致する。
        bool negativeBranch = reference.Inputs[index] < 0 || reference.Inputs[0] + reference.Inputs[1] < 0;
        return negativeBranch && Math.Abs(reference.Inputs[1 - index]) > 0.125 && double.IsNormal(expected)
            && double.IsFinite(scale) && scale / Math.Abs(expected) > ConditioningThreshold;
    }
}

/// <summary>最大誤差と、その点の入力・文脈</summary>
internal sealed class MaximumError
{
    internal double Value { get; private set; }
    internal double Input { get; private set; } = double.NaN;
    internal string? Context { get; private set; }

    internal void Update(double error, string context, double input = double.NaN)
    {
        if (Context is null || error > Value)
        {
            Value = error;
            Input = input;
            Context = context;
        }
    }
}

/// <summary>照合の判定から分離した誤差と件数の集計</summary>
internal sealed class ScalarFunctionReferenceStatistics
{
    private readonly MaximumError relative = new();
    private readonly MaximumError scaled = new();
    private readonly MaximumError conditionedScaled = new();
    private readonly MaximumError ordinaryRelative = new();
    private readonly MaximumError positiveRelative = new();
    private readonly MaximumError tinyIncrementRelative = new();
    private int samples;
    private int subnormalCount;
    private int conditionedCount;
    private int extendedOnlyCount;
    private ulong maximumUlps;
    private int tinyIncrementCount;
    private ulong tinyIncrementUlps;

    internal void Observe(ScalarFunctionReference reference, int index, ScalarFunctionReferenceError? comparison, string context)
    {
        samples++;
        if (comparison is not { } error)
        {
            return;
        }

        maximumUlps = Math.Max(maximumUlps, error.Ulps);
        double expected = reference.Derivatives[index];
        if (reference.Function == "LogBeta" && index == 0 && reference.Inputs[0] < 0 && Math.Abs(reference.Inputs[1]) <= 1e-16)
        {
            tinyIncrementCount++;
            tinyIncrementUlps = Math.Max(tinyIncrementUlps, error.Ulps);
            if (double.IsNormal(expected))
            {
                tinyIncrementRelative.Update(error.Relative, context);
            }
        }

        if (double.IsFinite(error.Scaled))
        {
            scaled.Update(error.Scaled, context);
        }

        if (error.Conditioned)
        {
            conditionedCount++;
            conditionedScaled.Update(error.Scaled, context);
            if (!error.StrictAccepted && error.ExtendedAccepted)
            {
                extendedOnlyCount++;
            }
        }
        else if (double.IsNormal(expected))
        {
            ordinaryRelative.Update(error.Relative, context);
        }

        if (double.IsNormal(expected))
        {
            relative.Update(error.Relative, context, reference.Inputs[index]);
            if (reference.Function == "LogBeta" && reference.Inputs[0] > 0 && reference.Inputs[1] > 0)
            {
                positiveRelative.Update(error.Relative, context);
            }
        }
        else if (double.IsSubnormal(expected))
        {
            subnormalCount++;
        }
    }

    internal void WriteTo(ITestOutputHelper output, string function, int cases)
    {
        output.WriteLine(FormattableString.Invariant(
            $"{function}: samples={samples}, max relative error (normal references)={relative.Value:E17} at x={relative.Input:R}, max ULP distance (all finite references)={maximumUlps}, subnormal references={subnormalCount}"));
        output.WriteLine($"{function}: input cases={cases}, worst normal reference={relative.Context}");
        if (scaled.Context is not null)
        {
            output.WriteLine(FormattableString.Invariant(
                $"{function}: max scaled absolute error (all scaled normal references)={scaled.Value:E17} at {scaled.Context}"));
        }

        if (conditionedCount != 0)
        {
            output.WriteLine($"{function}: conditioned references={conditionedCount}, accepted only by extended-precision bound={extendedOnlyCount}");
            output.WriteLine(FormattableString.Invariant(
                $"{function}: max scaled absolute error (conditioned normal references)={conditionedScaled.Value:E17} at {conditionedScaled.Context}"));
            output.WriteLine(FormattableString.Invariant(
                $"{function}: max relative error (ordinary normal references)={ordinaryRelative.Value:E17} at {ordinaryRelative.Context}"));
            output.WriteLine(FormattableString.Invariant(
                $"{function}: max relative error (positive normal references)={positiveRelative.Value:E17} at {positiveRelative.Context}"));
        }

        if (tinyIncrementCount != 0)
        {
            output.WriteLine(FormattableString.Invariant(
                $"{function}: tiny-increment finite references={tinyIncrementCount}, max relative error (normal references)={tinyIncrementRelative.Value:E17} at {tinyIncrementRelative.Context}, max ULP distance (all finite references)={tinyIncrementUlps}"));
        }
    }
}
