using System.Numerics;

namespace Tuna.Tests;

internal static class SpecialFunctionAssert
{
    // 零点近傍以外は相対誤差で比較し、小さい値を絶対誤差で無条件に通さない。
    internal static void Close(double expected, double actual,
        double relativeTolerance = 5e-13, double absoluteTolerance = 0)
    {
        Assert.True(double.IsFinite(expected), $"Reference must be finite: {expected:R}");
        Assert.True(double.IsFinite(actual), $"Expected {expected:R}, actual {actual:R}");
        double error = Math.Abs(actual - expected);
        double tolerance = Math.Max(absoluteTolerance, relativeTolerance * Math.Abs(expected));
        Assert.True(error <= tolerance,
            $"Expected {expected:R}, actual {actual:R}, error {error:R}, tolerance {tolerance:R}");
    }

    // 丸め済みの参照値との差を、参照値の ulp を単位として比較する（参照値の丸めの 1/2 ulp を含む）。
    internal static void WithinUlps(double expected, double actual, double ulps)
    {
        Assert.True(double.IsFinite(expected) && expected != 0, $"Reference must be finite and nonzero: {expected:R}");
        Assert.True(double.IsFinite(actual), $"Expected {expected:R}, actual {actual:R}");
        double magnitude = Math.Abs(expected);
        double error = Math.Abs(actual - expected) / (Math.BitIncrement(magnitude) - magnitude);
        Assert.True(error <= ulps, $"Expected {expected:R}, actual {actual:R}, error {error:F2} ulp, tolerance {ulps} ulp");
    }

    // 最大成分を基準に複素数全体を比較する。小さい成分に単独の相対精度を求めず、
    // 有限な2成分の絶対値（Euclidean norm）が溢れる場合も許容誤差を有限に保つ。
    internal static void ComplexClose(Complex expected, Complex actual,
        double relativeTolerance = 2e-12, double absoluteTolerance = 0)
    {
        double scale = Math.Max(Math.Abs(expected.Real), Math.Abs(expected.Imaginary));
        double tolerance = Math.Max(absoluteTolerance, relativeTolerance * scale);
        Close(expected.Real, actual.Real, relativeTolerance: 0, absoluteTolerance: tolerance);
        Close(expected.Imaginary, actual.Imaginary, relativeTolerance: 0, absoluteTolerance: tolerance);
    }
}
