using Tuna.Differentiation;
using Tuna.Tests.ReferenceData;

namespace Tuna.Tests.Differentiation;

/// <summary>自動微分の数値比較と一変数の勾配・数値微分</summary>
internal static class DifferentiationAssert
{
    internal const double SixteenEpsilon = ReferencePrecision.SixteenEpsilon;

    internal static void EqualBits(double expected, double actual, string? context = null)
        => ReferencePrecision.EqualBits(expected, actual, context: context);

    internal static double Gradient(Func<ScalarNode, ScalarNode> function, double x)
    {
        var tape = new Tape();
        var input = tape.Leaf(x);
        tape.Backward(function(input));
        return tape.GetGradient(input);
    }

    internal static double Richardson(Func<double, double> function, double x, double? step = null)
    {
        // x = 0 では 1 を基準にする。非正規化数でも h/2 が 0 にならない下限を置く。
        double h = step ?? Math.Max(Math.ScaleB(x == 0 ? 1 : Math.Abs(x), -12), 4 * double.Epsilon);
        double coarse = (function(x + h) - function(x - h)) / (2 * h);
        double fine = (function(x + h / 2) - function(x - h / 2)) / h;
        return (4 * fine - coarse) / 3;
    }

    internal static void Close(double expected, double actual, double relativeTolerance, double absoluteTolerance, string? context = null)
    {
        Assert.True(Math.Abs(actual - expected) <= Math.Max(relativeTolerance * Math.Abs(expected), absoluteTolerance),
            $"{context}: expected {expected:R}, actual {actual:R}");
    }

    internal static void SameNumber(double expected, double actual, string? context = null)
    {
        Assert.True(double.IsNaN(expected) ? double.IsNaN(actual) : actual == expected,
            $"{context}: expected {expected:R}, actual {actual:R}");
    }

    internal static void Throws<TException>(Action action, string context) where TException : Exception
    {
        Exception? exception = Record.Exception(action);
        Assert.True(exception?.GetType() == typeof(TException),
            $"{context}: expected {typeof(TException).Name}, actual {exception?.ToString() ?? "no exception"}");
    }

    internal static void WithinUlps(double expected, double actual, ulong tolerance, string? context = null)
    {
        ulong distance = UlpDistance(expected, actual);
        Assert.True(distance <= tolerance,
            $"{context}: expected {expected:R}, actual {actual:R}, distance {distance} ulp");
    }

    internal static ulong UlpDistance(double first, double second)
        => ReferencePrecision.UlpDistance(first, second);
}
