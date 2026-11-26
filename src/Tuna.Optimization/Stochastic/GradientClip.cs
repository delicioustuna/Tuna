using System.Numerics;

namespace Tuna.Optimization.Stochastic;

internal readonly struct GradientClip(double divisor, double root, double scale)
{
    private const double MinNormal = 2.2250738585072014E-308;

    internal bool IsActive => divisor != 0;

    internal double Apply(double value)
    {
        if (!IsActive)
        {
            return value;
        }

        return divisor == 1 && root == 1 ? value * scale : value / divisor / root * scale;
    }

    internal static bool TryCreate(double squaredNorm, double limit, out GradientClip clip)
    {
        clip = default;
        if (double.IsPositiveInfinity(limit) || double.IsNaN(squaredNorm))
        {
            return true;
        }

        if (squaredNorm < MinNormal || !double.IsFinite(squaredNorm))
        {
            return false;
        }

        double norm = Math.Sqrt(squaredNorm);
        if (norm > limit)
        {
            double factor = limit / norm;
            // 倍率が 0 や非正規化数に丸められる場合は、勾配をノルムで割ってから上限を掛ける。
            clip = factor >= MinNormal ? new(1, 1, factor) : new(norm, 1, limit);
        }

        return true;
    }

    internal static GradientClip CreateScaled(double max, double squaredNorm, double limit)
    {
        if (double.IsPositiveInfinity(max))
        {
            return new(1, 1, 0);
        }

        // ‖g‖₂ = max·√s を直接求めるとオーバーフローするため、比較と更新をスケーリング後に行う。
        double root = Math.Sqrt(squaredNorm);
        if (limit / max >= root)
        {
            return default;
        }

        double factor = limit / root;
        return factor >= MinNormal ? new(max, 1, factor) : new(max, root, limit);
    }

    internal static double SumOfSquares<T>(ReadOnlySpan<T> gradient, double sum)
        where T : IFloatingPointIeee754<T>
    {
        foreach (T value in gradient)
        {
            double g = double.CreateChecked(value);
            sum += g * g;
        }

        return sum;
    }

    internal static double MaxAbs<T>(ReadOnlySpan<T> gradient, double max)
        where T : IFloatingPointIeee754<T>
    {
        foreach (T value in gradient)
        {
            max = Math.Max(max, Math.Abs(double.CreateChecked(value)));
        }

        return max;
    }

    internal static double ScaledSumOfSquares<T>(ReadOnlySpan<T> gradient, double max, double sum)
        where T : IFloatingPointIeee754<T>
    {
        foreach (T value in gradient)
        {
            double ratio = double.CreateChecked(value) / max;
            sum += ratio * ratio;
        }

        return sum;
    }
}
