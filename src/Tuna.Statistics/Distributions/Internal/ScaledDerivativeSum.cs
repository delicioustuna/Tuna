namespace Tuna.Statistics.Distributions;

// 指数の範囲ごとに補償付きで集計し、大きな項どうしが打ち消し合ったときに残る小さな項を保つ。
internal struct ScaledDerivativeSum
{
    private CompensatedSum tiny;
    private CompensatedSum ordinary;
    private CompensatedSum large;
    private CompensatedSum huge;
    private double nonfinite;

    internal void Add(ScaledDerivative derivative)
    {
        double value = derivative.Mantissa;
        if (!double.IsFinite(value))
        {
            nonfinite += value;
            return;
        }

        if (value == 0)
        {
            return;
        }

        int exponent = derivative.Exponent;
        if (exponent == 0 && Math.Abs(value) is >= 1e-150 and <= 1e150)
        {
            ordinary.Add(value);
            return;
        }

        int magnitude = exponent + Math.ILogB(value);
        if (magnitude < -500)
        {
            tiny.Add(Math.ScaleB(value, exponent + 1024));
        }
        else if (magnitude <= 500)
        {
            ordinary.Add(Math.ScaleB(value, exponent));
        }
        else if (magnitude <= 1536)
        {
            large.Add(Math.ScaleB(value, exponent - 1024));
        }
        else
        {
            huge.Add(Math.ScaleB(value, exponent - 2048));
        }
    }

    internal readonly double Value
    {
        get
        {
            if (nonfinite != 0)
            {
                return nonfinite;
            }

            var result = Merge(new(huge.Value, 2048), new(large.Value, 1024));
            result = Merge(result, new(ordinary.Value));
            return Merge(result, new(tiny.Value, -1024)).Value;
        }
    }

    private static ScaledDerivative Merge(ScaledDerivative left, ScaledDerivative right)
    {
        if (left.Mantissa == 0)
        {
            return right;
        }

        if (right.Mantissa == 0)
        {
            return left;
        }

        int power = Math.Max(Math.ILogB(left.Mantissa) + left.Exponent, Math.ILogB(right.Mantissa) + right.Exponent);
        double total = Math.ScaleB(left.Mantissa, left.Exponent - power) + Math.ScaleB(right.Mantissa, right.Exponent - power);
        return new(total, power);
    }
}
