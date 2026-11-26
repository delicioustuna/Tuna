namespace Tuna.Optimization.Global;

// Directed rounding (toward −∞ / +∞) for +, −, ×, ÷, and √ using TwoSum / FMA error signs.
// Results near the subnormal range (< Tiny) are widened by 1 ulp; elementary functions (exp, log, pow) use a 2-ulp margin.
// Overflow from finite inputs clamps to double.MaxValue when rounding inward; interval endpoint products treat 0·∞ as 0.
internal static class DirectedRounding
{
    // Threshold below which FMA error terms may underflow into subnormals.
    private const double Tiny = 1e-290;

    public static double AddDown(double a, double b)
    {
        double s = a + b;
        if (double.IsFinite(s))
        {
            double bb = s - a;
            double err = (a - (s - bb)) + (b - bb);
            return err >= 0 ? s : Math.BitDecrement(s);
        }

        return s == double.PositiveInfinity && double.IsFinite(a) && double.IsFinite(b) ? double.MaxValue : s;
    }

    public static double AddUp(double a, double b) => -AddDown(-a, -b);

    public static double SubtractDown(double a, double b) => AddDown(a, -b);

    public static double SubtractUp(double a, double b) => -AddDown(-a, b);

    public static double MultiplyDown(double a, double b)
    {
        double p = a * b;
        if (double.IsNaN(p))
        {
            return double.IsNaN(a) || double.IsNaN(b) ? p : 0;
        }

        if (double.IsInfinity(p))
        {
            return p > 0 && double.IsFinite(a) && double.IsFinite(b) ? double.MaxValue : p;
        }

        if (Math.Abs(p) < Tiny)
        {
            return a == 0 || b == 0 ? 0 : Math.BitDecrement(p);
        }

        double err = Math.FusedMultiplyAdd(a, b, -p);
        return err >= 0 ? p : Math.BitDecrement(p);
    }

    public static double MultiplyUp(double a, double b) => -MultiplyDown(-a, b);

    public static double DivideDown(double a, double b)
    {
        double q = a / b;
        if (double.IsInfinity(q))
        {
            return q > 0 && double.IsFinite(a) ? double.MaxValue : q;
        }

        if (a == 0)
        {
            return 0;
        }

        if (Math.Abs(q) < Tiny || Math.Abs(a) < Tiny || Math.Abs(b) < Tiny)
        {
            return Math.BitDecrement(q);
        }

        // a − q·b is exact, and (true quotient) − q = (a − q·b) / b.
        double r = Math.FusedMultiplyAdd(-q, b, a);
        return r == 0 || (r > 0) == (b > 0) ? q : Math.BitDecrement(q);
    }

    public static double DivideUp(double a, double b) => -DivideDown(-a, b);

    public static double SqrtDown(double a)
    {
        double s = Math.Sqrt(a);
        if (!double.IsFinite(s) || s == 0)
        {
            return s;
        }

        if (a < Tiny)
        {
            return Math.BitDecrement(s);
        }

        double r = Math.FusedMultiplyAdd(-s, s, a);
        return r >= 0 ? s : Math.BitDecrement(s);
    }

    public static double SqrtUp(double a)
    {
        double s = Math.Sqrt(a);
        if (!double.IsFinite(s) || (s == 0 && a == 0))
        {
            return s;
        }

        if (a < Tiny)
        {
            return Math.BitIncrement(s);
        }

        double r = Math.FusedMultiplyAdd(-s, s, a);
        return r <= 0 ? s : Math.BitIncrement(s);
    }

    public static double Down2(double value) => double.IsNaN(value) ? value : Math.BitDecrement(Math.BitDecrement(value));

    public static double Up2(double value) => double.IsNaN(value) ? value : Math.BitIncrement(Math.BitIncrement(value));
}
