using System.Runtime.CompilerServices;

namespace Tuna.Differentiation;

public static partial class XMathExtensions
{
    extension(XMath)
    {
        /// <summary>
        /// Computes the power <c>x^y</c>.
        /// <para>冪乗 <c>x^y</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The base scalar node. / 底となるスカラーノード。</param>
        /// <param name="y">The exponent scalar node. / 指数となるスカラーノード。</param>
        /// <returns>A scalar node holding <c>x^y</c>. / <c>x^y</c> を値とするスカラーノード。</returns>
        /// <remarks>
        /// <para>
        /// For positive bases, the local derivatives are <c>y·x^(y - 1)</c> and <c>x^y·ln x</c>.
        /// For negative bases, only the base derivative at finite integer exponents is defined; the exponent derivative is <c>NaN</c>.
        /// At <c>±0</c>, the base derivative takes the right-hand limit (0 when <c>y = 0</c>, 1 when <c>y = 1</c>), and the exponent derivative is 0 for <c>y &gt; 0</c> and <c>NaN</c> otherwise.
        /// Infinite inputs use the finite-side limit; any <c>NaN</c> input yields <c>NaN</c> for both derivatives. Representable derivatives remain accurate even when <c>x^y</c> underflows to 0 or overflows to <c>±∞</c>.
        /// </para>
        /// <para>
        /// 正の底では、局所微分は <c>y·x^(y - 1)</c> と <c>x^y·ln x</c> とします。
        /// 負の底では有限整数の指数に対する底の局所微分のみを定義し、指数の局所微分は <c>NaN</c> とします。
        /// <c>±0</c> に対する底の局所微分は右側極限（<c>y = 0</c> で 0、<c>y = 1</c> で 1）とし、指数の局所微分は <c>y &gt; 0</c> で 0、それ以外は <c>NaN</c> とします。
        /// 無限大では有限側からの極限、入力に <c>NaN</c> を含む場合は両方の局所微分を <c>NaN</c> とします。値が 0 や <c>±∞</c> に丸まる場合でも、表現可能な局所微分の精度を保ちます。
        /// </para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if either node belongs to a stale generation, the nodes belong to different tapes, or the tape's backward pass has completed. / 世代が古い、異なるテープに属している、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarNode Pow(ScalarNode x, ScalarNode y)
        {
            double rawX = x.Value;
            double rawY = y.Value;
            double value = XMath.Pow<double>(rawX, rawY);
            if (x.Tape is null && y.Tape is null)
            {
                return new(value);
            }

            double dx = x.Tape is null ? 0 : PowerBaseDerivative(rawX, rawY, value);
            double dy = y.Tape is null ? 0 : PowerExponentDerivative(rawX, rawY, value);
            return ScalarNode.Binary(value, x, dx, y, dy);
        }
    }

    internal static double PowerBaseDerivative(double x, double y, double value)
    {
        if (double.IsNaN(x) || double.IsNaN(y))
        {
            return double.NaN;
        }

        if (y == 0)
        {
            return 0;
        }

        if (y == 1)
        {
            return 1;
        }

        if (x < 0 && (!double.IsFinite(y) || y != Math.Truncate(y)))
        {
            return double.NaN;
        }

        if (x == 0)
        {
            // Use the right-hand limit from positive bases for both +0 and -0.
            return y > 1 ? 0 : y > 0 ? double.PositiveInfinity : double.NegativeInfinity;
        }

        if (double.IsInfinity(y))
        {
            if (x == 1)
            {
                return y;
            }

            bool grows = (x > 1) == (y > 0);
            return Math.CopySign(grows ? double.PositiveInfinity : 0, y);
        }

        if (double.IsNormal(value) && double.IsFinite(x))
        {
            // Power-of-two scaling does not alter rounding while intermediates stay normal.
            double product = value * y;
            double direct = product / x;
            if (double.IsNormal(product) && double.IsNormal(direct))
            {
                return direct;
            }
        }

        double magnitude = double.IsInfinity(x)
            ? y > 1 ? double.PositiveInfinity : 0
            : PowerDerivativeMagnitude(Math.Abs(x), y, Math.Abs(value), Math.Abs(y), Math.Abs(x));
        double derivative = Math.CopySign(magnitude, y);
        return double.IsNegative(value) == double.IsNegative(x) ? derivative : -derivative;
    }

    internal static double PowerExponentDerivative(double x, double y, double value)
    {
        if (double.IsNaN(x) || double.IsNaN(y) || x < 0)
        {
            return double.NaN;
        }

        if (x == 0)
        {
            // 0^y is constant for y > 0; the isolated value at y = 0 has no derivative.
            return y > 0 ? 0 : double.NaN;
        }

        if (x == 1)
        {
            return 0;
        }

        if (double.IsPositiveInfinity(x))
        {
            return y < 0 ? 0 : double.PositiveInfinity;
        }

        double logarithm = Math.Log(x);
        if (double.IsInfinity(y))
        {
            bool grows = (x > 1) == (y > 0);
            return Math.CopySign(grows ? double.PositiveInfinity : 0, logarithm);
        }

        if (double.IsNormal(value))
        {
            double direct = value * logarithm;
            if (double.IsNormal(direct))
            {
                return direct;
            }
        }

        return Math.CopySign(PowerDerivativeMagnitude(x, y, value, Math.Abs(logarithm), 1), logarithm);
    }

    // Computes |x^y| * numerator / denominator by separating exponents to prevent premature underflow/overflow.
    private static double PowerDerivativeMagnitude(double x, double y, double value, double numerator, double denominator)
    {
        double normalizedPower;
        int powerExponent;
        if (double.IsNormal(value))
        {
            powerExponent = Math.ILogB(value);
            normalizedPower = Math.ScaleB(value, -powerExponent);
        }
        else
        {
            // When value is subnormal, 0, or ∞, recover the mantissa and exponent via (x^(y/4))^4 to avoid rounding y - 1.
            double quarter = Math.Pow(x, y / 4);
            if (quarter == 0 || double.IsPositiveInfinity(quarter))
            {
                // Unrounded quarter^4 is < 2^-4300 or > 2^4095, beyond any finite numerator/denominator ratio (< 2^2098).
                return quarter;
            }

            int quarterExponent = Math.ILogB(quarter);
            double normalizedQuarter = Math.ScaleB(quarter, -quarterExponent);
            double square = normalizedQuarter * normalizedQuarter;
            normalizedPower = square * square;
            powerExponent = 4 * quarterExponent;
        }

        int numeratorExponent = Math.ILogB(numerator);
        int denominatorExponent = Math.ILogB(denominator);
        double normalizedNumerator = Math.ScaleB(numerator, -numeratorExponent);
        double normalizedDenominator = Math.ScaleB(denominator, -denominatorExponent);
        return Math.ScaleB(normalizedPower * normalizedNumerator / normalizedDenominator,
            powerExponent + numeratorExponent - denominatorExponent);
    }
}
