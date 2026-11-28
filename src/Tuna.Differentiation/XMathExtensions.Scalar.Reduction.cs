using System.Buffers;
using System.Runtime.CompilerServices;
using Tuna.LinearAlgebra;

namespace Tuna.Differentiation;

public static partial class XMathExtensions
{
    // exp(-746) is smaller than half the smallest positive double subnormal.
    private const double ExpZeroBelow = -746;

    extension(XMath)
    {
        /// <summary>
        /// Computes <c>ln(eˣ + eʸ)</c> stably.
        /// <para>指数の和の対数 <c>ln(eˣ + eʸ)</c> を安定に計算します。</para>
        /// </summary>
        /// <param name="x">The first scalar node. / 第 1 入力スカラーノード。</param>
        /// <param name="y">The second scalar node. / 第 2 入力スカラーノード。</param>
        /// <returns>A scalar node holding <c>ln(eˣ + eʸ)</c>. / <c>ln(eˣ + eʸ)</c> を値とするスカラーノード。</returns>
        /// <remarks>
        /// <para>
        /// The local derivatives are <c>eˣ/(eˣ + eʸ)</c> and <c>eʸ/(eˣ + eʸ)</c>.
        /// Equal infinities of the same sign receive <c>1/2</c> each; otherwise, the larger input receives 1 and the smaller receives 0.
        /// If either input is <c>NaN</c>, both local derivatives are <c>NaN</c>.
        /// </para>
        /// <para>
        /// 局所微分は順に <c>eˣ/(eˣ + eʸ)</c>、<c>eʸ/(eˣ + eʸ)</c> とします。
        /// 同符号の無限大どうしでは両方を <c>1/2</c> とし、それ以外の無限大の組み合わせでは大きい入力を 1、もう一方を 0 とします。
        /// いずれかが <c>NaN</c> の場合は両方の局所微分を <c>NaN</c> とします。
        /// </para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if either node belongs to a stale generation, the nodes belong to different tapes, or the tape's backward pass has completed. / 世代が古い、異なるテープに属している、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarNode LogAddExp(ScalarNode x, ScalarNode y)
        {
            double a = x.Value, b = y.Value;
            double value = XMath.LogAddExp<double>(a, b);
            if (x.Tape is null && y.Tape is null)
            {
                return new(value);
            }

            LogAddExpDerivatives(a, b, out double dx, out double dy);
            return ScalarNode.Binary(value, x, dx, y, dy);
        }

        /// <summary>
        /// Computes <c>ln(eˣ - eʸ)</c> stably for <c>x ≥ y</c>.
        /// <para><c>x ≥ y</c> の範囲で指数の差の対数 <c>ln(eˣ - eʸ)</c> を安定に計算します。</para>
        /// </summary>
        /// <param name="x">The minuend log-value node. / 被減数となるスカラーノード。</param>
        /// <param name="y">The subtrahend log-value node. / 減数となるスカラーノード。</param>
        /// <returns>A scalar node holding <c>ln(eˣ - eʸ)</c>. / <c>ln(eˣ - eʸ)</c> を値とするスカラーノード。</returns>
        /// <remarks>
        /// <para>
        /// The local derivatives are <c>1/(1 - eʸ⁻ˣ)</c> and <c>-eʸ⁻ˣ/(1 - eʸ⁻ˣ)</c>.
        /// For equal finite inputs, the interior limits <c>+∞</c> and <c>-∞</c> are returned.
        /// For <c>x &lt; y</c>, equal infinities of the same sign, or <c>NaN</c>, both local derivatives are <c>NaN</c>.
        /// When <c>x = +∞</c> or <c>y = -∞</c> and <c>x ≠ y</c>, the local derivatives are 1 and 0 respectively.
        /// </para>
        /// <para>
        /// 局所微分は順に <c>1/(1 - eʸ⁻ˣ)</c>、<c>-eʸ⁻ˣ/(1 - eʸ⁻ˣ)</c> とします。
        /// 有限な同値入力では定義域の内側からの極限として <c>+∞</c>、<c>-∞</c> とします。
        /// <c>x &lt; y</c>、同符号の無限大どうし、または <c>NaN</c> の場合は両方を <c>NaN</c> とします。
        /// <c>x = +∞</c> または <c>y = -∞</c> で <c>x ≠ y</c> の場合は順に 1、0 とします。
        /// </para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if either node belongs to a stale generation, the nodes belong to different tapes, or the tape's backward pass has completed. / 世代が古い、異なるテープに属している、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarNode LogSubExp(ScalarNode x, ScalarNode y)
        {
            double a = x.Value, b = y.Value;
            double value = XMath.LogSubExp<double>(a, b);
            if (x.Tape is null && y.Tape is null)
            {
                return new(value);
            }

            LogSubExpDerivatives(a, b, out double dx, out double dy);
            return ScalarNode.Binary(value, x, dx, y, dy);
        }

        /// <summary>
        /// Computes the log-sum-exp <c>ln Σ eˣⁱ</c> over a span of scalar nodes.
        /// <para>スカラーノード列の指数の総和の対数 <c>ln Σ eˣⁱ</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The input scalar nodes; the caller may mutate the span after this call returns. / 入力スカラーノード列（呼び出し後にスパンを書き換えても影響ありません）。</param>
        /// <returns>A scalar node holding <c>ln Σ eˣⁱ</c>, or constant <c>-∞</c> if <paramref name="x"/> is empty. / 指数の総和の対数を値とするスカラーノード。入力が空の場合は定数の <c>-∞</c>。</returns>
        /// <remarks>
        /// <para>
        /// The local derivative for each element is <c>eˣⁱ / Σ eˣʲ</c>.
        /// If any input is <c>+∞</c>, the derivative is divided equally among the <c>+∞</c> inputs (0 elsewhere); if all inputs are <c>-∞</c>, it is divided equally among all inputs.
        /// If any input is <c>NaN</c>, all local derivatives are <c>NaN</c>.
        /// </para>
        /// <para>
        /// 各要素の局所微分は <c>eˣⁱ / Σ eˣʲ</c> とします。
        /// <c>+∞</c> を含む場合はその入力のみで等分（他は 0）し、すべてが <c>-∞</c> の場合は全入力で等分します。
        /// いずれかの入力が <c>NaN</c> の場合はすべての局所微分を <c>NaN</c> とします。
        /// </para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if any node belongs to a stale generation, nodes belong to different tapes, or the tape's backward pass has completed. / 世代が古い、異なるテープに属している、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static ScalarNode LogSumExp(ReadOnlySpan<ScalarNode> x)
        {
            double[]? rented = null;
            Span<double> values = x.Length <= 256
                ? stackalloc double[x.Length]
                : (rented = ArrayPool<double>.Shared.Rent(x.Length)).AsSpan(0, x.Length);
            try
            {
                bool requiresGradient = false;
                double maximum = double.NegativeInfinity;
                int positiveInfinities = 0;
                for (int i = 0; i < x.Length; i++)
                {
                    double raw = x[i].Value;
                    values[i] = raw;
                    requiresGradient |= x[i].Tape is not null;
                    maximum = Math.Max(maximum, raw);
                    if (double.IsPositiveInfinity(raw))
                    {
                        positiveInfinities++;
                    }
                }

                double value = XMath.LogSumExp<double>(new ReadOnlyVectorSpan<double>(values));
                if (!requiresGradient)
                {
                    return new(value);
                }

                if (double.IsNaN(value))
                {
                    values.Fill(double.NaN);
                }
                else if (positiveInfinities > 0)
                {
                    double share = 1.0 / positiveInfinities;
                    for (int i = 0; i < x.Length; i++)
                    {
                        values[i] = double.IsPositiveInfinity(values[i]) ? share : 0;
                    }
                }
                else if (double.IsNegativeInfinity(maximum))
                {
                    values.Fill(1.0 / x.Length);
                }
                else
                {
                    // Compute shifted weights with compensated summation rather than subtracting the rounded output value.
                    double sum = 0, correction = 0;
                    for (int i = 0; i < x.Length; i++)
                    {
                        double weight = ExpNonpositiveDifference(values[i], maximum);
                        values[i] = weight;
                        double next = sum + weight;
                        correction += sum >= weight ? (sum - next) + weight : (weight - next) + sum;
                        sum = next;
                    }

                    double denominator = sum + correction;
                    for (int i = 0; i < x.Length; i++)
                    {
                        double derivative = values[i] / denominator;
                        // Avoid premature subnormal rounding before dividing by denominator.
                        values[i] = double.IsSubnormal(derivative) || derivative == 0
                            ? ExpNonpositiveDifference(x[i].Value, maximum, denominator)
                            : derivative;
                    }
                }

                return ScalarNode.Record(value, x, values);
            }
            finally
            {
                if (rented is not null)
                {
                    ArrayPool<double>.Shared.Return(rented);
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void LogAddExpDerivatives(double a, double b, out double dx, out double dy)
    {
        if (double.IsNaN(a) || double.IsNaN(b))
        {
            dx = dy = double.NaN;
        }
        else if (a == b)
        {
            dx = dy = 0.5;
        }
        else if (a > b)
        {
            double t = ExpNonpositiveDifference(b, a);
            dx = 1 / (1 + t);
            dy = t / (1 + t);
        }
        else
        {
            double t = ExpNonpositiveDifference(a, b);
            dx = t / (1 + t);
            dy = 1 / (1 + t);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void LogSubExpDerivatives(double a, double b, out double dx, out double dy)
    {
        if (double.IsNaN(a) || double.IsNaN(b) || a < b || (a == b && double.IsInfinity(a)))
        {
            dx = dy = double.NaN;
        }
        else if (a == b)
        {
            dx = double.PositiveInfinity;
            dy = double.NegativeInfinity;
        }
        else
        {
            double difference = b - a;
            double t = ExpNonpositiveDifference(b, a);
            double denominator;
            if (difference < -XMath.Constants.Ln2)
            {
                denominator = 1 - t;
            }
            else
            {
                // Propagate the rounding error of (b - a) into expm1 to avoid cancellation in 1 - exp(b - a).
                double error = DifferenceError(b, a, difference);
                denominator = -XMath.Expm1<double>(difference) - Math.Exp(difference) * XMath.Expm1<double>(error);
            }

            dx = 1 / denominator;
            dy = -t / denominator;
        }
    }

    // Computes exp(smaller - larger) / denominator for smaller <= larger, compensating for rounding in (smaller - larger).
    private static double ExpNonpositiveDifference(double smaller, double larger, double denominator = 1)
    {
        if (smaller == larger)
        {
            return 1 / denominator;
        }

        double difference = smaller - larger;
        if (difference <= ExpZeroBelow)
        {
            return 0;
        }

        double error = DifferenceError(smaller, larger, difference);
        double factor = error == 0 ? 1 : Math.Exp(error);
        if (difference >= -708)
        {
            return Math.Exp(difference) * factor / denominator;
        }

        // Split the exponent in half so subnormal rounding occurs only once at the final product.
        double half = Math.Exp(0.5 * difference);
        return half * (half * (factor / denominator));
    }

    private static double DifferenceError(double left, double right, double difference)
    {
        double virtualRight = difference - left;
        return (left - (difference - virtualRight)) + (-right - virtualRight);
    }
}
