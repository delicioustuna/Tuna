namespace Tuna.Statistics.Distributions;

internal static class DistributionMath
{
    internal const double Epsilon = 2.220446049250313e-16;
    internal const double MinNormal = 2.2250738585072014e-308;
    internal const double Ln2 = 0.6931471805599453;
    internal const double LnPi = 1.1447298858494002;
    internal const double LnTwoPi = 1.8378770664093453;
    internal const double LnSqrt2Pi = 0.9189385332046728;

    // e^x が正規化数となる下限よりやや大きい閾値。これ未満では線形確率を経由せず対数空間で逆関数を解く
    internal const double LogProbabilityLinearLimit = -700;

    internal static double Log1p(double x) => XMath.Special.Log1p(x);

    // 差のみがオーバーフローする場合は 1/2 スケーリング（正確な 2 の冪乗）で回避する
    internal static double Standardize(double x, double location, double scale)
    {
        double z = (x - location) / scale;
        return double.IsInfinity(z) && double.IsFinite(x) ? (0.5 * x - 0.5 * location) / scale * 2 : z;
    }

    internal static double StandardizeCorrection(double x, double location, double scale, double z)
    {
        double difference = x - location;
        if (!double.IsFinite(z) || !double.IsFinite(difference))
        {
            return 0;
        }

        double displaced = difference - x;
        double error = (x - (difference - displaced)) - (location + displaced);
        return (Math.FusedMultiplyAdd(-z, scale, difference) + error) / scale;
    }

    // 積・和のみがオーバーフローする場合は 1/2 スケーリングで回避する
    internal static double Unstandardize(double location, double scale, double z)
    {
        double x = location + scale * z;
        return double.IsInfinity(x) && double.IsFinite(z) ? 2 * (0.5 * location + 0.5 * scale * z) : x;
    }

    // z = m * 2^e として、z 自体がオーバーフローする場合でも scale * z が表現可能なら有限値を返す
    internal static double UnstandardizeScaled(double location, double scale, double m, int e)
    {
        int s = Math.ILogB(scale);
        double half = Math.ScaleB(Math.ScaleB(scale, -s) * m, s + e - 1);
        double x = location + 2 * half;
        return double.IsInfinity(x) ? 2 * (0.5 * location + half) : x;
    }

    // z = ±e^logZ がオーバーフローする場合は scale * z / 2 = ±e^(logZ + ln(scale) - ln 2) から計算する
    internal static double UnstandardizeFromLog(double location, double scale, bool negative, double logZ)
    {
        double z = Math.Exp(logZ);
        if (double.IsFinite(z))
        {
            return Unstandardize(location, scale, negative ? -z : z);
        }

        double half = Math.Exp(logZ + Math.Log(scale) - Ln2);
        return 2 * (0.5 * location + (negative ? -half : half));
    }

    internal static double Expm1(double x) => XMath.Special.Expm1(x);

    internal static double Log1mExp(double x) => XMath.Special.Log1mExp(x);

    // ln(1 + e^x) をオーバーフロー・桁落ちなく計算する
    internal static double Log1pExp(double x)
    {
        if (x <= -37)
        {
            return Math.Exp(x);
        }

        if (x <= 18)
        {
            return Log1p(Math.Exp(x));
        }

        return x <= 33.3 ? x + Math.Exp(-x) : x;
    }

    internal static double Logit(double p) => Math.Log(p) - Log1p(-p);

    internal static double Log1pmx(double u) => XMath.Special.Log1pmx(u);

    // 各因子を [1, 2) の仮数と 2 の冪指数に分解して乗除し、中間オーバーフロー・アンダーフローを回避する
    internal static double ScaledRatio(ReadOnlySpan<double> numerators, ReadOnlySpan<double> denominators) =>
        ScaledRatio(numerators, denominators, 1, 0);

    internal static double ExponentialRatio((double Hi, double Lo) exponent, ReadOnlySpan<double> numerators,
        ReadOnlySpan<double> denominators)
    {
        if (double.IsNaN(exponent.Hi))
        {
            return double.NaN;
        }

        double mantissa = XMath.Special.ConfluentScaledExp(exponent.Hi, exponent.Lo, out long scale);
        return !double.IsFinite(mantissa) || mantissa == 0 ? mantissa : ScaledRatio(numerators, denominators, mantissa, scale);
    }

    internal static double ScaledRatio(ReadOnlySpan<double> numerators, ReadOnlySpan<double> denominators, double value, long exponent)
    {
        foreach (double factor in numerators)
        {
            if (factor == 0)
            {
                return 0;
            }

            int e = Math.ILogB(factor);
            value *= Math.ScaleB(factor, -e);
            exponent += e;
        }

        foreach (double factor in denominators)
        {
            int e = Math.ILogB(factor);
            value /= Math.ScaleB(factor, -e);
            exponent -= e;
        }

        return Math.ScaleB(value, (int)Math.Clamp(exponent, int.MinValue, int.MaxValue));
    }

    // スターリング残差 S(x) = ln Γ(x) - (x - 1/2) ln x + x - ln √(2π)（Loader (2000) の stirlerr）。
    // x < 10 は漸化差 S(y) - S(y+1) = Σ t^(2k)/(2k+1)（t = 1/(2y+1)）を正項級数で加算して桁落ちを避ける（y < 1 は直接式）。
    internal static double StirlingRemainder(double x)
    {
        if (x >= 10)
        {
            return XMath.Special.StirlingCorrection(x);
        }

        if (!(x > 0))
        {
            return x == 0 ? double.PositiveInfinity : double.NaN;
        }

        int steps = (int)Math.Ceiling(10 - x);
        double sum = 0;
        double y = x;
        for (int j = 0; j < steps; j++, y += 1)
        {
            if (y < 1)
            {
                sum += (y + 0.5) * Log1p(1 / y) - 1;
                continue;
            }

            double t = 1 / (2 * y + 1);
            double t2 = t * t;
            double power = t2;
            double series = 0;
            for (int k = 1; k < 200; k++)
            {
                double term = power / (2 * k + 1);
                series += term;
                if (term <= Epsilon * 0.25 * series)
                {
                    break;
                }

                power *= t2;
            }

            sum += series;
        }

        return XMath.Special.StirlingCorrection(y) + sum;
    }

    // S(a) + (1/2) ln(1 + x/a)。a < 1 では ln Γ(a) - a ln a + a - ln √(2π) + (1/2) ln(a + x) として 1/a のオーバーフローを防ぐ
    internal static double StirlingRemainderShifted(double a, double x) =>
        a >= 1 ? StirlingRemainder(a) + 0.5 * Log1p(x / a) : StirlingRemainderShiftedWithoutLog(a, x) - Math.Log(a);

    // a < 1 での StirlingRemainderShifted + ln a。呼び出し側が -ln a を他項の対数と組み合わせて桁落ちを防ぐために使う
    internal static double StirlingRemainderShiftedWithoutLog(double a, double x) =>
        XMath.Special.LogGamma(1 + a) - a * Math.Log(a) + a - 0.5 * LnTwoPi + 0.5 * Math.Log(a + x);

    // ディガンマ漸近残差 R(x) = ln x - 1/(2x) - ψ(x)。
    // x < 10 は漸化差 R(y) - R(y+1) = 2 Σ 2k t^(2k+1)/(2k+1)（t = 1/(2y+1)）を正項級数で加算する（y < 1 は直接式）。
    internal static double DigammaRemainder(double x)
    {
        if (!(x > 0))
        {
            return x == 0 ? double.NegativeInfinity : double.NaN;
        }

        double sum = 0;
        double y = x;
        while (y < 10)
        {
            if (y < 1)
            {
                sum += 0.5 * (1 / y + 1 / (y + 1)) - Log1p(1 / y);
            }
            else
            {
                double t = 1 / (2 * y + 1);
                double t2 = t * t;
                double power = t * t2;
                double series = 0;
                for (int k = 1; k < 200; k++)
                {
                    double term = power * (2.0 * k) / (2 * k + 1);
                    series += term;
                    if (term <= Epsilon * 0.25 * series)
                    {
                        break;
                    }

                    power *= t2;
                }

                sum += 2 * series;
            }

            y += 1;
        }

        double z2 = 1 / (y * y);
        double asymptotic = DigammaAsymptoticCoefficients[^1];
        for (int k = DigammaAsymptoticCoefficients.Length - 2; k >= 0; k--)
        {
            asymptotic = DigammaAsymptoticCoefficients[k] + z2 * asymptotic;
        }

        return z2 * asymptotic + sum;
    }

    // B_(2k)/(2k) (k = 1..10) for R(x) = Σ (B_(2k)/(2k)) x^(-2k)
    private static readonly double[] DigammaAsymptoticCoefficients =
    [
        1.0 / 12, -1.0 / 120, 1.0 / 252, -1.0 / 240, 1.0 / 132,
        -691.0 / 32760, 1.0 / 12, -3617.0 / 8160, 43867.0 / 14364, -174611.0 / 6600,
    ];

    // ψ(x + 1/2) - ψ(x) を ψ(x) = ln x - 1/(2x) - R(x) で展開し、大きな x での桁落ちを避ける
    internal static double DigammaHalfStep(double x)
    {
        if (x < 1)
        {
            return XMath.Special.Digamma(x + 0.5) - XMath.Special.Digamma(x);
        }

        return Log1p(0.5 / x) - 1 / (2 * x + 1) + 0.5 / x - DigammaRemainder(x + 0.5) + DigammaRemainder(x);
    }

    // ln(Γ(x + 1/2) / Γ(x))
    internal static double LogGammaHalfStepRatio(double x) => 0.5 * Math.Log(x) + LogGammaHalfStepCorrection(x);

    // ln(Γ(x + 1/2) / (Γ(x) √x)) をスターリング残差と Log1pmx で桁落ちなく求める
    internal static double LogGammaHalfStepCorrection(double x)
    {
        if (x < 1)
        {
            return XMath.Special.LogGamma(x + 0.5) - XMath.Special.LogGamma(x) - 0.5 * Math.Log(x);
        }

        // x ln(1 + u) - 1/2 = x (ln(1 + u) - u)（u = 1/(2x)）
        return x * Log1pmx(0.5 / x) + StirlingRemainder(x + 0.5) - StirlingRemainder(x);
    }

    // ln(Γ(x + h) / (Γ(x) x^h)) をスターリング残差と Log1pmx で桁落ちなく求める
    internal static double LogGammaShiftCorrection(double x, double h)
    {
        if (h == 0.5)
        {
            return LogGammaHalfStepCorrection(x);
        }

        if (x < 1)
        {
            return XMath.Special.LogGamma(x + h) - XMath.Special.LogGamma(x) - h * Math.Log(x);
        }

        double u = h / x;
        return x * Log1pmx(u) + (h - 0.5) * Log1p(u) + StirlingRemainder(x + h) - StirlingRemainder(x);
    }

    // 標準ガンマ分位点の Wilson-Hilferty 近似（初期値用、非正なら NaN）
    internal static double WilsonHilferty(double shape, double z)
    {
        double c = 1 - 1 / (9 * shape) + z / (3 * Math.Sqrt(shape));
        return c > 0 ? shape * c * c * c : double.NaN;
    }

    // 鞍点展開の偏差 bd0(x, m) = x ln(x/m) + m - x（Loader 2000）。
    // m/x >= 1/2 では -x * Log1pmx((m - x)/x) で x ≈ m の桁落ちを防ぎ、m/x < 1/2 は対数差で直接計算する。
    internal static double Bd0(double x, double m)
    {
        if (x == m)
        {
            return 0;
        }

        double ratio = m / x;
        if (ratio >= 0.5 && double.IsFinite(ratio))
        {
            double u = (m - x) / x;
            if (double.IsFinite(u))
            {
                return -x * Log1pmx(u);
            }
        }

        double logRatio = Math.Log(x) - Math.Log(m);
        return x * logRatio + (m - x);
    }

    // Loader (2000) の鞍点展開による ln(C(n, x) p^x q^(n-x))（実数 x, n にも対応）
    internal static double LogBinomialProbability(double x, double n, double p, double q)
    {
        if (x < 0 || x > n)
        {
            return double.NegativeInfinity;
        }

        if (p == 0)
        {
            return x == 0 ? 0 : double.NegativeInfinity;
        }

        if (q == 0)
        {
            return x == n ? 0 : double.NegativeInfinity;
        }

        if (x == 0)
        {
            return n == 0 ? 0 : n * (p < 0.5 ? Log1p(-p) : Math.Log(q));
        }

        if (x == n)
        {
            return n * (q < 0.5 ? Log1p(-q) : Math.Log(p));
        }

        return LogBinomialProbabilityCore(x, n - x, n, p, q);
    }

    // 失敗回数 y = n - x を明示的に受け取る内部計算（負の二項分布での n = r + k の丸め誤差を避ける）
    internal static double LogBinomialProbabilityCore(double x, double y, double n, double p, double q)
    {
        double lc = StirlingRemainder(n) - StirlingRemainder(x) - StirlingRemainder(y)
            - Bd0(x, n * p) - Bd0(y, n * q);
        double lf = LnTwoPi + Math.Log(x) + Math.Log(y) - Math.Log(n);
        return lc - 0.5 * lf;
    }

    // Loader (2000) の鞍点展開による ln(e^(-λ) λ^x / x!)
    internal static double LogPoissonProbability(double x, double lambda)
    {
        if (x < 0)
        {
            return double.NegativeInfinity;
        }

        if (lambda == 0)
        {
            return x == 0 ? 0 : double.NegativeInfinity;
        }

        if (x == 0)
        {
            return -lambda;
        }

        if (double.IsPositiveInfinity(lambda))
        {
            return double.NegativeInfinity;
        }

        return -StirlingRemainder(x) - Bd0(x, lambda) - 0.5 * (LnTwoPi + Math.Log(x));
    }

    // ln x から ln I_x(a, b) を求める。x < MinNormal の極裾は先頭項 x^a / (a B(a, b)) で近似する
    internal static double LogRegularizedIncompleteBetaFromLog(double a, double b, double logX)
    {
        if (logX >= Math.Log(MinNormal))
        {
            return XMath.Special.LogRegularizedIncompleteBeta(a, b, Math.Exp(logX));
        }

        return a * logX - Math.Log(a) - XMath.Special.LogBeta(a, b);
    }

    // ln x から ln P(a, x) を求める。x < MinNormal の極裾は先頭項 x^a / Γ(a + 1) で近似する
    internal static double LogRegularizedGammaPFromLog(double a, double logX)
    {
        if (logX >= Math.Log(MinNormal))
        {
            return XMath.Special.LogRegularizedGammaP(a, Math.Exp(logX));
        }

        return a * logX - XMath.Special.LogGamma1p(a);
    }

    internal delegate double ValueAndDerivative(double t, out double derivative);

    // 単調増加関数 function(t) = target を区間保持ニュートン法（二分法フォールバック付き）で解く
    internal static double SolveIncreasing(ValueAndDerivative function, double target, double guess, double step,
        double lower = double.NegativeInfinity, double upper = double.PositiveInfinity)
    {
        guess = Math.Clamp(guess, lower, upper);
        double value = function(guess, out double derivative) - target;
        if (value == 0)
        {
            return guess;
        }

        // 符号の変わる区間 [lo, hi] を、初期値から幅を倍にしながら探す
        double lo, hi, valueLo, valueHi;
        if (value < 0)
        {
            lo = guess;
            valueLo = value;
            hi = guess;
            valueHi = value;
            for (int i = 0; i < 2100 && valueHi < 0; i++)
            {
                lo = hi;
                valueLo = valueHi;
                hi = Math.Min(hi + step, upper);
                valueHi = function(hi, out _) - target;
                step *= 2;
                if (hi == upper)
                {
                    break;
                }
            }

            if (valueHi < 0)
            {
                return upper;
            }
        }
        else
        {
            hi = guess;
            valueHi = value;
            lo = guess;
            valueLo = value;
            for (int i = 0; i < 2100 && valueLo > 0; i++)
            {
                hi = lo;
                valueHi = valueLo;
                lo = Math.Max(lo - step, lower);
                valueLo = function(lo, out _) - target;
                step *= 2;
                if (lo == lower)
                {
                    break;
                }
            }

            if (valueLo > 0)
            {
                return lower;
            }
        }

        if (valueLo == 0)
        {
            return lo;
        }

        if (valueHi == 0)
        {
            return hi;
        }

        // 区間の中でニュートン法。区間の外へ出たり進みが遅かったりする場合は二分する
        double t = value < 0 ? lo : hi;
        double f = value < 0 ? valueLo : valueHi;
        function(t, out derivative);
        double previousWidth = hi - lo;
        for (int i = 0; i < 400; i++)
        {
            double next = t - f / derivative;
            if (!(next > lo && next < hi) || !double.IsFinite(next))
            {
                next = 0.5 * (lo + hi);
            }

            if (next == t || next == lo || next == hi)
            {
                break;
            }

            double fNext = function(next, out double derivativeNext) - target;
            if (fNext == 0)
            {
                return next;
            }

            if (fNext < 0)
            {
                lo = next;
            }
            else
            {
                hi = next;
            }

            double change = Math.Abs(next - t);
            t = next;
            f = fNext;
            derivative = derivativeNext;
            if (change <= 2 * Epsilon * Math.Abs(t) || hi - lo <= 2 * Epsilon * Math.Max(Math.Abs(lo), Math.Abs(hi)))
            {
                break;
            }

            // 区間が半分以下に縮まないことが続くなら、次は二分する
            double width = hi - lo;
            if (width > 0.5 * previousWidth)
            {
                derivative = double.PositiveInfinity;
                f = 0;
            }

            previousWidth = width;
        }

        return t;
    }

    // 単調述語を満たす最小整数を、指数的ブラケット探索と二分探索で求める
    internal static long SmallestSatisfying(Func<long, bool> predicate, long guess, long minimum, long maximum)
    {
        guess = Math.Clamp(guess, minimum, maximum);
        long lo, hi;
        if (predicate(guess))
        {
            hi = guess;
            long step = 1;
            lo = guess;
            while (true)
            {
                if (lo == minimum)
                {
                    return minimum;
                }

                long candidate = lo - step < minimum || lo - step > lo ? minimum : lo - step;
                if (!predicate(candidate))
                {
                    lo = candidate;
                    break;
                }

                hi = candidate;
                lo = candidate;
                step = step > long.MaxValue / 2 ? long.MaxValue : step * 2;
            }
        }
        else
        {
            lo = guess;
            long step = 1;
            while (true)
            {
                if (lo == maximum)
                {
                    return maximum;
                }

                long candidate = lo + step > maximum || lo + step < lo ? maximum : lo + step;
                if (predicate(candidate))
                {
                    hi = candidate;
                    break;
                }

                lo = candidate;
                step = step > long.MaxValue / 2 ? long.MaxValue : step * 2;
            }
        }

        // 不変条件: predicate(lo) は偽、predicate(hi) は真
        while (hi - lo > 1)
        {
            long middle = lo + (hi - lo) / 2;
            if (predicate(middle))
            {
                hi = middle;
            }
            else
            {
                lo = middle;
            }
        }

        return hi;
    }
}
