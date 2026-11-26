namespace Tuna.Statistics.Distributions;

// ν = 1 の位置・尺度の微分はコーシー分布でも使う。
internal readonly struct StudentTLogPdfGradient(double nu, double location, double scale, bool requiresNu = false)
{
    private const double AsymptoticLimit = 32;
    private readonly double normalizer = requiresNu ? NormalizerDerivative(nu) : 0;

    internal (ScaledDerivative Location, ScaledDerivative Scale) LocationScale(double x, double t, bool requiresLocation, bool requiresScale)
    {
        if (!requiresLocation && !requiresScale)
        {
            return default;
        }

        if (double.IsNaN(x))
        {
            return (new(double.NaN), new(double.NaN));
        }

        if (double.IsInfinity(x))
        {
            int numeratorPower = Math.ILogB(nu);
            int denominatorPower = Math.ILogB(scale);
            return (default, new(Math.ScaleB(nu, -numeratorPower) / Math.ScaleB(scale, -denominatorPower), numeratorPower - denominatorPower));
        }

        double absolute = Math.Abs(t);
        if (scale is >= 1e-100 and <= 1e100 && nu is >= 1e-100 and <= 1e100
            && (absolute is >= 1e-100 and <= 1e100 || x == location))
        {
            double denominator = Math.FusedMultiplyAdd(t, t, nu);
            double dLocation = requiresLocation ? ((nu + 1) * t / denominator) / scale : 0;
            double dScale = 0;
            if (requiresScale)
            {
                dScale = absolute is >= 0.75 and <= 1.5
                    ? ((nu / denominator) / scale) * (ScaleOffset(x, t) * (absolute + 1))
                    : (nu * Math.FusedMultiplyAdd(t, t, -1) / denominator) / scale;
            }

            return (new(dLocation), new(dScale));
        }

        return ScaledLocationScale(x, t, requiresLocation, requiresScale);
    }

    private (ScaledDerivative Location, ScaledDerivative Scale) ScaledLocationScale(double x, double t, bool requiresLocation, bool requiresScale)
    {
        if (x == location)
        {
            int power = Math.ILogB(scale);
            return (default, new(-1 / Math.ScaleB(scale, -power), -power));
        }

        var (residual, _, shift) = Difference(x);
        int rPower = Math.ILogB(residual) + shift;
        int sPower = Math.ILogB(scale);
        int nuPower = Math.ILogB(nu);
        double r = Math.ScaleB(residual, shift - rPower);
        double s = Math.ScaleB(scale, -sPower);
        double n = Math.ScaleB(nu, -nuPower);

        // νσ² + r² を最大の項の指数で割ってから、仮数の和を求める。
        int rSquarePower = 2 * rPower;
        int sSquarePower = 2 * sPower;
        int denominatorPower = Math.Max(nuPower + sSquarePower, rSquarePower);
        double denominator = Math.ScaleB(n * s * s, nuPower + sSquarePower - denominatorPower)
            + Math.ScaleB(r * r, rSquarePower - denominatorPower);
        ScaledDerivative dLocation = default, dScale = default;
        if (requiresLocation)
        {
            int coefficientPower = Math.Max(nuPower, 0);
            double coefficient = Math.ScaleB(nu + 1, -coefficientPower);
            dLocation = new(coefficient * r / denominator, coefficientPower + rPower - denominatorPower);
        }

        if (requiresScale)
        {
            int differencePower = Math.Max(rSquarePower, sSquarePower);
            double difference = Math.ScaleB(r * r, rSquarePower - differencePower)
                - Math.ScaleB(s * s, sSquarePower - differencePower);
            if (Math.Abs(t) is >= 0.75 and <= 1.5)
            {
                // r² − σ² = σ² (|t| − 1)(|t| + 1)。桁落ちが生じる最初の因子は差の下位桁から求める。
                difference = ScaleOffset(x, t) * (Math.Abs(t) + 1) * s * s;
                differencePower = sSquarePower;
            }

            dScale = new(n * difference / (s * denominator), nuPower + differencePower - sPower - denominatorPower);
        }

        return (dLocation, dScale);
    }

    private double ScaleOffset(double x, double t)
    {
        var (residual, error, shift) = Difference(x);
        double scaled = Math.ScaleB(scale, -shift);
        return ((Math.Abs(residual) - scaled) + (t < 0 ? -error : error)) / scaled;
    }

    private (double Residual, double Error, int Shift) Difference(double x)
    {
        double center = location;
        double residual = x - center;
        int shift = 0;
        if (double.IsInfinity(residual))
        {
            x *= 0.5;
            center *= 0.5;
            residual = x - center;
            shift = 1;
        }

        double error = CompensatedSum.DifferenceError(x, center, residual);
        return (residual, error, shift);
    }

    internal ScaledDerivative DegreesOfFreedom(double x, double t)
    {
        if (double.IsNaN(x))
        {
            return new(double.NaN);
        }

        if (double.IsInfinity(x))
        {
            return new(double.NegativeInfinity);
        }

        double square = t * t;
        double u = square / nu;
        if (double.IsFinite(t) && !double.IsNormal(square))
        {
            u = DistributionMath.ScaledRatio([t, t], [nu]);
        }

        if (nu >= AsymptoticLimit && square <= 16)
        {
            // 残る項をすべて 1/ν² の係数にそろえてから差を取る。
            double coefficient = Math.FusedMultiplyAdd(-0.5 * square * square, LogTermCoefficient(u),
                normalizer + 0.5 * square / (1 + u));
            int power = Math.ILogB(nu);
            double mantissa = Math.ScaleB(nu, -power);
            return new(coefficient / (mantissa * mantissa), -2 * power);
        }

        if (nu >= AsymptoticLimit && u <= 0.5)
        {
            // 大きな t では u² をくくり、観測の和をとるまでその指数を保つ。
            double coefficient = (normalizer / square) / square + 0.5 / (square * (1 + u)) - 0.5 * LogTermCoefficient(u);
            int power = Math.ILogB(u);
            double mantissa = Math.ScaleB(u, -power);
            return new(coefficient * mantissa * mantissa, 2 * power);
        }

        double weight = double.IsPositiveInfinity(u) ? 1 : u / (1 + u);
        double difference = u <= 0.5
            ? -(u * u) * LogTermCoefficient(u)
            : weight - LogOnePlusRatio(x, u);
        if (nu < 1)
        {
            // νA′(ν) と同じスケールで加算し、1/ν がオーバーフローする場合も最後に丸める。
            int power = Math.ILogB(nu);
            double mantissa = Math.ScaleB(nu, -power);
            return new((normalizer + 0.5 * weight + (0.5 * difference) * nu) / mantissa, -power);
        }

        double normalization = nu >= AsymptoticLimit ? normalizer / nu / nu : normalizer;
        return new(normalization + 0.5 * weight / nu + 0.5 * difference);
    }

    private double LogOnePlusRatio(double x, double u)
    {
        if (double.IsFinite(u))
        {
            return DistributionMath.Log1p(u);
        }

        var (residual, _, shift) = Difference(x);
        return 2 * (Math.Log(Math.Abs(residual)) + shift * DistributionMath.Ln2 - Math.Log(scale)) - Math.Log(nu);
    }

    // [ln(1 + u) − u/(1 + u)]/u²。u が小さい方では級数の先頭 5 項。
    private static double LogTermCoefficient(double u) =>
        u < 1e-4
            ? 0.5 + u * (-2.0 / 3 + u * (0.75 + u * (-0.8 + u * (5.0 / 6))))
            : 1 / (1 + u) + DistributionMath.Log1pmx(u) / (u * u);

    // A(ν) = ln Γ((ν+1)/2) − ln Γ(ν/2) − (ln ν)/2。
    // A′(ν) = A′(ν+2) + 1/(ν(ν+1)(ν+2))。ν < 1 では ν A′(ν)、ν ≥ 32 では ν² A′(ν) を持つ。
    private static double NormalizerDerivative(double nu)
    {
        if (nu < 1)
        {
            return 1 / ((nu + 1) * (nu + 2)) + nu * NormalizerDerivative(nu + 2);
        }

        if (nu >= AsymptoticLimit)
        {
            return AsymptoticCoefficient(nu);
        }

        double sum = 0;
        for (; nu < AsymptoticLimit; nu += 2)
        {
            sum += (1 / ((nu + 1) * (nu + 2))) / nu;
        }

        return sum + AsymptoticCoefficient(nu) / (nu * nu);
    }

    // 係数は Bernoulli 数 B₂ₖ による (2²ᵏ − 1)B₂ₖ/(2k)（k = 1〜8）。
    private static double AsymptoticCoefficient(double nu)
    {
        double inverse = 1 / nu;
        double q = inverse * inverse;
        return 0.25 + q * (-0.125 + q * (0.25 + q * (-1.0625 + q * (7.75 + q * (-86.375 + q * (1365.25 + q * (-929569.0 / 32)))))));
    }
}
