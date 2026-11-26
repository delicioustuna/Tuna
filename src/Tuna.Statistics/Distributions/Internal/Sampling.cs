using Tuna.LinearAlgebra;
using Tuna.RandomSource;

namespace Tuna.Statistics.Distributions;

internal static class Sampling
{
    private const double TwoPow64 = 18446744073709551616.0;

    // U と p の 2 進展開を 64 ビット単位で比較し、p が 2^-53 の倍数でなくても正確な成功確率を保つ（1 - 2^-64 の確率で 1 ワード消費）
    internal static bool Bernoulli(IRandomSource random, double p)
    {
        if (!(p > 0))
        {
            return false;
        }

        if (p >= 1)
        {
            return true;
        }

        double remainder = p;
        while (true)
        {
            // remainder · 2⁶⁴ は2の累乗倍なので正確で、整数部と小数部への分解も正確
            remainder *= TwoPow64;
            double whole = Math.Floor(remainder);
            ulong digits = (ulong)whole;
            ulong drawn = random.NextUInt64();
            if (drawn != digits)
            {
                return drawn < digits;
            }

            remainder -= whole;
            if (remainder == 0)
            {
                // p の展開がここで終わる: 残りの桁がすべて0でない限り U > p（U = p の確率は0）
                return false;
            }
        }
    }

    internal static double StandardGamma(IRandomSource random, double shape) =>
        shape >= 1 ? MarsagliaTsang(random, shape) : Math.Exp(LogStandardGamma(random, shape));

    // shape < 1 では Marsaglia & Tsang (2000) 第 6 節の G(a) = G(a + 1) U^(1/a) を対数空間で計算しアンダーフローを避ける
    internal static double LogStandardGamma(IRandomSource random, double shape) => LogStandardGamma(random, shape, out _);

    // logUniform には shape < 1 のときに使った ln U を返す（shape >= 1 では NaN）
    internal static double LogStandardGamma(IRandomSource random, double shape, out double logUniform)
    {
        if (shape >= 1)
        {
            logUniform = double.NaN;
            return Math.Log(MarsagliaTsang(random, shape));
        }

        double logGamma = Math.Log(MarsagliaTsang(random, shape + 1));
        logUniform = Math.Log(random.NextDoubleOpen());
        return logGamma + logUniform / shape;
    }

    // Marsaglia & Tsang (2000) による標準ガンマ乱数生成（shape >= 1）
    private static double MarsagliaTsang(IRandomSource random, double shape)
    {
        double d = shape - 1.0 / 3;
        double c = 1 / Math.Sqrt(9 * d);
        while (true)
        {
            double x, v;
            do
            {
                x = Ziggurat.StandardNormal(random);
                v = 1 + c * x;
            }
            while (v <= 0);

            v = v * v * v;
            double u = random.NextDoubleOpen();
            double x2 = x * x;
            if (u < 1 - 0.0331 * x2 * x2)
            {
                return d * v;
            }

            if (Math.Log(u) < 0.5 * x2 + d * (1 - v + Math.Log(v)))
            {
                return d * v;
            }
        }
    }

    // ポアソン乱数を long で表現できる平均の上限
    internal const double MaxPoissonRate = 1e18;

    // rate < 10 は逆関数法、10 <= rate <= MaxPoissonRate は Hörmann (1993) の PTRS（Loader (2000) の鞍点展開で受理判定）。
    // rate > MaxPoissonRate（負の二項分布のガンマ・ポアソン混合で起こりうる）は正規近似を丸めて long.MaxValue で飽和させる。
    internal static long Poisson(IRandomSource random, double rate)
    {
        if (!(rate > 0))
        {
            return 0;
        }

        if (rate < 10)
        {
            return PoissonInversion(random, rate);
        }

        if (rate > MaxPoissonRate)
        {
            if (double.IsPositiveInfinity(rate))
            {
                // ∞ + √∞·z は z < 0 で NaN になる
                return long.MaxValue;
            }

            double value = Math.Round(rate + Math.Sqrt(rate) * Ziggurat.StandardNormal(random));
            return value >= 9.2233720368547758e18 ? long.MaxValue : (long)value;
        }

        return PoissonPtrs(random, rate);
    }

    private static long PoissonInversion(IRandomSource random, double rate)
    {
        double start = Math.Exp(-rate);
        while (true)
        {
            double u = random.NextDouble();
            double probability = start;
            double cumulative = probability;
            long k = 0;
            while (u >= cumulative && k < 200)
            {
                k++;
                probability *= rate / k;
                cumulative += probability;
            }

            // 累積和の丸めで u に届かなかった場合（確率は 2⁻⁵³ 程度）は引き直す
            if (k < 200)
            {
                return k;
            }
        }
    }

    private static long PoissonPtrs(IRandomSource random, double rate)
    {
        double sqrtRate = Math.Sqrt(rate);
        double b = 0.931 + 2.53 * sqrtRate;
        double a = -0.059 + 0.02483 * b;
        double inverseAlpha = 1.1239 + 1.1328 / (b - 3.4);
        double vr = 0.9277 - 3.6224 / (b - 2);
        double logInverseAlpha = Math.Log(inverseAlpha);
        while (true)
        {
            double u = random.NextDoubleOpen() - 0.5;
            double v = random.NextDouble();
            double us = 0.5 - Math.Abs(u);
            double kd = Math.Floor((2 * a / us + b) * u + rate + 0.43);
            if (us >= 0.07 && v <= vr)
            {
                return (long)kd;
            }

            if (kd < 0 || kd > 9.2e18 || (us < 0.013 && v > us))
            {
                continue;
            }

            if (Math.Log(v) + logInverseAlpha - Math.Log(a / (us * us) + b)
                <= DistributionMath.LogPoissonProbability(kd, rate))
            {
                return (long)kd;
            }
        }
    }

    // 対数標準ガンマから正規化前の重みとその総和を求める。
    // 全成分で ln U / α がアンダーフローする極小 α では、極限として -ln U / α が最小の成分のみを 1 とする。
    internal static double DirichletWeights(IRandomSource random, ReadOnlySpan<double> alpha, Span<double> weights, Span<double> logUniform)
    {
        double largest = double.NegativeInfinity;
        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] = LogStandardGamma(random, alpha[i], out logUniform[i]);
            largest = Math.Max(largest, weights[i]);
        }

        if (double.IsNegativeInfinity(largest))
        {
            // 全成分で ln U / α が -∞ にオーバーフローした極限: -ln U / α が最小の成分を 1 とする
            int winner = 0;
            double smallest = double.PositiveInfinity;
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = 0;
                double key = Math.Log(-logUniform[i]) - Math.Log(alpha[i]);
                if (key < smallest)
                {
                    smallest = key;
                    winner = i;
                }
            }

            weights[winner] = 1;
            return 1;
        }

        double sum = 0;
        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] = Math.Exp(weights[i] - largest);
            sum += weights[i];
        }

        return sum;
    }

    // 条件付き二項分布 X_i ~ B(remaining, w_i / suffix[i]) を順次生成する（残り試行回数が 0 になったら乱数を消費しない）
    internal static void Multinomial(IRandomSource random, long trials, ReadOnlySpan<double> weights, ReadOnlySpan<double> suffix, VectorSpan<long> destination)
    {
        long remaining = trials;
        int last = weights.Length - 1;
        for (int i = 0; i < last; i++)
        {
            long count = 0;
            if (remaining > 0 && weights[i] > 0)
            {
                // 後続の重みがすべて 0 なら残りの試行回数をすべて割り当てる
                double rest = suffix[i + 1];
                count = rest > 0 ? Binomial(random, remaining, weights[i] / suffix[i], rest / suffix[i]) : remaining;
            }

            destination[i] = count;
            remaining -= count;
        }

        destination[last] = remaining;
    }

    // p <= 1/2 側で生成し、p > 1/2 は n - X に反転する。
    // n * min(p, q) < 10 は逆関数法、それ以上は Hörmann (1993) の BTRS（Loader (2000) の鞍点展開で受理判定）。
    internal static long Binomial(IRandomSource random, long trials, double p, double q)
    {
        if (trials == 0 || p == 0)
        {
            return 0;
        }

        if (q == 0)
        {
            return trials;
        }

        bool flip = p > 0.5;
        if (flip)
        {
            (p, q) = (q, p);
        }

        long k = trials * p < 10 ? BinomialInversion(random, trials, p, q) : BinomialBtrs(random, trials, p, q);
        return flip ? trials - k : k;
    }

    private static long BinomialInversion(IRandomSource random, long trials, double p, double q)
    {
        double n = trials;
        double start = Math.Exp(n * DistributionMath.Log1p(-p));
        double ratio = p / q;
        while (true)
        {
            double u = random.NextDouble();
            double probability = start;
            double cumulative = probability;
            long k = 0;
            while (u >= cumulative && k < trials && k < 200)
            {
                probability *= ratio * (n - k) / (k + 1);
                k++;
                cumulative += probability;
            }

            if (u < cumulative || k == trials)
            {
                return k;
            }
        }
    }

    private static long BinomialBtrs(IRandomSource random, long trials, double p, double q)
    {
        double n = trials;
        double spq = Math.Sqrt(n * p * q);
        double b = 1.15 + 2.53 * spq;
        double a = -0.0873 + 0.0248 * b + 0.01 * p;
        double c = n * p + 0.5;
        double vr = 0.92 - 4.2 / b;
        double alpha = (2.83 + 5.1 / b) * spq;
        double mode = Math.Floor((n + 1) * p);
        double logModeProbability = DistributionMath.LogBinomialProbability(mode, n, p, q);
        while (true)
        {
            double u = random.NextDoubleOpen() - 0.5;
            double v = random.NextDouble();
            double us = 0.5 - Math.Abs(u);
            double kd = Math.Floor((2 * a / us + b) * u + c);
            if (kd < 0 || kd > n)
            {
                continue;
            }

            if (us >= 0.07 && v <= vr)
            {
                return (long)kd;
            }

            if (Math.Log(v * alpha / (a / (us * us) + b))
                <= DistributionMath.LogBinomialProbability(kd, n, p, q) - logModeProbability)
            {
                return (long)kd;
            }
        }
    }
}
