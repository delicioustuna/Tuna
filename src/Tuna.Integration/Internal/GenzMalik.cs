using System.Numerics;

namespace Tuna.Integration.Internal;

// Genz & Malik (1980, https://doi.org/10.1016/0771-050X(80)90039-X) の7次・5次求積則（2^d + 2d² + 2d + 1 点、全節点が領域内部）。
// 誤差見積もりは |I₇ − I₅|（下限 50ε·V·Σ|wₖ f(xₖ)|）、分割軸のスコアは各軸の4階差分。
internal sealed class GenzMalik : ICubatureRule
{
    private static readonly double Lambda2 = Math.Sqrt(9.0 / 70);
    private static readonly double Lambda4 = Math.Sqrt(9.0 / 10);
    private static readonly double Lambda5 = Math.Sqrt(9.0 / 19);

    // (λ₂/λ₄)² = 1/7
    private const double DifferenceRatio = 1.0 / 7;

    private readonly int d;
    private readonly double w1, w2, w3, w4, w5;
    private readonly double v1, v2, v3, v4;
    private readonly double[] point;
    private readonly double[] fourthDifference;

    internal GenzMalik(int dimension)
    {
        d = dimension;
        double n = dimension;
        w1 = (12824 - 9120 * n + 400 * n * n) / 19683;
        w2 = 980.0 / 6561;
        w3 = (1820 - 400 * n) / 19683;
        w4 = 200.0 / 19683;
        w5 = 6859.0 / 19683 / Math.Pow(2, dimension);
        v1 = (729 - 950 * n + 50 * n * n) / 729;
        v2 = 245.0 / 486;
        v3 = (265 - 100 * n) / 1458;
        v4 = 25.0 / 729;
        point = new double[dimension];
        fourthDifference = new double[dimension];
        Points = (1L << dimension) + 2L * dimension * dimension + 2L * dimension + 1;
    }

    public long Points { get; }

    // 直前の Apply で計算した各軸の4階差分
    public ReadOnlySpan<double> AxisScores => fourthDifference;

    public bool Apply<TIntegrand>(ref TIntegrand f, ReadOnlySpan<double> lower, ReadOnlySpan<double> upper, out RuleEstimate estimate)
        where TIntegrand : IBoxIntegrand
    {
        Span<double> x = point;
        double volume = 1;
        for (int i = 0; i < d; i++)
        {
            x[i] = 0.5 * lower[i] + 0.5 * upper[i];
            volume *= upper[i] - lower[i];
        }

        double f0 = f.Evaluate(x);
        bool finite = double.IsFinite(f0);
        double sum2 = 0, sum3 = 0, abs2 = 0, abs3 = 0;
        for (int i = 0; i < d; i++)
        {
            double c = x[i];
            double h = 0.5 * upper[i] - 0.5 * lower[i];
            x[i] = c - Lambda2 * h;
            double a = f.Evaluate(x);
            x[i] = c + Lambda2 * h;
            double b = f.Evaluate(x);
            x[i] = c - Lambda4 * h;
            double p = f.Evaluate(x);
            x[i] = c + Lambda4 * h;
            double q = f.Evaluate(x);
            x[i] = c;
            finite &= double.IsFinite(a) & double.IsFinite(b) & double.IsFinite(p) & double.IsFinite(q);
            double two = a + b, three = p + q;
            sum2 += two;
            sum3 += three;
            abs2 += Math.Abs(a) + Math.Abs(b);
            abs3 += Math.Abs(p) + Math.Abs(q);
            fourthDifference[i] = Math.Abs(two - 2 * f0 - DifferenceRatio * (three - 2 * f0));
        }

        double sum4 = 0, abs4 = 0;
        for (int i = 0; i < d; i++)
        {
            double ci = x[i];
            double hi = Lambda4 * (0.5 * upper[i] - 0.5 * lower[i]);
            for (int j = i + 1; j < d; j++)
            {
                double cj = x[j];
                double hj = Lambda4 * (0.5 * upper[j] - 0.5 * lower[j]);
                for (int s = 0; s < 4; s++)
                {
                    x[i] = (s & 1) == 0 ? ci - hi : ci + hi;
                    x[j] = (s & 2) == 0 ? cj - hj : cj + hj;
                    double v = f.Evaluate(x);
                    finite &= double.IsFinite(v);
                    sum4 += v;
                    abs4 += Math.Abs(v);
                }

                x[j] = cj;
            }

            x[i] = ci;
        }

        // 2^d 個の頂点方向の点を、1軸ずつ符号を変える順（Gray code）で回る
        Span<double> centers = stackalloc double[d];
        Span<double> offsets = stackalloc double[d];
        for (int i = 0; i < d; i++)
        {
            centers[i] = x[i];
            offsets[i] = Lambda5 * (0.5 * upper[i] - 0.5 * lower[i]);
            x[i] = centers[i] + offsets[i];
        }

        double sum5 = 0, abs5 = 0;
        long corners = 1L << d;
        for (long m = 0; m < corners; m++)
        {
            if (m > 0)
            {
                // Gray code g = m ⊕ (m >> 1) の第 k ビットが立っている軸は負の側
                int k = BitOperations.TrailingZeroCount(m);
                long gray = m ^ (m >> 1);
                x[k] = ((gray >> k) & 1) != 0 ? centers[k] - offsets[k] : centers[k] + offsets[k];
            }

            double v = f.Evaluate(x);
            finite &= double.IsFinite(v);
            sum5 += v;
            abs5 += Math.Abs(v);
        }

        if (!finite)
        {
            estimate = default;
            return false;
        }

        double value = volume * (w1 * f0 + w2 * sum2 + w3 * sum3 + w4 * sum4 + w5 * sum5);
        double lowOrder = volume * (v1 * f0 + v2 * sum2 + v3 * sum3 + v4 * sum4);
        double error = Math.Abs(value - lowOrder);
        double resabs = Math.Abs(volume) * (Math.Abs(w1) * Math.Abs(f0) + w2 * abs2 + Math.Abs(w3) * abs3 + w4 * abs4 + w5 * abs5);
        double roundoff = 50 * GaussKronrod.MachineEpsilon * resabs;
        bool atRoundoff = false;
        if (resabs > GaussKronrod.MinNormal / (50 * GaussKronrod.MachineEpsilon) && error <= roundoff)
        {
            error = roundoff;
            atRoundoff = true;
        }

        estimate = new RuleEstimate(value, error, atRoundoff);
        return double.IsFinite(value) && double.IsFinite(error);
    }
}

// 変数変換後の座標（t 空間）で評価する Cubature 内部用の被積分関数
internal interface IBoxIntegrand
{
    double Evaluate(Span<double> t);
}
