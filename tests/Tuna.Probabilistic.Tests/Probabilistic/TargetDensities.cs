using Tuna.LinearAlgebra;
using Tuna.Probabilistic.Inference;

namespace Tuna.Tests.Probabilistic;

/// <summary>標本の平均と共分散が閉じた形で分かる目標分布</summary>
internal static class TargetDensities
{
    /// <summary>独立な正規分布 N(μᵢ, σᵢ²)</summary>
    internal readonly struct IndependentNormal(double[] mean, double[] sd) : IDifferentiableLogDensity
    {
        public double LogDensity(ReadOnlyVectorSpan<double> x)
        {
            double sum = 0;
            for (int i = 0; i < mean.Length; i++)
            {
                double z = (x[i] - mean[i]) / sd[i];
                sum -= 0.5 * z * z;
            }

            return sum;
        }

        public double LogDensity(ReadOnlyVectorSpan<double> x, VectorSpan<double> g)
        {
            double sum = 0;
            for (int i = 0; i < mean.Length; i++)
            {
                double z = (x[i] - mean[i]) / sd[i];
                sum -= 0.5 * z * z;
                g[i] = -z / sd[i];
            }

            return sum;
        }
    }

    /// <summary>2次元の相関のある正規分布（平均 μ、標準偏差 σ₁・σ₂、相関 ρ）</summary>
    internal readonly struct CorrelatedNormal(double mu1, double mu2, double s1, double s2, double rho) : IDifferentiableLogDensity
    {
        public double LogDensity(ReadOnlyVectorSpan<double> x)
        {
            double a = (x[0] - mu1) / s1, b = (x[1] - mu2) / s2;
            return -0.5 * (a * a - 2 * rho * a * b + b * b) / (1 - rho * rho);
        }

        public double LogDensity(ReadOnlyVectorSpan<double> x, VectorSpan<double> g)
        {
            double a = (x[0] - mu1) / s1, b = (x[1] - mu2) / s2, c = 1 - rho * rho;
            g[0] = -(a - rho * b) / c / s1;
            g[1] = -(b - rho * a) / c / s2;
            return -0.5 * (a * a - 2 * rho * a * b + b * b) / c;
        }

        public Matrix<double> Covariance => new(new[,] { { s1 * s1, rho * s1 * s2 }, { rho * s1 * s2, s2 * s2 } });
    }

    /// <summary>
    /// ねじれた正規分布（Haario, Saksman &amp; Tamminen 1999 の "banana"）。y = (x₁, x₂ + b(x₁² − 100), x₃, …) が N(0, diag(100, 1, 1, …))。
    /// 平均は 0、Var(x₁) = 100、Var(x₂) = 1 + 2·10⁴b²、Cov(x₁, x₂) = 0
    /// </summary>
    internal readonly struct Banana(int dimension, double b) : IDifferentiableLogDensity
    {
        public double LogDensity(ReadOnlyVectorSpan<double> x)
        {
            double y2 = x[1] + b * (x[0] * x[0] - 100);
            double sum = -0.5 * (x[0] * x[0] / 100 + y2 * y2);
            for (int i = 2; i < dimension; i++)
            {
                sum -= 0.5 * x[i] * x[i];
            }

            return sum;
        }

        public double LogDensity(ReadOnlyVectorSpan<double> x, VectorSpan<double> g)
        {
            double y2 = x[1] + b * (x[0] * x[0] - 100);
            g[0] = -x[0] / 100 - y2 * 2 * b * x[0];
            g[1] = -y2;
            double sum = -0.5 * (x[0] * x[0] / 100 + y2 * y2);
            for (int i = 2; i < dimension; i++)
            {
                sum -= 0.5 * x[i] * x[i];
                g[i] = -x[i];
            }

            return sum;
        }

        public double Variance2 => 1 + 2e4 * b * b;
    }

    /// <summary>2つの離れた山の等しい重みの混合 ½N(−c·1, I) + ½N(c·1, I)（d 次元）。平均 0、各成分の分散 1 + c²</summary>
    internal readonly struct TwoModes(int dimension, double c) : IDifferentiableLogDensity
    {
        public double LogDensity(ReadOnlyVectorSpan<double> x)
        {
            double a = 0, b = 0;
            for (int i = 0; i < dimension; i++)
            {
                a -= 0.5 * (x[i] + c) * (x[i] + c);
                b -= 0.5 * (x[i] - c) * (x[i] - c);
            }

            double max = Math.Max(a, b);
            return max + Math.Log(Math.Exp(a - max) + Math.Exp(b - max));
        }

        public double LogDensity(ReadOnlyVectorSpan<double> x, VectorSpan<double> g)
        {
            double a = 0, b = 0;
            for (int i = 0; i < dimension; i++)
            {
                a -= 0.5 * (x[i] + c) * (x[i] + c);
                b -= 0.5 * (x[i] - c) * (x[i] - c);
            }

            double max = Math.Max(a, b);
            double wa = Math.Exp(a - max), wb = Math.Exp(b - max), total = wa + wb;
            for (int i = 0; i < dimension; i++)
            {
                g[i] = (wa * -(x[i] + c) + wb * -(x[i] - c)) / total;
            }

            return max + Math.Log(total);
        }
    }

    /// <summary>単位超立方体 [0, 1]ᵈ 上の一様分布（台の外は −∞。不連続な密度）。平均 1/2、分散 1/12</summary>
    internal readonly struct UnitBox : ILogDensity
    {
        public double LogDensity(ReadOnlyVectorSpan<double> x)
        {
            for (int i = 0; i < x.Length; i++)
            {
                if (!(x[i] >= 0 && x[i] <= 1))
                {
                    return double.NegativeInfinity;
                }
            }

            return 0;
        }
    }

    /// <summary>Neal の漏斗 v ~ N(0, 3²)、x ~ N(0, eᵛ)（曲率が大きく変わり、ハミルトニアン法が発散しやすい）</summary>
    internal readonly struct Funnel : IDifferentiableLogDensity
    {
        public double LogDensity(ReadOnlyVectorSpan<double> p, VectorSpan<double> g)
        {
            double v = p[0], x = p[1], ev = Math.Exp(-v);
            g[0] = -v / 9 - 0.5 + 0.5 * x * x * ev;
            g[1] = -x * ev;
            return -v * v / 18 - 0.5 * v - 0.5 * x * x * ev;
        }
    }

    /// <summary>ベータ分布 Beta(a, b) の (0, 1) 上の対数密度と勾配。平均 a/(a + b)、分散 ab/((a + b)²(a + b + 1))</summary>
    internal readonly struct Beta(double a, double b) : IDifferentiableLogDensity
    {
        public double LogDensity(ReadOnlyVectorSpan<double> x, VectorSpan<double> g)
        {
            double p = x[0];
            g[0] = (a - 1) / p - (b - 1) / (1 - p);
            return (a - 1) * Math.Log(p) + (b - 1) * Math.Log(1 - p);
        }
    }

    /// <summary>ディリクレ分布 Dir(α) の単体上の対数密度（先頭 K − 1 成分についての密度。勾配は K 成分すべてについて書く）。平均 αₖ/Σα</summary>
    internal readonly struct Dirichlet(double[] alpha) : IDifferentiableLogDensity
    {
        public double LogDensity(ReadOnlyVectorSpan<double> x, VectorSpan<double> g)
        {
            double sum = 0;
            for (int k = 0; k < alpha.Length; k++)
            {
                sum += (alpha[k] - 1) * Math.Log(x[k]);
                g[k] = (alpha[k] - 1) / x[k];
            }

            return sum;
        }
    }
}
