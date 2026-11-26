using Tuna.LinearAlgebra;
using Tuna.Optimization;

namespace Tuna.Tests.Optimization;

/// <summary>勾配法のテストに使う関数（値と勾配は式から直接書いたもの）</summary>
internal static class GradientTestFunctions
{
    /// <summary>拡張 Rosenbrock 関数 Σ 100(xᵢ₊₁ − xᵢ²)² + (1 − xᵢ)²（最小は全成分 1 で値 0）</summary>
    public readonly struct Rosenbrock : IDifferentiableObjectiveFunction
    {
        public double Evaluate(ReadOnlyVectorSpan<double> x)
        {
            double sum = 0;
            for (int i = 0; i + 1 < x.Length; i++)
            {
                double a = x[i + 1] - x[i] * x[i], b = 1 - x[i];
                sum += 100 * a * a + b * b;
            }

            return sum;
        }

        public double Evaluate(ReadOnlyVectorSpan<double> x, VectorSpan<double> gradient)
        {
            double sum = 0;
            for (int i = 0; i < x.Length; i++)
            {
                gradient[i] = 0;
            }

            for (int i = 0; i + 1 < x.Length; i++)
            {
                double a = x[i + 1] - x[i] * x[i], b = 1 - x[i];
                sum += 100 * a * a + b * b;
                gradient[i] += -400 * a * x[i] - 2 * b;
                gradient[i + 1] += 200 * a;
            }

            return sum;
        }
    }

    /// <summary>二次形式 ½xᵀAx − bᵀx（A は対称正定値。最小点は A⁻¹b）</summary>
    public sealed class Quadratic(Matrix<double> a, Vector<double> b) : IDifferentiableObjectiveFunction
    {
        public Matrix<double> A { get; } = a;

        public Vector<double> B { get; } = b;

        public double Evaluate(ReadOnlyVectorSpan<double> x, VectorSpan<double> gradient)
        {
            int n = x.Length;
            double value = 0;
            for (int i = 0; i < n; i++)
            {
                double ax = 0;
                for (int j = 0; j < n; j++)
                {
                    ax += A[i, j] * x[j];
                }

                gradient[i] = ax - B[i];
                value += x[i] * (0.5 * ax - B[i]);
            }

            return value;
        }
    }

    /// <summary>一様乱数 [-1, 1) の行列から作る対称正定値行列 MᵀM + shift·I</summary>
    public static Matrix<double> RandomPositiveDefinite(int n, double shift, int seed)
    {
        var random = new Random(seed);
        var m = new Matrix<double>(n, n);
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < n; i++)
            {
                m[i, j] = 2 * random.NextDouble() - 1;
            }
        }

        var a = new Matrix<double>(n, n);
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < n; i++)
            {
                double sum = 0;
                for (int k = 0; k < n; k++)
                {
                    sum += m[k, i] * m[k, j];
                }

                a[i, j] = sum + (i == j ? shift : 0);
            }
        }

        return a;
    }

    public static Vector<double> RandomVector(int n, int seed, double scale = 1)
    {
        var random = new Random(seed);
        var v = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            v[i] = scale * (2 * random.NextDouble() - 1);
        }

        return v;
    }

    public static double MaxAbsDifference(ReadOnlyVectorSpan<double> a, ReadOnlyVectorSpan<double> b)
    {
        double worst = 0;
        for (int i = 0; i < a.Length; i++)
        {
            worst = Math.Max(worst, Math.Abs(a[i] - b[i]));
        }

        return worst;
    }
}
