using Tuna.Integration;
using Tuna.LinearAlgebra;

namespace Tuna.Tests.Integration;

/// <summary>
/// Genz（1984, "Testing multidimensional integration routines"）の6種類の検査関数。パラメータと参照値は
/// <see cref="IntegrationReferences.Genz"/>（mpmath の閉形式）
/// </summary>
internal readonly struct GenzFunction(GenzCase c) : IMultivariateIntegrand
{
    private readonly double[] a = c.A;
    private readonly double[] u = c.U;
    private readonly string family = c.Family;

    public double Evaluate(ReadOnlyVectorSpan<double> x)
    {
        int d = a.Length;
        switch (family)
        {
            case "Oscillatory":
                {
                    double s = 2 * Math.PI * u[0];
                    for (int i = 0; i < d; i++)
                    {
                        s += a[i] * x[i];
                    }

                    return Math.Cos(s);
                }
            case "ProductPeak":
                {
                    double p = 1;
                    for (int i = 0; i < d; i++)
                    {
                        p /= 1 / (a[i] * a[i]) + (x[i] - u[i]) * (x[i] - u[i]);
                    }

                    return p;
                }
            case "CornerPeak":
                {
                    double s = 1;
                    for (int i = 0; i < d; i++)
                    {
                        s += a[i] * x[i];
                    }

                    return Math.Pow(s, -(d + 1));
                }
            case "Gaussian":
                {
                    double s = 0;
                    for (int i = 0; i < d; i++)
                    {
                        s += a[i] * a[i] * (x[i] - u[i]) * (x[i] - u[i]);
                    }

                    return Math.Exp(-s);
                }
            case "Continuous":
                {
                    double s = 0;
                    for (int i = 0; i < d; i++)
                    {
                        s += a[i] * Math.Abs(x[i] - u[i]);
                    }

                    return Math.Exp(-s);
                }
            case "Discontinuous":
                {
                    if (x[0] > u[0] || x[1] > u[1])
                    {
                        return 0;
                    }

                    double s = 0;
                    for (int i = 0; i < d; i++)
                    {
                        s += a[i] * x[i];
                    }

                    return Math.Exp(s);
                }
            default:
                throw new InvalidOperationException(family);
        }
    }

    public static IEnumerable<GenzCase> Cases(params string[] families) =>
        IntegrationReferences.Genz.Where(c => families.Contains(c.Family));
}
