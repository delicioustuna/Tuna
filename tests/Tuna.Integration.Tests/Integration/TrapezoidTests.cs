using Tuna.Integration;
using Tuna.LinearAlgebra;

namespace Tuna.Tests.Integration;

/// <summary>標本値の台形則（<see cref="Quadrature.Trapezoid(ReadOnlyVectorSpan{double}, double)"/> など）</summary>
public class TrapezoidTests
{
    [Fact]
    public void UniformSpacing()
    {
        Assert.Equal(4, Quadrature.Trapezoid(new double[] { 1.0, 2.0, 3.0 }));
        Assert.Equal(0.4, Quadrature.Trapezoid(new double[] { 1.0, 2.0, 3.0 }, 0.1), 1e-15);
        Assert.Equal(-4, Quadrature.Trapezoid(new double[] { 1.0, 2.0, 3.0 }, -1));
        Assert.Equal(0, Quadrature.Trapezoid(new double[] { 5.0 }));
        Assert.Equal(0, Quadrature.Trapezoid(ReadOnlySpan<double>.Empty));

        // 1次関数は正確
        var y = Enumerable.Range(0, 101).Select(i => 2 * (i * 0.01) + 1).ToArray();
        Assert.Equal(2, Quadrature.Trapezoid(y, 0.01), 1e-14);
    }

    [Fact]
    public void ErrorIsSecondOrderInTheSpacing()
    {
        // ∫₀^π sin = 2 の誤差 ≈ π h²/12·…。間隔を半分にすると誤差は 1/4
        double Error(int n)
        {
            double h = Math.PI / n;
            var y = Enumerable.Range(0, n + 1).Select(i => Math.Sin(i * h)).ToArray();
            return 2 - Quadrature.Trapezoid(y, h);
        }

        Assert.Equal(4, Error(100) / Error(200), 1e-3);
    }

    [Fact]
    public void ArbitraryPoints()
    {
        Assert.Equal(6.5, Quadrature.Trapezoid(new double[] { 1.0, 2.0, 3.0 }, new double[] { 0.0, 1.0, 3.0 }));
        // 減る区間は負の寄与
        Assert.Equal(-6.5, Quadrature.Trapezoid(new double[] { 3.0, 2.0, 1.0 }, new double[] { 3.0, 1.0, 0.0 }));
        Assert.Throws<ArgumentException>(() => Quadrature.Trapezoid(new double[] { 1.0, 2.0 }, new double[] { 0.0 }));
    }

    [Fact]
    public void Cumulative()
    {
        double[] y = [1, 2, 3, 4];
        Assert.Equal(new double[] { 0, 1.5, 4, 7.5 }, Quadrature.CumulativeTrapezoid(y).ToArray());
        Assert.Equal(new double[] { 0, 0.75, 2, 3.75 }, Quadrature.CumulativeTrapezoid(y, 0.5).ToArray());
        Assert.Equal(new double[] { 0, 1.5, 6.5, 13.5 }, Quadrature.CumulativeTrapezoid(y, new double[] { 0.0, 1.0, 3.0, 5.0 }).ToArray());
        Assert.Equal(Quadrature.Trapezoid(y, new double[] { 0.0, 1.0, 3.0, 5.0 }), Quadrature.CumulativeTrapezoid(y, new double[] { 0.0, 1.0, 3.0, 5.0 })[3]);
        Assert.Empty(Quadrature.CumulativeTrapezoid(ReadOnlySpan<double>.Empty).ToArray());
    }

    [Fact]
    public void CumulativeInPlaceAndOnStridedViews()
    {
        // 出力先が標本と同じ（その場）でも、ずれて重なっていても、別の出力先と同じ結果
        double[] y = [1, 2, 3, 4, 5];
        var expected = Quadrature.CumulativeTrapezoid(y, 0.5).ToArray();
        var inPlace = (double[])y.Clone();
        Quadrature.CumulativeTrapezoid(inPlace, 0.5, inPlace);
        Assert.Equal(expected, inPlace);

        var shifted = new double[] { 1, 2, 3, 4, 5, 0 };
        Quadrature.CumulativeTrapezoid(shifted.AsSpan(0, 5), 0.5, shifted.AsSpan(1, 5));
        Assert.Equal(expected, shifted[1..]);

        // 行列の行（間隔のあるビュー）
        var m = new Matrix<double>(2, 5);
        for (int j = 0; j < 5; j++)
        {
            m[0, j] = y[j];
        }

        Quadrature.CumulativeTrapezoid(m.Row(0), 0.5, m.Row(1));
        Assert.Equal(expected, m.Row(1).ToArray());
        Assert.Equal(expected[^1], Quadrature.Trapezoid(m.Row(0), 0.5));

        var x = new double[] { 0, 1, 2, 3, 4 };
        var withPoints = Quadrature.CumulativeTrapezoid(y, x).ToArray();
        Quadrature.CumulativeTrapezoid(y, x, x);
        Assert.Equal(withPoints, x);
    }

    [Fact]
    public void NonFiniteSamplesPropagate()
    {
        const double Inf = double.PositiveInfinity;
        Assert.Equal(Inf, Quadrature.Trapezoid(new double[] { Inf, Inf }));
        Assert.Equal(-Inf, Quadrature.Trapezoid(new double[] { 1.0, -Inf }));
        Assert.Equal(Inf, Quadrature.Trapezoid(new double[] { Inf, 1.0 }, new double[] { 0.0, 2.0 }));
        Assert.True(double.IsNaN(Quadrature.Trapezoid(new double[] { Inf, -Inf })));
        Assert.True(double.IsNaN(Quadrature.Trapezoid(new double[] { 1.0, double.NaN, 1.0 })));
        // 有限の標本でも和が溢れたら ∞
        Assert.Equal(Inf, Quadrature.Trapezoid(new double[] { 1e308, 1e308, 1e308 }));

        Assert.Equal(new double[] { 0, Inf, Inf }, Quadrature.CumulativeTrapezoid(new double[] { 1.0, Inf, 1.0 }).ToArray());
        Assert.Equal(new double[] { 0, 1, -Inf }, Quadrature.CumulativeTrapezoid(new double[] { 1.0, 1.0, -Inf }, new double[] { 0.0, 1.0, 2.0 }).ToArray());
    }

    [Fact]
    public void InvalidArgumentsThrow()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Quadrature.Trapezoid(new double[] { 1.0, 2.0 }, double.NaN));
        Assert.Throws<ArgumentOutOfRangeException>(() => Quadrature.CumulativeTrapezoid(new double[] { 1.0, 2.0 }, double.PositiveInfinity));
        Assert.Throws<ArgumentException>(() => Quadrature.CumulativeTrapezoid(new double[] { 1.0, 2.0 }, 1, new double[3]));
        Assert.Throws<ArgumentException>(() => Quadrature.CumulativeTrapezoid(new double[] { 1.0, 2.0 }, new double[] { 0.0, 1.0 }, new double[1]));
        Assert.Throws<ArgumentException>(() => Quadrature.CumulativeTrapezoid(new double[] { 1.0, 2.0 }, new double[] { 0.0 }, new double[2]));
    }
}
