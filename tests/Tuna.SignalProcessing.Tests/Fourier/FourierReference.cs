using Complex = System.Numerics.Complex;

namespace Tuna.Tests.SignalProcessing;

/// <summary>二重倍精度（double-double、有効桁約 32 桁）の実数。FFT の基準値の計算にだけ使う</summary>
internal readonly struct DoubleDouble(double hi, double lo = 0)
{
    public double Hi { get; } = hi;

    public double Lo { get; } = lo;

    public double ToDouble() => Hi + Lo;

    public static DoubleDouble Pi { get; } = new(3.141592653589793, 1.2246467991473532e-16);

    private static DoubleDouble QuickTwoSum(double a, double b)
    {
        double s = a + b;
        return new DoubleDouble(s, b - (s - a));
    }

    private static (double Sum, double Error) TwoSum(double a, double b)
    {
        double s = a + b;
        double bb = s - a;
        return (s, (a - (s - bb)) + (b - bb));
    }

    public static DoubleDouble operator +(DoubleDouble a, DoubleDouble b)
    {
        var (s, e) = TwoSum(a.Hi, b.Hi);
        var (t, f) = TwoSum(a.Lo, b.Lo);
        e += t;
        var r = QuickTwoSum(s, e);
        return QuickTwoSum(r.Hi, r.Lo + f);
    }

    public static DoubleDouble operator -(DoubleDouble a) => new(-a.Hi, -a.Lo);

    public static DoubleDouble operator -(DoubleDouble a, DoubleDouble b) => a + -b;

    public static DoubleDouble operator *(DoubleDouble a, DoubleDouble b)
    {
        double p = a.Hi * b.Hi;
        double e = Math.FusedMultiplyAdd(a.Hi, b.Hi, -p);
        e += a.Hi * b.Lo + a.Lo * b.Hi;
        return QuickTwoSum(p, e);
    }

    public static DoubleDouble operator /(DoubleDouble a, double b)
    {
        double q1 = a.Hi / b;
        var r = a - new DoubleDouble(q1) * new DoubleDouble(b);
        double q2 = r.Hi / b;
        r -= new DoubleDouble(q2) * new DoubleDouble(b);
        double q3 = r.Hi / b;
        return QuickTwoSum(q1, q2) + new DoubleDouble(q3);
    }

    public static implicit operator DoubleDouble(double value) => new(value);
}

/// <summary>二重倍精度の直接 DFT（O(n²)）と誤差の測り方</summary>
internal static class FourierReference
{
    public const double Epsilon = 1.1102230246251565e-16; // 2^−53

    /// <summary>exp(−2πi·k/n) の実部と虚部（二重倍精度。k を整数のまま八分円に還元してからテイラー級数で計算する）</summary>
    public static (DoubleDouble Re, DoubleDouble Im) Root(long k, long n)
    {
        k %= n;
        if (k < 0)
        {
            k += n;
        }

        long eightK = 8 * k;
        long o = eightK / n;
        long r = eightK - o * n;
        DoubleDouble c, s;
        if ((o & 1) == 0)
        {
            (s, c) = SinCosQuarterPi(r, n);
        }
        else
        {
            (c, s) = SinCosQuarterPi(n - r, n);
        }

        (c, s) = (o >> 1) switch
        {
            0 => (c, s),
            1 => (-s, c),
            2 => (-c, -s),
            _ => (s, -c),
        };
        return (c, -s);
    }

    /// <summary>φ = (π/4)·(r/n)（0 ≤ r ≤ n）の sin と cos</summary>
    private static (DoubleDouble Sin, DoubleDouble Cos) SinCosQuarterPi(long r, long n)
    {
        var phi = DoubleDouble.Pi * (new DoubleDouble(r) / 4.0 / n);
        var x2 = phi * phi;
        DoubleDouble sin = phi, cos = 1.0;
        DoubleDouble sinTerm = phi, cosTerm = 1.0;
        for (int i = 1; i < 40; i++)
        {
            sinTerm = -(sinTerm * x2 / (2.0 * i) / (2.0 * i + 1));
            cosTerm = -(cosTerm * x2 / (2.0 * i - 1) / (2.0 * i));
            sin += sinTerm;
            cos += cosTerm;
            if (Math.Abs(sinTerm.Hi) < 1e-40 && Math.Abs(cosTerm.Hi) < 1e-40)
            {
                break;
            }
        }

        return (sin, cos);
    }

    /// <summary>倍率なしの直接 DFT（<paramref name="inverse"/> なら exp(+2πi·jk/n)）を二重倍精度で計算し、double に丸めずに返す</summary>
    public static (DoubleDouble Re, DoubleDouble Im)[] Dft(ReadOnlySpan<Complex> x, bool inverse)
    {
        int n = x.Length;
        var roots = new (DoubleDouble Re, DoubleDouble Im)[n];
        for (int r = 0; r < n; r++)
        {
            roots[r] = Root(r, n);
            if (inverse)
            {
                roots[r] = (roots[r].Re, -roots[r].Im);
            }
        }

        var result = new (DoubleDouble Re, DoubleDouble Im)[n];
        for (int k = 0; k < n; k++)
        {
            DoubleDouble re = 0, im = 0;
            long index = 0;
            for (int j = 0; j < n; j++)
            {
                var (c, s) = roots[index];
                DoubleDouble xr = x[j].Real, xi = x[j].Imaginary;
                re += xr * c - xi * s;
                im += xr * s + xi * c;
                index += k;
                if (index >= n)
                {
                    index -= n;
                }
            }

            result[k] = (re, im);
        }

        return result;
    }

    /// <summary>相対 RMS 誤差 ‖y − y_ref‖₂ / ‖y_ref‖₂ を ε = 2^−53 の単位で返す</summary>
    public static double RelativeRmsError(ReadOnlySpan<Complex> actual, ReadOnlySpan<(DoubleDouble Re, DoubleDouble Im)> expected, double scale = 1)
    {
        Assert.Equal(expected.Length, actual.Length);
        double error = 0, norm = 0;
        for (int k = 0; k < actual.Length; k++)
        {
            double er = (new DoubleDouble(actual[k].Real) - expected[k].Re * scale).ToDouble();
            double ei = (new DoubleDouble(actual[k].Imaginary) - expected[k].Im * scale).ToDouble();
            double rr = (expected[k].Re * scale).ToDouble();
            double ri = (expected[k].Im * scale).ToDouble();
            error += er * er + ei * ei;
            norm += rr * rr + ri * ri;
        }

        return norm == 0 ? Math.Sqrt(error) / Epsilon : Math.Sqrt(error / norm) / Epsilon;
    }

    /// <summary>実部・虚部が独立な標準正規乱数の複素数列（種を固定）</summary>
    public static Complex[] RandomComplex(int length, int seed)
    {
        var random = new Random(seed);
        var result = new Complex[length];
        for (int i = 0; i < length; i++)
        {
            result[i] = new Complex(Normal(random), Normal(random));
        }

        return result;
    }

    /// <summary>標準正規乱数の実数列（種を固定）</summary>
    public static double[] RandomReal(int length, int seed)
    {
        var random = new Random(seed);
        var result = new double[length];
        for (int i = 0; i < length; i++)
        {
            result[i] = Normal(random);
        }

        return result;
    }

    private static double Normal(Random random)
    {
        double u = 1 - random.NextDouble();
        double v = random.NextDouble();
        return Math.Sqrt(-2 * Math.Log(u)) * Math.Cos(2 * Math.PI * v);
    }
}
