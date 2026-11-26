using System.Collections.Concurrent;
using System.Numerics;
using System.Runtime.InteropServices;

namespace Tuna.SignalProcessing;

internal enum TrigonometricKind
{
    Cosine1,
    Cosine2,
    Cosine3,
    Cosine4,
    Sine1,
}

internal sealed class TrigonometricPlan
{
    private const int Capacity = 64;
    private static readonly ConcurrentDictionary<(TrigonometricKind Kind, int Length), TrigonometricPlan> Cache = new();

    private readonly RealFourierPlan? real;
    private readonly FourierPlan? complex;
    private readonly Complex[] pre = [];
    private readonly Complex[] post = [];

    private TrigonometricPlan(TrigonometricKind kind, int n)
    {
        Kind = kind;
        Length = n;
        switch (kind)
        {
            case TrigonometricKind.Cosine1:
                real = RealFourierPlan.Create(2 * (n - 1));
                break;
            case TrigonometricKind.Sine1:
                real = RealFourierPlan.Create(checked(2 * (n + 1)));
                break;
            case TrigonometricKind.Cosine2 or TrigonometricKind.Cosine3:
                real = RealFourierPlan.Create(n);
                post = new Complex[n / 2 + 1];
                for (int k = 0; k < post.Length; k++)
                {
                    post[k] = UnitRoots.Root(k, 4L * n);
                }

                break;
            default:
                if (n % 2 == 0)
                {
                    int h = n / 2;
                    complex = FourierPlan.Create(h);
                    pre = new Complex[h];
                    post = new Complex[h];
                    for (int m = 0; m < h; m++)
                    {
                        pre[m] = UnitRoots.Root(m, 2L * n);
                        post[m] = UnitRoots.Root(4L * m + 1, 8L * n);
                    }
                }
                else
                {
                    complex = FourierPlan.Create(checked(2 * n));
                    pre = new Complex[n];
                    post = new Complex[n];
                    for (int j = 0; j < n; j++)
                    {
                        pre[j] = UnitRoots.Root(j, 4L * n);
                        post[j] = UnitRoots.Root(2L * j + 1, 8L * n);
                    }
                }

                break;
        }
    }

    internal TrigonometricKind Kind { get; }

    internal int Length { get; }

    internal static TrigonometricPlan Create(TrigonometricKind kind, int length)
    {
        var key = (kind, length);
        if (Cache.TryGetValue(key, out var plan))
        {
            return plan;
        }

        if (Cache.Count >= Capacity)
        {
            Cache.Clear();
        }

        return Cache.GetOrAdd(key, static k => new TrigonometricPlan(k.Kind, k.Length));
    }

    internal int WorkLength
    {
        get
        {
            int n = Length;
            return Kind switch
            {
                TrigonometricKind.Cosine1 => (n - 1) + n + real!.ForwardWorkLength(1),
                TrigonometricKind.Sine1 => (n + 1) + (n + 2) + real!.ForwardWorkLength(1),
                TrigonometricKind.Cosine2 => (n + 1) / 2 + (n / 2 + 1) + real!.ForwardWorkLength(1),
                TrigonometricKind.Cosine3 => (n / 2 + 1) + (n + 1) / 2 + real!.InverseWorkLength(1),
                _ => (n % 2 == 0 ? n / 2 : 2 * n) + complex!.WorkLength(1),
            };
        }
    }

    internal void Execute(ReadOnlySpan<double> x, Span<double> y, Span<Complex> work, bool sine)
    {
        switch (Kind)
        {
            case TrigonometricKind.Cosine1:
                Cosine1(x, y, work);
                break;
            case TrigonometricKind.Sine1:
                Sine1(x, y, work);
                break;
            case TrigonometricKind.Cosine2:
                Cosine2(x, y, work, sine);
                break;
            case TrigonometricKind.Cosine3:
                Cosine3(x, y, work, sine);
                break;
            default:
                if (Length % 2 == 0)
                {
                    Cosine4Even(x, y, work, sine);
                }
                else
                {
                    Cosine4Odd(x, y, work, sine);
                }

                break;
        }
    }

    private void Cosine1(ReadOnlySpan<double> x, Span<double> y, Span<Complex> work)
    {
        int n = Length;
        int m = 2 * (n - 1);
        var extended = MemoryMarshal.Cast<Complex, double>(work[..(n - 1)]);
        var spectrum = work.Slice(n - 1, n);
        x.CopyTo(extended);
        for (int j = 1; j < n - 1; j++)
        {
            extended[m - j] = x[j];
        }

        real!.ForwardCore(ref extended[0], ref spectrum[0], work[(2 * n - 1)..], 1);
        for (int k = 0; k < n; k++)
        {
            y[k] = spectrum[k].Real;
        }
    }

    private void Sine1(ReadOnlySpan<double> x, Span<double> y, Span<Complex> work)
    {
        int n = Length;
        int m = 2 * (n + 1);
        var extended = MemoryMarshal.Cast<Complex, double>(work[..(n + 1)]);
        var spectrum = work.Slice(n + 1, n + 2);
        extended[0] = 0;
        extended[n + 1] = 0;
        for (int j = 0; j < n; j++)
        {
            extended[j + 1] = x[j];
            extended[m - 1 - j] = -x[j];
        }

        real!.ForwardCore(ref extended[0], ref spectrum[0], work[(2 * n + 3)..], 1);
        for (int k = 0; k < n; k++)
        {
            y[k] = -spectrum[k + 1].Imaginary;
        }
    }

    // Makhoul (1980) による長さ n の実数 FFT への帰着。DST-II は奇数番の符号反転と出力逆順を融合する。
    private void Cosine2(ReadOnlySpan<double> x, Span<double> y, Span<Complex> work, bool sine)
    {
        int n = Length;
        int half = (n + 1) / 2;
        var v = MemoryMarshal.Cast<Complex, double>(work[..half])[..n];
        var spectrum = work.Slice(half, n / 2 + 1);
        double oddSign = sine ? -1 : 1;
        for (int m = 0; m < n / 2; m++)
        {
            v[m] = x[2 * m];
            v[n - 1 - m] = oddSign * x[2 * m + 1];
        }

        if (n % 2 == 1)
        {
            v[n / 2] = x[n - 1];
        }

        real!.ForwardCore(ref v[0], ref spectrum[0], work[(half + n / 2 + 1)..], 1);
        int last = sine ? n - 1 : 0;
        y[last] = 2 * spectrum[0].Real;
        for (int k = 1; k <= n / 2; k++)
        {
            var w = post[k];
            var s = spectrum[k];
            double re = w.Real * s.Real - w.Imaginary * s.Imaginary;
            double im = w.Real * s.Imaginary + w.Imaginary * s.Real;
            y[Math.Abs(last - k)] = 2 * re;
            if (2 * k != n)
            {
                y[Math.Abs(last - (n - k))] = -2 * im;
            }
        }
    }

    // DCT-II の逆手順。DST-III は入力逆順と出力の奇数番符号反転を融合する。
    private void Cosine3(ReadOnlySpan<double> x, Span<double> y, Span<Complex> work, bool sine)
    {
        int n = Length;
        int count = n / 2 + 1;
        int half = (n + 1) / 2;
        var spectrum = work[..count];
        var v = MemoryMarshal.Cast<Complex, double>(work.Slice(count, half))[..n];
        int first = sine ? n - 1 : 0;
        spectrum[0] = new Complex(x[first], 0);
        for (int k = 1; k < count; k++)
        {
            var w = post[k];
            double a = x[Math.Abs(first - k)], b = x[Math.Abs(first - (n - k))];
            spectrum[k] = new Complex(a * w.Real - b * w.Imaginary, -(a * w.Imaginary + b * w.Real));
        }

        real!.InverseCore(ref spectrum[0], ref v[0], work[(count + half)..], 1);
        double oddSign = sine ? -1 : 1;
        for (int m = 0; m < n / 2; m++)
        {
            y[2 * m] = v[m];
            y[2 * m + 1] = oddSign * v[n - 1 - m];
        }

        if (n % 2 == 1)
        {
            y[n - 1] = v[n / 2];
        }
    }

    private void Cosine4Even(ReadOnlySpan<double> x, Span<double> y, Span<Complex> work, bool sine)
    {
        int n = Length;
        int h = n / 2;
        var z = work[..h];
        double oddSign = sine ? -1 : 1;
        for (int m = 0; m < h; m++)
        {
            var w = pre[m];
            double a = x[2 * m], b = oddSign * x[n - 1 - 2 * m];
            z[m] = new Complex(a * w.Real - b * w.Imaginary, a * w.Imaginary + b * w.Real);
        }

        complex!.Execute<ForwardDirection>(ref z[0], ref z[0], work[h..], 1);
        for (int p = 0; p < h; p++)
        {
            var w = post[p];
            var s = z[p];
            double even = 2 * (w.Real * s.Real - w.Imaginary * s.Imaginary);
            double odd = -2 * (w.Real * s.Imaginary + w.Imaginary * s.Real);
            y[sine ? n - 1 - 2 * p : 2 * p] = even;
            y[sine ? 2 * p : n - 1 - 2 * p] = odd;
        }
    }

    private void Cosine4Odd(ReadOnlySpan<double> x, Span<double> y, Span<Complex> work, bool sine)
    {
        int n = Length;
        var u = work[..(2 * n)];
        for (int j = 0; j < n; j++)
        {
            var w = pre[j];
            double value = sine && (j & 1) == 1 ? -x[j] : x[j];
            u[j] = new Complex(value * w.Real, value * w.Imaginary);
        }

        u[n..].Clear();
        complex!.Execute<ForwardDirection>(ref u[0], ref u[0], work[(2 * n)..], 1);
        for (int k = 0; k < n; k++)
        {
            var w = post[k];
            var s = u[k];
            y[sine ? n - 1 - k : k] = 2 * (w.Real * s.Real - w.Imaginary * s.Imaginary);
        }
    }
}
