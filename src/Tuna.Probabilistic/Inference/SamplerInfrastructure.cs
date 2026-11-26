using System.Runtime.ExceptionServices;
using Tuna.LinearAlgebra;
using Tuna.RandomSource;

namespace Tuna.Probabilistic.Inference;

internal readonly record struct SamplerSettings(int Warmup, int Draws, int Thin, ulong Seed, int MaxDegreeOfParallelism)
{
    public long SamplingIterations => (long)Draws * Thin;

    public static SamplerSettings Validate(SamplerOptions options)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(options.Warmup, nameof(options.Warmup));
        ArgumentOutOfRangeException.ThrowIfLessThan(options.Draws, 1, nameof(options.Draws));
        ArgumentOutOfRangeException.ThrowIfLessThan(options.Thin, 1, nameof(options.Thin));
        if (options.MaxDegreeOfParallelism is 0 or < -1)
        {
            throw new ArgumentOutOfRangeException(nameof(options.MaxDegreeOfParallelism), "The degree of parallelism must be positive or -1.");
        }

        ulong seed = options.Seed ?? new Philox4x32Random().GetState().Seed;
        return new SamplerSettings(options.Warmup, options.Draws, options.Thin, seed, options.MaxDegreeOfParallelism);
    }

    public void RequireStorage(int dimension, int chains)
    {
        if ((long)dimension * Draws * chains > Array.MaxLength)
        {
            throw new ArgumentOutOfRangeException(nameof(Draws), "The dimension times the draws times the chains must not exceed the maximum array length.");
        }
    }
}

// 乱数ストリーム ID の上位 8 ビットに割り当てる用途識別子
internal enum StreamPurpose : ulong
{
    Transition = 0,
    Initialization = 1,
    Exchange = 2,
}

// ストリーム ID のビット配置: [63..56] 用途、[55..32] レプリカ番号、[31..0] チェーン番号（再現性のため固定）
internal static class RandomStreams
{
    public static Philox4x32Random Create(ulong seed, StreamPurpose purpose, int chain, int replica = 0) =>
        new(seed, ((ulong)purpose << 56) | ((ulong)(uint)replica << 32) | (uint)chain);
}

internal static class ChainRunner
{
    // 並列実行時に発生した最初の例外を AggregateException でラップせずそのまま再スローする
    public static void Run(int count, int maxDegreeOfParallelism, CancellationToken cancellationToken, Action<int> runChain)
    {
        if (maxDegreeOfParallelism == 1 || count == 1)
        {
            for (int c = 0; c < count; c++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                runChain(c);
            }

            return;
        }

        var parallel = new ParallelOptions { MaxDegreeOfParallelism = maxDegreeOfParallelism, CancellationToken = cancellationToken };
        try
        {
            Parallel.For(0, count, parallel, runChain);
        }
        catch (AggregateException e) when (e.InnerExceptions.Count > 0)
        {
            ExceptionDispatchInfo.Capture(e.InnerExceptions[0]).Throw();
            throw;
        }
    }
}

internal static class InitialPoints
{
    public static double[] Broadcast(ReadOnlySpan<double> point, int chains)
    {
        if (point.IsEmpty)
        {
            throw new ArgumentException("The initial point must not be empty.", "initialPoint");
        }

        var result = new double[point.Length * chains];
        for (int c = 0; c < chains; c++)
        {
            point.CopyTo(result.AsSpan(c * point.Length));
        }

        return result;
    }

    public static double[] Columns(ReadOnlyMatrixSpan<double> points, int chains, string what = "chains")
    {
        if (points.Rows == 0)
        {
            throw new ArgumentException("The initial points must not be empty.", "initialPoints");
        }

        if (points.Columns != chains)
        {
            throw new ArgumentException($"The initial points must have one column per {what[..^1]} ({chains}).", "initialPoints");
        }

        var result = new double[points.Rows * chains];
        for (int c = 0; c < chains; c++)
        {
            for (int i = 0; i < points.Rows; i++)
            {
                result[c * points.Rows + i] = points[i, c];
            }
        }

        return result;
    }

    public static void RequireValid(ReadOnlySpan<double> point, double logDensity, int chain)
    {
        foreach (double v in point)
        {
            if (!double.IsFinite(v))
            {
                throw new ArgumentException($"The initial point of chain {chain} must be finite.", "initialPoints");
            }
        }

        if (!double.IsFinite(logDensity))
        {
            throw new ArgumentException($"The log density at the initial point of chain {chain} must be finite (got {logDensity}).", "initialPoints");
        }
    }
}

internal sealed class WarmupSchedule
{
    private readonly int[] starts;
    private readonly int[] ends;

    public WarmupSchedule(int warmup, WarmupWindowOptions options)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(options.InitialBuffer, nameof(options.InitialBuffer));
        ArgumentOutOfRangeException.ThrowIfLessThan(options.BaseWindow, 1, nameof(options.BaseWindow));
        ArgumentOutOfRangeException.ThrowIfNegative(options.TerminalBuffer, nameof(options.TerminalBuffer));
        var startList = new List<int>();
        var endList = new List<int>();
        if (warmup >= 20)
        {
            long initial = options.InitialBuffer, window = options.BaseWindow, terminal = options.TerminalBuffer;
            if (initial + window + terminal > warmup)
            {
                initial = (int)(0.15 * warmup);
                terminal = (int)(0.1 * warmup);
                window = warmup - initial - terminal;
            }

            long start = initial, last = warmup - terminal;
            while (start < last)
            {
                long end = start + window;
                if (end + 2 * window > last)
                {
                    end = last;
                }

                startList.Add((int)start);
                endList.Add((int)end);
                start = end;
                window *= 2;
            }
        }

        starts = [.. startList];
        ends = [.. endList];
    }

    public bool AdaptsMetric => starts.Length > 0;

    public bool InWindow(int iteration) => starts.Length > 0 && iteration >= starts[0] && iteration < ends[^1];

    public bool EndsWindow(int iteration) => Array.IndexOf(ends, iteration + 1) >= 0;
}

// 双対平均法によるステップ幅適応（Nesterov 2009、Hoffman & Gelman 2014 §3.2）
internal sealed class DualAveraging(double targetAcceptance, double centerFactor = 10, double gamma = 0.05)
{
    private const double T0 = 10;
    private const double Kappa = 0.75;

    private double mu;
    private double averageError;
    private double logAveraged;
    private int count;

    public void Restart(double stepSize)
    {
        mu = Math.Log(centerFactor * stepSize);
        averageError = 0;
        logAveraged = 0;
        count = 0;
    }

    public double Update(double acceptance)
    {
        acceptance = double.IsNaN(acceptance) ? 0 : Math.Clamp(acceptance, 0, 1);
        count++;
        double eta = 1 / (count + T0);
        averageError = (1 - eta) * averageError + eta * (targetAcceptance - acceptance);
        double logStep = mu - Math.Sqrt(count) / gamma * averageError;
        double weight = Math.Pow(count, -Kappa);
        logAveraged = weight * logStep + (1 - weight) * logAveraged;
        return Math.Exp(logStep);
    }

    public double Averaged => count > 0 ? Math.Exp(logAveraged) : double.NaN;
}

// Welford 法による標本共分散のオンライン推定と単位行列方向への縮小
internal sealed class CovarianceEstimator(int dimension, bool dense)
{
    private readonly double[] mean = new double[dimension];
    private readonly double[] delta = new double[dimension];
    private readonly double[] sums = new double[dense ? dimension * dimension : dimension];
    private long count;

    public long Count => count;

    public void Reset()
    {
        Array.Clear(mean);
        Array.Clear(sums);
        count = 0;
    }

    public void Add(ReadOnlySpan<double> x)
    {
        count++;
        for (int i = 0; i < dimension; i++)
        {
            delta[i] = x[i] - mean[i];
            mean[i] += delta[i] / count;
        }

        if (dense)
        {
            for (int j = 0; j < dimension; j++)
            {
                double after = x[j] - mean[j];
                for (int i = 0; i < dimension; i++)
                {
                    sums[j * dimension + i] += delta[i] * after;
                }
            }
        }
        else
        {
            for (int i = 0; i < dimension; i++)
            {
                sums[i] += delta[i] * (x[i] - mean[i]);
            }
        }
    }

    // Stan と同じ縮小推定: n/(n + 5) * Σ̂ + 1e-3 * 5/(n + 5) * I
    public void WriteRegularized(Span<double> destination)
    {
        double n = count;
        double scale = n / (n + 5) / Math.Max(n - 1, 1);
        double shrink = 1e-3 * 5 / (n + 5);
        if (dense)
        {
            for (int j = 0; j < dimension; j++)
            {
                for (int i = 0; i < dimension; i++)
                {
                    double value = 0.5 * (sums[j * dimension + i] + sums[i * dimension + j]) * scale;
                    destination[j * dimension + i] = i == j ? value + shrink : value;
                }
            }
        }
        else
        {
            for (int i = 0; i < dimension; i++)
            {
                destination[i] = sums[i] * scale + shrink;
            }
        }
    }
}

internal static class Gaussian
{
    public static void Fill(IRandomSource random, Span<double> destination)
    {
        Statistics.Distributions.NormalDistribution.Standard.Sample(random, destination);
    }
}
