using System.Globalization;
using Tuna.Statistics.Distributions;

namespace Tuna.Tests.Statistics;

// mpmath の参照値との比較。参照値と許容誤差は tools/DistributionReferences/generate_distribution_references.py が
// DiscreteReferences.Generated.cs に書き出す。確率と性質の許容誤差は条件数 κ から 64ε(1 + κ)、
// 逆関数と中央値は整数なので完全一致（確率は F(k-1) と F(k) の中点で、どちらにも相対 1e-9 以上離れている）。
public class DiscreteReferenceTests
{
    public static TheoryData<int, string> Cases()
    {
        var data = new TheoryData<int, string>();
        for (int i = 0; i < DiscreteReferences.Cases.Length; i++)
        {
            data.Add(i, Describe(i));
        }

        return data;
    }

    private static string Describe(int index)
    {
        var c = DiscreteReferences.Cases[index];
        string argument = double.IsNaN(c.Argument) ? "" : $"({c.Argument.ToString("R", CultureInfo.InvariantCulture)})";
        return $"{DistributionCatalog.Describe(c.Distribution, c.Parameters)}.{c.Function}{argument}";
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void MatchesReference(int index, string description)
    {
        var c = DiscreteReferences.Cases[index];
        IDiscreteDistribution d = DistributionCatalog.Discrete(c.Distribution, c.Parameters);
        long k = double.IsNaN(c.Argument) || c.Function.StartsWith("Inverse", StringComparison.Ordinal) ? 0 : checked((long)c.Argument);
        double p = c.Argument;
        double actual = c.Function switch
        {
            "Pmf" => d.Pmf(k),
            "LogPmf" => d.LogPmf(k),
            "Cdf" => d.Cdf(k),
            "Sf" => d.Sf(k),
            "LogCdf" => d.LogCdf(k),
            "LogSf" => d.LogSf(k),
            "InverseCdf" => d.InverseCdf(p),
            "InverseSf" => d.InverseSf(p),
            "InverseLogCdf" => d.InverseLogCdf(p),
            "InverseLogSf" => d.InverseLogSf(p),
            "Mean" => d.Mean,
            "Variance" => d.Variance,
            "StandardDeviation" => d.StandardDeviation,
            "Skewness" => d.Skewness,
            "ExcessKurtosis" => d.ExcessKurtosis,
            "Entropy" => d.Entropy,
            "Median" => d.Median,
            _ => throw new ArgumentOutOfRangeException(nameof(index), c.Function, null),
        };
        DistributionAssert.Close(c.Expected, actual, c.Tolerance, description);
    }
}
