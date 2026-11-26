using System.Globalization;
using Tuna.Statistics.Distributions;

namespace Tuna.Tests.Statistics;

// mpmath の参照値との比較。参照値と許容誤差は tools/DistributionReferences/generate_distribution_references.py が
// ContinuousReferences.Generated.cs に書き出す。許容誤差は C# の出力を見る前に、問題の条件数 κ から 64ε(1 + κ) と決めた
// （実装の近似の限界で広げた箇所は生成スクリプトの known_limit に理由がある）。
public class ContinuousReferenceTests
{
    public static TheoryData<int, string> Cases()
    {
        var data = new TheoryData<int, string>();
        for (int i = 0; i < ContinuousReferences.Cases.Length; i++)
        {
            data.Add(i, Describe(i));
        }

        return data;
    }

    private static string Describe(int index)
    {
        var c = ContinuousReferences.Cases[index];
        string argument = double.IsNaN(c.Argument) ? "" : $"({c.Argument.ToString("R", CultureInfo.InvariantCulture)})";
        return $"{DistributionCatalog.Describe(c.Distribution, c.Parameters)}.{c.Function}{argument}";
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void MatchesReference(int index, string description)
    {
        var c = ContinuousReferences.Cases[index];
        IContinuousDistribution d = DistributionCatalog.Continuous(c.Distribution, c.Parameters);
        double x = c.Argument;
        double actual = c.Function switch
        {
            "Pdf" => d.Pdf(x),
            "LogPdf" => d.LogPdf(x),
            "Cdf" => d.Cdf(x),
            "Sf" => d.Sf(x),
            "LogCdf" => d.LogCdf(x),
            "LogSf" => d.LogSf(x),
            "InverseCdf" => d.InverseCdf(x),
            "InverseSf" => d.InverseSf(x),
            "InverseLogCdf" => d.InverseLogCdf(x),
            "InverseLogSf" => d.InverseLogSf(x),
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
