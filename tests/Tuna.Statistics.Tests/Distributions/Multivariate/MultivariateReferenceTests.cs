using System.Globalization;

namespace Tuna.Tests.Statistics;

// mpmath の参照値との比較。参照値と許容誤差は tools/DistributionReferences/generate_multivariate_references.py が
// MultivariateReferences.Generated.cs に書き出す。許容誤差は C# の出力を見る前に、評価点と各パラメータの成分についての
// 条件数 κ から 64ε(1 + κ) と決めた（対称行列の成分は √|aᵢᵢ aⱼⱼ| の幅で動かす。離散分布は実装の式の項の大きさの和も κ に足す）。
public class MultivariateReferenceTests
{
    public static TheoryData<int, string> Cases()
    {
        var data = new TheoryData<int, string>();
        for (int i = 0; i < MultivariateReferences.Cases.Length; i++)
        {
            var c = MultivariateReferences.Cases[i];
            string point = string.Join(", ", c.Point.Select(v => v.ToString("R", CultureInfo.InvariantCulture)));
            data.Add(i, $"{c.Distribution}.{c.Function}([{point}])");
        }

        return data;
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void MatchesReference(int index, string description)
    {
        var c = MultivariateReferences.Cases[index];
        object distribution = MultivariateCatalog.Create(c.Distribution, c.Parameters);
        double actual = c.Function switch
        {
            "LogPdf" => MultivariateCatalog.LogPdf(distribution, c.Point),
            "LogPmf" => MultivariateCatalog.LogPmf(distribution, c.Point),
            _ => throw new ArgumentOutOfRangeException(nameof(index), c.Function, null),
        };
        DistributionAssert.Close(c.Expected, actual, c.Tolerance, description);
    }
}
