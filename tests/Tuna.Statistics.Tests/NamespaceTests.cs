using Tuna.Statistics.Distributions;

namespace Tuna.Tests.Statistics;

// 拡張アセンブリの公開型がアセンブリ名から始まる名前空間に属することを検証する。
public class NamespaceTests
{
    [Fact]
    public void PublicTypesStartWithTheAssemblyName()
    {
        var assembly = typeof(NormalDistribution).Assembly;
        Assert.Equal("Tuna.Statistics", assembly.GetName().Name);
        Assert.All(assembly.GetExportedTypes(), t => Assert.StartsWith("Tuna.Statistics.", t.Namespace));
    }
}
