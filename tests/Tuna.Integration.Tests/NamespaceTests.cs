using Tuna.Integration;

namespace Tuna.Tests.Integration;

/// <summary>拡張の公開する型が、アセンブリ名から始まる名前空間にあり、コアだけを参照することを確かめる</summary>
public class NamespaceTests
{
    [Fact]
    public void PublicTypesStartWithTheAssemblyName()
    {
        var assembly = typeof(Quadrature).Assembly;
        Assert.Equal("Tuna.Integration", assembly.GetName().Name);
        Assert.All(assembly.GetExportedTypes(), t => Assert.Equal("Tuna.Integration", t.Namespace));
    }

    [Fact]
    public void ReferencesOnlyTheCore()
    {
        var references = typeof(Quadrature).Assembly.GetReferencedAssemblies().Select(a => a.Name!).Where(n => n.StartsWith("Tuna", StringComparison.Ordinal));
        Assert.Equal(["Tuna"], references);
    }
}
