using Tuna.Differentiation;

namespace Tuna.Tests.Differentiation;

/// <summary>拡張の公開する型が、アセンブリ名の名前空間にあり、コアだけを参照することを確かめる</summary>
public class NamespaceTests
{
    [Fact]
    public void PublicTypesStartWithTheAssemblyName()
    {
        var assembly = typeof(Tape).Assembly;
        Assert.Equal("Tuna.Differentiation", assembly.GetName().Name);
        Assert.All(assembly.GetExportedTypes(), t => Assert.Equal("Tuna.Differentiation", t.Namespace));
    }

    [Fact]
    public void ReferencesOnlyTheCore()
    {
        var references = typeof(Tape).Assembly.GetReferencedAssemblies().Select(a => a.Name!).Where(n => n.StartsWith("Tuna", StringComparison.Ordinal));
        Assert.Equal(["Tuna"], references);
    }
}
