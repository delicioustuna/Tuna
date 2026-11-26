using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Tuna.Generators;

namespace Tuna.Differentiation.Tests;

public class DerivativeDiagnosticTests
{
    [Fact]
    public void InvalidDerivativeSpecificationsReportGeneratorErrors()
    {
        const string source = """
            using Tuna;
            public static partial class Functions
            {
                [ElementwiseOperation(Derivative = nameof(MissingResult))]
                public static double WrongSignature(double x) => x;
                private static double MissingResult(double x) => 1;

                [ElementwiseOperation(Over = ["x", "y"], Derivative = nameof(Partial))]
                public static double WrongProperty(double x, double y) => x + y;
                private static double Partial(double x, double y, double value) => 1;

                [ElementwiseOperation(Over = ["x", "y"], PartialDerivatives = [nameof(Partial)])]
                public static double TooFewPartials(double x, double y) => x + y;

                [ElementwiseOperation(Derivative = nameof(IntegerPartial))]
                public static int UnsupportedElementType(int x) => x;
                private static int IntegerPartial(int x, int value) => 1;

                [ElementwiseOperation(Derivative = "Unknown")]
                public static double MissingDerivative(double x) => x;
            }
            """;
        var references = AppDomain.CurrentDomain.GetAssemblies()
            .Where(a => !a.IsDynamic && a.Location.Length > 0)
            .Select(a => a.Location)
            .Append(typeof(ElementwiseOperationAttribute).Assembly.Location)
            .Append(typeof(ScalarNode).Assembly.Location)
            .Distinct()
            .Select(path => MetadataReference.CreateFromFile(path));
        var compilation = CSharpCompilation.Create("User", [CSharpSyntaxTree.ParseText(source)], references,
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        CSharpGeneratorDriver.Create(new OperationsGenerator())
            .RunGeneratorsAndUpdateCompilation(compilation, out _, out var diagnostics);
        var errors = diagnostics.Where(d => d.Severity == DiagnosticSeverity.Error).ToArray();
        Assert.Equal(5, errors.Length);
        Assert.All(errors, d => Assert.Equal("TNG001", d.Id));
        foreach (string name in new[] { "WrongSignature", "WrongProperty", "TooFewPartials", "UnsupportedElementType", "MissingDerivative" })
        {
            Assert.Single(errors, d => d.GetMessage().Contains(name, StringComparison.Ordinal));
        }
    }
}
