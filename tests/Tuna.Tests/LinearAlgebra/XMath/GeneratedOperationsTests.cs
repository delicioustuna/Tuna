using System.Collections.Immutable;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Tuna.Generators;
using Tuna.LinearAlgebra;
using DoubleVector = Tuna.LinearAlgebra.Vector<double>;

namespace Tuna.Tests.LinearAlgebra;

/// <summary>利用者が書くスカラー関数。属性を付けるだけで、ベクトル・行列版と演算子が生成される。</summary>
public static partial class Activations
{
    [ElementwiseOperation(Derivative = nameof(SoftplusDerivative))]
    public static double Softplus(double x) => Math.Log(1 + Math.Exp(x));

    private static double SoftplusDerivative(double x, double result) => 1 / (1 + Math.Exp(-x));

    [ElementwiseOperation]
    public static T Square<T>(T x) where T : unmanaged, System.Numerics.INumberBase<T> => x * x;

    /// <summary>追加の引数（offset）は、生成する演算子のフィールドになる</summary>
    [ElementwiseOperation]
    public static T Shifted<T>(T x, T offset) where T : unmanaged, System.Numerics.INumberBase<T> => x + offset;

    /// <summary>要素ごとに回す引数を Over で選ぶ。それ以外の引数（exponent）は演算子のフィールドになる</summary>
    [ElementwiseOperation(Over = [nameof(x)])]
    public static double Power(double exponent, double x) => Math.Pow(x, exponent);

    /// <summary>2入力（状態なし）。演算子の名前と置き場所を指定する</summary>
    [ElementwiseOperation(Over = [nameof(a), nameof(b)], OperatorName = "Hypotenuse", OperatorContainer = "Operators")]
    public static double Hypot(double a, double b) => Math.Sqrt(a * a + b * b);

    /// <summary>2入力で、追加の引数が入力の間にある（生成する版も引数の並びは元のまま）</summary>
    [ElementwiseOperation(Over = [nameof(x), nameof(y)])]
    public static double Affine(double x, double slope, double y) => slope * x + y;

    /// <summary>3入力・型引数・追加の引数</summary>
    [ElementwiseOperation(Over = [nameof(x), nameof(y), nameof(weight)])]
    public static T Blend<T>(T x, T y, T weight, T scale) where T : unmanaged, System.Numerics.INumberBase<T> => scale * (x + (y - x) * weight);

    /// <summary>非公開のメソッドの演算子は、入れ子のクラスに置くと internal になる</summary>
    [ElementwiseOperation(OperatorContainer = "Operators")]
    private static double Twice(double x) => 2 * x;

    [PairwiseOperation]
    public static double SquaredDistance(ReadOnlyVectorSpan<double> x, ReadOnlyVectorSpan<double> y)
    {
        double sum = 0;
        for (int i = 0; i < x.Length; i++)
        {
            sum += (x[i] - y[i]) * (x[i] - y[i]);
        }

        return sum;
    }

    [PairwiseOperation]
    public static double Gaussian(ReadOnlyVectorSpan<double> x, ReadOnlyVectorSpan<double> y, double lengthScale) =>
        Math.Exp(-SquaredDistance(x, y) / (2 * lengthScale * lengthScale));
}

/// <summary>利用者向けの属性（[ElementwiseOperation]・[PairwiseOperation]）で生成されるコード</summary>
[Trait("LinearAlgebra", "Generator")]
public class GeneratedOperationsTests
{
    private static double SoftplusOf(double v) => Math.Log(1 + Math.Exp(v));

    [Fact]
    public void ElementwiseOperationLiftsAScalarFunctionToVectorsAndMatrices()
    {
        var x = new DoubleVector([-2, 0, 3]);
        var expected = x.ToArray().Select(SoftplusOf).ToArray();
        Assert.Equal(SoftplusOf(1.5), Activations.Softplus(1.5));            // 元のスカラー関数はそのまま
        Assert.Equal(expected, Activations.Softplus(x).ToArray());           // ベクトル向け（新規割り当て）
        var z = new DoubleVector(3);
        Activations.Softplus(x, z);                                          // destination オーバーロード
        Assert.Equal(expected, z.ToArray());

        var a = Matrix<double>.FromColumnMajor(2, 2, [-1, 0, 1, 2]);
        Assert.Equal(a.AsSpan().ToArray().Select(SoftplusOf).ToArray(), Activations.Softplus(a).AsSpan().ToArray());
        Assert.Equal(new[] { SoftplusOf(-1), SoftplusOf(2) }, Activations.Softplus(a.Diagonal()).ToArray());

        Activations.InPlace.Softplus(x);                                     // その場更新
        Assert.Equal(expected, x.ToArray());

        // 生成した演算子は XMath.Map にも渡せる
        Assert.Equal(expected.Select(SoftplusOf).ToArray(), XMath.Map<Activations.SoftplusOperator>(x).ToArray());
    }

    [Fact]
    public void DerivativeMetadataDoesNotRequireTheDifferentiationAssembly()
    {
        // スカラーメソッドに加え、ベクトルと行列それぞれに新規割り当てと destination を受け取るオーバーロードがある。
        Assert.Equal(5, typeof(Activations).GetMethods().Count(m => m.Name == nameof(Activations.Softplus)));
    }

    [Fact]
    public void GenericAndStatefulOperationsInferTheElementType()
    {
        var x = new DoubleVector([1, 2, 3]);
        Assert.Equal(9.0, Activations.Square(3.0));
        Assert.Equal(new double[] { 1, 4, 9 }, Activations.Square(x).ToArray());                       // コンテナー
        var a = Matrix<double>.FromColumnMajor(2, 2, [1, 2, 3, 4]);
        Assert.Equal(new double[] { 1, 16 }, Activations.Square(a.Diagonal()).ToArray());             // 書き込み可能なビュー
        Assert.Equal(new float[] { 4, 9 }, Activations.Square(new float[] { 2, 3 }.ToVector()).ToArray());

        Assert.Equal(new double[] { 1.5, 2.5, 3.5 }, Activations.Shifted(x, 0.5).ToArray());
        Assert.Equal(new double[] { 1.5, 2.5, 3.5 }, XMath.Map(x, new Activations.ShiftedOperator<double>(0.5)).ToArray());
        Activations.InPlace.Shifted(x, -1.0);
        Assert.Equal(new double[] { 0, 1, 2 }, x.ToArray());
    }

    [Fact]
    public void PairwiseOperationBuildsDistanceAndKernelMatrices()
    {
        // 各列が1点
        var x = Matrix<double>.FromColumnMajor(2, 3, [0, 0, 1, 0, 0, 2]);
        var y = Matrix<double>.FromColumnMajor(2, 2, [1, 1, 0, 0]);
        var distances = Activations.SquaredDistance(x, y);
        Assert.Equal(new double[,] { { 2, 0 }, { 1, 1 }, { 2, 4 } }, distances.ToArray());
        Assert.Equal(distances.ToArray(), XMath.Pairwise<Activations.SquaredDistanceOperator>(x, y).ToArray());
        Assert.Equal(2.0, Activations.SquaredDistance(x.Column(0), y.Column(0)));   // 元の関数は2点のまま使える

        var kernel = Activations.Gaussian(x, x, 1.0);
        Assert.Equal(1.0, kernel[2, 2]);
        Assert.Equal(Math.Exp(-0.5), kernel[0, 1]);
        var destination = new Matrix<double>(3, 3);
        Assert.Same(destination, XMath.Pairwise(x, x, new Activations.GaussianOperator(1.0), destination));
        Assert.Equal(kernel.ToArray(), destination.ToArray());
    }

    [Fact]
    public void OverSelectsTheElementwiseArgumentsAndKeepsTheParameterOrder()
    {
        var x = new DoubleVector([1, 2, 3]);
        Assert.Equal(new double[] { 1, 4, 9 }, Activations.Power(2.0, x).ToArray());
        Assert.Equal(new double[] { 1, 8, 27 }, XMath.Map(x, new Activations.PowerOperator(3.0)).ToArray());
        Activations.InPlace.Power(2.0, x);
        Assert.Equal(new double[] { 1, 4, 9 }, x.ToArray());

        var a = new DoubleVector([3, 5, 8]);
        var b = new DoubleVector([4, 12, 15]);
        Assert.Equal(new double[] { 5, 13, 17 }, Activations.Hypot(a, b).ToArray());
        Assert.Equal(new double[] { 5, 13, 17 }, XMath.Zip<Activations.Operators.Hypotenuse>(a, b).ToArray());
        var destination = new DoubleVector(3);
        Activations.Hypot(a, b, destination);
        Assert.Equal(new double[] { 5, 13, 17 }, destination.ToArray());
        var m = Matrix<double>.FromColumnMajor(1, 2, [3, 5]);
        Assert.Equal(new double[,] { { 5, 13 } }, Activations.Hypot(m, Matrix<double>.FromColumnMajor(1, 2, [4, 12])).ToArray());

        Assert.Equal(new double[] { 10, 22, 31 }, Activations.Affine(a, 2.0, b).ToArray());
        Assert.Equal(new double[] { 10, 22, 31 }, XMath.Zip(a, b, new Activations.AffineOperator(2.0)).ToArray());

        var w = new DoubleVector([0, 0.5, 1]);
        Assert.Equal(new double[] { 6, 17, 30 }, Activations.Blend(a, b, w, 2.0).ToArray());                        // コンテナー
        Assert.Equal(new double[] { 6, 17, 30 }, Activations.Blend(a.AsVectorSpan(), b, (ReadOnlyVectorSpan<double>)w, 2.0).ToArray());
        Assert.Equal(new double[] { 6, 17, 30 }, XMath.Zip(a, b, w, new Activations.BlendOperator<double>(2.0)).ToArray());
        Assert.Equal(new float[] { 2 }, Activations.Blend(new float[] { 1 }.ToVector(), new float[] { 3 }.ToVector(), new float[] { 0.5f }.ToVector(), 1f).ToArray());

        Assert.Equal(new double[] { 6, 10, 16 }, XMath.Map<Activations.Operators.TwiceOperator>(a).ToArray());
        Assert.True(typeof(Activations.Operators.TwiceOperator).IsNestedAssembly);
        Assert.True(typeof(Activations.Operators.Hypotenuse).IsNestedPublic);

        // 入力が複数だと書き換える先が決まらないので、InPlace 版は作らない
        Assert.DoesNotContain(typeof(Activations.InPlace).GetMethods(), method => method.Name is "Hypot" or "Affine" or "Blend");
    }

    private static (ImmutableArray<Diagnostic> Generator, Compilation Output) RunGenerator(string source)
    {
        var references = AppDomain.CurrentDomain.GetAssemblies()
            .Where(a => !a.IsDynamic && a.Location.Length > 0)
            .Select(a => a.Location)
            .Append(typeof(ElementwiseOperationAttribute).Assembly.Location)
            .Distinct()
            .Select(location => MetadataReference.CreateFromFile(location));
        var compilation = CSharpCompilation.Create("User", [CSharpSyntaxTree.ParseText(source)], references,
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        CSharpGeneratorDriver.Create(new OperationsGenerator())
            .RunGeneratorsAndUpdateCompilation(compilation, out var output, out ImmutableArray<Diagnostic> diagnostics);
        return (diagnostics, output);
    }

    [Fact]
    public void MixedViewsAndContainersCompileAgainstThePublicAssembly()
    {
        const string source = """
            using Tuna;
            using Tuna.LinearAlgebra;
            public static class Consumer
            {
                public static void Run()
                {
                    var a = new Matrix<double>(2, 3);
                    var w = new Vector<double>(2);
                    var destination = new Vector<double>(3);
                    Vector<double> same = XMath.Multiply(a.T, w, destination);
                    Vector<double> sliced = XMath.Multiply(a[.., ..].T, w.AsVectorSpan(), destination);
                    VectorSpan<double> view = XMath.Multiply(a.T, w, destination.AsVectorSpan());
                    ReadOnlyMatrixSpan<double> readOnly = a.T;
                    Vector<double> readOnlyResult = XMath.Multiply(readOnly, w, destination);
                    var matrixOutput = new Matrix<double>(3, 3);
                    Matrix<double> product = XMath.Multiply(a.T, a, matrixOutput);
                    var tensor = new Tensor<double>(2, 3, 2);
                    var tensorOutput = new Tensor<double>(3, 3, 2);
                    Tensor<double> batch = XMath.Multiply(tensor.T, tensor, tensorOutput);
                }
            }
            """;
        var (diagnostics, output) = RunGenerator(source);
        Assert.Empty(diagnostics.Where(d => d.Severity == DiagnosticSeverity.Error));
        Assert.Empty(output.GetDiagnostics().Where(d => d.Severity == DiagnosticSeverity.Error));
    }

    [Fact]
    public void InferenceOverloadsInExtensionBlocksExtendTheEntry()
    {
        // 拡張（Tuna.SignalProcessing の畳み込みなど）が C# 14 の拡張ブロックで XMath に足す関数にも、型推論のためのオーバーロードを生成する
        const string source = """
            using Tuna;
            using Tuna.LinearAlgebra;
            public static partial class SignalLike
            {
                extension(XMath)
                {
                    [InferenceOverloads]
                    public static Vector<T> Twice<T>(ReadOnlyVectorSpan<T> x) where T : unmanaged, System.Numerics.INumberBase<T> =>
                        throw new System.NotImplementedException();

                    [InferenceOverloads]
                    public static VectorSpan<T> Twice<T>(ReadOnlyVectorSpan<T> x, VectorSpan<T> destination) where T : unmanaged, System.Numerics.INumberBase<T> =>
                        throw new System.NotImplementedException();
                }
            }
            public static class Consumer
            {
                public static void Run()
                {
                    var x = new Vector<double>(3);
                    var m = new Matrix<double>(2, 3);
                    var destination = new Vector<double>(3);
                    Vector<double> a = XMath.Twice(x);
                    Vector<double> b = XMath.Twice(m.Row(0));
                    Vector<double> c = XMath.Twice(x, destination);
                    Vector<double> d = SignalLike.Twice(x);
                }
            }
            """;
        var (diagnostics, output) = RunGenerator(source);
        Assert.Empty(diagnostics);
        Assert.Empty(output.GetDiagnostics().Where(d => d.Severity == DiagnosticSeverity.Error));
        var generated = Assert.Single(output.SyntaxTrees, t => Path.GetFileName(t.FilePath).StartsWith("global.SignalLike.Extension_XMath_", StringComparison.Ordinal));
        Assert.EndsWith(".InferenceOverloads.g.cs", generated.FilePath, StringComparison.Ordinal);
        Assert.Contains("extension(global::Tuna.XMath)", generated.ToString());
    }

    [Fact]
    public void ExtensionBlocksWithTheSameReceiverNameGetDistinctFiles()
    {
        // 受け手の型名が同じで、型引数や名前空間だけが違う拡張ブロックも別のファイルに生成する。
        // 同じファイル名だと AddSource が例外を投げ（CS8785）、同じコンパイルのほかのオーバーロードも生成されなくなる
        const string source = """
            using Tuna;
            using Tuna.LinearAlgebra;
            namespace Other { public sealed class Vector<T>; }
            public static partial class CustomExtensions
            {
                extension(Vector<double>)
                {
                    [InferenceOverloads]
                    public static double First(ReadOnlyVectorSpan<double> x) => x[0];
                }
                extension(Vector<float>)
                {
                    [InferenceOverloads]
                    public static float First(ReadOnlyVectorSpan<float> x) => x[0];
                }
                extension(Other.Vector<double>)
                {
                    [InferenceOverloads]
                    public static double Last(ReadOnlyVectorSpan<double> x) => x[x.Length - 1];
                }

                [InferenceOverloads]
                public static T Head<T>(ReadOnlyVectorSpan<T> x) where T : unmanaged, System.Numerics.INumberBase<T> => x[0];
            }
            public static class Consumer
            {
                public static void Run()
                {
                    var d = new Vector<double>(3);
                    var f = new Vector<float>(3);
                    double a = Vector<double>.First(d);
                    float b = Vector<float>.First(f);
                    double c = Other.Vector<double>.Last(d);
                    double e = CustomExtensions.Head(d);
                }
            }
            """;
        var (diagnostics, output) = RunGenerator(source);
        Assert.Empty(diagnostics);
        Assert.Empty(output.GetDiagnostics().Where(d => d.Severity == DiagnosticSeverity.Error));
        var files = output.SyntaxTrees.Select(t => Path.GetFileName(t.FilePath))
            .Where(name => name.StartsWith("global.CustomExtensions.Extension_Vector_", StringComparison.Ordinal)).ToArray();
        Assert.Equal(3, files.Length);
        Assert.Equal(files.Length, files.Distinct(StringComparer.OrdinalIgnoreCase).Count());
    }

    [Fact]
    public void ElementwiseOperationsInExtensionBlocksAreReported()
    {
        // 演算子の構造体を拡張ブロックの中に入れ子にできないので、[ElementwiseOperation] は partial class に直接置く
        const string source = """
            using Tuna;
            public static partial class SignalLike
            {
                extension(XMath)
                {
                    [ElementwiseOperation] public static double F(double x) => x;
                }
            }
            """;
        var (diagnostics, _) = RunGenerator(source);
        var error = Assert.Single(diagnostics, d => d.Id == "TNG001");
        Assert.Contains("拡張ブロック", error.GetMessage());
    }

    [Fact]
    public void InvalidTargetsAreReportedAsDiagnostics()
    {
        const string source = """
            using Tuna;
            public static class NotPartial
            {
                [ElementwiseOperation] public static double F(double x) => x;
            }
            public static partial class WrongShape
            {
                [ElementwiseOperation] public static double G(float x) => x;
            }
            """;
        var references = AppDomain.CurrentDomain.GetAssemblies()
            .Where(a => !a.IsDynamic && a.Location.Length > 0)
            .Select(a => MetadataReference.CreateFromFile(a.Location))
            .Append(MetadataReference.CreateFromFile(typeof(ElementwiseOperationAttribute).Assembly.Location));
        var compilation = CSharpCompilation.Create("User", [CSharpSyntaxTree.ParseText(source)], references,
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        CSharpGeneratorDriver.Create(new OperationsGenerator())
            .RunGeneratorsAndUpdateCompilation(compilation, out _, out ImmutableArray<Diagnostic> diagnostics);
        Assert.Equal(2, diagnostics.Count(d => d.Id == "TNG001"));
        Assert.Contains(diagnostics, d => d.GetMessage().Contains("NotPartial"));
        Assert.Contains(diagnostics, d => d.GetMessage().Contains("G"));
    }

    [Fact]
    public void InvalidSpecificationsAndDuplicateOperatorNamesAreReported()
    {
        const string source = """
            using Tuna;
            public static partial class Bad
            {
                [ElementwiseOperation(Over = ["y"])] public static double A(double x) => x;
                [ElementwiseOperation(Over = [])] public static double B(double x) => x;
                [ElementwiseOperation(Over = ["x", "x"])] public static double C(double x) => x;
                [ElementwiseOperation(Over = [nameof(n)])] public static double D(int n, double x) => x;
                [ElementwiseOperation(Over = ["a", "b", "c", "d"])] public static double E(double a, double b, double c, double d) => a;
                [ElementwiseOperation(OperatorName = "1x")] public static double F(double x) => x;
                [ElementwiseOperation(OperatorContainer = "InPlace")] public static double G(double x) => x;
                [ElementwiseOperation] public static double H(double x, double destination) => x;
                [ElementwiseOperation(OperatorName = "I")] public static double I(double x) => x;
            }
            public static partial class Overloads
            {
                [ElementwiseOperation] public static double Gamma(double x) => x;
                [ElementwiseOperation] public static float Gamma(float x) => x;
                [ElementwiseOperation(OperatorName = "HalfGammaOperator")] public static System.Half Gamma(System.Half x) => x;
                [ElementwiseOperation(Over = [nameof(x), nameof(t)], OperatorContainer = "Operators")]
                public static double Lerp(double x, double y, double t) => x + (y - x) * t;
                [PairwiseOperation(OperatorName = "Dot", OperatorContainer = "Operators")]
                public static double Inner(Tuna.LinearAlgebra.ReadOnlyVectorSpan<double> x, Tuna.LinearAlgebra.ReadOnlyVectorSpan<double> y) => 0;
            }
            """;
        var (diagnostics, output) = RunGenerator(source);
        static string Target(Diagnostic d) => d.Location.SourceTree!.GetText().ToString(d.Location.SourceSpan);
        Assert.Equal(["A", "B", "C", "D", "E", "F", "G", "H", "I"], diagnostics.Where(d => d.Id == "TNG001").Select(Target).Order());
        var duplicates = diagnostics.Where(d => d.Id == "TNG002").ToArray();
        Assert.Equal(2, duplicates.Length);
        Assert.All(duplicates, d => Assert.Contains("GammaOperator", d.GetMessage()));

        // 重複したものは生成せず、それ以外（名前を変えた版・Over・入れ子のクラス・Pairwise）の生成コードはコンパイルできる
        Assert.Empty(output.GetDiagnostics().Where(d => d.Severity == DiagnosticSeverity.Error));
        var overloads = output.GetTypeByMetadataName("Overloads")!;
        Assert.NotNull(overloads.GetTypeMembers("HalfGammaOperator").SingleOrDefault());
        Assert.Empty(overloads.GetTypeMembers("GammaOperator"));
        var operators = overloads.GetTypeMembers("Operators").Single();
        Assert.Equal(["Dot", "LerpOperator"], operators.GetTypeMembers().Select(t => t.Name).Order());
        Assert.Contains(overloads.GetMembers("Lerp").OfType<IMethodSymbol>(),
            m => m.Parameters.Select(p => p.Type.Name).SequenceEqual(["ReadOnlyVectorSpan", "Double", "ReadOnlyVectorSpan"]));
    }

    [Fact]
    public void SeveralAttributesOnOneMethodGenerateEachVersion()
    {
        const string source = """
            using Tuna;
            public static partial class Several
            {
                [ElementwiseOperation(OperatorName = "PowScalar", OperatorContainer = "Operators")]
                [ElementwiseOperation(Over = [nameof(x), nameof(y)], OperatorName = "Pow", OperatorContainer = "Operators")]
                public static double Pow(double x, double y) => System.Math.Pow(x, y);
                [ElementwiseOperation]
                [ElementwiseOperation(Over = [nameof(x), nameof(y)])]
                public static double Same(double x, double y) => x;
            }
            """;
        var (diagnostics, output) = RunGenerator(source);

        // 演算子名を分けなければ、既定の名前 SameOperator が重なる
        var duplicates = diagnostics.Where(d => d.Id == "TNG002").ToArray();
        Assert.Equal(2, duplicates.Length);
        Assert.All(duplicates, d => Assert.Contains("SameOperator", d.GetMessage()));
        Assert.Empty(output.GetDiagnostics().Where(d => d.Severity == DiagnosticSeverity.Error));
        var several = output.GetTypeByMetadataName("Several")!;
        Assert.Equal(["Pow", "PowScalar"], several.GetTypeMembers("Operators").Single().GetTypeMembers().Select(t => t.Name).Order());
        var pow = several.GetMembers("Pow").OfType<IMethodSymbol>().Select(m => string.Join(",", m.Parameters.Select(p => p.Type.Name))).ToArray();
        Assert.Contains("ReadOnlyVectorSpan,Double", pow);
        Assert.Contains("ReadOnlyVectorSpan,ReadOnlyVectorSpan", pow);
    }
}
