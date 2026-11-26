using Tuna.LinearAlgebra;
using static Tuna.XMath;
using DoubleVector = Tuna.LinearAlgebra.Vector<double>;

namespace Tuna.Tests.LinearAlgebra;

// XMath の規約（新規割り当て・destination・インプレース、エイリアシング、型推論）とビューの共有を検証する。
[Trait("LinearAlgebra", "XMath")]
public class XMathTests
{
    private static Matrix<double> Sequence(int rows, int columns, double start = 1) =>
        Matrix<double>.FromColumnMajor(rows, columns, Enumerable.Range(0, rows * columns).Select(i => start + i).ToArray());

    private readonly struct Softplus : IUnaryOperator<double>
    {
        public static double Invoke(double x) => Math.Log(1 + Math.Exp(x));
    }

    private readonly struct Shift(double offset) : IStatefulUnaryOperator<double>
    {
        public double Invoke(double x) => x + offset;
    }

    private readonly struct SquaredDistance : IPairwiseOperator<double>
    {
        public static double Invoke(ReadOnlyVectorSpan<double> x, ReadOnlyVectorSpan<double> y)
        {
            double sum = 0;
            for (int i = 0; i < x.Length; i++)
            {
                sum += (x[i] - y[i]) * (x[i] - y[i]);
            }

            return sum;
        }
    }

    private readonly struct Gaussian(double lengthScale) : IStatefulPairwiseOperator<double>
    {
        public double Invoke(ReadOnlyVectorSpan<double> x, ReadOnlyVectorSpan<double> y) =>
            Math.Exp(-SquaredDistance.Invoke(x, y) / (2 * lengthScale * lengthScale));
    }

    [Fact]
    public void OperatorsComputeImmediatelyAndDoNotTrackLaterInputChanges()
    {
        var a = Sequence(2, 2);
        var b = Sequence(2, 2, 10);
        var sum = a + b;
        a[0, 0] = 100;
        Assert.Equal(new double[,] { { 11, 15 }, { 13, 17 } }, sum.ToArray());
        Assert.Equal(new double[,] { { -100, -3 }, { -2, -4 } }, (-a).ToArray());
        Assert.Equal(new double[,] { { 200, 6 }, { 4, 8 } }, (2.0 * a).ToArray());
    }

    [Fact]
    public void AllocatingAndDestinationVersionsAgreeBitForBit()
    {
        var a = Sequence(17, 9, -40);
        var b = Sequence(9, 33, -100);
        var x = Sequence(9, 1, -3).AsVector();
        var c = new Matrix<double>(17, 33);
        var y = new DoubleVector(17);
        Assert.Equal(Multiply(a, b).AsSpan().ToArray(), Multiply(a, b, c).AsSpan().ToArray());
        Assert.Equal(Multiply(a, x).ToArray(), Multiply(a, x, y).ToArray());
        var e = new DoubleVector(9);
        Assert.Equal(Exp(x).ToArray(), Exp(x, e).ToArray());
        Assert.Same(c, Multiply(a, b, c));
    }

    [Fact]
    public void TransposedViewsGoStraightToGemmWithoutCopying()
    {
        var a = LinearAlgebraTestUtilities.Random(33, 31, 5);
        var b = LinearAlgebraTestUtilities.Random(33, 17, 6);
        var expected = Multiply(XMath.Transpose(a), b).AsSpan().ToArray();
        Assert.Equal(expected, Multiply(a.T, b).AsSpan().ToArray());
        Assert.Equal(expected, (a.T * b).AsSpan().ToArray());
        Assert.Equal(Multiply(b, XMath.Transpose(b)).AsSpan().ToArray(), Multiply(b, b.T).AsSpan().ToArray());

        // 単精度は GEMV を列ごとに呼ぶ経路（加算の順序が異なるので丸め誤差の範囲で一致）、
        // 一般のストライドは T.Zero から順に足す経路（単純な三重ループとビット単位で一致）を通る
        var f = a.ConvertChecked<float>();
        var g = b.ConvertChecked<float>();
        var naive = NaiveProduct(f.T, g);
        var actual = Multiply(f.T, g).ToArray();
        for (int i = 0; i < naive.GetLength(0); i++)
        {
            for (int j = 0; j < naive.GetLength(1); j++)
            {
                Assert.Equal(naive[i, j], actual[i, j], 1e-4f);
            }
        }

        var everyOtherRow = new MatrixSpan<float>(f.AsSpan(), [16, 31], [2, 33]);
        Assert.Equal(NaiveProduct(everyOtherRow, f[..31, ..]), Multiply(everyOtherRow, f[..31, ..]).ToArray());
    }

    private static float[,] NaiveProduct(ReadOnlyMatrixSpan<float> a, ReadOnlyMatrixSpan<float> b)
    {
        var result = new float[a.Rows, b.Columns];
        for (int i = 0; i < a.Rows; i++)
        {
            for (int j = 0; j < b.Columns; j++)
            {
                float sum = 0;
                for (int p = 0; p < a.Columns; p++)
                {
                    sum += a[i, p] * b[p, j];
                }

                result[i, j] = sum;
            }
        }

        return result;
    }

    [Fact]
    public void ViewsShareStorageWithTheContainer()
    {
        var a = Sequence(3, 4);
        a.T[3, 2] = 100;
        Assert.Equal(100, a[2, 3]);
        a.Row(1)[2] = 200;
        Assert.Equal(200, a[1, 2]);
        a[1.., 2..].Fill(0);
        Assert.Equal(new double[,] { { 1, 4, 7, 10 }, { 2, 5, 0, 0 }, { 3, 6, 0, 0 } }, a.ToArray());
        Assert.Equal(new double[] { 1, 5, 0 }, a.Diagonal().ToArray());
        Assert.Equal(new double[] { 4, 5, 6 }, a.Column(1).ToArray());
        Assert.Equal(XMath.Transpose(a).ToArray(), a.T.ToArray());
    }

    [Fact]
    public void CompoundAssignmentUpdatesTheSameStorage()
    {
        var x = new DoubleVector([1, 2, 3]);
        var alias = x;
        var before = x.AsSpan().ToArray();
        x += new DoubleVector([10, 20, 30]);
        x *= 2;
        Assert.Same(alias, x);
        Assert.Equal(new double[] { 22, 44, 66 }, alias.ToArray());
        Assert.Equal(new double[] { 1, 2, 3 }, before);

        var a = Sequence(2, 2);
        a -= XMath.Transpose(a);
        Assert.Equal(new double[,] { { 0, 1 }, { -1, 0 } }, a.ToArray());
    }

    [Fact]
    public void InPlaceOperationsRewriteTheirArgument()
    {
        var a = Sequence(3, 3);
        InPlace.Add(a.Diagonal(), 0.5);
        Assert.Equal(new double[] { 1.5, 5.5, 9.5 }, a.Diagonal().ToArray());

        var x = new DoubleVector([-1, 0, 2]);
        var expected = x.ToArray().Select(Math.Exp).ToArray();
        InPlace.Exp(x);
        Assert.Equal(expected, x.ToArray());

        InPlace.Map(x, new Shift(-1));
        Assert.Equal(expected.Select(v => v - 1).ToArray(), x.ToArray());

        XMath.InPlace.Multiply(a.Row(0), 2.0);
        Assert.Equal(new double[] { 3, 8, 14 }, a.Row(0).ToArray());
    }

    [Fact]
    public void MapTakesStatelessOperatorsAsTypeArgumentsAndStatefulOnesAsValues()
    {
        var x = new DoubleVector([-2, 0, 3]);
        var expected = x.ToArray().Select(v => Math.Log(1 + Math.Exp(v))).ToArray();
        Assert.Equal(expected, Map<Softplus>(x).ToArray());
        Assert.Equal(expected, Map<double, Softplus>(x).ToArray());
        var z = new DoubleVector(3);
        Assert.Same(z, Map(x, new Shift(1), z));
        Assert.Equal(x.ToArray().Select(v => v + 1).ToArray(), z.ToArray());
        Map<Softplus>(x, z);
        Assert.Equal(expected, z.ToArray());

        var a = Sequence(3, 3, -4);
        var diagonal = Map<Softplus>(a.Diagonal());
        Assert.Equal(a.Diagonal().ToArray().Select(v => Math.Log(1 + Math.Exp(v))).ToArray(), diagonal.ToArray());
        Assert.Equal(Map<Softplus>(a).ToArray(), Map<double, Softplus>(a).ToArray());
        InPlace.Map<Softplus>(a);
        Assert.Equal(diagonal.ToArray(), a.Diagonal().ToArray());
    }

    [Fact]
    public void SourceGeneratorEmitsFixedElementTypeAndInferenceOverloads()
    {
        // [FixedElementTypeOverloads] の一覧の型ごとに、型引数が演算子だけの Map がある
        foreach (var element in new[] { typeof(Half), typeof(float), typeof(double), typeof(decimal), typeof(System.Numerics.Complex) })
        {
            var view = typeof(ReadOnlyVectorSpan<>).MakeGenericType(element);
            Assert.Contains(typeof(XMath).GetMethods(), m => m.Name == "Map" && m.IsGenericMethodDefinition
                && m.GetGenericArguments().Length == 1 && m.GetParameters().Select(p => p.ParameterType).SequenceEqual([view]));
        }

        // [InferenceOverloads] の基本形 Add(ReadOnlyVectorSpan<T>, ReadOnlyVectorSpan<T>) から、書き込み可能なビュー版とコンテナー版ができている
        var adds = typeof(XMath).GetMethods().Where(m => m.Name == "Add" && m.GetParameters().Length == 2)
            .Select(m => m.GetParameters()[0].ParameterType.GetGenericTypeDefinition()).ToHashSet();
        Assert.Superset(new HashSet<Type> { typeof(ReadOnlyVectorSpan<>), typeof(VectorSpan<>), typeof(DoubleVector).GetGenericTypeDefinition(),
            typeof(ReadOnlyMatrixSpan<>), typeof(MatrixSpan<>), typeof(Matrix<>) }, adds);
    }

    [Fact]
    public void PairwiseAppliesOperatorsToEveryPairOfColumns()
    {
        // 各列が1点（2次元）。行が点の配置なら X.T を渡す
        var x = Matrix<double>.FromColumnMajor(2, 3, [0, 0, 1, 0, 0, 2]);
        var y = Matrix<double>.FromColumnMajor(2, 2, [1, 1, 0, 0]);
        var distances = Pairwise<SquaredDistance>(x, y);
        Assert.Equal(new double[,] { { 2, 0 }, { 1, 1 }, { 2, 4 } }, distances.ToArray());
        var kernel = Pairwise(x, x, new Gaussian(1));
        Assert.Equal(1.0, kernel[1, 1]);
        Assert.Equal(Math.Exp(-0.5), kernel[0, 1]);
        Assert.Equal(kernel.ToArray(), XMath.Transpose(kernel).ToArray());
        var rows = Pairwise<SquaredDistance>(XMath.Transpose(x).T, y);
        Assert.Equal(distances.ToArray(), rows.ToArray());
        Assert.Throws<ArgumentException>(() => { Pairwise<SquaredDistance>(x, Sequence(3, 2)); });
    }

    [Fact]
    public void MixedArgumentKindsInferTheElementType()
    {
        var a = Sequence(3, 3);
        var x = new DoubleVector([1, 2, 3]);
        // コンテナー・書き込み可能ビュー・読み取り専用ビューの組み合わせが、型引数を書かずに解決できること
        Assert.Equal(18, Sum(a.Row(2)));
        Assert.Equal(Sum(a.Diagonal()), Sum((ReadOnlyVectorSpan<double>)a.Diagonal()));
        Assert.Equal(Multiply(a, x).ToArray(), Multiply(a, a.Column(0)).ToArray());
        Assert.Equal(Multiply(a, x).ToArray(), (a * x).ToArray());
        Assert.Equal(Multiply(a, x).ToArray(), (a.AsMatrixSpan() * x.AsVectorSpan()).ToArray());
        Assert.Equal(Add(x, a.Row(0)).ToArray(), Add(a.Row(0), x).ToArray());
        Assert.Equal(Exp(a.Diagonal()).ToArray(), Exp(a).Diagonal().ToArray());
        Assert.Equal(new double[] { 6, 15, 24 }, ColumnSums(a).ToArray());
        Assert.Equal(new double[] { 12, 15, 18 }, RowSums(a).ToArray());
        Assert.Equal(RowSums(a).ToArray(), ColumnSums(a.T).ToArray());
        Assert.Equal(Hadamard(x, x).ToArray(), new double[] { 1, 4, 9 });
    }

    [Fact]
    public void ExactAliasIsAllowedButPartialOverlapIsRejected()
    {
        var x = new DoubleVector([1, 2, 3, 4]);
        Add(x, x, x);
        Assert.Equal(new double[] { 2, 4, 6, 8 }, x.ToArray());
        Assert.Throws<ArgumentException>(() => { Add(x[..3], x[..3], x[1..]); });

        // 範囲は重なるが要素を共有しない行どうしは受け付ける
        var a = Sequence(3, 4);
        Add(a.Row(0), a.Row(1), a.Row(2));
        Assert.Equal(new double[] { 3, 9, 15, 21 }, a.Row(2).ToArray());

        Assert.Throws<ArgumentException>(() => { Multiply(a, Sequence(4, 4), a[.., ..]); });
        var square = Sequence(3, 3);
        Assert.Throws<ArgumentException>(() => { XMath.Transpose(square, square); });
        Assert.Throws<ArgumentException>(() => { square += square.T; });
    }

    [Fact]
    public void InvalidShapesAndViewsAreRejected()
    {
        var a = Sequence(2, 3);
        Assert.Throws<ArgumentException>(() => { Add(a, Sequence(3, 2)); });
        Assert.Throws<ArgumentException>(() => { Multiply(a, a); });
        Assert.Throws<ArgumentException>(() => { Multiply(a, Sequence(3, 3), new Matrix<double>(3, 3)); });
        Assert.Throws<ArgumentException>(() => { _ = new MatrixSpan<double>(new double[6], [2, 3], [0, 2]); });
        Assert.Throws<ArgumentException>(() => { _ = new MatrixSpan<double>(new double[5], [2, 3], [1, 2]); });
        Assert.Throws<ArgumentOutOfRangeException>(() => { _ = a.T[3, 0]; });
        var broadcast = new ReadOnlyMatrixSpan<double>(new double[] { 1, 2, 3 }, [2, 3], [0, 1]);
        Assert.Equal(new double[,] { { 1, 2, 3 }, { 1, 2, 3 } }, broadcast.ToArray());
    }

    [Fact]
    public void CholeskyCanOverwriteItsInputAndSolveInPlace()
    {
        var original = LinearAlgebraTestUtilities.RandomPositiveDefinite(20, 3);
        var expected = original.Cholesky();
        var a = original.Clone();
        var chol = CholeskyDecomposition.Compute(a, a);
        Assert.Equal(expected.L.AsSpan().ToArray(), a.AsSpan().ToArray());
        Assert.Equal(expected.L.AsSpan().ToArray(), chol.L.AsSpan().ToArray());

        var b = Enumerable.Range(0, 20).Select(i => i + 1.0).ToArray();
        var solution = expected.Solve(b);
        chol.Solve(b, b);
        Assert.Equal(solution, b);
    }

    [Fact]
    public void CoreReferencesNoExtension()
    {
        // コアは拡張（FFT・ウェーブレット・畳み込みの Tuna.SignalProcessing、確率分布などの統計の Tuna.Statistics、
        // 最適化の Tuna.Optimization、推論の Tuna.Probabilistic、数値積分の Tuna.Integration、自動微分の Tuna.Differentiation）を参照せず、それらの型も含まない。
        // 畳み込みは拡張が XMath への静的拡張メンバーとして足すので、コアの XMath 自体には Convolve がない
        Assert.DoesNotContain(typeof(XMath).Assembly.GetReferencedAssemblies(), a => a.Name!.StartsWith("Tuna", StringComparison.Ordinal));
        Assert.DoesNotContain(typeof(XMath).GetMethods(), m => m.Name is "Convolve" or "Correlate" or "AutoCorrelate" or "ColumnAutoCorrelations");
        // 拡張の型は拡張のアセンブリ名から始まる名前空間にあるので、コアにはその名前空間がない
        string[] extensions = ["Tuna.SignalProcessing", "Tuna.Statistics", "Tuna.Optimization", "Tuna.Probabilistic", "Tuna.Integration", "Tuna.Differentiation"];
        Assert.DoesNotContain(typeof(XMath).Assembly.GetTypes(), t => t.Namespace is { } ns
            && extensions.Any(e => ns == e || ns.StartsWith(e + ".", StringComparison.Ordinal)));
        Assert.DoesNotContain(typeof(XMath).Assembly.GetTypes(), t => t.Name is "Fourier" or "Wavelets" or "XMathExtensions");
    }
}
