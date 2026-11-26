using System.Numerics;
using Tuna.Differentiation;
using Tuna.LinearAlgebra;
using static Tuna.Tests.Differentiation.ElementwiseOperations;

namespace Tuna.Tests.Differentiation;

/// <summary>
/// ブロードキャストのノード（<see cref="VectorNode{T}.Broadcast(int)"/>・<see cref="VectorNode{T}.BroadcastRows(int)"/>）。
/// 値はコアの同じビューに XMath を使った結果と同じビット。ベクトルの勾配は、ブロードキャストを写した行列を葉にして同じモデルを記録した勾配の
/// 行ごとの和（各行がベクトルなら列ごとの和）と同じビットで、ほかの葉の勾配も同じビット。
/// </summary>
public class ArrayBroadcastTests
{
    // 並べ方。Columns は v.Broadcast(n)（各列が v）、Rows は v.BroadcastRows(n)（各行が v）、Transposed の付くものはその転置。
    internal enum Arrangement
    {
        Columns,
        Rows,
        ColumnsTransposed,
        RowsTransposed,
    }

    private static readonly Arrangement[] Arrangements = [Arrangement.Columns, Arrangement.Rows, Arrangement.ColumnsTransposed, Arrangement.RowsTransposed];

    // ベクトルの長さと並べる本数。1 本はブロードキャストでない普通のビューになる。
    private static readonly (int Length, int Count)[] Shapes = [(5, 4), (1, 6), (6, 1), (37, 23)];

    public static TheoryData<string> BinaryCases() => BinaryNames();

    public static TheoryData<string> UnaryCases() => UnaryNames();

    public static TheoryData<string> ScalarCases() => ScalarNames();

    public static TheoryData<string> ActivationCases() => new(ArrayActivationTests.Names);

    public static TheoryData<string> ReductionCases() => new(ArrayReductionTests.MatrixNames);

    [Theory]
    [MemberData(nameof(BinaryCases))]
    public void BinaryValuesMatchXMathBitForBit(string name)
    {
        BinaryValues<double>(name);
        BinaryValues<float>(name);
    }

    [Theory]
    [MemberData(nameof(BinaryCases))]
    public void BinaryGradientsAreSumsOfTheMaterializedGradients(string name)
    {
        // 左の引数を (2, 3)、右の引数を (0.5, 1.5) から取り、どの演算も有限の勾配にする。
        foreach (bool left in new[] { true, false })
        {
            BinaryGradients<double>(name, left);
            BinaryGradients<float>(name, left);
        }
    }

    [Theory]
    [MemberData(nameof(UnaryCases))]
    public void UnaryGradientsAreSumsOfTheMaterializedGradients(string name)
    {
        var range = name == "Log1mExp" ? (-0.9, -0.1) : (0.1, 0.9);
        ForEachShape<double>(range, (tape, b, _, _) => Reduce(tape, Apply(name, b)), sameLoss: true, name);
        ForEachShape<float>(range, (tape, b, _, _) => Reduce(tape, Apply(name, b)), sameLoss: true, name);
    }

    [Theory]
    [MemberData(nameof(ActivationCases))]
    public void ActivationGradientsAreSumsOfTheMaterializedGradients(string name)
    {
        double parameter = ArrayActivationTests.DefaultParameter(name);
        ForEachShape<double>((-3, 3), (tape, b, _, _) => Reduce(tape, ArrayActivationTests.Apply(name, b, parameter)), sameLoss: true, name);
        ForEachShape<float>((-3, 3), (tape, b, _, _) => Reduce(tape, ArrayActivationTests.Apply(name, b, parameter)), sameLoss: true, name);
    }

    [Theory]
    [MemberData(nameof(ReductionCases))]
    public void ReductionGradientsAreSumsOfTheMaterializedGradients(string name)
    {
        // 列・行ごとの縮約とソフトマックス。並べたビューの値は、写した行列の値と同じビット。
        ForEachShape<double>((-3, 3), (tape, b, _, _) => ArrayReductionTests.Loss(tape, name, b), sameLoss: true, name);
        ForEachShape<float>((-3, 3), (tape, b, _, _) => ArrayReductionTests.Loss(tape, name, b), sameLoss: true, name);
    }

    [Theory]
    [MemberData(nameof(ScalarCases))]
    public void ScalarOperandGradientsAreSumsOfTheMaterializedGradients(string name)
    {
        // 記録つきのスカラーの勾配（ビューの論理的な要素すべての寄与の和）も同じビット。
        ForEachShape<double>((2, 3), (tape, b, _, s) => Reduce(tape, Apply(name, b, s)), sameLoss: true, name);
        ForEachShape<float>((2, 3), (tape, b, _, s) => Reduce(tape, Apply(name, b, s)), sameLoss: true, name);
    }

    [Fact]
    public void ProductsWithBroadcastOperandsMatchTheMaterializedGradients()
    {
        Products<double>();
        Products<float>();
    }

    [Fact]
    public void BothOperandsCanBeBroadcast()
    {
        // 外和 v·1ᵀ + 1·wᵀ と、同じベクトルを並べたもの同士の積。
        Outer<double>();
        Outer<float>();
    }

    [Fact]
    public void SlicesAndTransposesAreBroadcastsOfTheSlicedVector()
    {
        SlicesAndTransposes<double>();
        SlicesAndTransposes<float>();
    }

    [Fact]
    public void ContributionsFromSeveralUsesAccumulateInRecordOrder()
    {
        RepeatedUses<double>();
        RepeatedUses<float>();
    }

    [Fact]
    public void EmptyBroadcastsHaveNoContribution()
    {
        var tape = new Tape();
        var v = tape.Leaf(new LinearAlgebra.Vector<double>([1.5, -2, 3]));
        var empty = v.Broadcast(0);
        Assert.Equal((3, 0), (empty.Rows, empty.Columns));
        Assert.Equal((0, 3), (v.BroadcastRows(0).Rows, v.BroadcastRows(0).Columns));
        var none = tape.Leaf(new LinearAlgebra.Vector<double>(0));
        Assert.Equal((0, 4), (none.Broadcast(4).Rows, none.Broadcast(4).Columns));
        Assert.Equal((4, 0), (none.BroadcastRows(4).Rows, none.BroadcastRows(4).Columns));

        // 0 列のビューに演算と積を通しても、勾配は 0（届かない）。0 要素のベクトルを並べたものも同じ。
        var product = XMath.Exp(empty) * tape.Constant(new Matrix<double>(0, 2));
        ScalarNode loss = Reduce(tape, product) + Reduce(tape, XMath.Gelu(none.Broadcast(4))) + Reduce(tape, XMath.Exp(empty.T));
        tape.Backward(loss);
        Assert.Equal(0, loss.Value);
        Assert.Equal(new double[3], tape.GetGradient(v).ToArray());
        Assert.Equal(0, tape.GetGradient(none).Length);
        Assert.Equal((3, 0), (tape.GetGradient(empty).Rows, tape.GetGradient(empty).Columns));
    }

    [Fact]
    public void NonFiniteContributionsFollowIeeeSums()
    {
        // 寄与の行ごとの和は IEEE 754 のとおり（+∞ と −∞ を含む行と NaN を含む行は NaN、+∞ だけなら +∞）。
        double inf = double.PositiveInfinity;
        var c = new Matrix<double>(4, 3);
        double[][] rows = [[inf, 1, 1], [inf, -inf, 1], [double.NaN, 1, 1], [1, 2, 3]];
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                c[i, j] = rows[i][j];
            }
        }

        var tape = new Tape();
        var v = tape.Leaf(new LinearAlgebra.Vector<double>([1, 1, 1, 1]));
        var ones = new LinearAlgebra.Vector<double>([1, 1, 1]);
        var y = XMath.Hadamard(v.Broadcast(3), tape.Constant(c)) * tape.Constant(ones);
        tape.Backward(XMath.Sum(y));
        var gradient = tape.GetGradient(v);
        Assert.Equal(inf, gradient[0]);
        Assert.True(double.IsNaN(gradient[1]));
        Assert.True(double.IsNaN(gradient[2]));
        Assert.Equal(6, gradient[3]);

        // 列ごとの和（行として並べた場合）も同じ。
        var second = new Tape();
        var w = second.Leaf(new LinearAlgebra.Vector<double>([1, 1, 1, 1]));
        var z = XMath.Hadamard(w.BroadcastRows(3), second.Constant(c.AsMatrixSpan().T)) * second.Constant(new LinearAlgebra.Vector<double>([1, 1, 1, 1]));
        second.Backward(XMath.Sum(z));
        var columns = second.GetGradient(w);
        Assert.Equal(inf, columns[0]);
        Assert.True(double.IsNaN(columns[1]));
        Assert.True(double.IsNaN(columns[2]));
        Assert.Equal(6, columns[3]);
    }

    [Fact]
    public void InvalidBroadcastsThrow()
    {
        var tape = new Tape();
        var v = tape.Leaf(new LinearAlgebra.Vector<double>([1, 2, 3]));
        Assert.Throws<ArgumentOutOfRangeException>("count", () => v.Broadcast(-1));
        Assert.Throws<ArgumentOutOfRangeException>("rows", () => v.BroadcastRows(-1));

        // 要素数 Length·count が int に収まらないビューは作らない（0 要素のベクトルは何本でも並べられる）。
        Assert.Throws<ArgumentOutOfRangeException>("count", () => v.Broadcast(int.MaxValue / 3 + 1));
        Assert.Throws<ArgumentOutOfRangeException>("rows", () => v.BroadcastRows(int.MaxValue / 3 + 1));
        Assert.Equal(int.MaxValue / 3, v.Broadcast(int.MaxValue / 3).Columns);
        Assert.Equal(int.MaxValue, tape.Leaf(new LinearAlgebra.Vector<double>(0)).Broadcast(int.MaxValue).Columns);

        // 形の違う要素演算・積と、古い世代のノード。
        var x = tape.Leaf(new Matrix<double>(3, 2));
        Assert.Throws<ArgumentException>(() => x + v.Broadcast(3));
        Assert.Throws<ArgumentException>(() => x * v.BroadcastRows(3));
        tape.Reset();
        Assert.Throws<InvalidOperationException>(() => v.Broadcast(2));
        Assert.Throws<InvalidOperationException>(() => v.BroadcastRows(2));
    }

    [Fact]
    public void BroadcastModelGradientMatchesCentralDifferences()
    {
        var random = new Random(52);
        var point = ArrayProductTests.RandomVector<double>(random, 10);
        var constants = new LayerConstants<double>(random);
        ArrayElementwiseGradientTests.AssertMatchesCentralDifferences(point, (tape, v) => Layer(tape, v, constants), 1e-7);
    }

    [Fact]
    public void SingleGradientsAgreeWithDoubleGradients()
    {
        var random = new Random(53);
        var point = ArrayProductTests.RandomVector<double>(random, 10);
        var constants = new LayerConstants<double>(random);
        var tape = new Tape();
        var v = tape.Leaf(point);
        tape.Backward(Layer(tape, v, constants));
        var expected = tape.GetGradient(v).ToArray();

        var single = new Tape();
        var w = single.Leaf(point.ConvertChecked<float>());
        single.Backward(Layer(single, w, constants.Convert<float>()));
        var actual = single.GetGradient(w).ToArray();
        double scale = expected.Max(e => Math.Abs(e));
        for (int i = 0; i < expected.Length; i++)
        {
            DifferentiationAssert.Close(expected[i], actual[i], 1e-4, 1e-5 * scale, $"gradient[{i}]");
        }
    }

    // 損失の集約。Σᵢ sᵢ·(Y·r)ᵢ（逆伝播の随伴は s·rᵀ）。r・s は形から決める乱数。
    internal static ScalarNode Reduce<T>(Tape tape, MatrixNode<T> y) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var random = new Random(31 * y.Rows + y.Columns);
        var r = ArrayProductTests.RandomVector<T>(random, y.Columns);
        var s = ArrayProductTests.RandomVector<T>(random, y.Rows);
        return XMath.Dot(y * tape.Constant(r), tape.Constant(s));
    }

    internal static MatrixNode<T> ArrangeNode<T>(VectorNode<T> v, int count, Arrangement arrangement) where T : unmanaged, IFloatingPointIeee754<T> => arrangement switch
    {
        Arrangement.Columns => v.Broadcast(count),
        Arrangement.Rows => v.BroadcastRows(count),
        Arrangement.ColumnsTransposed => v.Broadcast(count).T,
        _ => v.BroadcastRows(count).T,
    };

    internal static ReadOnlyMatrixSpan<T> ArrangeSpan<T>(ReadOnlyVectorSpan<T> v, int count, Arrangement arrangement) where T : unmanaged, IFloatingPointIeee754<T> => arrangement switch
    {
        Arrangement.Columns => v.Broadcast(count),
        Arrangement.Rows => v.BroadcastRows(count),
        Arrangement.ColumnsTransposed => v.Broadcast(count).T,
        _ => v.BroadcastRows(count).T,
    };

    // 各列がベクトルになる並べ方（勾配は行ごとの和）か。
    private static bool EachColumnIsTheVector(Arrangement arrangement) => arrangement is Arrangement.Columns or Arrangement.RowsTransposed;

    private static (int Rows, int Columns) ShapeOf(int length, int count, Arrangement arrangement)
        => EachColumnIsTheVector(arrangement) ? (length, count) : (count, length);

    private static LinearAlgebra.Vector<T> Uniform<T>(Random random, int length, (double Low, double High) range) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var result = new LinearAlgebra.Vector<T>(length);
        for (int i = 0; i < length; i++)
        {
            result[i] = T.CreateTruncating(range.Low + (range.High - range.Low) * random.NextDouble());
        }

        return result;
    }

    private static Matrix<T> Uniform<T>(Random random, int rows, int columns, (double Low, double High) range) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var result = new Matrix<T>(rows, columns);
        Uniform<T>(random, rows * columns, range).AsSpan().CopyTo(result.AsSpan());
        return result;
    }

    private static void ForEachShape<T>((double Low, double High) range, Func<Tape, MatrixNode<T>, MatrixNode<T>, ScalarNode, ScalarNode> model, bool sameLoss, string context)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        var random = new Random(context.Length);
        foreach (var (length, count) in Shapes)
        {
            foreach (var arrangement in Arrangements)
            {
                var (rows, columns) = ShapeOf(length, count, arrangement);
                AssertMatchesMaterialized(Uniform<T>(random, length, range), count, arrangement, Uniform<T>(random, rows, columns, range), model, sameLoss,
                    $"{context} {typeof(T).Name} {arrangement} {length}x{count}");
            }
        }
    }

    // ブロードキャストのテープと、並べたものを写した行列を葉にしたテープで model を記録する。model は並べたノード b、葉の行列 x、葉のスカラー s を受け取る。
    // v の勾配が写した行列の勾配の行（列）ごとの和と、x・s の勾配と、b のノードの勾配（GetGradient）が同じビットであることを確かめる。
    // sameLoss なら損失も同じビット（行列積の値は、ブロードキャストのビューではコアの素朴な経路を通るので近さだけ比べる）。
    private static void AssertMatchesMaterialized<T>(LinearAlgebra.Vector<T> v, int count, Arrangement arrangement, Matrix<T> x,
        Func<Tape, MatrixNode<T>, MatrixNode<T>, ScalarNode, ScalarNode> model, bool sameLoss, string context) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var tape = new Tape();
        var leaf = tape.Leaf(v);
        var other = tape.Leaf(x);
        var scalar = tape.Leaf(0.75);
        var b = ArrangeNode(leaf, count, arrangement);
        AssertSameBits<T>(ArrangeSpan<T>(v, count, arrangement).ToMatrix().AsSpan(), b.Value.ToMatrix().AsSpan(), $"{context} value");
        ScalarNode loss = model(tape, b, other, scalar);
        tape.Backward(loss);

        // 1 本だけ並べたものはブロードキャストでない普通のビュー（行として並べたものはベクトルの転置）なので、写した列の転置と比べる。
        var reference = new Tape();
        bool single = count == 1 && !EachColumnIsTheVector(arrangement);
        var leafMatrix = reference.Leaf(single ? v.Broadcast(1) : ArrangeSpan<T>(v, count, arrangement));
        var materialized = single ? leafMatrix.T : leafMatrix;
        var referenceOther = reference.Leaf(x);
        var referenceScalar = reference.Leaf(0.75);
        ScalarNode expectedLoss = model(reference, materialized, referenceOther, referenceScalar);
        reference.Backward(expectedLoss);
        var gradient = reference.GetGradient(materialized);
        var expected = EachColumnIsTheVector(arrangement) ? XMath.RowSums(gradient) : XMath.ColumnSums(gradient);

        if (sameLoss)
        {
            AssertSameBits<double>([expectedLoss.Value], [loss.Value], $"{context} loss");
        }
        else
        {
            double tolerance = typeof(T) == typeof(float) ? 1e-5 : 1e-13;
            DifferentiationAssert.Close(expectedLoss.Value, loss.Value, tolerance, tolerance, $"{context} loss");
        }

        AssertSameBits<T>(expected.AsSpan(), tape.GetGradient(leaf).ToArray(), $"{context} gradient");
        AssertSameBits<T>(reference.GetGradient(referenceOther).ToMatrix().AsSpan(), tape.GetGradient(other).ToMatrix().AsSpan(), $"{context} other");
        AssertSameBits<double>([reference.GetGradient(referenceScalar)], [tape.GetGradient(scalar)], $"{context} scalar");
        AssertSameBits<T>(ArrangeSpan<T>(expected, count, arrangement).ToMatrix().AsSpan(), tape.GetGradient(b).ToMatrix().AsSpan(), $"{context} broadcast node");

        // Gradient(output, node, destination) も同じ形に並べたものを書く。
        var destination = new Matrix<T>(b.Rows, b.Columns);
        var again = new Tape();
        var againLeaf = again.Leaf(v);
        var againNode = ArrangeNode(againLeaf, count, arrangement);
        again.Gradient(model(again, againNode, again.Leaf(x), again.Leaf(0.75)), againNode, destination);
        AssertSameBits<T>(ArrangeSpan<T>(expected, count, arrangement).ToMatrix().AsSpan(), destination.AsSpan(), $"{context} Gradient");
    }

    private static void BinaryValues<T>(string name) where T : unmanaged, IFloatingPointIeee754<T>
    {
        // 特別な値（±0・非正規化数・裾・±∞・NaN）を含む入力で、ブロードキャストを左右・行と列・両方に置く。
        double[] inputs = Inputs(7, 30, 3);
        const int rows = 9, columns = 8;
        var data = new Matrix<T>(rows, columns);
        for (int k = 0; k < rows * columns; k++)
        {
            data.AsSpan()[k] = T.CreateTruncating(inputs[(5 * k) % inputs.Length]);
        }

        var column = new LinearAlgebra.Vector<T>(rows);
        var row = new LinearAlgebra.Vector<T>(columns);
        for (int i = 0; i < rows; i++)
        {
            column[i] = T.CreateTruncating(inputs[(3 * i + 1) % inputs.Length]);
        }

        for (int j = 0; j < columns; j++)
        {
            row[j] = T.CreateTruncating(inputs[(7 * j + 2) % inputs.Length]);
        }

        var tape = new Tape();
        var x = tape.Leaf(data);
        var v = tape.Leaf(column);
        var w = tape.Leaf(row);
        ReadOnlyMatrixSpan<T> vColumns = column.Broadcast(columns), wRows = row.BroadcastRows(rows);
        string context = $"{name} {typeof(T).Name}";
        AssertSameBits<T>(Expected<T>(name, data, vColumns).AsSpan(), Apply(name, x, v.Broadcast(columns)).Value.ToMatrix().AsSpan(), $"{context} x ⊕ v");
        AssertSameBits<T>(Expected<T>(name, vColumns, data).AsSpan(), Apply(name, v.Broadcast(columns), x).Value.ToMatrix().AsSpan(), $"{context} v ⊕ x");
        AssertSameBits<T>(Expected<T>(name, data, wRows).AsSpan(), Apply(name, x, w.BroadcastRows(rows)).Value.ToMatrix().AsSpan(), $"{context} x ⊕ w");
        AssertSameBits<T>(Expected<T>(name, wRows, data).AsSpan(), Apply(name, w.BroadcastRows(rows), x).Value.ToMatrix().AsSpan(), $"{context} w ⊕ x");
        AssertSameBits<T>(Expected<T>(name, vColumns, wRows).AsSpan(), Apply(name, v.Broadcast(columns), w.BroadcastRows(rows)).Value.ToMatrix().AsSpan(),
            $"{context} v ⊕ w");
        AssertSameBits<T>(Expected<T>(name, data.AsMatrixSpan().T, column.BroadcastRows(columns)).AsSpan(),
            Apply(name, x.T, v.Broadcast(columns).T).Value.ToMatrix().AsSpan(), $"{context} xᵀ ⊕ vᵀ");
    }

    private static void BinaryGradients<T>(string name, bool left) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var random = new Random(name.Length + (left ? 1 : 0));
        (double, double) broadcastRange = left ? (2, 3) : (0.5, 1.5), otherRange = left ? (0.5, 1.5) : (2, 3);
        foreach (var (length, count) in Shapes)
        {
            foreach (var arrangement in Arrangements)
            {
                var (rows, columns) = ShapeOf(length, count, arrangement);
                AssertMatchesMaterialized(Uniform<T>(random, length, broadcastRange), count, arrangement, Uniform<T>(random, rows, columns, otherRange),
                    (tape, b, x, _) => Reduce(tape, left ? Apply(name, b, x) : Apply(name, x, b)), sameLoss: true,
                    $"{name} {typeof(T).Name} {(left ? "left" : "right")} {arrangement} {length}x{count}");
            }
        }
    }

    private static void Products<T>() where T : unmanaged, IFloatingPointIeee754<T>
    {
        // 行列積の左右と行列ベクトル積に並べたノードを置く。値はコアにブロードキャストのビューを渡した XMath.Multiply と同じビット。
        var random = new Random(61);
        const int inner = 7;
        foreach (var (length, count) in Shapes)
        {
            foreach (var arrangement in Arrangements)
            {
                var (rows, columns) = ShapeOf(length, count, arrangement);
                var v = Uniform<T>(random, length, (-1, 1));
                string context = $"{typeof(T).Name} {arrangement} {length}x{count}";
                var right = ArrayProductTests.RandomMatrix<T>(random, columns, inner);
                var leftFactor = ArrayProductTests.RandomMatrix<T>(random, inner, rows);
                AssertMatchesMaterialized(v, count, arrangement, right, (tape, b, x, _) => Reduce(tape, b * x), sameLoss: false, $"{context} B·X");
                AssertMatchesMaterialized(v, count, arrangement, leftFactor, (tape, b, x, _) => Reduce(tape, x * b), sameLoss: false, $"{context} X·B");

                // 行列ベクトル積。x の勾配は、閉包で取り出したノードで比べる。
                var vector = ArrayProductTests.RandomVector<T>(random, columns);
                var vectors = new List<(Tape Tape, VectorNode<T> Node)>();
                AssertMatchesMaterialized(v, count, arrangement, leftFactor, (tape, b, _, _) =>
                {
                    var x = tape.Leaf(vector);
                    vectors.Add((tape, x));
                    var weights = ArrayProductTests.RandomVector<T>(new Random(b.Rows), b.Rows);
                    return XMath.Dot(b * x, tape.Constant(weights));
                }, sameLoss: false, $"{context} B·x");
                AssertSameBits<T>(vectors[1].Tape.GetGradient(vectors[1].Node).ToArray(), vectors[0].Tape.GetGradient(vectors[0].Node).ToArray(), $"{context} B·x vector");

                var tape = new Tape();
                var node = ArrangeNode(tape.Leaf(v), count, arrangement);
                var span = ArrangeSpan<T>(v, count, arrangement);
                AssertSameBits<T>(XMath.Multiply(span, right).AsSpan(), (node * tape.Constant(right)).Value.ToMatrix().AsSpan(), $"{context} B·X value");
                AssertSameBits<T>(XMath.Multiply(leftFactor.AsMatrixSpan(), span).AsSpan(), (tape.Constant(leftFactor) * node).Value.ToMatrix().AsSpan(), $"{context} X·B value");
                AssertSameBits<T>(XMath.Multiply(span, vector.AsVectorSpan()).AsSpan(), (node * tape.Constant(vector)).Value.ToArray(), $"{context} B·x value");
            }
        }
    }

    private static void Outer<T>() where T : unmanaged, IFloatingPointIeee754<T>
    {
        var random = new Random(71);
        const int rows = 6, columns = 5;
        var column = ArrayProductTests.RandomVector<T>(random, rows);
        var row = ArrayProductTests.RandomVector<T>(random, columns);
        var tape = new Tape();
        var v = tape.Leaf(column);
        var w = tape.Leaf(row);
        var sum = v.Broadcast(columns) + w.BroadcastRows(rows);
        AssertSameBits<T>(XMath.Add(column.Broadcast(columns), row.BroadcastRows(rows)).AsSpan(), sum.Value.ToMatrix().AsSpan(), $"{typeof(T).Name} outer sum");
        tape.Backward(Reduce(tape, XMath.Logistic(sum)));

        var reference = new Tape();
        var a = reference.Leaf(column.Broadcast(columns));
        var b = reference.Leaf(row.BroadcastRows(rows));
        reference.Backward(Reduce(reference, XMath.Logistic(a + b)));
        AssertSameBits<T>(XMath.RowSums(reference.GetGradient(a)).AsSpan(), tape.GetGradient(v).ToArray(), $"{typeof(T).Name} outer v");
        AssertSameBits<T>(XMath.ColumnSums(reference.GetGradient(b)).AsSpan(), tape.GetGradient(w).ToArray(), $"{typeof(T).Name} outer w");

        // 同じベクトルを並べたもの同士の要素積 (v·1ᵀ)∘(v·1ᵀ)。左右の寄与をそれぞれ和にして足す。
        var square = new Tape();
        var u = square.Leaf(column);
        square.Backward(Reduce(square, XMath.Hadamard(u.Broadcast(columns), u.Broadcast(columns))));
        var materialized = new Tape();
        var p = materialized.Leaf(column.Broadcast(columns));
        var q = materialized.Leaf(column.Broadcast(columns));
        materialized.Backward(Reduce(materialized, XMath.Hadamard(p, q)));
        var expected = XMath.Add(XMath.RowSums(materialized.GetGradient(p)), XMath.RowSums(materialized.GetGradient(q)));
        AssertSameBits<T>(expected.AsSpan(), square.GetGradient(u).ToArray(), $"{typeof(T).Name} v∘v");
    }

    private static void SlicesAndTransposes<T>() where T : unmanaged, IFloatingPointIeee754<T>
    {
        // 並べたノードを切り出したものは、切り出したベクトルを並べたもの。転置は逆向きに並べたものの転置。
        var random = new Random(81);
        var data = ArrayProductTests.RandomVector<T>(random, 6);
        (Func<VectorNode<T>, MatrixNode<T>> Sliced, Func<VectorNode<T>, MatrixNode<T>> Equivalent, string Name)[] cases =
        [
            (v => v.Broadcast(5)[1..3, 2..4], v => v[1..3].Broadcast(2), "Broadcast[1..3, 2..4]"),
            (v => v.BroadcastRows(5)[1..3, 2..4], v => v[2..4].BroadcastRows(2), "BroadcastRows[1..3, 2..4]"),
            (v => v.Broadcast(5).T[1..3, 2..4], v => v[2..4].Broadcast(2).T, "Broadcast.T[1..3, 2..4]"),
            (v => v.BroadcastRows(5).T[1..3, 2..4], v => v[1..3].BroadcastRows(2).T, "BroadcastRows.T[1..3, 2..4]"),
            (v => v.Broadcast(5)[.., 2..3], v => v.Broadcast(1), "Broadcast[.., 2..3]"),
            (v => v.Broadcast(5)[.., 2..2], v => v.Broadcast(0), "Broadcast[.., 2..2]"),
            (v => v.Broadcast(5).T, v => v.BroadcastRows(5), "Broadcast.T"),
            (v => v.BroadcastRows(5).T, v => v.Broadcast(5), "BroadcastRows.T"),
            (v => v[1..5].Broadcast(3)[1..3, ..], v => v[2..4].Broadcast(3), "v[1..5].Broadcast[1..3, ..]"),
        ];
        foreach (var (sliced, equivalent, name) in cases)
        {
            string context = $"{typeof(T).Name} {name}";
            var tape = new Tape();
            var v = tape.Leaf(data);
            var node = sliced(v);
            tape.Backward(Reduce(tape, XMath.Exp(node)) + XMath.Dot(v, tape.Constant(data)));
            var reference = new Tape();
            var u = reference.Leaf(data);
            var same = equivalent(u);
            reference.Backward(Reduce(reference, XMath.Exp(same)) + XMath.Dot(u, reference.Constant(data)));
            Assert.Equal((same.Rows, same.Columns), (node.Rows, node.Columns));
            AssertSameBits<T>(same.Value.ToMatrix().AsSpan(), node.Value.ToMatrix().AsSpan(), $"{context} value");
            AssertSameBits<T>(reference.GetGradient(u).ToArray(), tape.GetGradient(v).ToArray(), $"{context} gradient");
            AssertSameBits<T>(reference.GetGradient(same).ToMatrix().AsSpan(), tape.GetGradient(node).ToMatrix().AsSpan(), $"{context} node gradient");
        }

        // コアのビューを同じように切り出した値と同じビット。
        var check = new Tape();
        var leaf = check.Leaf(data);
        AssertSameBits<T>(data.Broadcast(5)[1..3, 2..4].ToMatrix().AsSpan(), leaf.Broadcast(5)[1..3, 2..4].Value.ToMatrix().AsSpan(), "core slice");
        AssertSameBits<T>(data.BroadcastRows(5).T[1..3, 2..4].ToMatrix().AsSpan(), leaf.BroadcastRows(5).T[1..3, 2..4].Value.ToMatrix().AsSpan(), "core transposed slice");
    }

    private static void RepeatedUses<T>() where T : unmanaged, IFloatingPointIeee754<T>
    {
        // 逆伝播は記録の逆順に寄与を足す。最初の寄与（全体を覆う並べ方）は上書き、部分（v[1..4]）とほかは足し込み。
        var random = new Random(91);
        var data = ArrayProductTests.RandomVector<T>(random, 6);
        var c = ArrayProductTests.RandomVector<T>(random, 6);
        var tape = new Tape();
        var v = tape.Leaf(data);
        ScalarNode loss = Reduce(tape, XMath.Exp(v.Broadcast(3)));
        loss += XMath.Dot(v, tape.Constant(c));
        loss += Reduce(tape, XMath.Logistic(v[1..4].BroadcastRows(2)));
        loss += Reduce(tape, XMath.Square(v.Broadcast(4)));
        tape.Backward(loss);

        var exp = Materialized(data.Broadcast(3), y => XMath.Exp(y), rowsOfVector: true);
        var logistic = Materialized(data[1..4].BroadcastRows(2), y => XMath.Logistic(y), rowsOfVector: false);
        var square = Materialized(data.Broadcast(4), y => XMath.Square(y), rowsOfVector: true);
        var expected = new LinearAlgebra.Vector<T>(6);
        for (int i = 0; i < 6; i++)
        {
            T sum = square[i];
            if (i is >= 1 and < 4)
            {
                sum += logistic[i - 1];
            }

            expected[i] = sum + c[i] + exp[i];
        }

        AssertSameBits<T>(expected.AsSpan(), tape.GetGradient(v).ToArray(), $"{typeof(T).Name} repeated uses");

        static LinearAlgebra.Vector<T> Materialized(ReadOnlyMatrixSpan<T> view, Func<MatrixNode<T>, MatrixNode<T>> f, bool rowsOfVector)
        {
            var tape = new Tape();
            var m = tape.Leaf(view);
            tape.Backward(Reduce(tape, f(m)));
            return rowsOfVector ? XMath.RowSums(tape.GetGradient(m)) : XMath.ColumnSums(tape.GetGradient(m));
        }
    }

    // 中心差分と float の比較に使う層。バイアス（列ごと）・尺度（行ごと）・活性化関数・積を組み合わせる。v は長さ 10。
    private static ScalarNode Layer<T>(Tape tape, VectorNode<T> v, LayerConstants<T> constants) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var z = tape.Constant(constants.Weights) * tape.Constant(constants.Inputs) + v[..4].Broadcast(6);
        var h = XMath.Hadamard(XMath.Gelu(z), v[4..].BroadcastRows(4));
        var y = XMath.Silu(v[..4].Broadcast(3).T * h);
        ScalarNode loss = XMath.Dot(y * tape.Constant(constants.Mix), tape.Constant(constants.Target));
        loss += XMath.Dot(v[3..7].Broadcast(5) * tape.Constant(constants.Mix[..5].ToVector()), tape.Constant(constants.Mix[2..].ToVector()));
        return loss + Reduce(tape, XMath.Softplus(XMath.Divide(h, v[..4].Broadcast(6) + 3)));
    }

    private sealed class LayerConstants<T> where T : unmanaged, IFloatingPointIeee754<T>
    {
        public LayerConstants(Random random)
            : this(ArrayProductTests.RandomMatrix<T>(random, 4, 5), ArrayProductTests.RandomMatrix<T>(random, 5, 6),
                ArrayProductTests.RandomVector<T>(random, 6), ArrayProductTests.RandomVector<T>(random, 3))
        {
        }

        private LayerConstants(Matrix<T> weights, Matrix<T> inputs, LinearAlgebra.Vector<T> mix, LinearAlgebra.Vector<T> target)
        {
            Weights = weights;
            Inputs = inputs;
            Mix = mix;
            Target = target;
        }

        public Matrix<T> Weights { get; }

        public Matrix<T> Inputs { get; }

        public LinearAlgebra.Vector<T> Mix { get; }

        public LinearAlgebra.Vector<T> Target { get; }

        public LayerConstants<TOther> Convert<TOther>() where TOther : unmanaged, IFloatingPointIeee754<TOther>
            => new(Weights.ConvertChecked<TOther>(), Inputs.ConvertChecked<TOther>(), Mix.ConvertChecked<TOther>(), Target.ConvertChecked<TOther>());
    }
}
