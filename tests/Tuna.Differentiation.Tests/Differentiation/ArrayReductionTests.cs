using System.Numerics;
using Tuna.Differentiation;
using Tuna.LinearAlgebra;
using static Tuna.Tests.Differentiation.ElementwiseOperations;

namespace Tuna.Tests.Differentiation;

/// <summary>
/// 縮約（ベクトルからスカラー、行列の列・行ごと）とソフトマックス・対数ソフトマックスの配列ノード。
/// 値は XMath の同じビューと同じビット。勾配は、文書の式を XMath の要素演算と列・行ごとの和で求めたものと同じビット。
/// ln Σ exp の局所微分の極限はスカラーの LogSumExp と同じ。中心差分、float と double の一致、多項ロジスティック回帰の解析的な勾配。
/// </summary>
public class ArrayReductionTests
{
    // 行列の列・行ごとの縮約と、行列のソフトマックス。
    internal static readonly string[] MatrixNames =
    [
        "ColumnSums", "RowSums", "ColumnSumsOfSquares", "RowSumsOfSquares", "ColumnMeans", "RowMeans", "ColumnLogSumExp", "RowLogSumExp",
        "ColumnSoftmax", "RowSoftmax", "ColumnLogSoftmax", "RowLogSoftmax",
    ];

    // ベクトルからスカラーへの縮約と、ベクトルのソフトマックス。
    internal static readonly string[] VectorNames = ["SumOfSquares", "Mean", "LogSumExp", "Softmax", "LogSoftmax"];

    // スカラーの出力に掛ける係数（逆伝播の随伴になる）。2 の冪でない値にして、随伴の丸めと 2 倍を確かめる。
    private const double Coefficient = 0.3;

    public static TheoryData<string> MatrixCases() => new(MatrixNames);

    public static TheoryData<string> VectorCases() => new(VectorNames);

    internal static bool IsSoftmax(string name) => name.EndsWith("Softmax", StringComparison.Ordinal);

    internal static bool ByColumns(string name) => name.StartsWith("Column", StringComparison.Ordinal);

    internal static VectorNode<T> Reduce<T>(string name, MatrixNode<T> x) where T : unmanaged, IFloatingPointIeee754<T> => name switch
    {
        "ColumnSums" => XMath.ColumnSums(x),
        "RowSums" => XMath.RowSums(x),
        "ColumnSumsOfSquares" => XMath.ColumnSumsOfSquares(x),
        "RowSumsOfSquares" => XMath.RowSumsOfSquares(x),
        "ColumnMeans" => XMath.ColumnMeans(x),
        "RowMeans" => XMath.RowMeans(x),
        "ColumnLogSumExp" => XMath.ColumnLogSumExp(x),
        "RowLogSumExp" => XMath.RowLogSumExp(x),
        _ => throw new ArgumentOutOfRangeException(nameof(name)),
    };

    internal static MatrixNode<T> Softmax<T>(string name, MatrixNode<T> x) where T : unmanaged, IFloatingPointIeee754<T> => name switch
    {
        "ColumnSoftmax" => XMath.ColumnSoftmax(x),
        "RowSoftmax" => XMath.RowSoftmax(x),
        "ColumnLogSoftmax" => XMath.ColumnLogSoftmax(x),
        "RowLogSoftmax" => XMath.RowLogSoftmax(x),
        _ => throw new ArgumentOutOfRangeException(nameof(name)),
    };

    internal static ScalarNode Scalar<T>(string name, VectorNode<T> x) where T : unmanaged, IFloatingPointIeee754<T> => name switch
    {
        "SumOfSquares" => XMath.SumOfSquares(x),
        "Mean" => XMath.Mean(x),
        "LogSumExp" => XMath.LogSumExp(x),
        _ => throw new ArgumentOutOfRangeException(nameof(name)),
    };

    internal static VectorNode<T> Softmax<T>(string name, VectorNode<T> x) where T : unmanaged, IFloatingPointIeee754<T> => name switch
    {
        "Softmax" => XMath.Softmax(x),
        "LogSoftmax" => XMath.LogSoftmax(x),
        _ => throw new ArgumentOutOfRangeException(nameof(name)),
    };

    // 行列の演算の出力を損失に集約する。縮約の出力は形から決める重みとの内積（随伴は重みそのもの）、ソフトマックスの出力は ArrayBroadcastTests.Reduce。
    internal static ScalarNode Loss<T>(Tape tape, string name, MatrixNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
        => IsSoftmax(name) ? ArrayBroadcastTests.Reduce(tape, Softmax(name, x)) : Weighted(tape, Reduce(name, x));

    // ベクトルの演算の出力を損失に集約する。スカラーの出力は Coefficient 倍、ベクトルのソフトマックスは重みとの内積。
    internal static ScalarNode Loss<T>(Tape tape, string name, VectorNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
        => IsSoftmax(name) ? Weighted(tape, Softmax(name, x)) : Coefficient * Scalar(name, x);

    // 長さから決める重みとの内積。
    internal static ScalarNode Weighted<T>(Tape tape, VectorNode<T> s) where T : unmanaged, IFloatingPointIeee754<T>
        => XMath.Dot(s, tape.Constant(ArrayProductTests.RandomVector<T>(new Random(17 * s.Length + 3), s.Length)));

    [Theory]
    [MemberData(nameof(MatrixCases))]
    public void MatrixValuesMatchXMathBitForBit(string name)
    {
        MatrixValues<double>(name);
        MatrixValues<float>(name);
    }

    [Theory]
    [MemberData(nameof(VectorCases))]
    public void VectorValuesMatchXMathBitForBit(string name)
    {
        VectorValues<double>(name);
        VectorValues<float>(name);
    }

    [Theory]
    [MemberData(nameof(MatrixCases))]
    public void MatrixGradientsFollowTheDocumentedFormulasBitForBit(string name)
    {
        MatrixGradients<double>(name);
        MatrixGradients<float>(name);
    }

    [Theory]
    [MemberData(nameof(VectorCases))]
    public void VectorGradientsFollowTheDocumentedFormulasBitForBit(string name)
    {
        VectorGradients<double>(name);
        VectorGradients<float>(name);
    }

    [Fact]
    public void LogSumExpLimitsMatchTheScalarLogSumExp()
    {
        // 最大値が +∞・−∞・NaN の入力で、ベクトルおよび行・列ごとの LogSumExp 勾配がスカラー LogSumExp と一致することを検証する。
        double inf = double.PositiveInfinity, nan = double.NaN;
        double[][] cases =
        [
            [inf, 0, inf], [-inf, -inf], [0, nan, inf], [-inf, inf, 1, inf, inf], [nan, -inf], [1, nan, 2], [inf], [-inf],
            [double.MaxValue, double.MaxValue, 0, -inf], [-inf, 3, -inf],
        ];
        foreach (double[] values in cases)
        {
            var scalar = new Tape();
            var leaves = values.Select(scalar.Leaf).ToArray();
            var reference = XMath.LogSumExp(leaves);
            scalar.Backward(reference);
            double[] expected = leaves.Select(scalar.GetGradient).ToArray();
            string context = $"[{string.Join(", ", values)}]";
            LimitGradients<double>(values, reference.Value, expected, context);
            LimitGradients<float>(values, (float)reference.Value, expected.Select(e => (double)(float)e).ToArray(), context);
        }
    }

    [Fact]
    public void ZeroScalarAdjointsSkipTheRecord()
    {
        // 随伴がちょうど 0（float では丸めて 0）のスカラーの出力は、∞・NaN の入力でも寄与を足さない（総和・内積と同じ）。
        double[] values = [double.PositiveInfinity, double.NaN, 1];
        foreach (string name in new[] { "SumOfSquares", "Mean", "LogSumExp" })
        {
            var tape = new Tape();
            var x = tape.Leaf(new LinearAlgebra.Vector<double>(values));
            tape.Backward(0 * Scalar(name, x) + XMath.Sum(x));
            Assert.Equal([1.0, 1, 1], tape.GetGradient(x).ToArray());

            var single = new Tape();
            var y = single.Leaf(new LinearAlgebra.Vector<double>(values).ConvertChecked<float>());
            single.Backward(1e-50 * Scalar(name, y) + XMath.Sum(y));
            Assert.Equal([1f, 1, 1], single.GetGradient(y).ToArray());
        }
    }

    [Fact]
    public void ReachedReductionsFollowIeeeProducts()
    {
        // 届いた列・行ごとの縮約は、随伴が 0 の列・行でも IEEE 754 のとおりに掛ける（0·∞ は NaN、計画 3.6）。
        var data = new Matrix<double>(2, 2);
        data[0, 0] = double.PositiveInfinity;
        data[0, 1] = 1;
        data[1, 0] = 2;
        data[1, 1] = 3;
        var weights = new LinearAlgebra.Vector<double>([0, 1]);

        var tape = new Tape();
        var x = tape.Leaf(data);
        tape.Backward(XMath.Dot(XMath.ColumnSumsOfSquares(x), tape.Constant(weights)));
        var columns = tape.GetGradient(x);
        Assert.True(double.IsNaN(columns[0, 0]));
        Assert.Equal((0.0, 2.0, 6.0), (columns[1, 0], columns[0, 1], columns[1, 1]));

        tape.Reset();
        x = tape.Leaf(data);
        tape.Backward(XMath.Dot(XMath.RowSumsOfSquares(x), tape.Constant(weights)));
        var rows = tape.GetGradient(x);
        Assert.True(double.IsNaN(rows[0, 0]));
        Assert.Equal((0.0, 4.0, 6.0), (rows[0, 1], rows[1, 0], rows[1, 1]));
    }

    [Fact]
    public void UnreachedOutputsAreSkipped()
    {
        // 出力から届かないソフトマックス・縮約の記録は飛ばす。NaN の列のソフトマックス（値はすべて NaN）が、ほかの経路の勾配を壊さない。
        var data = new Matrix<double>(3, 2);
        data[0, 0] = double.NaN;
        data[1, 1] = double.PositiveInfinity;
        var tape = new Tape();
        var x = tape.Leaf(data);
        _ = XMath.ColumnSoftmax(x);
        _ = XMath.RowLogSoftmax(x);
        _ = XMath.ColumnLogSumExp(x);
        _ = XMath.RowSumsOfSquares(x);
        var sums = XMath.ColumnSums(x);
        _ = XMath.Softmax(sums);
        tape.Backward(XMath.Dot(sums, tape.Constant(new LinearAlgebra.Vector<double>([2, 3]))));
        Assert.Equal([2.0, 2, 2, 3, 3, 3], tape.GetGradient(x).ToMatrix().AsSpan().ToArray());
    }

    [Fact]
    public void EmptyInputsFollowTheCoreConventions()
    {
        // 行のない行列の列ごとの縮約は 0・NaN（平均）・−∞（ln Σ exp）、列のない行列の行ごとの縮約も同じ。空のベクトルも同じ。勾配は空。
        var tape = new Tape();
        var none = tape.Leaf(new Matrix<double>(0, 3));
        var wide = tape.Leaf(new Matrix<double>(3, 0));
        var empty = tape.Leaf(new LinearAlgebra.Vector<double>(0));
        double nan = double.NaN, inf = double.PositiveInfinity;
        Assert.Equal([0.0, 0, 0], XMath.ColumnSums(none).Value.ToArray());
        Assert.Equal([0.0, 0, 0], XMath.ColumnSumsOfSquares(none).Value.ToArray());
        Assert.Equal([nan, nan, nan], XMath.ColumnMeans(none).Value.ToArray());
        Assert.Equal([-inf, -inf, -inf], XMath.ColumnLogSumExp(none).Value.ToArray());
        Assert.Equal([0.0, 0, 0], XMath.RowSums(wide).Value.ToArray());
        Assert.Equal([0.0, 0, 0], XMath.RowSumsOfSquares(wide).Value.ToArray());
        Assert.Equal([nan, nan, nan], XMath.RowMeans(wide).Value.ToArray());
        Assert.Equal([-inf, -inf, -inf], XMath.RowLogSumExp(wide).Value.ToArray());
        Assert.Equal(0, XMath.RowSums(none).Length);
        Assert.Equal(0, XMath.ColumnMeans(wide).Length);
        Assert.Equal(0, XMath.Softmax(empty).Length);
        Assert.Equal(0, XMath.SumOfSquares(empty).Value);
        Assert.True(double.IsNaN(XMath.Mean(empty).Value));
        Assert.Equal(-inf, XMath.LogSumExp(empty).Value);

        // すべての演算を通した損失の逆伝播は、空の勾配のまま例外を投げない。
        ScalarNode loss = XMath.SumOfSquares(empty) + XMath.Mean(empty) + XMath.LogSumExp(empty) + Loss(tape, "Softmax", empty) + Loss(tape, "LogSoftmax", empty);
        foreach (string name in MatrixNames)
        {
            loss += Loss(tape, name, none) + Loss(tape, name, wide);
        }

        tape.Backward(loss);
        Assert.Equal((0, 3), (tape.GetGradient(none).Rows, tape.GetGradient(none).Columns));
        Assert.Equal((3, 0), (tape.GetGradient(wide).Rows, tape.GetGradient(wide).Columns));
        Assert.Equal(0, tape.GetGradient(empty).Length);
    }

    [Fact]
    public void ReductionModelGradientMatchesCentralDifferences()
    {
        // 行列の演算をすべて、葉の転置のビューに通す。ソフトマックスの出力にさらに縮約を通す合成も含める。
        var random = new Random(71);
        var point = Uniform<double>(random, 5, 4, -2, 2);
        ArrayActivationTests.AssertMatrixMatchesCentralDifferences(point, MatrixModel, 1e-7);

        // ベクトルの演算をすべて、葉の部分ベクトルに通す。
        var vector = new LinearAlgebra.Vector<double>(12);
        Uniform<double>(random, 12, 1, -2, 2).AsSpan().CopyTo(vector.AsSpan());
        ArrayElementwiseGradientTests.AssertMatchesCentralDifferences(vector, VectorModel, 1e-7);
    }

    [Fact]
    public void SingleGradientsAgreeWithDoubleGradients()
    {
        var random = new Random(72);
        var point = Uniform<double>(random, 5, 4, -2, 2);
        var tape = new Tape();
        var x = tape.Leaf(point);
        tape.Backward(MatrixModel(tape, x));
        var single = new Tape();
        var y = single.Leaf(point.ConvertChecked<float>());
        single.Backward(MatrixModel(single, y));
        AssertClose(tape.GetGradient(x).ToMatrix().AsSpan().ToArray(), single.GetGradient(y).ToMatrix().AsSpan().ToArray(), "matrix");

        var vector = Uniform<double>(random, 12, 1, -2, 2).AsMatrixSpan().Column(0).ToVector();
        tape.Reset();
        var v = tape.Leaf(vector);
        tape.Backward(VectorModel(tape, v));
        single.Reset();
        var w = single.Leaf(vector.ConvertChecked<float>());
        single.Backward(VectorModel(single, w));
        AssertClose(tape.GetGradient(v).ToArray(), single.GetGradient(w).ToArray(), "vector");

        static void AssertClose<T>(double[] expected, T[] actual, string context) where T : unmanaged, IFloatingPointIeee754<T>
        {
            double scale = expected.Max(Math.Abs);
            for (int i = 0; i < expected.Length; i++)
            {
                DifferentiationAssert.Close(expected[i], double.CreateTruncating(actual[i]), 1e-4, 1e-5 * scale, $"{context}[{i}]");
            }
        }
    }

    [Fact]
    public void MultinomialLogisticRegressionMatchesTheAnalyticGradient()
    {
        // K 種類・D 次元・N 件。スコア Z = W·X + b·1ᵀ（列が標本）、損失 E = −Σ T ∘ ln softmax(Z)。解析的な勾配は (P − T)·Xᵀ と (P − T)·1。
        // 同じ損失を ln Σ exp と列ごとの和（Σ lse(Z) − Σ T ∘ Z）でも、標本を行に置いた行ごとの版（Zᵀ = Xᵀ·Wᵀ + 1·bᵀ）でも書く。
        const int classes = 4, features = 3, samples = 50;
        var random = new Random(73);
        var weight = Uniform<double>(random, classes, features, -1, 1);
        var bias = Uniform<double>(random, classes, 1, -1, 1).AsMatrixSpan().Column(0).ToVector();
        var data = Uniform<double>(random, features, samples, -3, 3);
        var targets = new Matrix<double>(classes, samples);
        for (int n = 0; n < samples; n++)
        {
            targets[random.Next(classes), n] = 1;
        }

        // 解析的な値と勾配。
        var scores = new Matrix<double>(classes, samples);
        double expectedLoss = 0;
        var residual = new Matrix<double>(classes, samples);
        for (int n = 0; n < samples; n++)
        {
            double max = double.NegativeInfinity;
            for (int k = 0; k < classes; k++)
            {
                double z = bias[k];
                for (int d = 0; d < features; d++)
                {
                    z += weight[k, d] * data[d, n];
                }

                scores[k, n] = z;
                max = Math.Max(max, z);
            }

            double total = 0;
            for (int k = 0; k < classes; k++)
            {
                total += Math.Exp(scores[k, n] - max);
            }

            double lse = max + Math.Log(total);
            for (int k = 0; k < classes; k++)
            {
                residual[k, n] = Math.Exp(scores[k, n] - lse) - targets[k, n];
                expectedLoss -= targets[k, n] * (scores[k, n] - lse);
            }
        }

        var expectedWeight = new Matrix<double>(classes, features);
        var expectedBias = new double[classes];
        for (int k = 0; k < classes; k++)
        {
            for (int n = 0; n < samples; n++)
            {
                expectedBias[k] += residual[k, n];
                for (int d = 0; d < features; d++)
                {
                    expectedWeight[k, d] += residual[k, n] * data[d, n];
                }
            }
        }

        for (int form = 0; form < 3; form++)
        {
            var tape = new Tape();
            var w = tape.Leaf(weight);
            var b = tape.Leaf(bias);
            ScalarNode loss;
            if (form < 2)
            {
                var z = w * tape.Constant(data) + b.Broadcast(samples);
                var t = tape.Constant(targets);
                loss = form == 0
                    ? -XMath.Sum(XMath.ColumnSums(XMath.Hadamard(t, XMath.ColumnLogSoftmax(z))))
                    : XMath.Sum(XMath.ColumnLogSumExp(z)) - XMath.Sum(XMath.ColumnSums(XMath.Hadamard(t, z)));
            }
            else
            {
                var z = tape.Constant(data).T * w.T + b.BroadcastRows(samples);
                loss = -XMath.Sum(XMath.RowSums(XMath.Hadamard(tape.Constant(targets).T, XMath.RowLogSoftmax(z))));
            }

            tape.Backward(loss);
            string context = $"form {form}";
            DifferentiationAssert.Close(expectedLoss, loss.Value, 1e-13, 1e-13, $"{context} loss");
            var gradient = tape.GetGradient(w);
            for (int k = 0; k < classes; k++)
            {
                DifferentiationAssert.Close(expectedBias[k], tape.GetGradient(b)[k], 1e-12, 1e-13, $"{context} b[{k}]");
                for (int d = 0; d < features; d++)
                {
                    DifferentiationAssert.Close(expectedWeight[k, d], gradient[k, d], 1e-12, 1e-13, $"{context} W[{k}, {d}]");
                }
            }
        }
    }

    [Fact]
    public void StaleNodesAndRecordingAfterBackwardThrow()
    {
        var tape = new Tape();
        var x = tape.Leaf(new Matrix<double>(2, 3));
        var v = tape.Leaf(new LinearAlgebra.Vector<double>(3));
        tape.Reset();
        foreach (string name in MatrixNames)
        {
            Assert.Throws<InvalidOperationException>(() => Loss(tape, name, x));
        }

        foreach (string name in VectorNames)
        {
            Assert.Throws<InvalidOperationException>(() => Loss(tape, name, v));
        }

        // 逆伝播の後は、記録する演算は例外、定数だけの演算は値を返す。
        var data = Uniform<double>(new Random(74), 2, 3, -1, 1);
        var y = tape.Leaf(data);
        var w = tape.Leaf(new LinearAlgebra.Vector<double>([1.0, 2, 3]));
        var c = tape.Constant(data);
        var d = tape.Constant(new LinearAlgebra.Vector<double>([1.0, 2, 3]));
        tape.Backward(XMath.Sum(w));
        foreach (string name in MatrixNames)
        {
            Assert.Throws<InvalidOperationException>(() => IsSoftmax(name) ? Softmax(name, y).Rows : Reduce(name, y).Length);
            Assert.False(IsSoftmax(name) ? Softmax(name, c).RequiresGradient : Reduce(name, c).RequiresGradient);
        }

        foreach (string name in VectorNames)
        {
            Assert.Throws<InvalidOperationException>(() => IsSoftmax(name) ? Softmax(name, w).Length : Scalar(name, w).Value);
            Assert.False(IsSoftmax(name) ? Softmax(name, d).RequiresGradient : Scalar(name, d).RequiresGradient);
        }

        Assert.Equal(XMath.LogSumExp<double>(new LinearAlgebra.Vector<double>([1.0, 2, 3])), XMath.LogSumExp(d).Value);
    }

    // 行列の演算をすべて通すモデル。点は 5 × 4 で、演算には転置のビュー（4 × 5）を渡す。
    private static ScalarNode MatrixModel<T>(Tape tape, MatrixNode<T> m) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var x = m.T;
        ScalarNode loss = 0;
        foreach (string name in MatrixNames)
        {
            loss += Loss(tape, name, x);
        }

        // 合成: 対数ソフトマックスの列ごとの ln Σ exp（値は 0 に近い）と、ソフトマックスの行ごとの2乗和。
        loss += Weighted(tape, XMath.ColumnLogSumExp(XMath.ColumnLogSoftmax(x) * 2)) + Weighted(tape, XMath.RowSumsOfSquares(XMath.RowSoftmax(m)));
        return loss;
    }

    // ベクトルの演算をすべて通すモデル。部分ベクトルとソフトマックスの出力への縮約も含める。
    private static ScalarNode VectorModel<T>(Tape tape, VectorNode<T> v) where T : unmanaged, IFloatingPointIeee754<T>
    {
        ScalarNode loss = 0;
        foreach (string name in VectorNames)
        {
            loss += Loss(tape, name, v) + Loss(tape, name, v[2..9]);
        }

        return loss + XMath.LogSumExp(XMath.LogSoftmax(v) * 3) + XMath.SumOfSquares(XMath.Softmax(v[1..]));
    }

    private static void MatrixValues<T>(string name) where T : unmanaged, IFloatingPointIeee754<T>
    {
        // 特別な値（±0・非正規化数・裾・±∞・NaN）を含む入力と、有限の乱数だけの入力。
        foreach (double[] inputs in new[] { Inputs(81, 30, 30), Uniform<double>(new Random(82), 63, 1, -30, 30).AsSpan().ToArray() })
        {
            var data = new LinearAlgebra.Vector<double>(inputs).ConvertChecked<T>();
            var storage = new Matrix<T>(9, 7);
            for (int k = 0; k < storage.Rows * storage.Columns; k++)
            {
                storage.AsSpan()[k] = data[k % data.Length];
            }

            var tape = new Tape();
            var leaf = tape.Leaf(storage);
            var vector = tape.Leaf(data);
            string context = $"{name} {typeof(T).Name}";
            MatrixValue(name, storage, leaf, $"{context} leaf");
            MatrixValue(name, storage, tape.Constant(storage), $"{context} constant");
            MatrixValue(name, storage.AsMatrixSpan()[1..8, 2..6].T, leaf[1..8, 2..6].T, $"{context} view");
            MatrixValue(name, data.AsVectorSpan()[..5].BroadcastRows(3), vector[..5].BroadcastRows(3), $"{context} rows");
            MatrixValue(name, data.AsVectorSpan()[3..9].Broadcast(4), vector[3..9].Broadcast(4), $"{context} columns");
        }
    }

    private static void MatrixValue<T>(string name, ReadOnlyMatrixSpan<T> expected, MatrixNode<T> x, string context) where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (IsSoftmax(name))
        {
            var result = new Matrix<T>(expected.Rows, expected.Columns);
            MatrixSpan<T> d = result;
            switch (name)
            {
                case "ColumnSoftmax": XMath.ColumnSoftmax(expected, d); break;
                case "RowSoftmax": XMath.RowSoftmax(expected, d); break;
                case "ColumnLogSoftmax": XMath.ColumnLogSoftmax(expected, d); break;
                default: XMath.RowLogSoftmax(expected, d); break;
            }

            AssertSameBits<T>(result.AsSpan(), Softmax(name, x).Value.ToMatrix().AsSpan(), context);
            return;
        }

        var sums = new LinearAlgebra.Vector<T>(ByColumns(name) ? expected.Columns : expected.Rows);
        VectorSpan<T> s = sums;
        switch (name)
        {
            case "ColumnSums": XMath.ColumnSums(expected, s); break;
            case "RowSums": XMath.RowSums(expected, s); break;
            case "ColumnSumsOfSquares": XMath.ColumnSumsOfSquares(expected, s); break;
            case "RowSumsOfSquares": XMath.RowSumsOfSquares(expected, s); break;
            case "ColumnMeans": XMath.ColumnMeans(expected, s); break;
            case "RowMeans": XMath.RowMeans(expected, s); break;
            case "ColumnLogSumExp": XMath.ColumnLogSumExp(expected, s); break;
            default: XMath.RowLogSumExp(expected, s); break;
        }

        AssertSameBits<T>(sums.AsSpan(), Reduce(name, x).Value.ToArray(), context);
    }

    private static void VectorValues<T>(string name) where T : unmanaged, IFloatingPointIeee754<T>
    {
        foreach (double[] inputs in new[] { Inputs(83, 30, 30), Uniform<double>(new Random(84), 63, 1, -30, 30).AsSpan().ToArray(), [-0.0], [] })
        {
            var data = new LinearAlgebra.Vector<double>(inputs).ConvertChecked<T>();
            var tape = new Tape();
            var leaf = tape.Leaf(data);
            string context = $"{name} {typeof(T).Name} length {inputs.Length}";
            VectorValue(name, data, leaf, $"{context} leaf");
            VectorValue(name, data, tape.Constant(data), $"{context} constant");
            if (inputs.Length > 10)
            {
                VectorValue(name, data.AsVectorSpan()[3..^4], leaf[3..^4], $"{context} slice");
            }
        }
    }

    private static void VectorValue<T>(string name, ReadOnlyVectorSpan<T> expected, VectorNode<T> x, string context) where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (IsSoftmax(name))
        {
            var result = new LinearAlgebra.Vector<T>(expected.Length);
            if (name == "Softmax")
            {
                XMath.Softmax(expected, result);
            }
            else
            {
                XMath.LogSoftmax(expected, result);
            }

            AssertSameBits<T>(result.AsSpan(), Softmax(name, x).Value.ToArray(), context);
            return;
        }

        T value = name switch
        {
            "SumOfSquares" => XMath.SumOfSquares(expected),
            "Mean" => XMath.Mean(expected),
            _ => XMath.LogSumExp(expected),
        };
        AssertSameBits<double>([double.CreateTruncating(value)], [Scalar(name, x).Value], context);
    }

    // 3通りの置き方で勾配を文書の式と比べる。0: 葉そのもの（勾配へ直接書く）、1: 大きい葉の転置の部分行列、2: 同じ葉に2回使う（2回目の寄与を足し込む）。
    private static void MatrixGradients<T>(string name) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var storage = Uniform<T>(new Random(200 + Array.IndexOf(MatrixNames, name)), 9, 7, -4, 4);
        for (int placement = 0; placement < 3; placement++)
        {
            var tape = new Tape();
            var leaf = tape.Leaf(storage);
            bool view = placement == 1;
            var x = view ? leaf[1..8, 2..6].T : leaf;
            ReadOnlyMatrixSpan<T> values = view ? storage.AsMatrixSpan()[1..8, 2..6].T : storage;
            Matrix<T> contribution;
            if (IsSoftmax(name))
            {
                var y = Softmax(name, x);
                ScalarNode loss = ArrayBroadcastTests.Reduce(tape, y);
                tape.Backward(placement == 2 ? loss + ArrayBroadcastTests.Reduce(tape, Softmax(name, x)) : loss);
                contribution = SoftmaxContribution(name, y.Value, tape.GetGradient(y));
            }
            else
            {
                var s = Reduce(name, x);
                ScalarNode loss = Weighted(tape, s);
                tape.Backward(placement == 2 ? loss + Weighted(tape, Reduce(name, x)) : loss);
                contribution = ReductionContribution(name, values, tape.GetGradient(s));
            }

            string context = $"{name} {typeof(T).Name} placement {placement}";
            var expected = placement == 2 ? Twice(contribution.AsSpan()) : contribution.AsSpan().ToArray();
            AssertSameBits<T>(expected, tape.GetGradient(x).ToMatrix().AsSpan(), context);
            if (view)
            {
                // ビューの外の葉の勾配は 0。
                var full = new Matrix<T>(9, 7);
                for (int j = 0; j < contribution.Columns; j++)
                {
                    for (int i = 0; i < contribution.Rows; i++)
                    {
                        full[j + 1, i + 2] = contribution[i, j];
                    }
                }

                AssertSameBits<T>(full.AsSpan(), tape.GetGradient(leaf).ToMatrix().AsSpan(), $"{context} leaf");
            }
        }
    }

    // 縮約の寄与。d は出力の勾配（列・行ごとの値）。
    private static Matrix<T> ReductionContribution<T>(string name, ReadOnlyMatrixSpan<T> x, ReadOnlyVectorSpan<T> d) where T : unmanaged, IFloatingPointIeee754<T>
    {
        bool columns = ByColumns(name);
        var result = new Matrix<T>(x.Rows, x.Columns);
        var weights = new Matrix<T>(x.Rows, x.Columns);
        if (columns)
        {
            XMath.ColumnSoftmax(x, weights);
        }
        else
        {
            XMath.RowSoftmax(x, weights);
        }

        for (int j = 0; j < x.Columns; j++)
        {
            for (int i = 0; i < x.Rows; i++)
            {
                T di = d[columns ? j : i];
                result[i, j] = name switch
                {
                    "ColumnSums" or "RowSums" => di,
                    "ColumnSumsOfSquares" or "RowSumsOfSquares" => x[i, j] * (di + di),
                    "ColumnMeans" or "RowMeans" => T.CreateTruncating(double.CreateTruncating(di) / (columns ? x.Rows : x.Columns)),
                    _ => weights[i, j] * di,
                };
            }
        }

        return result;
    }

    // ソフトマックスの寄与。y は出力の値、dy は出力の勾配。ソフトマックスは y ∘ (dy − Σ y ∘ dy)、対数ソフトマックスは dy − exp(y) ∘ Σ dy。
    private static Matrix<T> SoftmaxContribution<T>(string name, ReadOnlyMatrixSpan<T> y, ReadOnlyMatrixSpan<T> dy) where T : unmanaged, IFloatingPointIeee754<T>
    {
        bool columns = ByColumns(name) || name is "Softmax" or "LogSoftmax";
        bool log = name.EndsWith("LogSoftmax", StringComparison.Ordinal);
        var result = new Matrix<T>(y.Rows, y.Columns);
        var operand = new Matrix<T>(y.Rows, y.Columns);
        for (int j = 0; j < y.Columns; j++)
        {
            for (int i = 0; i < y.Rows; i++)
            {
                operand[i, j] = log ? dy[i, j] : y[i, j] * dy[i, j];
            }
        }

        var sums = new LinearAlgebra.Vector<T>(columns ? y.Columns : y.Rows);
        if (columns)
        {
            XMath.ColumnSums(operand, sums);
        }
        else
        {
            XMath.RowSums(operand, sums);
        }

        var exp = new Matrix<T>(y.Rows, y.Columns);
        XMath.Exp(y, exp);
        for (int j = 0; j < y.Columns; j++)
        {
            for (int i = 0; i < y.Rows; i++)
            {
                T s = sums[columns ? j : i];
                result[i, j] = log ? dy[i, j] - exp[i, j] * s : y[i, j] * (dy[i, j] - s);
            }
        }

        return result;
    }

    // 3通りの置き方（葉そのもの、葉の部分ベクトル、同じ葉に2回）で勾配を文書の式と比べる。
    private static void VectorGradients<T>(string name) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var data = Uniform<T>(new Random(300 + Array.IndexOf(VectorNames, name)), 23, 1, -4, 4).AsMatrixSpan().Column(0).ToVector();
        for (int placement = 0; placement < 3; placement++)
        {
            var tape = new Tape();
            var leaf = tape.Leaf(data);
            var x = placement == 1 ? leaf[3..20] : leaf;
            ReadOnlyVectorSpan<T> values = placement == 1 ? data.AsVectorSpan()[3..20] : data;
            T[] contribution;
            if (IsSoftmax(name))
            {
                var y = Softmax(name, x);
                ScalarNode loss = Weighted(tape, y);
                tape.Backward(placement == 2 ? loss + Weighted(tape, Softmax(name, x)) : loss);
                contribution = SoftmaxContribution(name, y.Value.Broadcast(1), tape.GetGradient(y).Broadcast(1)).AsSpan().ToArray();
            }
            else
            {
                ScalarNode loss = Coefficient * Scalar(name, x);
                tape.Backward(placement == 2 ? loss + Coefficient * Scalar(name, x) : loss);
                T g = T.CreateTruncating(Coefficient);
                var weights = new LinearAlgebra.Vector<T>(values.Length);
                XMath.Softmax(values, weights);
                contribution = new T[values.Length];
                for (int i = 0; i < values.Length; i++)
                {
                    contribution[i] = name switch
                    {
                        "SumOfSquares" => (g + g) * values[i],
                        "Mean" => T.CreateTruncating(double.CreateTruncating(g) / values.Length),
                        _ => g * weights[i],
                    };
                }
            }

            string context = $"{name} {typeof(T).Name} placement {placement}";
            var expected = placement == 2 ? Twice<T>(contribution) : contribution;
            AssertSameBits<T>(expected, tape.GetGradient(x).ToArray(), context);
            if (placement == 1)
            {
                var full = new T[data.Length];
                contribution.CopyTo(full, 3);
                AssertSameBits<T>(full, tape.GetGradient(leaf).ToArray(), $"{context} leaf");
            }
        }
    }

    // 極限の入力をベクトル、行列の第1列（列ごと）、行列の第1行（行ごと）に置き、ln Σ exp の値と勾配を比べる。ほかの列・行は有限の乱数。
    private static void LimitGradients<T>(double[] values, double expectedValue, double[] expected, string context) where T : unmanaged, IFloatingPointIeee754<T>
    {
        int n = values.Length;
        var data = new LinearAlgebra.Vector<double>(values).ConvertChecked<T>();
        context = $"{context} {typeof(T).Name}";
        var tape = new Tape();
        var x = tape.Leaf(data);
        var s = XMath.LogSumExp(x);
        tape.Backward(s);
        DifferentiationAssert.SameNumber(expectedValue, s.Value, $"{context} value");
        Compare(tape.GetGradient(x).ToArray(), "vector");

        var matrix = Uniform<T>(new Random(n), n, 3, -2, 2);
        for (int i = 0; i < n; i++)
        {
            matrix[i, 1] = data[i];
        }

        var weights = new LinearAlgebra.Vector<T>([T.CreateTruncating(0.5), T.One, T.CreateTruncating(0.25)]);
        tape.Reset();
        var columns = tape.Leaf(matrix);
        var lse = XMath.ColumnLogSumExp(columns);
        tape.Backward(XMath.Dot(lse, tape.Constant(weights)));
        DifferentiationAssert.SameNumber(expectedValue, double.CreateTruncating(lse.Value[1]), $"{context} column value");
        Compare(tape.GetGradient(columns).Column(1).ToArray(), "column");

        tape.Reset();
        var rows = tape.Leaf(matrix.AsMatrixSpan().T);
        lse = XMath.RowLogSumExp(rows);
        tape.Backward(XMath.Dot(lse, tape.Constant(weights)));
        DifferentiationAssert.SameNumber(expectedValue, double.CreateTruncating(lse.Value[1]), $"{context} row value");
        Compare(tape.GetGradient(rows).Row(1).ToArray(), "row");

        void Compare(T[] actual, string kind)
        {
            for (int i = 0; i < n; i++)
            {
                DifferentiationAssert.SameNumber(expected[i], double.CreateTruncating(actual[i]), $"{context} {kind}[{i}]");
            }
        }
    }

    private static T[] Twice<T>(ReadOnlySpan<T> values) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var result = new T[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            result[i] = values[i] + values[i];
        }

        return result;
    }

    private static Matrix<T> Uniform<T>(Random random, int rows, int columns, double low, double high) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var result = new Matrix<T>(rows, columns);
        var span = result.AsSpan();
        for (int i = 0; i < span.Length; i++)
        {
            span[i] = T.CreateTruncating(low + (high - low) * random.NextDouble());
        }

        return result;
    }
}
