using System.Numerics;
using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.NeuralNetwork.Tests.Training;

/// <summary>NN の層の記録（<see cref="XMathDifferentiationExtensions"/>）の値と勾配</summary>
public class XMathDifferentiationExtensionsTests
{
    public static TheoryData<string> Operations
    {
        get
        {
            var data = new TheoryData<string>();
            foreach (var reference in GradientReferences.Cases)
            {
                data.Add(reference.Operation);
            }

            return data;
        }
    }

    [Theory]
    [MemberData(nameof(Operations))]
    public void DoubleGradientsMatchTheReferences(string operation) => MatchesReference<double>(operation, 1e-14);

    [Theory]
    [MemberData(nameof(Operations))]
    public void SingleGradientsMatchTheReferences(string operation) => MatchesReference<float>(operation, 2e-6);

    // 入力は float でも厳密に表せる値なので、float と double で同じ参照値と比べる。誤差は最大値（1 未満なら 1）で正規化する。
    private static void MatchesReference<T>(string operation, double tolerance) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var reference = GradientReferences.Cases.Single(c => c.Operation == operation);
        var tape = new Tape();
        var leaves = new Leaves<T>(tape, reference.Inputs.Select(a => a.Values).ToArray(), reference.Inputs.Select(a => (a.Rows, a.Columns)).ToArray(),
            operation.StartsWith("Linear", StringComparison.Ordinal) ? 2 : 1);
        var y = Apply(operation, T.CreateTruncating(reference.Epsilon), leaves);
        Close(reference.Output.Values, ToArray(y.Value), tolerance, "output");
        tape.Backward(Loss(tape, y, reference.OutputGradient.Values));
        for (int i = 0; i < reference.Gradients.Length; i++)
        {
            Close(reference.Gradients[i].Values, leaves.Gradient(tape, i), tolerance, $"gradient {i}");
        }
    }

    [Theory]
    [MemberData(nameof(Operations))]
    public void GradientsMatchCentralDifferences(string operation)
    {
        // 参照値とは別の乱数の入力と大きさ（列ごとに平均と尺度を変え、PReLU の折れ目から離す）で、推論版の XMath の中心差分と比べる。
        var reference = GradientReferences.Cases.Single(c => c.Operation == operation);
        var (inputs, shapes, matrices) = RandomInputs(operation, 13, 6, 9, seed: 31);
        var g = TestData.Uniform(shapes[0].Rows * (operation.StartsWith("Linear", StringComparison.Ordinal) ? shapes[1].Columns : shapes[0].Columns), 32);
        var tape = new Tape();
        var leaves = new Leaves<double>(tape, inputs, shapes, matrices);
        var y = Apply(operation, reference.Epsilon, leaves);
        tape.Backward(Loss(tape, y, g));
        for (int i = 0; i < inputs.Length; i++)
        {
            var actual = leaves.Gradient(tape, i);
            var expected = new double[inputs[i].Length];
            for (int k = 0; k < expected.Length; k++)
            {
                double saved = inputs[i][k], h = 1e-6 * Math.Max(1, Math.Abs(saved));
                inputs[i][k] = saved + h;
                double plus = Evaluate(operation, reference.Epsilon, inputs, shapes, matrices, g);
                inputs[i][k] = saved - h;
                double minus = Evaluate(operation, reference.Epsilon, inputs, shapes, matrices, g);
                inputs[i][k] = saved;
                expected[k] = (plus - minus) / (2 * h);
            }

            Close(expected, actual, 1e-7, $"gradient {i}");
        }
    }

    [Theory]
    [MemberData(nameof(Operations))]
    public void ValuesAreBitIdenticalToInference(string operation)
    {
        ValuesAreBitIdentical<double>(operation);
        ValuesAreBitIdentical<float>(operation);
    }

    // 記録つきの値は、勾配を要する入力（統計量を保存する）、定数の入力、転置・部分行列のビューのどれでも推論版の XMath と同じビット。
    // ビューの入力の勾配は、連続な入力の勾配と同じ値（作業領域に求めてから 0 に加えるので、符号つきのゼロだけが違いうる）。
    private static void ValuesAreBitIdentical<T>(string operation) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var reference = GradientReferences.Cases.Single(c => c.Operation == operation);
        var (inputs, shapes, matrices) = RandomInputs(operation, 37, 23, 19, seed: 41);
        T epsilon = T.CreateTruncating(reference.Epsilon);
        var expected = Evaluate<T>(operation, epsilon, inputs, shapes, matrices);
        bool linear = operation.StartsWith("Linear", StringComparison.Ordinal);
        var g = TestData.Uniform(expected.Rows * expected.Columns, 42);

        var tape = new Tape();
        var leaves = new Leaves<T>(tape, inputs, shapes, matrices);
        var y = Apply(operation, epsilon, leaves);
        TestData.Same(expected, y.Value);
        tape.Backward(Loss(tape, y, g));

        var constants = new Tape();
        TestData.Same(expected, Apply(operation, epsilon, new Leaves<T>(constants, inputs, shapes, matrices, constant: true)).Value);

        var views = new Tape();
        var viewLeaves = new Leaves<T>(views, inputs, shapes, matrices, views: true);
        var z = Apply(operation, epsilon, viewLeaves);
        TestData.Same(expected, z.Value);
        views.Backward(Loss(views, z, g));
        for (int i = 0; i < inputs.Length; i++)
        {
            var contiguous = leaves.Gradient(tape, i);
            var strided = viewLeaves.Gradient(views, i);
            if (linear && i < 2)
            {
                // 転置のビューへの積は転置した積として求めるので、加算の順序が変わりうる。
                Close(contiguous, strided, typeof(T) == typeof(float) ? 1e-6 : 1e-14, $"gradient {i}");
            }
            else
            {
                Assert.Equal(contiguous, strided);
            }
        }
    }

    [Theory]
    [InlineData(3, 5, 7)]
    [InlineData(64, 48, 300)]
    public void LinearMatchesTheProductComposition(int outputs, int inner, int positions)
    {
        var (inputs, shapes, matrices) = RandomInputs("Linear", outputs, inner, positions, seed: outputs);
        var g = TestData.Uniform(outputs * positions, 51);
        var tape = new Tape();
        var leaves = new Leaves<float>(tape, inputs, shapes, matrices);
        var y = XMath.Linear(leaves.Matrices[0], leaves.Matrices[1], leaves.Vectors[0]);
        tape.Backward(Loss(tape, y, g));

        var reference = new Tape();
        var composed = new Leaves<float>(reference, inputs, shapes, matrices);
        var z = composed.Matrices[0] * composed.Matrices[1] + composed.Vectors[0].Broadcast(positions);
        reference.Backward(Loss(reference, z, g));
        TestData.Same(z.Value, y.Value);
        for (int i = 0; i < inputs.Length; i++)
        {
            Assert.Equal(composed.Gradient(reference, i), leaves.Gradient(tape, i));
        }
    }

    [Fact]
    public void PreluSkipsZeroLocalDerivatives()
    {
        // 傾き 0 の負の入力では、上流の勾配が無限大・NaN でも入力の勾配に寄与しない（3.6）。SIMD 経路と転置ビュー・共有傾きも含める。
        CheckPreluZeroDerivative<double>();
        CheckPreluZeroDerivative<float>();
    }

    private static void CheckPreluZeroDerivative<T>() where T : unmanaged, IFloatingPointIeee754<T>
    {
        const int rows = 18;
        const int columns = 3;
        var xData = new T[rows * columns];
        var dyData = new T[rows * columns];
        var slopeData = new T[rows];
        for (int i = 0; i < rows; i++)
        {
            slopeData[i] = i % 3 == 0 ? T.Zero : (i % 3 == 1 ? T.NegativeZero : T.CreateTruncating(0.5));
            for (int j = 0; j < columns; j++)
            {
                int idx = j * rows + i;
                xData[idx] = (i + j) % 2 == 0 ? T.CreateTruncating(-(i + 1)) : T.CreateTruncating(i + 1);
                dyData[idx] = i % 3 == 0
                    ? T.PositiveInfinity
                    : (i % 3 == 1 ? T.NaN : T.CreateTruncating(j + 1));
            }
        }

        // 連続入力（SIMD 経路）と転置ビューの両方を検証
        foreach (bool transposed in new[] { false, true })
        {
            var tape = new Tape();
            var xMatrix = Matrix<T>.FromColumnMajor(rows, columns, xData);
            var x = transposed ? tape.Leaf(xMatrix.T.ToMatrix()).T : tape.Leaf(xMatrix);
            var slopes = tape.Leaf(new LinearAlgebra.Vector<T>(slopeData));
            var y = XMath.Prelu(x, slopes);
            var dy = tape.Constant(Matrix<T>.FromColumnMajor(rows, columns, dyData));
            tape.Backward(XMath.Sum(XMath.ColumnSums(XMath.Hadamard(y, dy))));

            var dx = tape.GetGradient(x);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    int idx = j * rows + i;
                    if (xData[idx] < T.Zero && slopeData[i] == T.Zero)
                    {
                        Assert.Equal(T.Zero, dx[i, j]);
                    }
                    else if (xData[idx] >= T.Zero)
                    {
                        Assert.Equal(dyData[idx], dx[i, j]);
                    }
                    else
                    {
                        Assert.Equal(slopeData[i] * dyData[idx], dx[i, j]);
                    }
                }
            }
        }

        // 共有傾き（長さ 1 で 0.0）
        {
            var tape = new Tape();
            var x = tape.Leaf(Matrix<T>.FromColumnMajor(rows, 1, xData.AsSpan(0, rows).ToArray()));
            var shared = tape.Leaf(new LinearAlgebra.Vector<T>([T.Zero]));
            var y = XMath.Prelu(x, shared);
            var dy = tape.Constant(Matrix<T>.FromColumnMajor(rows, 1, dyData.AsSpan(0, rows).ToArray()));
            tape.Backward(XMath.Sum(XMath.ColumnSums(XMath.Hadamard(y, dy))));
            var dx = tape.GetGradient(x);
            for (int i = 0; i < rows; i++)
            {
                if (xData[i] < T.Zero)
                {
                    Assert.Equal(T.Zero, dx[i, 0]);
                }
                else
                {
                    Assert.Equal(dyData[i], dx[i, 0]);
                }
            }
        }
    }

    [Fact]
    public void InvalidArgumentsAreRejected()
    {
        var tape = new Tape();
        var other = new Tape();
        var x = tape.Leaf(new Matrix<double>(3, 4));
        var w = tape.Leaf(new Matrix<double>(2, 5));
        var validW = tape.Leaf(new Matrix<double>(2, 3));
        var v3 = tape.Leaf(new LinearAlgebra.Vector<double>(3));
        var v2 = tape.Leaf(new LinearAlgebra.Vector<double>(2));

        // 形状・パラメーター異常
        Assert.Throws<ArgumentException>(() => XMath.Linear(w, x));
        Assert.Throws<ArgumentException>(() => XMath.Linear(validW, x, v3));
        Assert.Throws<ArgumentException>(() => XMath.LayerNorm(x, v2, v3, 1e-5));
        Assert.Throws<ArgumentException>(() => XMath.LayerNorm(x, v3, v2, 1e-5));
        Assert.Throws<ArgumentException>(() => XMath.RmsNorm(x, tape.Leaf(new LinearAlgebra.Vector<double>(4)), 1e-5));
        Assert.Throws<ArgumentOutOfRangeException>(() => XMath.LayerNorm(x, -1.0));
        Assert.Throws<ArgumentOutOfRangeException>(() => XMath.RmsNorm(x, double.NaN));
        Assert.Throws<ArgumentException>(() => XMath.Prelu(x, v2));

        // 未初期化ノードは形状・epsilon 異常より優先して InvalidOperationException
        Assert.Throws<InvalidOperationException>(() => XMath.Linear(default(MatrixNode<double>), x));
        Assert.Throws<InvalidOperationException>(() => XMath.Linear(validW, x, default(VectorNode<double>)));
        Assert.Throws<InvalidOperationException>(() => XMath.LayerNorm(x, default(VectorNode<double>), v3, 1e-5));
        Assert.Throws<InvalidOperationException>(() => XMath.LayerNorm(default(MatrixNode<double>), -1.0));
        Assert.Throws<InvalidOperationException>(() => XMath.RmsNorm(x, default(VectorNode<double>), 1e-5));
        Assert.Throws<InvalidOperationException>(() => XMath.Prelu(x, default(VectorNode<double>)));

        // 別テープ混在
        Assert.Throws<InvalidOperationException>(() => XMath.Linear(other.Leaf(new Matrix<double>(2, 3)), x));
        Assert.Throws<InvalidOperationException>(() => XMath.LayerNorm(x, other.Leaf(new LinearAlgebra.Vector<double>(3)), v3, 1e-5));
        Assert.Throws<InvalidOperationException>(() => XMath.Prelu(x, other.Leaf(new LinearAlgebra.Vector<double>(3))));

        // 非対応要素型
        Assert.Throws<NotSupportedException>(() => XMath.LayerNorm(default(MatrixNode<Half>), (Half)1e-2f));

        // 例外後も同じテープで正常に記録・逆伝播できる
        var y = XMath.Linear(validW, x, v2);
        tape.Backward(XMath.Sum(XMath.ColumnSums(y)));
        Assert.Equal(2, tape.GetGradient(validW).Rows);

        // 逆伝播完了後の記録は拒否される
        Assert.Throws<InvalidOperationException>(() => XMath.LayerNorm(x, 1e-5));

        // Reset 後の失効ノードは形状・epsilon 異常より優先して InvalidOperationException
        tape.Reset();
        Assert.Throws<InvalidOperationException>(() => XMath.LayerNorm(x, -1.0));
        Assert.Throws<InvalidOperationException>(() => XMath.RmsNorm(x, -1.0));
        Assert.Throws<InvalidOperationException>(() => XMath.Prelu(x, v2));
    }

    [Fact]
    public void EmptyInputsProduceEmptyGradients()
    {
        // 3 × 0（0 列）
        {
            var tape = new Tape();
            var x = tape.Leaf(new Matrix<double>(3, 0));
            var w = tape.Leaf(new Matrix<double>(2, 3));
            var b = tape.Leaf(new LinearAlgebra.Vector<double>([1.0, 2.0]));
            var scale = tape.Leaf(new LinearAlgebra.Vector<double>([1.0, 2, 3]));
            var shift = tape.Leaf(new LinearAlgebra.Vector<double>(3));
            var y = XMath.Linear(w, XMath.Prelu(XMath.RmsNorm(XMath.LayerNorm(x, scale, shift, 1e-5), scale, 1e-5), shift), b);
            tape.Backward(XMath.Sum(XMath.ColumnSums(y)));
            Assert.Equal(0, tape.GetGradient(x).Columns);
            Assert.Equal(new double[6], tape.GetGradient(w).ToMatrix().AsSpan().ToArray());
            Assert.Equal(new double[2], tape.GetGradient(b).ToArray());
            Assert.Equal(new double[3], tape.GetGradient(scale).ToArray());
            Assert.Equal(new double[3], tape.GetGradient(shift).ToArray());
        }

        // 0 × 4（0 行）
        {
            var tape = new Tape();
            var x = tape.Leaf(new Matrix<double>(0, 4));
            var scale = tape.Leaf(new LinearAlgebra.Vector<double>(0));
            var shift = tape.Leaf(new LinearAlgebra.Vector<double>(0));
            var sharedSlope = tape.Leaf(new LinearAlgebra.Vector<double>([0.25]));
            var y = XMath.Prelu(XMath.RmsNorm(XMath.LayerNorm(x, scale, shift, 1e-5), 1e-5), sharedSlope);
            tape.Backward(XMath.Sum(XMath.ColumnSums(y)));
            Assert.Equal(0, tape.GetGradient(x).Rows);
            Assert.Equal(0, tape.GetGradient(scale).Length);
            Assert.Equal(0, tape.GetGradient(shift).Length);
            Assert.Equal(0.0, tape.GetGradient(sharedSlope)[0]);
        }
    }

    [Fact]
    public void PartialGradientRequirementsAndAccumulation()
    {
        double[] xData = [-0.8, 0.3, -1.2, 0.5, -0.4, 1.1, 0.7, -0.6, 0.2, -1.5, 0.9, -0.1];
        double[] gData = [0.4, -0.7, 0.2, 1.1, -0.3, 0.8, -0.5, 0.6, 0.9, -0.2, 0.3, -0.4];
        var xVal = Matrix<double>.FromColumnMajor(3, 4, xData);
        var scaleVal = new LinearAlgebra.Vector<double>([1.1, 0.9, 1.3]);
        var shiftVal = new LinearAlgebra.Vector<double>([-0.2, 0.4, 0.1]);
        var gVal = Matrix<double>.FromColumnMajor(3, 4, gData);

        // LayerNorm で shift のみ勾配要（x と scale は定数、stats 保存なし）
        {
            var tape = new Tape();
            var shift = tape.Leaf(shiftVal);
            var y = XMath.LayerNorm(tape.Constant(xVal), tape.Constant(scaleVal), shift, 1e-5);
            tape.Backward(Loss(tape, y, gData));
            var expected = XMath.RowSums(gVal);
            var actual = tape.GetGradient(shift);
            for (int i = 0; i < 3; i++)
            {
                Assert.Equal(expected[i], actual[i], 14);
            }
        }

        // 既存勾配への加算蓄積とエイリアシング（Linear(w, w)）
        {
            var wParam = Parameter<double>.Matrix(3, 3);
            double[] wData = [0.5, -0.2, 0.8, -0.4, 0.7, 0.1, 0.3, -0.6, 0.9];
            double[] g3Data = [0.2, -0.5, 0.4, 0.7, -0.1, 0.3, -0.8, 0.6, 0.5];
            wData.CopyTo(wParam.Value);
            wParam.Gradient.Fill(1.5);

            var tape = new Tape();
            var wNode = tape.UseMatrix(wParam);
            var y = XMath.Linear(wNode, wNode);
            tape.Backward(Loss(tape, y, g3Data));

            var reference = new Tape();
            var wRef = reference.Leaf(Matrix<double>.FromColumnMajor(3, 3, wData));
            var yRef = wRef * wRef;
            reference.Backward(Loss(reference, yRef, g3Data));
            var expected = reference.GetGradient(wRef);
            for (int j = 0; j < 3; j++)
            {
                for (int i = 0; i < 3; i++)
                {
                    Assert.Equal(1.5 + expected[i, j], wParam.Gradient[j * 3 + i], 13);
                }
            }
        }

        // 混合の勾配要否（x 定数 + slopes 葉、および x 葉 + slopes 定数）
        {
            var slopeVal = new LinearAlgebra.Vector<double>([0.2, -0.3, 0.5]);
            var refTape = new Tape();
            var rx = refTape.Leaf(xVal);
            var rs = refTape.Leaf(slopeVal);
            refTape.Backward(Loss(refTape, XMath.Prelu(rx, rs), gData));

            var tapeXOnly = new Tape();
            var lx = tapeXOnly.Leaf(xVal);
            tapeXOnly.Backward(Loss(tapeXOnly, XMath.Prelu(lx, tapeXOnly.Constant(slopeVal)), gData));
            TestData.Same(refTape.GetGradient(rx), tapeXOnly.GetGradient(lx));

            var tapeSOnly = new Tape();
            var ls = tapeSOnly.Leaf(slopeVal);
            tapeSOnly.Backward(Loss(tapeSOnly, XMath.Prelu(tapeSOnly.Constant(xVal), ls), gData));
            Assert.Equal(refTape.GetGradient(rs).ToArray(), tapeSOnly.GetGradient(ls).ToArray());
        }
    }

    internal static MatrixNode<T> Apply<T>(string operation, T epsilon, Leaves<T> leaves) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var m = leaves.Matrices;
        var v = leaves.Vectors;
        return operation switch
        {
            "Linear" => XMath.Linear(m[0], m[1], v[0]),
            "LinearNoBias" => XMath.Linear(m[0], m[1]),
            "LayerNorm" => XMath.LayerNorm(m[0], v[0], v[1], epsilon),
            "LayerNormNoAffine" => XMath.LayerNorm(m[0], epsilon),
            "RmsNorm" => XMath.RmsNorm(m[0], v[0], epsilon),
            "RmsNormNoScale" => XMath.RmsNorm(m[0], epsilon),
            "PreluPerRow" or "PreluShared" => XMath.Prelu(m[0], v[0]),
            _ => throw new ArgumentException(operation),
        };
    }

    // 記録しない推論版の XMath で同じ演算を求める。
    internal static Matrix<T> Evaluate<T>(string operation, T epsilon, double[][] inputs, (int Rows, int Columns)[] shapes, int matrices)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        var m = Enumerable.Range(0, matrices).Select(i => TestData.Matrix<T>(inputs[i], shapes[i].Rows, shapes[i].Columns)).ToArray();
        var v = inputs.Skip(matrices).Select(TestData.Vector<T>).ToArray();
        return operation switch
        {
            "Linear" => XMath.Linear<T>(m[0], m[1], v[0]),
            "LinearNoBias" => XMath.Linear<T>(m[0], m[1]),
            "LayerNorm" => XMath.LayerNorm<T>(m[0], v[0], v[1], epsilon),
            "LayerNormNoAffine" => XMath.LayerNorm<T>(m[0], epsilon),
            "RmsNorm" => XMath.RmsNorm<T>(m[0], v[0], epsilon),
            "RmsNormNoScale" => XMath.RmsNorm<T>(m[0], epsilon),
            "PreluPerRow" or "PreluShared" => XMath.Prelu<T>(m[0], v[0]),
            _ => throw new ArgumentException(operation),
        };
    }

    private static double Evaluate(string operation, double epsilon, double[][] inputs, (int Rows, int Columns)[] shapes, int matrices, double[] g)
    {
        var y = Evaluate(operation, epsilon, inputs, shapes, matrices);
        double sum = 0;
        for (int j = 0; j < y.Columns; j++)
        {
            for (int i = 0; i < y.Rows; i++)
            {
                sum += g[j * y.Rows + i] * y[i, j];
            }
        }

        return sum;
    }

    // L = Σ G ⊙ Y。Y の勾配はちょうど G になる。
    internal static ScalarNode Loss<T>(Tape tape, MatrixNode<T> y, double[] g) where T : unmanaged, IFloatingPointIeee754<T> =>
        XMath.Sum(XMath.ColumnSums(XMath.Hadamard(y, tape.Constant(TestData.Matrix<T>(g, y.Rows, y.Columns)))));

    // 演算の入力の乱数（列ごとに平均と尺度を変える。PReLU の入力は 0 から 1e-3 以上離す）。
    internal static (double[][] Inputs, (int Rows, int Columns)[] Shapes, int Matrices) RandomInputs(string operation, int rows, int inner, int columns, int seed)
    {
        if (operation.StartsWith("Linear", StringComparison.Ordinal))
        {
            double[][] linear = [TestData.Uniform(rows * inner, seed), TestData.Uniform(inner * columns, seed + 1, 2), TestData.Uniform(rows, seed + 2)];
            (int, int)[] shapes = [(rows, inner), (inner, columns), (rows, 1)];
            return operation == "Linear" ? (linear, shapes, 2) : (linear[..2], shapes[..2], 2);
        }

        var random = new Random(seed);
        var x = new double[rows * columns];
        for (int j = 0; j < columns; j++)
        {
            double offset = 4 * random.NextDouble() - 2, spread = Math.Pow(10, -2 * random.NextDouble());
            for (int i = 0; i < rows; i++)
            {
                double value = offset + spread * (2 * random.NextDouble() - 1);
                x[j * rows + i] = operation.StartsWith("Prelu", StringComparison.Ordinal) && Math.Abs(value) < 1e-3 ? 1e-3 : value;
            }
        }

        var scale = TestData.Uniform(rows, seed + 1, 0.5, 1.5);
        var shift = TestData.Uniform(rows, seed + 2);
        double[][] inputs = operation switch
        {
            "LayerNorm" => [x, scale, shift],
            "RmsNorm" => [x, scale],
            "PreluPerRow" => [x, TestData.Uniform(rows, seed + 3, 0.5)],
            "PreluShared" => [x, [0.2]],
            _ => [x],
        };
        return (inputs, inputs.Select((a, i) => i == 0 ? (rows, columns) : (a.Length, 1)).ToArray(), 1);
    }

    internal static double[] ToArray<T>(ReadOnlyMatrixSpan<T> x) where T : unmanaged, INumberBase<T>
    {
        var result = new double[x.Rows * x.Columns];
        for (int j = 0; j < x.Columns; j++)
        {
            for (int i = 0; i < x.Rows; i++)
            {
                result[j * x.Rows + i] = double.CreateTruncating(x[i, j]);
            }
        }

        return result;
    }

    private static void Close(double[] expected, double[] actual, double tolerance, string context)
    {
        Assert.Equal(expected.Length, actual.Length);
        double scale = Math.Max(1, expected.Select(Math.Abs).DefaultIfEmpty().Max());
        double worst = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.False(double.IsNaN(actual[i]), $"{context}: NaN at {i}");
            worst = Math.Max(worst, Math.Abs(actual[i] - expected[i]));
        }

        Assert.True(worst <= tolerance * scale, $"{context}: max error {worst:E3} > {tolerance * scale:E3}");
    }

    // 演算の入力を葉（または定数）にする。先頭の matrices 個は行列、残りはベクトル。
    // views なら、行列は転置した葉の転置のビュー（行が連続でない）と、大きい葉の部分行列のビューにする。
    internal sealed class Leaves<T> where T : unmanaged, IFloatingPointIeee754<T>
    {
        private readonly MatrixNode<T>[] sources;

        public Leaves(Tape tape, double[][] inputs, (int Rows, int Columns)[] shapes, int matrices, bool constant = false, bool views = false)
        {
            Matrices = new MatrixNode<T>[matrices];
            sources = new MatrixNode<T>[matrices];
            for (int i = 0; i < matrices; i++)
            {
                var value = TestData.Matrix<T>(inputs[i], shapes[i].Rows, shapes[i].Columns);
                if (!views)
                {
                    sources[i] = Matrices[i] = constant ? tape.Constant(value) : tape.Leaf(value);
                }
                else if (i == 0)
                {
                    sources[i] = tape.Leaf(XMath.Transpose(value));
                    Matrices[i] = sources[i].T;
                }
                else
                {
                    var large = new Matrix<T>(value.Rows + 3, value.Columns + 2);
                    value.AsMatrixSpan().CopyTo(large[2..(value.Rows + 2), 1..(value.Columns + 1)]);
                    sources[i] = tape.Leaf(large);
                    Matrices[i] = sources[i][2..(value.Rows + 2), 1..(value.Columns + 1)];
                }
            }

            Vectors = inputs.Skip(matrices).Select(a => constant ? tape.Constant(TestData.Vector<T>(a)) : tape.Leaf(TestData.Vector<T>(a))).ToArray();
            Views = views;
        }

        public MatrixNode<T>[] Matrices { get; }

        public VectorNode<T>[] Vectors { get; }

        private bool Views { get; }

        // 入力 index の勾配を、その入力の形の列優先の配列で返す。
        public double[] Gradient(Tape tape, int index)
        {
            if (index >= Matrices.Length)
            {
                var g = tape.GetGradient(Vectors[index - Matrices.Length]);
                var result = new double[g.Length];
                for (int i = 0; i < g.Length; i++)
                {
                    result[i] = double.CreateTruncating(g[i]);
                }

                return result;
            }

            var gradient = tape.GetGradient(sources[index]);
            if (!Views)
            {
                return ToArray(gradient);
            }

            return index == 0
                ? ToArray(gradient.T)
                : ToArray(gradient[2..(Matrices[index].Rows + 2), 1..(Matrices[index].Columns + 1)]);
        }
    }
}
