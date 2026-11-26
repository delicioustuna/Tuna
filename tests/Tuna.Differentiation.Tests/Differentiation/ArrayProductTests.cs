using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.Tests.Differentiation;

/// <summary>行列積・行列ベクトル積・総和・内積の値（XMath とのビット一致）と勾配（中心差分・解析式）</summary>
public class ArrayProductTests
{
    // 行列積の向きと部分行列の組。部分行列は大きい格納の一部で、列の間隔が行数と違う。
    public static TheoryData<int, int, int, bool, bool, bool> ProductShapes()
    {
        var data = new TheoryData<int, int, int, bool, bool, bool>();
        foreach (var (m, k, n) in new[] { (3, 4, 5), (1, 5, 3), (5, 1, 3), (24, 24, 24), (25, 30, 7), (64, 100, 50), (130, 140, 120) })
        {
            foreach (bool ta in new[] { false, true })
            {
                foreach (bool tb in new[] { false, true })
                {
                    data.Add(m, k, n, ta, tb, false);
                }
            }

            data.Add(m, k, n, false, false, true);
            data.Add(m, k, n, true, true, true);
        }

        return data;
    }

    [Theory]
    [MemberData(nameof(ProductShapes))]
    public void MatrixProductValuesMatchXMathBitForBit(int m, int k, int n, bool ta, bool tb, bool part)
    {
        ProductValues<double>(m, k, n, ta, tb, part);
        ProductValues<float>(m, k, n, ta, tb, part);
    }

    [Theory]
    [InlineData(1, 1)]
    [InlineData(5, 7)]
    [InlineData(64, 3)]
    [InlineData(100, 300)]
    [InlineData(1000, 1000)]
    [InlineData(2100, 40)]
    public void MatrixVectorValuesMatchXMathBitForBit(int m, int k)
    {
        MatrixVectorValues<double>(m, k);
        MatrixVectorValues<float>(m, k);
    }

    [Theory]
    [InlineData(17, 9)]
    [InlineData(64, 33)]
    [InlineData(301, 203)]
    [InlineData(1000, 1000)]
    [InlineData(2100, 40)]
    public void CopiedViewsKeepTheBitsOfXMathOnTheSameViews(int rows, int columns)
    {
        CopiedViewValues<double>(rows, columns);
        CopiedViewValues<float>(rows, columns);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(63)]
    [InlineData(64)]
    [InlineData(65)]
    [InlineData(1000)]
    [InlineData(200_000)]
    public void SumAndDotValuesMatchXMathWidened(int length)
    {
        var random = new Random(length);
        var x = RandomVector<double>(random, length);
        var y = RandomVector<double>(random, length);
        var tape = new Tape();
        var xn = tape.Leaf(x);
        var yn = tape.Constant(y);
        DifferentiationAssert.EqualBits(XMath.Sum(x), XMath.Sum(xn).Value);
        DifferentiationAssert.EqualBits(XMath.Dot(x, y), XMath.Dot(xn, yn).Value);
        DifferentiationAssert.EqualBits(XMath.Dot(y, x), XMath.Dot(yn, xn).Value);
        if (length > 2)
        {
            DifferentiationAssert.EqualBits(XMath.Sum(x[1..^1]), XMath.Sum(xn[1..^1]).Value);
        }

        var xf = RandomVector<float>(random, length);
        var yf = RandomVector<float>(random, length);
        var xfn = tape.Leaf(xf);
        DifferentiationAssert.EqualBits(XMath.Sum(xf), XMath.Sum(xfn).Value);
        DifferentiationAssert.EqualBits(XMath.Dot(xf, yf), XMath.Dot(xfn, tape.Constant(yf)).Value);
    }

    [Theory]
    [InlineData(3, 4, 5)]
    [InlineData(1, 3, 1)]
    [InlineData(6, 1, 4)]
    [InlineData(9, 7, 8)]
    public void MatrixProductGradientsMatchCentralDifferences(int m, int k, int n)
    {
        var random = new Random(m * 100 + k * 10 + n);
        var a = RandomMatrix<double>(random, m, k);
        var b = RandomMatrix<double>(random, k, n);
        var u = RandomVector<double>(random, m);
        var v = RandomVector<double>(random, n);
        // f(A, B) = uᵀ(AB)v
        double Value(Matrix<double> left, Matrix<double> right) => XMath.Dot(u, XMath.Multiply(XMath.Multiply(left, right), v));
        var tape = new Tape();
        var an = tape.Leaf(a);
        var bn = tape.Leaf(b);
        tape.Backward(XMath.Dot(tape.Constant(u), an * bn * tape.Constant(v)));
        var da = tape.GetGradient(an).ToMatrix();
        var db = tape.GetGradient(bn).ToMatrix();
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < k; j++)
            {
                double numeric = Numeric(a, i, j, () => Value(a, b));
                DifferentiationAssert.Close(numeric, da[i, j], 1e-7, 1e-8, $"dA[{i}, {j}]");
            }
        }

        for (int i = 0; i < k; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double numeric = Numeric(b, i, j, () => Value(a, b));
                DifferentiationAssert.Close(numeric, db[i, j], 1e-7, 1e-8, $"dB[{i}, {j}]");
            }
        }
    }

    [Theory]
    [MemberData(nameof(ProductShapes))]
    public void MatrixProductGradientsOfViewsMatchTheAnalyticForm(int m, int k, int n, bool ta, bool tb, bool part)
    {
        var random = new Random(m + 7 * k + 31 * n + (ta ? 1000 : 0) + (tb ? 2000 : 0) + (part ? 4000 : 0));
        var (aStorage, aRange) = Storage<double>(random, m, k, ta, part);
        var (bStorage, bRange) = Storage<double>(random, k, n, tb, part);
        var g = RandomMatrix<double>(random, m, n);
        // f = Σ_j g[:, j]ᵀ (AB)e_j。要素ごとの積はまだないので、AB の列を行列ベクトル積で取り出して内積をとる。
        var tape = new Tape();
        var aLeaf = tape.Leaf(aStorage);
        var bLeaf = tape.Leaf(bStorage);
        var an = View(aLeaf, aRange, ta);
        var bn = View(bLeaf, bRange, tb);
        var c = an * bn;
        ScalarNode loss = 0;
        for (int j = 0; j < n; j++)
        {
            var column = c * tape.Constant(Unit(n, j));
            loss += XMath.Dot(tape.Constant(g.Column(j).ToVector()), column);
        }

        tape.Backward(loss);
        var a = an.Value.ToMatrix();
        var b = bn.Value.ToMatrix();
        // dA = G·Bᵀ、dB = Aᵀ·G
        var expectedA = XMath.Multiply(g, b.T);
        var expectedB = XMath.Multiply(a.T, g);
        AssertClose(expectedA, tape.GetGradient(an).ToMatrix(), 1e-12);
        AssertClose(expectedB, tape.GetGradient(bn).ToMatrix(), 1e-12);
        // 格納の、ビューの外側の勾配は 0。
        double outside = Total(tape.GetGradient(aLeaf).ToMatrix()) - Total(tape.GetGradient(an).ToMatrix());
        Assert.Equal(0, outside, 1e-9);
    }

    [Theory]
    [InlineData(4, 3, false)]
    [InlineData(4, 3, true)]
    [InlineData(1, 6, false)]
    [InlineData(7, 1, true)]
    [InlineData(40, 30, true)]
    public void MatrixVectorGradientsMatchCentralDifferences(int m, int k, bool transposed)
    {
        var random = new Random(m * 13 + k + (transposed ? 500 : 0));
        var storage = transposed ? RandomMatrix<double>(random, k, m) : RandomMatrix<double>(random, m, k);
        var x = RandomVector<double>(random, k);
        var u = RandomVector<double>(random, m);
        double Value() => XMath.Dot(u, XMath.Multiply(transposed ? storage.T : storage, x));
        var tape = new Tape();
        var leaf = tape.Leaf(storage);
        var a = transposed ? leaf.T : leaf;
        var xn = tape.Leaf(x);
        tape.Backward(XMath.Dot(tape.Constant(u), a * xn));
        var ds = tape.GetGradient(leaf).ToMatrix();
        for (int i = 0; i < storage.Rows; i++)
        {
            for (int j = 0; j < storage.Columns; j++)
            {
                DifferentiationAssert.Close(Numeric(storage, i, j, Value), ds[i, j], 1e-7, 1e-8, $"dA[{i}, {j}]");
            }
        }

        var dx = tape.GetGradient(xn);
        for (int j = 0; j < k; j++)
        {
            double saved = x[j];
            double numeric = DifferentiationAssert.Richardson(t =>
            {
                x[j] = t;
                double value = Value();
                x[j] = saved;
                return value;
            }, saved);
            DifferentiationAssert.Close(numeric, dx[j], 1e-7, 1e-8, $"dx[{j}]");
        }
    }

    [Fact]
    public void SumAndDotGradientsCoverRepeatedAndOverlappingInputs()
    {
        var tape = new Tape();
        var x = tape.Leaf(new Vector<double>([1, 2, 3, 4]));
        // Σx + x·x + x[0..3]·x[1..4] + 2·Σx[2..4]
        ScalarNode loss = XMath.Sum(x) + XMath.Dot(x, x) + XMath.Dot(x[0..3], x[1..4]) + 2 * XMath.Sum(x[2..4]);
        Assert.Equal(10 + 30 + 20 + 14, loss.Value);
        tape.Backward(loss);
        // 1 + 2x + (x[1..4] を前に、x[0..3] を後ろにずらした和) + 2·[0, 0, 1, 1]
        Assert.Equal(new double[] { 1 + 2 + 2, 1 + 4 + 3 + 1, 1 + 6 + 4 + 2 + 2, 1 + 8 + 3 + 2 }, tape.GetGradient(x).ToArray());
    }

    [Fact]
    public void ProductOfAMatrixWithItselfAddsBothContributions()
    {
        var random = new Random(5);
        var a = RandomMatrix<double>(random, 4, 4);
        var g = RandomMatrix<double>(random, 4, 4);
        var tape = new Tape();
        var an = tape.Leaf(a);
        var c = an * an;
        ScalarNode loss = 0;
        for (int j = 0; j < 4; j++)
        {
            loss += XMath.Dot(tape.Constant(g.Column(j).ToVector()), c * tape.Constant(Unit(4, j)));
        }

        tape.Backward(loss);
        // d(AA) = G·Aᵀ + Aᵀ·G
        AssertClose(XMath.Add(XMath.Multiply(g, a.T), XMath.Multiply(a.T, g)), tape.GetGradient(an).ToMatrix(), 1e-13);
    }

    [Fact]
    public void SingleGradientsAgreeWithDoubleGradients()
    {
        var random = new Random(11);
        var a = RandomMatrix<double>(random, 30, 20);
        var b = RandomMatrix<double>(random, 20, 25);
        var v = RandomVector<double>(random, 25);
        var u = RandomVector<double>(random, 30);
        var (da, db, dv) = Gradients(a, b, v, u);
        var (fa, fb, fv) = Gradients(a.ConvertChecked<float>(), b.ConvertChecked<float>(), v.ConvertChecked<float>(), u.ConvertChecked<float>());
        AssertClose(da, fa.ConvertChecked<double>(), 2e-5);
        AssertClose(db, fb.ConvertChecked<double>(), 2e-5);
        AssertClose(dv.AsMatrix(25, 1), fv.ConvertChecked<double>().AsMatrix(25, 1), 2e-5);

        static (Matrix<T>, Matrix<T>, Vector<T>) Gradients<T>(Matrix<T> a, Matrix<T> b, Vector<T> v, Vector<T> u)
            where T : unmanaged, System.Numerics.IFloatingPointIeee754<T>
        {
            var tape = new Tape();
            var an = tape.Leaf(a);
            var bn = tape.Leaf(b);
            var vn = tape.Leaf(v);
            tape.Backward(XMath.Dot(tape.Constant(u), an * (bn * vn)) + XMath.Sum(vn));
            return (tape.GetGradient(an).ToMatrix(), tape.GetGradient(bn).ToMatrix(), tape.GetGradient(vn).ToVector());
        }
    }

    [Theory]
    [InlineData(16, 3)]
    [InlineData(500, 20)]
    public void LinearRegressionGradientMatchesTheAnalyticGradient(int samples, int features)
    {
        var random = new Random(samples + features);
        var x = RandomMatrix<double>(random, features, samples);   // [特徴 × 標本]
        var y = RandomVector<double>(random, samples);
        var theta = RandomVector<double>(random, features);
        var tape = new Tape();
        var w = tape.Leaf(theta);
        var yn = tape.Constant(y);
        var r = tape.Constant(x).T * w;
        // ½‖Xᵀw − y‖² = ½ rᵀr − yᵀr + ½ yᵀy
        ScalarNode loss = 0.5 * XMath.Dot(r, r) - XMath.Dot(yn, r) + 0.5 * XMath.Dot(yn, yn);
        double[] gradient = new double[features];
        double value = tape.Gradient(loss, new VectorSpan<double>(gradient));
        var residual = XMath.Subtract(XMath.Multiply(x.T, theta), y);
        var expected = XMath.Multiply(x, residual);
        DifferentiationAssert.Close(0.5 * XMath.Dot(residual, residual), value, 1e-11, 1e-12, "value");
        for (int i = 0; i < features; i++)
        {
            DifferentiationAssert.Close(expected[i], gradient[i], 1e-11, 1e-11 * samples, $"gradient[{i}]");
        }
    }

    // 定数・葉にビュー（転置の部分行列と、列優先の部分行列）を渡して写しても、XMath に同じビューを渡した結果と同じビットになる。
    // 行列ベクトル積は転置の有無で部分和の順序が変わるので、転置のビューを列優先に並べ直すと値が変わる。
    private static void CopiedViewValues<T>(int rows, int columns) where T : unmanaged, System.Numerics.IFloatingPointIeee754<T>
    {
        var random = new Random(rows * 7 + columns);
        var storage = RandomMatrix<T>(random, rows + 3, columns + 2);
        ReadOnlyMatrixSpan<T> part = ((ReadOnlyMatrixSpan<T>)storage)[1..(rows + 1), 2..];
        var w = RandomVector<T>(random, rows);
        var x = RandomVector<T>(random, columns);
        var b = RandomMatrix<T>(random, rows, 5);
        var tape = new Tape();
        var wn = tape.Constant(w);
        var xn = tape.Constant(x);
        var bn = tape.Constant(b);
        foreach (var node in new[] { tape.Constant(part.T), tape.Leaf(part.T) })
        {
            EqualBits(XMath.Multiply(part.T, w).AsSpan(), (node * wn).Value.ToVector().AsSpan());
            EqualBits(XMath.Multiply(part.T, b).AsSpan(), (node * bn).Value.ToMatrix().AsSpan());
        }

        foreach (var node in new[] { tape.Constant(part), tape.Leaf(part) })
        {
            EqualBits(XMath.Multiply(part, x).AsSpan(), (node * xn).Value.ToVector().AsSpan());
        }

        static void EqualBits(ReadOnlySpan<T> expected, ReadOnlySpan<T> actual)
        {
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                DifferentiationAssert.EqualBits(double.CreateTruncating(expected[i]), double.CreateTruncating(actual[i]));
            }
        }
    }

    [Theory]
    [InlineData(3, 5)]
    [InlineData(40, 300)]
    public void FrozenParametersLetTheInputBeSolvedFor(int outputs, int inputs)
    {
        // 重みを定数にして入力を葉にする逆算。½‖Wx − t‖² + bᵀx の x についての勾配は Wᵀ(Wx − t) + b。
        var random = new Random(outputs * 1000 + inputs);
        var weight = Parameter<double>.Matrix(outputs, inputs);
        RandomMatrix<double>(random, outputs, inputs).AsSpan().CopyTo(weight.Value);
        var bias = Parameter<double>.Vector(inputs);
        RandomVector<double>(random, inputs).AsSpan().CopyTo(bias.Value);
        weight.Gradient.Fill(7);
        bias.Gradient.Fill(7);
        var x = RandomVector<double>(random, inputs);
        var t = RandomVector<double>(random, outputs);
        var tape = new Tape();
        var w = tape.ConstantMatrix(weight);
        var b = tape.ConstantVector(bias);
        Assert.False(w.RequiresGradient);
        Assert.False(b.RequiresGradient);
        var xn = tape.Leaf(x);
        var tn = tape.Constant(t);
        var r = w * xn;
        ScalarNode loss = 0.5 * XMath.Dot(r, r) - XMath.Dot(tn, r) + 0.5 * XMath.Dot(tn, tn) + XMath.Dot(b, xn);
        double[] gradient = new double[inputs];
        tape.Gradient(loss, new VectorSpan<double>(gradient));

        var matrix = new Matrix<double>(outputs, inputs);
        weight.Value.CopyTo(matrix.AsSpan());
        var expected = XMath.Add(XMath.Multiply(matrix.T, XMath.Subtract(XMath.Multiply(matrix, x), t)), new Vector<double>(bias.Value.ToArray()));
        for (int i = 0; i < inputs; i++)
        {
            DifferentiationAssert.Close(expected[i], gradient[i], 1e-12, 1e-12 * outputs, $"gradient[{i}]");
        }

        // 重みの勾配は変えず、値は写さずに参照する。
        Assert.All(weight.Gradient.ToArray(), g => Assert.Equal(7, g));
        Assert.All(bias.Gradient.ToArray(), g => Assert.Equal(7, g));
        weight.Value[0] = 123;
        bias.Value[0] = 456;
        Assert.Equal(123, w.Value[0, 0]);
        Assert.Equal(456, b.Value[0]);
        Assert.Throws<ArgumentException>(() => tape.ConstantVector(weight));
        Assert.Throws<ArgumentException>(() => tape.ConstantMatrix(bias));
        Assert.Throws<ArgumentNullException>(() => tape.ConstantMatrix<double>(null!));
    }

    private static void ProductValues<T>(int m, int k, int n, bool ta, bool tb, bool part)
        where T : unmanaged, System.Numerics.IFloatingPointIeee754<T>
    {
        var random = new Random(m * 1000 + k * 10 + n);
        var (aStorage, aRange) = Storage<T>(random, m, k, ta, part);
        var (bStorage, bRange) = Storage<T>(random, k, n, tb, part);
        var aView = SpanView(aStorage, aRange, ta);
        var bView = SpanView(bStorage, bRange, tb);
        var expected = XMath.Multiply(aView, bView);
        var tape = new Tape();
        var recorded = View(tape.Leaf(aStorage), aRange, ta) * View(tape.Constant(bStorage), bRange, tb);
        var constant = View(tape.Constant(aStorage), aRange, ta) * View(tape.Constant(bStorage), bRange, tb);
        Assert.True(recorded.RequiresGradient);
        Assert.False(constant.RequiresGradient);
        Assert.Equal(expected.AsSpan().ToArray(), recorded.Value.ToMatrix().AsSpan().ToArray());
        Assert.Equal(expected.AsSpan().ToArray(), constant.Value.ToMatrix().AsSpan().ToArray());
    }

    private static void MatrixVectorValues<T>(int m, int k) where T : unmanaged, System.Numerics.IFloatingPointIeee754<T>
    {
        var random = new Random(m + k);
        var a = RandomMatrix<T>(random, m, k);
        var at = RandomMatrix<T>(random, k, m);
        var x = RandomVector<T>(random, k);
        var tape = new Tape();
        var xn = tape.Leaf(x);
        Assert.Equal(XMath.Multiply(a, x).ToArray(), (tape.Constant(a) * xn).Value.ToArray());
        Assert.Equal(XMath.Multiply(at.T, x).ToArray(), (tape.Constant(at).T * xn).Value.ToArray());
        if (m > 2 && k > 2)
        {
            Assert.Equal(XMath.Multiply(a[1..^1, 1..^1], x[1..^1]).ToArray(), (tape.Constant(a)[1..^1, 1..^1] * xn[1..^1]).Value.ToArray());
        }
    }

    // m × k の論理的な形の行列を、転置なら k × m で、部分なら上下左右に余白のある大きい格納に置く。
    private static (Matrix<T> Storage, (Range Rows, Range Columns) Range) Storage<T>(Random random, int rows, int columns, bool transposed, bool part)
        where T : unmanaged, System.Numerics.IFloatingPointIeee754<T>
    {
        int storedRows = transposed ? columns : rows, storedColumns = transposed ? rows : columns;
        int pad = part ? 2 : 0;
        var storage = RandomMatrix<T>(random, storedRows + 2 * pad, storedColumns + 2 * pad);
        return (storage, (pad..(pad + storedRows), pad..(pad + storedColumns)));
    }

    private static ReadOnlyMatrixSpan<T> SpanView<T>(Matrix<T> storage, (Range Rows, Range Columns) range, bool transposed)
        where T : unmanaged, System.Numerics.IFloatingPointIeee754<T>
    {
        ReadOnlyMatrixSpan<T> view = storage[range.Rows, range.Columns];
        return transposed ? view.T : view;
    }

    private static MatrixNode<T> View<T>(MatrixNode<T> node, (Range Rows, Range Columns) range, bool transposed)
        where T : unmanaged, System.Numerics.IFloatingPointIeee754<T>
    {
        var view = node[range.Rows, range.Columns];
        return transposed ? view.T : view;
    }

    internal static Matrix<T> RandomMatrix<T>(Random random, int rows, int columns) where T : unmanaged, System.Numerics.IFloatingPointIeee754<T>
    {
        var result = new Matrix<T>(rows, columns);
        var span = result.AsSpan();
        for (int i = 0; i < span.Length; i++)
        {
            span[i] = T.CreateTruncating(2 * random.NextDouble() - 1);
        }

        return result;
    }

    internal static Vector<T> RandomVector<T>(Random random, int length) where T : unmanaged, System.Numerics.IFloatingPointIeee754<T>
    {
        var result = new Vector<T>(length);
        for (int i = 0; i < length; i++)
        {
            result[i] = T.CreateTruncating(2 * random.NextDouble() - 1);
        }

        return result;
    }

    private static Vector<double> Unit(int length, int index)
    {
        var result = new Vector<double>(length);
        result[index] = 1;
        return result;
    }

    private static double Numeric(Matrix<double> matrix, int row, int column, Func<double> value)
    {
        double saved = matrix[row, column];
        double result = DifferentiationAssert.Richardson(t =>
        {
            matrix[row, column] = t;
            double v = value();
            matrix[row, column] = saved;
            return v;
        }, saved);
        return result;
    }

    private static double Total(Matrix<double> matrix) => XMath.Sum(matrix.AsVector());

    private static void AssertClose(Matrix<double> expected, Matrix<double> actual, double relative)
    {
        Assert.Equal((expected.Rows, expected.Columns), (actual.Rows, actual.Columns));
        double scale = Math.Max(1, XMath.NormInfinity(expected.AsVector()));
        for (int i = 0; i < expected.Length; i++)
        {
            double e = expected.AsSpan()[i], a = actual.AsSpan()[i];
            Assert.True(Math.Abs(e - a) <= relative * scale, $"[{i}]: expected {e:R}, actual {a:R}");
        }
    }
}
