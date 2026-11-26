using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.Tests.Differentiation;

/// <summary>配列のノードの作り方・ビュー・定数・重み・勾配の受け取りと、誤った使い方の例外</summary>
public class ArrayNodeTests
{
    [Fact]
    public void ConstantContainersAreReferencedAndViewsAreCopied()
    {
        var tape = new Tape();
        var matrix = Matrix<double>.FromRowMajor(2, 3, [1, 2, 3, 4, 5, 6]);
        var vector = new Vector<double>([7, 8, 9]);
        var a = tape.Constant(matrix);
        var v = tape.Constant(vector);
        var copiedMatrix = tape.Constant(matrix.AsMatrixSpan());
        var copiedVector = tape.Constant(vector.AsVectorSpan());
        matrix[0, 0] = 100;
        vector[1] = 200;
        Assert.Equal(100, a.Value[0, 0]);
        Assert.Equal(200, v.Value[1]);
        Assert.Equal(1, copiedMatrix.Value[0, 0]);
        Assert.Equal(8, copiedVector.Value[1]);
        Assert.False(a.RequiresGradient);
        Assert.False(v.RequiresGradient);
        Assert.False(XMath.Multiply(a, v).RequiresGradient);
        Assert.False(XMath.Sum(v).RequiresGradient);
    }

    [Fact]
    public void LeavesCopyValuesAndRequireGradients()
    {
        var tape = new Tape();
        double[] data = [1, 2, 3, 4];
        var x = tape.Leaf(new ReadOnlyVectorSpan<double>(data));
        var a = tape.Leaf(new ReadOnlyMatrixSpan<double>(data, 2, 2));
        data[0] = 100;
        Assert.Equal(1, x.Value[0]);
        Assert.Equal(1, a.Value[0, 0]);
        Assert.True(x.RequiresGradient);
        Assert.True(a.RequiresGradient);
        Assert.Equal(4, x.Length);
        Assert.Equal((2, 2), (a.Rows, a.Columns));
    }

    [Fact]
    public void TransposeAndRangesAreViewsOfTheSameStorage()
    {
        var tape = new Tape();
        var source = Matrix<double>.FromRowMajor(3, 4, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        var a = tape.Leaf(source);
        var t = a.T;
        Assert.Equal((4, 3), (t.Rows, t.Columns));
        Assert.Equal(source.T.ToArray(), t.Value.ToArray());
        var part = a[1..3, 1..4];
        Assert.Equal(source[1..3, 1..4].ToArray(), part.Value.ToArray());
        var partOfTranspose = a.T[1..3, 0..2];
        Assert.Equal(source.T[1..3, 0..2].ToArray(), partOfTranspose.Value.ToArray());
        Assert.Equal(source[0..2, 1..3].ToArray(), partOfTranspose.T.Value.ToArray());
        var x = tape.Leaf(new Vector<double>([1, 2, 3, 4, 5]));
        Assert.Equal(new double[] { 2, 3, 4 }, x[1..4].Value.ToArray());
        Assert.Equal(new double[] { 3 }, x[1..4][1..2].Value.ToArray());
    }

    [Fact]
    public void GradientsOfViewsAccumulateIntoTheirStorage()
    {
        var tape = new Tape();
        var a = tape.Leaf(Matrix<double>.FromRowMajor(2, 3, [1, 2, 3, 4, 5, 6]));
        var u = tape.Constant(new Vector<double>([1, 10]));
        var v = tape.Constant(new Vector<double>([2, 3]));
        // loss = Σ (a.T[0..2, ..]·u) + Σ (a[.., 1..3]·v)
        ScalarNode loss = XMath.Sum(a.T[0..2, ..] * u) + XMath.Sum(a[.., 1..3] * v);
        tape.Backward(loss);
        // ∂/∂a[i, j] = u[i]（j < 2）+ v[j − 1]（j ≥ 1）
        var expected = Matrix<double>.FromRowMajor(2, 3, [1, 1 + 2, 3, 10, 10 + 2, 3]);
        Assert.Equal(expected.ToArray(), tape.GetGradient(a).ToArray());
        Assert.Equal(expected.T.ToArray(), tape.GetGradient(a.T).ToArray());
        Assert.Equal(expected[.., 1..3].ToArray(), tape.GetGradient(a[.., 1..3]).ToArray());
    }

    [Fact]
    public void FlatGradientInterleavesScalarAndArrayLeavesInCreationOrder()
    {
        var tape = new Tape();
        var s = tape.Leaf(2);
        var x = tape.Leaf(new Vector<double>([1, 2]));
        var t = tape.Leaf(3);
        var a = tape.Leaf(Matrix<float>.FromRowMajor(2, 2, [1, 2, 3, 4]));
        var f = tape.Leaf(new Vector<float>([5, 6]));
        // loss = s·Σx + t·Σ(a·f) = 2·3 + 3·(17 + 39)
        ScalarNode loss = s * XMath.Sum(x) + t * XMath.Sum(a * f);
        double[] gradient = new double[1 + 2 + 1 + 4 + 2];
        Assert.Equal(6 + 3 * 56, tape.Gradient(loss, new VectorSpan<double>(gradient)));
        // ∂/∂a[i, j] = 3·f[j]（列優先）、∂/∂f[j] = 3·Σ_i a[i, j]
        Assert.Equal(new double[] { 3, 2, 2, 56, 15, 15, 18, 18, 12, 18 }, gradient);
        Assert.Throws<ArgumentException>(() => new Tape().Gradient(0, new VectorSpan<double>(new double[1])));
    }

    [Fact]
    public void FlatGradientOfATransposedLeafFollowsTheLeafInColumnMajorOrder()
    {
        var tape = new Tape();
        var storage = Matrix<double>.FromRowMajor(3, 2, [1, 2, 3, 4, 5, 6]);
        var a = tape.Leaf(storage.T);                     // 2×3 の [[1, 3, 5], [2, 4, 6]]
        var x = tape.Constant(new Vector<double>([10, 20, 30]));
        var u = tape.Constant(new Vector<double>([1, -1]));
        // loss = uᵀ(Ax)。∂/∂A[i, j] = u[i]·x[j]
        double[] gradient = new double[6];
        Assert.Equal((10 + 60 + 150) - (20 + 80 + 180), tape.Gradient(XMath.Dot(u, a * x), new VectorSpan<double>(gradient)));
        Assert.Equal(new double[] { 10, -10, 20, -20, 30, -30 }, gradient);
        Assert.Equal(new double[,] { { 10, 20, 30 }, { -10, -20, -30 } }, tape.GetGradient(a).ToArray());
        Assert.Equal(new double[,] { { 1, 3, 5 }, { 2, 4, 6 } }, a.Value.ToArray());
    }

    [Fact]
    public void FlatGradientRejectsTheScalarOnlyLengthWhenArraysAreLeaves()
    {
        var tape = new Tape();
        tape.Leaf(1);
        var x = tape.Leaf(new Vector<double>([1, 2, 3]));
        var loss = XMath.Sum(x);
        Assert.Throws<ArgumentException>(() => tape.Gradient(loss, new VectorSpan<double>(new double[1])));
        double[] gradient = new double[4];
        tape.Gradient(loss, new VectorSpan<double>(gradient));
        Assert.Equal(new double[] { 0, 1, 1, 1 }, gradient);
    }

    [Fact]
    public void GradientOfOneNodeWritesOnlyThatNode()
    {
        var tape = new Tape();
        var x = tape.Leaf(new Vector<double>([1, 2, 3]));
        var y = tape.Leaf(new Vector<double>([4, 5, 6]));
        double[] gradient = [-1, -1, -1, -1, -1];
        Assert.Equal(32, tape.Gradient(XMath.Dot(x, y), x, new VectorSpan<double>(gradient, [3], [2])));
        Assert.Equal(new double[] { 4, -1, 5, -1, 6 }, gradient);
        Assert.Equal(new double[] { 1, 2, 3 }, tape.GetGradient(y).ToArray());

        tape.Reset();
        var a = tape.Leaf(Matrix<double>.FromRowMajor(2, 2, [1, 2, 3, 4]));
        var v = tape.Constant(new Vector<double>([1, 1]));
        var result = new Matrix<double>(2, 2);
        Assert.Equal(10, tape.Gradient(XMath.Sum(a * v), a, result));
        Assert.Equal(new double[,] { { 1, 1 }, { 1, 1 } }, result.ToArray());
        tape.Reset();
        a = tape.Leaf(new Matrix<double>(2, 2));
        Assert.Throws<ArgumentException>(() => tape.Gradient(XMath.Sum(a * tape.Constant(new Vector<double>(2))), a, new Matrix<double>(2, 3)));
        Assert.Throws<ArgumentException>(() => tape.Gradient(0, tape.Leaf(new Vector<double>(2)), new Vector<double>(3)));
    }

    [Fact]
    public void UnreachedAndConstantNodesHaveZeroGradients()
    {
        var tape = new Tape();
        var x = tape.Leaf(new Vector<double>([1, 2]));
        var unused = tape.Leaf(new Vector<double>([3, 4]));
        var c = tape.Constant(new Vector<double>([5, 6]));
        var a = tape.Leaf(new Matrix<double>(2, 2));
        _ = a * unused;
        tape.Backward(XMath.Sum(x));
        Assert.Equal(new double[] { 1, 1 }, tape.GetGradient(x).ToArray());
        Assert.Equal(new double[] { 0, 0 }, tape.GetGradient(unused).ToArray());
        Assert.Equal(new double[] { 0, 0 }, tape.GetGradient(c).ToArray());
        Assert.Equal(new double[,] { { 0, 0 }, { 0, 0 } }, tape.GetGradient(a).ToArray());
    }

    [Fact]
    public void UnreachedRecordsDoNotTurnInfinitiesIntoNaN()
    {
        var tape = new Tape();
        var w = tape.Leaf(new Vector<double>([1, 2]));
        var infinite = tape.Constant(Matrix<double>.FromRowMajor(2, 2, [double.PositiveInfinity, 1, 1, 1]));
        _ = infinite * w;                                   // 出力から届かない
        ScalarNode zeroed = 0 * XMath.Dot(infinite * w, tape.Constant(new Vector<double>([1, 1])));
        tape.Backward(XMath.Sum(w) + zeroed);
        Assert.Equal(new double[] { 1, 1 }, tape.GetGradient(w).ToArray());
    }

    [Fact]
    public void ReachedProductsFollowIeeeArithmetic()
    {
        var tape = new Tape();
        var w = tape.Leaf(new Vector<double>([1, 2]));
        var infinite = tape.Constant(Matrix<double>.FromRowMajor(2, 2, [double.PositiveInfinity, 1, 1, 1]));
        // dy = 0 が届くので dw = Aᵀ·0 は ∞·0 = NaN を含む。
        tape.Backward(XMath.Dot(tape.Constant(new Vector<double>([0, 0])), infinite * w));
        Assert.True(double.IsNaN(tape.GetGradient(w)[0]));
        Assert.Equal(0, tape.GetGradient(w)[1]);
    }

    [Theory]
    [InlineData(1, 1, 3)]
    [InlineData(3, 2, 3)]
    [InlineData(37, 41, 3)]
    [InlineData(2, 41, 10)]
    [InlineData(5, 100, 33)]
    public void ReachedMatrixVectorProductsFollowIeeeArithmeticInBothOrientations(int m, int n, int zeroPeriod)
    {
        // Blas.Ger・転置しない Blas.Gemv は 0 の成分の列を省くが、届いた記録では ∞·0 = NaN にする。
        // A の向き（そのままか、転置を格納して .T）によらず同じ結果になる。
        // x は zeroPeriod ごとに 0 で、間隔が広いと 0 を含まないベクトルの区間と、末尾の端数の 0 を通る。
        var x = new Vector<double>(n);
        for (int j = 0; j < n; j++)
        {
            x[j] = j % zeroPeriod == 0 ? 0 : j % 2 == 1 ? j : -j;
        }

        var infinite = new Matrix<double>(m, n);
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < m; i++)
            {
                infinite[i, j] = (i + j) % 5 == 0 ? double.PositiveInfinity : (i + 2 * j) % 11 == 3 ? double.NaN : i - j;
            }
        }

        foreach (bool transposed in new[] { false, true })
        {
            // dA = dy·xᵀ。A = 0 なので Sqrt(Σ Ax) の随伴 dy はすべて ∞ で、x[j] = 0 の列は NaN。
            var tape = new Tape();
            var a = transposed ? tape.Leaf(new Matrix<double>(n, m)).T : tape.Leaf(new Matrix<double>(m, n));
            tape.Backward(XMath.Sqrt(XMath.Sum(a * tape.Constant(x))));
            var da = tape.GetGradient(a);
            for (int j = 0; j < n; j++)
            {
                for (int i = 0; i < m; i++)
                {
                    Assert.Equal(x[j] == 0 ? double.NaN : x[j] * double.PositiveInfinity, da[i, j]);
                }
            }

            // dx = Aᵀ·dy。dy = 0 なので、∞・NaN を含む列は NaN、ほかは 0。
            tape.Reset();
            var c = transposed ? tape.Constant(infinite.T.ToMatrix()).T : tape.Constant(infinite);
            var w = tape.Leaf(x);
            tape.Backward(XMath.Dot(tape.Constant(new Vector<double>(m)), c * w));
            var dw = tape.GetGradient(w);
            for (int j = 0; j < n; j++)
            {
                bool hasNonFinite = false;
                for (int i = 0; i < m; i++)
                {
                    hasNonFinite |= !double.IsFinite(infinite[i, j]);
                }

                Assert.Equal(hasNonFinite ? double.NaN : 0, dw[j]);
            }
        }
    }

    [Fact]
    public void EverySubviewOfSmallMatricesRecordsAndBackpropagates()
    {
        // 0〜3 行・列の行列のすべての部分矩形（空を含む）と、その転置。
        for (int rows = 0; rows <= 3; rows++)
        {
            for (int columns = 0; columns <= 3; columns++)
            {
                for (int r0 = 0; r0 <= rows; r0++)
                {
                    for (int r1 = r0; r1 <= rows; r1++)
                    {
                        for (int c0 = 0; c0 <= columns; c0++)
                        {
                            for (int c1 = c0; c1 <= columns; c1++)
                            {
                                CheckSubview(rows, columns, r0..r1, c0..c1, transposed: false);
                                CheckSubview(rows, columns, r0..r1, c0..c1, transposed: true);
                            }
                        }
                    }
                }
            }
        }

        static void CheckSubview(int rows, int columns, Range rowRange, Range columnRange, bool transposed)
        {
            var tape = new Tape();
            var storage = new Matrix<double>(rows, columns);
            for (int j = 0; j < columns; j++)
            {
                for (int i = 0; i < rows; i++)
                {
                    storage[i, j] = 10 * i + j + 1;
                }
            }

            var a = tape.Leaf(storage);
            var view = a[rowRange, columnRange];
            if (transposed)
            {
                view = view.T;
            }

            // loss = Σ V·x + Σ (V·B)·u。∂/∂V[p, q] = x[q] + (B·u)[q]。
            var x = new Vector<double>(view.Columns);
            var b = new Matrix<double>(view.Columns, 2);
            for (int q = 0; q < view.Columns; q++)
            {
                x[q] = q + 1;
                b[q, 0] = 2 * q - 1;
                b[q, 1] = 0.5 * q;
            }

            var u = new Vector<double>([1, -2]);
            ScalarNode loss = XMath.Sum(view * tape.Constant(x)) + XMath.Sum(view * tape.Constant(b) * tape.Constant(u));
            tape.Backward(loss);
            var bu = XMath.Multiply(b, u);
            var (r0, rowCount) = rowRange.GetOffsetAndLength(rows);
            var (c0, columnCount) = columnRange.GetOffsetAndLength(columns);
            var gradient = tape.GetGradient(a);
            for (int j = 0; j < columns; j++)
            {
                for (int i = 0; i < rows; i++)
                {
                    bool inside = i >= r0 && i < r0 + rowCount && j >= c0 && j < c0 + columnCount;
                    // 転置なら V[p, q] = A[r0 + q, c0 + p] なので、A[i, j] は V の列 i − r0 にあたる。
                    int q = transposed ? i - r0 : j - c0;
                    Assert.Equal(inside ? x[q] + bu[q] : 0, gradient[i, j]);
                }
            }
        }
    }

    [Fact]
    public void ParametersAccumulateUntilTheUserZeroesThem()
    {
        var w = Parameter<double>.Matrix(2, 2);
        var b = Parameter<double>.Vector(2);
        w.Value.CopyTo(w.Value);
        new double[] { 1, 2, 3, 4 }.CopyTo(w.Value);
        new double[] { 5, 6 }.CopyTo(b.Value);
        var parameters = new ParameterSet<double>(w, b);
        Assert.Equal(2, parameters.Count);
        Assert.Equal(6, parameters.Length);
        var tape = new Tape();
        for (int round = 1; round <= 2; round++)
        {
            tape.Reset();
            var x = tape.Constant(new Vector<double>([1, -1]));
            var wn = tape.UseMatrix(w);
            // 同じ重みを2回使うと、両方の寄与を足し込む。
            ScalarNode loss = XMath.Sum(wn * x) + XMath.Sum(tape.UseMatrix(w).T * x) + XMath.Dot(tape.UseVector(b), x);
            tape.Backward(loss);
            // ∂/∂w[i, j] = x[j] + x[i]、∂/∂b = x
            Assert.Equal(new double[] { 2 * round, 0, 0, -2 * round }, w.Gradient.ToArray());
            Assert.Equal(new double[] { round, -round }, b.Gradient.ToArray());
            Assert.Equal(new double[] { 2 * round, 0, 0, -2 * round }, tape.GetGradient(wn).ToArray().Cast<double>());
        }

        parameters.ZeroGradients();
        Assert.All(w.Gradient.ToArray().Concat(b.Gradient.ToArray()), value => Assert.Equal(0, value));
        Assert.Equal(new double[] { 1, 2, 3, 4 }, w.Value.ToArray());
    }

    [Fact]
    public void ParametersCheckTheirShapesAndSets()
    {
        var tape = new Tape();
        var vector = Parameter<float>.Vector(3);
        var matrix = Parameter<float>.Matrix(2, 3);
        Assert.Equal((1, 3, 1, 3), (vector.Rank, vector.Rows, vector.Columns, vector.Length));
        Assert.Equal((2, 2, 3, 6), (matrix.Rank, matrix.Rows, matrix.Columns, matrix.Length));
        Assert.Throws<ArgumentException>(() => tape.UseMatrix(vector));
        Assert.Throws<ArgumentException>(() => tape.UseVector(matrix));
        Assert.Throws<ArgumentNullException>(() => tape.UseVector<float>(null!));
        Assert.Throws<ArgumentException>(() => new ParameterSet<float>(vector, matrix, vector));
        Assert.Throws<ArgumentNullException>(() => new ParameterSet<float>(vector, null!));
        Assert.Throws<ArgumentOutOfRangeException>(() => Parameter<float>.Vector(-1));
        Assert.Throws<NotSupportedException>(() => Parameter<Half>.Vector(1));
        Assert.Throws<NotSupportedException>(() => tape.Constant(new Vector<Half>(1)));
        Assert.Equal([vector, matrix], new ParameterSet<float>(vector, matrix));
    }

    [Fact]
    public void ResetInvalidatesArrayNodesInEveryPublicUse()
    {
        var tape = new Tape();
        var x = tape.Leaf(new Vector<double>([1, 2]));
        var a = tape.Leaf(new Matrix<double>(2, 2));
        tape.Reset();
        var current = tape.Leaf(new Vector<double>([1, 2]));
        var currentMatrix = tape.Leaf(new Matrix<double>(2, 2));
        Assert.Throws<InvalidOperationException>(() => x.Value.Length);
        Assert.Throws<InvalidOperationException>(() => x.RequiresGradient);
        Assert.Throws<InvalidOperationException>(() => x[0..1]);
        Assert.Throws<InvalidOperationException>(() => a.Value.Rows);
        Assert.Throws<InvalidOperationException>(() => a.RequiresGradient);
        Assert.Throws<InvalidOperationException>(() => a.T);
        Assert.Throws<InvalidOperationException>(() => a[0..1, 0..1]);
        Assert.Throws<InvalidOperationException>(() => XMath.Sum(x));
        Assert.Throws<InvalidOperationException>(() => XMath.Dot(x, current));
        Assert.Throws<InvalidOperationException>(() => XMath.Dot(current, x));
        Assert.Throws<InvalidOperationException>(() => a * current);
        Assert.Throws<InvalidOperationException>(() => currentMatrix * x);
        Assert.Throws<InvalidOperationException>(() => a * currentMatrix);
        Assert.Throws<InvalidOperationException>(() => XMath.Multiply(currentMatrix, a));
        tape.Backward(XMath.Sum(current));
        Assert.Throws<InvalidOperationException>(() => tape.GetGradient(x));
        Assert.Throws<InvalidOperationException>(() => tape.GetGradient(a));
        Assert.Equal(new double[] { 1, 1 }, tape.GetGradient(current).ToArray());
    }

    [Fact]
    public void BorrowedScalarLeavesStayInvalidWhenAnArrayLeafReusesTheirPosition()
    {
        var tape = new Tape();
        var borrowed = tape.Leaves(new ReadOnlyVectorSpan<double>(new double[] { 2, 3 }));
        tape.Reset();
        tape.Leaf(new Vector<double>([4, 5]));
        ScalarNode old = borrowed[0];
        Assert.Throws<InvalidOperationException>(() => old.Value);
        Assert.Throws<InvalidOperationException>(() => old + 1);
    }

    [Fact]
    public void UninitializedAndForeignNodesAreRejected()
    {
        var tape = new Tape();
        var other = new Tape();
        var x = tape.Leaf(new Vector<double>([1, 2]));
        var y = other.Leaf(new Vector<double>([1, 2]));
        Assert.Throws<InvalidOperationException>(() => default(VectorNode<double>).Value.Length);
        Assert.Throws<InvalidOperationException>(() => default(MatrixNode<double>).T);
        Assert.Throws<InvalidOperationException>(() => XMath.Sum(default(VectorNode<double>)));
        Assert.Throws<InvalidOperationException>(() => XMath.Dot(x, default));
        Assert.Throws<InvalidOperationException>(() => XMath.Dot(x, y));
        Assert.Throws<InvalidOperationException>(() => other.Gradient(XMath.Sum(y), x, new Vector<double>(2)));
        tape.Backward(XMath.Sum(x));
        Assert.Throws<InvalidOperationException>(() => other.GetGradient(x));
    }

    [Fact]
    public void ShapesAreChecked()
    {
        var tape = new Tape();
        var a = tape.Leaf(new Matrix<double>(2, 3));
        Assert.Throws<ArgumentException>(() => a * a);
        Assert.Throws<ArgumentException>(() => a * tape.Leaf(new Vector<double>(2)));
        Assert.Throws<ArgumentException>(() => XMath.Dot(tape.Leaf(new Vector<double>(2)), tape.Leaf(new Vector<double>(3))));
        Assert.Throws<ArgumentOutOfRangeException>(() => a[0..3, ..]);
        Assert.Throws<ArgumentOutOfRangeException>(() => tape.Leaf(new Vector<double>(2))[1..3]);
    }

    [Fact]
    public void BackwardRunsOnceAndOnlyConstantsMayFollowIt()
    {
        var tape = new Tape();
        var x = tape.Leaf(new Vector<double>([1, 2]));
        var a = tape.Constant(Matrix<double>.FromRowMajor(2, 2, [1, 2, 3, 4]));
        var output = XMath.Sum(a * x);
        tape.Backward(output);
        Assert.Throws<InvalidOperationException>(() => tape.Backward(output));
        Assert.Throws<InvalidOperationException>(() => a * x);
        Assert.Throws<InvalidOperationException>(() => XMath.Sum(x));
        Assert.Throws<InvalidOperationException>(() => tape.Leaf(new Vector<double>(1)));
        Assert.Throws<InvalidOperationException>(() => tape.UseVector(Parameter<double>.Vector(1)));
        // 記録しない定数の演算は続けられる。
        var c = tape.Constant(new Vector<double>([1, 1]));
        Assert.Equal(new double[] { 3, 7 }, (a * c).Value.ToArray());
        Assert.Equal(2, XMath.Sum(c).Value);
    }

    [Fact]
    public void GetGradientRequiresBackward()
    {
        var tape = new Tape();
        var x = tape.Leaf(new Vector<double>([1, 2]));
        Assert.Throws<InvalidOperationException>(() => tape.GetGradient(x));
        Assert.Throws<InvalidOperationException>(() => tape.GetGradient(tape.Leaf(new Matrix<double>(1, 1))));
    }

    [Fact]
    public void EmptyShapesHaveEmptyOrZeroGradients()
    {
        var tape = new Tape();
        var a = tape.Leaf(new Matrix<double>(3, 0));
        var b = tape.Leaf(new Matrix<double>(0, 2));
        var x = tape.Leaf(new Vector<double>(0));
        var c = a * b;
        Assert.Equal((3, 2), (c.Rows, c.Columns));
        Assert.All(c.Value.ToArray().Cast<double>(), value => Assert.Equal(0, value));
        ScalarNode loss = XMath.Sum(c * tape.Constant(new Vector<double>([1, 2]))) + XMath.Sum(a * x) + XMath.Dot(x, x);
        Assert.Equal(0, loss.Value);
        double[] gradient = [];
        tape.Gradient(loss, new VectorSpan<double>(gradient));
        Assert.Equal(0, tape.GetGradient(a).Rows * tape.GetGradient(a).Columns);
        Assert.Equal(0, tape.GetGradient(x).Length);
    }

    [Fact]
    public void ParameterViewAccessorsShareBackingStorageAndValidateRank()
    {
        var vec = Parameter<float>.Vector(3);
        vec.AsVectorSpan()[1] = 2.5f;
        Assert.Equal(2.5f, vec.Value[1]);
        Assert.Throws<InvalidOperationException>(() => vec.AsMatrixSpan());
        Assert.Throws<InvalidOperationException>(() => vec.AsTensorSpan());

        var mat = Parameter<float>.Matrix(2, 3);
        var matSpan = mat.AsMatrixSpan();
        matSpan[1, 2] = 4.25f;
        Assert.Equal(4.25f, mat.Value[1 + 2 * 2]);
        Assert.Throws<InvalidOperationException>(() => mat.AsVectorSpan());
        Assert.Throws<InvalidOperationException>(() => mat.AsTensorSpan());

        var x = new Vector<float>([1f, 2f, 3f]);
        Vector<float> y = XMath.Multiply(mat.AsMatrixSpan(), x);
        Assert.Equal(12.75f, y[1]);

        var tensor = Parameter<double>.Tensor(2, 2, 3);
        tensor.AsTensorSpan()[0, 1, 2] = 7.5;
        Assert.Equal(7.5, tensor.Value[0 + 1 * 2 + 2 * 4]);
        Assert.Throws<InvalidOperationException>(() => tensor.AsVectorSpan());
        Assert.Throws<InvalidOperationException>(() => tensor.AsMatrixSpan());
    }
}
