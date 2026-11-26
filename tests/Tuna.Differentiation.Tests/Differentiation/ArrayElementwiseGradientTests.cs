using System.Numerics;
using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.Tests.Differentiation;

/// <summary>要素演算を組み合わせたモデルの勾配（中心差分）、0 の局所微分・随伴の規則、スカラーのノードとの演算、ビューへの書き戻し</summary>
public class ArrayElementwiseGradientTests
{
    private const int Length = 37;

    [Fact]
    public void UnaryModelGradientMatchesCentralDifferences()
    {
        var random = new Random(21);
        var x = Uniform(random, Length, 0.2, 0.8);
        var c = Enumerable.Range(0, ElementwiseOperations.Unary.Length).Select(_ => ArrayProductTests.RandomVector<double>(random, Length)).ToArray();
        ScalarNode Model(Tape tape, VectorNode<double> v)
        {
            ScalarNode loss = 0;
            for (int k = 0; k < ElementwiseOperations.Unary.Length; k++)
            {
                string name = ElementwiseOperations.Unary[k];
                // ln(1 − eˣ) は負の入力で定義する。
                var input = name == "Log1mExp" ? -v : v;
                loss += XMath.Dot(tape.Constant(c[k]), ElementwiseOperations.Apply(name, input));
            }

            return loss;
        }

        AssertMatchesCentralDifferences(x, Model, 1e-7);
    }

    [Fact]
    public void BinaryModelGradientMatchesCentralDifferences()
    {
        // x と y は1つの葉の前半と後半の部分ベクトル。同じスロットへの寄与を足し込む経路を通る。
        var random = new Random(22);
        var z = Uniform(random, 2 * Length, 0.3, 0.7);
        for (int i = 0; i < Length; i++)
        {
            // 最小・最大の切り替わる点を差分の幅（約 10⁻⁴）より離す。
            z[Length + i] = z[i] + (random.Next(2) == 0 ? -1 : 1) * (0.05 + 0.1 * random.NextDouble());
        }

        var c = Enumerable.Range(0, ElementwiseOperations.Binary.Length).Select(_ => ArrayProductTests.RandomVector<double>(random, Length)).ToArray();
        ScalarNode Model(Tape tape, VectorNode<double> v)
        {
            var x = v[..Length];
            var y = v[Length..];
            ScalarNode loss = 0;
            for (int k = 0; k < ElementwiseOperations.Binary.Length; k++)
            {
                string name = ElementwiseOperations.Binary[k];
                // ln(eˣ − eʸ) は x > y で定義する。
                var left = name == "LogSubExp" ? x + 2 : x;
                loss += XMath.Dot(tape.Constant(c[k]), ElementwiseOperations.Apply(name, left, y));
            }

            // 同じ入力どうしの演算（両方の寄与が同じ要素に入る）。
            return loss + XMath.Sum(XMath.Hadamard(x, x)) + XMath.Dot(tape.Constant(c[0]), XMath.Divide(y, y + x));
        }

        AssertMatchesCentralDifferences(z, Model, 1e-7);
    }

    [Fact]
    public void ScalarOperandModelGradientMatchesCentralDifferences()
    {
        // 最後の要素をスカラーのノードとして使う。
        var random = new Random(23);
        var z = Uniform(random, Length + 1, 0.2, 0.8);
        z[Length] = 0.5;
        for (int i = 0; i < Length; i++)
        {
            // 最小・最大の切り替わる点（x = s）を差分の幅より離す。
            if (Math.Abs(z[i] - 0.5) < 0.02)
            {
                z[i] += 0.05;
            }
        }

        var c = Enumerable.Range(0, ElementwiseOperations.WithScalar.Length).Select(_ => ArrayProductTests.RandomVector<double>(random, Length)).ToArray();
        ScalarNode Model(Tape tape, VectorNode<double> v)
        {
            var x = v[..Length];
            ScalarNode s = XMath.Sum(v[Length..]);
            ScalarNode loss = 0;
            for (int k = 0; k < ElementwiseOperations.WithScalar.Length; k++)
            {
                string name = ElementwiseOperations.WithScalar[k];
                var left = name == "LogSubExp" ? x + 2 : x;
                loss += XMath.Dot(tape.Constant(c[k]), ElementwiseOperations.Apply(name, left, s));
            }

            // スカラーの式を通したスカラー（s² と 1/s）も使う。
            return loss + XMath.Sum(x * (s * s)) + XMath.Dot(tape.Constant(c[1]), 1 / s - x);
        }

        AssertMatchesCentralDifferences(z, Model, 1e-7);
    }

    [Fact]
    public void ViewsOfMatricesReceiveTheirGradientsInStorageOrder()
    {
        // 同じ格納の、転置の部分行列（論理的に 3×5）と、列優先の部分行列（3×5、重なりあり）。
        var random = new Random(24);
        var storage = Uniform(random, 6 * 7, 0.2, 0.8);
        var c = ArrayProductTests.RandomVector<double>(random, 5);
        var d = ArrayProductTests.RandomVector<double>(random, 5);
        double Value(Matrix<double> s)
        {
            var tape = new Tape();
            return Evaluate(tape, tape.Constant(s)).Value;
        }

        ScalarNode Evaluate(Tape tape, MatrixNode<double> s)
        {
            var x = s[1..6, 2..5].T;
            var y = s[..3, 1..6];
            var cn = tape.Constant(c);
            var dn = tape.Constant(d);
            return XMath.Sum(XMath.Hadamard(XMath.Exp(x), y) * cn) + XMath.Sum(XMath.Divide(x, y + 1) * dn)
                + XMath.Sum(XMath.Logistic(x - y) * cn) + XMath.Sum(XMath.Pow(y, x) * dn) + XMath.Sum(XMath.Multiply(XMath.Log(x), XMath.Sum(y * cn)) * dn);
        }

        var matrix = new Matrix<double>(6, 7);
        storage.AsSpan().CopyTo(matrix.AsSpan());
        var gradientTape = new Tape();
        var leaf = gradientTape.Leaf(matrix);
        gradientTape.Backward(Evaluate(gradientTape, leaf));
        var gradient = gradientTape.GetGradient(leaf).ToMatrix();
        for (int i = 0; i < 6; i++)
        {
            for (int j = 0; j < 7; j++)
            {
                double saved = matrix[i, j];
                double numeric = DifferentiationAssert.Richardson(t =>
                {
                    matrix[i, j] = t;
                    double value = Value(matrix);
                    matrix[i, j] = saved;
                    return value;
                }, saved);
                DifferentiationAssert.Close(numeric, gradient[i, j], 1e-7, 1e-9, $"dS[{i}, {j}]");
            }
        }
    }

    [Fact]
    public void SingleGradientsAgreeWithDoubleGradients()
    {
        var random = new Random(25);
        var z = Uniform(random, 2 * Length + 1, 0.2, 0.8);
        var c = ArrayProductTests.RandomVector<double>(random, Length);
        var expected = Gradient<double>(z, c);
        var actual = Gradient<float>(z.ConvertChecked<float>(), c.ConvertChecked<float>());
        for (int i = 0; i < z.Length; i++)
        {
            DifferentiationAssert.Close(expected[i], actual[i], 5e-5, 5e-5, $"gradient[{i}]");
        }

        static double[] Gradient<T>(LinearAlgebra.Vector<T> values, LinearAlgebra.Vector<T> weights) where T : unmanaged, IFloatingPointIeee754<T>
        {
            var tape = new Tape();
            var v = tape.Leaf(values);
            var x = v[..Length];
            var y = v[Length..^1];
            ScalarNode s = XMath.Sum(v[^1..]);
            var w = tape.Constant(weights);
            var u = XMath.Softplus(XMath.Hadamard(x, y) - s) + XMath.Sqrt(XMath.Divide(x, y)) + XMath.Log1p(XMath.Pow(x, s)) - XMath.Max(x, y) * s;
            ScalarNode loss = XMath.Dot(w, u) + XMath.Sum(XMath.LogLogistic(-x)) + XMath.Sum(XMath.Logit(XMath.Min(x, y)));
            var gradient = new double[values.Length];
            tape.Gradient(loss, new VectorSpan<double>(gradient));
            return gradient;
        }
    }

    [Fact]
    public void ZeroAdjointsDoNotMultiplyInfiniteLocalDerivatives()
    {
        // 随伴が 0 の要素には、局所微分が ±∞ でも寄与を足さない（0·∞ の NaN を作らない）。
        var tape = new Tape();
        var x = tape.Leaf(new LinearAlgebra.Vector<double>([0.0, 4.0, -0.0, 1.0]));
        var p = tape.Leaf(new LinearAlgebra.Vector<double>([0.0, 0.5, 1.0, -1.0]));
        var weights = tape.Constant(new LinearAlgebra.Vector<double>([0.0, 1.0, 0.0, 1.0]));
        ScalarNode loss = XMath.Dot(weights, XMath.Sqrt(x)) + XMath.Dot(weights, XMath.Log(x)) + XMath.Dot(weights, XMath.Reciprocal(x))
            + XMath.Dot(weights, XMath.Logit(p)) + XMath.Dot(weights, XMath.Log1p(p));
        tape.Backward(loss);
        // dx = 1/(2√x) + 1/x − 1/x²、dp = 1/(p(1 − p)) + 1/(1 + p)（重みが 0 の要素は 0）。
        Assert.Equal(new[] { 0.0, 0.25 + 0.25 - 1.0 / 16, 0.0, 0.5 + 1 - 1 }, tape.GetGradient(x).ToArray());
        Assert.Equal(0.0, tape.GetGradient(p)[0]);
        Assert.Equal(4 + 1 / 1.5, tape.GetGradient(p)[1], 15);
        Assert.Equal(0.0, tape.GetGradient(p)[2]);
        // p = −1 は Logit の定義域の外（NaN）で、Log1p の局所微分は +∞。
        Assert.True(double.IsNaN(tape.GetGradient(p)[3]));
    }

    [Fact]
    public void ZeroLocalDerivativesDoNotPassNonFiniteAdjoints()
    {
        // 局所微分がちょうど 0 の要素には、随伴が NaN・∞ でも寄与を足さない（スカラーの記録と同じ）。
        double[] xs = [1.0, 2.0, 0.0, 1.0, 5.0];
        double[] ys = [0.0, 3.0, 4.0, 1.0, 2.0];
        double[] weights = [double.PositiveInfinity, 1.0, double.NaN, double.NegativeInfinity, double.NaN];
        foreach (bool transposed in new[] { false, true })
        {
            var tape = new Tape();
            var x = Leaf(tape, xs, transposed);
            var y = Leaf(tape, ys, transposed);
            var w = tape.Constant(new LinearAlgebra.Vector<double>(weights));
            var s = tape.Leaf(0.0);
            var u = tape.Leaf(-1.0);
            // 要素積（相手が 0 の要素）、最小（選ばれなかった側）、スカラー倍（s = 0 と、x が 0 の要素）、絶対値の右側の局所微分。
            ScalarNode loss = XMath.Sum(XMath.Hadamard(x, y) * w) + XMath.Sum(XMath.Min(x, y) * w) + XMath.Sum(XMath.Multiply(x, s) * w)
                + XMath.Sum(XMath.Max(x, u) * w);
            tape.Backward(loss);
            var expected = ScalarModel(xs, ys, weights);
            var dx = tape.GetGradient(x);
            var dy = tape.GetGradient(y);
            for (int i = 0; i < xs.Length; i++)
            {
                DifferentiationAssert.SameNumber(expected.Dx[i], dx[0, i], $"dx[{i}] transposed={transposed}");
                DifferentiationAssert.SameNumber(expected.Dy[i], dy[0, i], $"dy[{i}] transposed={transposed}");
            }

            DifferentiationAssert.SameNumber(expected.Ds, tape.GetGradient(s), "ds");
            DifferentiationAssert.SameNumber(expected.Du, tape.GetGradient(u), "du");
        }

        // 同じ式をスカラーの記録で書いた勾配。
        static (double[] Dx, double[] Dy, double Ds, double Du) ScalarModel(double[] xs, double[] ys, double[] weights)
        {
            var tape = new Tape();
            var x = tape.Leaves(xs).ToArray();
            var y = tape.Leaves(ys).ToArray();
            var s = tape.Leaf(0.0);
            var u = tape.Leaf(-1.0);
            ScalarNode loss = 0;
            for (int i = 0; i < xs.Length; i++)
            {
                loss += (x[i] * y[i]) * weights[i] + XMath.Min(x[i], y[i]) * weights[i] + (x[i] * s) * weights[i] + XMath.Max(x[i], u) * weights[i];
            }

            tape.Backward(loss);
            return (x.Select(tape.GetGradient).ToArray(), y.Select(tape.GetGradient).ToArray(), tape.GetGradient(s), tape.GetGradient(u));
        }

        static MatrixNode<double> Leaf(Tape tape, double[] values, bool transposed)
        {
            // 1×5 の行（論理的な形）を、列優先の格納の部分行列か、転置の格納（5×1 の部分行列の転置）に置く。
            var storage = transposed ? new Matrix<double>(values.Length + 2, 2) : new Matrix<double>(3, values.Length);
            for (int i = 0; i < values.Length; i++)
            {
                if (transposed)
                {
                    storage[i + 1, 1] = values[i];
                }
                else
                {
                    storage[1, i] = values[i];
                }
            }

            var leaf = tape.Leaf(storage);
            return transposed ? leaf[1..(values.Length + 1), 1..2].T : leaf[1..2, ..];
        }
    }

    [Fact]
    public void ScalarOperandGradientsAreSumsOfTheElementContributions()
    {
        var random = new Random(26);
        var x = ArrayProductTests.RandomVector<double>(random, 101);
        var xf = x.ConvertChecked<float>();
        var tape = new Tape();
        var s = tape.Leaf(1.25);
        var xn = tape.Leaf(x);
        var xc = tape.Constant(xf);
        // 定数の配列と記録つきのスカラーの演算も記録し、s の勾配だけを求める。float の配列では float の和を double に広げる。
        var product = XMath.Multiply(xc, s);
        Assert.True(product.RequiresGradient);
        ScalarNode loss = XMath.Sum(XMath.Multiply(xn, s)) + XMath.Sum(product) + XMath.Sum(s - xn) + XMath.Sum(xn + 2.0);
        tape.Backward(loss);
        Assert.Equal(XMath.Sum(x) + XMath.Sum(xf) + 101, tape.GetGradient(s), 12);
        Assert.Equal((double)XMath.Sum(xf), FloatContribution(xf), 0);
        var dx = tape.GetGradient(xn);
        for (int i = 0; i < x.Length; i++)
        {
            Assert.Equal(1.25 - 1 + 1, dx[i]);
        }

        static double FloatContribution(LinearAlgebra.Vector<float> values)
        {
            var tape = new Tape();
            var s = tape.Leaf(3.0);
            tape.Backward(XMath.Sum(XMath.Multiply(tape.Constant(values), s)));
            return tape.GetGradient(s);
        }
    }

    [Fact]
    public void ConstantScalarsAreNotDifferentiated()
    {
        var tape = new Tape();
        var x = tape.Leaf(new LinearAlgebra.Vector<double>([1.0, 2.0, 3.0]));
        var y = XMath.Pow(x, 2.0) - 1.0;
        tape.Backward(XMath.Sum(y));
        Assert.Equal(new[] { 2.0, 4.0, 6.0 }, tape.GetGradient(x).ToArray());

        var constants = new Tape();
        var c = constants.Constant(new LinearAlgebra.Vector<double>([1.0, 2.0]));
        Assert.False(XMath.Pow(c, 2.0).RequiresGradient);
        Assert.False((c * 3.0).RequiresGradient);
    }

    [Fact]
    public void InvalidOperandsThrow()
    {
        var tape = new Tape();
        var other = new Tape();
        var x = tape.Leaf(new LinearAlgebra.Vector<double>(3));
        var y = other.Leaf(new LinearAlgebra.Vector<double>(3));
        var a = tape.Leaf(new Matrix<double>(2, 3));
        var s = other.Leaf(1.0);
        Assert.Throws<InvalidOperationException>(() => x + y);
        Assert.Throws<InvalidOperationException>(() => XMath.Pow(x, s));
        Assert.Throws<InvalidOperationException>(() => s - a);
        Assert.Throws<ArgumentException>(() => x - tape.Leaf(new LinearAlgebra.Vector<double>(4)));
        Assert.Throws<ArgumentException>(() => XMath.Hadamard(a, a.T));
        Assert.Throws<ArgumentException>(() => XMath.Divide(a, tape.Leaf(new Matrix<double>(3, 2))));

        // Reset より前のスカラーのノード。
        var stale = tape.Leaf(2.0);
        tape.Reset();
        var z = tape.Leaf(new LinearAlgebra.Vector<double>(3));
        Assert.Throws<InvalidOperationException>(() => z * stale);
        Assert.Throws<InvalidOperationException>(() => XMath.Exp(x));

        // 逆伝播の後は、記録する演算は例外、定数だけの演算は値を返す。
        var c = tape.Constant(new LinearAlgebra.Vector<double>([1.0, 2.0, 3.0]));
        tape.Backward(XMath.Sum(z));
        Assert.Throws<InvalidOperationException>(() => XMath.Exp(z));
        Assert.Throws<InvalidOperationException>(() => c * tape.Leaf(1.0));
        Assert.Equal(XMath.Exp(new LinearAlgebra.Vector<double>([1.0, 2.0, 3.0])).ToArray(), XMath.Exp(c).Value.ToArray());
    }

    [Fact]
    public void UnreachedElementwiseRecordsAreSkipped()
    {
        // 出力から届かない記録は、∞ の局所微分を持っていても勾配に寄与しない。
        var tape = new Tape();
        var x = tape.Leaf(new LinearAlgebra.Vector<double>([0.0, 1.0]));
        _ = XMath.Log(x);
        _ = XMath.Reciprocal(x) * tape.Leaf(double.NaN);
        tape.Backward(XMath.Sum(x));
        Assert.Equal(new[] { 1.0, 1.0 }, tape.GetGradient(x).ToArray());
    }

    [Fact]
    public void LogisticRegressionGradientMatchesTheAnalyticGradient()
    {
        // 負の対数尤度 Σ softplus(η) − yᵀη（η = Xᵀw）の勾配は X(σ(η) − y)。
        foreach (int samples in new[] { 16, 500 })
        {
            var random = new Random(samples);
            const int features = 5;
            var x = ArrayProductTests.RandomMatrix<double>(random, features, samples);
            var y = new LinearAlgebra.Vector<double>(samples);
            for (int i = 0; i < samples; i++)
            {
                y[i] = random.NextDouble() < 0.5 ? 0 : 1;
            }

            var theta = ArrayProductTests.RandomVector<double>(random, features);
            var tape = new Tape();
            var w = tape.Leaf(theta);
            var eta = tape.Constant(x).T * w;
            ScalarNode loss = XMath.Sum(XMath.Softplus(eta)) - XMath.Dot(tape.Constant(y), eta);
            var gradient = new double[features];
            double value = tape.Gradient(loss, new VectorSpan<double>(gradient));
            var etaValues = XMath.Multiply(x.T, theta);
            var expected = XMath.Multiply(x, XMath.Subtract(XMath.Logistic(etaValues), y));
            DifferentiationAssert.Close(XMath.Sum(XMath.Softplus(etaValues)) - XMath.Dot(y, etaValues), value, 1e-13, 1e-13, "value");
            for (int i = 0; i < features; i++)
            {
                DifferentiationAssert.Close(expected[i], gradient[i], 1e-12, 1e-12 * samples, $"gradient[{i}]");
            }
        }
    }

    internal static void AssertMatchesCentralDifferences(LinearAlgebra.Vector<double> point, Func<Tape, VectorNode<double>, ScalarNode> model, double tolerance)
    {
        var tape = new Tape();
        var leaf = tape.Leaf(point);
        tape.Backward(model(tape, leaf));
        var gradient = tape.GetGradient(leaf).ToArray();
        var values = new Tape();
        for (int i = 0; i < point.Length; i++)
        {
            double saved = point[i];
            double numeric = DifferentiationAssert.Richardson(t =>
            {
                point[i] = t;
                values.Reset();
                double value = model(values, values.Constant(point)).Value;
                point[i] = saved;
                return value;
            }, saved);
            DifferentiationAssert.Close(numeric, gradient[i], tolerance, 1e-9, $"gradient[{i}]");
        }
    }

    private static LinearAlgebra.Vector<double> Uniform(Random random, int length, double low, double high)
    {
        var result = new LinearAlgebra.Vector<double>(length);
        for (int i = 0; i < length; i++)
        {
            result[i] = low + (high - low) * random.NextDouble();
        }

        return result;
    }
}
