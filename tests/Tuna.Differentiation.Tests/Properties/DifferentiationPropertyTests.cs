using CsCheck;
using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.Differentiation.Tests.Properties;

[Trait("Category", "Property")]
public class DifferentiationPropertyTests
{
    [Fact]
    public void ScalarGraph_ReverseModeGradientMatchesCentralFiniteDifferenceAndTapeReset()
    {
        var gen = Gen.Select(
            Gen.Int[2, 5],
            Gen.Int[3, 10],
            Gen.Int[0, 100000]);

        gen.Sample((varCount, opCount, seed) =>
        {
            var rng = new Random(seed);
            var inputs = new double[varCount];
            for (int i = 0; i < varCount; i++)
            {
                inputs[i] = rng.NextDouble() * 1.6 - 0.8;
            }

            // ランダムな計算グラフ（DAG）を固定の操作列として生成する
            var ops = new (int Kind, int Left, int Right, double Constant)[opCount];
            for (int step = 0; step < opCount; step++)
            {
                int nodeCount = varCount + step;
                int kind = rng.Next(0, 7);
                int left = rng.Next(0, nodeCount);
                int right = rng.Next(0, nodeCount);
                double constant = rng.NextDouble() * 1.5 - 0.75;
                ops[step] = (kind, left, right, constant);
            }

            var tape = new Tape();
            var adGrad1 = EvaluateAndDifferentiate(tape, inputs, ops);

            // Tape.Reset() 後の再利用で同じ結果が得られることも併せて検証する
            tape.Reset();
            var adGrad2 = EvaluateAndDifferentiate(tape, inputs, ops);
            for (int i = 0; i < varCount; i++)
            {
                Assert.Equal(adGrad1[i], adGrad2[i]);
            }

            // 中心有限差分による数値微分と比較する
            const double h = 1e-6;
            for (int i = 0; i < varCount; i++)
            {
                double orig = inputs[i];
                inputs[i] = orig + h;
                double fPlus = EvaluateScalarOnly(inputs, ops);
                inputs[i] = orig - h;
                double fMinus = EvaluateScalarOnly(inputs, ops);
                inputs[i] = orig;

                double numericGrad = (fPlus - fMinus) / (2.0 * h);
                double tol = 1e-5 * (Math.Abs(numericGrad) + Math.Abs(adGrad1[i]) + 1.0);
                Assert.True(
                    Math.Abs(adGrad1[i] - numericGrad) <= tol,
                    $"Gradient mismatch for input[{i}]: ad={adGrad1[i]}, fd={numericGrad}");
            }
        }, iter: 150, seed: "e5f6a1b2c3d4");
    }

    [Fact]
    public void MatrixAndVectorGraph_ReverseModeGradientMatchesCentralFiniteDifference()
    {
        var gen = Gen.Select(
            Gen.Int[1, 6],
            Gen.Int[1, 6],
            Gen.Int[1, 6],
            Gen.Int[0, 100000]);

        gen.Sample((m, k, n, seed) =>
        {
            var rng = new Random(seed);
            var a = Matrix<double>.Generate(m, k, (_, _) => rng.NextDouble() - 0.5);
            var b = Matrix<double>.Generate(k, n, (_, _) => rng.NextDouble() - 0.5);
            var w = Vector<double>.Generate(m, _ => rng.NextDouble() - 0.5);
            var v = Vector<double>.Generate(n, _ => rng.NextDouble() - 0.5);

            var tape = new Tape();
            var aNode = tape.Leaf(a);
            var bNode = tape.Leaf(b);
            var wNode = tape.Leaf(w);
            var vNode = tape.Leaf(v);

            // loss = Dot(w, (A * B) * v) + Dot(v, XMath.ColumnLogSumExp(A * B))
            var ab = aNode * bNode;
            var abv = ab * vNode;
            var lse = XMath.ColumnLogSumExp(ab);
            var loss = XMath.Dot(wNode, abv) + XMath.Dot(vNode, lse);

            tape.Backward(loss);
            var gradA = tape.GetGradient(aNode);
            var gradB = tape.GetGradient(bNode);
            var gradW = tape.GetGradient(wNode);
            var gradV = tape.GetGradient(vNode);

            const double h = 1e-6;
            for (int j = 0; j < k; j++)
            {
                for (int i = 0; i < m; i++)
                {
                    double orig = a[i, j];
                    a[i, j] = orig + h;
                    double fPlus = EvalMatrixLoss(a, b, w, v);
                    a[i, j] = orig - h;
                    double fMinus = EvalMatrixLoss(a, b, w, v);
                    a[i, j] = orig;

                    double fd = (fPlus - fMinus) / (2.0 * h);
                    Assert.True(Math.Abs(gradA[i, j] - fd) <= 1e-5 * (Math.Abs(fd) + 1.0));
                }
            }

            for (int j = 0; j < n; j++)
            {
                for (int i = 0; i < k; i++)
                {
                    double orig = b[i, j];
                    b[i, j] = orig + h;
                    double fPlus = EvalMatrixLoss(a, b, w, v);
                    b[i, j] = orig - h;
                    double fMinus = EvalMatrixLoss(a, b, w, v);
                    b[i, j] = orig;

                    double fd = (fPlus - fMinus) / (2.0 * h);
                    Assert.True(Math.Abs(gradB[i, j] - fd) <= 1e-5 * (Math.Abs(fd) + 1.0));
                }
            }

            for (int i = 0; i < m; i++)
            {
                double orig = w[i];
                w[i] = orig + h;
                double fPlus = EvalMatrixLoss(a, b, w, v);
                w[i] = orig - h;
                double fMinus = EvalMatrixLoss(a, b, w, v);
                w[i] = orig;

                double fd = (fPlus - fMinus) / (2.0 * h);
                Assert.True(Math.Abs(gradW[i] - fd) <= 1e-5 * (Math.Abs(fd) + 1.0));
            }

            for (int i = 0; i < n; i++)
            {
                double orig = v[i];
                v[i] = orig + h;
                double fPlus = EvalMatrixLoss(a, b, w, v);
                v[i] = orig - h;
                double fMinus = EvalMatrixLoss(a, b, w, v);
                v[i] = orig;

                double fd = (fPlus - fMinus) / (2.0 * h);
                Assert.True(Math.Abs(gradV[i] - fd) <= 1e-5 * (Math.Abs(fd) + 1.0));
            }
        }, iter: 80, seed: "f6a1b2c3d4e5");
    }

    private static double[] EvaluateAndDifferentiate(
        Tape tape,
        double[] inputs,
        (int Kind, int Left, int Right, double Constant)[] ops)
    {
        var nodes = new ScalarNode[inputs.Length + ops.Length];
        for (int i = 0; i < inputs.Length; i++)
        {
            nodes[i] = tape.Leaf(inputs[i]);
        }

        for (int step = 0; step < ops.Length; step++)
        {
            var (kind, left, right, c) = ops[step];
            ScalarNode l = nodes[left];
            ScalarNode r = nodes[right];
            nodes[inputs.Length + step] = kind switch
            {
                0 => l + r,
                1 => l - r,
                2 => l * r,
                3 => l / (XMath.Exp(r) + 1.0),
                4 => XMath.Sin(l + c),
                5 => XMath.Cos(l) + c,
                _ => XMath.Atan(l) * r
            };
        }

        ScalarNode output = nodes[^1];
        tape.Backward(output);

        var grads = new double[inputs.Length];
        for (int i = 0; i < inputs.Length; i++)
        {
            grads[i] = tape.GetGradient(nodes[i]);
        }

        return grads;
    }

    private static double EvaluateScalarOnly(
        double[] inputs,
        (int Kind, int Left, int Right, double Constant)[] ops)
    {
        var vals = new double[inputs.Length + ops.Length];
        Array.Copy(inputs, vals, inputs.Length);

        for (int step = 0; step < ops.Length; step++)
        {
            var (kind, left, right, c) = ops[step];
            double l = vals[left];
            double r = vals[right];
            vals[inputs.Length + step] = kind switch
            {
                0 => l + r,
                1 => l - r,
                2 => l * r,
                3 => l / (Math.Exp(r) + 1.0),
                4 => Math.Sin(l + c),
                5 => Math.Cos(l) + c,
                _ => Math.Atan(l) * r
            };
        }

        return vals[^1];
    }

    private static double EvalMatrixLoss(Matrix<double> a, Matrix<double> b, Vector<double> w, Vector<double> v)
    {
        var ab = a * b;
        var abv = ab * v;
        var lse = XMath.ColumnLogSumExp<double>(ab);
        return w.Dot(abv) + v.Dot(lse);
    }
}
