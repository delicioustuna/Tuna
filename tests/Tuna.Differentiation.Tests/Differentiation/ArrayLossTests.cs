using System.Numerics;
using Tuna.Differentiation;
using Tuna.LinearAlgebra;
using static Tuna.Tests.Differentiation.ElementwiseOperations;

namespace Tuna.Tests.Differentiation;

/// <summary>
/// 損失 SoftmaxCrossEntropy（ラベル・目標の分布、行列とベクトル）と LogisticCrossEntropy（ベクトルと行列）。
/// 値は XML コメントに書いたコアの関数の組み合わせと同じビット、勾配は文書の式を XMath で求めたものと同じビット。
/// 特別な値・裾・空・例外、中心差分、float と double の一致、多項・2値のロジスティック回帰の解析的な勾配。
/// </summary>
public class ArrayLossTests
{
    // 損失に掛ける係数（逆伝播の随伴になる）。2 の冪でない値にして、随伴の丸めを確かめる。
    private const double Coefficient = 0.3;

    [Fact]
    public void LabelValuesFollowTheDocumentedFormulaBitForBit()
    {
        LabelValues<double>();
        LabelValues<float>();
    }

    [Fact]
    public void TargetValuesFollowTheDocumentedFormulaBitForBit()
    {
        TargetValues<double>();
        TargetValues<float>();
    }

    [Fact]
    public void LogisticValuesFollowTheDocumentedFormulaBitForBit()
    {
        LogisticValues<double>();
        LogisticValues<float>();
    }

    [Fact]
    public void DoubleSoftplusTermsEqualXMathSoftplus()
    {
        // 文書の softplus(±η)（η + s・s・s − η）は、double では XMath.Softplus(±η) と同じビット。
        double[] inputs = Inputs(91, 200, 60);
        var negative = new LinearAlgebra.Vector<double>(inputs.Length);
        for (int i = 0; i < inputs.Length; i++)
        {
            negative[i] = -Math.Abs(inputs[i]);
        }

        var s = XMath.Softplus<double>(negative);
        var plus = XMath.Softplus<double>(new LinearAlgebra.Vector<double>(inputs));
        var minus = XMath.Softplus<double>(new LinearAlgebra.Vector<double>(inputs.Select(x => -x).ToArray()));
        for (int i = 0; i < inputs.Length; i++)
        {
            double x = inputs[i];
            DifferentiationAssert.SameNumber(plus[i], x >= 0 ? x + s[i] : s[i], $"softplus({x:R})");
            DifferentiationAssert.SameNumber(minus[i], x >= 0 ? s[i] : s[i] - x, $"softplus(-{x:R})");
        }
    }

    [Fact]
    public void LabelGradientsFollowTheDocumentedFormulaBitForBit()
    {
        LabelGradients<double>();
        LabelGradients<float>();
    }

    [Fact]
    public void TargetGradientsFollowTheDocumentedFormulaBitForBit()
    {
        TargetGradients<double>();
        TargetGradients<float>();
    }

    [Fact]
    public void LogisticGradientsFollowTheDocumentedFormulaBitForBit()
    {
        LogisticGradients<double>();
        LogisticGradients<float>();
    }

    [Fact]
    public void VectorVersionsEqualSingleColumnMatrices()
    {
        // ベクトルの SoftmaxCrossEntropy は1列の行列の版と同じ値と勾配。
        var data = Uniform<double>(new Random(92), 7, 1, -3, 3).AsMatrixSpan().Column(0).ToVector();
        var soft = Uniform<double>(new Random(93), 7, 1, 0, 1).AsMatrixSpan().Column(0).ToVector();
        soft[2] = 0;
        for (int form = 0; form < 2; form++)
        {
            var tape = new Tape();
            var v = tape.Leaf(data);
            var t = tape.Leaf(soft);
            var loss = form == 0 ? XMath.SoftmaxCrossEntropy(v, 4) : XMath.SoftmaxCrossEntropy(v, t);
            tape.Backward(loss);
            var matrix = new Tape();
            var m = matrix.Leaf(data);
            var u = matrix.Leaf(soft);
            var reference = form == 0 ? XMath.SoftmaxCrossEntropy(m.Broadcast(1), [4]) : XMath.SoftmaxCrossEntropy(m.Broadcast(1), u.Broadcast(1));
            matrix.Backward(reference);
            DifferentiationAssert.EqualBits(reference.Value, loss.Value, $"form {form} value");
            AssertSameBits<double>(matrix.GetGradient(m).ToArray(), tape.GetGradient(v).ToArray(), $"form {form} logits");
            AssertSameBits<double>(matrix.GetGradient(u).ToArray(), tape.GetGradient(t).ToArray(), $"form {form} targets");
            DifferentiationAssert.EqualBits(0 - XMath.LogSoftmax<double>(data)[4], XMath.SoftmaxCrossEntropy(new Tape().Constant(data), 4).Value, "single label");
        }
    }

    [Fact]
    public void OneHotTargetsMatchLabelsBitForBit()
    {
        // one-hot の目標の分布は、ラベルの版と値も勾配も同じビット。
        var data = Uniform<double>(new Random(94), 6, 40, -5, 5);
        int[] labels = RandomLabels(new Random(95), 6, 40);
        var oneHot = new Matrix<double>(6, 40);
        for (int n = 0; n < labels.Length; n++)
        {
            oneHot[labels[n], n] = 1;
        }

        foreach (bool single in new[] { false, true })
        {
            if (single)
            {
                Compare(data.ConvertChecked<float>(), oneHot.ConvertChecked<float>());
            }
            else
            {
                Compare(data, oneHot);
            }
        }

        void Compare<T>(Matrix<T> z, Matrix<T> t) where T : unmanaged, IFloatingPointIeee754<T>
        {
            var tape = new Tape();
            var a = tape.Leaf(z);
            var la = XMath.SoftmaxCrossEntropy(a, labels);
            tape.Backward(Coefficient * la);
            var other = new Tape();
            var b = other.Leaf(z);
            var lb = XMath.SoftmaxCrossEntropy(b, other.Constant(t));
            other.Backward(Coefficient * lb);
            DifferentiationAssert.EqualBits(la.Value, lb.Value, typeof(T).Name);
            AssertSameBits<T>(tape.GetGradient(a).ToMatrix().AsSpan(), other.GetGradient(b).ToMatrix().AsSpan(), typeof(T).Name);
        }
    }

    [Fact]
    public void SoftmaxSpecialColumnsFollowTheCoreLogSoftmax()
    {
        // 列ごとに: 有限、ラベルの要素が −∞（値 +∞、勾配は有限）、ほかの要素が −∞、最大値が +∞ でラベルが有限の要素（値 +∞）、
        // ラベルが +∞ の要素（値 NaN）、すべて −∞（値 NaN）、NaN を含む（値・勾配とも NaN）。
        // 値はコアの対数ソフトマックス、勾配は P − onehot で、P は ln Σ exp の局所微分（最大値が有限でない列は極限の重み）。
        double inf = double.PositiveInfinity, nan = double.NaN;
        double[][] columns = [[1, 2, 3], [-inf, 0, 1], [-inf, 0, 1], [inf, 0, 1], [inf, 0, inf], [-inf, -inf, -inf], [nan, 0, 1]];
        int[] labels = [2, 0, 1, 1, 0, 0, 2];
        double third = 1.0 / 3;
        double[] values = [Math.Log(Math.Exp(1) + Math.Exp(2) + Math.Exp(3)) - 3, inf, Math.Log(1 + Math.E), inf, nan, nan, nan];
        double[][] gradients = [[], [], [], [1, -1, 0], [-0.5, 0, 0.5], [third - 1, third, third], [nan, nan, nan]];
        var data = new Matrix<double>(3, columns.Length);
        for (int j = 0; j < columns.Length; j++)
        {
            for (int i = 0; i < 3; i++)
            {
                data[i, j] = columns[j][i];
            }
        }

        for (int j = 0; j < columns.Length; j++)
        {
            var tape = new Tape();
            var x = tape.Leaf(data);
            var loss = XMath.SoftmaxCrossEntropy(x[.., j..(j + 1)], labels.AsSpan(j, 1));
            tape.Backward(loss);
            var gradient = tape.GetGradient(x);
            string context = $"column {j}";
            if (double.IsFinite(values[j]))
            {
                DifferentiationAssert.Close(values[j], loss.Value, 1e-15, 0, context);
            }
            else
            {
                DifferentiationAssert.SameNumber(values[j], loss.Value, context);
            }

            for (int i = 0; i < 3; i++)
            {
                if (gradients[j].Length > 0)
                {
                    DifferentiationAssert.SameNumber(gradients[j][i], gradient[i, j], $"{context} [{i}]");
                    continue;
                }

                // 最大値が有限の列は exp(L) から 1 を引く。
                double lse = j == 0 ? Math.Log(Math.Exp(1) + Math.Exp(2) + Math.Exp(3)) : Math.Log(1 + Math.E);
                double p = Math.Exp(data[i, j] - lse);
                DifferentiationAssert.Close(p - (i == labels[j] ? 1 : 0), gradient[i, j], 1e-15, 1e-16, $"{context} [{i}]");
            }

            // ほかの列の勾配は 0。
            for (int k = 0; k < columns.Length; k++)
            {
                for (int i = 0; k != j && i < 3; i++)
                {
                    Assert.Equal(0.0, gradient[i, k]);
                }
            }
        }

        // 目標の分布では、T が 0 の要素の L = −∞ を +0 の項にする。T が正なら値は +∞、dT は +∞。
        var target = new Tape();
        var z = target.Leaf(new LinearAlgebra.Vector<double>([-inf, 0, 1]));
        var masked = target.Leaf(new LinearAlgebra.Vector<double>([0, 0.5, 0.5]));
        var positive = target.Leaf(new LinearAlgebra.Vector<double>([0.25, 0.25, 0.5]));
        var finite = XMath.SoftmaxCrossEntropy(z, masked);
        var infinite = XMath.SoftmaxCrossEntropy(z, positive);
        target.Backward(finite + infinite);
        double l1 = -Math.Log(1 + Math.E), l2 = 1 - Math.Log(1 + Math.E);
        DifferentiationAssert.Close(-0.5 * l1 - 0.5 * l2, finite.Value, 1e-15, 0, "masked value");
        Assert.Equal(inf, infinite.Value);
        foreach (var t in new[] { masked, positive })
        {
            var dt = target.GetGradient(t);
            Assert.Equal(inf, dt[0]);
            DifferentiationAssert.Close(-l1, dt[1], 1e-15, 0, "dT[1]");
            DifferentiationAssert.Close(-l2, dt[2], 1e-15, 0, "dT[2]");
        }

        var dz = target.GetGradient(z);
        Assert.Equal(-0.25, dz[0]);
        DifferentiationAssert.Close(2 * Math.Exp(l1) - 0.75, dz[1], 1e-15, 1e-16, "dz[1]");
        DifferentiationAssert.Close(2 * Math.Exp(l2) - 1, dz[2], 1e-15, 1e-16, "dz[2]");
    }

    [Fact]
    public void LogisticTailsAndInfinitiesFollowTheLimits()
    {
        // 損失の小さい裾（y = 1 で η が大きい、y = 0 で η が小さい）も相対誤差で求める。
        double tail = Math.Exp(-40);
        var tape = new Tape();
        var eta = tape.Leaf(new LinearAlgebra.Vector<double>([40, -40]));
        var y = tape.Leaf(new LinearAlgebra.Vector<double>([1, 0]));
        var loss = XMath.LogisticCrossEntropy(eta, y);
        tape.Backward(loss);
        DifferentiationAssert.Close(2 * tail, loss.Value, 1e-15, 0, "value");
        DifferentiationAssert.Close(-tail, tape.GetGradient(eta)[0], 1e-15, 0, "d eta[0]");
        DifferentiationAssert.Close(tail, tape.GetGradient(eta)[1], 1e-15, 0, "d eta[1]");
        Assert.Equal([-40.0, 40], tape.GetGradient(y).ToArray());

        var single = new Tape();
        var f = single.Leaf(new LinearAlgebra.Vector<float>([20f, -20f]));
        var g = single.Constant(new LinearAlgebra.Vector<float>([1f, 0f]));
        var singleLoss = XMath.LogisticCrossEntropy(f, g);
        single.Backward(singleLoss);
        DifferentiationAssert.Close(2 * Math.Exp(-20), singleLoss.Value, 1e-6, 0, "float value");
        DifferentiationAssert.Close(-Math.Exp(-20), single.GetGradient(f)[0], 1e-6, 0, "float d eta[0]");

        // η = ±∞ でも 0·∞ にならない。y = 1、η = +∞ と y = 0、η = −∞ は 0、外れた側は +∞。NaN は NaN。
        double inf = double.PositiveInfinity;
        (double Eta, double Y, double Value, double Derivative)[] cases =
        [
            (inf, 1, 0, 0), (-inf, 0, 0, 0), (inf, 0, inf, 1), (-inf, 1, inf, -1), (inf, 0.5, inf, 0.5), (-inf, 0.25, inf, -0.25),
            (double.NaN, 1, double.NaN, double.NaN),
        ];
        foreach (var (e, t, value, derivative) in cases)
        {
            var limit = new Tape();
            var x = limit.Leaf(new LinearAlgebra.Vector<double>([e]));
            var target = limit.Leaf(new LinearAlgebra.Vector<double>([t]));
            var result = XMath.LogisticCrossEntropy(x, target);
            limit.Backward(result);
            string context = $"eta {e}, y {t}";
            DifferentiationAssert.SameNumber(value, result.Value, $"{context} value");
            DifferentiationAssert.SameNumber(derivative, limit.GetGradient(x)[0], $"{context} d eta");
            DifferentiationAssert.SameNumber(-e, limit.GetGradient(target)[0], $"{context} dy");
        }
    }

    [Fact]
    public void SpecialValuesGiveTheSameBitsInEveryLane()
    {
        // 損失の要素ごとのループは、SIMD のループと端数のスカラーのループに同じ式を書いている。
        // 特別な値（±∞・±0・NaN・裾）を各レーンと端数に置いた長い配列の結果を、1要素ずつ（スカラーのループだけを通る）求めたものと比べる。
        Run<double>();
        Run<float>();

        static void Run<T>() where T : unmanaged, IFloatingPointIeee754<T>
        {
            T[] logits = [T.PositiveInfinity, T.NegativeInfinity, T.Zero, T.NegativeZero, T.NaN, T.CreateTruncating(40), T.CreateTruncating(-40), T.One, -T.One,
                T.CreateTruncating(1e-30), T.CreateTruncating(-3.5)];
            T[] targets = [T.Zero, T.One, T.CreateTruncating(0.5), T.CreateTruncating(0.25), T.NegativeZero, T.CreateTruncating(0.75), T.One];
            const int Length = 67;
            var eta = new LinearAlgebra.Vector<T>(Length);
            var y = new LinearAlgebra.Vector<T>(Length);
            for (int i = 0; i < Length; i++)
            {
                eta[i] = logits[i % logits.Length];
                y[i] = targets[(i + i / logits.Length) % targets.Length];
            }

            var tape = new Tape();
            var x = tape.Leaf(eta);
            var t = tape.Leaf(y);
            var loss = XMath.LogisticCrossEntropy(x, t);
            tape.Backward(loss);
            var terms = new LinearAlgebra.Vector<T>(Length);
            for (int i = 0; i < Length; i++)
            {
                var single = new Tape();
                var xi = single.Leaf(new LinearAlgebra.Vector<T>([eta[i]]));
                var ti = single.Leaf(new LinearAlgebra.Vector<T>([y[i]]));
                var term = XMath.LogisticCrossEntropy(xi, ti);
                single.Backward(term);
                terms[i] = T.CreateTruncating(term.Value);
                string context = $"{typeof(T).Name} [{i}] eta {eta[i]}, y {y[i]}";
                Assert.True(SameBits(single.GetGradient(xi)[0], tape.GetGradient(x)[i]), $"{context} d eta");
                Assert.True(SameBits(single.GetGradient(ti)[0], tape.GetGradient(t)[i]), $"{context} dy");
            }

            Assert.True(SameBits(XMath.Sum(terms), T.CreateTruncating(loss.Value)), $"{typeof(T).Name} value");

            // 目標の分布の項 0 − T∘L も、T が 0 の要素（L = −∞ を含む）を +0 にする式が各レーンで同じか、文書の式で求めた値と比べる。
            var z = new Matrix<T>(Length, 1);
            var soft = new Matrix<T>(Length, 1);
            for (int i = 0; i < Length; i++)
            {
                z[i, 0] = i % 5 == 0 ? T.NegativeInfinity : T.CreateTruncating(i % 7 - 3.0);
                soft[i, 0] = i % 5 == 0 || i % 3 == 0 ? (i % 2 == 0 ? T.Zero : T.NegativeZero) : T.CreateTruncating(0.01 * (1 + i % 4));
            }

            var constants = new Tape();
            var masked = XMath.SoftmaxCrossEntropy(constants.Constant(z), constants.Constant(soft));
            Assert.True(double.IsFinite(masked.Value), $"{typeof(T).Name} targets");
            Assert.Equal(BitConverter.DoubleToInt64Bits(TargetValue<T>(z, soft, out _)), BitConverter.DoubleToInt64Bits(masked.Value));
        }

        static bool SameBits<T>(T expected, T actual) where T : unmanaged, IFloatingPointIeee754<T>
            => typeof(T) == typeof(double)
                ? BitConverter.DoubleToInt64Bits(double.CreateTruncating(expected)) == BitConverter.DoubleToInt64Bits(double.CreateTruncating(actual))
                : BitConverter.SingleToInt32Bits(float.CreateTruncating(expected)) == BitConverter.SingleToInt32Bits(float.CreateTruncating(actual));
    }

    [Fact]
    public void ZeroScalarAdjointsSkipTheRecord()
    {
        // 随伴がちょうど 0（float では丸めて 0）の損失は、∞・NaN の入力でも寄与を足さない（総和・内積と同じ）。
        var data = new Matrix<double>(2, 2);
        data[0, 0] = double.NaN;
        data[1, 1] = double.PositiveInfinity;
        Run(data, 0);
        Run(data.ConvertChecked<float>(), 1e-50);

        static void Run<T>(Matrix<T> data, double coefficient) where T : unmanaged, IFloatingPointIeee754<T>
        {
            var tape = new Tape();
            var x = tape.Leaf(data);
            var t = tape.Leaf(data);
            var v = tape.Leaf(new LinearAlgebra.Vector<T>([T.NaN, T.PositiveInfinity]));
            ScalarNode losses = XMath.SoftmaxCrossEntropy(x, [0, 1]) + XMath.SoftmaxCrossEntropy(x, t) + XMath.LogisticCrossEntropy(x, t)
                + XMath.LogisticCrossEntropy(v, v) + XMath.SoftmaxCrossEntropy(v, 1);
            tape.Backward(coefficient * losses + XMath.Sum(XMath.ColumnSums(x)) + XMath.Sum(XMath.ColumnSums(t)) + XMath.Sum(v));
            Assert.Equal([T.One, T.One, T.One, T.One], tape.GetGradient(x).ToMatrix().AsSpan().ToArray());
            Assert.Equal([T.One, T.One, T.One, T.One], tape.GetGradient(t).ToMatrix().AsSpan().ToArray());
            Assert.Equal([T.One, T.One], tape.GetGradient(v).ToArray());
        }
    }

    [Fact]
    public void UnreachedLossesAndConstantsAreNotDifferentiated()
    {
        // 出力から届かない損失は飛ばす。入力がすべて定数なら記録しない。目標だけが葉なら dT だけを求める。
        var data = Uniform<double>(new Random(96), 3, 4, -2, 2);
        var tape = new Tape();
        var x = tape.Leaf(data);
        var c = tape.Constant(data);
        _ = XMath.SoftmaxCrossEntropy(x, [0, 1, 2, 0]);
        _ = XMath.LogisticCrossEntropy(x, c);
        Assert.False(XMath.SoftmaxCrossEntropy(c, [0, 1, 2, 0]).RequiresGradient);
        Assert.False(XMath.SoftmaxCrossEntropy(c, c).RequiresGradient);
        Assert.False(XMath.LogisticCrossEntropy(c, c).RequiresGradient);
        var t = tape.Leaf(data);
        var loss = XMath.SoftmaxCrossEntropy(c, t);
        Assert.True(loss.RequiresGradient);
        tape.Backward(loss + XMath.Sum(XMath.ColumnSums(x)));
        Assert.Equal(Enumerable.Repeat(1.0, 12), tape.GetGradient(x).ToMatrix().AsSpan().ToArray());
        var logSoftmax = XMath.ColumnLogSoftmax<double>(data);
        var expected = logSoftmax.AsSpan().ToArray().Select(l => -1.0 * l).ToArray();
        AssertSameBits<double>(expected, tape.GetGradient(t).ToMatrix().AsSpan(), "dT");
    }

    [Fact]
    public void LabelsAreNotKeptAfterTheCall()
    {
        // 係数を記録するときに求めるので、呼んだ後にラベルの配列を書き換えても勾配は変わらない。
        var data = Uniform<double>(new Random(97), 4, 5, -2, 2);
        int[] labels = [0, 3, 1, 2, 3];
        var reference = new Tape();
        var r = reference.Leaf(data);
        reference.Backward(XMath.SoftmaxCrossEntropy(r, labels));
        var tape = new Tape();
        var x = tape.Leaf(data);
        var loss = XMath.SoftmaxCrossEntropy(x, labels);
        Array.Fill(labels, 2);
        tape.Backward(loss);
        AssertSameBits<double>(reference.GetGradient(r).ToMatrix().AsSpan(), tape.GetGradient(x).ToMatrix().AsSpan(), "gradient");
    }

    [Fact]
    public void EmptyInputsGivePositiveZero()
    {
        // 標本・クラス・要素のない損失は +0。勾配は空のまま例外を投げない。
        var tape = new Tape();
        var wide = tape.Leaf(new Matrix<double>(3, 0));
        var none = tape.Leaf(new Matrix<double>(0, 4));
        var empty = tape.Leaf(new LinearAlgebra.Vector<double>(0));
        ScalarNode[] losses =
        [
            XMath.SoftmaxCrossEntropy(wide, []), XMath.SoftmaxCrossEntropy(wide, wide), XMath.SoftmaxCrossEntropy(none, none),
            XMath.SoftmaxCrossEntropy(empty, empty), XMath.LogisticCrossEntropy(empty, empty), XMath.LogisticCrossEntropy(wide, wide),
            XMath.LogisticCrossEntropy(none, none),
        ];
        ScalarNode total = 0;
        foreach (var loss in losses)
        {
            Assert.Equal(0L, BitConverter.DoubleToInt64Bits(loss.Value));
            total += loss;
        }

        tape.Backward(total);
        Assert.Equal((3, 0), (tape.GetGradient(wide).Rows, tape.GetGradient(wide).Columns));
        Assert.Equal((0, 4), (tape.GetGradient(none).Rows, tape.GetGradient(none).Columns));
        Assert.Equal(0, tape.GetGradient(empty).Length);
    }

    [Fact]
    public void InvalidArgumentsThrow()
    {
        var tape = new Tape();
        var x = tape.Leaf(new Matrix<double>(3, 2));
        var v = tape.Leaf(new LinearAlgebra.Vector<double>(3));
        var other = new Tape().Leaf(new Matrix<double>(3, 2));
        DifferentiationAssert.Throws<ArgumentException>(() => XMath.SoftmaxCrossEntropy(x, [0]), "label count");
        DifferentiationAssert.Throws<ArgumentOutOfRangeException>(() => XMath.SoftmaxCrossEntropy(x, [0, 3]), "label too large");
        DifferentiationAssert.Throws<ArgumentOutOfRangeException>(() => XMath.SoftmaxCrossEntropy(x, [-1, 0]), "negative label");
        DifferentiationAssert.Throws<ArgumentOutOfRangeException>(() => XMath.SoftmaxCrossEntropy(tape.Leaf(new Matrix<double>(0, 1)), [0]), "no classes");
        DifferentiationAssert.Throws<ArgumentOutOfRangeException>(() => XMath.SoftmaxCrossEntropy(v, 3), "vector label");
        DifferentiationAssert.Throws<ArgumentOutOfRangeException>(() => XMath.SoftmaxCrossEntropy(v, -1), "negative vector label");
        DifferentiationAssert.Throws<ArgumentException>(() => XMath.SoftmaxCrossEntropy(x, x.T), "target shape");
        DifferentiationAssert.Throws<ArgumentException>(() => XMath.SoftmaxCrossEntropy(v, v[1..]), "target length");
        DifferentiationAssert.Throws<ArgumentException>(() => XMath.LogisticCrossEntropy(x, x.T), "logistic shape");
        DifferentiationAssert.Throws<ArgumentException>(() => XMath.LogisticCrossEntropy(v, v[1..]), "logistic length");
        DifferentiationAssert.Throws<InvalidOperationException>(() => XMath.SoftmaxCrossEntropy(x, other), "different tapes");
        DifferentiationAssert.Throws<InvalidOperationException>(() => XMath.LogisticCrossEntropy(other, x), "different tapes logistic");
    }

    [Fact]
    public void StaleNodesAndRecordingAfterBackwardThrow()
    {
        var tape = new Tape();
        var x = tape.Leaf(new Matrix<double>(2, 3));
        var v = tape.Leaf(new LinearAlgebra.Vector<double>(3));
        tape.Reset();
        DifferentiationAssert.Throws<InvalidOperationException>(() => XMath.SoftmaxCrossEntropy(x, [0, 1, 0]), "stale labels");
        DifferentiationAssert.Throws<InvalidOperationException>(() => XMath.SoftmaxCrossEntropy(x, x), "stale targets");
        DifferentiationAssert.Throws<InvalidOperationException>(() => XMath.SoftmaxCrossEntropy(v, 0), "stale vector");
        DifferentiationAssert.Throws<InvalidOperationException>(() => XMath.LogisticCrossEntropy(v, v), "stale logistic");

        // 逆伝播の後は、記録する損失は例外、定数だけの損失は値を返す。
        var data = Uniform<double>(new Random(98), 2, 3, -1, 1);
        var y = tape.Leaf(data);
        var w = tape.Leaf(new LinearAlgebra.Vector<double>([1.0, 2, 3]));
        var c = tape.Constant(data);
        var d = tape.Constant(new LinearAlgebra.Vector<double>([1.0, 2, 3]));
        tape.Backward(XMath.Sum(w));
        DifferentiationAssert.Throws<InvalidOperationException>(() => XMath.SoftmaxCrossEntropy(y, [0, 1, 0]), "after backward labels");
        DifferentiationAssert.Throws<InvalidOperationException>(() => XMath.SoftmaxCrossEntropy(c, y), "after backward targets");
        DifferentiationAssert.Throws<InvalidOperationException>(() => XMath.SoftmaxCrossEntropy(w, 2), "after backward vector");
        DifferentiationAssert.Throws<InvalidOperationException>(() => XMath.LogisticCrossEntropy(w, d), "after backward logistic");
        Assert.False(XMath.SoftmaxCrossEntropy(c, [0, 1, 0]).RequiresGradient);
        Assert.False(XMath.SoftmaxCrossEntropy(c, c).RequiresGradient);
        Assert.False(XMath.LogisticCrossEntropy(d, d).RequiresGradient);
        DifferentiationAssert.EqualBits(0 - XMath.LogSoftmax<double>(new LinearAlgebra.Vector<double>([1.0, 2, 3]))[2], XMath.SoftmaxCrossEntropy(d, 2).Value);
    }

    [Fact]
    public void LossModelGradientMatchesCentralDifferences()
    {
        var random = new Random(99);
        var point = Uniform<double>(random, 5, 4, -2, 2);
        ArrayActivationTests.AssertMatrixMatchesCentralDifferences(point, MatrixModel, 1e-7);
        var vector = Uniform<double>(random, 12, 1, -2, 2).AsMatrixSpan().Column(0).ToVector();
        ArrayElementwiseGradientTests.AssertMatchesCentralDifferences(vector, VectorModel, 1e-7);
    }

    [Fact]
    public void SingleGradientsAgreeWithDoubleGradients()
    {
        var random = new Random(100);
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

        static void AssertClose(double[] expected, float[] actual, string context)
        {
            double scale = expected.Max(Math.Abs);
            for (int i = 0; i < expected.Length; i++)
            {
                DifferentiationAssert.Close(expected[i], actual[i], 1e-4, 1e-5 * scale, $"{context}[{i}]");
            }
        }
    }

    [Fact]
    public void MultinomialLogisticRegressionMatchesTheAnalyticGradient()
    {
        // K 種類・D 次元・N 件。スコア Z = W·X + b·1ᵀ（列が標本）、損失 E = −Σ ln softmax(Z)[t]。解析的な勾配は (P − T)·Xᵀ と (P − T)·1。
        // ラベルの版、one-hot の目標の版、標本を行に置いたスコアの転置のビュー（(Xᵀ·Wᵀ + 1·bᵀ).T）で書く。
        const int classes = 4, features = 3, samples = 50;
        var random = new Random(101);
        var weight = Uniform<double>(random, classes, features, -1, 1);
        var bias = Uniform<double>(random, classes, 1, -1, 1).AsMatrixSpan().Column(0).ToVector();
        var data = Uniform<double>(random, features, samples, -3, 3);
        int[] labels = RandomLabels(random, classes, samples);
        var targets = new Matrix<double>(classes, samples);
        for (int n = 0; n < samples; n++)
        {
            targets[labels[n], n] = 1;
        }

        double expectedLoss = 0;
        var expectedWeight = new Matrix<double>(classes, features);
        var expectedBias = new double[classes];
        for (int n = 0; n < samples; n++)
        {
            var scores = new double[classes];
            for (int k = 0; k < classes; k++)
            {
                scores[k] = bias[k];
                for (int d = 0; d < features; d++)
                {
                    scores[k] += weight[k, d] * data[d, n];
                }
            }

            double max = scores.Max();
            double lse = max + Math.Log(scores.Sum(z => Math.Exp(z - max)));
            expectedLoss += lse - scores[labels[n]];
            for (int k = 0; k < classes; k++)
            {
                double residual = Math.Exp(scores[k] - lse) - targets[k, n];
                expectedBias[k] += residual;
                for (int d = 0; d < features; d++)
                {
                    expectedWeight[k, d] += residual * data[d, n];
                }
            }
        }

        for (int form = 0; form < 3; form++)
        {
            var tape = new Tape();
            var w = tape.Leaf(weight);
            var b = tape.Leaf(bias);
            var z = form < 2 ? w * tape.Constant(data) + b.Broadcast(samples) : (tape.Constant(data).T * w.T + b.BroadcastRows(samples)).T;
            var loss = form == 1 ? XMath.SoftmaxCrossEntropy(z, tape.Constant(targets)) : XMath.SoftmaxCrossEntropy(z, labels);
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
    public void LogisticRegressionMatchesTheAnalyticGradient()
    {
        // η = Xᵀw、y は 0・1 と確率。値は Σ softplus(η) − yᵀη、勾配は X(σ(η) − y)。行列の版は2つの出力を独立に並べる。
        const int features = 4, samples = 60;
        var random = new Random(102);
        var data = Uniform<double>(random, features, samples, -2, 2);
        var weights = Uniform<double>(random, features, 2, -1.5, 1.5);
        var targets = new Matrix<double>(2, samples);
        for (int n = 0; n < samples; n++)
        {
            targets[0, n] = random.Next(2);
            targets[1, n] = random.NextDouble();
        }

        var expectedLoss = new double[2];
        var expectedGradient = new Matrix<double>(features, 2);
        for (int o = 0; o < 2; o++)
        {
            for (int n = 0; n < samples; n++)
            {
                double eta = 0;
                for (int d = 0; d < features; d++)
                {
                    eta += data[d, n] * weights[d, o];
                }

                double softplus = Math.Max(eta, 0) + Math.Log(1 + Math.Exp(-Math.Abs(eta)));
                expectedLoss[o] += softplus - targets[o, n] * eta;
                for (int d = 0; d < features; d++)
                {
                    expectedGradient[d, o] += data[d, n] * (1 / (1 + Math.Exp(-eta)) - targets[o, n]);
                }
            }
        }

        // ベクトルの版（出力ごと）。
        for (int o = 0; o < 2; o++)
        {
            var tape = new Tape();
            var w = tape.Leaf(weights.AsMatrixSpan().Column(o).ToVector());
            var y = tape.Constant(targets.AsMatrixSpan().Row(o).ToVector());
            var loss = XMath.LogisticCrossEntropy(tape.Constant(data).T * w, y);
            tape.Backward(loss);
            DifferentiationAssert.Close(expectedLoss[o], loss.Value, 1e-13, 1e-13, $"output {o} loss");
            for (int d = 0; d < features; d++)
            {
                DifferentiationAssert.Close(expectedGradient[d, o], tape.GetGradient(w)[d], 1e-12, 1e-13, $"output {o} w[{d}]");
            }
        }

        // 行列の版（出力が行、標本が列）。
        var matrixTape = new Tape();
        var m = matrixTape.Leaf(weights);
        var matrixLoss = XMath.LogisticCrossEntropy(m.T * matrixTape.Constant(data), matrixTape.Constant(targets));
        matrixTape.Backward(matrixLoss);
        DifferentiationAssert.Close(expectedLoss[0] + expectedLoss[1], matrixLoss.Value, 1e-13, 1e-13, "matrix loss");
        for (int o = 0; o < 2; o++)
        {
            for (int d = 0; d < features; d++)
            {
                DifferentiationAssert.Close(expectedGradient[d, o], matrixTape.GetGradient(m)[d, o], 1e-12, 1e-13, $"matrix W[{d}, {o}]");
            }
        }
    }

    // 値: 特別な値を含む入力と有限の乱数の入力を 9 × 7 に並べ、葉・定数・転置の部分行列・ブロードキャストに通す。
    private static void LabelValues<T>() where T : unmanaged, IFloatingPointIeee754<T>
    {
        foreach (var (storage, vector) in ValueInputs<T>(110))
        {
            var tape = new Tape();
            var leaf = tape.Leaf(storage);
            var v = tape.Leaf(vector);
            int[] labels = RandomLabels(new Random(111), 9, 7);
            int[] viewLabels = RandomLabels(new Random(112), 4, 7);
            string context = typeof(T).Name;
            AssertSameValue(LabelValue<T>(storage, labels, out _), XMath.SoftmaxCrossEntropy(leaf, labels).Value, $"{context} leaf");
            AssertSameValue(LabelValue<T>(storage, labels, out _), XMath.SoftmaxCrossEntropy(tape.Constant(storage), labels).Value, $"{context} constant");
            AssertSameValue(LabelValue<T>(storage.AsMatrixSpan()[1..8, 2..6].T, viewLabels, out _), XMath.SoftmaxCrossEntropy(leaf[1..8, 2..6].T, viewLabels).Value, $"{context} view");
            AssertSameValue(LabelValue<T>(vector.AsVectorSpan()[3..9].Broadcast(4), viewLabels[..4], out _), XMath.SoftmaxCrossEntropy(v[3..9].Broadcast(4), viewLabels[..4]).Value, $"{context} columns");
            AssertSameValue(LabelValue<T>(vector.AsVectorSpan()[..5].BroadcastRows(3), [2, 0, 1, 1, 0], out _), XMath.SoftmaxCrossEntropy(v[..5].BroadcastRows(3), [2, 0, 1, 1, 0]).Value, $"{context} rows");
        }
    }

    private static void TargetValues<T>() where T : unmanaged, IFloatingPointIeee754<T>
    {
        foreach (var (storage, vector) in ValueInputs<T>(120))
        {
            var tape = new Tape();
            var leaf = tape.Leaf(storage);
            var v = tape.Leaf(vector);
            var targets = SoftTargets<T>(new Random(121), 9, 7);
            var t = tape.Leaf(targets);
            string context = typeof(T).Name;
            AssertSameValue(TargetValue<T>(storage, targets, out _), XMath.SoftmaxCrossEntropy(leaf, t).Value, $"{context} leaf");
            AssertSameValue(TargetValue<T>(storage, targets, out _), XMath.SoftmaxCrossEntropy(tape.Constant(storage), tape.Constant(targets)).Value, $"{context} constant");
            AssertSameValue(TargetValue<T>(storage.AsMatrixSpan()[1..8, 2..6].T, targets.AsMatrixSpan()[2..6, ..7], out _),
                XMath.SoftmaxCrossEntropy(leaf[1..8, 2..6].T, t[2..6, ..7]).Value, $"{context} view");
            AssertSameValue(TargetValue<T>(vector.AsVectorSpan()[3..9].Broadcast(4), targets.AsMatrixSpan()[3..9, 1..5], out _),
                XMath.SoftmaxCrossEntropy(v[3..9].Broadcast(4), t[3..9, 1..5]).Value, $"{context} columns");
            AssertSameValue(TargetValue<T>(targets.AsMatrixSpan()[..5, ..4], vector.AsVectorSpan()[4..8].BroadcastRows(5), out _),
                XMath.SoftmaxCrossEntropy(t[..5, ..4], v[4..8].BroadcastRows(5)).Value, $"{context} broadcast targets");
        }
    }

    private static void LogisticValues<T>() where T : unmanaged, IFloatingPointIeee754<T>
    {
        foreach (var (storage, vector) in ValueInputs<T>(130))
        {
            var tape = new Tape();
            var leaf = tape.Leaf(storage);
            var v = tape.Leaf(vector);
            var targets = BinaryTargets<T>(new Random(131), 9, 7);
            var t = tape.Leaf(targets);
            var y = tape.Leaf(BinaryTargets<T>(new Random(132), vector.Length, 1).AsMatrixSpan().Column(0).ToVector());
            string context = typeof(T).Name;
            AssertSameValue(LogisticValue<T>(storage, targets, matrix: true), XMath.LogisticCrossEntropy(leaf, t).Value, $"{context} leaf");
            AssertSameValue(LogisticValue<T>(storage, targets, matrix: true), XMath.LogisticCrossEntropy(tape.Constant(storage), tape.Constant(targets)).Value, $"{context} constant");
            AssertSameValue(LogisticValue<T>(storage.AsMatrixSpan()[1..8, 2..6].T, targets.AsMatrixSpan()[2..6, ..7], matrix: true),
                XMath.LogisticCrossEntropy(leaf[1..8, 2..6].T, t[2..6, ..7]).Value, $"{context} view");
            AssertSameValue(LogisticValue<T>(vector.AsVectorSpan()[3..9].Broadcast(4), targets.AsMatrixSpan()[3..9, 1..5], matrix: true),
                XMath.LogisticCrossEntropy(v[3..9].Broadcast(4), t[3..9, 1..5]).Value, $"{context} columns");
            AssertSameValue(LogisticValue<T>(vector.AsVectorSpan().Broadcast(1), y.Value.Broadcast(1), matrix: false), XMath.LogisticCrossEntropy(v, y).Value, $"{context} vector");
            AssertSameValue(LogisticValue<T>(vector.AsVectorSpan()[5..40].Broadcast(1), y.Value[5..40].Broadcast(1), matrix: false),
                XMath.LogisticCrossEntropy(v[5..40], y[5..40]).Value, $"{context} slice");
        }
    }

    // 勾配: 4通りの置き方で文書の式と比べる。0: 葉そのもの、1: 大きい葉の転置の部分行列、2: 同じ葉に2回、3: ベクトルを並べたブロードキャスト。
    private static void LabelGradients<T>() where T : unmanaged, IFloatingPointIeee754<T>
    {
        var storage = Uniform<T>(new Random(140), 9, 7, -4, 4);
        T g = T.CreateTruncating(Coefficient);
        for (int placement = 0; placement < 4; placement++)
        {
            var tape = new Tape();
            var leaf = tape.Leaf(storage);
            var vector = tape.Leaf(storage.AsMatrixSpan().Column(0).ToVector());
            var x = placement switch { 1 => leaf[1..8, 2..6].T, 3 => vector.Broadcast(5), _ => leaf };
            ReadOnlyMatrixSpan<T> values = placement switch
            {
                1 => storage.AsMatrixSpan()[1..8, 2..6].T,
                3 => storage.AsMatrixSpan().Column(0).Broadcast(5),
                _ => storage,
            };
            int[] labels = RandomLabels(new Random(141 + placement), values.Rows, values.Columns);
            ScalarNode loss = Coefficient * XMath.SoftmaxCrossEntropy(x, labels);
            tape.Backward(placement == 2 ? loss + Coefficient * XMath.SoftmaxCrossEntropy(x, labels) : loss);
            // P − onehot（P は L = XMath.ColumnLogSoftmax の XMath.Exp）に随伴を掛ける。
            var weights = XMath.Exp(XMath.ColumnLogSoftmax(values));
            for (int n = 0; n < labels.Length; n++)
            {
                weights[labels[n], n] -= T.One;
            }

            AssertGradient(tape, placement, leaf, vector, Scale(weights, g), $"labels {typeof(T).Name} placement {placement}");
        }
    }

    private static void TargetGradients<T>() where T : unmanaged, IFloatingPointIeee754<T>
    {
        var storage = Uniform<T>(new Random(150), 9, 7, -4, 4);
        var soft = SoftTargets<T>(new Random(151), 9, 7);
        T g = T.CreateTruncating(Coefficient);
        for (int placement = 0; placement < 4; placement++)
        {
            var tape = new Tape();
            var leaf = tape.Leaf(storage);
            var vector = tape.Leaf(storage.AsMatrixSpan().Column(0).ToVector());
            var targetLeaf = tape.Leaf(soft);
            var x = placement switch { 1 => leaf[1..8, 2..6].T, 3 => vector.Broadcast(5), _ => leaf };
            ReadOnlyMatrixSpan<T> values = placement switch
            {
                1 => storage.AsMatrixSpan()[1..8, 2..6].T,
                3 => storage.AsMatrixSpan().Column(0).Broadcast(5),
                _ => storage,
            };
            // 目標も、置き方 1 では大きい葉の転置の部分行列にする。
            var t = placement == 1 ? targetLeaf[2..9, 1..5].T : targetLeaf[..values.Rows, ..values.Columns];
            ReadOnlyMatrixSpan<T> targets = placement == 1 ? soft.AsMatrixSpan()[2..9, 1..5].T : soft.AsMatrixSpan()[..values.Rows, ..values.Columns];
            ScalarNode loss = Coefficient * XMath.SoftmaxCrossEntropy(x, t);
            tape.Backward(placement == 2 ? loss + Coefficient * XMath.SoftmaxCrossEntropy(x, t) : loss);
            TargetValue(values, targets, out var logSoftmax);
            var sums = XMath.ColumnSums(targets);
            var weights = XMath.Exp(logSoftmax);
            var factor = new Matrix<T>(values.Rows, values.Columns);
            var negative = new Matrix<T>(values.Rows, values.Columns);
            for (int j = 0; j < values.Columns; j++)
            {
                for (int i = 0; i < values.Rows; i++)
                {
                    factor[i, j] = weights[i, j] * sums[j] - targets[i, j];
                    negative[i, j] = logSoftmax[i, j] * -g;
                }
            }

            string context = $"targets {typeof(T).Name} placement {placement}";
            AssertGradient(tape, placement, leaf, vector, Scale(factor, g), context);

            // dT += −g·L（置き方 2 では 2 回）。T のビューの外の勾配は 0。
            var expected = new Matrix<T>(9, 7);
            for (int j = 0; j < values.Columns; j++)
            {
                for (int i = 0; i < values.Rows; i++)
                {
                    T d = placement == 2 ? negative[i, j] + negative[i, j] : negative[i, j];
                    if (placement == 1)
                    {
                        expected[j + 2, i + 1] = d;
                    }
                    else
                    {
                        expected[i, j] = d;
                    }
                }
            }

            AssertSameBits<T>(expected.AsSpan(), tape.GetGradient(targetLeaf).ToMatrix().AsSpan(), $"{context} dT");
        }
    }

    private static void LogisticGradients<T>() where T : unmanaged, IFloatingPointIeee754<T>
    {
        var storage = Uniform<T>(new Random(160), 9, 7, -6, 6);
        var binary = BinaryTargets<T>(new Random(161), 9, 7);
        T g = T.CreateTruncating(Coefficient);
        for (int placement = 0; placement < 4; placement++)
        {
            var tape = new Tape();
            var leaf = tape.Leaf(storage);
            var vector = tape.Leaf(storage.AsMatrixSpan().Column(0).ToVector());
            var targetLeaf = tape.Leaf(binary);
            var x = placement switch { 1 => leaf[1..8, 2..6].T, 3 => vector.Broadcast(5), _ => leaf };
            ReadOnlyMatrixSpan<T> values = placement switch
            {
                1 => storage.AsMatrixSpan()[1..8, 2..6].T,
                3 => storage.AsMatrixSpan().Column(0).Broadcast(5),
                _ => storage,
            };
            var t = targetLeaf[..values.Rows, ..values.Columns];
            ReadOnlyMatrixSpan<T> targets = binary.AsMatrixSpan()[..values.Rows, ..values.Columns];
            ScalarNode loss = Coefficient * XMath.LogisticCrossEntropy(x, t);
            tape.Backward(placement == 2 ? loss + Coefficient * XMath.LogisticCrossEntropy(x, t) : loss);
            var (factor, negative) = LogisticDerivatives(values, targets, g);
            string context = $"logistic {typeof(T).Name} placement {placement}";
            AssertGradient(tape, placement, leaf, vector, Scale(factor, g), context);
            var expected = new Matrix<T>(9, 7);
            for (int j = 0; j < values.Columns; j++)
            {
                for (int i = 0; i < values.Rows; i++)
                {
                    expected[i, j] = placement == 2 ? negative[i, j] + negative[i, j] : negative[i, j];
                }
            }

            AssertSameBits<T>(expected.AsSpan(), tape.GetGradient(targetLeaf).ToMatrix().AsSpan(), $"{context} dy");
        }

        // ベクトルの版（葉と部分ベクトル）。
        var data = Uniform<T>(new Random(162), 23, 1, -6, 6).AsMatrixSpan().Column(0).ToVector();
        var labels = BinaryTargets<T>(new Random(163), 23, 1).AsMatrixSpan().Column(0).ToVector();
        for (int placement = 0; placement < 2; placement++)
        {
            var tape = new Tape();
            var leaf = tape.Leaf(data);
            var y = tape.Leaf(labels);
            var x = placement == 1 ? leaf[3..20] : leaf;
            var target = placement == 1 ? y[3..20] : y;
            tape.Backward(Coefficient * XMath.LogisticCrossEntropy(x, target));
            ReadOnlyVectorSpan<T> values = placement == 1 ? data.AsVectorSpan()[3..20] : data;
            ReadOnlyVectorSpan<T> targets = placement == 1 ? labels.AsVectorSpan()[3..20] : labels;
            var (factor, negative) = LogisticDerivatives(values.Broadcast(1), targets.Broadcast(1), g);
            var dx = new T[data.Length];
            var dy = new T[data.Length];
            int start = placement == 1 ? 3 : 0;
            for (int i = 0; i < values.Length; i++)
            {
                dx[start + i] = factor[i, 0] * g;
                dy[start + i] = negative[i, 0];
            }

            string context = $"logistic vector {typeof(T).Name} placement {placement}";
            AssertSameBits<T>(dx, tape.GetGradient(leaf).ToArray(), $"{context} d eta");
            AssertSameBits<T>(dy, tape.GetGradient(y).ToArray(), $"{context} dy");
        }
    }

    // 置き方ごとに、葉（またはブロードキャストしたベクトル）の勾配を寄与 contribution（ノードの論理的な形）と比べる。
    private static void AssertGradient<T>(Tape tape, int placement, MatrixNode<T> leaf, VectorNode<T> vector, Matrix<T> contribution, string context)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (placement == 3)
        {
            // 各列が同じベクトルなので、寄与の行ごとの和がベクトルの勾配。
            var sums = XMath.RowSums<T>(contribution);
            AssertSameBits<T>(sums.AsSpan(), tape.GetGradient(vector).ToArray(), context);
            AssertSameBits<T>(new T[63], tape.GetGradient(leaf).ToMatrix().AsSpan(), $"{context} leaf");
            return;
        }

        var full = new Matrix<T>(9, 7);
        for (int j = 0; j < contribution.Columns; j++)
        {
            for (int i = 0; i < contribution.Rows; i++)
            {
                T c = contribution[i, j];
                if (placement == 1)
                {
                    full[j + 1, i + 2] = c;
                }
                else
                {
                    full[i, j] = placement == 2 ? c + c : c;
                }
            }
        }

        AssertSameBits<T>(full.AsSpan(), tape.GetGradient(leaf).ToMatrix().AsSpan(), context);
    }

    // ラベルの損失の文書の値。L = XMath.ColumnLogSoftmax(Z)、ℓ[n] = 0 − L[t[n], n]、XMath.Sum(ℓ)。
    private static double LabelValue<T>(ReadOnlyMatrixSpan<T> z, int[] labels, out Matrix<T> logSoftmax) where T : unmanaged, IFloatingPointIeee754<T>
    {
        logSoftmax = new Matrix<T>(z.Rows, z.Columns);
        XMath.ColumnLogSoftmax(z, logSoftmax);
        var losses = new LinearAlgebra.Vector<T>(z.Columns);
        for (int n = 0; n < z.Columns; n++)
        {
            losses[n] = T.Zero - logSoftmax[labels[n], n];
        }

        return double.CreateTruncating(XMath.Sum<T>(losses));
    }

    // 目標の分布の損失の文書の値。項は 0 − XMath.Hadamard(T, L)（T が 0 の要素は +0）、ℓ = XMath.ColumnSums(項)、XMath.Sum(ℓ)。
    private static double TargetValue<T>(ReadOnlyMatrixSpan<T> z, ReadOnlyMatrixSpan<T> t, out Matrix<T> logSoftmax) where T : unmanaged, IFloatingPointIeee754<T>
    {
        logSoftmax = new Matrix<T>(z.Rows, z.Columns);
        XMath.ColumnLogSoftmax(z, logSoftmax);
        var terms = new Matrix<T>(z.Rows, z.Columns);
        XMath.Hadamard(t, logSoftmax, terms);
        for (int j = 0; j < z.Columns; j++)
        {
            for (int i = 0; i < z.Rows; i++)
            {
                terms[i, j] = T.IsZero(t[i, j]) ? T.Zero : T.Zero - terms[i, j];
            }
        }

        return double.CreateTruncating(XMath.Sum<T>(XMath.ColumnSums<T>(terms)));
    }

    // ロジスティック損失の文書の値。s = XMath.Softplus(−|η|)（論理的な順に並べたベクトル）、ℓ = y·softplus(−η) + (1 − y)·softplus(η)（係数 0 の項は +0）。
    private static double LogisticValue<T>(ReadOnlyMatrixSpan<T> eta, ReadOnlyMatrixSpan<T> y, bool matrix) where T : unmanaged, IFloatingPointIeee754<T>
    {
        int rows = eta.Rows, columns = eta.Columns;
        var negative = new LinearAlgebra.Vector<T>(rows * columns);
        for (int j = 0; j < columns; j++)
        {
            for (int i = 0; i < rows; i++)
            {
                negative[i + j * rows] = -T.Abs(eta[i, j]);
            }
        }

        var s = XMath.Softplus<T>(negative);
        var losses = new Matrix<T>(rows, columns);
        for (int j = 0; j < columns; j++)
        {
            for (int i = 0; i < rows; i++)
            {
                T x = eta[i, j], target = y[i, j], complement = T.One - target, si = s[i + j * rows];
                T positive = x >= T.Zero ? x + si : si;
                T negativeTerm = x >= T.Zero ? si : si - x;
                losses[i, j] = (T.IsZero(target) ? T.Zero : target * negativeTerm) + (T.IsZero(complement) ? T.Zero : complement * positive);
            }
        }

        T value = matrix ? XMath.Sum<T>(XMath.ColumnSums<T>(losses)) : XMath.Sum(losses.AsMatrixSpan().Column(0).AsReadOnly());
        return double.CreateTruncating(value);
    }

    // ロジスティック損失の局所微分。η について (1 − y)·σ(η) − y·σ(−η)（σ(±η) は p = XMath.Logistic(−|η|) と 1 − p）、y について −g·η。
    private static (Matrix<T> Factor, Matrix<T> Negative) LogisticDerivatives<T>(ReadOnlyMatrixSpan<T> eta, ReadOnlyMatrixSpan<T> y, T g)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        int rows = eta.Rows, columns = eta.Columns;
        var negative = new LinearAlgebra.Vector<T>(rows * columns);
        for (int j = 0; j < columns; j++)
        {
            for (int i = 0; i < rows; i++)
            {
                negative[i + j * rows] = -T.Abs(eta[i, j]);
            }
        }

        var p = XMath.Logistic<T>(negative);
        var factor = new Matrix<T>(rows, columns);
        var dy = new Matrix<T>(rows, columns);
        for (int j = 0; j < columns; j++)
        {
            for (int i = 0; i < rows; i++)
            {
                T x = eta[i, j], pi = p[i + j * rows], q = T.One - pi;
                factor[i, j] = (T.One - y[i, j]) * (x >= T.Zero ? q : pi) - y[i, j] * (x >= T.Zero ? pi : q);
                dy[i, j] = x * -g;
            }
        }

        return (factor, dy);
    }

    // 寄与 g·factor（逆伝播と同じく、係数に随伴を掛ける順）。
    private static Matrix<T> Scale<T>(Matrix<T> factor, T g) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var result = new Matrix<T>(factor.Rows, factor.Columns);
        for (int k = 0; k < factor.AsSpan().Length; k++)
        {
            result.AsSpan()[k] = factor.AsSpan()[k] * g;
        }

        return result;
    }

    // 損失をすべて通すモデル。点は 5 × 4 で、転置のビュー（4 クラス × 5 標本）と点そのもの（5 クラス × 4 標本）に通す。
    // 目標の分布と 2 値の目標を点のソフトマックス・ロジスティックにした項で、dT・dy の経路も通す。
    private static ScalarNode MatrixModel<T>(Tape tape, MatrixNode<T> m) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var z = m.T;
        var soft = tape.Constant(SoftTargets<T>(new Random(170), 4, 5));
        var binary = tape.Constant(BinaryTargets<T>(new Random(171), 5, 4));
        var teacher = tape.Constant(Uniform<T>(new Random(172), 5, 4, -1, 1));
        return XMath.SoftmaxCrossEntropy(z, [0, 3, 1, 2, 1]) + 0.7 * XMath.SoftmaxCrossEntropy(z, soft) + 1.3 * XMath.SoftmaxCrossEntropy(m, [4, 0, 2, 2])
            + XMath.LogisticCrossEntropy(m, binary) + XMath.SoftmaxCrossEntropy(teacher, XMath.ColumnSoftmax(m))
            + 0.5 * XMath.LogisticCrossEntropy(teacher * 2, XMath.Logistic(m));
    }

    // ベクトルの損失をすべて通すモデル（部分ベクトルを含む）。
    private static ScalarNode VectorModel<T>(Tape tape, VectorNode<T> v) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var y = tape.Constant(BinaryTargets<T>(new Random(173), 12, 1).AsMatrixSpan().Column(0).ToVector());
        var soft = tape.Constant(SoftTargets<T>(new Random(174), 11, 1).AsMatrixSpan().Column(0).ToVector());
        return XMath.LogisticCrossEntropy(v, y) + XMath.LogisticCrossEntropy(v[2..9], y[3..10]) + XMath.SoftmaxCrossEntropy(v, 3)
            + XMath.SoftmaxCrossEntropy(v[1..], soft) + XMath.SoftmaxCrossEntropy(v[1..], XMath.Softmax(v[..11] * 0.5));
    }

    // 値の入力: 特別な値（±0・非正規化数・裾・±∞・NaN）を含む入力と、有限の乱数だけの入力。行列 9 × 7 と、同じ値のベクトル。
    private static IEnumerable<(Matrix<T> Storage, LinearAlgebra.Vector<T> Vector)> ValueInputs<T>(int seed) where T : unmanaged, IFloatingPointIeee754<T>
    {
        foreach (double[] inputs in new[] { Inputs(seed, 30, 30), Uniform<double>(new Random(seed + 1), 63, 1, -30, 30).AsSpan().ToArray() })
        {
            var data = new LinearAlgebra.Vector<double>(inputs).ConvertChecked<T>();
            var storage = new Matrix<T>(9, 7);
            for (int k = 0; k < 63; k++)
            {
                storage.AsSpan()[k] = data[k % data.Length];
            }

            yield return (storage, data);
        }
    }

    private static int[] RandomLabels(Random random, int classes, int samples)
    {
        var labels = new int[samples];
        for (int n = 0; n < samples; n++)
        {
            labels[n] = random.Next(classes);
        }

        return labels;
    }

    // 目標の分布: 各列の和が 1 になる正の値に、0 の要素（約 3 割）を混ぜる。
    private static Matrix<T> SoftTargets<T>(Random random, int rows, int columns) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var result = new Matrix<T>(rows, columns);
        for (int j = 0; j < columns; j++)
        {
            var weights = new double[rows];
            for (int i = 0; i < rows; i++)
            {
                weights[i] = random.NextDouble() < 0.3 ? 0 : random.NextDouble();
            }

            double total = weights.Sum();
            for (int i = 0; i < rows; i++)
            {
                result[i, j] = T.CreateTruncating(total > 0 ? weights[i] / total : 1.0 / rows);
            }
        }

        return result;
    }

    // 2 値の目標: 0・1 と、(0, 1) の確率を混ぜる。
    private static Matrix<T> BinaryTargets<T>(Random random, int rows, int columns) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var result = new Matrix<T>(rows, columns);
        for (int k = 0; k < rows * columns; k++)
        {
            int kind = random.Next(4);
            result.AsSpan()[k] = T.CreateTruncating(kind == 0 ? 0 : kind == 1 ? 1 : random.NextDouble());
        }

        return result;
    }

    private static void AssertSameValue(double expected, double actual, string context)
    {
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual), $"{context}: expected NaN, actual {actual:R}");
            return;
        }

        DifferentiationAssert.EqualBits(expected, actual, context);
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
