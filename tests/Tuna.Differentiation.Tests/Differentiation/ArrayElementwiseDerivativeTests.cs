using System.Numerics;
using Tuna.Differentiation;
using Tuna.LinearAlgebra;
using Tuna.Tests.ReferenceData;
using Xunit.Abstractions;
using static Tuna.Tests.Differentiation.ElementwiseOperations;

namespace Tuna.Tests.Differentiation;

// 要素別演算の局所微分をスカラー自動微分および高精度参照表と比較して検証する。
// 出力の総和を微分すると随伴はすべて 1 になるため、入力の勾配は局所微分そのものと一致する。
public class ArrayElementwiseDerivativeTests(ITestOutputHelper output)
{
    // スカラー参照表と同じ許容基準（相対 16ε または 4 ULP）。float は float の ε と ULP で評価する。
    private const ulong UlpTolerance = 4;
    private const float SingleSixteenEpsilon = 16 * 1.1920929e-7f;

    // double でスカラー自動微分と同一の式・関数を使う演算（0 の符号を除き局所微分が一致する）。
    private static readonly HashSet<string> SameFormula =
        ["Negate", "Log", "Sqrt", "Square", "Abs", "Log1p", "Logit", "Add", "Subtract", "Hadamard", "Multiply", "Min", "Max",
         "Pow", "LogAddExp", "LogSubExp", "ScalarSubtract"];

    public static TheoryData<string> UnaryCases() => UnaryNames();

    public static TheoryData<string> BinaryCases() => BinaryNames();

    public static TheoryData<string> ScalarCases() => ScalarNames();

    [Theory]
    [MemberData(nameof(UnaryCases))]
    public void UnaryLocalDerivativesFollowTheScalarRecords(string name)
    {
        double[] inputs = Inputs(11, 200, 40);
        var expected = inputs.Select(x => DifferentiationAssert.Gradient(v => Scalar(name, v), x)).ToArray();
        var expectedSingle = inputs.Select(x => (float)DifferentiationAssert.Gradient(v => Scalar(name, v), (float)x)).ToArray();
        // ベクトル（連続）と、転置の部分行列（一時領域に集めて勾配のビューへ書き戻す経路）。
        var vector = UnaryGradient<double>(name, inputs, transposed: false);
        var view = UnaryGradient<double>(name, inputs, transposed: true);
        var single = UnaryGradient<float>(name, inputs, transposed: false);
        var singleView = UnaryGradient<float>(name, inputs, transposed: true);
        bool exact = SameFormula.Contains(name);
        for (int i = 0; i < inputs.Length; i++)
        {
            string context = $"{name}({inputs[i]:R})";
            Matches(expected[i], vector[i], exact, context);
            Matches(expected[i], view[i], exact, context + " view");
            MatchesSingle(expectedSingle[i], single[i], context + " float");
            MatchesSingle(expectedSingle[i], singleView[i], context + " float view");
        }
    }

    [Theory]
    [MemberData(nameof(BinaryCases))]
    public void BinaryLocalDerivativesFollowTheScalarRecords(string name)
    {
        double[] values = Inputs(12, 0, 1);
        var random = new Random(13);
        var pairs = new List<(double X, double Y)>();
        foreach (double x in values)
        {
            foreach (double y in values)
            {
                pairs.Add((x, y));
            }
        }

        for (int i = 0; i < 300; i++)
        {
            pairs.Add((8 * random.NextDouble() - 4, 8 * random.NextDouble() - 4));
        }

        double[] xs = pairs.Select(p => p.X).ToArray(), ys = pairs.Select(p => p.Y).ToArray();
        var (dx, dy) = BinaryGradient<double>(name, xs, ys, transposed: false);
        var (vx, vy) = BinaryGradient<double>(name, xs, ys, transposed: true);
        var (fx, fy) = BinaryGradient<float>(name, xs, ys, transposed: false);
        bool exact = SameFormula.Contains(name);
        for (int i = 0; i < pairs.Count; i++)
        {
            var (ex, ey) = ScalarGradients(name, xs[i], ys[i]);
            var (sx, sy) = ScalarGradients(name, (float)xs[i], (float)ys[i]);
            string context = $"{name}({xs[i]:R}, {ys[i]:R})";
            Matches(ex, dx[i], exact, context + " dx");
            Matches(ey, dy[i], exact && name != "Divide", context + " dy");
            Matches(ex, vx[i], exact, context + " dx view");
            Matches(ey, vy[i], exact && name != "Divide", context + " dy view");
            MatchesSingle((float)sx, fx[i], context + " dx float");
            MatchesSingle((float)sy, fy[i], context + " dy float");
        }
    }

    [Theory]
    [MemberData(nameof(ScalarCases))]
    public void ScalarOperandLocalDerivativesFollowTheScalarRecords(string name)
    {
        double[] inputs = Inputs(14, 40, 4);
        bool exact = SameFormula.Contains(name);
        foreach (double s in Inputs(15, 0, 1).Where((_, i) => i % 3 == 0).Append(0.3).Append(-2.5))
        {
            var dx = ScalarOperandGradient<double>(name, inputs, s);
            var fx = ScalarOperandGradient<float>(name, inputs, s);
            for (int i = 0; i < inputs.Length; i++)
            {
                var (ex, es) = ScalarGradients(name, inputs[i], s);
                var (sx, ss) = ScalarGradients(name, (float)inputs[i], (float)s);
                string context = $"{name}({inputs[i]:R}, s={s:R})";
                Matches(ex, dx[i], exact, context + " dx");
                MatchesSingle((float)sx, fx[i], context + " dx float");
                // 1要素の配列なら、スカラーの勾配はその要素の寄与そのもの。
                Matches(es, ScalarGradient<double>(name, inputs[i], s), exact && name != "Divide", context + " ds");
                MatchesSingle((float)ss, (float)ScalarGradient<float>(name, inputs[i], s), context + " ds float");
            }
        }
    }

    [Theory]
    [InlineData("Exp")]
    [InlineData("Log")]
    [InlineData("Sqrt")]
    [InlineData("Square")]
    [InlineData("Abs")]
    [InlineData("Expm1")]
    [InlineData("Log1p")]
    [InlineData("Logistic")]
    [InlineData("LogLogistic")]
    [InlineData("Logit")]
    [InlineData("Softplus")]
    [InlineData("Log1mExp")]
    public void ArrayLocalDerivativesMatchHighPrecisionReferences(string function)
    {
        var rows = ElementaryReferences.Rows.Where(row => row.Function == function).ToArray();
        double[] inputs = rows.Select(row => row.X).ToArray();
        var vector = UnaryGradient<double>(function, inputs, transposed: false);
        var view = UnaryGradient<double>(function, inputs, transposed: true);
        double maximumRelativeError = 0;
        ulong maximumUlps = 0;
        for (int i = 0; i < rows.Length; i++)
        {
            double expected = rows[i].Derivative;
            foreach (double actual in new[] { vector[i], view[i] })
            {
                string context = $"{function}({rows[i].X:R}): expected {expected:R}, actual {actual:R}";
                if (double.IsInfinity(expected))
                {
                    Assert.True(actual == expected, context);
                    continue;
                }

                Assert.True(double.IsFinite(actual), context);
                double relative = expected == 0 ? (actual == 0 ? 0 : double.PositiveInfinity) : Math.Abs((actual - expected) / expected);
                ulong ulps = DifferentiationAssert.UlpDistance(actual, expected);
                Assert.True(relative <= DifferentiationAssert.SixteenEpsilon || ulps <= UlpTolerance, $"{context}, relative {relative:E3}, {ulps} ulp");
                maximumUlps = Math.Max(maximumUlps, ulps);
                if (double.IsNormal(expected))
                {
                    maximumRelativeError = Math.Max(maximumRelativeError, relative);
                }
            }
        }

        output.WriteLine(FormattableString.Invariant(
            $"{function}: samples={rows.Length}, max relative error (normal references)={maximumRelativeError:E3}, max ULP distance={maximumUlps}"));
    }

    [Theory]
    [InlineData("Pow")]
    [InlineData("LogAddExp")]
    [InlineData("LogSubExp")]
    public void ArrayBinaryDerivativesMatchHighPrecisionReferences(string function)
    {
        var rows = ScalarFunctionReferences.Rows.Where(row => row.Function == function).ToArray();
        double[] xs = rows.Select(row => row.Inputs[0]).ToArray(), ys = rows.Select(row => row.Inputs[1]).ToArray();
        var (dx, dy) = BinaryGradient<double>(function, xs, ys, transposed: false);
        var statistics = new ScalarFunctionReferenceStatistics();
        for (int i = 0; i < rows.Length; i++)
        {
            string context = $"{function}({xs[i]:R}, {ys[i]:R})";
            statistics.Observe(rows[i], 0, ScalarFunctionReferenceAssert.Matches(rows[i], 0, dx[i], context + " dx"), context);
            statistics.Observe(rows[i], 1, ScalarFunctionReferenceAssert.Matches(rows[i], 1, dy[i], context + " dy"), context);
            // スカラーとの演算（x が配列、y がスカラーのノード）も同じ関数で求める。
            double scalarDx = ScalarOperandGradient<double>(function, [xs[i]], ys[i])[0];
            ScalarFunctionReferenceAssert.Matches(rows[i], 0, scalarDx, context + " dx (scalar operand)");
            ScalarFunctionReferenceAssert.Matches(rows[i], 1, ScalarGradient<double>(function, xs[i], ys[i]), context + " ds (scalar operand)");
        }

        statistics.WriteTo(output, function, rows.Length);
    }

    // 入力を葉にして、出力の総和の勾配を返す。transposed なら、入力を大きい格納の転置の部分行列に置く。
    private static double[] UnaryGradient<T>(string name, double[] inputs, bool transposed) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var tape = new Tape();
        if (!transposed)
        {
            var x = tape.Leaf(new LinearAlgebra.Vector<double>(inputs).ConvertChecked<T>());
            tape.Backward(XMath.Sum(Apply(name, x)));
            return tape.GetGradient(x).ToArray().Select(v => double.CreateTruncating(v)).ToArray();
        }

        var (leaf, view) = TransposedView<T>(tape, inputs);
        var y = Apply(name, view);
        // 行列の総和は、出力（連続した列優先の行列）を1列に並べ直す代わりに、1 のベクトルとの積の総和で求める。
        tape.Backward(XMath.Sum(y * tape.Constant(Ones<T>(y.Columns))));
        return FromView(tape.GetGradient(view));
    }

    private static (double[] Dx, double[] Dy) BinaryGradient<T>(string name, double[] xs, double[] ys, bool transposed)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        var tape = new Tape();
        if (!transposed)
        {
            var x = tape.Leaf(new LinearAlgebra.Vector<double>(xs).ConvertChecked<T>());
            var y = tape.Leaf(new LinearAlgebra.Vector<double>(ys).ConvertChecked<T>());
            tape.Backward(XMath.Sum(Apply(name, x, y)));
            return (Widen(tape.GetGradient(x).ToArray()), Widen(tape.GetGradient(y).ToArray()));
        }

        // x は転置の部分行列、y は列優先の部分行列。
        var (_, xv) = TransposedView<T>(tape, xs);
        int rows = xv.Rows, columns = xv.Columns;
        var storage = new Matrix<T>(rows + 2, columns + 1);
        for (int i = 0; i < ys.Length; i++)
        {
            storage[1 + i % rows, i / rows] = T.CreateTruncating(ys[i]);
        }

        var yLeaf = tape.Leaf(storage);
        var yv = yLeaf[1..(rows + 1), ..columns];
        var z = Apply(name, xv, yv);
        tape.Backward(XMath.Sum(z * tape.Constant(Ones<T>(columns))));
        return (FromView(tape.GetGradient(xv)), FromView(tape.GetGradient(yv)));
    }

    private static double[] ScalarOperandGradient<T>(string name, double[] inputs, double s) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var tape = new Tape();
        var x = tape.Leaf(new LinearAlgebra.Vector<double>(inputs).ConvertChecked<T>());
        tape.Backward(XMath.Sum(Apply(name, x, tape.Leaf(s))));
        return Widen(tape.GetGradient(x).ToArray());
    }

    // 1要素の配列とスカラーのノードの演算の、スカラーの勾配。
    private static double ScalarGradient<T>(string name, double x, double s) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var tape = new Tape();
        var sn = tape.Leaf(s);
        var xn = tape.Constant(new LinearAlgebra.Vector<T>([T.CreateTruncating(x)]));
        tape.Backward(XMath.Sum(Apply(name, xn, sn)));
        return tape.GetGradient(sn);
    }

    // 入力を、列の間隔が行数と違う大きい格納の転置の部分行列（論理的に 5 行）に、論理的な列優先の順で置く。
    private static (MatrixNode<T> Leaf, MatrixNode<T> View) TransposedView<T>(Tape tape, double[] inputs) where T : unmanaged, IFloatingPointIeee754<T>
    {
        const int rows = 5;
        int columns = (inputs.Length + rows - 1) / rows;
        // 論理的な (i, j) は格納の (j + 1, i + 2)。端数の要素は 0 で埋める。
        var storage = new Matrix<T>(columns + 3, rows + 4);
        for (int k = 0; k < rows * columns; k++)
        {
            storage[k / rows + 1, k % rows + 2] = k < inputs.Length ? T.CreateTruncating(inputs[k]) : T.One;
        }

        var leaf = tape.Leaf(storage);
        return (leaf, leaf[1..(columns + 1), 2..(rows + 2)].T);
    }

    private static double[] FromView<T>(ReadOnlyMatrixSpan<T> gradient) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var result = new double[gradient.Rows * gradient.Columns];
        for (int j = 0; j < gradient.Columns; j++)
        {
            for (int i = 0; i < gradient.Rows; i++)
            {
                result[i + j * gradient.Rows] = double.CreateTruncating(gradient[i, j]);
            }
        }

        return result;
    }

    private static LinearAlgebra.Vector<T> Ones<T>(int length) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var result = new LinearAlgebra.Vector<T>(length);
        result.AsSpan().Fill(T.One);
        return result;
    }

    private static double[] Widen<T>(T[] values) where T : unmanaged, IFloatingPointIeee754<T>
        => values.Select(v => double.CreateTruncating(v)).ToArray();

    private static (double Dx, double Dy) ScalarGradients(string name, double x, double y)
    {
        var tape = new Tape();
        var a = tape.Leaf(x);
        var b = tape.Leaf(y);
        tape.Backward(Scalar(name, a, b));
        return (tape.GetGradient(a), tape.GetGradient(b));
    }

    private static void Matches(double expected, double actual, bool exact, string context)
    {
        if (double.IsNaN(expected) || double.IsNaN(actual))
        {
            Assert.True(double.IsNaN(expected) && double.IsNaN(actual), $"{context}: expected {expected:R}, actual {actual:R}");
            return;
        }

        // 0 の符号は約束しない。
        if (exact || expected == 0 || double.IsInfinity(expected))
        {
            Assert.True(actual == expected, $"{context}: expected {expected:R}, actual {actual:R}");
            return;
        }

        double relative = Math.Abs((actual - expected) / expected);
        ulong ulps = DifferentiationAssert.UlpDistance(actual, expected);
        Assert.True(relative <= DifferentiationAssert.SixteenEpsilon || ulps <= UlpTolerance,
            $"{context}: expected {expected:R}, actual {actual:R}, relative {relative:E3}, {ulps} ulp");
    }

    private static void MatchesSingle(float expected, double actualWide, string context)
    {
        float actual = (float)actualWide;
        if (float.IsNaN(expected) || float.IsNaN(actual))
        {
            Assert.True(float.IsNaN(expected) && float.IsNaN(actual), $"{context}: expected {expected:R}, actual {actual:R}");
            return;
        }

        float relative = expected == 0 || float.IsInfinity(expected) ? (actual == expected ? 0 : float.PositiveInfinity) : Math.Abs((actual - expected) / expected);
        uint ulps = SingleUlpDistance(expected, actual);
        Assert.True(relative <= SingleSixteenEpsilon || ulps <= UlpTolerance,
            $"{context}: expected {expected:R}, actual {actual:R}, relative {relative:E3}, {ulps} ulp");
    }

    // ±0 を同じ点に写し、∞ は最大の有限値の隣とする。
    private static uint SingleUlpDistance(float first, float second)
    {
        static uint Ordered(float value)
        {
            uint bits = BitConverter.SingleToUInt32Bits(value);
            const uint sign = 1U << 31;
            return (bits & sign) == 0 ? sign + bits : sign - (bits & ~sign);
        }

        uint a = Ordered(first), b = Ordered(second);
        return a >= b ? a - b : b - a;
    }
}
