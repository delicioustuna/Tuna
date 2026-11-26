using System.Numerics;
using Tuna.Differentiation;
using Tuna.LinearAlgebra;
using Xunit.Abstractions;

namespace Tuna.Tests.Differentiation;

/// <summary>
/// 活性化関数の値（XMath と同じビット）、局所微分（高精度の参照表）、float と double の一致、0 の局所微分の規則、中心差分。
/// 出力の総和を微分すると随伴はすべて 1 なので、入力の勾配は局所微分そのもの（0 の局所微分は寄与を足さないので 0）。
/// </summary>
public class ArrayActivationTests(ITestOutputHelper output)
{
    // 近似版は名前の末尾に Approximate を付ける。LeakyRelu・Elu の引数は Parameter で渡す。
    internal static readonly string[] Names = ["Relu", "LeakyRelu", "Elu", "Selu", "Gelu", "GeluApproximate", "Silu", "SiluApproximate"];

    private const ulong UlpTolerance = 4;
    private const float SingleSixteenEpsilon = 16 * 1.1920929e-7f;

    // 相殺する2項の絶対値の和 S が導関数の絶対値の 4 倍を超える点（導関数の零点の近く）では、|誤差| ≤ 16ε·S も認める（実装の前に決めた基準）。
    private const double CancellationRatio = 4;

    public static TheoryData<string> Cases() => new(Names);

    public static TheoryData<string, double> ReferenceCases()
    {
        var data = new TheoryData<string, double>();
        foreach (var (function, parameter) in ActivationReferences.Rows.Select(row => (row.Function, row.Parameter)).Distinct())
        {
            data.Add(function, parameter);
        }

        return data;
    }

    internal static VectorNode<T> Apply<T>(string name, VectorNode<T> x, double parameter) where T : unmanaged, IFloatingPointIeee754<T> => name switch
    {
        "Relu" => XMath.Relu(x),
        "LeakyRelu" => XMath.LeakyRelu(x, T.CreateTruncating(parameter)),
        "Elu" => XMath.Elu(x, T.CreateTruncating(parameter)),
        "Selu" => XMath.Selu(x),
        "Gelu" => XMath.Gelu(x),
        "GeluApproximate" or "GeluTanh" => XMath.Gelu(x, approximate: true),
        "Silu" => XMath.Silu(x),
        "SiluApproximate" => XMath.Silu(x, approximate: true),
        _ => throw new ArgumentOutOfRangeException(nameof(name)),
    };

    internal static MatrixNode<T> Apply<T>(string name, MatrixNode<T> x, double parameter) where T : unmanaged, IFloatingPointIeee754<T> => name switch
    {
        "Relu" => XMath.Relu(x),
        "LeakyRelu" => XMath.LeakyRelu(x, T.CreateTruncating(parameter)),
        "Elu" => XMath.Elu(x, T.CreateTruncating(parameter)),
        "Selu" => XMath.Selu(x),
        "Gelu" => XMath.Gelu(x),
        "GeluApproximate" or "GeluTanh" => XMath.Gelu(x, approximate: true),
        "Silu" => XMath.Silu(x),
        "SiluApproximate" => XMath.Silu(x, approximate: true),
        _ => throw new ArgumentOutOfRangeException(nameof(name)),
    };

    internal static LinearAlgebra.Vector<T> Expected<T>(string name, ReadOnlyVectorSpan<T> x, double parameter) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var result = new LinearAlgebra.Vector<T>(x.Length);
        VectorSpan<T> d = result;
        T p = T.CreateTruncating(parameter);
        switch (name)
        {
            case "Relu": XMath.Relu(x, d); break;
            case "LeakyRelu": XMath.LeakyRelu(x, p, d); break;
            case "Elu": XMath.Elu(x, p, d); break;
            case "Selu": XMath.Selu(x, d); break;
            case "Gelu": XMath.Gelu(x, d); break;
            case "GeluApproximate": XMath.Gelu(x, true, d); break;
            case "Silu": XMath.Silu(x, d); break;
            case "SiluApproximate": XMath.Silu(x, true, d); break;
            default: throw new ArgumentOutOfRangeException(nameof(name));
        }

        return result;
    }

    internal static Matrix<T> Expected<T>(string name, ReadOnlyMatrixSpan<T> x, double parameter) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var result = new Matrix<T>(x.Rows, x.Columns);
        MatrixSpan<T> d = result;
        T p = T.CreateTruncating(parameter);
        switch (name)
        {
            case "Relu": XMath.Relu(x, d); break;
            case "LeakyRelu": XMath.LeakyRelu(x, p, d); break;
            case "Elu": XMath.Elu(x, p, d); break;
            case "Selu": XMath.Selu(x, d); break;
            case "Gelu": XMath.Gelu(x, d); break;
            case "GeluApproximate": XMath.Gelu(x, true, d); break;
            case "Silu": XMath.Silu(x, d); break;
            case "SiluApproximate": XMath.Silu(x, true, d); break;
            default: throw new ArgumentOutOfRangeException(nameof(name));
        }

        return result;
    }

    // テストで使う引数（LeakyRelu の傾き・Elu の α）。
    internal static double DefaultParameter(string name) => name switch
    {
        "LeakyRelu" => 0.05,
        "Elu" => 0.7,
        _ => 0,
    };

    [Theory]
    [MemberData(nameof(Cases))]
    public void ValuesMatchXMathBitForBit(string name)
    {
        double[] inputs = Inputs(31);
        ValuesMatch<double>(name, inputs);
        ValuesMatch<float>(name, inputs);
    }

    [Theory]
    [MemberData(nameof(ReferenceCases))]
    public void LocalDerivativesMatchHighPrecisionReferences(string function, double parameter)
    {
        var rows = ActivationReferences.Rows.Where(row => row.Function == function && row.Parameter == parameter).ToArray();
        double[] inputs = rows.Select(row => row.X).ToArray();
        var vector = VectorGradient<double>(function, inputs, parameter);
        var view = ViewGradient<double>(function, inputs, parameter);
        double maximumRelativeError = 0, maximumScaledError = 0;
        ulong maximumUlps = 0;
        int scaledOnly = 0;
        for (int i = 0; i < rows.Length; i++)
        {
            double expected = rows[i].Derivative;
            foreach (double actual in new[] { vector[i], view[i] })
            {
                string context = $"{function}[{parameter:R}]({rows[i].X:R}): expected {expected:R}, actual {actual:R}";
                Assert.True(double.IsFinite(actual), context);
                double relative = expected == 0 ? (actual == 0 ? 0 : double.PositiveInfinity) : Math.Abs((actual - expected) / expected);
                ulong ulps = DifferentiationAssert.UlpDistance(actual, expected);
                bool ordinary = relative <= DifferentiationAssert.SixteenEpsilon || ulps <= UlpTolerance;
                double scale = rows[i].Scale;
                double scaled = scale > 0 ? Math.Abs(actual - expected) / scale : 0;
                bool cancellation = scale > CancellationRatio * Math.Abs(expected);
                Assert.True(ordinary || (cancellation && scaled <= DifferentiationAssert.SixteenEpsilon),
                    $"{context}, relative {relative:E3}, {ulps} ulp, scaled {scaled:E3}");
                scaledOnly += ordinary ? 0 : 1;
                maximumScaledError = Math.Max(maximumScaledError, scaled);
                if (!cancellation)
                {
                    maximumUlps = Math.Max(maximumUlps, ulps);
                    maximumRelativeError = double.IsNormal(expected) ? Math.Max(maximumRelativeError, relative) : maximumRelativeError;
                }
            }
        }

        output.WriteLine(FormattableString.Invariant(
            $"{function}[{parameter}]: samples={rows.Length}, no cancellation: max relative error (normal)={maximumRelativeError:E3}, max ULP distance={maximumUlps}; max scaled error={maximumScaledError:E3}, passed only by the scaled criterion={scaledOnly}"));
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void SingleLocalDerivativesMatchDoubleAtTheSameInputs(string name)
    {
        // float に丸めた入力で double の勾配（参照表で検査済みの式）を求め、float に丸めたものと比べる。近似版も正確な版の局所微分の式を使う。
        double parameter = DefaultParameter(name);
        double[] inputs = Inputs(32).Concat(SingleTails()).Select(v => (double)(float)v).Distinct().ToArray();
        string exactName = name.Replace("Approximate", "", StringComparison.Ordinal);
        string doubleName = name == "GeluApproximate" ? "GeluTanh" : exactName;
        var expected = VectorGradient<double>(doubleName, inputs, parameter);
        var single = VectorGradient<float>(name, inputs, parameter);
        var singleView = ViewGradient<float>(name, inputs, parameter);
        for (int i = 0; i < inputs.Length; i++)
        {
            double scale = Scale(doubleName, inputs[i]);
            MatchesSingle((float)expected[i], single[i], scale, $"{name}({inputs[i]:R}) float");
            MatchesSingle((float)expected[i], singleView[i], scale, $"{name}({inputs[i]:R}) float view");
        }
    }

    [Theory]
    [InlineData(1e15f)]
    [InlineData(1e30f)]
    [InlineData(3e38f)]
    public void SingleEluWithLargeAlphaKeepsTheRepresentableTail(float alpha)
    {
        // α·eˣ が float で表せるのに e^(x + 64) がアンダーフローする負の裾（x = −170、α = 1e30 では約 1.5e−44）。
        // 期待値は double の α·eˣ（x ≥ −745 で Math.Exp は 0 にならない）を float に丸めたもの。double の大きな α は参照表で調べる。
        double[] inputs = [-0.5, -20, -86.5, -87.5, -100, -120, -150, -151, -160, -170, -175, -180, -190, -200, -1e30];
        var vector = VectorGradient<float>("Elu", inputs, alpha);
        var view = ViewGradient<float>("Elu", inputs, alpha);
        for (int i = 0; i < inputs.Length; i++)
        {
            float expected = (float)(alpha * Math.Exp(inputs[i]));
            MatchesSingle(expected, vector[i], 0, $"Elu[{alpha:R}]({inputs[i]:R}) float");
            MatchesSingle(expected, view[i], 0, $"Elu[{alpha:R}]({inputs[i]:R}) float view");
        }
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void ZeroLocalDerivativesIgnoreNonFiniteAdjoints(string name)
    {
        // 局所微分がちょうど 0 の要素（ReLU・Leaky ReLU（傾き 0）の負の側、ELU・SELU・GELU・SiLU の −∞）には、随伴が NaN でも寄与を足さない。
        double parameter = name == "LeakyRelu" ? 0 : DefaultParameter(name);
        double[] inputs = [-3, -0.5, double.NegativeInfinity, 2, -1e300];
        var tape = new Tape();
        var x = tape.Leaf(new LinearAlgebra.Vector<double>(inputs));
        var weights = new LinearAlgebra.Vector<double>([double.NaN, double.NaN, double.NaN, 1, double.NaN]);
        tape.Backward(XMath.Dot(Apply(name, x, parameter), tape.Constant(weights)));
        var gradient = tape.GetGradient(x);
        bool piecewise = name is "Relu" or "LeakyRelu";
        for (int i = 0; i < inputs.Length; i++)
        {
            bool zero = double.IsNegativeInfinity(inputs[i]) || (inputs[i] < 0 && (piecewise || inputs[i] < -1e3));
            if (zero)
            {
                Assert.True(gradient[i] == 0, $"{name}({inputs[i]:R}): {gradient[i]:R}");
            }
            else if (double.IsNaN(weights[i]))
            {
                Assert.True(double.IsNaN(gradient[i]), $"{name}({inputs[i]:R}): {gradient[i]:R}");
            }
        }

        // 随伴がちょうど 0 の要素は、局所微分を掛けない（NaN の入力でも寄与は 0）。
        var second = new Tape();
        var y = second.Leaf(new LinearAlgebra.Vector<double>([double.NaN, 1]));
        second.Backward(XMath.Dot(Apply(name, y, parameter), second.Constant(new LinearAlgebra.Vector<double>([0, 1]))));
        Assert.Equal(0, second.GetGradient(y)[0]);
    }

    [Fact]
    public void DerivativesAtTheKinksAndSpecialValuesFollowTheDocumentedSide()
    {
        double[] inputs = [0, -0.0, double.PositiveInfinity, double.NegativeInfinity, double.NaN];
        double[] Gradient(string name, double parameter) => VectorGradient<double>(name, inputs, parameter);
        // ReLU・Leaky ReLU・ELU・SELU の 0 は x ≤ 0 の側（左）。
        AssertSequence([0, 0, 1, 0, double.NaN], Gradient("Relu", 0), "Relu");
        AssertSequence([0.25, 0.25, 1, 0.25, double.NaN], Gradient("LeakyRelu", 0.25), "LeakyRelu");
        AssertSequence([0.5, 0.5, 1, 0, double.NaN], Gradient("Elu", 0.5), "Elu");
        AssertSequence([1.7580993408473768, 1.7580993408473768, 1.0507009873554805, 0, double.NaN], Gradient("Selu", 0), "Selu");
        AssertSequence([0.5, 0.5, 1, 0, double.NaN], Gradient("Gelu", 0), "Gelu");
        AssertSequence([0.5, 0.5, 1, 0, double.NaN], Gradient("GeluTanh", 0), "GeluTanh");
        AssertSequence([0.5, 0.5, 1, 0, double.NaN], Gradient("Silu", 0), "Silu");
    }

    [Fact]
    public void ActivationModelGradientMatchesCentralDifferences()
    {
        // 折れ目（0）を差分の幅より離した点で、ベクトルと行列（転置のビュー）の両方を通す。
        var random = new Random(41);
        const int length = 24;
        var x = new double[length];
        for (int i = 0; i < length; i++)
        {
            double v = 6 * random.NextDouble() - 3;
            x[i] = Math.Abs(v) < 0.05 ? v + 0.1 : v;
        }

        var c = Enumerable.Range(0, Names.Length).Select(_ => ArrayProductTests.RandomVector<double>(random, length)).ToArray();
        var d = ArrayProductTests.RandomVector<double>(random, 4);
        ScalarNode Model(Tape tape, VectorNode<double> v)
        {
            ScalarNode loss = 0;
            for (int k = 0; k < Names.Length; k++)
            {
                loss += XMath.Dot(tape.Constant(c[k]), Apply(Names[k], v, DefaultParameter(Names[k])));
            }

            return loss;
        }

        ArrayElementwiseGradientTests.AssertMatchesCentralDifferences(new LinearAlgebra.Vector<double>(x), Model, 1e-7);

        // 行列（6 × 4 の葉の転置のビュー）でも同じ。
        ScalarNode MatrixModel(Tape tape, MatrixNode<double> m)
        {
            ScalarNode loss = 0;
            for (int k = 0; k < Names.Length; k++)
            {
                loss += XMath.Sum(Apply(Names[k], m.T, DefaultParameter(Names[k])) * tape.Constant(d));
            }

            return loss;
        }

        var matrix = new Matrix<double>(4, 6);
        x.AsSpan().CopyTo(matrix.AsSpan());
        AssertMatrixMatchesCentralDifferences(matrix, MatrixModel, 1e-7);
    }

    private static void ValuesMatch<T>(string name, double[] inputs) where T : unmanaged, IFloatingPointIeee754<T>
    {
        double parameter = DefaultParameter(name);
        var data = new LinearAlgebra.Vector<double>(inputs).ConvertChecked<T>();
        var tape = new Tape();
        var leaf = tape.Leaf(data);
        AssertBits(Expected<T>(name, data, parameter), Apply(name, leaf, parameter).Value, $"{name} {typeof(T).Name} vector");
        AssertBits(Expected<T>(name, data, parameter), Apply(name, tape.Constant(data), parameter).Value, $"{name} {typeof(T).Name} constant");

        // 転置の部分行列のビューと、ベクトルを並べたブロードキャストのビュー。
        var storage = new Matrix<T>(9, 7);
        for (int k = 0; k < storage.Rows * storage.Columns; k++)
        {
            storage.AsSpan()[k] = data[k % data.Length];
        }

        var matrix = tape.Leaf(storage);
        var view = matrix[1..8, 2..6].T;
        AssertBits(Expected<T>(name, storage.AsMatrixSpan()[1..8, 2..6].T, parameter), Apply(name, view, parameter).Value, $"{name} {typeof(T).Name} view");
        var broadcast = leaf[..5].BroadcastRows(3);
        AssertBits(Expected<T>(name, data.AsVectorSpan()[..5].BroadcastRows(3), parameter), Apply(name, broadcast, parameter).Value,
            $"{name} {typeof(T).Name} broadcast");
    }

    // 入力を葉にして、出力の総和の勾配を返す。
    private static double[] VectorGradient<T>(string name, double[] inputs, double parameter) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var tape = new Tape();
        var x = tape.Leaf(new LinearAlgebra.Vector<double>(inputs).ConvertChecked<T>());
        tape.Backward(XMath.Sum(Apply(name, x, parameter)));
        return tape.GetGradient(x).ToArray().Select(v => double.CreateTruncating(v)).ToArray();
    }

    // 入力を大きい格納の転置の部分行列（一時領域に集めて勾配のビューへ書き戻す経路）に置き、出力の総和の勾配を返す。
    private static double[] ViewGradient<T>(string name, double[] inputs, double parameter) where T : unmanaged, IFloatingPointIeee754<T>
    {
        const int rows = 5;
        int columns = (inputs.Length + rows - 1) / rows;
        var storage = new Matrix<T>(columns + 3, rows + 4);
        for (int k = 0; k < rows * columns; k++)
        {
            storage[k / rows + 1, k % rows + 2] = k < inputs.Length ? T.CreateTruncating(inputs[k]) : T.One;
        }

        var tape = new Tape();
        var leaf = tape.Leaf(storage);
        var view = leaf[1..(columns + 1), 2..(rows + 2)].T;
        var y = Apply(name, view, parameter);
        var ones = new LinearAlgebra.Vector<T>(y.Columns);
        ones.AsSpan().Fill(T.One);
        tape.Backward(XMath.Sum(y * tape.Constant(ones)));
        var gradient = tape.GetGradient(view);
        var result = new double[inputs.Length];
        for (int k = 0; k < inputs.Length; k++)
        {
            result[k] = double.CreateTruncating(gradient[k % rows, k / rows]);
        }

        return result;
    }

    // 選んだ点（符号付きのゼロ、非正規化数、裾、導関数の零点の近く）と乱数の点。
    private static double[] Inputs(int seed)
    {
        var random = new Random(seed);
        var values = new List<double>
        {
            0, -0.0, double.Epsilon, -double.Epsilon, 1e-300, -1e-300, 0.5, -0.5, 1, -1, 2, -2, -0.7517915246935645, -0.7525, -1.2784645427610738,
            -5, -10, -20, -21.9, -25, -37.6, -38.6, -39, -87.5, -100, -104, -709, -720, -745, -750, -1000, 9, 10, 11, 40, 1e300,
            double.MaxValue, -double.MaxValue, double.PositiveInfinity, double.NegativeInfinity, double.NaN,
        };
        for (int i = 0; i < 40; i++)
        {
            values.Add(20 * random.NextDouble() - 10);
        }

        return values.ToArray();
    }

    // float の負の裾（指数が非正規化数になる x < −87 から、導関数が 0 に丸まる辺りまで）。
    private static double[] SingleTails() => [-86, -87.5, -88, -90, -95, -100, -103, -104, -105, -110, -150, -160, 88, 89];

    // 相殺する2項の絶対値の和（double で求めた目安）。相殺のない関数は 0。
    private static double Scale(string name, double x) => name switch
    {
        "Gelu" => 0.5 * XMath.Special.Erfc(-x / Math.Sqrt(2)) + Math.Abs(x) * Math.Exp(-0.5 * x * x) / Math.Sqrt(2 * Math.PI),
        "GeluTanh" => LogisticScale(x, 1.5957691216057308 * (x + 0.044715 * x * x * x), 1.5957691216057308 * (1 + 3 * 0.044715 * x * x)),
        "Silu" => LogisticScale(x, x, 1),
        _ => 0,
    };

    // σ(z) + |x·z′|·σ(z)σ(−z)。
    private static double LogisticScale(double x, double z, double slope)
    {
        double t = Math.Exp(-Math.Abs(z));
        double logistic = z >= 0 ? 1 / (1 + t) : t / (1 + t);
        return logistic + Math.Abs(x * slope) * t / ((1 + t) * (1 + t));
    }

    private static void MatchesSingle(float expected, double actualWide, double scale, string context)
    {
        float actual = (float)actualWide;
        if (float.IsNaN(expected) || float.IsNaN(actual))
        {
            Assert.True(float.IsNaN(expected) && float.IsNaN(actual), $"{context}: expected {expected:R}, actual {actual:R}");
            return;
        }

        if (expected == 0 || float.IsInfinity(expected))
        {
            Assert.True(actual == expected || Math.Abs(actual) <= 4 * float.Epsilon, $"{context}: expected {expected:R}, actual {actual:R}");
            return;
        }

        float relative = Math.Abs((actual - expected) / expected);
        int ulps = Math.Abs(BitConverter.SingleToInt32Bits(actual) - BitConverter.SingleToInt32Bits(expected));
        bool cancellation = scale > CancellationRatio * Math.Abs(expected);
        bool scaled = cancellation && Math.Abs(actual - expected) <= SingleSixteenEpsilon * scale;
        Assert.True(relative <= SingleSixteenEpsilon || ulps <= (int)UlpTolerance || scaled,
            $"{context}: expected {expected:R}, actual {actual:R}, relative {relative:E3}, {ulps} ulp, scale {scale:R}");
    }

    private static void AssertSequence(double[] expected, double[] actual, string context)
    {
        for (int i = 0; i < expected.Length; i++)
        {
            DifferentiationAssert.SameNumber(expected[i], actual[i], $"{context}[{i}]");
        }
    }

    private static void AssertBits<T>(ReadOnlyVectorSpan<T> expected, ReadOnlyVectorSpan<T> actual, string context) where T : unmanaged, IFloatingPointIeee754<T>
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(BitsOf(expected[i]) == BitsOf(actual[i]), $"{context}[{i}]: expected {expected[i]}, actual {actual[i]}");
        }
    }

    private static void AssertBits<T>(ReadOnlyMatrixSpan<T> expected, ReadOnlyMatrixSpan<T> actual, string context) where T : unmanaged, IFloatingPointIeee754<T>
    {
        Assert.Equal(expected.Rows, actual.Rows);
        Assert.Equal(expected.Columns, actual.Columns);
        for (int j = 0; j < expected.Columns; j++)
        {
            for (int i = 0; i < expected.Rows; i++)
            {
                Assert.True(BitsOf(expected[i, j]) == BitsOf(actual[i, j]), $"{context}[{i}, {j}]: expected {expected[i, j]}, actual {actual[i, j]}");
            }
        }
    }

    private static long BitsOf<T>(T value) where T : unmanaged, IFloatingPointIeee754<T>
        => typeof(T) == typeof(float) ? BitConverter.SingleToInt32Bits(float.CreateTruncating(value)) : BitConverter.DoubleToInt64Bits(double.CreateTruncating(value));

    internal static void AssertMatrixMatchesCentralDifferences(Matrix<double> point, Func<Tape, MatrixNode<double>, ScalarNode> model, double tolerance)
    {
        var tape = new Tape();
        var leaf = tape.Leaf(point);
        tape.Backward(model(tape, leaf));
        var gradient = tape.GetGradient(leaf).ToMatrix();
        for (int j = 0; j < point.Columns; j++)
        {
            for (int i = 0; i < point.Rows; i++)
            {
                double saved = point[i, j];
                double numeric = DifferentiationAssert.Richardson(t =>
                {
                    point[i, j] = t;
                    var evaluation = new Tape();
                    double value = model(evaluation, evaluation.Constant(point)).Value;
                    point[i, j] = saved;
                    return value;
                }, saved);
                DifferentiationAssert.Close(numeric, gradient[i, j], tolerance, tolerance, $"({i}, {j})");
            }
        }
    }
}
