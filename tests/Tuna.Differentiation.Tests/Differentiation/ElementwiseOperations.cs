using System.Numerics;
using System.Runtime.InteropServices;
using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.Tests.Differentiation;

/// <summary>要素演算のテストで使う、演算の名前から記録つきの版・XMath の版・スカラーの記録の版への対応</summary>
internal static class ElementwiseOperations
{
    // 近似版は名前の末尾に Approximate を付ける。
    internal static readonly string[] Unary =
    [
        "Negate", "Exp", "ExpApproximate", "Log", "Sqrt", "Square", "Abs", "Expm1", "Log1p", "Logistic", "LogisticApproximate",
        "LogLogistic", "Logit", "Softplus", "SoftplusApproximate", "Log1mExp", "Reciprocal", "Tanh", "TanhApproximate",
    ];

    internal static readonly string[] Binary = ["Add", "Subtract", "Hadamard", "Divide", "Min", "Max", "Pow", "Hypot", "LogAddExp", "LogSubExp"];

    // ScalarSubtract は演算子 s − x。
    internal static readonly string[] WithScalar =
        ["Add", "Subtract", "Multiply", "Divide", "Min", "Max", "Pow", "Hypot", "LogAddExp", "LogSubExp", "ScalarSubtract"];

    internal static TheoryData<string> UnaryNames() => new(Unary);

    internal static TheoryData<string> BinaryNames() => new(Binary);

    internal static TheoryData<string> ScalarNames() => new(WithScalar);

    internal static VectorNode<T> Apply<T>(string name, VectorNode<T> x) where T : unmanaged, IFloatingPointIeee754<T> => name switch
    {
        "Negate" => -x,
        "Exp" => XMath.Exp(x),
        "ExpApproximate" => XMath.Exp(x, approximate: true),
        "Log" => XMath.Log(x),
        "Sqrt" => XMath.Sqrt(x),
        "Square" => XMath.Square(x),
        "Abs" => XMath.Abs(x),
        "Expm1" => XMath.Expm1(x),
        "Log1p" => XMath.Log1p(x),
        "Tanh" => XMath.Tanh(x),
        "TanhApproximate" => XMath.Tanh(x, approximate: true),
        "Logistic" => XMath.Logistic(x),
        "LogisticApproximate" => XMath.Logistic(x, approximate: true),
        "LogLogistic" => XMath.LogLogistic(x),
        "Logit" => XMath.Logit(x),
        "Softplus" => XMath.Softplus(x),
        "SoftplusApproximate" => XMath.Softplus(x, approximate: true),
        "Log1mExp" => XMath.Log1mExp(x),
        "Reciprocal" => XMath.Reciprocal(x),
        _ => throw new ArgumentOutOfRangeException(nameof(name)),
    };

    internal static MatrixNode<T> Apply<T>(string name, MatrixNode<T> x) where T : unmanaged, IFloatingPointIeee754<T> => name switch
    {
        "Negate" => -x,
        "Exp" => XMath.Exp(x),
        "ExpApproximate" => XMath.Exp(x, approximate: true),
        "Log" => XMath.Log(x),
        "Sqrt" => XMath.Sqrt(x),
        "Square" => XMath.Square(x),
        "Abs" => XMath.Abs(x),
        "Expm1" => XMath.Expm1(x),
        "Log1p" => XMath.Log1p(x),
        "Tanh" => XMath.Tanh(x),
        "TanhApproximate" => XMath.Tanh(x, approximate: true),
        "Logistic" => XMath.Logistic(x),
        "LogisticApproximate" => XMath.Logistic(x, approximate: true),
        "LogLogistic" => XMath.LogLogistic(x),
        "Logit" => XMath.Logit(x),
        "Softplus" => XMath.Softplus(x),
        "SoftplusApproximate" => XMath.Softplus(x, approximate: true),
        "Log1mExp" => XMath.Log1mExp(x),
        "Reciprocal" => XMath.Reciprocal(x),
        _ => throw new ArgumentOutOfRangeException(nameof(name)),
    };

    internal static LinearAlgebra.Vector<T> Expected<T>(string name, ReadOnlyVectorSpan<T> x) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var result = new LinearAlgebra.Vector<T>(x.Length);
        VectorSpan<T> d = result;
        switch (name)
        {
            case "Negate": XMath.Negate(x, d); break;
            case "Exp": XMath.Exp(x, d); break;
            case "ExpApproximate": XMath.Exp(x, true, d); break;
            case "Log": XMath.Log(x, d); break;
            case "Sqrt": XMath.Sqrt(x, d); break;
            case "Square": XMath.Square(x, d); break;
            case "Abs": XMath.Abs(x, d); break;
            case "Expm1": XMath.Expm1(x, d); break;
            case "Log1p": XMath.Log1p(x, d); break;
            case "Tanh": XMath.Tanh(x, d); break;
            case "TanhApproximate": XMath.Tanh(x, true, d); break;
            case "Logistic": XMath.Logistic(x, d); break;
            case "LogisticApproximate": XMath.Logistic(x, true, d); break;
            case "LogLogistic": XMath.LogLogistic(x, d); break;
            case "Logit": XMath.Logit(x, d); break;
            case "Softplus": XMath.Softplus(x, d); break;
            case "SoftplusApproximate": XMath.Softplus(x, true, d); break;
            case "Log1mExp": XMath.Log1mExp(x, d); break;
            case "Reciprocal": XMath.Reciprocal(x, d); break;
            default: throw new ArgumentOutOfRangeException(nameof(name));
        }

        return result;
    }

    internal static Matrix<T> Expected<T>(string name, ReadOnlyMatrixSpan<T> x) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var result = new Matrix<T>(x.Rows, x.Columns);
        MatrixSpan<T> d = result;
        switch (name)
        {
            case "Negate": XMath.Negate(x, d); break;
            case "Exp": XMath.Exp(x, d); break;
            case "ExpApproximate": XMath.Exp(x, true, d); break;
            case "Log": XMath.Log(x, d); break;
            case "Sqrt": XMath.Sqrt(x, d); break;
            case "Square": XMath.Square(x, d); break;
            case "Abs": XMath.Abs(x, d); break;
            case "Expm1": XMath.Expm1(x, d); break;
            case "Log1p": XMath.Log1p(x, d); break;
            case "Tanh": XMath.Tanh(x, d); break;
            case "TanhApproximate": XMath.Tanh(x, true, d); break;
            case "Logistic": XMath.Logistic(x, d); break;
            case "LogisticApproximate": XMath.Logistic(x, true, d); break;
            case "LogLogistic": XMath.LogLogistic(x, d); break;
            case "Logit": XMath.Logit(x, d); break;
            case "Softplus": XMath.Softplus(x, d); break;
            case "SoftplusApproximate": XMath.Softplus(x, true, d); break;
            case "Log1mExp": XMath.Log1mExp(x, d); break;
            case "Reciprocal": XMath.Reciprocal(x, d); break;
            default: throw new ArgumentOutOfRangeException(nameof(name));
        }

        return result;
    }

    // スカラーの記録（D1）。近似版は正確な版の局所微分と比べる。
    internal static ScalarNode Scalar(string name, ScalarNode x) => name switch
    {
        "Negate" => -x,
        "Exp" or "ExpApproximate" => XMath.Exp(x),
        "Log" => XMath.Log(x),
        "Sqrt" => XMath.Sqrt(x),
        "Square" => XMath.Square(x),
        "Abs" => XMath.Abs(x),
        "Expm1" => XMath.Expm1(x),
        "Log1p" => XMath.Log1p(x),
        "Tanh" or "TanhApproximate" => XMath.Tanh(x),
        "Logistic" or "LogisticApproximate" => XMath.Logistic(x),
        "LogLogistic" => XMath.LogLogistic(x),
        "Logit" => XMath.Logit(x),
        "Softplus" or "SoftplusApproximate" => XMath.Softplus(x),
        "Log1mExp" => XMath.Log1mExp(x),
        "Reciprocal" => 1 / x,
        _ => throw new ArgumentOutOfRangeException(nameof(name)),
    };

    internal static VectorNode<T> Apply<T>(string name, VectorNode<T> x, VectorNode<T> y) where T : unmanaged, IFloatingPointIeee754<T> => name switch
    {
        "Add" => x + y,
        "Subtract" => x - y,
        "Hadamard" => XMath.Hadamard(x, y),
        "Divide" => XMath.Divide(x, y),
        "Min" => XMath.Min(x, y),
        "Max" => XMath.Max(x, y),
        "Pow" => XMath.Pow(x, y),
        "Hypot" => XMath.Hypot(x, y),
        "LogAddExp" => XMath.LogAddExp(x, y),
        "LogSubExp" => XMath.LogSubExp(x, y),
        _ => throw new ArgumentOutOfRangeException(nameof(name)),
    };

    internal static MatrixNode<T> Apply<T>(string name, MatrixNode<T> x, MatrixNode<T> y) where T : unmanaged, IFloatingPointIeee754<T> => name switch
    {
        "Add" => x + y,
        "Subtract" => x - y,
        "Hadamard" => XMath.Hadamard(x, y),
        "Divide" => XMath.Divide(x, y),
        "Min" => XMath.Min(x, y),
        "Max" => XMath.Max(x, y),
        "Pow" => XMath.Pow(x, y),
        "Hypot" => XMath.Hypot(x, y),
        "LogAddExp" => XMath.LogAddExp(x, y),
        "LogSubExp" => XMath.LogSubExp(x, y),
        _ => throw new ArgumentOutOfRangeException(nameof(name)),
    };

    internal static LinearAlgebra.Vector<T> Expected<T>(string name, ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        var result = new LinearAlgebra.Vector<T>(x.Length);
        VectorSpan<T> d = result;
        switch (name)
        {
            case "Add": XMath.Add(x, y, d); break;
            case "Subtract": XMath.Subtract(x, y, d); break;
            case "Hadamard": XMath.Hadamard(x, y, d); break;
            case "Divide": XMath.Divide(x, y, d); break;
            case "Min": XMath.Min(x, y, d); break;
            case "Max": XMath.Max(x, y, d); break;
            case "Pow": XMath.Pow(x, y, d); break;
            case "Hypot": XMath.Hypot(x, y, d); break;
            case "LogAddExp": XMath.LogAddExp(x, y, d); break;
            case "LogSubExp": XMath.LogSubExp(x, y, d); break;
            default: throw new ArgumentOutOfRangeException(nameof(name));
        }

        return result;
    }

    internal static Matrix<T> Expected<T>(string name, ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var result = new Matrix<T>(x.Rows, x.Columns);
        MatrixSpan<T> d = result;
        switch (name)
        {
            case "Add": XMath.Add(x, y, d); break;
            case "Subtract": XMath.Subtract(x, y, d); break;
            case "Hadamard": XMath.Hadamard(x, y, d); break;
            case "Divide": XMath.Divide(x, y, d); break;
            case "Min": XMath.Min(x, y, d); break;
            case "Max": XMath.Max(x, y, d); break;
            case "Pow": XMath.Pow(x, y, d); break;
            case "Hypot": XMath.Hypot(x, y, d); break;
            case "LogAddExp": XMath.LogAddExp(x, y, d); break;
            case "LogSubExp": XMath.LogSubExp(x, y, d); break;
            default: throw new ArgumentOutOfRangeException(nameof(name));
        }

        return result;
    }

    internal static ScalarNode Scalar(string name, ScalarNode x, ScalarNode y) => name switch
    {
        "Add" => x + y,
        "Subtract" => x - y,
        "Hadamard" or "Multiply" => x * y,
        "Divide" => x / y,
        "Min" => XMath.Min(x, y),
        "Max" => XMath.Max(x, y),
        "Pow" => XMath.Pow(x, y),
        "Hypot" => XMath.Hypot(x, y),
        "LogAddExp" => XMath.LogAddExp(x, y),
        "LogSubExp" => XMath.LogSubExp(x, y),
        "ScalarSubtract" => y - x,
        _ => throw new ArgumentOutOfRangeException(nameof(name)),
    };

    internal static VectorNode<T> Apply<T>(string name, VectorNode<T> x, ScalarNode s) where T : unmanaged, IFloatingPointIeee754<T> => name switch
    {
        "Add" => XMath.Add(x, s),
        "Subtract" => XMath.Subtract(x, s),
        "Multiply" => XMath.Multiply(x, s),
        "Divide" => XMath.Divide(x, s),
        "Min" => XMath.Min(x, s),
        "Max" => XMath.Max(x, s),
        "Pow" => XMath.Pow(x, s),
        "Hypot" => XMath.Hypot(x, s),
        "LogAddExp" => XMath.LogAddExp(x, s),
        "LogSubExp" => XMath.LogSubExp(x, s),
        "ScalarSubtract" => s - x,
        _ => throw new ArgumentOutOfRangeException(nameof(name)),
    };

    internal static MatrixNode<T> Apply<T>(string name, MatrixNode<T> x, ScalarNode s) where T : unmanaged, IFloatingPointIeee754<T> => name switch
    {
        "Add" => XMath.Add(x, s),
        "Subtract" => XMath.Subtract(x, s),
        "Multiply" => XMath.Multiply(x, s),
        "Divide" => XMath.Divide(x, s),
        "Min" => XMath.Min(x, s),
        "Max" => XMath.Max(x, s),
        "Pow" => XMath.Pow(x, s),
        "Hypot" => XMath.Hypot(x, s),
        "LogAddExp" => XMath.LogAddExp(x, s),
        "LogSubExp" => XMath.LogSubExp(x, s),
        "ScalarSubtract" => s - x,
        _ => throw new ArgumentOutOfRangeException(nameof(name)),
    };

    internal static LinearAlgebra.Vector<T> Expected<T>(string name, ReadOnlyVectorSpan<T> x, T s) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var result = new LinearAlgebra.Vector<T>(x.Length);
        VectorSpan<T> d = result;
        switch (name)
        {
            case "Add": XMath.Add(x, s, d); break;
            case "Subtract": XMath.Subtract(x, s, d); break;
            case "Multiply": XMath.Multiply(x, s, d); break;
            case "Divide": XMath.Divide(x, s, d); break;
            case "Min": XMath.Min(x, s, d); break;
            case "Max": XMath.Max(x, s, d); break;
            case "Pow": XMath.Pow(x, s, d); break;
            case "Hypot": XMath.Hypot(x, s, d); break;
            case "LogAddExp": XMath.LogAddExp(x, s, d); break;
            case "LogSubExp": XMath.LogSubExp(x, s, d); break;
            case "ScalarSubtract":
                // 文書に書いた求め方。NaN の符号のほかは IEEE 754 の s − x と同じ（ScalarSubtractIsTheIeeeDifference）。
                XMath.Negate(x, d);
                XMath.Add(d.AsReadOnly(), s, d);
                break;
            default: throw new ArgumentOutOfRangeException(nameof(name));
        }

        return result;
    }

    internal static Matrix<T> Expected<T>(string name, ReadOnlyMatrixSpan<T> x, T s) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var result = new Matrix<T>(x.Rows, x.Columns);
        MatrixSpan<T> d = result;
        switch (name)
        {
            case "Add": XMath.Add(x, s, d); break;
            case "Subtract": XMath.Subtract(x, s, d); break;
            case "Multiply": XMath.Multiply(x, s, d); break;
            case "Divide": XMath.Divide(x, s, d); break;
            case "Min": XMath.Min(x, s, d); break;
            case "Max": XMath.Max(x, s, d); break;
            case "Pow": XMath.Pow(x, s, d); break;
            case "Hypot": XMath.Hypot(x, s, d); break;
            case "LogAddExp": XMath.LogAddExp(x, s, d); break;
            case "LogSubExp": XMath.LogSubExp(x, s, d); break;
            case "ScalarSubtract":
                XMath.Negate(x, d);
                XMath.Add(d.AsReadOnly(), s, d);
                break;
            default: throw new ArgumentOutOfRangeException(nameof(name));
        }

        return result;
    }

    internal static void AssertSameBits<T>(ReadOnlySpan<T> expected, ReadOnlySpan<T> actual, string context) where T : unmanaged
    {
        Assert.True(expected.Length == actual.Length, $"{context}: length {expected.Length} != {actual.Length}");
        ReadOnlySpan<byte> e = MemoryMarshal.AsBytes(expected), a = MemoryMarshal.AsBytes(actual);
        int size = e.Length / Math.Max(1, expected.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(e.Slice(i * size, size).SequenceEqual(a.Slice(i * size, size)),
                $"{context}[{i}]: expected {expected[i]}, actual {actual[i]}");
        }
    }

    // 入力の例。乱数の点と、0・非正規化数・定義域の端・裾・±∞・NaN を混ぜる。
    internal static double[] Inputs(int seed, int randomCount, double scale)
    {
        var random = new Random(seed);
        double[] special =
        [
            0.0, -0.0, 5e-324, -5e-324, 1e-310, -1e-310, 2.2250738585072014e-308, -2.2250738585072014e-308, 1e-300, -1e-300,
            1e-16, -1e-16, 1e-8, -1e-8, 0.25, -0.25, 0.5, -0.5, 0.6931471805599453, -0.6931471805599453, -0.6931471805599454,
            0.75, 0.9999999999999999, 1.0, -1.0, -1.0000000000000002, -0.9999999999999999, 1.5, 2.0, -2.0, 3.0, 10.0, -10.0,
            37.5, -37.5, 88.5, -88.5, 104.0, -104.0, 709.0, -709.0, 710.0, -710.0, 745.5, -745.5, 1e300, -1e300,
            double.PositiveInfinity, double.NegativeInfinity, double.NaN,
        ];
        var values = new double[special.Length + randomCount];
        special.CopyTo(values, 0);
        for (int i = special.Length; i < values.Length; i++)
        {
            values[i] = scale * (2 * random.NextDouble() - 1);
        }

        return values;
    }
}
