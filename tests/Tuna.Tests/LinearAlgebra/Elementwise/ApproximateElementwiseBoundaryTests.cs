using System.Numerics;
using static Tuna.XMath;

namespace Tuna.Tests.LinearAlgebra;

using Tuna.LinearAlgebra;

[Trait("LinearAlgebra", "XMath")]
[Trait("Category", "IntegrationCoverage20261102")]
public class ApproximateElementwiseBoundaryTests
{
    private delegate Vector<T> AllocateVector<T>(ReadOnlyVectorSpan<T> x, bool approximate) where T : unmanaged, IFloatingPointIeee754<T>;
    private delegate void IntoVector<T>(ReadOnlyVectorSpan<T> x, bool approximate, VectorSpan<T> destination) where T : unmanaged, IFloatingPointIeee754<T>;
    private delegate void UpdateVector<T>(VectorSpan<T> x, bool approximate) where T : unmanaged, IFloatingPointIeee754<T>;
    private delegate Matrix<T> AllocateMatrix<T>(ReadOnlyMatrixSpan<T> x, bool approximate) where T : unmanaged, IFloatingPointIeee754<T>;
    private delegate void IntoMatrix<T>(ReadOnlyMatrixSpan<T> x, bool approximate, MatrixSpan<T> destination) where T : unmanaged, IFloatingPointIeee754<T>;
    private delegate void UpdateMatrix<T>(MatrixSpan<T> x, bool approximate) where T : unmanaged, IFloatingPointIeee754<T>;

    private sealed record Operation<T>(Func<T, bool, T>? Scalar, AllocateVector<T> Vector, IntoVector<T> VectorInto,
        UpdateVector<T> VectorInPlace, AllocateMatrix<T> Matrix, IntoMatrix<T> MatrixInto, UpdateMatrix<T> MatrixInPlace)
        where T : unmanaged, IFloatingPointIeee754<T>;

    private static Operation<T> GetOperation<T>(string name) where T : unmanaged, IFloatingPointIeee754<T> => name switch
    {
        "Exp" => new(null, (x, a) => Exp(x, a), (x, a, d) => Exp(x, a, d), (x, a) => InPlace.Exp(x, a),
            (x, a) => Exp(x, a), (x, a, d) => Exp(x, a, d), (x, a) => InPlace.Exp(x, a)),
        "Logistic" => new((x, a) => Logistic(x, a), (x, a) => Logistic(x, a), (x, a, d) => Logistic(x, a, d), (x, a) => InPlace.Logistic(x, a),
            (x, a) => Logistic(x, a), (x, a, d) => Logistic(x, a, d), (x, a) => InPlace.Logistic(x, a)),
        "Silu" => new((x, a) => Silu(x, a), (x, a) => Silu(x, a), (x, a, d) => Silu(x, a, d), (x, a) => InPlace.Silu(x, a),
            (x, a) => Silu(x, a), (x, a, d) => Silu(x, a, d), (x, a) => InPlace.Silu(x, a)),
        "Softplus" => new((x, a) => Softplus(x, a), (x, a) => Softplus(x, a), (x, a, d) => Softplus(x, a, d), (x, a) => InPlace.Softplus(x, a),
            (x, a) => Softplus(x, a), (x, a, d) => Softplus(x, a, d), (x, a) => InPlace.Softplus(x, a)),
        "Gelu" => new((x, a) => Gelu(x, a), (x, a) => Gelu(x, a), (x, a, d) => Gelu(x, a, d), (x, a) => InPlace.Gelu(x, a),
            (x, a) => Gelu(x, a), (x, a, d) => Gelu(x, a, d), (x, a) => InPlace.Gelu(x, a)),
        _ => throw new ArgumentException(name)
    };

    // Independent values from the defining real functions; no Tuna function supplies the expected value.
    // Regenerate only when needed: tools/SpecialFunctionReferences/generate_approximate_boundary_references.py.
    [Theory]
    // mpmath 1.3.0; exact binary32 inputs; 120/200-digit rounding agrees.
    [InlineData("Exp", -103.972084f, 7.006443546669571e-46, 1.07)]
    [InlineData("Exp", -103.972076f, 7.006497001795564e-46, 1.07)]
    [InlineData("Exp", -87.3365555f, 1.1754817373607687e-38, 1.07)]
    [InlineData("Exp", -87.3365402f, 1.1754996739254907e-38, 1.07)]
    [InlineData("Exp", -0.346573591f, 0.7071067805131506, 1.07)]
    [InlineData("Exp", 0f, 1.0, 1.07)]
    [InlineData("Exp", 0.346573591f, 1.414213563719889, 1.07)]
    [InlineData("Exp", 1f, 2.718281828459045, 1.07)]
    [InlineData("Exp", 88.7228317f, 3.4027985374118487e+38, 1.07)]
    [InlineData("Exp", 88.7228394f, 3.402824498803435e+38, 1.07)]
    [InlineData("Logistic", -103.972084f, 7.006443546669571e-46, 2.41)]
    [InlineData("Logistic", -103.972076f, 7.006497001795564e-46, 2.41)]
    [InlineData("Logistic", -100f, 3.720075976020836e-44, 2.41)]
    [InlineData("Logistic", -87.3365555f, 1.1754817373607687e-38, 2.41)]
    [InlineData("Logistic", -87.3365402f, 1.1754996739254907e-38, 2.41)]
    [InlineData("Logistic", -20f, 2.0611536181902037e-09, 2.41)]
    [InlineData("Logistic", 0f, 0.5, 2.41)]
    [InlineData("Logistic", 1.00000001e-10f, 0.500000000025, 2.41)]
    [InlineData("Logistic", 20f, 0.9999999979388464, 2.41)]
    [InlineData("Silu", -108.660309f, -7.0064560943784485e-46, 3.35)]
    [InlineData("Silu", -108.660301f, -7.006509057650296e-46, 3.35)]
    [InlineData("Silu", -105f, -2.631895849694951e-44, 3.35)]
    [InlineData("Silu", -91.856781f, -1.1754877452505621e-38, 3.35)]
    [InlineData("Silu", -91.8567734f, -1.175496615910731e-38, 3.35)]
    [InlineData("Silu", -90f, -7.374611361591464e-38, 3.35)]
    [InlineData("Silu", -1f, -0.2689414213699951, 3.35)]
    [InlineData("Silu", 0f, 0.0, 3.35)]
    [InlineData("Silu", 1f, 0.7310585786300049, 3.35)]
    [InlineData("Silu", 80f, 80.0, 3.35)]
    [InlineData("Softplus", -103.972084f, 7.006443546669571e-46, 2.8)]
    [InlineData("Softplus", -103.972076f, 7.006497001795564e-46, 2.8)]
    [InlineData("Softplus", -100f, 3.720075976020836e-44, 2.8)]
    [InlineData("Softplus", -87.3365555f, 1.1754817373607687e-38, 2.8)]
    [InlineData("Softplus", -87.3365402f, 1.1754996739254907e-38, 2.8)]
    [InlineData("Softplus", -20f, 2.061153620314381e-09, 2.8)]
    [InlineData("Softplus", 0f, 0.6931471805599453, 2.8)]
    [InlineData("Softplus", 20f, 20.000000002061153, 2.8)]
    [InlineData("Softplus", 80f, 80.0, 2.8)]
    [InlineData("Gelu", -10.5f, -7.42490055607904e-43, 203)]
    [InlineData("Gelu", -10.3000002f, -1.0281763755783076e-40, 203)]
    [InlineData("Gelu", -10.1006374f, -1.1754931884043879e-38, 203)]
    [InlineData("Gelu", -10.1006365f, -1.175519349493002e-38, 203)]
    [InlineData("Gelu", -10f, -1.204092348209806e-37, 203)]
    [InlineData("Gelu", -6f, -8.439646700762297e-11, 203)]
    [InlineData("Gelu", -3f, -0.003637392081773019, 203)]
    [InlineData("Gelu", -0.00100000005f, -0.0004996010814972463, 203)]
    [InlineData("Gelu", 0f, 0.0, 203)]
    [InlineData("Gelu", 1f, 0.8411919906082767, 203)]
    [InlineData("Gelu", 2.70000005f, 2.6911124542205034, 203)]
    [InlineData("Gelu", 3f, 2.996362607918227, 203)]
    public void SingleBoundaryValuesMatchIndependentReferences(string name, float input, double expected, double maxUlps)
    {
        var operation = GetOperation<float>(name);
        var vector = new Vector<float>(Enumerable.Repeat(input, 17).ToArray());
        if (operation.Scalar is not null)
        {
            Check(operation.Scalar(input, true));
        }

        foreach (float value in operation.Vector(vector, true).ToArray())
        {
            Check(value);
        }

        // 17 = one AVX512 block plus a tail; the diagonal forces the strided/scalar route.
        var matrix = DiagonalStorage(input);
        var original = matrix.ToArray();
        foreach (float value in operation.Vector(matrix.Diagonal(), true).ToArray())
        {
            Check(value);
        }

        var destination = DiagonalStorage(0);
        operation.VectorInto(matrix.Diagonal(), true, destination.Diagonal());
        CheckDiagonal(destination);
        Assert.Equal(original, matrix.ToArray());
        operation.VectorInPlace(matrix.Diagonal(), true);
        CheckDiagonal(matrix);
        Assert.All(vector.ToArray(), actual => Assert.Equal(input, actual));

        void Check(float actual) => CheckSingle(name, expected, actual, maxUlps);
        void CheckDiagonal(Matrix<float> result)
        {
            for (int row = 0; row < 17; row++)
            {
                for (int column = 0; column < 17; column++)
                {
                    if (row == column)
                    {
                        Check(result[row, column]);
                    }
                    else
                    {
                        Assert.Equal(12345f, result[row, column]);
                    }
                }
            }
        }
    }

    private static Matrix<float> DiagonalStorage(float value)
    {
        var matrix = new Matrix<float>(17, 17);
        for (int row = 0; row < 17; row++)
        {
            for (int column = 0; column < 17; column++)
            {
                matrix[row, column] = row == column ? value : 12345f;
            }
        }

        return matrix;
    }

    private static double SingleUlp(double value) => Math.Abs(value) < 1.1754943508222875e-38
        ? Math.ScaleB(1.0, -149) : Math.ScaleB(1.0, Math.ILogB(Math.Abs(value)) - 23);

    private static void CheckSingle(string name, double expected, float actual, double maxUlps)
    {
        if (!float.IsFinite((float)expected))
        {
            Assert.Equal((float)expected, actual);
            return;
        }

        Assert.True(float.IsFinite(actual), $"{name}: expected {expected:R}, actual {actual:R}");
        double error = Math.Abs(actual - expected) / SingleUlp(expected);
        Assert.True(error <= maxUlps, $"{name}: expected {expected:R}, actual {actual:R}, error {error:R} ulp");
    }

    [Theory]
    [InlineData("Exp", false, 0.049787068367863944)]
    [InlineData("Exp", true, 0.049787068367863944)]
    [InlineData("Logistic", false, 0.04742587317756678)]
    [InlineData("Logistic", true, 0.04742587317756678)]
    [InlineData("Silu", false, -0.14227761953270035)]
    [InlineData("Silu", true, -0.14227761953270035)]
    [InlineData("Softplus", false, 0.04858735157374206)]
    [InlineData("Softplus", true, 0.04858735157374206)]
    [InlineData("Gelu", false, -0.0040496940948902835)]
    [InlineData("Gelu", true, -0.003637392081773019)]
    public void PublicOverloadsUseTheSelectedFormula(string name, bool approximate, double expected)
    {
        CheckOverloads<float>(name, approximate, expected);
        CheckOverloads<double>(name, approximate, expected);
        CheckOverloads<Half>(name, approximate, expected);
    }

    // 現行の近似版と末尾のビットが異なる固定点。falseのAPIが既定の値を保つことを各版で確認する。
    [Theory]
    [InlineData("Exp", -4.9994998f, 0.00674131839f)]
    [InlineData("Logistic", -4.99989986f, 0.00669351686f)]
    [InlineData("Silu", -4.99980021f, -0.0334695578f)]
    [InlineData("Softplus", -4.99980021f, 0.00671668584f)]
    public void DefaultSingleOverloadsKeepTheirFixedValues(string name, float input, float expected) =>
        CheckOverloads<float>(name, false, expected, input, exactBits: true);

    private static void CheckOverloads<T>(string name, bool approximate, double expected, double input = -3, bool exactBits = false) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var operation = GetOperation<T>(name);
        T value = T.CreateTruncating(input);
        var x = new Vector<T>(Enumerable.Repeat(value, 17).ToArray());
        if (operation.Scalar is not null)
        {
            Check(operation.Scalar(value, approximate));
        }

        CheckVector(operation.Vector(x, approximate));
        var destination = new Vector<T>(17);
        operation.VectorInto(x, approximate, destination);
        CheckVector(destination);
        var inPlace = new Vector<T>(x.ToArray());
        operation.VectorInPlace(inPlace, approximate);
        CheckVector(inPlace);

        var a = new Matrix<T>(3, 7);
        for (int row = 0; row < a.Rows; row++)
        {
            for (int column = 0; column < a.Columns; column++)
            {
                a[row, column] = value;
            }
        }

        var result = operation.Matrix(a.T, approximate);
        Assert.Equal(7, result.Rows);
        Assert.Equal(3, result.Columns);
        CheckMatrix(result);
        var matrixDestination = new Matrix<T>(3, 7);
        operation.MatrixInto(a.T, approximate, matrixDestination.T);
        CheckMatrix(matrixDestination);
        var matrixInPlace = a.Clone();
        operation.MatrixInPlace(matrixInPlace.T, approximate);
        CheckMatrix(matrixInPlace);
        Assert.All(x.ToArray(), actual => Assert.Equal(value, actual));
        foreach (T actual in a.ToArray())
        {
            Assert.Equal(value, actual);
        }

        void CheckVector(Vector<T> values)
        {
            foreach (T actual in values.ToArray())
            {
                Check(actual);
            }
        }

        void CheckMatrix(Matrix<T> values)
        {
            foreach (T actual in values.ToArray())
            {
                Check(actual);
            }
        }

        void Check(T actual)
        {
            if (typeof(T) == typeof(float))
            {
                if (exactBits)
                {
                    Assert.Equal(BitConverter.SingleToInt32Bits((float)expected), BitConverter.SingleToInt32Bits(float.CreateTruncating(actual)));
                    return;
                }

                double ulps = !approximate ? 1 : name switch { "Exp" => 1.07, "Logistic" => 2.41, "Silu" => 3.35, "Softplus" => 2.8, _ => 203 };
                CheckSingle(name, expected, float.CreateTruncating(actual), ulps);
                return;
            }

            double tolerance = typeof(T) == typeof(double) ? Math.Max(2e-12 * Math.Abs(expected), 2 * double.Epsilon)
                : 2 * double.CreateTruncating(T.BitIncrement(T.CreateTruncating(Math.Abs(expected))) - T.CreateTruncating(Math.Abs(expected)));
            Assert.True(T.IsFinite(actual));
            Assert.True(Math.Abs(double.CreateTruncating(actual) - expected) <= tolerance,
                $"{name}<{typeof(T).Name}>, approximate={approximate}: expected {expected:R}, actual {actual}");
        }
    }

    [Theory]
    [InlineData("Exp", false)]
    [InlineData("Exp", true)]
    [InlineData("Logistic", false)]
    [InlineData("Logistic", true)]
    [InlineData("Silu", false)]
    [InlineData("Silu", true)]
    [InlineData("Softplus", false)]
    [InlineData("Softplus", true)]
    [InlineData("Gelu", false)]
    [InlineData("Gelu", true)]
    public void PartialOverlapIsRejectedBeforeWriting(string name, bool approximate)
    {
        CheckOverlap<float>(name, approximate);
        CheckOverlap<double>(name, approximate);
        CheckOverlap<Half>(name, approximate);
    }

    private static void CheckOverlap<T>(string name, bool approximate) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var operation = GetOperation<T>(name);
        T[] storage = Enumerable.Repeat(T.CreateTruncating(-3), 19).ToArray();
        var original = (T[])storage.Clone();
        Assert.Throws<ArgumentException>(() =>
            operation.VectorInto(new ReadOnlyVectorSpan<T>(storage.AsSpan(0, 17)), approximate, new VectorSpan<T>(storage.AsSpan(1, 17))));
        Assert.Equal(original, storage);
        Assert.Throws<ArgumentException>(() => operation.MatrixInto(
            new ReadOnlyMatrixSpan<T>(storage.AsSpan(0, 18), new nint[] { 3, 6 }, new nint[] { 1, 3 }), approximate,
            new MatrixSpan<T>(storage.AsSpan(1, 18), new nint[] { 3, 6 }, new nint[] { 1, 3 })));
        Assert.Equal(original, storage);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void EmptyShapesPreserveTheirShape(bool approximate)
    {
        CheckEmpty<float>(approximate);
        CheckEmpty<double>(approximate);
        CheckEmpty<Half>(approximate);
    }

    private static void CheckEmpty<T>(bool approximate) where T : unmanaged, IFloatingPointIeee754<T>
    {
        foreach (string name in new[] { "Exp", "Logistic", "Silu", "Softplus", "Gelu" })
        {
            var operation = GetOperation<T>(name);
            var vector = new Vector<T>(0);
            Assert.Empty(operation.Vector(vector, approximate).ToArray());
            operation.VectorInto(vector, approximate, vector);
            operation.VectorInPlace(vector, approximate);
            foreach (var (rows, columns) in new[] { (0, 0), (0, 3), (3, 0) })
            {
                var matrix = new Matrix<T>(rows, columns);
                var result = operation.Matrix(matrix, approximate);
                Assert.Equal(rows, result.Rows);
                Assert.Equal(columns, result.Columns);
                operation.MatrixInto(matrix, approximate, matrix);
                operation.MatrixInPlace(matrix, approximate);
            }
        }
    }
}
