using Tuna.LinearAlgebra;
using Tuna.SignalProcessing;
using static Tuna.Tests.SignalProcessing.FourierReference;

namespace Tuna.Tests.SignalProcessing;

/// <summary>DCT・DST（type I〜IV）: SciPy との照合、二重倍精度の定義式との誤差、逆変換、直交性、行・列の一括変換、例外</summary>
[Trait("Fourier", "Trigonometric")]
public class TrigonometricTransformTests
{
    private static readonly FourierNormalization[] Normalizations =
        [FourierNormalization.Backward, FourierNormalization.Orthonormal, FourierNormalization.Forward];

    /// <summary>実数 FFT の偶奇、DCT-IV の偶奇、Bluestein 法を通る長さ</summary>
    public static TheoryData<int> Lengths => new() { 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 31, 32, 37, 64, 100, 127, 128, 255, 256, 1000, 1009, 1024 };

    private static Vector<double> Transform(bool sine, int type, bool inverse, ReadOnlyVectorSpan<double> x, FourierNormalization normalization) =>
        (sine, inverse) switch
        {
            (false, false) => Fourier.Cosine(x, type, normalization),
            (false, true) => Fourier.InverseCosine(x, type, normalization),
            (true, false) => Fourier.Sine(x, type, normalization),
            _ => Fourier.InverseSine(x, type, normalization),
        };

    private static string Name(bool sine, int type) => $"{(sine ? "DST" : "DCT")}-{type}";

    private static void AssertBitwiseEqual(ReadOnlyVectorSpan<double> expected, ReadOnlyVectorSpan<double> actual, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            if (BitConverter.DoubleToInt64Bits(expected[i]) != BitConverter.DoubleToInt64Bits(actual[i]))
            {
                Assert.Fail($"{label}[{i}] expected {expected[i]:R}, actual {actual[i]:R}");
            }
        }
    }

    // ---- SciPy との照合 ----

    [Fact]
    public void MatchesSciPy()
    {
        foreach (var (sine, type, normalization, inverse, x, expected) in SciPyReferences.Trigonometric)
        {
            var actual = Transform(sine, type, inverse, x, normalization);
            double scale = Math.Max(1, expected.Max(Math.Abs));
            for (int k = 0; k < expected.Length; k++)
            {
                Assert.True(
                    Math.Abs(actual[k] - expected[k]) <= 1e-14 * scale,
                    $"{(inverse ? "inverse " : "")}{Name(sine, type)} {normalization} n={x.Length} [{k}]: {actual[k]:R} vs {expected[k]:R}");
            }
        }
    }

    // ---- 二重倍精度の定義式との誤差 ----

    /// <summary>
    /// 倍率なしの定義式を二重倍精度で直接計算する。cos・sin の引数はどれも π の有理数倍なので、
    /// exp(−2πi·r/M) の表（<see cref="FourierReference.Root"/>）から引く
    /// </summary>
    private static DoubleDouble[] Definition(bool sine, int type, double[] x)
    {
        int n = x.Length;
        long modulus = (sine, type) switch
        {
            (false, 1) => 2L * (n - 1),
            (true, 1) => 2L * (n + 1),
            (_, 4) => 8L * n,
            _ => 4L * n,
        };
        var table = new (DoubleDouble Re, DoubleDouble Im)[modulus];
        for (long r = 0; r < modulus; r++)
        {
            table[r] = Root(r, modulus);
        }

        // cos(2πr/M) = Re、sin(2πr/M) = −Im
        DoubleDouble Cos(long r) => table[r % modulus].Re;
        DoubleDouble Sin(long r) => -table[r % modulus].Im;

        var result = new DoubleDouble[n];
        for (int k = 0; k < n; k++)
        {
            DoubleDouble sum = 0;
            for (int j = 0; j < n; j++)
            {
                DoubleDouble term = (sine, type) switch
                {
                    (false, 1) => j == 0 ? 1.0 : j == n - 1 ? (k % 2 == 0 ? 1.0 : -1.0) : 2.0 * Cos((long)k * j),
                    (false, 2) => 2.0 * Cos((long)k * (2 * j + 1)),
                    (false, 3) => j == 0 ? 1.0 : 2.0 * Cos((long)j * (2 * k + 1)),
                    (false, _) => 2.0 * Cos((long)(2 * j + 1) * (2 * k + 1)),
                    (true, 1) => 2.0 * Sin((long)(k + 1) * (j + 1)),
                    (true, 2) => 2.0 * Sin((long)(k + 1) * (2 * j + 1)),
                    (true, 3) => j == n - 1 ? (k % 2 == 0 ? 1.0 : -1.0) : 2.0 * Sin((long)(2 * k + 1) * (j + 1)),
                    (true, _) => 2.0 * Sin((long)(2 * j + 1) * (2 * k + 1)),
                };
                sum += term * x[j];
            }

            result[k] = sum;
        }

        return result;
    }

    [Theory]
    [MemberData(nameof(Lengths))]
    public void AccuracyAgainstDoubleDoubleDefinition(int n)
    {
        // 相対 RMS 誤差の上限（ε 単位）。FFT のテストの 2(1 + log₂ n) に前後の処理の分を足した
        double bound = 4 * (1 + Math.Log2(n));
        var failures = new List<string>();
        foreach (bool sine in new[] { false, true })
        {
            for (int type = 1; type <= 4; type++)
            {
                var x = RandomReal(n, 1000 * n + 10 * type + (sine ? 1 : 0));
                var expected = Definition(sine, type, x);
                var actual = sine ? Fourier.Sine(x, type) : Fourier.Cosine(x, type);
                double error = 0, norm = 0;
                for (int k = 0; k < n; k++)
                {
                    double e = (new DoubleDouble(actual[k]) - expected[k]).ToDouble();
                    double r = expected[k].ToDouble();
                    error += e * e;
                    norm += r * r;
                }

                double relative = Math.Sqrt(error / norm) / Epsilon;
                if (relative > bound)
                {
                    failures.Add($"{Name(sine, type)} n={n}: {relative:F2} ε > {bound:F2} ε");
                }
            }
        }

        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    // ---- 逆変換と直交性 ----

    [Theory]
    [MemberData(nameof(Lengths))]
    public void InverseRestoresInput(int n)
    {
        foreach (bool sine in new[] { false, true })
        {
            for (int type = 1; type <= 4; type++)
            {
                var x = RandomReal(n, 7 * n + type);
                foreach (var normalization in Normalizations)
                {
                    var y = Transform(sine, type, false, x, normalization);
                    var back = Transform(sine, type, true, y, normalization);
                    for (int j = 0; j < n; j++)
                    {
                        Assert.True(
                            Math.Abs(back[j] - x[j]) <= 1e-13 * (1 + Math.Abs(x[j])),
                            $"{Name(sine, type)} {normalization} n={n} [{j}]: {back[j]:R} vs {x[j]:R}");
                    }
                }
            }
        }
    }

    [Theory]
    [InlineData(2)]
    [InlineData(6)]
    [InlineData(7)]
    [InlineData(16)]
    public void OrthonormalTransformsAreOrthogonalAndInverseIsTranspose(int n)
    {
        foreach (bool sine in new[] { false, true })
        {
            for (int type = 1; type <= 4; type++)
            {
                // 単位ベクトルの変換を並べた行列 M（第 j 列 = M·e_j）と、逆変換の行列 N
                var m = new Matrix<double>(n, n);
                var inverse = new Matrix<double>(n, n);
                for (int j = 0; j < n; j++)
                {
                    var e = new Vector<double>(n);
                    e[j] = 1;
                    Transform(sine, type, false, e, FourierNormalization.Orthonormal).AsVectorSpan().CopyTo(m.Column(j));
                    Transform(sine, type, true, e, FourierNormalization.Orthonormal).AsVectorSpan().CopyTo(inverse.Column(j));
                }

                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        double dot = 0;
                        for (int k = 0; k < n; k++)
                        {
                            dot += m[k, i] * m[k, j];
                        }

                        Assert.True(Math.Abs(dot - (i == j ? 1 : 0)) < 1e-14, $"{Name(sine, type)} n={n}: (MᵀM)[{i},{j}] = {dot:R}");
                        Assert.True(Math.Abs(inverse[i, j] - m[j, i]) < 1e-15, $"{Name(sine, type)} n={n}: N[{i},{j}] ≠ M[{j},{i}]");
                    }
                }
            }
        }
    }

    [Fact]
    public void TypeTwoAndThreeAreInversesUpToScale()
    {
        // idct(type=2) は dct(type=3)/(2n)、idct(type=3) は dct(type=2)/(2n)（倍率なしの組）
        int n = 12;
        var x = RandomReal(n, 3);
        var inverse2 = Fourier.InverseCosine(x, 2);
        var inverse3 = Fourier.InverseCosine(x, 3);
        var forward3 = Fourier.Cosine(x, 3);
        var forward2 = Fourier.Cosine(x, 2);
        for (int k = 0; k < n; k++)
        {
            Assert.Equal(forward3[k] / (2 * n), inverse2[k], 15);
            Assert.Equal(forward2[k] / (2 * n), inverse3[k], 15);
        }
    }

    [Fact]
    public void KnownValues()
    {
        // 定数列の DCT-II は第0係数だけ、DCT-I の両端の重み
        var constant = Fourier.Cosine(new double[] { 1, 1, 1, 1 }, 2);
        Assert.Equal(8, constant[0], 14);
        for (int k = 1; k < 4; k++)
        {
            Assert.Equal(0, constant[k], 14);
        }

        var two = Fourier.Cosine(new double[] { 3, 5 }, 1);
        Assert.Equal(8, two[0], 15);
        Assert.Equal(-2, two[1], 15);
        Assert.Equal(6, Fourier.Cosine(new double[] { 3 }, 2)[0]);
        Assert.Equal(3, Fourier.Cosine(new double[] { 3 }, 3)[0]);
        Assert.Equal(3 * Math.Sqrt(2), Fourier.Cosine(new double[] { 3 }, 4)[0], 15);
        Assert.Equal(6, Fourier.Sine(new double[] { 3 }, 1)[0], 15);
    }

    // ---- 行・列の一括変換、ビューの配置、その場更新 ----

    [Fact]
    public void RowsColumnsStridedAndInPlaceMatchVectorTransformsBitwise()
    {
        foreach (int n in new[] { 2, 5, 8, 33 })
        {
            int rows = 5;
            var x = Matrix<double>.FromColumnMajor(rows, n, RandomReal(rows * n, n));
            foreach (bool sine in new[] { false, true })
            {
                for (int type = 1; type <= 4; type++)
                {
                    foreach (var normalization in Normalizations)
                    {
                        string label = $"{Name(sine, type)} {normalization} n={n}";
                        var byRows = sine ? Fourier.SineRows(x, type, normalization) : Fourier.CosineRows(x, type, normalization);
                        var byColumns = sine ? Fourier.SineColumns(x.T, type, normalization) : Fourier.CosineColumns(x.T, type, normalization);
                        var inverseRows = sine ? Fourier.InverseSineRows(x, type, normalization) : Fourier.InverseCosineRows(x, type, normalization);
                        var inverseColumns = sine
                            ? Fourier.InverseSineColumns(x.T, type, normalization)
                            : Fourier.InverseCosineColumns(x.T, type, normalization);
                        for (int i = 0; i < rows; i++)
                        {
                            var row = x.Row(i).ToVector();
                            var expected = Transform(sine, type, false, row, normalization);
                            AssertBitwiseEqual(expected, byRows.Row(i), $"{label} row {i}");
                            AssertBitwiseEqual(expected, byColumns.Column(i), $"{label} column {i}");
                            AssertBitwiseEqual(expected, Transform(sine, type, false, x.Row(i), normalization), $"{label} strided row {i}");
                            var expectedInverse = Transform(sine, type, true, row, normalization);
                            AssertBitwiseEqual(expectedInverse, inverseRows.Row(i), $"{label} inverse row {i}");
                            AssertBitwiseEqual(expectedInverse, inverseColumns.Column(i), $"{label} inverse column {i}");

                            // 出力先が入力と同じビュー
                            var inPlace = x.Row(i).ToVector();
                            if (sine)
                            {
                                Fourier.Sine(inPlace, type, normalization, inPlace);
                            }
                            else
                            {
                                Fourier.Cosine(inPlace, type, normalization, inPlace);
                            }

                            AssertBitwiseEqual(expected, inPlace, $"{label} in place {i}");
                        }

                        var copy = x.AsMatrixSpan().ToMatrix();
                        if (sine)
                        {
                            Fourier.SineRows(copy, type, normalization, copy);
                        }
                        else
                        {
                            Fourier.CosineRows(copy, type, normalization, copy);
                        }

                        for (int i = 0; i < rows; i++)
                        {
                            AssertBitwiseEqual(byRows.Row(i), copy.Row(i), $"{label} rows in place {i}");
                        }
                    }
                }
            }
        }
    }

    [Fact]
    public void TwoDimensionalTransformIsSeparable()
    {
        // dctn = 各行の DCT の後に各列の DCT。順序を入れ替えても同じ値（丸め誤差の範囲）
        var x = Matrix<double>.FromColumnMajor(6, 9, RandomReal(54, 5));
        var rowsFirst = Fourier.CosineColumns(Fourier.CosineRows(x, 2), 2);
        var columnsFirst = Fourier.CosineRows(Fourier.CosineColumns(x, 2), 2);
        var back = Fourier.InverseCosineRows(Fourier.InverseCosineColumns(rowsFirst, 2), 2);
        for (int i = 0; i < 6; i++)
        {
            for (int j = 0; j < 9; j++)
            {
                Assert.True(Math.Abs(rowsFirst[i, j] - columnsFirst[i, j]) < 1e-12, $"[{i},{j}] order");
                Assert.True(Math.Abs(x[i, j] - back[i, j]) < 1e-13, $"[{i},{j}] inverse");
            }
        }
    }

    // ---- 例外と空の入力 ----

    [Fact]
    public void RejectsInvalidArguments()
    {
        var x = new Vector<double>(4);
        Assert.Throws<ArgumentOutOfRangeException>(() => Fourier.Cosine(x, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => Fourier.Sine(x, 5));
        Assert.Throws<ArgumentOutOfRangeException>(() => Fourier.Cosine(x, 2, (FourierNormalization)7));
        Assert.Throws<ArgumentException>(() => Fourier.Cosine(new Vector<double>(1), 1));
        Assert.Throws<ArgumentException>(() => Fourier.InverseCosine(new Vector<double>(1), 1));
        Assert.Throws<ArgumentException>(() => Fourier.Cosine(x, 2, new Vector<double>(5)));

        // 部分的に重なる出力先
        var buffer = new Vector<double>(6);
        Assert.Throws<ArgumentException>(() => Fourier.Cosine(buffer[..4], 2, buffer[1..5]));
    }

    [Fact]
    public void EmptyInputGivesEmptyOutput()
    {
        for (int type = 1; type <= 4; type++)
        {
            Assert.Equal(0, Fourier.Cosine(new Vector<double>(0), type).Length);
            Assert.Equal(0, Fourier.InverseSine(new Vector<double>(0), type).Length);
            Assert.Equal(0, Fourier.CosineRows(new Matrix<double>(3, 0), type).Columns);
        }

        // 長さ 1 の DST-I の逆変換は 1/(2(n + 1)) 倍
        Assert.Equal(3, Fourier.InverseSine(new double[] { 6 }, 1)[0], 15);
    }
}
