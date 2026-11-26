using Tuna.LinearAlgebra;
using Tuna.SignalProcessing;
using Complex = System.Numerics.Complex;
using static Tuna.Tests.SignalProcessing.FourierReference;

namespace Tuna.Tests.SignalProcessing;

/// <summary>FFT の API: 一括変換と1本ずつの変換のビット一致、ビューの配置、その場更新、倍率、例外、補助関数</summary>
[Trait("Fourier", "Transform")]
public class FourierTransformTests
{
    /// <summary>Stockham の段数の偶奇、一般の奇数基数、Bluestein 法、実数の偶奇の各経路を通る長さ</summary>
    public static TheoryData<int> Lengths => new() { 0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 15, 16, 30, 31, 37, 49, 64, 97, 100, 128, 210, 256, 1009 };

    private static readonly int[] RowCounts = [1, 2, 3, 4, 5, 7, 8, 9, 17];

    private static Matrix<Complex> RandomMatrix(int rows, int columns, int seed) =>
        Matrix<Complex>.FromColumnMajor(rows, columns, RandomComplex(rows * columns, seed));

    private static Matrix<double> RandomRealMatrix(int rows, int columns, int seed) =>
        Matrix<double>.FromColumnMajor(rows, columns, RandomReal(rows * columns, seed));

    private static void AssertBitwiseEqual(ReadOnlyVectorSpan<Complex> expected, ReadOnlyVectorSpan<Complex> actual, string label = "")
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            if (BitConverter.DoubleToInt64Bits(expected[i].Real) != BitConverter.DoubleToInt64Bits(actual[i].Real)
                || BitConverter.DoubleToInt64Bits(expected[i].Imaginary) != BitConverter.DoubleToInt64Bits(actual[i].Imaginary))
            {
                Assert.Fail($"{label}[{i}] expected {expected[i]}, actual {actual[i]}");
            }
        }
    }

    private static void AssertBitwiseEqual(ReadOnlyVectorSpan<double> expected, ReadOnlyVectorSpan<double> actual, string label = "")
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

    private static void AssertBitwiseEqual(ReadOnlyMatrixSpan<Complex> expected, ReadOnlyMatrixSpan<Complex> actual)
    {
        Assert.Equal(expected.Rows, actual.Rows);
        Assert.Equal(expected.Columns, actual.Columns);
        for (int j = 0; j < expected.Columns; j++)
        {
            AssertBitwiseEqual(expected.Column(j), actual.Column(j), $"column {j}");
        }
    }

    private static void AssertBitwiseEqual(ReadOnlyMatrixSpan<double> expected, ReadOnlyMatrixSpan<double> actual)
    {
        Assert.Equal(expected.Rows, actual.Rows);
        Assert.Equal(expected.Columns, actual.Columns);
        for (int j = 0; j < expected.Columns; j++)
        {
            AssertBitwiseEqual(expected.Column(j), actual.Column(j), $"column {j}");
        }
    }

    // ---- 一括変換と1本ずつの変換のビット一致 ----

    [Theory]
    [MemberData(nameof(Lengths))]
    public void RowTransformsMatchVectorTransformsBitwise(int n)
    {
        foreach (int rows in RowCounts)
        {
            var x = RandomMatrix(rows, n, 100 * n + rows);
            var forward = Fourier.ForwardRows(x);
            var inverse = Fourier.InverseRows(x);
            for (int i = 0; i < rows; i++)
            {
                // 行は間隔 rows のビュー（集めてから変換する経路）、配列に写した行は連続の経路
                var row = x.Row(i).ToVector();
                AssertBitwiseEqual(Fourier.Forward(row), forward.Row(i), $"rows {rows}, row {i}");
                AssertBitwiseEqual(Fourier.Forward(x.Row(i)), forward.Row(i), $"rows {rows}, strided row {i}");
                AssertBitwiseEqual(Fourier.Inverse(row), inverse.Row(i), $"rows {rows}, inverse row {i}");
            }
        }
    }

    [Theory]
    [MemberData(nameof(Lengths))]
    public void ColumnTransformsMatchVectorTransformsBitwise(int n)
    {
        foreach (int columns in RowCounts)
        {
            // 列優先の行列の各列は連続（1列ずつの経路）、転置ビューの各列は間隔のある行（一括変換の経路）
            var x = RandomMatrix(n, columns, 7 * n + columns);
            var forward = Fourier.ForwardColumns(x);
            var transposed = Fourier.ForwardColumns(x.T.T);
            var viaRows = Fourier.ForwardRows(x.T);
            for (int j = 0; j < columns; j++)
            {
                AssertBitwiseEqual(Fourier.Forward(x.Column(j)), forward.Column(j), $"column {j}");
                AssertBitwiseEqual(forward.Column(j), viaRows.Row(j), $"rows of transpose {j}");
            }

            AssertBitwiseEqual(forward, transposed);

            var y = RandomMatrix(columns, n, 11 * n + columns);
            var batched = Fourier.InverseColumns(y.T);
            for (int j = 0; j < columns; j++)
            {
                AssertBitwiseEqual(Fourier.Inverse(y.Row(j)), batched.Column(j), $"inverse column {j}");
            }
        }
    }

    [Theory]
    [MemberData(nameof(Lengths))]
    public void RealRowAndColumnTransformsMatchVectorTransformsBitwise(int n)
    {
        int spectrum = n == 0 ? 0 : n / 2 + 1;
        foreach (int rows in RowCounts)
        {
            var x = RandomRealMatrix(rows, n, 13 * n + rows);
            var forward = Fourier.ForwardRealRows(x);
            Assert.Equal(spectrum, forward.Columns);
            var spectra = RandomMatrix(rows, spectrum, 17 * n + rows);
            var inverse = Fourier.InverseRealRows(spectra, n);
            var forwardColumns = Fourier.ForwardRealColumns(x.T);
            var inverseColumns = Fourier.InverseRealColumns(spectra.T, n);
            for (int i = 0; i < rows; i++)
            {
                var expected = Fourier.ForwardReal(x.Row(i).ToVector());
                AssertBitwiseEqual(expected, forward.Row(i), $"rows {rows}, row {i}");
                AssertBitwiseEqual(expected, Fourier.ForwardReal(x.Row(i)), $"rows {rows}, strided row {i}");
                AssertBitwiseEqual(expected, forwardColumns.Column(i), $"rows {rows}, column {i}");

                var signal = Fourier.InverseReal(spectra.Row(i).ToVector(), n);
                AssertBitwiseEqual(signal, inverse.Row(i), $"rows {rows}, inverse row {i}");
                AssertBitwiseEqual(signal, Fourier.InverseReal(spectra.Row(i), n), $"rows {rows}, strided inverse row {i}");
                AssertBitwiseEqual(signal, inverseColumns.Column(i), $"rows {rows}, inverse column {i}");
            }
        }
    }

    [Theory]
    [InlineData(5, 16)]
    [InlineData(3, 37)]
    [InlineData(6, 30)]
    public void StridedViewsMatchDenseCopiesBitwise(int rows, int n)
    {
        var big = RandomMatrix(rows + 4, n + 5, rows * n);
        var view = big[2..(2 + rows), 3..(3 + n)];
        var dense = view.ToMatrix();
        AssertBitwiseEqual(Fourier.ForwardRows(dense), Fourier.ForwardRows(view));
        AssertBitwiseEqual(Fourier.ForwardColumns(dense.T.ToMatrix()), Fourier.ForwardColumns(view.T));

        // 出力先もストライド付きのビュー
        var target = new Matrix<Complex>(rows + 3, n + 2);
        Fourier.ForwardRows(view, target[1..(1 + rows), 2..(2 + n)]);
        AssertBitwiseEqual(Fourier.ForwardRows(dense), target[1..(1 + rows), 2..(2 + n)]);

        var realBig = RandomRealMatrix(rows + 4, n + 5, rows + n);
        var realView = realBig[1..(1 + rows), 2..(2 + n)];
        var spectrum = new Matrix<Complex>(rows + 2, n / 2 + 3);
        Fourier.ForwardRealRows(realView, spectrum[1..(1 + rows), 1..(2 + n / 2)]);
        AssertBitwiseEqual(Fourier.ForwardRealRows(realView.ToMatrix()), spectrum[1..(1 + rows), 1..(2 + n / 2)]);

        var back = new Matrix<double>(rows + 2, n + 1);
        Fourier.InverseRealRows(spectrum[1..(1 + rows), 1..(2 + n / 2)], back[1..(1 + rows), 1..(1 + n)]);
        AssertBitwiseEqual(Fourier.InverseRealRows(spectrum[1..(1 + rows), 1..(2 + n / 2)].ToMatrix(), n), back[1..(1 + rows), 1..(1 + n)]);
    }

    /// <summary>
    /// 何行かずつ集めて変換する経路（全体が 512 KiB を超える行列）。40 × 2048 は 16 行ずつ（実数は 28 行ずつ）列優先に集め、
    /// 9 × 4099 は Bluestein 法を4行ずつ、6 × 40000 は1行ずつ行優先に集める（列優先で連続な行列はそのまま一括変換）
    /// </summary>
    [Theory]
    [InlineData(40, 2048)]
    [InlineData(9, 4099)]
    [InlineData(6, 40000)]
    public void LargeMatricesMatchVectorTransformsBitwise(int rows, int n)
    {
        var big = RandomMatrix(rows + 3, n + 2, rows + n);
        var view = big[1..(1 + rows), 2..];
        var dense = view.ToMatrix();
        var forward = Fourier.ForwardRows(dense);
        var inverse = Fourier.InverseRows(view);
        var inPlace = view.ToMatrix();
        Fourier.InPlace.ForwardRows(inPlace.T.T);
        var stridedInPlace = big.Clone();
        Fourier.InPlace.ForwardRows(stridedInPlace[1..(1 + rows), 2..]);
        AssertBitwiseEqual(forward, Fourier.ForwardRows(view));
        AssertBitwiseEqual(forward, inPlace);
        AssertBitwiseEqual(forward, stridedInPlace[1..(1 + rows), 2..]);
        for (int i = 0; i < rows; i++)
        {
            var row = dense.Row(i).ToVector();
            AssertBitwiseEqual(Fourier.Forward(row), forward.Row(i), $"row {i}");
            AssertBitwiseEqual(Fourier.Inverse(row), inverse.Row(i), $"inverse row {i}");
        }

        var realBig = RandomRealMatrix(rows + 2, n + 3, rows * n);
        var realView = realBig[2.., 1..(1 + n)];
        var realForward = Fourier.ForwardRealRows(realView.ToMatrix());
        AssertBitwiseEqual(realForward, Fourier.ForwardRealRows(realView));
        var realInverse = Fourier.InverseRealRows(realForward, n);
        var spectra = view[.., ..(n / 2 + 1)];
        var signals = Fourier.InverseRealRows(spectra, n);
        AssertBitwiseEqual(signals, Fourier.InverseRealRows(spectra.ToMatrix(), n));
        for (int i = 0; i < rows; i++)
        {
            AssertBitwiseEqual(Fourier.ForwardReal(realView.Row(i).ToVector()), realForward.Row(i), $"real row {i}");
            AssertBitwiseEqual(Fourier.InverseReal(realForward.Row(i).ToVector(), n), realInverse.Row(i), $"real round trip row {i}");
            AssertBitwiseEqual(Fourier.InverseReal(spectra.Row(i).ToVector(), n), signals.Row(i), $"real inverse row {i}");
        }
    }

    // ---- その場更新・計画の再利用・入力を書き換えないこと ----

    [Theory]
    [MemberData(nameof(Lengths))]
    public void InPlaceMatchesOutOfPlaceBitwise(int n)
    {
        var x = RandomComplex(n, n + 3);
        var vector = new Vector<Complex>(x);
        Fourier.InPlace.Forward(vector);
        AssertBitwiseEqual(Fourier.Forward(new ReadOnlyVectorSpan<Complex>(x)), vector);
        Fourier.InPlace.Inverse(vector, FourierNormalization.Orthonormal);
        AssertBitwiseEqual(Fourier.Inverse(Fourier.Forward(new ReadOnlyVectorSpan<Complex>(x)), FourierNormalization.Orthonormal), vector);

        var m = RandomMatrix(3, n, n + 5);
        var copy = m.Clone();
        Fourier.InPlace.ForwardRows(m);
        AssertBitwiseEqual(Fourier.ForwardRows(copy), m);
        Fourier.InPlace.InverseRows(m);
        AssertBitwiseEqual(Fourier.InverseRows(Fourier.ForwardRows(copy)), m);

        var c = RandomMatrix(n, 3, n + 6);
        var cCopy = c.Clone();
        Fourier.InPlace.ForwardColumns(c);
        AssertBitwiseEqual(Fourier.ForwardColumns(cCopy), c);
        Fourier.InPlace.InverseColumns(c.T.T);
        AssertBitwiseEqual(Fourier.InverseColumns(Fourier.ForwardColumns(cCopy)), c);

        // 転置ビューのその場更新（集めて書き戻す経路）
        var t = RandomMatrix(n, 4, n + 8);
        var tCopy = t.Clone();
        Fourier.InPlace.ForwardRows(t.T);
        AssertBitwiseEqual(Fourier.ForwardColumns(tCopy), t);
    }

    [Theory]
    [MemberData(nameof(Lengths))]
    public void PlansGiveTheSameResultsAsTheConvenienceFunctions(int n)
    {
        var x = RandomComplex(n, n + 9);
        var plan = FourierPlan.Create(n);
        Assert.Same(plan, FourierPlan.Create(n));
        Assert.Equal(n, plan.Length);
        var first = new Vector<Complex>(n);
        var second = new Vector<Complex>(n);
        plan.Forward(new ReadOnlyVectorSpan<Complex>(x), first);
        plan.Forward(new ReadOnlyVectorSpan<Complex>(x), second);
        AssertBitwiseEqual(first, second);
        AssertBitwiseEqual(Fourier.Forward(new ReadOnlyVectorSpan<Complex>(x)), first);

        var real = RandomReal(n, n + 10);
        var realPlan = RealFourierPlan.Create(n);
        Assert.Equal(n == 0 ? 0 : n / 2 + 1, realPlan.SpectrumLength);
        var spectrum = new Vector<Complex>(realPlan.SpectrumLength);
        realPlan.Forward(new ReadOnlyVectorSpan<double>(real), spectrum);
        AssertBitwiseEqual(Fourier.ForwardReal(new ReadOnlyVectorSpan<double>(real)), spectrum);
        var signal = new Vector<double>(n);
        realPlan.Inverse(spectrum, signal);
        AssertBitwiseEqual(Fourier.InverseReal(spectrum, n), signal);
    }

    [Theory]
    [MemberData(nameof(Lengths))]
    public void InputsAreNotModified(int n)
    {
        var x = RandomComplex(n, n + 11);
        var original = (Complex[])x.Clone();
        Fourier.Forward(new ReadOnlyVectorSpan<Complex>(x));
        Fourier.Inverse(new ReadOnlyVectorSpan<Complex>(x));
        AssertBitwiseEqual(new ReadOnlyVectorSpan<Complex>(original), new ReadOnlyVectorSpan<Complex>(x));

        var real = RandomReal(n, n + 12);
        var realOriginal = (double[])real.Clone();
        Fourier.ForwardReal(new ReadOnlyVectorSpan<double>(real));
        AssertBitwiseEqual(new ReadOnlyVectorSpan<double>(realOriginal), new ReadOnlyVectorSpan<double>(real));

        var spectrum = RandomComplex(n == 0 ? 0 : n / 2 + 1, n + 13);
        var spectrumOriginal = (Complex[])spectrum.Clone();
        Fourier.InverseReal(new ReadOnlyVectorSpan<Complex>(spectrum), n);
        AssertBitwiseEqual(new ReadOnlyVectorSpan<Complex>(spectrumOriginal), new ReadOnlyVectorSpan<Complex>(spectrum));

        var m = RandomRealMatrix(3, n, n + 14);
        var mOriginal = m.Clone();
        Fourier.ForwardRealRows(m);
        AssertBitwiseEqual(mOriginal, m);
    }

    // ---- 倍率 ----

    [Theory]
    [InlineData(1)]
    [InlineData(8)]
    [InlineData(45)]
    [InlineData(37)]
    public void NormalizationsScaleTheUnnormalizedTransform(int n)
    {
        var x = new ReadOnlyVectorSpan<Complex>(RandomComplex(n, n + 15));
        var plain = Fourier.Forward(x);
        var unscaledInverse = Fourier.Inverse(x, FourierNormalization.Forward);
        double orthonormal = 1 / Math.Sqrt(n);
        double backward = 1.0 / n;
        var ortho = Fourier.Forward(x, FourierNormalization.Orthonormal);
        var forward = Fourier.Forward(x, FourierNormalization.Forward);
        var inverse = Fourier.Inverse(x);
        var orthoInverse = Fourier.Inverse(x, FourierNormalization.Orthonormal);
        for (int k = 0; k < n; k++)
        {
            Assert.Equal(new Complex(plain[k].Real * orthonormal, plain[k].Imaginary * orthonormal), ortho[k]);
            Assert.Equal(new Complex(plain[k].Real * backward, plain[k].Imaginary * backward), forward[k]);
            Assert.Equal(new Complex(unscaledInverse[k].Real * backward, unscaledInverse[k].Imaginary * backward), inverse[k]);
            Assert.Equal(new Complex(unscaledInverse[k].Real * orthonormal, unscaledInverse[k].Imaginary * orthonormal), orthoInverse[k]);
        }

        // 実数の逆変換の倍率も同じ規則
        var real = new ReadOnlyVectorSpan<double>(RandomReal(n, n + 16));
        var spectrum = Fourier.ForwardReal(real, FourierNormalization.Orthonormal);
        var back = Fourier.InverseReal(spectrum, n, FourierNormalization.Orthonormal);
        for (int i = 0; i < n; i++)
        {
            Assert.True(Math.Abs(back[i] - real[i]) <= 1e-14, $"[{i}] {back[i]} vs {real[i]}");
        }
    }

    // ---- 既知の変換 ----

    [Theory]
    [InlineData(1)]
    [InlineData(6)]
    [InlineData(16)]
    [InlineData(41)]
    public void KnownTransforms(int n)
    {
        // インパルス → 全部 1（Stockham の長さでは厳密、Bluestein 法の長さでは丸め誤差がある）
        var impulse = new Vector<Complex>(n);
        impulse[0] = 1;
        foreach (var value in Fourier.Forward(impulse).AsSpan())
        {
            Assert.True(Complex.Abs(value - 1) <= 1e-14, $"{value}");
        }

        // 定数 → 周波数 0 に n
        var constant = new Vector<double>(n);
        constant.AsSpan().Fill(1);
        var spectrum = Fourier.ForwardReal(constant);
        Assert.Equal(n, spectrum[0].Real, 12);
        for (int k = 1; k < spectrum.Length; k++)
        {
            Assert.True(Complex.Abs(spectrum[k]) <= 1e-13 * n, $"[{k}] {spectrum[k]}");
        }

        // exp(2πi·m·j/n) → 周波数 m に n
        int m = n / 3;
        var wave = new Vector<Complex>(n);
        for (int j = 0; j < n; j++)
        {
            wave[j] = Complex.Exp(new Complex(0, 2 * Math.PI * m * j / n));
        }

        var waveSpectrum = Fourier.Forward(wave);
        for (int k = 0; k < n; k++)
        {
            double expected = k == m ? n : 0;
            Assert.True(Complex.Abs(waveSpectrum[k] - expected) <= 1e-12 * n, $"[{k}] {waveSpectrum[k]}");
        }
    }

    [Fact]
    public void RealTransformIgnoresImaginaryPartsOfZeroAndNyquistCoefficients()
    {
        var spectrum = RandomComplex(9, 3);
        var cleaned = (Complex[])spectrum.Clone();
        cleaned[0] = new Complex(cleaned[0].Real, 0);
        cleaned[8] = new Complex(cleaned[8].Real, 0);
        AssertBitwiseEqual(
            Fourier.InverseReal(new ReadOnlyVectorSpan<Complex>(cleaned), 16),
            Fourier.InverseReal(new ReadOnlyVectorSpan<Complex>(spectrum), 16));
        AssertBitwiseEqual(
            Fourier.InverseReal(new ReadOnlyVectorSpan<Complex>(cleaned), 17),
            Fourier.InverseReal(new ReadOnlyVectorSpan<Complex>(cleaned.Select((c, k) => k == 0 ? new Complex(c.Real, 5) : c).ToArray()), 17));
    }

    [Fact]
    public void ConcurrentCallsGiveTheSameResults()
    {
        int[] lengths = [16, 37, 97, 100, 1009, 1024, 4099, 30];
        var inputs = lengths.Select(n => RandomComplex(n, n)).ToArray();
        var expected = inputs.Select(x => Fourier.Forward(new ReadOnlyVectorSpan<Complex>(x))).ToArray();
        Parallel.For(0, 200, i =>
        {
            int index = i % lengths.Length;
            var actual = Fourier.Forward(new ReadOnlyVectorSpan<Complex>(inputs[index]));
            AssertBitwiseEqual(expected[index], actual);
        });
    }

    // ---- 例外 ----

    [Fact]
    public void InvalidArgumentsThrow()
    {
        var x = new Vector<Complex>(RandomComplex(8, 1));
        Assert.Throws<ArgumentException>(() => Fourier.Forward(x, new Vector<Complex>(7)));
        Assert.Throws<ArgumentException>(() => FourierPlan.Create(4).Forward(x, new Vector<Complex>(8)));

        // 部分的な重なり（同じ配列の1つずれた範囲）
        var buffer = new Vector<Complex>(RandomComplex(9, 2));
        Assert.Throws<ArgumentException>(() => Fourier.Forward(buffer[0..8], buffer[1..9]));

        // 実数 ↔ 複素は重なりを許さない
        var complexStorage = new Complex[8];
        Assert.Throws<ArgumentException>(() =>
        {
            var asReal = System.Runtime.InteropServices.MemoryMarshal.Cast<Complex, double>(complexStorage.AsSpan());
            Fourier.ForwardReal(new ReadOnlyVectorSpan<double>(asReal[..8]), new VectorSpan<Complex>(complexStorage.AsSpan(0, 5)));
        });

        // 片側スペクトルの長さが合わない
        Assert.Throws<ArgumentException>(() => Fourier.InverseReal(new Vector<Complex>(4), 8));
        Assert.Throws<ArgumentException>(() => Fourier.ForwardReal(new Vector<double>(8), new Vector<Complex>(4)));
        Assert.Throws<ArgumentOutOfRangeException>(() => Fourier.InverseReal(new Vector<Complex>(1), -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => FourierPlan.Create(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => Fourier.Forward(x, (FourierNormalization)7));
        Assert.Throws<ArgumentOutOfRangeException>(() => Fourier.NextFastLength(-1));
    }

    [Fact]
    public void EmptyAndSingleElementInputs()
    {
        Assert.Equal(0, Fourier.Forward(new Vector<Complex>(0)).Length);
        Assert.Equal(0, Fourier.ForwardReal(new Vector<double>(0)).Length);
        Assert.Equal(0, Fourier.InverseReal(new Vector<Complex>(0), 0).Length);
        Assert.Equal(new Complex(2, -3), Fourier.Forward(new Vector<Complex>([new Complex(2, -3)]))[0]);
        Assert.Equal(new Complex(2, -3), Fourier.Inverse(new Vector<Complex>([new Complex(2, -3)]))[0]);
        Assert.Equal(new Complex(4, 0), Fourier.ForwardReal(new Vector<double>([4.0]))[0]);
        Assert.Equal(4.0, Fourier.InverseReal(new Vector<Complex>([new Complex(4, 1)]), 1)[0]);
        Assert.Equal(new Complex[] { 3, -1 }, Fourier.ForwardReal(new Vector<double>([1.0, 2.0])).ToArray());
        Assert.Equal(0, Fourier.ForwardRows(new Matrix<Complex>(0, 5)).Rows);
        Assert.Equal(3, Fourier.ForwardRealRows(new Matrix<double>(3, 0)).Rows);
    }

    // ---- 補助関数（期待値は numpy 2.5.3 の出力） ----

    [Fact]
    public void FrequenciesMatchNumpy()
    {
        Assert.Equal([0.0, 2.0, 4.0, -4.0, -2.0], Fourier.Frequencies(5, 0.1).ToArray());
        Assert.Equal([0.0, 0.25, -0.5, -0.25], Fourier.Frequencies(4).ToArray());
        Assert.Equal([0.0, 0.2, 0.4], Fourier.RealFrequencies(5).ToArray());
        Assert.Equal([0.0, 0.25, 0.5], Fourier.RealFrequencies(4).ToArray());
        Assert.Equal(
            [0.0, 0.47619047619047616, 0.9523809523809523, 1.4285714285714284, -1.4285714285714284, -0.9523809523809523, -0.47619047619047616],
            Fourier.Frequencies(7, 0.3).ToArray());
        Assert.Empty(Fourier.Frequencies(0).ToArray());
    }

    [Fact]
    public void ShiftsMatchNumpy()
    {
        var odd = new Vector<int>([0, 1, 2, 3, 4]);
        Assert.Equal([3, 4, 0, 1, 2], Fourier.Shift(odd).ToArray());
        Assert.Equal([2, 3, 4, 0, 1], Fourier.InverseShift(odd).ToArray());
        Assert.Equal([0, 1, 2, 3, 4], Fourier.InverseShift(Fourier.Shift(odd)).ToArray());
        var even = new Vector<double>([0, 1, 2, 3]);
        Assert.Equal([2.0, 3, 0, 1], Fourier.Shift(even).ToArray());
        Assert.Empty(Fourier.Shift(new Vector<double>(0)).ToArray());

        var m = Matrix<double>.FromRowMajor(2, 3, [0, 1, 2, 10, 11, 12]);
        Assert.Equal(new double[,] { { 2, 0, 1 }, { 12, 10, 11 } }, Fourier.ShiftRows(m).ToArray());
        Assert.Equal(new double[,] { { 1, 2, 0 }, { 11, 12, 10 } }, Fourier.InverseShiftRows(m).ToArray());
        Assert.Equal(new double[,] { { 10, 11, 12 }, { 0, 1, 2 } }, Fourier.ShiftColumns(m).ToArray());
        Assert.Equal(new double[,] { { 10, 11, 12 }, { 0, 1, 2 } }, Fourier.InverseShiftColumns(m).ToArray());

        var buffer = new Vector<double>(5);
        Assert.Throws<ArgumentException>(() => Fourier.Shift(buffer[0..4], buffer[1..5]));
    }

    [Theory]
    [InlineData(0, 1)]
    [InlineData(1, 1)]
    [InlineData(2, 2)]
    [InlineData(11, 12)]
    [InlineData(13, 14)]
    [InlineData(17, 18)]
    [InlineData(37, 40)]
    [InlineData(97, 98)]
    [InlineData(1021, 1024)]
    [InlineData(1025, 1029)]
    [InlineData(2017, 2025)]
    [InlineData(131073, 131220)]
    [InlineData(1000001, 1000188)]
    public void NextFastLengthMatchesScipy(int length, int expected) => Assert.Equal(expected, Fourier.NextFastLength(length));
}
