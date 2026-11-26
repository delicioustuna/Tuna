using System.Numerics;
using System.Runtime.InteropServices;
using Tuna.LinearAlgebra;
using static Tuna.XMath;
using DoubleVector = Tuna.LinearAlgebra.Vector<double>;

namespace Tuna.Tests.LinearAlgebra;

/// <summary>
/// 分散・標準偏差が、値の大きさによらず正確なこと（偏差の平方和が溢れるときと、非正規化数の範囲に近いときは2の冪で縮尺して求め直す）。
/// 参照は入力の値（丸めたあとの値）から分散を有理数として正確に求めたもの（BigInteger）で、ライブラリの実装には頼らない。
/// </summary>
[Trait("LinearAlgebra", "XMath")]
public class DispersionScalingTests
{
    // 縮尺の指数。±490 あたりから平方和が溢れるか非正規化数の範囲に近づき、-1022 より下では入力そのものが非正規化数になる
    private static readonly int[] Scales = [-1068, -1060, -1040, -1000, -700, -600, -520, -500, -490, -400, -100, 0, 100, 400, 490, 500, 505, 510, 520, 600, 1000, 1015];

    private static readonly int[] Lengths = [1, 2, 3, 5, 17, 64, 65, 200, 1000];

    /// <summary>符号と大きさ（2⁻⁴〜2⁴）を混ぜた値。shift を足すと平均が標準偏差より大きくなる</summary>
    private static double[] Values(int length, int seed, double shift = 0)
    {
        var random = new Random(seed);
        return Enumerable.Range(0, length).Select(_ => shift + (2 * random.NextDouble() - 1) * Math.ScaleB(1, random.Next(-4, 5))).ToArray();
    }

    // x[i] = a[i]·2^E（a[i] は整数、E は 0 でない要素の最小の指数）とおくと、n(n - c)·分散 = (nΣa² - (Σa)²)·2^(2E)。
    // 分散を「分子 / 分母·2^指数」の形で正確に返す
    private static (BigInteger Numerator, BigInteger Denominator, int Exponent) ExactVariance(double[] x, int correction)
    {
        var parts = new (long Mantissa, int Exponent)[x.Length];
        int smallest = int.MaxValue;
        for (int i = 0; i < x.Length; i++)
        {
            long bits = BitConverter.DoubleToInt64Bits(x[i]);
            int biased = (int)((bits >> 52) & 0x7FF);
            long mantissa = bits & 0xF_FFFF_FFFF_FFFF;
            int exponent = biased == 0 ? -1074 : biased - 1075;
            if (biased != 0)
            {
                mantissa |= 1L << 52;
            }

            if (bits < 0)
            {
                mantissa = -mantissa;
            }

            parts[i] = (mantissa, exponent);
            if (mantissa != 0)
            {
                smallest = Math.Min(smallest, exponent);
            }
        }

        if (smallest == int.MaxValue)
        {
            return (BigInteger.Zero, BigInteger.One, 0);
        }

        BigInteger sum = 0, squares = 0;
        foreach (var (mantissa, exponent) in parts)
        {
            BigInteger a = new BigInteger(mantissa) << (exponent - smallest);
            sum += a;
            squares += a * a;
        }

        int n = x.Length;
        return (n * squares - sum * sum, new BigInteger(n) * (n - correction), 2 * smallest);
    }

    private static BigInteger IntegerSqrt(BigInteger value)
    {
        if (value.IsZero)
        {
            return value;
        }

        BigInteger x = BigInteger.One << (int)((value.GetBitLength() + 1) / 2);
        while (true)
        {
            BigInteger y = (x + value / x) >> 1;
            if (y >= x)
            {
                return x;
            }

            x = y;
        }
    }

    // 正確な分散（または標準偏差）を double に丸める（およそ 100 ビットまで求めてから丸めるので、非正規化数の範囲を除いて正しく丸めた値）
    private static double Exact(double[] x, int correction, bool standardDeviation)
    {
        var (numerator, denominator, exponent) = ExactVariance(x, correction);
        if (numerator.IsZero)
        {
            return 0;
        }

        int shift = (int)Math.Max(0, 200 - (numerator.GetBitLength() - denominator.GetBitLength()));
        shift += shift & 1;
        BigInteger ratio = (numerator << shift) / denominator;
        return standardDeviation
            ? Math.ScaleB((double)IntegerSqrt(ratio), (exponent - shift) / 2)
            : Math.ScaleB((double)ratio, exponent - shift);
    }

    // 相対誤差 relative 以内（非正規化数の範囲では最小の非正規化数の 2 倍まで）。溢れるはずなら +∞
    private static void AssertAccurate(string label, double expected, double actual, double relative)
    {
        if (double.IsPositiveInfinity(expected))
        {
            Assert.True(double.IsPositiveInfinity(actual), $"{label}: expected +∞, actual {actual:R}");
            return;
        }

        Assert.True(Math.Abs(actual - expected) <= relative * expected + 2 * double.Epsilon, $"{label}: expected {expected:R}, actual {actual:R}");
    }

    private static double[] Scaled(double[] x, int k) => x.Select(v => Math.ScaleB(v, k)).ToArray();

    [Fact]
    public void VarianceAndStandardDeviationAreAccurateAtEveryScale()
    {
        foreach (int n in Lengths)
        {
            foreach (double shift in new[] { 0.0, 20.0 })
            {
                double[] x = Values(n, n, shift);
                foreach (int k in Scales)
                {
                    double[] y = Scaled(x, k);
                    var strided = new Matrix<double>(3, n);
                    for (int i = 0; i < n; i++)
                    {
                        strided[1, i] = y[i];
                    }

                    foreach (int correction in new[] { 0, 1 })
                    {
                        if (n <= correction)
                        {
                            continue;
                        }

                        string label = $"n={n} shift={shift} k={k} correction={correction}";
                        double variance = Exact(y, correction, standardDeviation: false);
                        double deviation = Exact(y, correction, standardDeviation: true);
                        AssertAccurate($"Variance {label}", variance, Variance(new DoubleVector(y), correction), 1e-14);
                        AssertAccurate($"StandardDeviation {label}", deviation, StandardDeviation(new DoubleVector(y), correction), 1e-14);
                        AssertAccurate($"Variance strided {label}", variance, Variance(strided.Row(1), correction), 1e-14);
                        AssertAccurate($"StandardDeviation strided {label}", deviation, StandardDeviation(strided.Row(1), correction), 1e-14);
                    }
                }
            }
        }
    }

    [Fact]
    public void StandardDeviationStaysFiniteAndPreciseWhereTheVarianceIsNotRepresentable()
    {
        // 分散 4e400 は溢れるが、標準偏差 2e200 は表現できる（以前は √∞ = ∞ だった）
        double[] large = [1e200, -1e200, 3e200];
        Assert.True(double.IsPositiveInfinity(Variance(new DoubleVector(large), 1)));
        AssertAccurate("large", Exact(large, 1, true), StandardDeviation(new DoubleVector(large), 1), 1e-15);

        // 偏差そのもの（-4M/3）が溢れても、母標準偏差 √(8/9)·M は表現できる
        double[] extreme = [double.MaxValue, -double.MaxValue, double.MaxValue];
        AssertAccurate("extreme", Math.Sqrt(8.0 / 9) * double.MaxValue, StandardDeviation(new DoubleVector(extreme), 0), 1e-15);

        // 偏差の平方 1e-400 は 0 に丸まるが、標準偏差 1e-200 は表現できる（以前は 0 だった）
        double[] small = [1e-200, 2e-200, 3e-200];
        Assert.Equal(0.0, Variance(new DoubleVector(small), 1));
        AssertAccurate("small", Exact(small, 1, true), StandardDeviation(new DoubleVector(small), 1), 1e-15);

        // 非正規化数だけの入力: 分散 ε² は 0 に丸まり、標準偏差はちょうど最小の非正規化数
        double[] subnormal = [double.Epsilon, 2 * double.Epsilon, 3 * double.Epsilon];
        Assert.Equal(0.0, Variance(new DoubleVector(subnormal), 1));
        Assert.Equal(double.Epsilon, StandardDeviation(new DoubleVector(subnormal), 1));

        // 分散が非正規化数になる範囲でも、2^(2k) 倍の縮尺どおり（正規化数の範囲で求めた値を1回丸めたもの）
        double[] values = Values(100, 7);
        foreach (int k in new[] { -530, -540, -560 })
        {
            AssertAccurate($"subnormal variance k={k}", Exact(Scaled(values, k), 1, false), Variance(new DoubleVector(Scaled(values, k)), 1), 1e-14);
        }

        // float・Half・そのほかの型（NFloat は汎用の経路）も同じ
        float[] smallSingle = [1e-30f, 2e-30f, 3e-30f];
        double expectedSingle = Exact(smallSingle.Select(v => (double)v).ToArray(), 1, true);
        Assert.True(Math.Abs(StandardDeviation(smallSingle.ToVector(), 1) - expectedSingle) <= 1e-6 * expectedSingle);
        float[] largeSingle = [1e30f, -1e30f, 3e30f];
        Assert.True(float.IsPositiveInfinity(Variance(largeSingle.ToVector(), 1)));
        Assert.True(Math.Abs(StandardDeviation(largeSingle.ToVector(), 1) - 2e30) <= 1e-6 * 2e30);

        // Half の偏差 1e-4 の平方は最小の非正規化数（6e-8）より小さい
        Half[] smallHalf = [(Half)0.001, (Half)0.0011, (Half)0.0012];
        double expectedHalf = Exact(smallHalf.Select(v => (double)v).ToArray(), 1, true);
        Assert.True(Math.Abs((double)StandardDeviation(smallHalf.ToVector(), 1) - expectedHalf) <= 2e-3 * expectedHalf);

        NFloat[] largeNative = [(NFloat)1e200, (NFloat)(-1e200), (NFloat)3e200];
        if (NFloat.Size == sizeof(double))
        {
            Assert.True(Math.Abs((double)StandardDeviation(largeNative.ToVector(), 1) - 2e200) <= 1e-15 * 2e200);
        }
    }

    [Fact]
    public void HalfIsComputedInSingleAndHandlesSubnormalsAndLongInputs()
    {
        static void AssertHalf(string label, Half expected, Half actual) =>
            Assert.True(BitConverter.HalfToInt16Bits(expected) == BitConverter.HalfToInt16Bits(actual), $"{label}: expected {(double)expected:R}, actual {(double)actual:R}");

        // 平均はちょうど 0。[1, 2) にそろえた偏差の平方和を Half のまま足すと 4n > 65504 で溢れていた
        Half magnitude = (Half)0.00023;
        Half[] alternating = Enumerable.Range(0, 20000).Select(i => i % 2 == 0 ? magnitude : -magnitude).ToArray();
        // .NET 10 の Half.ILogB は非正規化数で誤った指数を返す（Half.ILogB(Half.Epsilon) が -40）。縮尺がそれに頼ると NaN になっていた
        Half e = Half.Epsilon;
        Half[] subnormal = [e, (Half)((Half)2 * e), (Half)((Half)3 * e)];
        foreach (var (name, values, correction, variance, deviation) in new[]
        {
            ("alternating", alternating, 0, (Half)((double)magnitude * (double)magnitude), magnitude),
            ("subnormal", subnormal, 1, Half.Zero, Half.Epsilon),
        })
        {
            var row = Matrix<Half>.FromColumnMajor(1, values.Length, values);
            var column = Matrix<Half>.FromColumnMajor(values.Length, 1, values);
            AssertHalf($"Variance {name}", variance, Variance(values.ToVector(), correction));
            AssertHalf($"StandardDeviation {name}", deviation, StandardDeviation(values.ToVector(), correction));
            AssertHalf($"RowVariances {name}", variance, RowVariances(row, correction)[0]);
            AssertHalf($"RowStandardDeviations {name}", deviation, RowStandardDeviations(row, correction)[0]);
            AssertHalf($"ColumnVariances {name}", variance, ColumnVariances(column, correction)[0]);
            AssertHalf($"ColumnStandardDeviations {name}", deviation, ColumnStandardDeviations(column, correction)[0]);
        }

        // ノルムも非正規化数の指数に Half.ILogB を使わない
        Half[] pythagorean = [(Half)((Half)3 * e), (Half)((Half)4 * e)];
        AssertHalf("Norm ε", e, Norm(new[] { e }.ToVector()));
        AssertHalf("Norm 3ε, 4ε", (Half)((Half)5 * e), Norm(pythagorean.ToVector()));
        AssertHalf("RowNorms 3ε, 4ε", (Half)((Half)5 * e), RowNorms(Matrix<Half>.FromColumnMajor(1, 2, pythagorean))[0]);

        // float に広げて求め、最後に1回 Half に丸める（Half の丸めの幅の中で正確な値に一致する）
        var random = new Random(3);
        foreach (int n in new[] { 2, 3, 65, 1000, 70000 })
        {
            Half[] values = Enumerable.Range(0, n).Select(_ => (Half)((random.NextDouble() * 2 - 1) * Math.ScaleB(1, random.Next(-20, 15)))).ToArray();
            double[] widened = values.Select(v => (double)v).ToArray();
            foreach (bool standardDeviation in new[] { false, true })
            {
                double expected = Exact(widened, 1, standardDeviation);
                Half actual = standardDeviation ? StandardDeviation(values.ToVector(), 1) : Variance(values.ToVector(), 1);
                double tolerance = expected > (double)Half.MaxValue ? double.PositiveInfinity : Math.ScaleB(1, -11) * expected + (double)Half.Epsilon;
                Assert.True(Math.Abs((double)actual - expected) <= tolerance || Half.IsPositiveInfinity(actual) && expected > (double)Half.MaxValue,
                    $"Half n={n} standardDeviation={standardDeviation}: expected {expected:R}, actual {(double)actual:R}");
            }
        }
    }

    [Fact]
    public void RescaledResultsFollowThePowerOfTwoExactly()
    {
        // 縮尺して求め直す範囲では、入力を 2^k 倍すると標準偏差はちょうど 2^k 倍、分散はちょうど 2^(2k) 倍になる（正規化数で表現できる限り）
        int checkedVariances = 0;
        foreach (int n in new[] { 2, 3, 64, 65, 1000 })
        {
            double[] x = Values(n, n + 1, shift: 3);
            double reference = StandardDeviation(new DoubleVector(Scaled(x, -600)), 1);
            foreach (int k in new[] { -1000, -700, -600, -520, 520, 600, 1000 })
            {
                Assert.Equal(Math.ScaleB(reference, k + 600), StandardDeviation(new DoubleVector(Scaled(x, k)), 1));
            }

            // 分散が溢れず非正規化数にもならない範囲は、平方和が非正規化数に近い側の 2^-510〜2^-490 あたりだけ
            double referenceVariance = Variance(new DoubleVector(Scaled(x, -500)), 1);
            foreach (int k in new[] { -505, -500, -495, -490 })
            {
                double expected = Math.ScaleB(referenceVariance, 2 * (k + 500));
                if (!double.IsNormal(expected) || !double.IsNormal(referenceVariance))
                {
                    continue;
                }

                Assert.Equal(expected, Variance(new DoubleVector(Scaled(x, k)), 1));
                checkedVariances++;
            }
        }

        Assert.True(checkedVariances >= 10);
    }

    [Fact]
    public void ConstantAndNonFiniteInputsKeepTheirResults()
    {
        Assert.Equal(0.0, Variance(new DoubleVector([3, 3, 3]), 1));
        Assert.Equal(0.0, Variance(new DoubleVector([0, -0.0, 0, 0]), 0));
        Assert.Equal(0.0, StandardDeviation(new DoubleVector([1e-310, 1e-310, 1e-310, 1e-310]), 1));
        Assert.Equal(0.0, StandardDeviation(new DoubleVector([double.MaxValue, double.MaxValue]), 1));
        Assert.True(double.IsNaN(Variance(new DoubleVector([1, double.NaN, 1e300]), 1)));
        Assert.True(double.IsNaN(StandardDeviation(new DoubleVector([1, double.PositiveInfinity]), 1)));
        Assert.True(double.IsNaN(StandardDeviation(new DoubleVector([1e-300, double.NegativeInfinity]), 1)));
    }

    [Fact]
    public void RowsAndColumnsMatchTheVectorFunctionsBitwise()
    {
        // 行ごとに大きさの違う行列（溢れる行・非正規化数に近い行・ふつうの行・NaN と ∞ の行）
        int[] scales = [-1060, -700, -520, 0, 0, 505, 600, 1000];
        foreach (int columns in new[] { 1, 2, 3, 64, 65, 130 })
        {
            int rows = scales.Length + 2;
            var a = new Matrix<double>(rows, columns);
            for (int i = 0; i < scales.Length; i++)
            {
                double[] values = Scaled(Values(columns, 31 * i + columns, shift: i % 2 == 0 ? 0 : 5), scales[i]);
                for (int j = 0; j < columns; j++)
                {
                    a[i, j] = values[j];
                }
            }

            for (int j = 0; j < columns; j++)
            {
                a[rows - 2, j] = j == columns / 2 ? double.NaN : j;
                a[rows - 1, j] = j == 0 ? double.PositiveInfinity : 1e300 * j;
            }

            foreach (int correction in new[] { 0, 1 })
            {
                var rowVariances = RowVariances(a, correction);
                var rowDeviations = RowStandardDeviations(a, correction);
                var columnVariances = ColumnVariances(a.T, correction);
                var columnDeviations = ColumnStandardDeviations(a.T, correction);
                for (int i = 0; i < rows; i++)
                {
                    double variance = Variance(a.Row(i), correction), deviation = StandardDeviation(a.Row(i), correction);
                    string label = $"columns={columns} row={i} correction={correction}";
                    Assert.True(BitConverter.DoubleToInt64Bits(variance) == BitConverter.DoubleToInt64Bits(rowVariances[i]) || double.IsNaN(variance) && double.IsNaN(rowVariances[i]), $"RowVariances {label}");
                    Assert.True(BitConverter.DoubleToInt64Bits(deviation) == BitConverter.DoubleToInt64Bits(rowDeviations[i]) || double.IsNaN(deviation) && double.IsNaN(rowDeviations[i]), $"RowStandardDeviations {label}");
                    Assert.True(BitConverter.DoubleToInt64Bits(variance) == BitConverter.DoubleToInt64Bits(columnVariances[i]) || double.IsNaN(variance) && double.IsNaN(columnVariances[i]), $"ColumnVariances {label}");
                    Assert.True(BitConverter.DoubleToInt64Bits(deviation) == BitConverter.DoubleToInt64Bits(columnDeviations[i]) || double.IsNaN(deviation) && double.IsNaN(columnDeviations[i]), $"ColumnStandardDeviations {label}");
                }
            }
        }

        var destination = new DoubleVector(3);
        Assert.Throws<ArgumentException>(() => { RowStandardDeviations(new Matrix<double>(2, 4), 1, destination); });
        Assert.Throws<ArgumentOutOfRangeException>(() => { ColumnStandardDeviations(new Matrix<double>(2, 3), -1, destination); });
    }
}
