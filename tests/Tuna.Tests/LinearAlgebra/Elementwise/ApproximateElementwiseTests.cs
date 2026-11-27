using static Tuna.XMath;

namespace Tuna.Tests.LinearAlgebra;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む。
using Tuna.LinearAlgebra;

// approximate: true の Exp・Tanh・Logistic・Silu・Softplus・Gelu を検証する。
// float 近似式の ULP 誤差上限、特殊値、スカラーと SIMD のビット一致、および double・Half の挙動（Gelu 以外は approximate: false と同一）を確認する。
[Trait("LinearAlgebra", "XMath")]
public class ApproximateElementwiseTests
{
    private sealed record Approximation(string Name, Func<Vector<float>, Vector<float>> Vector, Func<float, float>? Scalar, Func<double, double> Truth, double MaxUlp);

    private static readonly Approximation[] Approximations =
    [
        new("Exp", x => Exp(x, true), null, Math.Exp, 1.07),
        new("Tanh", x => Tanh(x, true), v => Tanh(v, true), Math.Tanh, 1.1),
        new("Logistic", x => Logistic(x, true), v => Logistic(v, true), Logistic, 2.41),
        new("Silu", x => Silu(x, true), v => Silu(v, true), Silu, 3.35),
        new("Softplus", x => Softplus(x, true), v => Softplus(v, true), Softplus, 2.8),
        // 負の側は指数の引数 z = c·(x + k·x³) の float の丸めが |z| 倍に広がる（x ≈ -10 で z ≈ -87、値は 10⁻³⁷ 程度）
        new("Gelu", x => Gelu(x, true), v => Gelu(v, true), v => Gelu(v, true), 203),
    ];

    /// <summary>float の間隔（真の値の位置。非正規化数は 2^-149）</summary>
    private static double Ulp(double v)
    {
        v = Math.Abs(v);
        return v < 1.1754943508222875e-38 ? Math.ScaleB(1.0, -149) : Math.ScaleB(1.0, Math.ILogB(v) - 23);
    }

    private static float[] Inputs(int seed)
    {
        var random = new Random(seed);
        return Enumerable.Range(0, 50_000).Select(_ => (float)(240 * random.NextDouble() - 140))
            .Concat(Enumerable.Range(0, 50_000).Select(_ => (float)(20 * random.NextDouble() - 10)))
            .Concat(Enumerable.Range(0, 50_000).Select(_ => BitConverter.Int32BitsToSingle(random.Next() * (random.Next(2) == 0 ? 1 : -1))))
            .Where(float.IsFinite)
            .ToArray();
    }

    [Fact]
    public void SingleApproximationsStayWithinTheirErrorBounds()
    {
        var x = Inputs(17);
        foreach (var f in Approximations)
        {
            var y = f.Vector(new Vector<float>(x));
            for (int i = 0; i < x.Length; i++)
            {
                double truth = f.Truth(x[i]);
                float rounded = (float)truth;
                if (!float.IsFinite(rounded))
                {
                    // 溢れ（Exp の x > 88.72 など）は ∞ で、正しく丸めた値と同じ
                    Assert.True(rounded == y[i], $"{f.Name}({x[i]:R}) = {y[i]:R}, reference {truth:R}");
                    continue;
                }

                double error = Math.Abs(y[i] - truth) / Ulp(truth);
                Assert.True(error <= f.MaxUlp, $"{f.Name}({x[i]:R}) = {y[i]:R}, reference {truth:R}, error {error:F3} ulp");
            }
        }

        // Gelu は |x| が小さい側では誤差が小さい（x ≥ -3 で 15.2 ulp 以下）
        var small = x.Where(v => v >= -3).ToArray();
        var gelu = Gelu(new Vector<float>(small), true);
        for (int i = 0; i < small.Length; i++)
        {
            double truth = Gelu((double)small[i], true);
            Assert.True(Math.Abs(gelu[i] - truth) / Ulp(truth) <= 15.2, $"Gelu({small[i]:R}) = {gelu[i]:R}, reference {truth:R}");
        }
    }

    [Fact]
    public void ScalarVersionsMatchTheVectorLanesBitwise()
    {
        float[] special =
        [
            0, -0f, 1, -1, float.Epsilon, -float.Epsilon, float.MaxValue, float.MinValue, float.PositiveInfinity, float.NegativeInfinity,
            88.72283f, 88.7229f, -87.33655f, -103.97f, -104, -150, -151, -1e30f, 1e30f, 1e-30f, -1e-30f,
        ];
        var x = special.Concat(Inputs(19).Take(20_000)).ToArray();
        foreach (var f in Approximations.Where(f => f.Scalar is not null))
        {
            var y = f.Vector(new Vector<float>(x));
            for (int i = 0; i < x.Length; i++)
            {
                float scalar = f.Scalar!(x[i]);
                Assert.True(BitConverter.SingleToInt32Bits(scalar) == BitConverter.SingleToInt32Bits(y[i]),
                    $"{f.Name}({x[i]:R}): scalar {scalar:R}, vector {y[i]:R}");
            }
        }
    }

    /// <summary>1要素（端数の経路）と 37 要素の各位置（SIMD の本体と端数）の結果</summary>
    private static float[] Everywhere(Func<Vector<float>, Vector<float>> function, float value)
    {
        var results = new List<float> { function(new Vector<float>([value]))[0] };
        for (int position = 0; position < 37; position++)
        {
            var x = Enumerable.Repeat(0.5f, 37).ToArray();
            x[position] = value;
            results.Add(function(new Vector<float>(x))[position]);
        }

        return results.ToArray();
    }

    private static void AssertEverywhere(string name, Func<Vector<float>, Vector<float>> function, float value, Func<float, bool> expected)
    {
        foreach (float y in Everywhere(function, value))
        {
            Assert.True(expected(y), $"{name}({value:R}) = {y:R} ({BitConverter.SingleToInt32Bits(y):X8})");
        }
    }

    private static void AssertBits(string name, Func<Vector<float>, Vector<float>> function, float value, float expected) =>
        AssertEverywhere(name, function, value, y => BitConverter.SingleToInt32Bits(y) == BitConverter.SingleToInt32Bits(expected));

    private static void AssertNear(string name, Func<Vector<float>, Vector<float>> function, float value, double truth, double ulps) =>
        AssertEverywhere(name, function, value, y => Math.Abs(y - truth) <= ulps * Ulp(truth));

    [Fact]
    public void SpecialInputsGiveTheLimitingValuesAndNaNPassesThrough()
    {
        Func<Vector<float>, Vector<float>> exp = x => Exp(x, true), logistic = x => Logistic(x, true), silu = x => Silu(x, true),
            softplus = x => Softplus(x, true), gelu = x => Gelu(x, true), tanh = x => Tanh(x, true);

        // 指数: ±∞・±0・溢れの境目・非正規化数・0 への下位桁あふれ
        AssertBits("Exp", exp, float.PositiveInfinity, float.PositiveInfinity);
        AssertBits("Exp", exp, float.NegativeInfinity, 0);
        AssertBits("Exp", exp, 0, 1);
        AssertBits("Exp", exp, -0f, 1);
        AssertBits("Exp", exp, 88.7229f, float.PositiveInfinity);
        AssertBits("Exp", exp, 1000, float.PositiveInfinity);
        AssertNear("Exp", exp, 88.72283f, Math.Exp(88.72283f), 1.07);
        AssertNear("Exp", exp, -87.33655f, Math.Exp(-87.33655f), 1.07);
        AssertNear("Exp", exp, -100, Math.Exp(-100), 1);   // 非正規化数（2^-149 の倍数に丸める）
        AssertBits("Exp", exp, -103.97f, float.Epsilon);
        AssertBits("Exp", exp, -104, 0);
        AssertBits("Exp", exp, -1000, 0);

        AssertBits("Tanh", tanh, float.PositiveInfinity, 1);
        AssertBits("Tanh", tanh, float.NegativeInfinity, -1);
        AssertBits("Tanh", tanh, float.MaxValue, 1);
        AssertBits("Tanh", tanh, float.MinValue, -1);
        AssertBits("Tanh", tanh, 0, 0);
        AssertBits("Tanh", tanh, -0f, -0f);
        AssertBits("Tanh", tanh, float.Epsilon, float.Epsilon);
        AssertBits("Tanh", tanh, -float.Epsilon, -float.Epsilon);

        // ロジスティック関数
        AssertBits("Logistic", logistic, float.PositiveInfinity, 1);
        AssertBits("Logistic", logistic, float.NegativeInfinity, 0);
        AssertBits("Logistic", logistic, 0, 0.5f);
        AssertBits("Logistic", logistic, -0f, 0.5f);
        AssertBits("Logistic", logistic, 200, 1);
        AssertNear("Logistic", logistic, -100, Math.Exp(-100), 1);
        AssertBits("Logistic", logistic, -200, 0);

        // SiLU: 負の大きな x でも eˣ が非正規化数になる前に x を掛けるので、x·eˣ が正規化数なら精度を保つ
        AssertBits("Silu", silu, float.PositiveInfinity, float.PositiveInfinity);
        AssertBits("Silu", silu, float.NegativeInfinity, -0f);
        AssertBits("Silu", silu, -0f, -0f);
        AssertBits("Silu", silu, 0, 0);
        AssertNear("Silu", silu, -90, -90 * Math.Exp(-90), 3.35);
        AssertNear("Silu", silu, -105, -105 * Math.Exp(-105), 1);
        AssertBits("Silu", silu, -200, -0f);
        AssertBits("Silu", silu, -1e30f, -0f);
        AssertBits("Silu", silu, 1e30f, 1e30f);

        // Softplus
        AssertBits("Softplus", softplus, float.PositiveInfinity, float.PositiveInfinity);
        AssertBits("Softplus", softplus, float.NegativeInfinity, 0);
        AssertNear("Softplus", softplus, 0, Math.Log(2), 2.8);
        AssertBits("Softplus", softplus, 100, 100);
        AssertBits("Softplus", softplus, 1e30f, 1e30f);
        AssertNear("Softplus", softplus, -100, Math.Exp(-100), 1);
        AssertBits("Softplus", softplus, -1000, 0);

        // tanh で近似した GELU
        AssertBits("Gelu", gelu, float.PositiveInfinity, float.PositiveInfinity);
        AssertBits("Gelu", gelu, float.NegativeInfinity, -0f);
        AssertBits("Gelu", gelu, -0f, -0f);
        AssertBits("Gelu", gelu, 0, 0);
        AssertBits("Gelu", gelu, 1e30f, 1e30f);
        AssertBits("Gelu", gelu, -1e13f, -0f);
        AssertBits("Gelu", gelu, -100, -0f);
        AssertNear("Gelu", gelu, -11, Gelu(-11.0, true), 1);   // z ≈ -112.5。真の値はfloatの最小非正規数より小さい

        // NaN は入力そのもの（符号・内容・シグナリングを変えない）
        float[] nans =
        [
            float.NaN, BitConverter.Int32BitsToSingle(0x7FC0_0042), BitConverter.Int32BitsToSingle(unchecked((int)0xFFC0_1234)),
            BitConverter.Int32BitsToSingle(0x7F80_0001), BitConverter.Int32BitsToSingle(unchecked((int)0xFFA0_0000)),
        ];
        foreach (float nan in nans)
        {
            foreach (var f in Approximations)
            {
                AssertBits(f.Name, f.Vector, nan, nan);
                if (f.Scalar is not null)
                {
                    Assert.Equal(BitConverter.SingleToInt32Bits(nan), BitConverter.SingleToInt32Bits(f.Scalar(nan)));
                }
            }
        }
    }

    [Fact]
    public void DoubleAndHalfGiveTheSameValuesAsTheExactVersions()
    {
        var random = new Random(23);
        var x = new double[] { 0, -0.0, double.PositiveInfinity, double.NegativeInfinity, double.NaN, -745.2, 709.9, -1000, 1000 }
            .Concat(Enumerable.Range(0, 3_000).Select(_ => 100 * random.NextDouble() - 50)).ToArray();
        var v = new Vector<double>(x);
        Assert.Equal(Exp(v).ToArray(), Exp(v, true).ToArray());
        Assert.Equal(Tanh(v).ToArray().Select(BitConverter.DoubleToInt64Bits), Tanh(v, true).ToArray().Select(BitConverter.DoubleToInt64Bits));
        Assert.Equal(Logistic(v).ToArray(), Logistic(v, true).ToArray());
        Assert.Equal(Silu(v).ToArray(), Silu(v, true).ToArray());
        Assert.Equal(Softplus(v).ToArray(), Softplus(v, true).ToArray());
        foreach (double value in x)
        {
            Assert.Equal(BitConverter.DoubleToInt64Bits(Tanh(value)), BitConverter.DoubleToInt64Bits(Tanh(value, true)));
            Assert.Equal(Logistic(value), Logistic(value, true));
            Assert.Equal(Silu(value), Silu(value, true));
            Assert.Equal(Softplus(value), Softplus(value, true));
        }

        // Half は1要素ずつ、approximate: false と同じ値（Gelu は tanh による近似の式を double で計算して丸める）
        var h = x.Select(value => (Half)value).ToArray();
        var hv = new Vector<Half>(h);
        Assert.Equal(Exp(hv).ToArray(), Exp(hv, true).ToArray());
        Assert.Equal(Tanh(hv).ToArray().Select(BitConverter.HalfToInt16Bits), Tanh(hv, true).ToArray().Select(BitConverter.HalfToInt16Bits));
        foreach (Half value in h)
        {
            Assert.Equal(BitConverter.HalfToInt16Bits(Tanh(value)), BitConverter.HalfToInt16Bits(Tanh(value, true)));
        }

        Assert.Equal(Logistic(hv).ToArray(), Logistic(hv, true).ToArray());
        Assert.Equal(Silu(hv).ToArray(), Silu(hv, true).ToArray());
        Assert.Equal(Softplus(hv).ToArray(), Softplus(hv, true).ToArray());
        Assert.Equal(h.Select(value => (Half)Gelu((double)value, true)).ToArray(), Gelu(hv, true).ToArray());
        Assert.Equal(Gelu(hv).ToArray(), Gelu(hv, false).ToArray());
    }

    private static float[] ColumnMajor(Matrix<float> m) =>
        Enumerable.Range(0, m.Columns).SelectMany(j => Enumerable.Range(0, m.Rows).Select(i => m[i, j])).ToArray();

    [Fact]
    public void EveryOverloadComputesTheSameValues()
    {
        var random = new Random(29);
        var values = Enumerable.Range(0, 6 * 7).Select(_ => (float)(20 * random.NextDouble() - 10)).ToArray();
        var a = Matrix<float>.FromColumnMajor(6, 7, values);
        var x = new Vector<float>(values);

        void Check(string name, Func<Vector<float>, bool, Vector<float>> allocating, Action<Vector<float>, bool, Vector<float>> into,
            Func<Matrix<float>, bool, Matrix<float>> matrix, Action<Matrix<float>, bool, Matrix<float>> matrixInto,
            Action<Vector<float>, bool> inPlace, Action<Matrix<float>, bool> matrixInPlace, Func<Vector<float>, Vector<float>> exact)
        {
            var expected = allocating(x, true).ToArray();
            var destination = new Vector<float>(x.Length);
            into(x, true, destination);
            Assert.Equal(expected, destination.ToArray());
            var copy = x.ToArray().ToVector();
            inPlace(copy, true);
            Assert.Equal(expected, copy.ToArray());
            Assert.Equal(expected, ColumnMajor(matrix(a, true)));
            var md = new Matrix<float>(6, 7);
            matrixInto(a, true, md);
            Assert.Equal(expected, ColumnMajor(md));
            var mc = a.Clone();
            matrixInPlace(mc, true);
            Assert.Equal(expected, ColumnMajor(mc));
            Assert.Equal(matrix(a, true).T.ToArray(), matrix(a.T.ToMatrix(), true).ToArray());

            // approximate: false は元の版と同じ
            Assert.True(exact(x).ToArray().SequenceEqual(allocating(x, false).ToArray()), $"{name}: approximate: false differs from the exact version");
        }

        Check("Exp", (v, p) => Exp(v, p), (v, p, d) => Exp(v, p, d), (m, p) => Exp(m, p), (m, p, d) => Exp(m, p, d),
            (v, p) => InPlace.Exp(v, p), (m, p) => InPlace.Exp(m, p), v => Exp(v));
        Check("Tanh", (v, p) => Tanh(v, p), (v, p, d) => Tanh(v, p, d), (m, p) => Tanh(m, p), (m, p, d) => Tanh(m, p, d),
            (v, p) => InPlace.Tanh(v, p), (m, p) => InPlace.Tanh(m, p), v => Tanh(v));
        Check("Logistic", (v, p) => Logistic(v, p), (v, p, d) => Logistic(v, p, d), (m, p) => Logistic(m, p), (m, p, d) => Logistic(m, p, d),
            (v, p) => InPlace.Logistic(v, p), (m, p) => InPlace.Logistic(m, p), v => Logistic(v));
        Check("Silu", (v, p) => Silu(v, p), (v, p, d) => Silu(v, p, d), (m, p) => Silu(m, p), (m, p, d) => Silu(m, p, d),
            (v, p) => InPlace.Silu(v, p), (m, p) => InPlace.Silu(m, p), v => Silu(v));
        Check("Softplus", (v, p) => Softplus(v, p), (v, p, d) => Softplus(v, p, d), (m, p) => Softplus(m, p), (m, p, d) => Softplus(m, p, d),
            (v, p) => InPlace.Softplus(v, p), (m, p) => InPlace.Softplus(m, p), v => Softplus(v));
        Check("Gelu", (v, p) => Gelu(v, p), (v, p, d) => Gelu(v, p, d), (m, p) => Gelu(m, p), (m, p, d) => Gelu(m, p, d),
            (v, p) => InPlace.Gelu(v, p), (m, p) => InPlace.Gelu(m, p), v => Gelu(v));
    }
}
