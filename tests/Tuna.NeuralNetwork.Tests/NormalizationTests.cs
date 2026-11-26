using System.Numerics;

namespace Tuna.NeuralNetwork.Tests;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む。
using Tuna.LinearAlgebra;

/// <summary>LayerNorm・RmsNorm・PReLU・Linear を倍精度の定義と比べる</summary>
public class NormalizationTests
{
    // 列の長さは SIMD の幅の倍数・端数の両方、位置は1つと複数
    public static TheoryData<int, int> Shapes => new() { { 1, 1 }, { 7, 3 }, { 16, 5 }, { 64, 33 }, { 67, 9 } };

    [Theory]
    [MemberData(nameof(Shapes))]
    public void LayerNormMatchesDefinition(int channels, int positions)
    {
        LayerNorm<float>(channels, positions);
        LayerNorm<double>(channels, positions);
    }

    private static void LayerNorm<T>(int channels, int positions) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var x = TestData.Rounded<T>(TestData.Uniform(channels * positions, 1, 3));
        for (int i = 0; i < x.Length; i++)
        {
            x[i] += 2; // 平均が 0 でない入力
        }

        var scale = TestData.Rounded<T>(TestData.Uniform(channels, 2, 0.5, 1.5));
        var shift = TestData.Rounded<T>(TestData.Uniform(channels, 3));
        T epsilon = T.CreateTruncating(1e-5);
        var expected = Reference.LayerNorm(x, channels, positions, scale, shift, double.CreateTruncating(epsilon));
        var input = TestData.Matrix<T>(x, channels, positions);
        var actual = XMath.LayerNorm(input, TestData.Vector<T>(scale), TestData.Vector<T>(shift), epsilon);
        TestData.Close<T>(expected, actual, TestData.Tolerance<T>());

        // 倍率・加算の省略
        TestData.Close<T>(Reference.LayerNorm(x, channels, positions, null, null, double.CreateTruncating(epsilon)),
            XMath.LayerNorm(input, epsilon), TestData.Tolerance<T>());

        // インプレース版と destination オーバーロード
        var inPlace = input.Clone();
        XMath.InPlace.LayerNorm<T>(inPlace, TestData.Vector<T>(scale), TestData.Vector<T>(shift), epsilon);
        TestData.Same<T>(actual, inPlace);
        var destination = new Matrix<T>(channels, positions);
        XMath.LayerNorm<T>(input, TestData.Vector<T>(scale), TestData.Vector<T>(shift), epsilon, destination);
        TestData.Same<T>(actual, destination);
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void RmsNormMatchesDefinition(int channels, int positions)
    {
        RmsNorm<float>(channels, positions);
        RmsNorm<double>(channels, positions);
    }

    private static void RmsNorm<T>(int channels, int positions) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var x = TestData.Rounded<T>(TestData.Uniform(channels * positions, 4, 2));
        var scale = TestData.Rounded<T>(TestData.Uniform(channels, 5, 0.5, 1.5));
        T epsilon = T.CreateTruncating(1e-6);
        var expected = Reference.LayerNorm(x, channels, positions, scale, null, double.CreateTruncating(epsilon), center: false);
        var input = TestData.Matrix<T>(x, channels, positions);
        var actual = XMath.RmsNorm(input, TestData.Vector<T>(scale), epsilon);
        TestData.Close<T>(expected, actual, TestData.Tolerance<T>());
        var inPlace = input.Clone();
        XMath.InPlace.RmsNorm<T>(inPlace, TestData.Vector<T>(scale), epsilon);
        TestData.Same<T>(actual, inPlace);
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void PreluMatchesDefinition(int channels, int positions)
    {
        Prelu<float>(channels, positions);
        Prelu<double>(channels, positions);
    }

    private static void Prelu<T>(int channels, int positions) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var x = TestData.Rounded<T>(TestData.Uniform(channels * positions, 6, 2));
        var slopes = TestData.Rounded<T>(TestData.Uniform(channels, 7, 0, 0.5));
        var input = TestData.Matrix<T>(x, channels, positions);

        // チャネルごとの傾き。PReLU は丸めが1回なので、倍精度の定義を T に丸めた値とビット単位で同じ
        var actual = XMath.Prelu(input, TestData.Vector<T>(slopes));
        TestData.Same<T>(TestData.Matrix<T>(Reference.Prelu(x, channels, slopes), channels, positions), actual);

        // 全チャネルで共通の傾き
        var shared = XMath.Prelu<T>(input, TestData.Vector<T>([0.25]));
        TestData.Same<T>(TestData.Matrix<T>(Reference.Prelu(x, channels, [0.25]), channels, positions), shared);

        var inPlace = input.Clone();
        XMath.InPlace.Prelu<T>(inPlace, TestData.Vector<T>(slopes));
        TestData.Same<T>(actual, inPlace);
    }

    [Theory]
    [InlineData(1, 1, 1)]
    [InlineData(5, 3, 4)]
    [InlineData(64, 128, 33)]
    public void LinearMatchesDefinition(int outputs, int inputs, int positions)
    {
        Linear<float>(outputs, inputs, positions);
        Linear<double>(outputs, inputs, positions);
    }

    private static void Linear<T>(int outputs, int inputs, int positions) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var w = TestData.Rounded<T>(TestData.Uniform(outputs * inputs, 8, 1 / Math.Sqrt(inputs)));
        var x = TestData.Rounded<T>(TestData.Uniform(inputs * positions, 9));
        var b = TestData.Rounded<T>(TestData.Uniform(outputs, 10));

        // PyTorch の nn.Linear.weight（行優先 [outputs, inputs]）は転置ビューで渡せる
        var weight = new ReadOnlyMatrixSpan<T>(TestData.To<T>(w), inputs, outputs).T;
        var input = TestData.Matrix<T>(x, inputs, positions);
        var actual = XMath.Linear(weight, input, TestData.Vector<T>(b));
        TestData.Close<T>(Reference.Linear(w, x, b, outputs, inputs, positions), actual, TestData.Tolerance<T>());
        TestData.Close<T>(Reference.Linear(w, x, null, outputs, inputs, positions), XMath.Linear(weight, input),
            TestData.Tolerance<T>());
    }

    /// <summary>
    /// 共通の大きな値を持つ列でも、平均の丸め誤差が結果に残らない。定数の列はどの経路でもちょうど 0（倍率・加算があれば加算）になり、
    /// 小さなゆらぎを足した列は倍精度の定義に合う
    /// </summary>
    [Fact]
    public void LayerNormIsStableUnderCommonOffset()
    {
        CommonOffset<float>();
        CommonOffset<double>();
    }

    private static void CommonOffset<T>() where T : unmanaged, IFloatingPointIeee754<T>
    {
        const int positions = 3;
        T epsilon = T.CreateTruncating(1e-5);
        var none = new Vector<T>(0);
        foreach (int channels in new[] { 1, 7, 128, 1024 })
        {
            var scale = TestData.Vector<T>(TestData.Uniform(channels, 4, 0.5, 1.5));
            var shift = TestData.Vector<T>(TestData.Uniform(channels, 5));
            foreach (double value in new[] { 1000.1, 1.1, -3e5, 1e-3 })
            {
                var input = new Matrix<T>(channels, positions);
                input.AsSpan().Fill(T.CreateTruncating(value));
                var transposed = new Matrix<T>(positions, channels);
                transposed.AsSpan().Fill(T.CreateTruncating(value));

                void Zero(ReadOnlyMatrixSpan<T> y, ReadOnlyVectorSpan<T> expected)
                {
                    for (int j = 0; j < y.Columns; j++)
                    {
                        for (int i = 0; i < y.Rows; i++)
                        {
                            T e = expected.Length == 0 ? T.Zero : expected[i];
                            Assert.True(y[i, j] == e, $"{typeof(T).Name} n={channels} x={value}: ({i}, {j}) = {y[i, j]}, expected {e}");
                        }
                    }
                }

                Zero(XMath.LayerNorm(input, epsilon), none);
                Zero(XMath.LayerNorm(input, scale, shift, epsilon), shift);
                var destination = new Matrix<T>(channels, positions);
                XMath.LayerNorm<T>(input, none, shift, epsilon, destination);
                Zero(destination, shift);
                var inPlace = input.Clone();
                XMath.InPlace.LayerNorm<T>(inPlace, scale, none, epsilon);
                Zero(inPlace, none);
                Zero(XMath.LayerNorm(transposed.T, scale, shift, epsilon), shift);
            }

            // 1000 に ±0.01 のゆらぎ（float の平均を要素型に丸めると、その誤差がゆらぎの標準偏差の 1% 近くになる）。
            // 基準も倍精度の平均の丸め誤差を含まないよう、1000 を引いた値（どちらの要素型でも誤差なく引ける）から求める
            var x = TestData.Rounded<T>(TestData.Uniform(channels * positions, 6, 0.01).Select(v => 1000 + v).ToArray());
            var expected = Reference.LayerNorm(x.Select(v => v - 1000).ToArray(), channels, positions, null, null, double.CreateTruncating(epsilon));
            TestData.Close<T>(expected, XMath.LayerNorm(TestData.Matrix<T>(x, channels, positions), epsilon), TestData.Tolerance<T>());
        }
    }

    [Fact]
    public void StridedViewsGiveTheSameBits()
    {
        Strided<float>();
        Strided<double>();
    }

    private static void Strided<T>() where T : unmanaged, IFloatingPointIeee754<T>
    {
        const int channels = 24, positions = 10;
        var x = TestData.Uniform(channels * positions, 11, 2);
        var scale = TestData.Vector<T>(TestData.Uniform(channels, 12, 0.5, 1.5));
        var shift = TestData.Vector<T>(TestData.Uniform(channels, 13));
        T epsilon = T.CreateTruncating(1e-5);
        var dense = TestData.Matrix<T>(x, channels, positions);
        var expected = XMath.LayerNorm(dense, scale, shift, epsilon);

        // 行の間隔が 1 でない入力（転置の転置）と、行の間隔が 1 でない出力先
        var transposed = TestData.Matrix<T>(x, channels, positions).T.ToMatrix(); // [positions × channels]
        var destination = new Matrix<T>(positions, channels);
        XMath.LayerNorm(transposed.T, scale, shift, epsilon, destination.T);
        TestData.Same<T>(expected, destination.T);

        // 大きな行列の一部（列の間隔が行数より大きい）
        var wide = new Matrix<T>(channels + 5, positions);
        dense.AsMatrixSpan().CopyTo(wide[2..(channels + 2), ..]);
        TestData.Same<T>(expected, XMath.LayerNorm(wide[2..(channels + 2), ..], scale, shift, epsilon));

        // 間隔のある倍率（行列の行）
        var scales = new Matrix<T>(3, channels);
        scale.AsVectorSpan().CopyTo(scales.Row(1));
        TestData.Same<T>(expected, XMath.LayerNorm<T>(dense, scales.Row(1), shift, epsilon));
    }

    [Fact]
    public void InvalidArgumentsThrow()
    {
        var x = new Matrix<float>(4, 3);
        var none = new Vector<float>(0);
        Assert.Throws<ArgumentException>(() => XMath.LayerNorm(x, new Vector<float>(3), none, 1e-5f));
        Assert.Throws<ArgumentException>(() => XMath.LayerNorm(x, none, new Vector<float>(5), 1e-5f));
        Assert.Throws<ArgumentOutOfRangeException>(() => XMath.LayerNorm(x, -1f));
        Assert.Throws<ArgumentOutOfRangeException>(() => XMath.LayerNorm(x, float.NaN));
        Assert.Throws<ArgumentException>(() => XMath.LayerNorm(x, none, none, 1e-5f, new Matrix<float>(4, 2)));
        Assert.Throws<ArgumentException>(() => XMath.Prelu(x, new Vector<float>(2)));
        Assert.Throws<ArgumentException>(() => XMath.Linear(new Matrix<float>(2, 5), x));
        Assert.Throws<NotSupportedException>(() => XMath.LayerNorm(new Matrix<Half>(2, 2), Half.Epsilon));

        // 出力先が入力と一部だけ重なる
        var buffer = new Matrix<float>(4, 6);
        Assert.Throws<ArgumentException>(() => XMath.LayerNorm<float>(buffer[.., 0..3], none, none, 1e-5f, buffer[.., 1..4]));

        // 同じ行列の行の範囲（列の間隔が同じ）: 行が重ならなければよく、重なれば例外
        var stacked = new Matrix<float>(8, 5);
        XMath.LayerNorm<float>(stacked[0..4, ..], none, none, 1e-5f, stacked[4..8, ..]);
        XMath.LayerNorm<float>(stacked[4..8, 1..], none, none, 1e-5f, stacked[0..4, ..4]);
        Assert.Throws<ArgumentException>(() => XMath.LayerNorm<float>(stacked[0..4, ..], none, none, 1e-5f, stacked[2..6, ..]));
        Assert.Throws<ArgumentException>(() => XMath.LayerNorm<float>(stacked[4..8, 1..], none, none, 1e-5f, stacked[4..8, ..4]));
        XMath.LayerNorm<float>(stacked[4..8, 1..3], none, none, 1e-5f, stacked[4..8, 3..5]);
    }

    /// <summary>
    /// 同じ行列から切り出した同じ形の部分行列のすべての組で、重なりの判定（行の範囲の速い経路を含む）を要素の位置の集合と比べる。
    /// LayerNorm は入力と同じ出力先を許し、因果畳み込みは許さない
    /// </summary>
    [Fact]
    public void OverlapCheckMatchesElementSets()
    {
        const int rows = 8, columns = 6;
        var buffer = new Matrix<float>(rows, columns);
        var none = new Vector<float>(0);
        var views = new List<(int R, int C, int H, int W)>();
        for (int h = 1; h <= rows; h++)
        {
            for (int w = 1; w <= columns; w++)
            {
                for (int r = 0; r + h <= rows; r++)
                {
                    for (int c = 0; c + w <= columns; c++)
                    {
                        views.Add((r, c, h, w));
                    }
                }
            }
        }

        static HashSet<int> Elements((int R, int C, int H, int W) v)
        {
            var set = new HashSet<int>();
            for (int j = 0; j < v.W; j++)
            {
                for (int i = 0; i < v.H; i++)
                {
                    set.Add((v.C + j) * rows + v.R + i);
                }
            }

            return set;
        }

        int checkedPairs = 0;
        foreach (var group in views.GroupBy(v => (v.H, v.W)))
        {
            foreach (var a in group)
            {
                var elements = Elements(a);
                foreach (var b in group)
                {
                    bool exact = a == b, overlaps = elements.Overlaps(Elements(b));
                    void Normalize() => XMath.LayerNorm<float>(buffer[a.R..(a.R + a.H), a.C..(a.C + a.W)], none, none, 1e-5f,
                        buffer[b.R..(b.R + b.H), b.C..(b.C + b.W)]);
                    void Convolve() => XMath.CausalDepthwiseConvolve<float>(buffer[a.R..(a.R + a.H), a.C..(a.C + a.W)],
                        new Matrix<float>(a.H, 1), none, buffer[b.R..(b.R + b.H), b.C..(b.C + b.W)]);
                    if (overlaps && !exact)
                    {
                        Assert.Throws<ArgumentException>(Normalize);
                    }
                    else
                    {
                        Normalize();
                    }

                    if (overlaps)
                    {
                        Assert.Throws<ArgumentException>(Convolve);
                    }
                    else
                    {
                        Convolve();
                    }

                    checkedPairs++;
                }
            }
        }

        Assert.Equal(18564, checkedPairs);
    }
}
