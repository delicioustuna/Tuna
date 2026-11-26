using System.Numerics;

namespace Tuna.NeuralNetwork.Tests;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む。
using Tuna.LinearAlgebra;

/// <summary>因果畳み込みと選択的走査を倍精度の定義と比べ、区切り方・1ステップ版との一致を確かめる</summary>
public class StateSpaceTests
{
    [Theory]
    [InlineData(1, 1, 1)]
    [InlineData(5, 7, 4)]
    [InlineData(32, 20, 4)]
    [InlineData(37, 3, 6)] // 位置が核より短い
    public void CausalDepthwiseConvolveMatchesDefinition(int channels, int positions, int taps)
    {
        CausalDepthwise<float>(channels, positions, taps);
        CausalDepthwise<double>(channels, positions, taps);
    }

    private static void CausalDepthwise<T>(int channels, int positions, int taps) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var x = TestData.Rounded<T>(TestData.Uniform(channels * positions, 20));
        var w = TestData.Rounded<T>(TestData.Uniform(channels * taps, 21));
        var b = TestData.Rounded<T>(TestData.Uniform(channels, 22));

        // nn.Conv1d(D, D, K, groups=D) の重み（行優先 [D, 1, K]）は [K × D] の列優先の転置として渡せる
        var kernel = new ReadOnlyMatrixSpan<T>(TestData.To<T>(w), taps, channels).T;
        var input = TestData.Matrix<T>(x, channels, positions);
        var actual = XMath.CausalDepthwiseConvolve(input, kernel, TestData.Vector<T>(b));
        TestData.Close<T>(Reference.CausalDepthwise(x, w, b, channels, positions, taps), actual, TestData.Tolerance<T>());
        TestData.Close<T>(Reference.CausalDepthwise(x, w, null, channels, positions, taps),
            XMath.CausalDepthwiseConvolve(input, kernel), TestData.Tolerance<T>());
    }

    /// <summary>走査の入力（Mamba と同じ範囲の値: Δ > 0、A &lt; 0）</summary>
    private sealed class ScanInputs
    {
        public ScanInputs(int channels, int states, int positions, int seed, bool skip = true)
        {
            Channels = channels;
            States = states;
            Positions = positions;
            X = TestData.Uniform(channels * positions, seed);
            Delta = TestData.Uniform(channels * positions, seed + 1, 0.001, 0.5);
            A = TestData.Uniform(channels * states, seed + 2, -8, -0.5);
            B = TestData.Uniform(states * positions, seed + 3);
            C = TestData.Uniform(states * positions, seed + 4);
            D = skip ? TestData.Uniform(channels, seed + 5, 0.5, 1.5) : [];
        }

        public int Channels { get; }
        public int States { get; }
        public int Positions { get; }
        public double[] X { get; private set; }
        public double[] Delta { get; private set; }
        public double[] A { get; private set; }
        public double[] B { get; private set; }
        public double[] C { get; private set; }
        public double[] D { get; private set; }

        public ScanInputs Rounded<T>() where T : INumberBase<T>
        {
            var copy = (ScanInputs)MemberwiseClone();
            copy.X = TestData.Rounded<T>(X);
            copy.Delta = TestData.Rounded<T>(Delta);
            copy.A = TestData.Rounded<T>(A);
            copy.B = TestData.Rounded<T>(B);
            copy.C = TestData.Rounded<T>(C);
            copy.D = TestData.Rounded<T>(D);
            return copy;
        }

        public Matrix<T> XMatrix<T>() where T : unmanaged, INumberBase<T> => TestData.Matrix<T>(X, Channels, Positions);
        public Matrix<T> DeltaMatrix<T>() where T : unmanaged, INumberBase<T> => TestData.Matrix<T>(Delta, Channels, Positions);

        /// <summary>A は行優先 [D, N] で持つので、列優先の [D × N] は転置ビューで作る</summary>
        public Matrix<T> AMatrix<T>() where T : unmanaged, INumberBase<T> =>
            new ReadOnlyMatrixSpan<T>(TestData.To<T>(A), States, Channels).T.ToMatrix();

        public Matrix<T> BMatrix<T>() where T : unmanaged, INumberBase<T> => TestData.Matrix<T>(B, States, Positions);
        public Matrix<T> CMatrix<T>() where T : unmanaged, INumberBase<T> => TestData.Matrix<T>(C, States, Positions);
        public Vector<T> DVector<T>() where T : unmanaged, INumberBase<T> => TestData.Vector<T>(D);
    }

    [Theory]
    [InlineData(1, 1, 1, true)]
    [InlineData(5, 3, 7, true)]
    [InlineData(16, 16, 9, false)]
    [InlineData(70, 16, 12, true)] // チャネルが Vector512 の幅の倍数でない
    public void SelectiveScanMatchesDefinition(int channels, int states, int positions, bool skip)
    {
        Scan<float>(new ScanInputs(channels, states, positions, 30, skip));
        Scan<double>(new ScanInputs(channels, states, positions, 30, skip));
    }

    private static void Scan<T>(ScanInputs inputs) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var r = inputs.Rounded<T>();
        var expected = Reference.SelectiveScan(r.X, r.Delta, r.A, r.B, r.C, r.D.Length == 0 ? null : r.D,
            r.Channels, r.States, r.Positions, new double[r.Channels * r.States]);
        var actual = XMath.SelectiveScan(r.XMatrix<T>(), r.DeltaMatrix<T>(), r.AMatrix<T>(), r.BMatrix<T>(), r.CMatrix<T>(),
            r.DVector<T>(), approximate: false);
        TestData.Close<T>(expected, actual, TestData.Tolerance<T>() * 10);

        // 近似版の exp（float は数 ulp、double は正確な版と同じ）
        var approximate = XMath.SelectiveScan(r.XMatrix<T>(), r.DeltaMatrix<T>(), r.AMatrix<T>(), r.BMatrix<T>(), r.CMatrix<T>(),
            r.DVector<T>(), approximate: true);
        TestData.Close<T>(expected, approximate, TestData.Tolerance<T>() * 10);
    }

    [Fact]
    public void SplitScanContinuesFromState()
    {
        SplitScan<float>();
        SplitScan<double>();
    }

    /// <summary>系列を区切って状態を渡しながら走査しても、まとめて走査したのと同じビット</summary>
    private static void SplitScan<T>() where T : unmanaged, IFloatingPointIeee754<T>
    {
        var inputs = new ScanInputs(40, 8, 13, 40);
        var (x, delta, a, b, c, d) = (inputs.XMatrix<T>(), inputs.DeltaMatrix<T>(), inputs.AMatrix<T>(), inputs.BMatrix<T>(),
            inputs.CMatrix<T>(), inputs.DVector<T>());
        var whole = XMath.SelectiveScan(x, delta, a, b, c, d, approximate: true);
        var state = new Matrix<T>(40, 8);
        var parts = new Matrix<T>(40, 13);
        foreach (var (start, end) in new[] { (0, 5), (5, 6), (6, 13) })
        {
            XMath.SelectiveScan<T>(x[.., start..end], delta[.., start..end], a, b[.., start..end], c[.., start..end], d, state,
                true, parts[.., start..end]);
        }

        TestData.Same<T>(whole, parts);

        // 最後の状態も定義と合う
        var h = new double[40 * 8];
        var r = inputs.Rounded<T>();
        Reference.SelectiveScan(r.X, r.Delta, r.A, r.B, r.C, r.D, 40, 8, 13, h);
        TestData.Close<T>(new ReadOnlyMatrixSpan<double>(h, 8, 40).T.ToMatrix().AsSpan().ToArray(), state, TestData.Tolerance<T>() * 10);
    }

    [Fact]
    public void StepMatchesScanForEachSequence()
    {
        Step<float>();
        Step<double>();
    }

    /// <summary>S 本の系列を1位置ずつ進めると、各系列をまとめて走査したのと同じビット</summary>
    private static void Step<T>() where T : unmanaged, IFloatingPointIeee754<T>
    {
        const int channels = 36, states = 5, positions = 6, sequences = 3;
        var all = Enumerable.Range(0, sequences).Select(s => new ScanInputs(channels, states, positions, 50 + 10 * s)).ToArray();
        var a = all[0].AMatrix<T>();
        var d = all[0].DVector<T>();
        var expected = all.Select(s => XMath.SelectiveScan(s.XMatrix<T>(), s.DeltaMatrix<T>(), a, s.BMatrix<T>(), s.CMatrix<T>(), d,
            approximate: true)).ToArray();
        var state = new Tensor<T>(channels, states, sequences);
        var y = new Matrix<T>(channels, sequences);
        var x = new Matrix<T>(channels, sequences);
        var delta = new Matrix<T>(channels, sequences);
        var b = new Matrix<T>(states, sequences);
        var c = new Matrix<T>(states, sequences);
        for (int t = 0; t < positions; t++)
        {
            for (int s = 0; s < sequences; s++)
            {
                all[s].XMatrix<T>().Column(t).CopyTo(x.Column(s));
                all[s].DeltaMatrix<T>().Column(t).CopyTo(delta.Column(s));
                all[s].BMatrix<T>().Column(t).CopyTo(b.Column(s));
                all[s].CMatrix<T>().Column(t).CopyTo(c.Column(s));
            }

            XMath.SelectiveScanStep<T>(x, delta, a, b, c, d, state, true, y);
            for (int s = 0; s < sequences; s++)
            {
                TestData.Same<T>(expected[s][.., t..(t + 1)], y[.., s..(s + 1)]);
            }
        }
    }

    [Fact]
    public void InvalidScanArgumentsThrow()
    {
        var inputs = new ScanInputs(8, 4, 3, 60);
        var (x, delta, a, b, c, d) = (inputs.XMatrix<float>(), inputs.DeltaMatrix<float>(), inputs.AMatrix<float>(), inputs.BMatrix<float>(),
            inputs.CMatrix<float>(), inputs.DVector<float>());
        Assert.Throws<ArgumentException>(() => XMath.SelectiveScan<float>(x, delta[.., ..2], a, b, c, d, false));
        Assert.Throws<ArgumentException>(() => XMath.SelectiveScan<float>(x, delta, a[..4, ..], b, c, d, false));
        Assert.Throws<ArgumentException>(() => XMath.SelectiveScan<float>(x, delta, a, b[..3, ..], c, d, false));
        Assert.Throws<ArgumentException>(() => XMath.SelectiveScan<float>(x, delta, a, b, c, new Vector<float>(3), false));
        Assert.Throws<ArgumentException>(() => XMath.SelectiveScan<float>(x, delta, a, b, c, d, new Matrix<float>(8, 3), false, new Matrix<float>(8, 3)));

        // 出力先が入力と重なる
        Assert.Throws<ArgumentException>(() => XMath.SelectiveScan<float>(x, delta, a, b, c, d, new Matrix<float>(8, 4), false, x));
        Assert.Throws<ArgumentException>(() =>
            XMath.SelectiveScanStep<float>(x, delta, a, b, c, d, new Tensor<float>(8, 4, 2), false, new Matrix<float>(8, 3)));
    }

    /// <summary>
    /// 1位置ずつの走査も、まとめた走査と同じく、書き換える状態・出力先がどの入力とも、また互いに重なれば例外にする
    /// （1つの配列の中に置いた各引数の位置を1つずつずらして確かめる）
    /// </summary>
    [Fact]
    public void StepRejectsOverlappingStateAndDestination()
    {
        const int channels = 2, states = 3, sequences = 4;
        var buffer = new float[200];
        Array.Fill(buffer, 0.5f);
        const int state = 0, x = 100, delta = 110, a = 120, b = 130, c = 150, d = 170, y = 180; // 状態は [0, 24)

        void Step(int xAt = x, int deltaAt = delta, int aAt = a, int bAt = b, int cAt = c, int dAt = d, int stateAt = state, int yAt = y) =>
            XMath.SelectiveScanStep<float>(
                new ReadOnlyMatrixSpan<float>(buffer.AsSpan(xAt, channels * sequences), channels, sequences),
                new ReadOnlyMatrixSpan<float>(buffer.AsSpan(deltaAt, channels * sequences), channels, sequences),
                new ReadOnlyMatrixSpan<float>(buffer.AsSpan(aAt, channels * states), channels, states),
                new ReadOnlyMatrixSpan<float>(buffer.AsSpan(bAt, states * sequences), states, sequences),
                new ReadOnlyMatrixSpan<float>(buffer.AsSpan(cAt, states * sequences), states, sequences),
                new ReadOnlyVectorSpan<float>(buffer.AsSpan(dAt, channels)),
                new TensorSpan<float>(buffer.AsSpan(stateAt, channels * states * sequences), channels, states, sequences),
                false,
                new MatrixSpan<float>(buffer.AsSpan(yAt, channels * sequences), channels, sequences));

        Step();
        Step(yAt: 24);   // 状態のすぐ後ろ（重ならない）
        Step(aAt: 24);
        Assert.Throws<ArgumentException>(() => Step(xAt: 5));
        Assert.Throws<ArgumentException>(() => Step(deltaAt: 20));
        Assert.Throws<ArgumentException>(() => Step(aAt: 18));
        Assert.Throws<ArgumentException>(() => Step(bAt: 0));
        Assert.Throws<ArgumentException>(() => Step(cAt: 12));
        Assert.Throws<ArgumentException>(() => Step(dAt: 23));
        Assert.Throws<ArgumentException>(() => Step(yAt: 17));        // 出力先と状態
        Assert.Throws<ArgumentException>(() => Step(aAt: y + 7));     // A の先頭と出力先の最後の要素
        Assert.Throws<ArgumentException>(() => Step(aAt: y - 5));     // A の最後の要素と出力先の先頭

        // まとめた走査（状態は行列）も同じ組を拒む
        var xs = new Matrix<float>(channels, sequences);
        var h = new Matrix<float>(channels, states);
        var ab = new Matrix<float>(channels, states + sequences);
        var bc = new Matrix<float>(states, sequences);
        var none = new Vector<float>(0);
        Assert.Throws<ArgumentException>(() => XMath.SelectiveScan<float>(xs, xs, ab[.., ..states], bc, bc, none, h, false, ab[.., (states - 1)..^1]));
        Assert.Throws<ArgumentException>(() => XMath.SelectiveScan<float>(xs, xs, h, bc, bc, none, h, false, new Matrix<float>(channels, sequences)));
    }
}
