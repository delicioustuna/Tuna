using System.Numerics;

namespace Tuna.NeuralNetwork.Tests;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む。
using Tuna.LinearAlgebra;

/// <summary>Mamba のブロックを倍精度の定義と比べ、状態の引き継ぎと1ステップ版がまとめて通したのと同じビットになることを確かめる</summary>
public class MambaBlockTests
{
    public static TheoryData<int, int, int, int, int, int, bool> Cases => new()
    {
        // model, inner, state, kernel, rank, length, projection biases
        { 1, 1, 1, 1, 1, 1, false },
        { 8, 16, 4, 4, 1, 9, false },
        { 12, 24, 16, 4, 1, 17, true },
        { 10, 30, 3, 2, 3, 5, false },  // 内側のチャネルが SIMD の幅の倍数でない
        { 64, 128, 16, 4, 4, 33, false }, // RE-USE に近い大きさ（Inner = 2·Model）
    };

    [Theory]
    [MemberData(nameof(Cases))]
    public void ForwardMatchesDefinition(int model, int inner, int state, int kernel, int rank, int length, bool biases)
    {
        var dimensions = new MambaDimensions(model, inner, state, kernel, rank);
        Forward<float>(dimensions, length, biases);
        Forward<double>(dimensions, length, biases);
    }

    private static void Forward<T>(MambaDimensions dimensions, int length, bool biases) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var weights = new MambaWeights(dimensions, 100, biases).RoundedTo<T>();
        var block = weights.Create<T>();
        var u = TestData.Rounded<T>(TestData.Uniform(dimensions.Model * length, 120));
        var expected = Reference.Mamba(weights, u, length);
        var input = TestData.Matrix<T>(u, dimensions.Model, length);
        TestData.Close<T>(expected, block.Forward(input, approximate: false), TestData.Tolerance<T>() * 10);
        TestData.Close<T>(expected, block.Forward(input, approximate: true), TestData.Tolerance<T>() * 20);

        // 出力先が入力と同じ（その場で x ← Mamba(x)）
        var inPlace = input.Clone();
        block.Forward(inPlace, null, false, inPlace);
        TestData.Same<T>(block.Forward(input, false), inPlace);
    }

    [Fact]
    public void ForwardContinuesFromState()
    {
        Continue<float>();
        Continue<double>();
    }

    /// <summary>系列を区切って状態を渡しながら通しても、まとめて通したのと同じビット（区切りが核より短い場合も）</summary>
    private static void Continue<T>() where T : unmanaged, IFloatingPointIeee754<T>
    {
        var dimensions = new MambaDimensions(16, 32, 8, 4, 2);
        var block = new MambaWeights(dimensions, 130).Create<T>();
        const int length = 20;
        var u = TestData.Matrix<T>(TestData.Uniform(dimensions.Model * length, 131), dimensions.Model, length);
        var whole = block.Forward(u, approximate: true);
        var state = block.CreateState(1);
        var parts = new Matrix<T>(dimensions.Model, length);
        foreach (var (start, end) in new[] { (0, 1), (1, 3), (3, 10), (10, 11), (11, 20) })
        {
            block.Forward(u[.., start..end], state, true, parts[.., start..end]);
        }

        TestData.Same<T>(whole, parts);

        // 状態を戻すと最初から
        state.Reset();
        block.Forward(u[.., 0..5], state, true, parts[.., 0..5]);
        TestData.Same<T>(whole[.., 0..5], parts[.., 0..5]);
    }

    [Fact]
    public void StepMatchesForwardForEachSequence()
    {
        Step<float>();
        Step<double>();
    }

    /// <summary>S 本の系列を1位置ずつ進めると、各系列をまとめて通したのと同じビット</summary>
    private static void Step<T>() where T : unmanaged, IFloatingPointIeee754<T>
    {
        var dimensions = new MambaDimensions(12, 40, 6, 4, 2);
        var block = new MambaWeights(dimensions, 140, projectionBiases: true).Create<T>();
        const int length = 7, sequences = 5;
        var inputs = Enumerable.Range(0, sequences)
            .Select(s => TestData.Matrix<T>(TestData.Uniform(dimensions.Model * length, 150 + s), dimensions.Model, length)).ToArray();
        var expected = inputs.Select(u => block.Forward(u, approximate: true)).ToArray();
        var state = block.CreateState(sequences);
        var x = new Matrix<T>(dimensions.Model, sequences);
        var y = new Matrix<T>(dimensions.Model, sequences);
        for (int t = 0; t < length; t++)
        {
            for (int s = 0; s < sequences; s++)
            {
                inputs[s].Column(t).CopyTo(x.Column(s));
            }

            block.Step(x, state, true, y);
            for (int s = 0; s < sequences; s++)
            {
                TestData.Same<T>(expected[s][.., t..(t + 1)], y[.., s..(s + 1)]);
            }
        }

        // 1本の状態なら Step と Forward を混ぜて使える
        var single = block.CreateState(1);
        block.Forward(inputs[0][.., ..3], single, true, new Matrix<T>(dimensions.Model, 3));
        var next = block.Step(inputs[0][.., 3..4], single, true);
        TestData.Same<T>(expected[0][.., 3..4], next);
        var rest = new Matrix<T>(dimensions.Model, length - 4);
        block.Forward(inputs[0][.., 4..], single, true, rest);
        TestData.Same<T>(expected[0][.., 4..], rest);
    }

    [Fact]
    public void DefaultDimensionsFollowMambaSsm()
    {
        Assert.Equal(new MambaDimensions(64, 128, 16, 4, 4), MambaDimensions.Default(64));
        Assert.Equal(new MambaDimensions(768, 1536, 16, 4, 48), MambaDimensions.Default(768));
        Assert.Equal(new MambaDimensions(20, 40, 16, 4, 2), MambaDimensions.Default(20));
    }

    /// <summary>
    /// 出力先は入力と同じか、重ならないこと。一部だけ重なると、並列に進む区間が別の区間の読む前の入力を書き換えるので、
    /// まとめて通す版と1位置ずつの版のどちらも例外にする。同じなら、別の出力先に書いたのと同じビット
    /// </summary>
    [Fact]
    public void DestinationMustBeTheInputOrDisjoint()
    {
        var dimensions = new MambaDimensions(8, 16, 4, 4, 1);
        var block = new MambaWeights(dimensions, 170).Create<float>();
        const int columns = 6;
        var u = TestData.Matrix<float>(TestData.Uniform(dimensions.Model * columns, 171), dimensions.Model, columns);
        var shared = new Matrix<float>(dimensions.Model, 2 * columns);
        u.AsMatrixSpan().CopyTo(shared[.., ..columns]);

        Assert.Throws<ArgumentException>(() => block.Forward(shared[.., ..columns], null, false, shared[.., 2..(columns + 2)]));
        Assert.Throws<ArgumentException>(() => block.Step(shared[.., ..columns], block.CreateState(columns), false, shared[.., 1..(columns + 1)]));
        var top = new Matrix<float>(2 * dimensions.Model, columns);
        u.AsMatrixSpan().CopyTo(top[..dimensions.Model, ..]);
        Assert.Throws<ArgumentException>(() => block.Step(top[..dimensions.Model, ..], block.CreateState(columns), false, top[4..(dimensions.Model + 4), ..]));

        // 重ならない（同じ配列の隣の列・行）
        var forward = block.Forward(u, false);
        block.Forward(shared[.., ..columns], null, false, shared[.., columns..]);
        TestData.Same<float>(forward, shared[.., columns..]);
        var step = block.Step(u, block.CreateState(columns), false);
        block.Step(top[..dimensions.Model, ..], block.CreateState(columns), false, top[dimensions.Model.., ..]);
        TestData.Same<float>(step, top[dimensions.Model.., ..]);

        // 入力と同じ
        var same = u.Clone();
        block.Forward(same, null, false, same);
        TestData.Same<float>(forward, same);
        same = u.Clone();
        block.Step(same, block.CreateState(columns), false, same);
        TestData.Same<float>(step, same);
    }

    [Fact]
    public void InvalidArgumentsThrow()
    {
        var dimensions = new MambaDimensions(4, 8, 2, 3, 1);
        var weights = new MambaWeights(dimensions, 160);
        var block = weights.Create<float>();
        Assert.Throws<ArgumentException>(() => block.Forward(new Matrix<float>(3, 2), false));
        Assert.Throws<ArgumentException>(() => block.Forward(new Matrix<float>(4, 2), null, false, new Matrix<float>(4, 3)));
        Assert.Throws<ArgumentException>(() => block.Forward(new Matrix<float>(4, 2), block.CreateState(2), false, new Matrix<float>(4, 2)));
        Assert.Throws<ArgumentException>(() => block.Step(new Matrix<float>(4, 3), block.CreateState(2), false));
        var other = new MambaWeights(new MambaDimensions(4, 8, 2, 4, 1), 161).Create<float>();
        Assert.Throws<ArgumentException>(() => block.Step(new Matrix<float>(4, 2), other.CreateState(2), false));
        Assert.Throws<ArgumentException>(() => new MambaBlock<float>(dimensions, new float[3], [], [], [], [], [], [], [], []));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new MambaBlock<float>(new MambaDimensions(0, 8, 2, 3, 1), [], [], [], [], [], [], [], [], []));
    }
}
