using System.Reflection;
using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.Tests.Differentiation;

/// <summary>テープの世代、逆伝播、入出力と領域の再利用</summary>
public class TapeTests
{
    [Fact]
    public void LeavesAndGradientRespectStridesAndCreationOrder()
    {
        var tape = new Tape();
        var first = tape.Leaf(7);
        double[] data = [2, -99, 3, -99, 5];
        var leaves = tape.Leaves(new ReadOnlyVectorSpan<double>(data, [3], [2]));
        Assert.Equal(new double[] { 2, 3, 5 }, leaves.ToArray().Select(x => x.Value));
        double[] result = [-99, -99, -99, -99, -99, -99, -99];
        var output = first + leaves[0] * leaves[1] + 4 * leaves[2];
        Assert.Equal(33, tape.Gradient(output, new VectorSpan<double>(result, [4], [2])));
        Assert.Equal(new double[] { 1, -99, 3, -99, 2, -99, 4 }, result);
    }

    [Fact]
    public void MultipleLeavesCallsKeepPreviouslyReturnedNodes()
    {
        var tape = new Tape();
        double[] a = [2, 3];
        double[] b = Enumerable.Range(4, 33).Select(i => (double)i).ToArray();
        var first = tape.Leaves(new ReadOnlyVectorSpan<double>(a));
        var second = tape.Leaves(new ReadOnlyVectorSpan<double>(b));
        var output = first[0] * second[1] + first[1] * second[0];
        double[] gradient = new double[a.Length + b.Length];
        Assert.Equal(22, tape.Gradient(output, new VectorSpan<double>(gradient)));
        Assert.Equal(new double[] { 5, 4, 3, 2 }, gradient.Take(4));
        Assert.All(gradient.Skip(4), value => Assert.Equal(0, value));
    }

    [Fact]
    public void LeavesCanBeInterleavedWithOperationsAndStorageCanGrow()
    {
        var tape = new Tape();
        var x = tape.Leaf(2);
        ScalarNode sum = x * x;
        for (int i = 0; i < 100; i++)
        {
            var next = tape.Leaf(i);
            sum += (i + 1) * next;
        }

        double[] gradient = new double[101];
        tape.Gradient(sum, new VectorSpan<double>(gradient));
        Assert.Equal(4, gradient[0]);
        for (int i = 0; i < 100; i++)
        {
            Assert.Equal(i + 1, gradient[i + 1]);
        }

        tape.Reset();
        x = tape.Leaf(7);
        tape.Backward(x);
        Assert.Equal(1, tape.GetGradient(x));
    }

    [Fact]
    public void LeavesCopyInputValuesBeforeTheyAreChanged()
    {
        var tape = new Tape();
        double[] values = [2, 3];
        var p = tape.Leaves(new ReadOnlyVectorSpan<double>(values));
        values[0] = 100;
        double[] gradient = new double[2];
        Assert.Equal(6, tape.Gradient(p[0] * p[1], new VectorSpan<double>(gradient)));
        Assert.Equal(new double[] { 3, 2 }, gradient);
    }

    [Fact]
    public void ConstantOutputOverwritesEveryGradientAndConsumesBackward()
    {
        var tape = new Tape();
        var x = tape.Leaf(3);
        double[] gradient = [double.NaN];
        Assert.Equal(7, tape.Gradient(7, new VectorSpan<double>(gradient)));
        Assert.Equal(0, gradient[0]);
        Assert.Equal(0, tape.GetGradient(x));
        Assert.Equal(0, tape.GetGradient(100));
        Assert.Throws<InvalidOperationException>(() => tape.Backward(7));
    }

    [Fact]
    public void EmptyTapeAndEmptyLeavesCanDifferentiateAConstant()
    {
        var tape = new Tape();
        Assert.True(tape.Leaves(new ReadOnlyVectorSpan<double>(ReadOnlySpan<double>.Empty)).IsEmpty);
        Assert.Equal(5, tape.Gradient(5, new VectorSpan<double>(Span<double>.Empty)));
        tape.Reset();
        var x = tape.Leaf(2);
        tape.Backward(x);
        Assert.Equal(1, tape.GetGradient(x));
    }

    [Fact]
    public void ResetInvalidatesNodesInEveryPublicUse()
    {
        var tape = new Tape();
        var old = tape.Leaf(3);
        tape.Reset();
        var current = tape.Leaf(4);
        Assert.Throws<InvalidOperationException>(() => old.Value);
        Assert.Throws<InvalidOperationException>(() => old.RequiresGradient);
        Assert.Throws<InvalidOperationException>(() => old + 1);
        Assert.Throws<InvalidOperationException>(() => 1 + old);
        Assert.Throws<InvalidOperationException>(() => old - 1);
        Assert.Throws<InvalidOperationException>(() => 1 - old);
        Assert.Throws<InvalidOperationException>(() => old * 1);
        Assert.Throws<InvalidOperationException>(() => 1 * old);
        Assert.Throws<InvalidOperationException>(() => old / 1);
        Assert.Throws<InvalidOperationException>(() => 1 / old);
        Assert.Throws<InvalidOperationException>(() => -old);
        Assert.Throws<InvalidOperationException>(() => old < current);
        Assert.Throws<InvalidOperationException>(() => ScalarNode.Record(1, [(old, 0)]));
        Assert.Throws<InvalidOperationException>(() => tape.Backward(old));
        tape.Backward(current);
        Assert.Throws<InvalidOperationException>(() => tape.GetGradient(old));
        Assert.Equal(1, tape.GetGradient(current));
    }

    [Theory]
    [InlineData(int.MaxValue)]
    [InlineData(-1)]
    public void ResetWrapsTheGenerationAndKeepsTheTapeUsable(int generation)
    {
        var tape = new Tape();
        // 数十億回の Reset を回さず、int.MaxValue と -1 の直後の世代を再現する。
        typeof(Tape).GetField("generation", BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(tape, generation);
        var old = tape.Leaf(3);
        tape.Reset();
        var current = tape.Leaf(4);
        Assert.Throws<InvalidOperationException>(() => old.Value);
        Assert.Throws<InvalidOperationException>(() => tape.Backward(old));
        tape.Backward(current * current);
        Assert.Equal(8, tape.GetGradient(current));
        tape.Reset();
        current = tape.Leaf(5);
        tape.Backward(current * current);
        Assert.Equal(10, tape.GetGradient(current));
    }

    [Fact]
    public void ResetLeavesBorrowedNodesInvalidUntilTheStorageIsReused()
    {
        var tape = new Tape();
        var borrowed = tape.Leaves(new ReadOnlyVectorSpan<double>(new double[] { 2, 3 }));
        tape.Reset();
        // span 自体はラムダに閉じ込めず、そこから読んだノードが失効していることを確かめる。
        ScalarNode old = borrowed[0];
        Assert.Throws<InvalidOperationException>(() => old.Value);
        Assert.Throws<InvalidOperationException>(() => old.RequiresGradient);
    }

    [Fact]
    public void IndependentTapesRejectMixedNodesWithoutDamagingEitherTape()
    {
        var a = new Tape();
        var b = new Tape();
        var x = a.Leaf(2);
        var y = b.Leaf(3);
        Assert.Throws<InvalidOperationException>(() => x + y);
        Assert.Throws<InvalidOperationException>(() => x - y);
        Assert.Throws<InvalidOperationException>(() => x * y);
        Assert.Throws<InvalidOperationException>(() => x / y);
        Assert.Throws<InvalidOperationException>(() => ScalarNode.Record(1, [(x, 1), (y, 0)]));
        Assert.Throws<InvalidOperationException>(() => a.Backward(y));
        a.Backward(x * x);
        b.Backward(y * y);
        Assert.Equal(4, a.GetGradient(x));
        Assert.Equal(6, b.GetGradient(y));
        Assert.Throws<InvalidOperationException>(() => a.GetGradient(y));
    }

    [Fact]
    public void BackwardIsSingleUseAndRecordingRequiresReset()
    {
        var tape = new Tape();
        var x = tape.Leaf(2);
        Assert.Throws<InvalidOperationException>(() => tape.GetGradient(x));
        tape.Backward(x * x);
        Assert.Throws<InvalidOperationException>(() => tape.Backward(x));
        Assert.Throws<InvalidOperationException>(() => tape.Leaf(1));
        Assert.Throws<InvalidOperationException>(() => tape.Leaves(new ReadOnlyVectorSpan<double>(new double[] { 1 })));
        Assert.Throws<InvalidOperationException>(() => x + 1);
        Assert.Throws<InvalidOperationException>(() => -x);
        Assert.Throws<InvalidOperationException>(() => x / 1);
        Assert.Throws<InvalidOperationException>(() => 1 / x);
        Assert.Throws<InvalidOperationException>(() => x * x);
        Assert.Throws<InvalidOperationException>(() => ScalarNode.Record(1, [(x, 0)]));
        Assert.Equal(4, tape.GetGradient(x));
        tape.Reset();
        var next = tape.Leaf(5);
        tape.Backward(next * next);
        Assert.Equal(10, tape.GetGradient(next));
    }

    [Fact]
    public void WrongGradientLengthDoesNotConsumeBackward()
    {
        var tape = new Tape();
        var x = tape.Leaf(2);
        var y = tape.Leaf(3);
        var output = x * y;
        Assert.Throws<ArgumentException>(() => tape.Gradient(output, new VectorSpan<double>(new double[1])));
        Assert.Throws<ArgumentException>(() => tape.Gradient(output, new VectorSpan<double>(new double[3])));
        double[] gradient = new double[2];
        tape.Gradient(output, new VectorSpan<double>(gradient));
        Assert.Equal(new double[] { 3, 2 }, gradient);
    }

    [Fact]
    public void RepeatingTheSameModelAfterResetAllocatesNoManagedBytes()
    {
        var tape = new Tape();
        double[] values = [2, 3, 4];
        double[] gradient = new double[3];
        for (int i = 0; i < 100; i++)
        {
            Evaluate(tape, values, gradient);
        }

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 100; i++)
        {
            Evaluate(tape, values, gradient);
        }

        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.Equal(0, allocated);
        Assert.Equal(new double[] { 3, 2, 3 }, gradient);
    }

    private static void Evaluate(Tape tape, double[] values, double[] gradient)
    {
        tape.Reset();
        var p = tape.Leaves(new ReadOnlyVectorSpan<double>(values));
        var product = p[0] * p[1];
        var custom = ScalarNode.Record(3 * p[2].Value, [(p[2], 3), ((ScalarNode)5, double.NaN)]);
        tape.Gradient(product + custom, new VectorSpan<double>(gradient));
    }
}
