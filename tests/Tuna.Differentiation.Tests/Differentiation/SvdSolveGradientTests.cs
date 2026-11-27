using System.Runtime.InteropServices;
using Tuna.Differentiation;
using Tuna.Differentiation.Tests;
using Tuna.LinearAlgebra;
using static Tuna.Tests.Differentiation.LinearAlgebraTests;

namespace Tuna.Tests.Differentiation;

public class SvdSolveGradientTests
{
    [Theory]
    [InlineData(6, 3, false)]
    [InlineData(3, 6, false)]
    [InlineData(4, 4, true)]
    public void FullRankPullbacksMatchFiniteDifferences(int m, int n, bool repeated)
    {
        var random = new Random(482 + m);
        var a = repeated ? Matrix<double>.FromDiagonal([2, 2, 3, 3]) : RandomMatrix(random, m, n);
        if (!repeated)
        {
            for (int i = 0; i < Math.Min(m, n); i++)
            {
                a[i, i] += 3;
            }
        }

        var rhs = RandomMatrix(random, m, 2);
        var b = RandomVector(random, m);
        var g = RandomVector(random, n);
        var gx = RandomMatrix(random, n, 2);
        var gp = RandomMatrix(random, n, m);
        const double relative = 2e-7;

        var tape = new Tape();
        var node = SingularValueDecompositionNode.Compute(tape.Leaf(a));
        var expected = a.Svd();
        AssertSameBits(expected.Solve(b).AsSpan(), node.Solve(tape.Leaf(b)).Value.Data);
        AssertSameBits(expected.Solve(rhs, tolerance: 0.05).AsSpan(), node.Solve(tape.Leaf(rhs), tolerance: 0.05).Value.Data);
        AssertSameBits(expected.PseudoInverse().AsSpan(), node.PseudoInverse().Value.Data);

        AssertGradients([a], [b], (t, x, v) => XMath.Dot(t.Constant(g), SingularValueDecompositionNode.Compute(x[0]).Solve(v[0])), relative);
        AssertGradients([a, rhs], [], (t, x, _) => Pair(t, SingularValueDecompositionNode.Compute(x[0]).Solve(x[1], tolerance: 0.05), gx), relative);
        AssertGradients([a], [], (t, x, _) => Pair(t, SingularValueDecompositionNode.Compute(x[0]).PseudoInverse(), gp), relative);
    }

    [Fact]
    public void RankDeficientPullbacksMatchFiniteDifferencesAlongRankPreservingPaths()
    {
        const int m = 5, n = 6;
        // Varying either factor of L·R preserves rank two, including along finite-difference paths.
        var left = Matrix<double>.FromRowMajor(m, 2, [1, 0, 0, 1, 0.2, 0.3, 0, -0.4, 0.4, 0]);
        var right = Matrix<double>.FromRowMajor(2, n, [1, 0, 0.4, 0, 0.2, -0.5, 0, 1, 0, 0.5, 0, 0.3]);
        var random = new Random(477);
        var b = RandomVector(random, m);
        var rhs = RandomMatrix(random, m, 3);
        var g = RandomVector(random, n);
        var gx = RandomMatrix(random, n, 3);
        var gp = RandomMatrix(random, n, m);
        const double relative = 2e-7;
        const double tolerance = 0.1;

        AssertGradients([left, right], [b], (t, x, v) => XMath.Dot(t.Constant(g), SingularValueDecompositionNode.Compute(x[0] * x[1]).Solve(v[0])), relative);
        AssertGradients([left, right, rhs], [], (t, x, _) => Pair(t, SingularValueDecompositionNode.Compute(x[0] * x[1]).Solve(x[2], tolerance), gx), relative);
        AssertGradients([left, right], [], (t, x, _) => Pair(t, SingularValueDecompositionNode.Compute(x[0] * x[1]).PseudoInverse(tolerance), gp), relative);
    }

    [Fact]
    public void ExplicitCutoffKeepsTheForwardRankInAllPullbacks()
    {
        var a = Matrix<double>.FromDiagonal([4, 1]);
        var b = new Vector<double>([4, 2]);
        var g = new Vector<double>([2, 3]);
        var gp = Matrix<double>.FromDiagonal([5, 7]);
        var tape = new Tape();
        var coefficient = tape.Leaf(a);
        var rhs = tape.Leaf(b);
        var svd = SingularValueDecompositionNode.Compute(coefficient);
        var x = svd.Solve(rhs, tolerance: 0.5);
        var inverse = svd.PseudoInverse(tolerance: 0.5);
        AssertSameBits(a.Svd().Solve(b, tolerance: 0.5).AsSpan(), x.Value.Data);
        AssertSameBits(a.Svd().PseudoInverse(tolerance: 0.5).AsSpan(), inverse.Value.Data);
        tape.Backward(XMath.Dot(tape.Constant(g), x) + Pair(tape, inverse, gp));

        double Value(double d)
        {
            var current = Matrix<double>.FromDiagonal([d, 1]).Svd();
            return XMath.Dot(g, current.Solve(b, tolerance: 0.5)) + 5 * current.PseudoInverse(tolerance: 0.5)[0, 0];
        }

        var gradient = tape.GetGradient(coefficient);
        double numeric = DifferentiationAssert.Richardson(Value, 4);
        DifferentiationAssert.Close(numeric, gradient[0, 0], 1e-8, 0, "retained singular direction");
        Assert.Equal(0, gradient[1, 1]);
        Assert.Equal(new double[] { 0.5, 0 }, tape.GetGradient(rhs).ToArray());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AnEmptyRetainedSpectrumHasZeroPullbacks(bool truncate)
    {
        var a = new Matrix<double>(3, 2);
        if (truncate)
        {
            a[0, 0] = 2;
            a[1, 1] = 1;
        }

        double tolerance = truncate ? 1 : 0;
        var tape = new Tape();
        var coefficient = tape.Leaf(a);
        var b = tape.Leaf(new Vector<double>([1, 2, 3]));
        var rhs = tape.Leaf(Matrix<double>.FromColumnMajor(3, 2, [1, 2, 3, 4, 5, 6]));
        var svd = SingularValueDecompositionNode.Compute(coefficient);
        var loss = XMath.Sum(svd.Solve(b, tolerance)) + Sum(svd.Solve(rhs, tolerance)) + Sum(svd.PseudoInverse(tolerance));
        tape.Backward(loss);
        Assert.Equal(0, loss.Value);
        Assert.Equal(new double[6], tape.GetGradient(coefficient).ToMatrix().AsSpan().ToArray());
        Assert.Equal(new double[3], tape.GetGradient(b).ToArray());
        Assert.Equal(new double[6], tape.GetGradient(rhs).ToMatrix().AsSpan().ToArray());

        if (truncate)
        {
            AssertGradients([a], [new Vector<double>([1, 2, 3])], (t, x, v) =>
            {
                var decomposition = SingularValueDecompositionNode.Compute(x[0]);
                return XMath.Sum(decomposition.Solve(v[0], tolerance)) + Sum(decomposition.PseudoInverse(tolerance));
            }, 1e-8);
        }
    }

    [Fact]
    public void SubnormalCutoffRetainsBothDirectionsForARightHandSidePullback()
    {
        double tiny = double.Epsilon;
        var tape = new Tape();
        var a = tape.Constant(Matrix<double>.FromDiagonal([3 * tiny, 2 * tiny]));
        var b = tape.Leaf(new Vector<double>([3 * tiny, 2 * tiny]));
        var g = tape.Constant(new Vector<double>([3 * tiny, 2 * tiny]));
        var svd = SingularValueDecompositionNode.Compute(a);
        var x = svd.Solve(b, tolerance: 0.5);
        Assert.Equal(new double[] { 1, 1 }, x.Value.ToArray());
        tape.Backward(XMath.Dot(g, x));
        Assert.Equal(new double[] { 1, 1 }, tape.GetGradient(b).ToArray());
    }

    [Theory]
    [InlineData(-600, -900, 500)]
    [InlineData(600, 900, -500)]
    [InlineData(-600, 600, -900)]
    [InlineData(600, -600, 900)]
    public void SolvePullbacksPreserveFiniteGradientsAcrossExtremeScales(int aExponent, int bExponent, int gExponent)
    {
        var tape = new Tape();
        var a = tape.Leaf(Matrix<double>.FromDiagonal([Math.ScaleB(1, aExponent)]));
        var b = tape.Leaf(new Vector<double>([Math.ScaleB(1, bExponent)]));
        var g = tape.Constant(new Vector<double>([Math.ScaleB(1, gExponent)]));
        var svd = SingularValueDecompositionNode.Compute(a);
        tape.Backward(XMath.Dot(g, svd.Solve(b)));
        Assert.Equal(Math.ScaleB(-1, bExponent + gExponent - 2 * aExponent), tape.GetGradient(a)[0, 0]);
        Assert.Equal(Math.ScaleB(1, gExponent - aExponent), tape.GetGradient(b)[0]);
    }

    [Theory]
    [InlineData(-700, -600)]
    [InlineData(700, 600)]
    [InlineData(-1040, -1070)]
    public void PseudoInversePullbackAppliesBothInverseFactorsWithoutIntermediateOverflow(int aExponent, int gExponent)
    {
        var tape = new Tape();
        var a = tape.Leaf(Matrix<double>.FromDiagonal([Math.ScaleB(1, aExponent), 0]));
        var svd = SingularValueDecompositionNode.Compute(a);
        double weight = Math.ScaleB(1, gExponent);
        var g = Matrix<double>.FromRowMajor(2, 2, [weight, -weight, weight, weight]);
        tape.Backward(Pair(tape, svd.PseudoInverse(), g));
        double coefficient = Math.ScaleB(1, gExponent - 2 * aExponent);
        Assert.Equal(new double[,] { { -coefficient, coefficient }, { -coefficient, 0 } }, tape.GetGradient(a).ToArray());
    }

    [Fact]
    public void SolveScatterGradientsThroughTransposedViews()
    {
        var random = new Random(498);
        var a = RandomMatrix(random, 4, 5);
        var rhs = RandomMatrix(random, 3, 5);
        var g = RandomMatrix(random, 3, 2);
        AssertGradients([a, rhs], [], (t, x, _) => Pair(t, SingularValueDecompositionNode.Compute(x[0][1.., 1..].T).Solve(x[1][..2, 1..].T), g), 2e-7);
    }

    [Fact]
    public void FailedSvdOperationsLeaveTheTapeUsable()
    {
        var tape = new Tape();
        var a = tape.Leaf(Matrix<double>.FromDiagonal([2]));
        var b = tape.Leaf(new Vector<double>([3]));
        var svd = SingularValueDecompositionNode.Compute(a);
        Assert.Throws<ArgumentOutOfRangeException>("tolerance", () => svd.Solve(b, double.NaN));
        Assert.Throws<ArgumentOutOfRangeException>("tolerance", () => svd.PseudoInverse(-1));
        tape.Backward(XMath.Sum(svd.Solve(b)) + Sum(svd.PseudoInverse()));
        Assert.Equal(-1, tape.GetGradient(a)[0, 0]);
        Assert.Equal(0.5, tape.GetGradient(b)[0]);
    }

    private static ScalarNode Sum(MatrixNode<double> matrix) => XMath.Sum(XMath.ColumnSums(matrix));

    private static void AssertSameBits(ReadOnlySpan<double> expected, ReadOnlySpan<double> actual) =>
        Assert.True(MemoryMarshal.AsBytes(expected).SequenceEqual(MemoryMarshal.AsBytes(actual)));
}
