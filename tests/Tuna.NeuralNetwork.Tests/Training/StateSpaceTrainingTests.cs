using System.Numerics;
using Tuna.Differentiation;

namespace Tuna.NeuralNetwork.Tests.Training;

using Tuna.LinearAlgebra;

[Collection(nameof(ParallelismCollection))]
public class StateSpaceTrainingTests
{
    [Fact]
    public void DoubleScanMatchesInferenceAndNumPy() => CheckScan<double>(approximate: false);

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void FloatScanMatchesInferenceAndNumPy(bool approximate) => CheckScan<float>(approximate);

    [Fact]
    public void DepthwiseConvolutionMatchesInferenceAndNumPy()
    {
        var x = MatrixFrom<float>(2, 5, StateSpaceReferences.DepthwiseInput);
        var kernel = MatrixFrom<float>(2, 3, StateSpaceReferences.Kernel);
        var bias = new Vector<float>(Convert<float>(StateSpaceReferences.Bias));
        var tape = new Tape();
        var nx = tape.Leaf(x);
        var nk = tape.Leaf(kernel);
        var nb = tape.Leaf(bias);
        var y = XMath.CausalDepthwiseConvolve(nx, nk, nb);
        Assert.Equal(Convert<float>(StateSpaceReferences.DepthwiseOutput), y.Value.Data.ToArray());
        Assert.True(XMath.CausalDepthwiseConvolve(x, kernel, bias).AsSpan().SequenceEqual(y.Value.Data));
        var seed = MatrixFrom<float>(2, 5, StateSpaceReferences.DepthwiseSeed);
        tape.Backward(WeightedSum(tape, y, seed));
        Assert.Equal(Convert<float>(StateSpaceReferences.DepthwiseInputGradient), tape.GetGradient(nx).Data.ToArray());
        Assert.Equal(Convert<float>(StateSpaceReferences.KernelGradient), tape.GetGradient(nk).Data.ToArray());
        Assert.Equal(Convert<float>(StateSpaceReferences.BiasGradient), tape.GetGradient(nb).Data.ToArray());
    }

    [Fact]
    public void AScanWithNoStatesStillDifferentiatesItsSkipConnection()
    {
        var tape = new Tape();
        var x = tape.Leaf(MatrixFrom<double>(1, 3, [1, 2, 3]));
        var delta = tape.Constant(new Matrix<double>(1, 3));
        var a = tape.Constant(new Matrix<double>(1, 0));
        var projection = tape.Constant(new Matrix<double>(0, 3));
        var d = tape.Leaf(new Vector<double>([2]));
        var y = XMath.SelectiveScan(x, delta, a, projection, projection, d);
        Assert.Equal(new double[] { 2, 4, 6 }, y.Value.Data.ToArray());
        tape.Backward(XMath.Sum(XMath.ColumnSums(y)));
        Assert.Equal(new double[] { 2, 2, 2 }, tape.GetGradient(x).Data.ToArray());
        Assert.Equal(6d, tape.GetGradient(d)[0]);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    public void CheckpointedTrainingDoesNotAllocateAfterWarmup(int degree)
    {
        int saved = LinearAlgebraSettings.MaxDegreeOfParallelism;
        try
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = degree;
            var tape = new Tape();
            var x = new Matrix<float>(72, 128);
            var delta = new Matrix<float>(72, 128);
            var a = new Matrix<float>(72, 8);
            var projection = new Matrix<float>(8, 128);
            var kernel = new Matrix<float>(72, 4);
            x.AsSpan().Fill(0.125f);
            delta.AsSpan().Fill(0.25f);
            a.AsSpan().Fill(-0.5f);
            projection.AsSpan().Fill(0.25f);
            kernel.AsSpan().Fill(-0.125f);
            for (int i = 0; i < 5; i++)
            {
                Step();
            }

            long bytes = -1;
            for (int window = 0; window < 3 && bytes != 0; window++)
            {
                GC.Collect(0, GCCollectionMode.Forced, blocking: true);
                long before = GC.GetAllocatedBytesForCurrentThread();
                for (int i = 0; i < 4; i++)
                {
                    Step();
                }

                bytes = GC.GetAllocatedBytesForCurrentThread() - before;
            }

            Assert.Equal(0, bytes);

            void Step()
            {
                tape.Reset();
                var features = XMath.CausalDepthwiseConvolve(tape.Leaf(x), tape.Leaf(kernel));
                var p = tape.Leaf(projection);
                var y = XMath.SelectiveScan(features, tape.Leaf(delta), tape.Leaf(a), p, p);
                tape.Backward(XMath.Sum(XMath.ColumnSums(y)));
            }
        }
        finally
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = saved;
        }
    }

    [Theory]
    [InlineData(65)]
    [InlineData(96)]
    [InlineData(129)]
    public void FullHistoryAndCheckpointedScanProduceIdenticalBitsOnPrefix(int longLength)
    {
        // 系列長 64（全状態保存: interval=1）と長系列（チェックポイント方式: interval=32）で先頭 64 位置の値と勾配がビット一致する
        const int channels = 6;
        const int states = 4;
        const int prefix = 64;

        var xLong = MakeMatrix<double>(channels, longLength, 1);
        var dtLong = MakePositiveMatrix<double>(channels, longLength, 2);
        var a = MakeNegativeMatrix<double>(channels, states, 3);
        var bLong = MakeMatrix<double>(states, longLength, 4);
        var cLong = MakeMatrix<double>(states, longLength, 5);
        var d = MakeVector<double>(channels, 6);
        var seedLong = new Matrix<double>(channels, longLength);
        var seedShort = MakeMatrix<double>(channels, prefix, 7);
        seedShort.AsMatrixSpan().CopyTo(seedLong[.., ..prefix]);

        // 全状態保存（length = 64）
        var shortTape = new Tape();
        var sx = shortTape.Leaf(xLong[.., ..prefix].ToMatrix());
        var sdt = shortTape.Leaf(dtLong[.., ..prefix].ToMatrix());
        var sa = shortTape.Leaf(a);
        var sb = shortTape.Leaf(bLong[.., ..prefix].ToMatrix());
        var sc = shortTape.Leaf(cLong[.., ..prefix].ToMatrix());
        var sd = shortTape.Leaf(d);
        var sy = XMath.SelectiveScan(sx, sdt, sa, sb, sc, sd);
        shortTape.Backward(WeightedSum(shortTape, sy, seedShort));

        // チェックポイント方式（length = longLength > 64）
        var longTape = new Tape();
        var lx = longTape.Leaf(xLong);
        var ldt = longTape.Leaf(dtLong);
        var la = longTape.Leaf(a);
        var lb = longTape.Leaf(bLong);
        var lc = longTape.Leaf(cLong);
        var ld = longTape.Leaf(d);
        var ly = XMath.SelectiveScan(lx, ldt, la, lb, lc, ld);
        longTape.Backward(WeightedSum(longTape, ly, seedLong));

        Assert.Equal(sy.Value.ToMatrix().AsSpan().ToArray(), ly.Value[.., ..prefix].ToMatrix().AsSpan().ToArray());
        Assert.Equal(shortTape.GetGradient(sx).ToMatrix().AsSpan().ToArray(), longTape.GetGradient(lx)[.., ..prefix].ToMatrix().AsSpan().ToArray());
        Assert.Equal(shortTape.GetGradient(sdt).ToMatrix().AsSpan().ToArray(), longTape.GetGradient(ldt)[.., ..prefix].ToMatrix().AsSpan().ToArray());
        Assert.Equal(shortTape.GetGradient(sa).ToMatrix().AsSpan().ToArray(), longTape.GetGradient(la).ToMatrix().AsSpan().ToArray());
        Assert.Equal(shortTape.GetGradient(sb).ToMatrix().AsSpan().ToArray(), longTape.GetGradient(lb)[.., ..prefix].ToMatrix().AsSpan().ToArray());
        Assert.Equal(shortTape.GetGradient(sc).ToMatrix().AsSpan().ToArray(), longTape.GetGradient(lc)[.., ..prefix].ToMatrix().AsSpan().ToArray());
        Assert.Equal(shortTape.GetGradient(sd).ToArray(), longTape.GetGradient(ld).ToArray());
    }

    [Fact]
    public void DepthwiseAndScanHandleEdgeCasesConstantsAliasingAndAccumulation()
    {
        // CausalDepthwiseConvolve: double、bias 省略、taps > length、taps = 1、空入力、転置ビュー
        {
            var x = MakeMatrix<double>(3, 2, 11);
            var kernel = MakeMatrix<double>(3, 5, 12);
            var tape = new Tape();
            var nx = tape.Leaf(x.T.ToMatrix()).T;
            var nk = tape.Leaf(kernel.T.ToMatrix()).T;
            var y = XMath.CausalDepthwiseConvolve(nx, nk);
            Assert.True(XMath.CausalDepthwiseConvolve(x, kernel).AsSpan().SequenceEqual(y.Value.Data));
            var seed = MakeMatrix<double>(3, 2, 13);
            tape.Backward(WeightedSum(tape, y, seed));

            // 中心差分でカーネル・入力勾配を検証
            double h = 1e-6;
            double orig = kernel[1, 4];
            kernel[1, 4] = orig + h;
            double plus = double.CreateChecked(XMath.Sum(XMath.ColumnSums(XMath.Hadamard(XMath.CausalDepthwiseConvolve(x, kernel), seed))));
            kernel[1, 4] = orig - h;
            double minus = double.CreateChecked(XMath.Sum(XMath.ColumnSums(XMath.Hadamard(XMath.CausalDepthwiseConvolve(x, kernel), seed))));
            kernel[1, 4] = orig;
            Assert.Equal((plus - minus) / (2 * h), tape.GetGradient(nk)[1, 4], 7);
        }

        // 空の系列長・0 チャネル
        {
            var tape = new Tape();
            var emptyLen = tape.Leaf(new Matrix<double>(2, 0));
            var k1 = tape.Leaf(MakeMatrix<double>(2, 1, 14));
            var yConv = XMath.CausalDepthwiseConvolve(emptyLen, k1);
            var yScan = XMath.SelectiveScan(emptyLen, emptyLen, tape.Leaf(new Matrix<double>(2, 3)),
                tape.Leaf(new Matrix<double>(3, 0)), tape.Leaf(new Matrix<double>(3, 0)));
            tape.Backward(XMath.Sum(XMath.ColumnSums(yConv)) + XMath.Sum(XMath.ColumnSums(yScan)));
            Assert.Equal(0, tape.GetGradient(emptyLen).Columns);
            Assert.All(tape.GetGradient(k1).ToMatrix().AsSpan().ToArray(), v => Assert.Equal(0.0, v));

            tape.Reset();
            var zeroCh = tape.Leaf(new Matrix<double>(0, 4));
            var zeroA = tape.Leaf(new Matrix<double>(0, 2));
            var zeroB = tape.Leaf(MakeMatrix<double>(2, 4, 15));
            var yZero = XMath.SelectiveScan(zeroCh, zeroCh, zeroA, zeroB, zeroB);
            tape.Backward(XMath.Sum(XMath.ColumnSums(yZero)));
            Assert.Equal(0, tape.GetGradient(zeroCh).Rows);
            Assert.All(tape.GetGradient(zeroB).ToMatrix().AsSpan().ToArray(), v => Assert.Equal(0.0, v));
        }

        // SelectiveScan: 全定数、d のみ要勾配（履歴保存スキップ経路）、d 省略、b == c エイリアシング、既存勾配への蓄積
        {
            var x = MakeMatrix<double>(3, 8, 21);
            var dt = MakePositiveMatrix<double>(3, 8, 22);
            var a = MakeNegativeMatrix<double>(3, 2, 23);
            var bc = MakeMatrix<double>(2, 8, 24);
            var d = MakeVector<double>(3, 25);
            var seed = MakeMatrix<double>(3, 8, 26);

            // 全定数
            var tape = new Tape();
            var constY = XMath.SelectiveScan(tape.Constant(x), tape.Constant(dt), tape.Constant(a), tape.Constant(bc), tape.Constant(bc), tape.Constant(d));
            Assert.False(constY.RequiresGradient);

            // d のみ要勾配（NeedsStateGradient == false）
            var dOnly = tape.Leaf(d);
            var yDOnly = XMath.SelectiveScan(tape.Constant(x), tape.Constant(dt), tape.Constant(a), tape.Constant(bc), tape.Constant(bc), dOnly);
            tape.Backward(WeightedSum(tape, yDOnly, seed));
            var expectedDd = XMath.RowSums(XMath.Hadamard(x, seed));
            for (int i = 0; i < 3; i++)
            {
                Assert.Equal(expectedDd[i], tape.GetGradient(dOnly)[i], 13);
            }

            // d 省略 + b == c エイリアシング + Parameter 既存勾配への蓄積
            tape.Reset();
            var pBc = Parameter<double>.Matrix(2, 8);
            bc.AsSpan().CopyTo(pBc.Value);
            pBc.Gradient.Fill(1.25);
            var nodeBc = tape.UseMatrix(pBc);
            var yAlias = XMath.SelectiveScan(tape.Constant(x), tape.Constant(dt), tape.Constant(a), nodeBc, nodeBc);
            tape.Backward(WeightedSum(tape, yAlias, seed));

            var refTape = new Tape();
            var refB = refTape.Leaf(bc);
            var refC = refTape.Leaf(bc);
            var refY = XMath.SelectiveScan(refTape.Constant(x), refTape.Constant(dt), refTape.Constant(a), refB, refC);
            refTape.Backward(WeightedSum(refTape, refY, seed));
            var expB = refTape.GetGradient(refB).ToMatrix().AsSpan();
            var expC = refTape.GetGradient(refC).ToMatrix().AsSpan();
            for (int i = 0; i < pBc.Gradient.Length; i++)
            {
                Assert.Equal(1.25 + expB[i] + expC[i], pBc.Gradient[i], 13);
            }
        }
    }

    [Theory]
    [InlineData(64)]
    [InlineData(128)]
    public void StateSpaceValuesAndGradientsDoNotDependOnDegreeOfParallelism(int length)
    {
        int saved = LinearAlgebraSettings.MaxDegreeOfParallelism;
        try
        {
            var x = MakeMatrix<float>(32, length, 31);
            var kernel = MakeMatrix<float>(32, 4, 32);
            var dt = MakePositiveMatrix<float>(32, length, 33);
            var a = MakeNegativeMatrix<float>(32, 8, 34);
            var b = MakeMatrix<float>(8, length, 35);
            var c = MakeMatrix<float>(8, length, 36);
            var d = MakeVector<float>(32, 37);

            (float[] y, float[] dx, float[] dk, float[] da, float[] dd) Run(int degree)
            {
                LinearAlgebraSettings.MaxDegreeOfParallelism = degree;
                var tape = new Tape();
                var nx = tape.Leaf(x);
                var nk = tape.Leaf(kernel);
                var na = tape.Leaf(a);
                var nd = tape.Leaf(d);
                var conv = XMath.CausalDepthwiseConvolve(nx, nk);
                var y = XMath.SelectiveScan(conv, tape.Constant(dt), na, tape.Constant(b), tape.Constant(c), nd);
                tape.Backward(XMath.Sum(XMath.ColumnSums(y)));
                return (y.Value.Data.ToArray(), tape.GetGradient(nx).Data.ToArray(), tape.GetGradient(nk).Data.ToArray(),
                    tape.GetGradient(na).Data.ToArray(), tape.GetGradient(nd).ToArray());
            }

            var single = Run(1);
            var parallel = Run(4);
            Assert.Equal(single.y, parallel.y);
            Assert.Equal(single.dx, parallel.dx);
            Assert.Equal(single.dk, parallel.dk);
            Assert.Equal(single.da, parallel.da);
            Assert.Equal(single.dd, parallel.dd);
        }
        finally
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = saved;
        }
    }

    [Fact]
    public void InvalidStateSpaceArgumentsAndTapeErrorsAreRejected()
    {
        var tape = new Tape();
        var other = new Tape();
        var x = tape.Leaf(new Matrix<double>(2, 5));
        var kernel = tape.Leaf(new Matrix<double>(2, 3));
        var dt = tape.Leaf(new Matrix<double>(2, 5));
        var a = tape.Leaf(new Matrix<double>(2, 4));
        var b = tape.Leaf(new Matrix<double>(4, 5));
        var c = tape.Leaf(new Matrix<double>(4, 5));

        // 未初期化ノード
        Assert.Throws<InvalidOperationException>(() => XMath.CausalDepthwiseConvolve(default, kernel));
        Assert.Throws<InvalidOperationException>(() => XMath.CausalDepthwiseConvolve(x, default));
        Assert.Throws<InvalidOperationException>(() => XMath.SelectiveScan(default, dt, a, b, c));
        Assert.Throws<InvalidOperationException>(() => XMath.SelectiveScan(x, dt, default, b, c));

        // 形状・タップ異常（AllocateSaved より前に ArgumentException / ArgumentOutOfRangeException）
        Assert.Throws<ArgumentOutOfRangeException>(() => XMath.CausalDepthwiseConvolve(x, tape.Leaf(new Matrix<double>(2, 0))));
        Assert.Throws<ArgumentException>(() => XMath.CausalDepthwiseConvolve(x, tape.Leaf(new Matrix<double>(3, 3))));
        Assert.Throws<ArgumentException>(() => XMath.CausalDepthwiseConvolve(x, kernel, tape.Leaf(new Vector<double>(3))));
        Assert.Throws<ArgumentException>(() => XMath.SelectiveScan(x, tape.Leaf(new Matrix<double>(3, 5)), a, b, c));
        Assert.Throws<ArgumentException>(() => XMath.SelectiveScan(x, dt, tape.Leaf(new Matrix<double>(3, 4)), b, c));
        Assert.Throws<ArgumentException>(() => XMath.SelectiveScan(x, dt, a, tape.Leaf(new Matrix<double>(3, 5)), c));
        Assert.Throws<ArgumentException>(() => XMath.SelectiveScan(x, dt, a, b, tape.Leaf(new Matrix<double>(4, 4))));
        Assert.Throws<ArgumentException>(() => XMath.SelectiveScan(x, dt, a, b, c, tape.Leaf(new Vector<double>(3))));

        // 非対応要素型
        Assert.Throws<NotSupportedException>(() => XMath.CausalDepthwiseConvolve(default(MatrixNode<Half>), default));
        Assert.Throws<NotSupportedException>(() => XMath.SelectiveScan(default(MatrixNode<Half>), default, default, default, default));

        // 別テープ混在
        Assert.Throws<InvalidOperationException>(() => XMath.CausalDepthwiseConvolve(x, other.Leaf(new Matrix<double>(2, 3))));
        Assert.Throws<InvalidOperationException>(() => XMath.SelectiveScan(x, dt, other.Leaf(new Matrix<double>(2, 4)), b, c));

        // 例外後も同じテープで正常に記録・逆伝播できる
        var y = XMath.SelectiveScan(XMath.CausalDepthwiseConvolve(x, kernel), dt, a, b, c);
        tape.Backward(XMath.Sum(XMath.ColumnSums(y)));
        Assert.Equal((2, 5), (y.Rows, y.Columns));

        // 逆伝播完了後の記録拒否と Reset 後の失効ノード
        Assert.Throws<InvalidOperationException>(() => XMath.CausalDepthwiseConvolve(x, kernel));
        tape.Reset();
        Assert.Throws<InvalidOperationException>(() => XMath.SelectiveScan(x, dt, a, b, c));
    }

    private static Matrix<T> MakeMatrix<T>(int rows, int columns, int seed) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var m = new Matrix<T>(rows, columns);
        var span = m.AsSpan();
        for (int i = 0; i < span.Length; i++)
        {
            span[i] = T.CreateTruncating(((i * 7 + seed * 13) % 19 - 9) * 0.0625);
        }

        return m;
    }

    private static Matrix<T> MakePositiveMatrix<T>(int rows, int columns, int seed) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var m = new Matrix<T>(rows, columns);
        var span = m.AsSpan();
        for (int i = 0; i < span.Length; i++)
        {
            span[i] = T.CreateTruncating(0.05 + ((i * 5 + seed * 11) % 11) * 0.02);
        }

        return m;
    }

    private static Matrix<T> MakeNegativeMatrix<T>(int rows, int columns, int seed) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var m = new Matrix<T>(rows, columns);
        var span = m.AsSpan();
        for (int i = 0; i < span.Length; i++)
        {
            span[i] = T.CreateTruncating(-0.1 - ((i * 3 + seed * 7) % 9) * 0.05);
        }

        return m;
    }

    private static Vector<T> MakeVector<T>(int length, int seed) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var v = new Vector<T>(length);
        for (int i = 0; i < length; i++)
        {
            v[i] = T.CreateTruncating(((i * 5 + seed * 3) % 13 - 6) * 0.125);
        }

        return v;
    }

    private static void CheckScan<T>(bool approximate) where T : unmanaged, IFloatingPointIeee754<T>
    {
        const int length = 69;
        var x = MatrixFrom<T>(2, length, StateSpaceReferences.ScanInput);
        var delta = MatrixFrom<T>(2, length, StateSpaceReferences.Delta);
        var a = MatrixFrom<T>(2, 3, StateSpaceReferences.A);
        var b = MatrixFrom<T>(3, length, StateSpaceReferences.B);
        var c = MatrixFrom<T>(3, length, StateSpaceReferences.C);
        var d = new Vector<T>(Convert<T>(StateSpaceReferences.Skip));
        var seed = MatrixFrom<T>(2, length, StateSpaceReferences.ScanSeed);
        var tape = new Tape();
        var nx = tape.Leaf(x);
        var nd = tape.Leaf(delta);
        var na = tape.Leaf(a);
        var nb = tape.Leaf(b);
        var nc = tape.Leaf(c);
        var ns = tape.Leaf(d);
        var y = XMath.SelectiveScan(nx, nd, na, nb, nc, ns, approximate);
        Assert.True(XMath.SelectiveScan(x, delta, a, b, c, d, approximate).AsSpan().SequenceEqual(y.Value.Data));
        double tolerance = typeof(T) == typeof(float) ? 6e-6 : 2e-14;
        Check<T>(StateSpaceReferences.ScanOutput, y.Value.Data, tolerance);
        tape.Backward(WeightedSum(tape, y, seed));
        Check<T>(StateSpaceReferences.ScanInputGradient, tape.GetGradient(nx).Data, tolerance);
        Check<T>(StateSpaceReferences.DeltaGradient, tape.GetGradient(nd).Data, tolerance);
        Check<T>(StateSpaceReferences.AGradient, tape.GetGradient(na).Data, tolerance);
        Check<T>(StateSpaceReferences.BGradient, tape.GetGradient(nb).Data, tolerance);
        Check<T>(StateSpaceReferences.CGradient, tape.GetGradient(nc).Data, tolerance);
        Check<T>(StateSpaceReferences.SkipGradient, tape.GetGradient(ns).Data, tolerance);

        if (typeof(T) == typeof(double))
        {
            T original = a[0, 0], h = T.CreateChecked(1e-5);
            a[0, 0] = original + h;
            double plus = RawLoss();
            a[0, 0] = original - h;
            double minus = RawLoss();
            Assert.Equal(StateSpaceReferences.AGradient[0], (plus - minus) / 2e-5, 8);
        }

        double RawLoss()
        {
            var raw = XMath.SelectiveScan(x, delta, a, b, c, d, approximate);
            return double.CreateChecked(XMath.Sum(XMath.ColumnSums(XMath.Hadamard(raw, seed))));
        }
    }

    private static void Check<T>(double[] expected, ReadOnlySpan<T> actual, double tolerance)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            double error = Math.Abs(double.CreateChecked(actual[i]) - expected[i]);
            Assert.InRange(error, 0, tolerance * Math.Max(1, Math.Abs(expected[i])));
        }
    }

    private static ScalarNode WeightedSum<T>(Tape tape, MatrixNode<T> y, Matrix<T> seed)
        where T : unmanaged, IFloatingPointIeee754<T>
        => XMath.Sum(XMath.ColumnSums(XMath.Hadamard(y, tape.Constant(seed))));

    private static Matrix<T> MatrixFrom<T>(int rows, int columns, double[] values)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        var matrix = new Matrix<T>(rows, columns);
        Convert<T>(values).CopyTo(matrix.AsSpan());
        return matrix;
    }

    private static T[] Convert<T>(double[] values) where T : unmanaged, IFloatingPointIeee754<T> =>
        Array.ConvertAll(values, value => T.CreateChecked(value));
}
