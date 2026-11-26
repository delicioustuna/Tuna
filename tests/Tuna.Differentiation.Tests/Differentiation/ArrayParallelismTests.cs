using System.Numerics;
using System.Runtime.CompilerServices;
using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.Tests.Differentiation;

/// <summary>
/// <see cref="LinearAlgebraSettings.MaxDegreeOfParallelism"/> を書き換えるテストをまとめ、他のテストと同時に走らせない。
/// </summary>
[CollectionDefinition(nameof(ParallelismCollection), DisableParallelization = true)]
public class ParallelismCollection;

/// <summary>配列の記録の、並列数によらない値と勾配と、2回目以降の確保 0</summary>
[Collection(nameof(ParallelismCollection))]
public class ArrayParallelismTests
{
    [Fact]
    public void ValuesAndGradientsDoNotDependOnTheDegreeOfParallelism()
    {
        // 並列の経路に入る大きさ（行列積 300 次、行列ベクトル積 1000 次、内積・ln Σ exp 20 万要素）を含める。
        // 検査する並列数は Environment.ProcessorCount で頭打ちになる（DOTNET_PROCESSOR_COUNT で指定できる）。
        int saved = LinearAlgebraSettings.MaxDegreeOfParallelism;
        try
        {
            string? expectedDouble = null, expectedSingle = null;
            for (int degree = 1; degree <= 4; degree++)
            {
                LinearAlgebraSettings.MaxDegreeOfParallelism = degree;
                string actualDouble = Fingerprint(new Model<double>(300, 1000, 200_000)) + Fingerprint(new ElementwiseModel<double>(20, 200_000, 300))
                    + Fingerprint(new BroadcastModel<double>(64, 20_000)) + Fingerprint(new ReductionModel<double>(64, 20_000, 200_000))
                    + Fingerprint(new LossModel<double>(64, 20_000, 200_000));
                string actualSingle = Fingerprint(new Model<float>(300, 1000, 200_000)) + Fingerprint(new ElementwiseModel<float>(20, 200_000, 300))
                    + Fingerprint(new BroadcastModel<float>(64, 20_000)) + Fingerprint(new ReductionModel<float>(64, 20_000, 200_000))
                    + Fingerprint(new LossModel<float>(64, 20_000, 200_000));
                expectedDouble ??= actualDouble;
                expectedSingle ??= actualSingle;
                Assert.Equal(expectedDouble, actualDouble);
                Assert.Equal(expectedSingle, actualSingle);
            }
        }
        finally
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = saved;
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(0)]
    public void RepeatingTheSameModelAllocatesNoManagedBytes(int degree)
    {
        int saved = LinearAlgebraSettings.MaxDegreeOfParallelism;
        try
        {
            if (degree > 0)
            {
                LinearAlgebraSettings.MaxDegreeOfParallelism = degree;
            }

            AssertNoAllocation(new Model<double>(300, 1000, 200_000));
            AssertNoAllocation(new Model<float>(300, 1000, 200_000));
            AssertNoAllocation(new Model<double>(4, 16, 16));
            AssertNoAllocation(new ElementwiseModel<double>(20, 200_000, 300));
            AssertNoAllocation(new ElementwiseModel<float>(20, 200_000, 300));
            AssertNoAllocation(new ElementwiseModel<double>(4, 16, 5));
            AssertNoAllocation(new BroadcastModel<double>(64, 20_000));
            AssertNoAllocation(new BroadcastModel<float>(64, 20_000));
            AssertNoAllocation(new BroadcastModel<double>(3, 7));
            AssertNoAllocation(new ReductionModel<double>(64, 20_000, 200_000));
            AssertNoAllocation(new ReductionModel<float>(64, 20_000, 200_000));
            AssertNoAllocation(new ReductionModel<double>(3, 7, 5));
            AssertNoAllocation(new LossModel<double>(64, 20_000, 200_000));
            AssertNoAllocation(new LossModel<float>(64, 20_000, 200_000));
            AssertNoAllocation(new LossModel<double>(3, 7, 5));
        }
        finally
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = saved;
        }
    }

    [Fact]
    public void LinearAlgebraValuesAndGradientsDoNotDependOnTheDegreeOfParallelism()
    {
        // 1000 次の分解・求解と、その逆伝播（Cholesky のブロック版の逆伝播・逆行列・三角求解・Blas.Gemm）は並列の経路に入る。
        int saved = LinearAlgebraSettings.MaxDegreeOfParallelism;
        try
        {
            string? expected = null;
            for (int degree = 1; degree <= 4; degree++)
            {
                LinearAlgebraSettings.MaxDegreeOfParallelism = degree;
                string actual = Fingerprint(new LinearAlgebraModel(1000)) + Fingerprint(new GaussianProcessModel(300, lu: false))
                    + Fingerprint(new GaussianProcessModel(300, lu: true)) + Fingerprint(new OrthogonalModel(300, 200))
                    + Fingerprint(new OrthogonalModel(200, 500));
                expected ??= actual;
                Assert.Equal(expected, actual);
            }
        }
        finally
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = saved;
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(0)]
    public void RepeatingLinearAlgebraAllocatesNoManagedBytes(int degree)
    {
        int saved = LinearAlgebraSettings.MaxDegreeOfParallelism;
        try
        {
            if (degree > 0)
            {
                LinearAlgebraSettings.MaxDegreeOfParallelism = degree;
            }

            AssertNoAllocation(new LinearAlgebraModel(300));
            AssertNoAllocation(new LinearAlgebraModel(5));
            AssertNoAllocation(new GaussianProcessModel(128, lu: false));
            AssertNoAllocation(new GaussianProcessModel(128, lu: true));
            AssertNoAllocation(new GaussianProcessModel(16, lu: false));
            AssertNoAllocation(new OrthogonalModel(300, 200));
            AssertNoAllocation(new OrthogonalModel(200, 300));
            AssertNoAllocation(new OrthogonalModel(6, 4));
        }
        finally
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = saved;
        }
    }

    [Fact]
    public void ChangingShapesReusesStorageOnceEachShapeHasBeenSeen()
    {
        var small = new Model<double>(4, 16, 16);
        var large = new Model<double>(40, 64, 100);
        var tape = new Tape();
        for (int i = 0; i < 3; i++)
        {
            small.Run(tape);
            large.Run(tape);
        }

        GC.Collect(0, GCCollectionMode.Forced, blocking: true);
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 5; i++)
        {
            small.Run(tape);
            large.Run(tape);
        }

        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
    }

    // 測るスレッドの割り当て領域の残りを増分に入れないよう、数える直前に gen0 の GC を入れる（計画 3.7）。
    // 並列数 2 以上の作業の中で借りる配列も、コアが作業を実行するスロットごとの置き場から出すので、どのスレッドがどの作業を受け持っても確保しない。
    // 失敗したときは、モデル（呼び出しの式）と確保を出す。
    private static void AssertNoAllocation(IModel model, [CallerArgumentExpression(nameof(model))] string? expression = null)
    {
        var tape = new Tape();
        for (int i = 0; i < 3; i++)
        {
            model.Run(tape);
        }

        GC.Collect(0, GCCollectionMode.Forced, blocking: true);
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 5; i++)
        {
            model.Run(tape);
        }

        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.True(allocated == 0, $"{expression} (degree {LinearAlgebraSettings.MaxDegreeOfParallelism}) allocated {allocated} B in 5 runs");
    }

    private static string Fingerprint(IModel model)
    {
        var tape = new Tape();
        double value = model.Run(tape);
        var bits = new System.Text.StringBuilder();
        bits.Append(BitConverter.DoubleToInt64Bits(value).ToString("X16"));
        foreach (var gradient in model.Gradients)
        {
            bits.Append(';').Append(Convert.ToHexString(gradient));
        }

        bits.Append(';').Append(Convert.ToHexString(System.Runtime.InteropServices.MemoryMarshal.AsBytes<double>(model.FlatGradient)));
        return bits.ToString();
    }

    private interface IModel
    {
        double[] FlatGradient { get; }

        // 重みの勾配のバイト列。
        IEnumerable<byte[]> Gradients { get; }

        double Run(Tape tape);
    }

    // 行列積・行列ベクトル積・転置のビュー・内積・総和と、重み・定数として使う重み・葉を使うモデル。
    private sealed class Model<T> : IModel where T : unmanaged, IFloatingPointIeee754<T>
    {
        private readonly Matrix<T> data;
        private readonly Matrix<T> square;
        private readonly Parameter<T> wide;
        private readonly LinearAlgebra.Vector<T> theta;
        private readonly LinearAlgebra.Vector<T> target;
        private readonly LinearAlgebra.Vector<T> longVector;
        private readonly LinearAlgebra.Vector<T> mix;
        private readonly Parameter<T> weight;

        public Model(int order, int columns, int length)
        {
            var random = new Random(order + columns + length);
            square = ArrayProductTests.RandomMatrix<T>(random, order, order);
            data = ArrayProductTests.RandomMatrix<T>(random, order, order);
            wide = Parameter<T>.Matrix(columns, columns);
            ArrayProductTests.RandomMatrix<T>(random, columns, columns).AsSpan().CopyTo(wide.Value);
            theta = ArrayProductTests.RandomVector<T>(random, columns);
            target = ArrayProductTests.RandomVector<T>(random, columns);
            longVector = ArrayProductTests.RandomVector<T>(random, length);
            mix = ArrayProductTests.RandomVector<T>(random, order);
            weight = Parameter<T>.Matrix(order, order);
            ArrayProductTests.RandomMatrix<T>(random, order, order).AsSpan().CopyTo(weight.Value);
            FlatGradient = new double[columns + length];
        }

        public double[] FlatGradient { get; }

        public IEnumerable<byte[]> Gradients => [System.Runtime.InteropServices.MemoryMarshal.AsBytes(weight.Gradient).ToArray()];

        public double Run(Tape tape)
        {
            tape.Reset();
            weight.ZeroGradient();
            var w = tape.Leaf(theta);
            var v = tape.Leaf(longVector);
            var h = tape.UseMatrix(weight) * tape.Constant(data).T * tape.Constant(square);
            var r = tape.ConstantMatrix(wide).T * w;
            ScalarNode loss = XMath.Sum(h * tape.Constant(mix)) + XMath.Dot(r, tape.Constant(target)) + 0.5 * XMath.Dot(r, r)
                + XMath.Dot(v, v) + XMath.Sum(v[1..]);
            return tape.Gradient(loss, new VectorSpan<double>(FlatGradient));
        }
    }

    // 要素演算のモデル: Softplus と内積で書いたロジスティック回帰に、記録つきのスカラーとの演算、1要素ずつ局所微分を求める冪、
    // 転置のビュー（一時領域に集めて勾配のビューへ書き戻す経路）と重みの要素演算を加える。
    private sealed class ElementwiseModel<T> : IModel where T : unmanaged, IFloatingPointIeee754<T>
    {
        private readonly Matrix<T> data;
        private readonly LinearAlgebra.Vector<T> labels;
        private readonly LinearAlgebra.Vector<T> theta;
        private readonly LinearAlgebra.Vector<T> mix;
        private readonly Parameter<T> weight;

        public ElementwiseModel(int features, int samples, int order)
        {
            var random = new Random(features + samples + order);
            data = ArrayProductTests.RandomMatrix<T>(random, features, samples);
            labels = new LinearAlgebra.Vector<T>(samples);
            for (int i = 0; i < samples; i++)
            {
                labels[i] = random.Next(2) == 0 ? T.Zero : T.One;
            }

            theta = ArrayProductTests.RandomVector<T>(random, features);
            mix = ArrayProductTests.RandomVector<T>(random, order);
            weight = Parameter<T>.Matrix(order, order);
            ArrayProductTests.RandomMatrix<T>(random, order, order).AsSpan().CopyTo(weight.Value);
            FlatGradient = new double[features + 1];
        }

        public double[] FlatGradient { get; }

        public IEnumerable<byte[]> Gradients => [System.Runtime.InteropServices.MemoryMarshal.AsBytes(weight.Gradient).ToArray()];

        public double Run(Tape tape)
        {
            tape.Reset();
            weight.ZeroGradient();
            var w = tape.Leaf(theta);
            var scale = tape.Leaf(0.75);
            var eta = tape.Constant(data).T * w;
            var p = tape.UseMatrix(weight);
            var m = tape.Constant(mix);
            ScalarNode loss = XMath.Sum(XMath.Softplus(eta)) - XMath.Dot(tape.Constant(labels), eta)
                + XMath.Sum(XMath.Exp(XMath.Multiply(eta, scale))) + XMath.Sum(XMath.Pow(XMath.Square(eta) + 1, scale))
                + XMath.Sum(XMath.Log1mExp(-XMath.Abs(eta) - 0.1))
                + XMath.Sum(XMath.Hadamard(XMath.Logistic(p.T), p) * m) + XMath.Sum(XMath.Divide(p, XMath.Sqrt(XMath.Square(p.T) + 1)) * m);
            return tape.Gradient(loss, new VectorSpan<double>(FlatGradient));
        }
    }
    // ブロードキャストと活性化関数のモデル: 全結合層 W·X + b·1ᵀ に活性化関数を通し、列ごとの尺度（行として並べた葉）を掛ける。
    // 行列積の左に並べたノードを置く経路（逆伝播で写す）と、ブロードキャストへの寄与の行・列ごとの和を含める。
    private sealed class BroadcastModel<T> : IModel where T : unmanaged, IFloatingPointIeee754<T>
    {
        private readonly Matrix<T> data;
        private readonly LinearAlgebra.Vector<T> scales;
        private readonly LinearAlgebra.Vector<T> mix;
        private readonly LinearAlgebra.Vector<T> target;
        private readonly Parameter<T> weight;
        private readonly Parameter<T> bias;

        public BroadcastModel(int rows, int columns)
        {
            var random = new Random(rows + columns);
            data = ArrayProductTests.RandomMatrix<T>(random, rows, columns);
            scales = ArrayProductTests.RandomVector<T>(random, columns);
            mix = ArrayProductTests.RandomVector<T>(random, columns);
            target = ArrayProductTests.RandomVector<T>(random, rows);
            weight = Parameter<T>.Matrix(rows, rows);
            ArrayProductTests.RandomMatrix<T>(random, rows, rows).AsSpan().CopyTo(weight.Value);
            bias = Parameter<T>.Vector(rows);
            ArrayProductTests.RandomVector<T>(random, rows).AsSpan().CopyTo(bias.Value);
            FlatGradient = new double[columns];
        }

        public double[] FlatGradient { get; }

        public IEnumerable<byte[]> Gradients =>
        [
            System.Runtime.InteropServices.MemoryMarshal.AsBytes(weight.Gradient).ToArray(),
            System.Runtime.InteropServices.MemoryMarshal.AsBytes(bias.Gradient).ToArray(),
        ];

        public double Run(Tape tape)
        {
            tape.Reset();
            weight.ZeroGradient();
            bias.ZeroGradient();
            var s = tape.Leaf(scales);
            var b = tape.UseVector(bias);
            var z = tape.UseMatrix(weight) * tape.Constant(data) + b.Broadcast(data.Columns);
            var columnScale = s.BroadcastRows(data.Rows);
            var m = tape.Constant(mix);
            var h = XMath.Hadamard(XMath.Gelu(z), columnScale) + XMath.Gelu(z, approximate: true) + XMath.Silu(z) + XMath.Silu(z, approximate: true);
            var k = XMath.Relu(z) + XMath.LeakyRelu(z, T.CreateTruncating(0.1)) + XMath.Elu(z - columnScale, T.CreateTruncating(0.5)) + XMath.Selu(z);
            ScalarNode loss = XMath.Sum(h * m) + XMath.Dot(k * m, tape.Constant(target)) + XMath.Sum(b.Broadcast(data.Rows) * b);
            return tape.Gradient(loss, new VectorSpan<double>(FlatGradient));
        }
    }

    // 縮約とソフトマックスのモデル: 重みをスコアの行列そのものにした多項ロジスティック回帰の損失（列ごとの対数ソフトマックス）に、
    // 行・列ごとのソフトマックスと縮約、長いベクトルの ln Σ exp・平均・2乗和・ソフトマックスを加える。行列積・行列ベクトル積を含まない。
    private sealed class ReductionModel<T> : IModel where T : unmanaged, IFloatingPointIeee754<T>
    {
        private readonly Matrix<T> targets;
        private readonly LinearAlgebra.Vector<T> longVector;
        private readonly LinearAlgebra.Vector<T> longMix;
        private readonly LinearAlgebra.Vector<T> classMix;
        private readonly LinearAlgebra.Vector<T> sampleMix;
        private readonly Parameter<T> scores;

        public ReductionModel(int classes, int samples, int length)
        {
            var random = new Random(classes + samples + length);
            targets = new Matrix<T>(classes, samples);
            for (int n = 0; n < samples; n++)
            {
                targets[random.Next(classes), n] = T.One;
            }

            longVector = ArrayProductTests.RandomVector<T>(random, length);
            longMix = ArrayProductTests.RandomVector<T>(random, length);
            classMix = ArrayProductTests.RandomVector<T>(random, classes);
            sampleMix = ArrayProductTests.RandomVector<T>(random, samples);
            scores = Parameter<T>.Matrix(classes, samples);
            ArrayProductTests.RandomMatrix<T>(random, classes, samples).AsSpan().CopyTo(scores.Value);
            FlatGradient = new double[length];
        }

        public double[] FlatGradient { get; }

        public IEnumerable<byte[]> Gradients => [System.Runtime.InteropServices.MemoryMarshal.AsBytes(scores.Gradient).ToArray()];

        public double Run(Tape tape)
        {
            tape.Reset();
            scores.ZeroGradient();
            var z = tape.UseMatrix(scores);
            var v = tape.Leaf(longVector);
            var c = tape.Constant(classMix);
            var m = tape.Constant(sampleMix);
            var mix = tape.Constant(longMix);
            int classes = z.Rows;
            ScalarNode loss = -XMath.Sum(XMath.ColumnSums(XMath.Hadamard(tape.Constant(targets), XMath.ColumnLogSoftmax(z))))
                + XMath.Dot(XMath.RowLogSumExp(z), c) + XMath.Dot(XMath.RowSums(XMath.Hadamard(XMath.RowSoftmax(z), m.BroadcastRows(classes))), c)
                + XMath.Dot(XMath.ColumnSums(XMath.Hadamard(XMath.ColumnSoftmax(z.T), m.Broadcast(classes))), c)
                + XMath.Dot(XMath.ColumnMeans(z), m) + XMath.Dot(XMath.RowMeans(z), c) + XMath.Dot(XMath.ColumnSumsOfSquares(z), m)
                + XMath.Dot(XMath.RowSumsOfSquares(z), c) + XMath.Dot(XMath.ColumnLogSumExp(XMath.RowLogSoftmax(z)), m)
                + XMath.LogSumExp(v) + XMath.Mean(v) + 0.5 * XMath.SumOfSquares(v) + XMath.Dot(XMath.Softmax(v), mix) + XMath.Dot(XMath.LogSoftmax(v), mix);
            return tape.Gradient(loss, new VectorSpan<double>(FlatGradient));
        }
    }

    // 損失のモデル: スコアの行列を重みにした、ラベル・目標の分布（葉）のソフトマックス交差エントロピーと行列のロジスティック交差エントロピー、
    // 転置のビューのロジスティック交差エントロピー、長いベクトルのロジスティック交差エントロピーと、その先頭のソフトマックス交差エントロピー。
    // 行列積・行列ベクトル積を含まない。
    private sealed class LossModel<T> : IModel where T : unmanaged, IFloatingPointIeee754<T>
    {
        private readonly int[] labels;
        private readonly Matrix<T> soft;
        private readonly Matrix<T> binary;
        private readonly LinearAlgebra.Vector<T> longVector;
        private readonly LinearAlgebra.Vector<T> longTargets;
        private readonly Parameter<T> scores;

        public LossModel(int classes, int samples, int length)
        {
            var random = new Random(classes * 7 + samples + length);
            labels = new int[samples];
            soft = new Matrix<T>(classes, samples);
            binary = new Matrix<T>(samples, classes);
            for (int n = 0; n < samples; n++)
            {
                labels[n] = random.Next(classes);
                for (int k = 0; k < classes; k++)
                {
                    soft[k, n] = T.CreateTruncating(random.NextDouble() < 0.3 ? 0 : random.NextDouble());
                    binary[n, k] = T.CreateTruncating(random.Next(3) switch { 0 => 0, 1 => 1, _ => random.NextDouble() });
                }
            }

            longVector = ArrayProductTests.RandomVector<T>(random, length);
            longTargets = new LinearAlgebra.Vector<T>(length);
            for (int i = 0; i < length; i++)
            {
                longTargets[i] = T.CreateTruncating(random.Next(2));
            }

            scores = Parameter<T>.Matrix(classes, samples);
            ArrayProductTests.RandomMatrix<T>(random, classes, samples).AsSpan().CopyTo(scores.Value);
            FlatGradient = new double[length + classes * samples];
        }

        public double[] FlatGradient { get; }

        public IEnumerable<byte[]> Gradients => [System.Runtime.InteropServices.MemoryMarshal.AsBytes(scores.Gradient).ToArray()];

        public double Run(Tape tape)
        {
            tape.Reset();
            scores.ZeroGradient();
            var z = tape.UseMatrix(scores);
            var v = tape.Leaf(longVector);
            var t = tape.Leaf(soft);
            var y = tape.Constant(longTargets);
            ScalarNode loss = XMath.SoftmaxCrossEntropy(z, labels) + 0.5 * XMath.SoftmaxCrossEntropy(z, t)
                + XMath.LogisticCrossEntropy(z.T, tape.Constant(binary)) + XMath.LogisticCrossEntropy(z, XMath.Logistic(t))
                + XMath.LogisticCrossEntropy(v, y) + XMath.SoftmaxCrossEntropy(v[..z.Rows], labels[0]) + XMath.SoftmaxCrossEntropy(v[1..(z.Rows + 1)], y[..z.Rows]);
            return tape.Gradient(loss, new VectorSpan<double>(FlatGradient));
        }
    }

    // 線形代数のモデル: 対称正定値行列の Cholesky 分解（因子・ln det・求解・二次形式）、因子を係数にした三角求解、
    // 一般の行列の LU 分解（求解・転置の求解・ln |det|）と XMath.Solve。分解の入力・右辺・因子のどれも勾配を要る。
    private sealed class LinearAlgebraModel : IModel
    {
        private readonly Matrix<double> spd, general, factorWeight, rhs, rhsWeight;
        private readonly LinearAlgebra.Vector<double> b, mix;

        public LinearAlgebraModel(int n)
        {
            var random = new Random(n + 31);
            spd = LinearAlgebraTests.SymmetricPositiveDefinite(random, n, garbleUpper: true);
            general = LinearAlgebraTests.RandomMatrix(random, n, n);
            factorWeight = LinearAlgebraTests.RandomMatrix(random, n, n);
            rhs = LinearAlgebraTests.RandomMatrix(random, n, 3);
            rhsWeight = LinearAlgebraTests.RandomMatrix(random, n, 3);
            b = LinearAlgebraTests.RandomVector(random, n);
            mix = LinearAlgebraTests.RandomVector(random, n);
            FlatGradient = new double[2 * n * n + n * 3 + n];
        }

        public double[] FlatGradient { get; }

        public IEnumerable<byte[]> Gradients => [];

        public double Run(Tape tape)
        {
            tape.Reset();
            var a = tape.Leaf(spd);
            var m = tape.Leaf(general);
            var r = tape.Leaf(rhs);
            var x = tape.Leaf(b);
            var w = tape.Constant(mix);
            var cholesky = CholeskyDecompositionNode.Compute(a);
            var lu = LuDecompositionNode.Compute(m);
            var z = XMath.SolveLowerTriangular(cholesky.L, x);
            ScalarNode loss = LinearAlgebraTests.Pair(tape, cholesky.L, factorWeight) + cholesky.LogDeterminant + cholesky.InverseQuadraticForm(x)
                + LinearAlgebraTests.Pair(tape, cholesky.Solve(r), rhsWeight) + XMath.Dot(w, XMath.SolveUpperTriangular(cholesky.L.T, z))
                + XMath.Dot(w, lu.Solve(z)) + LinearAlgebraTests.Pair(tape, lu.SolveTransposed(r), rhsWeight) + lu.LogAbsDeterminant
                + XMath.Dot(w, XMath.Solve(m, x)) + LinearAlgebraTests.Pair(tape, XMath.SolveLowerTriangular(cholesky.L, r), rhsWeight);
            return tape.Gradient(loss, new VectorSpan<double>(FlatGradient));
        }
    }

    // QR 分解・対称固有値分解・特異値分解のモデル（m × n の一般の行列と m 次の対称行列）。どの結果も勾配を要る。
    // m ≥ 1.6n の特異値分解は先に QR 分解する経路、m > 25 の対称固有値分解は分割統治法の経路に入る。
    private sealed class OrthogonalModel : IModel
    {
        private readonly Matrix<double> general, symmetric, weightQ, weightR, weightU, weightV, weightProduct, weightVectors;
        private readonly LinearAlgebra.Vector<double> weightValues, weightEigenvalues;

        public OrthogonalModel(int m, int n)
        {
            var random = new Random(m * 7 + n);
            int k = Math.Min(m, n);
            general = LinearAlgebraTests.RandomMatrix(random, m, n);
            symmetric = OrthogonalDecompositionTests.SymmetricMatrix(random, m, garbleUpper: true);
            weightQ = LinearAlgebraTests.RandomMatrix(random, m, k);
            weightR = LinearAlgebraTests.RandomMatrix(random, k, n);
            weightU = LinearAlgebraTests.RandomMatrix(random, m, k);
            weightV = LinearAlgebraTests.RandomMatrix(random, n, k);
            weightProduct = LinearAlgebraTests.RandomMatrix(random, m, n);
            weightVectors = LinearAlgebraTests.RandomMatrix(random, m, m);
            weightValues = LinearAlgebraTests.RandomVector(random, k);
            weightEigenvalues = LinearAlgebraTests.RandomVector(random, m);
            FlatGradient = new double[m * n + m * m];
        }

        public double[] FlatGradient { get; }

        public IEnumerable<byte[]> Gradients => [];

        public double Run(Tape tape)
        {
            tape.Reset();
            var a = tape.Leaf(general);
            var s = tape.Leaf(symmetric);
            var qr = QrDecompositionNode.Compute(a);
            var svd = SingularValueDecompositionNode.Compute(a);
            var eigen = SymmetricEigenDecompositionNode.Compute(s);
            ScalarNode loss = LinearAlgebraTests.Pair(tape, qr.Q, weightQ) + LinearAlgebraTests.Pair(tape, qr.R, weightR)
                + XMath.Dot(tape.Constant(weightValues), svd.Values) + LinearAlgebraTests.Pair(tape, XMath.Hadamard(svd.U, svd.U), weightU)
                + LinearAlgebraTests.Pair(tape, XMath.Hadamard(svd.V, svd.V), weightV) + LinearAlgebraTests.Pair(tape, svd.U * svd.V.T, weightProduct)
                + XMath.Dot(tape.Constant(weightEigenvalues), eigen.Values) + LinearAlgebraTests.Pair(tape, XMath.Hadamard(eigen.Vectors, eigen.Vectors), weightVectors);
            return tape.Gradient(loss, new VectorSpan<double>(FlatGradient));
        }
    }

    // ガウス過程回帰の対数周辺尤度（GaussianProcessExample）。
    private sealed class GaussianProcessModel(int observations, bool lu) : IModel
    {
        private readonly GaussianProcessExample example = new(observations);

        public double[] FlatGradient { get; } = new double[3];

        public IEnumerable<byte[]> Gradients => [];

        public double Run(Tape tape) => example.Automatic(tape, FlatGradient, lu);
    }
}
