using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Tuna.Differentiation;
using Tuna.LinearAlgebra;
using Tuna.Optimization.Stochastic;
using Xunit.Abstractions;

namespace Tuna.NeuralNetwork.Tests.Training;

/// <summary>NN の層の記録を組み合わせた学習の1ステップ（並列数によらない値と勾配、2回目以降の確保 0、損失の減少）</summary>
[Collection(nameof(ParallelismCollection))]
public class TrainingTests(ITestOutputHelper output)
{
    private static readonly int[] Degrees = [1, 2, 3, 4, 8];

    [Fact]
    public void ValuesAndGradientsDoNotDependOnTheDegreeOfParallelism()
    {
        // 全結合（行列積）・正規化・PReLU のどれも並列の経路に入る大きさにする。
        string? expected = null;
        foreach (int degree in Degrees)
        {
            string actual = TestData.WithThreads(degree,
                () => Fingerprint(new Model<float>(32, 64, 10, 4500, seed: 1)) + Fingerprint(new Model<double>(32, 64, 10, 2500, seed: 2)));
            expected ??= actual;
            Assert.True(expected == actual, $"degree {degree}");
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(0)]
    public void RepeatingATrainingStepAllocatesNoManagedBytes(int degree)
    {
        int saved = LinearAlgebraSettings.MaxDegreeOfParallelism;
        try
        {
            if (degree > 0)
            {
                LinearAlgebraSettings.MaxDegreeOfParallelism = degree;
            }

            AssertNoAllocation(new Model<float>(32, 64, 10, 4500, seed: 3));
            AssertNoAllocation(new Model<double>(32, 64, 10, 2500, seed: 4));
            AssertNoAllocation(new Model<float>(2, 8, 3, 5, seed: 5));
            AssertNoAllocation(new Model<double>(2, 8, 3, 5, seed: 6));
        }
        finally
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = saved;
        }
    }

    [Fact]
    public void GradientDescentReducesTheLossOfASmallNetwork()
    {
        // 3 クラスの渦巻きの分類。全結合 → LayerNorm → PReLU → RMSNorm → 全結合を確率的でない勾配降下で学習する。
        var model = new Model<float>(2, 32, 3, 150, seed: 7, spiral: true);
        var tape = new Tape();
        double initial = model.Run(tape) / model.Samples;
        double loss = initial;
        for (int step = 0; step < 300; step++)
        {
            foreach (var parameter in model.Parameters)
            {
                var value = parameter.Value;
                var gradient = parameter.Gradient;
                for (int i = 0; i < value.Length; i++)
                {
                    value[i] -= 0.5f * gradient[i] / model.Samples;
                }
            }

            loss = model.Run(tape) / model.Samples;
        }

        Assert.True(loss < 0.25 * initial, $"loss {initial:F4} -> {loss:F4}");
        Assert.True(model.Accuracy() > 0.9, $"accuracy {model.Accuracy():P1}");
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void AdamWithGlobalClippingTrainsASmallNetwork(bool singlePrecision)
    {
        if (singlePrecision)
        {
            TrainWithAdam<float>();
        }
        else
        {
            TrainWithAdam<double>();
        }
    }

    private void TrainWithAdam<T>() where T : unmanaged, IFloatingPointIeee754<T>
    {
        var model = new Model<T>(2, 32, 3, 150, seed: 7, spiral: true);
        var optimizer = new Adam<T>(model.Parameters, new AdamOptions { LearningRate = 0.01, GradientClipNorm = 10 });
        var tape = new Tape();
        double initial = model.Run(tape) / model.Samples;
        double loss = initial;
        for (int step = 0; step < 250; step++)
        {
            optimizer.Step();
            loss = model.Run(tape) / model.Samples;
        }

        double accuracy = model.Accuracy();
        output.WriteLine($"{typeof(T).Name}: loss {initial:F6} -> {loss:F6}, accuracy {accuracy:P1}");
        Assert.True(loss < 0.25 * initial, $"loss {initial:F4} -> {loss:F4}");
        Assert.True(accuracy > 0.9, $"accuracy {accuracy:P1}");
    }

    // 測るスレッドの割り当て領域の残りを増分に入れないよう、数える直前に gen0 の GC を入れる。数える窓は3回まで測り、どれかが 0 なら確保しないとみなす（設計 3.7）。
    private static void AssertNoAllocation<T>(Model<T> model, [CallerArgumentExpression(nameof(model))] string? expression = null)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        var tape = new Tape();
        var optimizer = new Adam<T>(model.Parameters, new AdamOptions { GradientClipNorm = 1 });
        for (int i = 0; i < 3; i++)
        {
            model.Run(tape);
            optimizer.Step();
        }

        long allocated = 0;
        for (int window = 0; window < 3; window++)
        {
            GC.Collect(0, GCCollectionMode.Forced, blocking: true);
            long before = GC.GetAllocatedBytesForCurrentThread();
            for (int i = 0; i < 5; i++)
            {
                model.Run(tape);
                optimizer.Step();
            }

            allocated = GC.GetAllocatedBytesForCurrentThread() - before;
            if (allocated == 0)
            {
                return;
            }
        }

        Assert.Fail($"{expression} (degree {LinearAlgebraSettings.MaxDegreeOfParallelism}) allocated {allocated} B in 5 runs");
    }

    private static string Fingerprint<T>(Model<T> model) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var tape = new Tape();
        double loss = model.Run(tape);
        new Adam<T>(model.Parameters, new AdamOptions { GradientClipNorm = 1 }).Step();
        var bits = new System.Text.StringBuilder(BitConverter.DoubleToInt64Bits(loss).ToString("X16"));
        foreach (var parameter in model.Parameters)
        {
            bits.Append(';').Append(Convert.ToHexString(MemoryMarshal.AsBytes<T>(parameter.Gradient)));
            bits.Append(';').Append(Convert.ToHexString(MemoryMarshal.AsBytes<T>(parameter.Value)));
        }

        bits.Append(';').Append(Convert.ToHexString(MemoryMarshal.AsBytes<T>(model.InputGradient)));
        return bits.ToString();
    }

    // 全結合 → LayerNorm → PReLU（行ごとの傾き）→ RMSNorm → 全結合（バイアスなし）→ ソフトマックス交差エントロピー。
    // 入力も葉にして、入力の勾配（全結合の dX）の経路も通す。
    private sealed class Model<T> where T : unmanaged, IFloatingPointIeee754<T>
    {
        private readonly Parameter<T> w1, b1, scale, shift, slopes, rms, w2;
        private readonly Matrix<T> data;
        private readonly int[] labels;
        private readonly T[] inputGradient;
        private readonly T epsilon = T.CreateTruncating(1e-5);
        private Matrix<T>? logits;

        public Model(int inputs, int hidden, int classes, int samples, int seed, bool spiral = false)
        {
            var random = new Random(seed);
            Samples = samples;
            data = new Matrix<T>(inputs, samples);
            labels = new int[samples];
            for (int j = 0; j < samples; j++)
            {
                labels[j] = j % classes;
                if (spiral)
                {
                    double radius = (double)(j / classes + 1) / (samples / classes), angle = 4 * labels[j] + 4 * radius + 0.2 * random.NextDouble();
                    data[0, j] = T.CreateTruncating(radius * Math.Sin(angle));
                    data[1, j] = T.CreateTruncating(radius * Math.Cos(angle));
                }
                else
                {
                    for (int i = 0; i < inputs; i++)
                    {
                        data[i, j] = T.CreateTruncating(2 * random.NextDouble() - 1);
                    }
                }
            }

            w1 = Random(Parameter<T>.Matrix(hidden, inputs), random, Math.Sqrt(6.0 / inputs));
            b1 = Random(Parameter<T>.Vector(hidden), random, 0.1);
            scale = Random(Parameter<T>.Vector(hidden), random, 0.1, 1);
            shift = Random(Parameter<T>.Vector(hidden), random, 0.1);
            slopes = Random(Parameter<T>.Vector(hidden), random, 0.1, 0.25);
            rms = Random(Parameter<T>.Vector(hidden), random, 0.1, 1);
            w2 = Random(Parameter<T>.Matrix(classes, hidden), random, Math.Sqrt(6.0 / hidden));
            Parameters = new ParameterSet<T>([w1, b1, scale, shift, slopes, rms, w2]);
            inputGradient = new T[inputs * samples];
        }

        public ParameterSet<T> Parameters { get; }

        public int Samples { get; }

        public ReadOnlySpan<T> InputGradient => inputGradient;

        public double Run(Tape tape)
        {
            tape.Reset();
            Parameters.ZeroGradients();
            var x = tape.Leaf(data);
            var h = XMath.Linear(tape.UseMatrix(w1), x, tape.UseVector(b1));
            h = XMath.LayerNorm(h, tape.UseVector(scale), tape.UseVector(shift), epsilon);
            h = XMath.Prelu(h, tape.UseVector(slopes));
            h = XMath.RmsNorm(h, tape.UseVector(rms), epsilon);
            var z = XMath.Linear(tape.UseMatrix(w2), h);
            var loss = XMath.SoftmaxCrossEntropy(z, labels);
            tape.Backward(loss);
            tape.GetGradient(x).CopyTo(new MatrixSpan<T>(inputGradient, data.Rows, data.Columns));
            logits ??= new Matrix<T>(z.Rows, z.Columns);
            z.Value.CopyTo(logits);
            return loss.Value;
        }

        public double Accuracy()
        {
            int correct = 0;
            for (int j = 0; j < logits!.Columns; j++)
            {
                int best = 0;
                for (int i = 1; i < logits.Rows; i++)
                {
                    best = logits[i, j] > logits[best, j] ? i : best;
                }

                correct += best == labels[j] ? 1 : 0;
            }

            return (double)correct / logits.Columns;
        }

        private static Parameter<T> Random(Parameter<T> parameter, Random random, double scale, double center = 0)
        {
            var value = parameter.Value;
            for (int i = 0; i < value.Length; i++)
            {
                value[i] = T.CreateTruncating(center + scale * (2 * random.NextDouble() - 1));
            }

            return parameter;
        }
    }
}
