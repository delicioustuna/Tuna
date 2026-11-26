using Tuna.LinearAlgebra;
using Tuna.Optimization;
using Tuna.Optimization.Stochastic;

namespace Tuna.Tests.Optimization;

[Trait("Optimization", "Stochastic")]
public class StochasticGradientTests
{
    // ---------- 1回の更新を式どおりに計算した値と比べる ----------

    private static double[] Run(StochasticGradientOptimizer optimizer, double[] start, params double[][] gradients)
    {
        var x = (double[])start.Clone();
        foreach (var g in gradients)
        {
            optimizer.Step(x, g);
        }

        return x;
    }

    [Fact]
    public void SgdStepsMatchTheFormulas()
    {
        Assert.Equal(0.95, Run(new StochasticGradientDescent(1, new() { LearningRate = 0.1 }), [1.0], [0.5])[0], 1e-15);

        // v₁ = 1、x = 0.9。v₂ = 0.9·1 + 1 = 1.9、x = 0.9 − 0.19
        var heavyBall = new StochasticGradientDescent(1, new() { LearningRate = 0.1, Momentum = 0.9 });
        Assert.Equal(0.71, Run(heavyBall, [1.0], [1.0], [1.0])[0], 1e-15);

        // Nesterov: x −= η(g + μv)。1回目 1.9、2回目 1 + 0.9·1.9 = 2.71
        var nesterov = new StochasticGradientDescent(1, new() { LearningRate = 0.1, Momentum = 0.9, Nesterov = true });
        Assert.Equal(1 - 0.19 - 0.271, Run(nesterov, [1.0], [1.0], [1.0])[0], 1e-15);
        Assert.True(nesterov.Nesterov);
        Assert.False(new StochasticGradientDescent(1, new() { Nesterov = true }).Nesterov);   // モーメンタムなしでは無効
    }

    [Fact]
    public void AdaptiveStepsMatchTheFormulas()
    {
        const double eta = 0.01;
        double[] g1 = [2.0, -0.5], g2 = [-1.0, 0.0];

        // AdaGrad: G = g₁² + g₂²
        var adagrad = Run(new AdaGrad(2, new() { LearningRate = eta, Epsilon = 1e-10 }), [0.0, 0.0], g1, g2);
        Assert.Equal(-eta * 2 / (2 + 1e-10) + eta * 1 / (Math.Sqrt(5) + 1e-10), adagrad[0], 1e-15);
        Assert.Equal(eta * 0.5 / (0.5 + 1e-10), adagrad[1], 1e-15);

        // RMSProp（ρ = 0.9）: s₁ = 0.1g₁²
        var rms = Run(new RmsProp(2, new() { LearningRate = eta }), [0.0, 0.0], g1);
        Assert.Equal(-eta * 2 / (Math.Sqrt(0.4) + 1e-8), rms[0], 1e-15);

        // 中心化: s − m² = 0.1g² − 0.01g² = 0.09g²
        var centered = Run(new RmsProp(2, new() { LearningRate = eta, Centered = true }), [0.0, 0.0], g1);
        Assert.Equal(-eta * 2 / (0.3 * 2 + 1e-8), centered[0], 1e-14);

        // モーメンタム: b₂ = μb₁ + g₂ / r₂
        var momentum = Run(new RmsProp(1, new() { LearningRate = eta, Momentum = 0.5 }), [0.0], [1.0], [1.0]);
        double r1 = Math.Sqrt(0.1) + 1e-8, r2 = Math.Sqrt(0.19) + 1e-8;
        double b1 = 1 / r1, b2 = 0.5 * b1 + 1 / r2;
        Assert.Equal(-eta * (b1 + b2), momentum[0], 1e-14);
    }

    [Fact]
    public void AdamFamilyStepsMatchTheFormulas()
    {
        const double eta = 0.001, b1 = 0.9, b2 = 0.999, eps = 1e-8;

        // 1回目は m̂ = g、v̂ = g² なので、歩幅はほぼ η·sign(g)
        var adam = new Adam(1);
        Assert.Equal(1 - eta * 0.5 / (0.5 + eps), Run(adam, [1.0], [0.5])[0], 1e-15);

        // 2回目（g = 1 の後に g = 0）
        double m = (1 - b1) * 1, v = (1 - b2) * 1;
        double mHat2 = b1 * m / (1 - b1 * b1);
        double v2 = b2 * v;
        double adamStep2 = eta * mHat2 / (Math.Sqrt(v2 / (1 - b2 * b2)) + eps);
        double amsStep2 = eta * mHat2 / (Math.Sqrt(v / (1 - b2 * b2)) + eps);   // v₂ < v₁ なので最大値は v₁
        double first = eta * 1 / (1 + eps);
        Assert.Equal(-first - adamStep2, Run(new Adam(1), [0.0], [1.0], [0.0])[0], 1e-15);
        Assert.Equal(-first - amsStep2, Run(new Adam(1, new() { Amsgrad = true }), [0.0], [1.0], [0.0])[0], 1e-15);

        // NAdam の1回目: m̂ = β₁(1 − β₁)g / (1 − β₁²) + (1 − β₁)g / (1 − β₁)
        double nadamHat = b1 * (1 - b1) / (1 - b1 * b1) + 1;
        Assert.Equal(-0.002 * nadamHat / (1 + eps), Run(new NAdam(1), [0.0], [1.0])[0], 1e-15);

        // AdaMax: 1回目の歩幅は η（m / u = 1 − β₁ を 1 − β₁ で割る）。勾配 0 の成分は動かない
        var adamax = Run(new AdaMax(2), [0.0, 0.0], [-3.0, 0.0]);
        Assert.Equal(0.002, adamax[0], 1e-15);
        Assert.Equal(0.0, adamax[1]);
    }

    // ---------- 重み減衰・切り詰め・予定 ----------

    [Fact]
    public void WeightDecayIsCoupledOrDecoupled()
    {
        // SGD では両者は同じ: x ← x − ηλx
        Assert.Equal(0.99, Run(new StochasticGradientDescent(1, new() { LearningRate = 0.1, WeightDecay = 0.1 }), [1.0], [0.0])[0], 1e-15);
        Assert.Equal(0.99, Run(new StochasticGradientDescent(1, new() { LearningRate = 0.1, WeightDecay = 0.1, DecoupledWeightDecay = true }), [1.0], [0.0])[0], 1e-15);

        // Adam: 勾配に足した λx は正規化されて歩幅 ≈ η。切り離せば (1 − ηλ)x だけ
        var coupled = Run(new Adam(1, new() { WeightDecay = 0.1 }), [1.0], [0.0]);
        Assert.Equal(1 - 0.001 * 0.1 / (0.1 + 1e-8), coupled[0], 1e-15);
        var decoupled = Run(new Adam(1, new() { WeightDecay = 0.1, DecoupledWeightDecay = true }), [1.0], [0.0]);
        Assert.Equal(1 - 0.001 * 0.1, decoupled[0], 1e-15);
    }

    [Fact]
    public void ClipsTheGradientNormBeforeAddingWeightDecay()
    {
        var sgd = new StochasticGradientDescent(2, new() { LearningRate = 1, GradientClipNorm = 1 });
        var x = Run(sgd, [0.0, 0.0], [3.0, 4.0]);
        Assert.Equal(-0.6, x[0], 1e-15);
        Assert.Equal(-0.8, x[1], 1e-15);

        // 上限以下の勾配はそのまま
        Assert.Equal(-0.5, Run(new StochasticGradientDescent(1, new() { LearningRate = 1, GradientClipNorm = 1 }), [0.0], [0.5])[0]);

        // 減衰の項は切り詰めの後に足す: g = 10 → 1、+ λx = 0.5
        var withDecay = new StochasticGradientDescent(1, new() { LearningRate = 0.1, GradientClipNorm = 1, WeightDecay = 0.5 });
        Assert.Equal(1 - 0.1 * 1.5, Run(withDecay, [1.0], [10.0])[0], 1e-15);
    }

    // 平方和がオーバーフローまたはアンダーフローする勾配のクリッピング回帰テスト。
    [Theory]
    [InlineData(3.0, 4.0, 1.0)]
    [InlineData(3e200, 4e200, 1.0)]
    [InlineData(-3e300, 4e300, 1.0)]
    [InlineData(3e-170, 4e-170, 1e-200)]
    [InlineData(3e-170, 4e-170, 1e300)]
    public void GradientClippingKeepsTheDirectionOfRepresentableGradients(double a, double b, double clip)
    {
        var solver = new StochasticGradientDescent(2, new StochasticGradientDescentOptions { LearningRate = 1, GradientClipNorm = clip });
        double[] x = [0, 0];
        solver.Step(x, new double[] { a, b });

        // ‖(a, b)‖ = 5|a|/3。clip を超えれば長さ clip に、超えなければそのまま
        double norm = 5 * Math.Abs(a) / 3;
        double scale = norm > clip ? clip / norm : 1;
        Assert.Equal(-a * scale, x[0], Math.Abs(a * scale) * 1e-14);
        Assert.Equal(-b * scale, x[1], Math.Abs(b * scale) * 1e-14);
    }

    [Theory]
    [InlineData(3e150, 4e150, 1e-300)]
    [InlineData(-3e200, 4e200, 1e-300)]
    [InlineData(3e10, 4e10, 1e-300)]
    public void GradientClippingToATinyNormKeepsTheClippedGradient(double a, double b, double clip)
    {
        // 倍率 clip/‖g‖ は double で表せない（2e-451）か非正規化数（2e-311）だが、切り詰めた勾配 (±0.6, 0.8)·clip は表せる。
        // 以前は [3e150, 4e150] で更新が [0, 0]、[3e10, 4e10] で相対誤差が 1e-13 程度になった
        var solver = new StochasticGradientDescent(2, new StochasticGradientDescentOptions { LearningRate = 1, GradientClipNorm = clip });
        double[] x = [0, 0];
        solver.Step(x, new double[] { a, b });

        double first = -0.6 * Math.Sign(a) * clip, second = -0.8 * clip;
        Assert.Equal(first, x[0], Math.Abs(first) * 1e-14);
        Assert.Equal(second, x[1], Math.Abs(second) * 1e-14);
    }

    [Fact]
    public void GradientClippingOfManyHugeComponentsDoesNotOverflow()
    {
        // ‖g‖ = 1e308·√16 は double であふれるが、切り詰めた結果は表現できる
        const int n = 16;
        var solver = new Adam(n, new AdamOptions { LearningRate = 1, GradientClipNorm = 2 });
        var gradient = Enumerable.Repeat(1e308, n).ToArray();
        var clipped = new double[n];
        var plain = new StochasticGradientDescent(n, new StochasticGradientDescentOptions { LearningRate = 1, GradientClipNorm = 2 });
        plain.Step(clipped, gradient);
        foreach (double v in clipped)
        {
            Assert.Equal(-0.5, v, 1e-15);
        }

        var x = new double[n];
        solver.Step(x, gradient);
        Assert.All(x, v => Assert.True(double.IsFinite(v) && v < 0));
    }

    [Fact]
    public void SchedulesGiveTheDocumentedMultipliers()
    {
        var step = new StepDecaySchedule(10, 0.5);
        Assert.Equal(1.0, step.Multiplier(0));
        Assert.Equal(1.0, step.Multiplier(9));
        Assert.Equal(0.5, step.Multiplier(10));
        Assert.Equal(0.125, step.Multiplier(35));

        var exponential = new ExponentialDecaySchedule(0.5, decaySteps: 4);
        Assert.Equal(1.0, exponential.Multiplier(0));
        Assert.Equal(Math.Sqrt(0.5), exponential.Multiplier(2), 1e-15);
        Assert.Equal(0.25, exponential.Multiplier(8), 1e-15);

        Assert.Equal(0.2, new InverseTimeDecaySchedule(0.5).Multiplier(8), 1e-15);

        var cosine = new CosineAnnealingSchedule(100, minimumMultiplier: 0.1);
        Assert.Equal(1.0, cosine.Multiplier(0));
        Assert.Equal(0.55, cosine.Multiplier(50), 1e-15);
        Assert.Equal(0.1, cosine.Multiplier(100), 1e-15);
        Assert.Equal(0.1, cosine.Multiplier(1000), 1e-15);
        var restart = new CosineAnnealingSchedule(100, restart: true);
        Assert.Equal(1.0, restart.Multiplier(200));
        Assert.Equal(restart.Multiplier(30), restart.Multiplier(130));

        var warmup = new LinearWarmupSchedule(4, new StepDecaySchedule(2, 0.1));
        double[] expected = [0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 0.1];
        for (int t = 0; t < expected.Length; t++)
        {
            Assert.Equal(expected[t], warmup.Multiplier(t), 1e-15);
        }

        Assert.Throws<ArgumentOutOfRangeException>(() => new StepDecaySchedule(0, 0.5));
        Assert.Throws<ArgumentOutOfRangeException>(() => new StepDecaySchedule(1, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ExponentialDecaySchedule(1.5));
        Assert.Throws<ArgumentOutOfRangeException>(() => new InverseTimeDecaySchedule(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new CosineAnnealingSchedule(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new CosineAnnealingSchedule(10, 2));
        Assert.Throws<ArgumentOutOfRangeException>(() => new LinearWarmupSchedule(0));
    }

    [Fact]
    public void StepUsesTheScheduledLearningRate()
    {
        var sgd = new StochasticGradientDescent(1, new() { LearningRate = 1, Schedule = new StepDecaySchedule(1, 0.5) });
        var x = new double[1];
        double[] g = [1.0];
        double[] expected = [-1, -1.5, -1.75];
        for (int k = 0; k < 3; k++)
        {
            Assert.Equal(Math.Pow(0.5, k), sgd.CurrentLearningRate);
            sgd.Step(x, g);
            Assert.Equal(expected[k], x[0]);
        }

        Assert.Equal(3, sgd.StepCount);
        sgd.LearningRate = 8;
        Assert.Equal(1.0, sgd.CurrentLearningRate);
        sgd.Reset();
        Assert.Equal(0, sgd.StepCount);
        Assert.Equal(8.0, sgd.CurrentLearningRate);
        Assert.Throws<ArgumentOutOfRangeException>(() => sgd.LearningRate = 0);

        var broken = new StochasticGradientDescent(1, new() { Schedule = new NegativeSchedule() });
        Assert.Throws<InvalidOperationException>(() => broken.Step(x, g));
    }

    private sealed class NegativeSchedule : ILearningRateSchedule
    {
        public double Multiplier(long step) => -1;
    }

    // ---------- 状態・引数 ----------

    [Fact]
    public void ResetClearsTheState()
    {
        foreach (var make in Optimizers())
        {
            var optimizer = make(3, new CosineAnnealingSchedule(5));
            double[] start = [1.0, -2.0, 0.5];
            double[][] gradients = [[0.3, -1.0, 2.0], [0.1, 0.4, -0.2], [-0.5, 0.0, 1.0]];
            var a = Run(optimizer, start, gradients);
            optimizer.Reset();
            var b = Run(optimizer, start, gradients);
            Assert.Equal(a, b);
        }
    }

    [Fact]
    public void UpdatesStridedParameters()
    {
        // 1つおきの成分だけを更新し、間の成分は触らない
        var buffer = new[] { 1.0, 9.0, 2.0, 9.0, 3.0 };
        var parameters = new VectorSpan<double>(buffer, [(nint)3], [(nint)2]);
        var adam = new Adam(3);
        adam.Step(parameters, new double[] { 1.0, -1.0, 0.0 });
        Assert.Equal(9.0, buffer[1]);
        Assert.Equal(9.0, buffer[3]);
        Assert.Equal(1 - 0.001 / (1 + 1e-8), buffer[0], 1e-15);
        Assert.Equal(2 + 0.001 / (1 + 1e-8), buffer[2], 1e-15);
        Assert.Equal(3.0, buffer[4]);
    }

    [Fact]
    public void RejectsInvalidArguments()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new Adam(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Adam(1, new() { LearningRate = 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Adam(1, new() { LearningRate = double.PositiveInfinity }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Adam(1, new() { Beta1 = 1 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Adam(1, new() { Beta2 = -0.1 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Adam(1, new() { Epsilon = 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Adam(1, new() { WeightDecay = -1 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Adam(1, new() { GradientClipNorm = 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new StochasticGradientDescent(1, new() { Momentum = 1 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new AdaGrad(1, new() { InitialAccumulator = -1 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new RmsProp(1, new() { Decay = 1 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new NAdam(1, new() { Epsilon = double.NaN }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new AdaMax(1, new() { Beta2 = 1 }));

        var adam = new Adam(2);
        Assert.Throws<ArgumentException>(() => adam.Step(new double[3], new double[2]));
        Assert.Throws<ArgumentException>(() => adam.Step(new double[2], new double[3]));
        Assert.Throws<ArgumentException>(() => adam.Minimize((x, g) => 0, [1.0], 10));
        Assert.Throws<ArgumentException>(() => adam.Minimize((x, g) => 0, [1.0, double.NaN], 10));
        Assert.Throws<ArgumentOutOfRangeException>(() => adam.Minimize((x, g) => 0, [1.0, 1.0], -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => adam.Minimize((x, g) => 0, [1.0, 1.0], 1, gradientTolerance: -1));
    }

    // ---------- 収束 ----------

    // 次元と学習率スケジュールから最適化器を生成する
    private static IEnumerable<Func<int, ILearningRateSchedule?, StochasticGradientOptimizer>> Optimizers()
    {
        yield return (n, s) => new StochasticGradientDescent(n, new() { LearningRate = 0.02, Schedule = s });
        yield return (n, s) => new StochasticGradientDescent(n, new() { LearningRate = 0.02, Momentum = 0.9, Nesterov = true, Schedule = s });
        yield return (n, s) => new AdaGrad(n, new() { LearningRate = 0.5, Schedule = s });
        yield return (n, s) => new RmsProp(n, new() { LearningRate = 0.01, Centered = true, Momentum = 0.5, Schedule = s });
        yield return (n, s) => new Adam(n, new() { LearningRate = 0.05, Schedule = s });
        yield return (n, s) => new Adam(n, new() { LearningRate = 0.05, Amsgrad = true, Schedule = s });
        yield return (n, s) => new NAdam(n, new() { LearningRate = 0.05, Schedule = s });
        yield return (n, s) => new AdaMax(n, new() { LearningRate = 0.05, Schedule = s });
    }

    public static TheoryData<int> OptimizerIndices => new(Enumerable.Range(0, Optimizers().Count()));

    [Theory]
    [MemberData(nameof(OptimizerIndices))]
    public void FitsALinearModelFromNoisyMiniBatches(int index)
    {
        // y = wᵀa + 0.1·雑音 の最小二乗を、8 個ずつのミニバッチの勾配で解く。学習率は余弦で 0 まで下げる
        const int n = 4, samples = 512, batch = 8, epochs = 60;
        double[] truth = [1.5, -2.0, 0.5, 3.0];
        var random = new Random(17);
        var features = new double[samples, n];
        var targets = new double[samples];
        for (int s = 0; s < samples; s++)
        {
            double y = 0;
            for (int j = 0; j < n; j++)
            {
                features[s, j] = 2 * random.NextDouble() - 1;
                y += truth[j] * features[s, j];
            }

            targets[s] = y + 0.1 * (2 * random.NextDouble() - 1);
        }

        var optimizer = Optimizers().ElementAt(index)(n, new CosineAnnealingSchedule(epochs * samples / batch));
        var w = new double[n];
        var gradient = new double[n];
        var order = Enumerable.Range(0, samples).ToArray();
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            random.Shuffle(order);
            for (int start = 0; start < samples; start += batch)
            {
                Array.Clear(gradient);
                for (int k = start; k < start + batch; k++)
                {
                    int s = order[k];
                    double residual = -targets[s];
                    for (int j = 0; j < n; j++)
                    {
                        residual += w[j] * features[s, j];
                    }

                    for (int j = 0; j < n; j++)
                    {
                        gradient[j] += residual * features[s, j] / batch;
                    }
                }

                optimizer.Step(w, gradient);
            }
        }

        for (int j = 0; j < n; j++)
        {
            Assert.True(Math.Abs(w[j] - truth[j]) < 0.05, $"{optimizer.GetType().Name} w[{j}] = {w[j]}");
        }
    }

    [Theory]
    [MemberData(nameof(OptimizerIndices))]
    public void MinimizesADeterministicQuadratic(int index)
    {
        // f = Σ aᵢ(xᵢ − cᵢ)² + (x₀ − x₁)²、余弦の予定で 3000 回
        double[] a = [1.0, 4.0, 0.5, 2.0], c = [1.0, -1.0, 2.0, 0.5];
        var optimizer = Optimizers().ElementAt(index)(4, new CosineAnnealingSchedule(3000));
        var result = optimizer.Minimize(
            (x, g) =>
            {
                double sum = 0;
                for (int i = 0; i < 4; i++)
                {
                    double r = x[i] - c[i];
                    sum += a[i] * r * r;
                    g[i] = 2 * a[i] * r;
                }

                double d = x[0] - x[1];
                sum += d * d;
                g[0] += 2 * d;
                g[1] -= 2 * d;
                return sum;
            },
            new double[4],
            maxIterations: 3000);

        // 最小点: x₂, x₃ はそのまま、(x₀, x₁) は [[2a₀ + 2, −2], [−2, 2a₁ + 2]] x = [2a₀c₀, 2a₁c₁]
        double det = (2 * a[0] + 2) * (2 * a[1] + 2) - 4;
        double x0 = (2 * a[0] * c[0] * (2 * a[1] + 2) + 2 * 2 * a[1] * c[1]) / det;
        double x1 = ((2 * a[0] + 2) * 2 * a[1] * c[1] + 2 * 2 * a[0] * c[0]) / det;
        double[] expected = [x0, x1, c[2], c[3]];
        // 勾配がちょうど 0 になれば（許容誤差 0 でも）そこで止まる
        if (result.Termination == MinimizationTermination.MaxIterations)
        {
            Assert.Equal(3000, result.Iterations);
            Assert.Equal(3001, result.Evaluations);
        }
        else
        {
            Assert.Equal(MinimizationTermination.GradientTolerance, result.Termination);
            Assert.Equal(0.0, result.GradientNorm);
        }

        for (int i = 0; i < 4; i++)
        {
            Assert.True(Math.Abs(result.Point[i] - expected[i]) < 1e-3, $"{optimizer.GetType().Name} x[{i}] = {result.Point[i]}");
        }
    }

    [Fact]
    public void MinimizeStopsOnGradientToleranceAndNonFiniteValues()
    {
        // f = x²/2 を SGD（η = 0.5）で: 勾配は毎回半分になる
        var sgd = new StochasticGradientDescent(1, new() { LearningRate = 0.5 });
        var result = sgd.Minimize((x, g) =>
        {
            g[0] = x[0];
            return 0.5 * x[0] * x[0];
        }, [1.0], maxIterations: 100, gradientTolerance: 1e-3);
        Assert.Equal(MinimizationTermination.GradientTolerance, result.Termination);
        Assert.Equal(10, result.Iterations);                  // 2⁻¹⁰ < 1e-3 < 2⁻⁹
        Assert.Equal(Math.Pow(0.5, 10), result.Point[0]);
        Assert.True(result.Converged);

        // 学習率が大きすぎて発散する: 有限だった最良の点を返す
        var diverging = new StochasticGradientDescent(1, new() { LearningRate = 3 });
        var blown = diverging.Minimize((x, g) =>
        {
            g[0] = x[0] * x[0] * x[0];
            return 0.25 * Math.Pow(x[0], 4);
        }, [1.0], maxIterations: 100);
        Assert.Equal(MinimizationTermination.NonFiniteValue, blown.Termination);
        Assert.Equal(1.0, blown.Point[0]);
        Assert.Equal(0.25, blown.Value);

        // 初期点で有限でない
        var nan = new Adam(1).Minimize((x, g) => double.NaN, [1.0], maxIterations: 10);
        Assert.Equal(MinimizationTermination.NonFiniteValue, nan.Termination);
        Assert.Equal(0, nan.Iterations);
        Assert.Equal(1.0, nan.Point[0]);
    }
}
