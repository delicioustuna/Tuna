using Tuna.Differentiation;
using Tuna.Optimization.Stochastic;

namespace Tuna.Optimization.Tests.Optimization;

public class ParameterOptimizerTests
{
    [Theory]
    [InlineData("SGD", false)]
    [InlineData("SGD", true)]
    [InlineData("AdaGrad", false)]
    [InlineData("AdaGrad", true)]
    [InlineData("RMSProp", false)]
    [InlineData("RMSProp", true)]
    [InlineData("Adam", false)]
    [InlineData("Adam", true)]
    [InlineData("NAdam", false)]
    [InlineData("NAdam", true)]
    [InlineData("AdaMax", false)]
    [InlineData("AdaMax", true)]
    public void DoubleParameterUpdatesMatchTheFlatVector(string method, bool decoupled)
    {
        var weights = Parameter<double>.Matrix(2, 2);
        var bias = Parameter<double>.Vector(2);
        var parameters = new ParameterSet<double>(weights, Parameter<double>.Vector(0), bias);
        var (flat, grouped) = CreateOptimizers(method, decoupled, parameters);
        double[] initial = [1.1, -0.8, 0.3, -0.4, 0.5, 1.6];
        double[] values = new double[initial.Length];
        double[] gradient = new double[initial.Length];

        for (int run = 0; run < 2; run++)
        {
            flat.Reset();
            grouped.Reset();
            Assert.Equal(0, grouped.StepCount);
            initial.CopyTo(values, 0);
            initial.AsSpan(0, 4).CopyTo(weights.Value);
            initial.AsSpan(4).CopyTo(bias.Value);

            for (int step = 0; step < 12; step++)
            {
                for (int i = 0; i < gradient.Length; i++)
                {
                    gradient[i] = (i + 0.2 * step) * (i % 2 == 0 ? 1 : -1);
                }

                gradient.AsSpan(0, 4).CopyTo(weights.Gradient);
                gradient.AsSpan(4).CopyTo(bias.Gradient);
                Assert.Equal(flat.CurrentLearningRate, grouped.CurrentLearningRate);
                flat.Step(values, gradient);
                grouped.Step();

                Assert.Equal(flat.StepCount, grouped.StepCount);
                for (int i = 0; i < values.Length; i++)
                {
                    double actual = i < 4 ? weights.Value[i] : bias.Value[i - 4];
                    Assert.Equal(BitConverter.DoubleToInt64Bits(values[i]), BitConverter.DoubleToInt64Bits(actual));
                }

                Assert.True(gradient.AsSpan(0, 4).SequenceEqual(weights.Gradient));
                Assert.True(gradient.AsSpan(4).SequenceEqual(bias.Gradient));
            }
        }
    }

    [Fact]
    public void FloatClippingUsesOneNormAcrossTheSet()
    {
        var weight = Parameter<float>.Matrix(1, 1);
        var bias = Parameter<float>.Vector(1);
        var optimizer = new StochasticGradientDescent<float>(new ParameterSet<float>(weight, bias),
            new StochasticGradientDescentOptions { LearningRate = 0.5, GradientClipNorm = 5 });
        weight.Gradient[0] = 3e30f;
        bias.Gradient[0] = -4e30f;

        optimizer.Step();

        Assert.Equal(-1.5f, weight.Value[0]);
        Assert.Equal(2f, bias.Value[0]);
        Assert.Equal(3e30f, weight.Gradient[0]);
        Assert.Equal(-4e30f, bias.Gradient[0]);
    }

    [Theory]
    [InlineData(3e200, -4e200, 5e-300)]
    [InlineData(3e-200, -4e-200, 5e-250)]
    public void DoubleClippingHandlesOverflowAndUnderflowInTheSquaredNorm(double first, double second, double limit)
    {
        var weight = Parameter<double>.Matrix(1, 1);
        var bias = Parameter<double>.Vector(1);
        var optimizer = new StochasticGradientDescent<double>(new ParameterSet<double>(weight, bias),
            new StochasticGradientDescentOptions { LearningRate = 1, GradientClipNorm = limit });
        weight.Gradient[0] = first;
        bias.Gradient[0] = second;

        optimizer.Step();

        Assert.Equal(-0.6, weight.Value[0] / limit, 14);
        Assert.Equal(0.8, bias.Value[0] / limit, 14);
    }

    [Fact]
    public void InvalidScheduledRateLeavesTheWholeStepUntouched()
    {
        var weight = Parameter<float>.Vector(1);
        weight.Value[0] = 4;
        weight.Gradient[0] = 1;
        var optimizer = new Adam<float>(new ParameterSet<float>(weight),
            new AdamOptions { Schedule = new InvalidSchedule(), WeightDecay = 0.25, DecoupledWeightDecay = true });

        Assert.Throws<InvalidOperationException>(optimizer.Step);

        Assert.Equal(4, weight.Value[0]);
        Assert.Equal(1, weight.Gradient[0]);
        Assert.Equal(0, optimizer.StepCount);
    }

    [Theory]
    [InlineData("SGD", false)]
    [InlineData("SGD", true)]
    [InlineData("AdaGrad", false)]
    [InlineData("AdaGrad", true)]
    [InlineData("RMSProp", false)]
    [InlineData("RMSProp", true)]
    [InlineData("Adam", false)]
    [InlineData("Adam", true)]
    [InlineData("NAdam", false)]
    [InlineData("NAdam", true)]
    [InlineData("AdaMax", false)]
    [InlineData("AdaMax", true)]
    public void FloatOptimizersTrackDoubleUpdatesAndSupportDefaultNoClipping(string method, bool clipEnabled)
    {
        var wFloat = Parameter<float>.Matrix(2, 2);
        var bFloat = Parameter<float>.Vector(2);
        var pFloat = new ParameterSet<float>(wFloat, bFloat);

        var wDouble = Parameter<double>.Matrix(2, 2);
        var bDouble = Parameter<double>.Vector(2);
        var pDouble = new ParameterSet<double>(wDouble, bDouble);

        double? clip = clipEnabled ? 1.5 : null;
        var optFloat = CreateGenericOptimizer(method, pFloat, clip);
        var optDouble = CreateGenericOptimizer(method, pDouble, clip);

        double[] initial = [0.8, -0.5, 0.3, -0.7, 0.2, 1.1];
        for (int i = 0; i < 4; i++)
        {
            wDouble.Value[i] = initial[i];
            wFloat.Value[i] = (float)initial[i];
        }

        for (int i = 0; i < 2; i++)
        {
            bDouble.Value[i] = initial[4 + i];
            bFloat.Value[i] = (float)initial[4 + i];
        }

        for (int step = 0; step < 8; step++)
        {
            for (int i = 0; i < 4; i++)
            {
                double g = (0.3 + 0.1 * i + 0.05 * step) * (i % 2 == 0 ? 1 : -1);
                wDouble.Gradient[i] = g;
                wFloat.Gradient[i] = (float)g;
            }

            for (int i = 0; i < 2; i++)
            {
                double g = -0.4 + 0.2 * i + 0.03 * step;
                bDouble.Gradient[i] = g;
                bFloat.Gradient[i] = (float)g;
            }

            optFloat.Step();
            optDouble.Step();

            for (int i = 0; i < 4; i++)
            {
                Assert.True(float.IsFinite(wFloat.Value[i]));
                Assert.True(Math.Abs(wFloat.Value[i] - wDouble.Value[i]) < 1e-4, $"{method}: mismatch at w[{i}]");
            }

            for (int i = 0; i < 2; i++)
            {
                Assert.True(float.IsFinite(bFloat.Value[i]));
                Assert.True(Math.Abs(bFloat.Value[i] - bDouble.Value[i]) < 1e-4, $"{method}: mismatch at b[{i}]");
            }
        }

        optFloat.Reset();
        Assert.Equal(0, optFloat.StepCount);
    }

    [Fact]
    public void ValidationAndEdgeCasesAreHandled()
    {
        var valid = new ParameterSet<float>(Parameter<float>.Vector(2));

        // null パラメーター集合
        Assert.Throws<ArgumentNullException>(() => new Adam<float>(null!));
        Assert.Throws<ArgumentNullException>(() => new StochasticGradientDescent<float>(null!));

        // 非対応要素型
        Assert.Throws<NotSupportedException>(() => new Adam<Half>(new ParameterSet<Half>()));

        // float で 1.0f に丸まる減衰・モーメンタム係数を拒否する
        const double roundsToOneInFloat = 0.99999999;
        Assert.Throws<ArgumentOutOfRangeException>(() => new Adam<float>(valid, new AdamOptions { Beta1 = roundsToOneInFloat }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Adam<float>(valid, new AdamOptions { Beta2 = roundsToOneInFloat }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new NAdam<float>(valid, new NAdamOptions { Beta1 = roundsToOneInFloat }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new NAdam<float>(valid, new NAdamOptions { Beta2 = roundsToOneInFloat }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new AdaMax<float>(valid, new AdaMaxOptions { Beta1 = roundsToOneInFloat }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new AdaMax<float>(valid, new AdaMaxOptions { Beta2 = roundsToOneInFloat }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new RmsProp<float>(valid, new RmsPropOptions { Decay = roundsToOneInFloat }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new RmsProp<float>(valid, new RmsPropOptions { Momentum = roundsToOneInFloat }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new StochasticGradientDescent<float>(valid, new StochasticGradientDescentOptions { Momentum = roundsToOneInFloat }));

        // float で 0 / ∞ に丸まる Epsilon・InitialAccumulator
        Assert.Throws<ArgumentOutOfRangeException>(() => new Adam<float>(valid, new AdamOptions { Epsilon = 1e-50 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new AdaGrad<float>(valid, new AdaGradOptions { InitialAccumulator = 1e40 }));

        // LearningRate セッターとプロパティ
        var sgd = new StochasticGradientDescent<float>(valid, new StochasticGradientDescentOptions { Momentum = 0.5, Nesterov = true });
        Assert.Same(valid, sgd.Parameters);
        Assert.Equal(2, sgd.Dimension);
        Assert.Null(sgd.Schedule);
        Assert.Equal(0.5, sgd.Momentum);
        Assert.True(sgd.Nesterov);
        Assert.Throws<ArgumentOutOfRangeException>(() => sgd.LearningRate = 0);
        Assert.Throws<ArgumentOutOfRangeException>(() => sgd.LearningRate = double.NaN);

        // スケジュールが NaN や float オーバーフローを返す場合
        var nanOpt = new Adam<float>(valid, new AdamOptions { Schedule = new ConstantSchedule(double.NaN) });
        Assert.Throws<InvalidOperationException>(nanOpt.Step);
        var overflowOpt = new Adam<float>(valid, new AdamOptions { LearningRate = 1.0, Schedule = new ConstantSchedule(1e40) });
        Assert.Throws<InvalidOperationException>(overflowOpt.Step);

        // 空の ParameterSet<T>
        var empty = new ParameterSet<float>();
        var emptyOpt = new Adam<float>(empty);
        Assert.Equal(0, emptyOpt.Dimension);
        emptyOpt.Step();
        Assert.Equal(1, emptyOpt.StepCount);
        emptyOpt.Reset();
        Assert.Equal(0, emptyOpt.StepCount);

        // クリッピングの境界（全0勾配・上限以下・±Infinity・NaN）
        var p1 = Parameter<double>.Vector(1);
        var p2 = Parameter<double>.Vector(1);
        var clipOpt = new StochasticGradientDescent<double>(new ParameterSet<double>(p1, p2),
            new StochasticGradientDescentOptions { LearningRate = 1.0, GradientClipNorm = 5.0 });

        // 全 0 勾配
        p1.Value[0] = 2.0;
        p2.Value[0] = -3.0;
        p1.Gradient[0] = 0.0;
        p2.Gradient[0] = 0.0;
        clipOpt.Step();
        Assert.Equal(2.0, p1.Value[0]);
        Assert.Equal(-3.0, p2.Value[0]);

        // 上限以下（ノルム 2.5 <= 5.0）はクリップしない
        p1.Gradient[0] = 1.5;
        p2.Gradient[0] = -2.0;
        clipOpt.Step();
        Assert.Equal(0.5, p1.Value[0], 14);
        Assert.Equal(-1.0, p2.Value[0], 14);

        // NaN を含む場合はクリップせずそのまま伝播する
        p1.Gradient[0] = double.NaN;
        p2.Gradient[0] = 1.0;
        clipOpt.Step();
        Assert.True(double.IsNaN(p1.Value[0]));
        Assert.Equal(-2.0, p2.Value[0], 14);

        // ±Infinity を含む場合は有限成分が 0、無限大成分が NaN になる
        p1.Value[0] = 1.0;
        p2.Value[0] = 2.0;
        p1.Gradient[0] = double.PositiveInfinity;
        p2.Gradient[0] = 3.0;
        clipOpt.Step();
        Assert.True(double.IsNaN(p1.Value[0]));
        Assert.Equal(2.0, p2.Value[0]);
    }

    private sealed class InvalidSchedule : ILearningRateSchedule
    {
        public double Multiplier(long step) => -1;
    }

    private sealed class ConstantSchedule(double value) : ILearningRateSchedule
    {
        public double Multiplier(long step) => value;
    }

    private static StochasticGradientOptimizer<T> CreateGenericOptimizer<T>(string method, ParameterSet<T> parameters, double? clip)
        where T : unmanaged, System.Numerics.IFloatingPointIeee754<T> => method switch
        {
            "SGD" => new StochasticGradientDescent<T>(parameters, new StochasticGradientDescentOptions { LearningRate = 0.05, Momentum = 0.7, Nesterov = true, GradientClipNorm = clip }),
            "AdaGrad" => new AdaGrad<T>(parameters, new AdaGradOptions { LearningRate = 0.05, InitialAccumulator = 0.1, GradientClipNorm = clip }),
            "RMSProp" => new RmsProp<T>(parameters, new RmsPropOptions { LearningRate = 0.02, Centered = true, Momentum = 0.5, GradientClipNorm = clip }),
            "Adam" => new Adam<T>(parameters, new AdamOptions { LearningRate = 0.02, Amsgrad = true, GradientClipNorm = clip }),
            "NAdam" => new NAdam<T>(parameters, new NAdamOptions { LearningRate = 0.02, GradientClipNorm = clip }),
            "AdaMax" => new AdaMax<T>(parameters, new AdaMaxOptions { LearningRate = 0.02, GradientClipNorm = clip }),
            _ => throw new ArgumentOutOfRangeException(nameof(method)),
        };

    private static (StochasticGradientOptimizer, StochasticGradientOptimizer<double>) CreateOptimizers(
        string method, bool decoupled, ParameterSet<double> parameters)
    {
        const double clip = 2.5;
        const double decay = 0.1;
        const int length = 6;
        var schedule = new StepDecaySchedule(3, 0.5);
        switch (method)
        {
            case "SGD":
                var sgd = new StochasticGradientDescentOptions
                {
                    Momentum = 0.8,
                    Nesterov = true,
                    Schedule = schedule,
                    WeightDecay = decay,
                    GradientClipNorm = clip,
                    DecoupledWeightDecay = decoupled,
                };
                return (new StochasticGradientDescent(length, sgd), new StochasticGradientDescent<double>(parameters, sgd));
            case "AdaGrad":
                var adagrad = new AdaGradOptions
                {
                    InitialAccumulator = 0.125,
                    Schedule = schedule,
                    WeightDecay = decay,
                    GradientClipNorm = clip,
                    DecoupledWeightDecay = decoupled,
                };
                return (new AdaGrad(length, adagrad), new AdaGrad<double>(parameters, adagrad));
            case "RMSProp":
                var rms = new RmsPropOptions
                {
                    Centered = true,
                    Momentum = 0.8,
                    Schedule = schedule,
                    WeightDecay = decay,
                    GradientClipNorm = clip,
                    DecoupledWeightDecay = decoupled,
                };
                return (new RmsProp(length, rms), new RmsProp<double>(parameters, rms));
            case "Adam":
                var adam = new AdamOptions
                {
                    Amsgrad = true,
                    Schedule = schedule,
                    WeightDecay = decay,
                    GradientClipNorm = clip,
                    DecoupledWeightDecay = decoupled,
                };
                return (new Adam(length, adam), new Adam<double>(parameters, adam));
            case "NAdam":
                var nadam = new NAdamOptions
                {
                    Schedule = schedule,
                    WeightDecay = decay,
                    GradientClipNorm = clip,
                    DecoupledWeightDecay = decoupled,
                };
                return (new NAdam(length, nadam), new NAdam<double>(parameters, nadam));
            case "AdaMax":
                var adamax = new AdaMaxOptions
                {
                    Schedule = schedule,
                    WeightDecay = decay,
                    GradientClipNorm = clip,
                    DecoupledWeightDecay = decoupled,
                };
                return (new AdaMax(length, adamax), new AdaMax<double>(parameters, adamax));
            default:
                throw new ArgumentOutOfRangeException(nameof(method));
        }
    }
}
