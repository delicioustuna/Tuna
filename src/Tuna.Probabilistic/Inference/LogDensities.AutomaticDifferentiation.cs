using System.Collections.Concurrent;
using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.Probabilistic.Inference;

public static partial class LogDensities
{
    /// <summary>
    /// Creates a log density that differentiates a scalar-node model when evaluated.
    /// <para>スカラーノードで記述したモデルから、勾配付き対数密度を作成します。</para>
    /// </summary>
    /// <param name="model">The log density model, called with the parameter nodes in input order. / 入力順のパラメータノードから対数密度を求めるモデル。</param>
    /// <returns>The log density for use by a sampler. / サンプラーに渡せる対数密度。</returns>
    /// <remarks>
    /// <para>Parameter nodes are valid only during evaluation. Concurrent evaluations use separate tapes. The model must support concurrent calls when shared by parallel chains.</para>
    /// <para>パラメータノードは評価中だけ有効です。同時評価には別のテープを使います。並列チェーンで共有するモデルは同時呼び出しに対応してください。</para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="model"/> is null. / <paramref name="model"/> が <see langword="null"/> の場合。</exception>
    public static IDifferentiableLogDensity FromAutomaticDifferentiation(Func<ReadOnlySpan<ScalarNode>, ScalarNode> model)
    {
        ArgumentNullException.ThrowIfNull(model);
        return new AutomaticLogDensity((tape, theta) => model(tape.Leaves(theta)));
    }

    /// <summary>
    /// Creates a log density that differentiates a vector-node model when evaluated.
    /// <para>ベクトルノードで記述したモデルから、勾配付き対数密度を作成します。</para>
    /// </summary>
    /// <param name="model">The log density model, called with the current tape and the parameter vector node. / 評価用テープとパラメータを表すベクトルノードから対数密度を求めるモデル。</param>
    /// <returns>The log density for use by a sampler. / サンプラーに渡せる対数密度。</returns>
    /// <remarks>
    /// <para>The supplied vector is the parameter leaf. Use the tape to record constants and intermediate results. Concurrent evaluations use separate tapes, and the model must support concurrent calls when shared by parallel chains.</para>
    /// <para>渡されたベクトルノードがパラメータの入力ノード（葉）になります。テープは定数や中間結果の記録に使えます。同時評価には別のテープを使うため、並列チェーンで共有するモデルは同時呼び出しに対応してください。</para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="model"/> is null. / <paramref name="model"/> が <see langword="null"/> の場合。</exception>
    public static IDifferentiableLogDensity FromAutomaticDifferentiation(Func<Tape, VectorNode<double>, ScalarNode> model)
    {
        ArgumentNullException.ThrowIfNull(model);
        return new AutomaticLogDensity((tape, theta) => model(tape, tape.Leaf(theta)));
    }

    private sealed class AutomaticLogDensity(Func<Tape, ReadOnlyVectorSpan<double>, ScalarNode> record) : IDifferentiableLogDensity
    {
        private readonly ConcurrentBag<Tape> tapes = new();

        public double LogDensity(ReadOnlyVectorSpan<double> theta) => Evaluate(theta, default, requiresGradient: false);

        public double LogDensity(ReadOnlyVectorSpan<double> theta, VectorSpan<double> gradient)
        {
            if (theta.Length != gradient.Length)
            {
                throw new ArgumentException("Parameter and gradient lengths must agree.", nameof(gradient));
            }

            return Evaluate(theta, gradient, requiresGradient: true);
        }

        private double Evaluate(ReadOnlyVectorSpan<double> theta, VectorSpan<double> gradient, bool requiresGradient)
        {
            if (!tapes.TryTake(out Tape? tape))
            {
                tape = new Tape();
            }

            try
            {
                ScalarNode value = record(tape, theta);
                return requiresGradient ? tape.Gradient(value, gradient) : value.Value;
            }
            finally
            {
                tape.Reset();
                tapes.Add(tape);
            }
        }
    }
}
