using System.Numerics;
using System.Runtime.CompilerServices;
using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.Statistics.Distributions;

internal static class PointwiseLogLikelihood
{
    internal static ScalarNode RecordSum<T>(ReadOnlyVectorSpan<T> observations, EvaluateLikelihood<T> evaluate, ValidateLikelihood validate,
        ScalarSetNode<double> first, ScalarSetNode<double> second = default, ScalarSetNode<double> third = default,
        [CallerArgumentExpression(nameof(first))] string firstName = "",
        [CallerArgumentExpression(nameof(second))] string secondName = "",
        [CallerArgumentExpression(nameof(third))] string thirdName = "")
        where T : unmanaged, INumberBase<T> =>
        RecordSum(observations, new DirectEvaluator<T>(evaluate, validate), first, second, third, firstName, secondName, thirdName);

    internal static VectorNode<double> RecordVector<T>(ReadOnlyVectorSpan<T> observations, EvaluateLikelihood<T> evaluate, ValidateLikelihood validate,
        ScalarSetNode<double> first, ScalarSetNode<double> second = default, ScalarSetNode<double> third = default, Tape? tape = null,
        [CallerArgumentExpression(nameof(first))] string firstName = "",
        [CallerArgumentExpression(nameof(second))] string secondName = "",
        [CallerArgumentExpression(nameof(third))] string thirdName = "")
        where T : unmanaged, INumberBase<T> =>
        RecordVector(observations, new DirectEvaluator<T>(evaluate, validate), first, second, third, tape, firstName, secondName, thirdName);

    internal static ScalarNode RecordSum<T, TEvaluator>(ReadOnlyVectorSpan<T> observations, TEvaluator evaluate,
        ScalarSetNode<double> first, ScalarSetNode<double> second = default, ScalarSetNode<double> third = default,
        [CallerArgumentExpression(nameof(first))] string firstName = "",
        [CallerArgumentExpression(nameof(second))] string secondName = "",
        [CallerArgumentExpression(nameof(third))] string thirdName = "")
        where T : unmanaged, INumberBase<T>
        where TEvaluator : ILikelihoodEvaluator<T>, allows ref struct
    {
        Tape? tape = null;
        int length = observations.Length;
        var a = first.Bind(length, ref tape, firstName);
        var b = second.Bind(length, ref tape, secondName);
        var c = third.Bind(length, ref tape, thirdName);
        if (length == 0)
        {
            evaluate.ValidateShared(a.SharedValue, b.SharedValue, c.SharedValue);
        }

        var operation = new LikelihoodOperation<T, TEvaluator>(observations, evaluate, a, b, c);
        return tape is null
            ? operation.Forward(new(sum: true))
            : tape.RecordPointwiseSum<double, LikelihoodOperation<T, TEvaluator>>(length, [a.Input, b.Input, c.Input], operation);
    }

    internal static VectorNode<double> RecordVector<T, TEvaluator>(ReadOnlyVectorSpan<T> observations, TEvaluator evaluate,
        ScalarSetNode<double> first, ScalarSetNode<double> second = default, ScalarSetNode<double> third = default, Tape? tape = null,
        [CallerArgumentExpression(nameof(first))] string firstName = "",
        [CallerArgumentExpression(nameof(second))] string secondName = "",
        [CallerArgumentExpression(nameof(third))] string thirdName = "")
        where T : unmanaged, INumberBase<T>
        where TEvaluator : ILikelihoodEvaluator<T>, allows ref struct
    {
        int length = observations.Length;
        var a = first.Bind(length, ref tape, firstName);
        var b = second.Bind(length, ref tape, secondName);
        var c = third.Bind(length, ref tape, thirdName);
        if (length == 0)
        {
            evaluate.ValidateShared(a.SharedValue, b.SharedValue, c.SharedValue);
        }

        if (tape is null)
        {
            throw new ArgumentException("Supply a tape when all parameters are scalar constants.", nameof(tape));
        }

        var operation = new LikelihoodOperation<T, TEvaluator>(observations, evaluate, a, b, c);
        return tape.RecordPointwiseVector<double, LikelihoodOperation<T, TEvaluator>>(length, [a.Input, b.Input, c.Input], operation);
    }

    internal static LikelihoodEvaluation Density<T>(T distribution, double observation, LikelihoodGradients gradients,
        Func<T, double, double, LogProbability, (double First, double Second, double X)> gradient)
        where T : struct, IContinuousDistribution
    {
        double value = distribution.LogPdf(observation);
        if (gradients == LikelihoodGradients.None)
        {
            return new(value);
        }

        var (first, second, _) = gradient(distribution, observation, value, LogProbability.Density);
        return new(value, new(first), new(second));
    }

    internal static LikelihoodEvaluation Mass<T>(T distribution, long observation, LikelihoodGradients gradients,
        Func<T, long, double, LogProbability, (double First, double Second)> gradient)
        where T : struct, IDiscreteDistribution
    {
        double value = distribution.LogPmf(observation);
        if (gradients == LikelihoodGradients.None)
        {
            return new(value);
        }

        var (first, second) = gradient(distribution, observation, value, LogProbability.Density);
        return new(value, new(first), new(second));
    }

    private readonly struct DirectEvaluator<T>(EvaluateLikelihood<T> evaluate, ValidateLikelihood validate) : ILikelihoodEvaluator<T>
    {
        public LikelihoodEvaluation Evaluate(int index, T observation, double first, double second, double third, LikelihoodGradients gradients) =>
            evaluate(observation, first, second, third, gradients);

        public void ValidateShared(double? first, double? second, double? third) => validate(first, second, third);
    }

    private readonly ref struct LikelihoodOperation<T, TEvaluator>(ReadOnlyVectorSpan<T> observations, TEvaluator evaluate,
        BoundScalarSet<double> first, BoundScalarSet<double> second, BoundScalarSet<double> third) : IPointwiseOperation<double>
        where T : unmanaged, INumberBase<T>
        where TEvaluator : ILikelihoodEvaluator<T>, allows ref struct
    {
        private readonly ReadOnlyVectorSpan<T> points = observations;
        private readonly TEvaluator evaluator = evaluate;
        private readonly BoundScalarSet<double> firstParameter = first;
        private readonly BoundScalarSet<double> secondParameter = second;
        private readonly BoundScalarSet<double> thirdParameter = third;

        public double Forward(scoped PointwiseOutput<double> output)
        {
            Span<double> values = output.Values;
            var a = new SavedDerivative(output.Derivatives(0), output.Sum && firstParameter.SharedValue.HasValue);
            var b = new SavedDerivative(output.Derivatives(1), output.Sum && secondParameter.SharedValue.HasValue);
            var c = new SavedDerivative(output.Derivatives(2), output.Sum && thirdParameter.SharedValue.HasValue);
            LikelihoodGradients gradients = LikelihoodGradients.None;
            if (a.Required)
            {
                gradients |= LikelihoodGradients.First;
            }

            if (b.Required)
            {
                gradients |= LikelihoodGradients.Second;
            }

            if (c.Required)
            {
                gradients |= LikelihoodGradients.Third;
            }

            double sum = 0;
            for (int i = 0; i < points.Length; i++)
            {
                var point = evaluator.Evaluate(i, points[i], firstParameter[i], secondParameter[i], thirdParameter[i], gradients);
                if (output.Sum)
                {
                    sum += point.Value;
                }
                else
                {
                    values[i] = point.Value;
                }

                a.Add(i, point.First);
                b.Add(i, point.Second);
                c.Add(i, point.Third);
            }

            a.Complete();
            b.Complete();
            c.Complete();
            return sum;
        }
    }

    private ref struct SavedDerivative(Span<double> values, bool reduce)
    {
        private readonly Span<double> buffer = values;
        private ScaledDerivativeSum sum;

        internal readonly bool Required => !buffer.IsEmpty;

        internal void Add(int index, ScaledDerivative derivative)
        {
            if (!Required)
            {
                return;
            }

            if (reduce)
            {
                sum.Add(derivative);
            }
            else
            {
                buffer[index] = derivative.Value;
            }
        }

        internal readonly void Complete()
        {
            if (reduce && Required)
            {
                buffer[0] = sum.Value;
            }
        }
    }
}

internal delegate LikelihoodEvaluation EvaluateLikelihood<T>(T observation, double first, double second, double third, LikelihoodGradients gradients);

internal delegate void ValidateLikelihood(double? first, double? second, double? third);

internal interface ILikelihoodEvaluator<T>
{
    LikelihoodEvaluation Evaluate(int index, T observation, double first, double second, double third, LikelihoodGradients gradients);

    void ValidateShared(double? first, double? second, double? third);
}

[Flags]
internal enum LikelihoodGradients
{
    None = 0,
    First = 1,
    Second = 2,
    Third = 4,
}

internal readonly record struct LikelihoodEvaluation(
    double Value, ScaledDerivative First = default, ScaledDerivative Second = default, ScaledDerivative Third = default);
