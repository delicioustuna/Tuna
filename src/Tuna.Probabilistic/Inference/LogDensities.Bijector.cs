using System.Buffers;
using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.Probabilistic.Inference;

public static partial class LogDensities
{
    /// <summary>
    /// Records a constrained log density and the transform's log Jacobian as one operation.
    /// <para>制約付き対数密度と制約変換の対数ヤコビアンを 1 回の演算として記録します。</para>
    /// </summary>
    /// <param name="density">The differentiable density in the constrained coordinates. / 制約付き座標での勾配付き対数密度。</param>
    /// <param name="bijector">The transform to the constrained coordinates. / 制約付き座標への変換。</param>
    /// <param name="unconstrained">The input nodes in the unconstrained coordinates. / 制約のない座標の入力ノード。</param>
    /// <returns>The transformed log density node. / 変換後の対数密度を表すノード。</returns>
    /// <exception cref="ArgumentNullException">The density or bijector is null. / 対数密度または制約変換が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException">The number of nodes does not match the input dimension. / ノードの数が入力次元と一致しない場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode RecordTransformed(IDifferentiableLogDensity density, IBijector bijector, ReadOnlySpan<ScalarNode> unconstrained)
    {
        var transformed = new TransformedDifferentiableLogDensity<IDifferentiableLogDensity, IBijector>(density, bijector);
        int count = unconstrained.Length;
        Bijection.RequireLengths(bijector, count, bijector.ConstrainedDimension);
        int length = checked(2 * count);
        double[]? rented = length > Bijection.StackLimit ? ArrayPool<double>.Shared.Rent(length) : null;
        Span<double> working = rented is null ? stackalloc double[length] : rented.AsSpan(0, length);
        try
        {
            bool requiresGradient = false;
            for (int i = 0; i < count; i++)
            {
                working[i] = unconstrained[i].Value;
                requiresGradient |= unconstrained[i].RequiresGradient;
            }

            var point = new ReadOnlyVectorSpan<double>(working[..count]);
            if (!requiresGradient)
            {
                return transformed.LogDensity(point);
            }

            Span<double> gradient = working[count..];
            double value = transformed.LogDensity(point, new VectorSpan<double>(gradient));
            var pairs = ArrayPool<(ScalarNode Input, double Derivative)>.Shared.Rent(count);
            try
            {
                for (int i = 0; i < count; i++)
                {
                    pairs[i] = (unconstrained[i], gradient[i]);
                }

                return ScalarNode.Record(value, pairs.AsSpan(0, count));
            }
            finally
            {
                Array.Clear(pairs, 0, count);
                ArrayPool<(ScalarNode Input, double Derivative)>.Shared.Return(pairs);
            }
        }
        finally
        {
            if (rented is not null)
            {
                ArrayPool<double>.Shared.Return(rented);
            }
        }
    }
}
