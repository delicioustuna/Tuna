using System.Numerics;
using Tuna.LinearAlgebra;

namespace Tuna.Differentiation;

internal sealed partial class ArrayStore<T>
{
    private void ActivationBackward(in ArrayRecord record)
    {
        int output = record.Output;
        ArrayView x = record.Left;
        if (!Touched(output) || !RequiresGradient(x.Slot))
        {
            return;
        }

        int n = x.Length;
        ReadOnlySpan<T> dy = gradients[output].AsSpan(0, n);
        ReadOnlySpan<T> values = LogicalValues(x, LeftValues);
        Span<T> d = Scratch(LeftDerivative, n);
        switch (record.Operation)
        {
            case ArrayOperation.Relu:
                ElementwiseKernels<T>.Map<ReluDerivative<T>>(values, d);
                break;
            case ArrayOperation.LeakyRelu:
                ElementwiseKernels<T>.Map<LeakyReluDerivative<T>>(values, FromDouble(record.Scalar), d);
                break;
            case ArrayOperation.Elu:
                ExponentialLinearDerivatives(values, FromDouble(record.Scalar), T.One, d);
                break;
            case ArrayOperation.Selu:
                ExponentialLinearDerivatives(values, FromDouble(ActivationDerivatives.SeluScaleAlpha), FromDouble(ActivationDerivatives.SeluScale), d);
                break;
            case ArrayOperation.Silu:
                SiluDerivatives(values, d);
                break;
            default:
                GeluDerivatives(record.Operation == ArrayOperation.GeluTanh, values, dy, d);
                break;
        }

        AddMaskedProduct(x, dy, d);
    }

    private void ExponentialLinearDerivatives(ReadOnlySpan<T> x, T a, T p, Span<T> d)
    {
        var (shift, coefficient) = ExponentialLinearTail(a);
        Span<T> t = Scratch(RightDerivative, d.Length);
        ElementwiseKernels<T>.Map<ExponentialLinearArgument<T>>(x, shift, t);
        XMath.Exp(new ReadOnlyVectorSpan<T>(t), new VectorSpan<T>(t));
        ElementwiseKernels<T>.Map<ExponentialLinearDerivative<T>>(x, t, a, coefficient, p, d);
    }

    // Computes integer shift s and coefficient c for the negative tail (x < Below) so that a * exp(x) = (c * exp(x + s)) * exp(-64).
    // When |a| > exp(27), adds k = min(ceil(ln|a|), 704 [double] / 64 [float]) so exp(x + s) stays normal whenever a * exp(x) is non-zero.
    private static (T Shift, T Coefficient) ExponentialLinearTail(T a)
    {
        double alpha = double.CreateTruncating(a);
        double log = Math.Log(Math.Abs(alpha));
        if (!(log > 27))
        {
            return (ActivationTail<T>.Shift, a);
        }

        double k = Math.Min(Math.Ceiling(log), typeof(T) == typeof(float) ? 64 : 704);
        return (T.CreateTruncating(ActivationDerivatives.TailShift + k), T.CreateTruncating(alpha * Math.Exp(-k)));
    }

    private void SiluDerivatives(ReadOnlySpan<T> x, Span<T> d)
    {
        Span<T> t = Scratch(RightDerivative, d.Length);
        ElementwiseKernels<T>.Map<SiluArgument<T>>(x, t);
        XMath.Exp(new ReadOnlyVectorSpan<T>(t), new VectorSpan<T>(t));
        ElementwiseKernels<T>.Map<SiluDerivative<T>>(x, t, d);
    }

    private static void GeluDerivatives(bool tanh, ReadOnlySpan<T> x, ReadOnlySpan<T> dy, Span<T> d)
    {
        for (int i = 0; i < d.Length; i++)
        {
            if (T.IsZero(dy[i]))
            {
                d[i] = T.Zero;
                continue;
            }

            double v = double.CreateTruncating(x[i]);
            d[i] = FromDouble(tanh ? ActivationDerivatives.GeluTanh(v) : ActivationDerivatives.Gelu(v));
        }
    }
}
