using System.Numerics;
using Tuna.LinearAlgebra;

namespace Tuna.Differentiation;

internal sealed partial class ArrayStore<T>
{
    // When keep >= 0, stores P - onehot(labels) in slot keep for backward reuse.
    public double SoftmaxCrossEntropy(in ArrayView z, ReadOnlySpan<int> labels, int keep)
    {
        int classes = z.Rows, samples = z.Columns;
        Span<T> logSoftmax = keep >= 0 ? Values(keep) : Scratch(LeftValues, checked(classes * samples));
        var matrix = new MatrixSpan<T>(logSoftmax, classes, samples);
        XMath.ColumnLogSoftmax(MatrixView(z), matrix);
        Span<T> losses = Scratch(ReducedValues, samples);
        LossKernels<T>.LabelLosses(logSoftmax, labels, classes, losses);
        double value = double.CreateTruncating(XMath.Sum(new ReadOnlyVectorSpan<T>(losses)));
        if (keep >= 0)
        {
            LossWeights(MatrixView(z), logSoftmax, logSoftmax);
            LossKernels<T>.SubtractLabels(logSoftmax, labels, classes);
        }

        return value;
    }

    // When keep >= 0, retains L = ColumnLogSoftmax(z) in slot keep for backward reuse.
    public double SoftmaxCrossEntropy(in ArrayView z, in ArrayView t, int keep)
    {
        int classes = z.Rows, samples = z.Columns;
        int length = checked(classes * samples);
        Span<T> logSoftmax = keep >= 0 ? Values(keep) : Scratch(LeftValues, length);
        XMath.ColumnLogSoftmax(MatrixView(z), new MatrixSpan<T>(logSoftmax, classes, samples));
        ReadOnlySpan<T> targets = LogicalValues(t, RightValues);
        Span<T> terms = Scratch(Contribution, length);
        LossKernels<T>.TargetTerms(targets, logSoftmax, terms);
        Span<T> losses = Scratch(ReducedValues, samples);
        XMath.ColumnSums(new ReadOnlyMatrixSpan<T>(terms, classes, samples), new VectorSpan<T>(losses));
        return double.CreateTruncating(XMath.Sum(new ReadOnlyVectorSpan<T>(losses)));
    }

    public double LogisticCrossEntropy(in ArrayView eta, in ArrayView y, bool matrix)
    {
        int length = checked(eta.Rows * eta.Columns);
        ReadOnlySpan<T> logits = LogicalValues(eta, LeftValues);
        ReadOnlySpan<T> targets = LogicalValues(y, RightValues);
        Span<T> losses = Scratch(Contribution, length);
        LossKernels<T>.NegatedMagnitudes(logits, losses);
        var vector = new VectorSpan<T>(losses);
        XMath.Softplus(vector.AsReadOnly(), vector);
        LossKernels<T>.LogisticLosses(logits, targets, losses);
        if (!matrix)
        {
            return double.CreateTruncating(XMath.Sum(vector.AsReadOnly()));
        }

        Span<T> sums = Scratch(ReducedValues, eta.Columns);
        XMath.ColumnSums(new ReadOnlyMatrixSpan<T>(losses, eta.Rows, eta.Columns), new VectorSpan<T>(sums));
        return double.CreateTruncating(XMath.Sum(new ReadOnlyVectorSpan<T>(sums)));
    }

    // Reuses P = exp(L) instead of recomputing softmax; repairs only columns with non-finite maxima (where L produces NaN) via LimitWeights.
    private static void LossWeights(ReadOnlyMatrixSpan<T> z, ReadOnlySpan<T> logSoftmax, Span<T> weights)
    {
        int classes = z.Rows, samples = z.Columns;
        XMath.Exp(new ReadOnlyMatrixSpan<T>(logSoftmax, classes, samples), new MatrixSpan<T>(weights, classes, samples));
        if (!LossKernels<T>.ContainsNaN(weights))
        {
            return;
        }

        for (int j = 0; j < samples; j++)
        {
            Span<T> column = weights.Slice(j * classes, classes);
            if (LossKernels<T>.ContainsNaN(column))
            {
                LimitWeights(z.Column(j), new VectorSpan<T>(column));
            }
        }
    }

    private void LossBackward(in ArrayRecord record, double adjoint)
    {
        T g = FromDouble(adjoint);
        if (T.IsZero(g))
        {
            return;
        }

        switch (record.Operation)
        {
            case ArrayOperation.SoftmaxCrossEntropyLabels:
                // Cached P - onehot(labels) was written to ScalarInput during the forward pass.
                AddScaledMatrix(record.Left, g, Values(record.ScalarInput));
                break;
            case ArrayOperation.SoftmaxCrossEntropyTargets:
                SoftmaxCrossEntropyBackward(record, g);
                break;
            default:
                LogisticCrossEntropyBackward(record, g);
                break;
        }
    }

    private void SoftmaxCrossEntropyBackward(in ArrayRecord record, T g)
    {
        ArrayView z = record.Left, t = record.Right;
        int classes = z.Rows, samples = z.Columns;
        ReadOnlySpan<T> logSoftmax = Values(record.ScalarInput);
        if (RequiresGradient(z.Slot))
        {
            ReadOnlyMatrixSpan<T> targets = MatrixView(t);
            Span<T> sums = Scratch(ReducedValues, samples);
            XMath.ColumnSums(targets, new VectorSpan<T>(sums));
            Span<T> scratch = Scratch(LeftDerivative, logSoftmax.Length);
            LossWeights(MatrixView(z), logSoftmax, scratch);
            var factor = new MatrixSpan<T>(scratch, classes, samples);
            XMath.Hadamard(factor.AsReadOnly(), Repeat(sums, columns: true, classes, samples), factor);
            XMath.Subtract(factor.AsReadOnly(), targets, factor);
            AddScaledMatrix(z, g, scratch);
        }

        if (RequiresGradient(t.Slot))
        {
            AddScaledMatrix(t, -g, logSoftmax);
        }
    }

    // Evaluates (1 - y) * sigma(eta) - y * sigma(-eta) via p = sigma(-|eta|) to avoid tail cancellation when y is 0 or 1.
    private void LogisticCrossEntropyBackward(in ArrayRecord record, T g)
    {
        ArrayView eta = record.Left, y = record.Right;
        ReadOnlySpan<T> logits = LogicalValues(eta, LeftValues);
        if (RequiresGradient(eta.Slot))
        {
            ReadOnlySpan<T> targets = LogicalValues(y, RightValues);
            Span<T> d = Scratch(LeftDerivative, logits.Length);
            LossKernels<T>.NegatedMagnitudes(logits, d);
            XMath.Logistic(new ReadOnlyVectorSpan<T>(d), new VectorSpan<T>(d));
            LossKernels<T>.LogisticDerivatives(logits, targets, d);
            AddScaledMatrix(eta, g, d);
        }

        if (RequiresGradient(y.Slot))
        {
            AddScaledMatrix(y, -g, logits);
        }
    }

    private void AddScaledMatrix(in ArrayView target, T g, ReadOnlySpan<T> source)
    {
        if (IsContiguous(target))
        {
            Span<T> gradient = BeginGradient(target, out bool overwrite);
            ElementwiseKernels<T>.AddScaled(source, g, gradient.Slice(Offset(target), target.Length), overwrite);
            return;
        }

        Span<T> contribution = Scratch(Contribution, target.Length);
        ElementwiseKernels<T>.AddScaled(source, g, contribution, overwrite: true);
        AddGradient(target, contribution, negate: false);
    }
}
