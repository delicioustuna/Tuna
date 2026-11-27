using System.Numerics;
using System.Runtime.CompilerServices;
using Tuna.LinearAlgebra;

namespace Tuna.Differentiation;

internal sealed partial class ArrayStore<T>
{
    private const int LeftValues = 0;
    private const int RightValues = 1;
    private const int LeftDerivative = 2;
    private const int Contribution = 3;
    private const int RightDerivative = 4;
    private const int Reduction = 5;
    internal const int ReducedValues = 6;
    // Scratch slots shared with LinearAlgebraBackward (never live concurrently with elementwise ops).
    internal const int SolveAdjoint = 7;
    internal const int BlockFirst = 8;
    internal const int BlockSecond = LeftDerivative;
    internal const int MatrixContribution = Contribution;
    internal const int DecompositionFirst = SolveAdjoint;
    internal const int DecompositionSecond = BlockFirst;
    internal const int DecompositionThird = LeftDerivative;
    internal const int DecompositionFourth = RightDerivative;
    internal const int DecompositionFifth = LeftValues;

    // Relies on ArrayOperation enum grouping: unary, activations, binary, scalar-operand, reductions/softmax, losses, linear algebra.
    private void ElementwiseBackward(in ArrayRecord record, double[] adjoints)
    {
        ArrayOperation operation = record.Operation;
        if (operation <= ArrayOperation.Reciprocal)
        {
            UnaryBackward(record);
        }
        else if (operation <= ArrayOperation.Silu)
        {
            ActivationBackward(record);
        }
        else if (operation <= ArrayOperation.LogSubExp)
        {
            BinaryBackward(record);
        }
        else if (operation <= ArrayOperation.ScalarSubtract)
        {
            ScalarOperandBackward(record, adjoints);
        }
        else if (operation <= ArrayOperation.RowLogSoftmax)
        {
            ReductionBackward(record, adjoints);
        }
        else if (operation <= ArrayOperation.LogisticCrossEntropy)
        {
            LossBackward(record, adjoints[record.Output]);
        }
        else
        {
            LinearAlgebraBackward.Backward(Unsafe.As<ArrayStore<double>>(this), record, adjoints);
        }
    }

    private void UnaryBackward(in ArrayRecord record)
    {
        int output = record.Output;
        ArrayView x = record.Left;
        if (!Touched(output) || !RequiresGradient(x.Slot))
        {
            return;
        }

        int n = x.Length;
        ReadOnlySpan<T> dy = gradients[output].AsSpan(0, n);
        ReadOnlySpan<T> y = Values(output).Slice(0, n);
        if (record.Operation == ArrayOperation.Negate)
        {
            AddGradient(x, dy, negate: true);
            return;
        }

        if (record.Operation == ArrayOperation.Exp)
        {
            AddMaskedProduct(x, dy, y);
            return;
        }

        if (record.Operation == ArrayOperation.Reciprocal)
        {
            Span<T> reciprocal = Scratch(LeftDerivative, n);
            ElementwiseKernels<T>.Map<ReciprocalDerivative<T>>(y, reciprocal);
            AddMaskedProduct(x, dy, reciprocal);
            return;
        }

        ReadOnlySpan<T> values = LogicalValues(x, LeftValues);
        Span<T> d = Scratch(LeftDerivative, n);
        var vector = new ReadOnlyVectorSpan<T>(values);
        var derivative = new VectorSpan<T>(d);
        switch (record.Operation)
        {
            case ArrayOperation.Log:
                ElementwiseKernels<T>.Map<LogDerivative<T>>(values, d);
                break;
            case ArrayOperation.Sqrt:
                ElementwiseKernels<T>.Map<SqrtDerivative<T>>(values, y, d);
                break;
            case ArrayOperation.Square:
                ElementwiseKernels<T>.Map<TwiceKernel<T>>(values, d);
                break;
            case ArrayOperation.Abs:
                ElementwiseKernels<T>.Map<AbsDerivative<T>>(values, d);
                break;
            case ArrayOperation.Expm1:
                XMath.Exp(vector, derivative);
                break;
            case ArrayOperation.Log1p:
                ElementwiseKernels<T>.Map<Log1pDerivative<T>>(values, d);
                break;
            case ArrayOperation.Tanh:
                ElementwiseKernels<T>.Map<NegativeAbsKernel<T>>(values, d);
                XMath.Exp(derivative.AsReadOnly(), derivative);
                ElementwiseKernels<T>.Map<TanhDerivativeFromExp<T>>(d, d);
                break;
            case ArrayOperation.Logistic:
                ElementwiseKernels<T>.Map<NegativeAbsKernel<T>>(values, d);
                XMath.Exp(derivative.AsReadOnly(), derivative);
                ElementwiseKernels<T>.Map<LogisticDerivativeFromExp<T>>(d, d);
                break;
            case ArrayOperation.LogLogistic:
                XMath.Negate(vector, derivative);
                XMath.Logistic(derivative.AsReadOnly(), derivative);
                break;
            case ArrayOperation.Logit:
                ElementwiseKernels<T>.Map<LogitDerivative<T>>(values, d);
                break;
            case ArrayOperation.Softplus:
                XMath.Logistic(vector, derivative);
                break;
            case ArrayOperation.Log1mExp:
                Span<T> m = Scratch(RightDerivative, n);
                var expm1 = new VectorSpan<T>(m);
                XMath.Exp(vector, derivative);
                XMath.Negate(vector, expm1);
                XMath.Expm1(expm1.AsReadOnly(), expm1);
                ElementwiseKernels<T>.Map<Log1mExpDerivative<T>>(values, d, m, d);
                break;
        }

        AddMaskedProduct(x, dy, d);
    }

    private void BinaryBackward(in ArrayRecord record)
    {
        int output = record.Output;
        if (!Touched(output))
        {
            return;
        }

        ArrayView x = record.Left, y = record.Right;
        bool left = RequiresGradient(x.Slot), right = RequiresGradient(y.Slot);
        int n = x.Length;
        ReadOnlySpan<T> dz = gradients[output].AsSpan(0, n);
        switch (record.Operation)
        {
            case ArrayOperation.Add:
            case ArrayOperation.Subtract:
                if (left)
                {
                    AddGradient(x, dz, negate: false);
                }

                if (right)
                {
                    AddGradient(y, dz, negate: record.Operation == ArrayOperation.Subtract);
                }

                return;
            case ArrayOperation.Hadamard:
                if (left)
                {
                    AddMaskedProduct(x, dz, LogicalValues(y, RightValues));
                }

                if (right)
                {
                    AddMaskedProduct(y, dz, LogicalValues(x, LeftValues));
                }

                return;
            case ArrayOperation.Divide:
                DivideBackward(x, y, left, right, dz, Values(output).Slice(0, n));
                return;
        }

        ReadOnlySpan<T> xValues = LogicalValues(x, LeftValues), yValues = LogicalValues(y, RightValues);
        Span<T> dx = left ? Scratch(LeftDerivative, n) : default;
        Span<T> dy = right ? Scratch(RightDerivative, n) : default;
        switch (record.Operation)
        {
            case ArrayOperation.Min:
                MinMaxDerivatives<MinLeftDerivative<T>, MinRightDerivative<T>>(xValues, yValues, left, right, dx, dy);
                break;
            case ArrayOperation.Max:
                MinMaxDerivatives<MaxLeftDerivative<T>, MaxRightDerivative<T>>(xValues, yValues, left, right, dx, dy);
                break;
            case ArrayOperation.Pow:
                PowDerivatives(xValues, yValues, Values(output).Slice(0, n), dz, left, right, dx, dy);
                break;
            case ArrayOperation.Hypot:
                ReadOnlySpan<T> norm = Values(output).Slice(0, n);
                if (left)
                {
                    ElementwiseKernels<T>.Map<HypotDerivative<T>>(xValues, yValues, norm, dx);
                }

                if (right)
                {
                    ElementwiseKernels<T>.Map<HypotDerivative<T>>(yValues, xValues, norm, dy);
                }

                break;
            case ArrayOperation.LogAddExp:
            case ArrayOperation.LogSubExp:
                LogExpDerivatives(record.Operation == ArrayOperation.LogSubExp, xValues, yValues, dz, left, right, dx, dy);
                break;
        }

        if (left)
        {
            AddMaskedProduct(x, dz, dx);
        }

        if (right)
        {
            AddMaskedProduct(y, dz, dy);
        }
    }

    private void DivideBackward(in ArrayView x, in ArrayView y, bool left, bool right, ReadOnlySpan<T> dz, ReadOnlySpan<T> z)
    {
        int n = dz.Length;
        ReadOnlySpan<T> denominator = LogicalValues(y, RightValues);
        if (left)
        {
            Span<T> d = Scratch(LeftDerivative, n);
            ElementwiseKernels<T>.Map<InverseKernel<T>>(denominator, d);
            AddMaskedProduct(x, dz, d);
        }

        if (right)
        {
            Span<T> d = Scratch(RightDerivative, n);
            ElementwiseKernels<T>.DenominatorDerivatives(LogicalValues(x, LeftValues), denominator, z, d);
            AddMaskedProduct(y, dz, d);
        }
    }

    private void ScalarOperandBackward(in ArrayRecord record, double[] adjoints)
    {
        int output = record.Output;
        if (!Touched(output))
        {
            return;
        }

        ArrayView x = record.Left;
        bool left = RequiresGradient(x.Slot), right = record.ScalarInput >= 0;
        T s = FromDouble(record.Scalar);
        int n = x.Length;
        ReadOnlySpan<T> dz = gradients[output].AsSpan(0, n);
        ReadOnlySpan<T> z = Values(output).Slice(0, n);
        ReadOnlySpan<T> scalar = new(in s);
        T ds = T.Zero;
        switch (record.Operation)
        {
            case ArrayOperation.AddScalar:
            case ArrayOperation.SubtractScalar:
            case ArrayOperation.ScalarSubtract:
                if (left)
                {
                    AddGradient(x, dz, negate: record.Operation == ArrayOperation.ScalarSubtract);
                }

                if (right)
                {
                    ds = XMath.Sum(new ReadOnlyVectorSpan<T>(dz));
                    if (record.Operation == ArrayOperation.SubtractScalar)
                    {
                        ds = -ds;
                    }
                }

                break;
            case ArrayOperation.MultiplyScalar:
                if (left)
                {
                    AddMaskedScale(x, dz, s);
                }

                if (right)
                {
                    ds = MaskedSum(dz, LogicalValues(x, LeftValues));
                }

                break;
            case ArrayOperation.DivideScalar:
                if (left)
                {
                    AddMaskedScale(x, dz, T.One / s);
                }

                if (right)
                {
                    Span<T> d = Scratch(RightDerivative, n);
                    ElementwiseKernels<T>.DenominatorDerivatives(LogicalValues(x, LeftValues), scalar, z, d);
                    ds = MaskedSum(dz, d);
                }

                break;
            default:
                ds = ScalarOperandDerivatives(record.Operation, x, s, left, right, dz, z);
                break;
        }

        if (right)
        {
            adjoints[record.ScalarInput] += double.CreateTruncating(ds);
        }
    }

    private T ScalarOperandDerivatives(ArrayOperation operation, in ArrayView x, T s, bool left, bool right, ReadOnlySpan<T> dz, ReadOnlySpan<T> z)
    {
        int n = dz.Length;
        ReadOnlySpan<T> xValues = LogicalValues(x, LeftValues);
        ReadOnlySpan<T> scalar = new(in s);
        Span<T> dx = left ? Scratch(LeftDerivative, n) : default;
        Span<T> d = right ? Scratch(RightDerivative, n) : default;
        switch (operation)
        {
            case ArrayOperation.MinScalar:
                MinMaxDerivatives<MinLeftDerivative<T>, MinRightDerivative<T>>(xValues, s, left, right, dx, d);
                break;
            case ArrayOperation.MaxScalar:
                MinMaxDerivatives<MaxLeftDerivative<T>, MaxRightDerivative<T>>(xValues, s, left, right, dx, d);
                break;
            case ArrayOperation.PowScalar:
                PowDerivatives(xValues, scalar, z, dz, left, right, dx, d);
                break;
            case ArrayOperation.HypotScalar:
                if (left)
                {
                    ElementwiseKernels<T>.Map<HypotDerivative<T>>(xValues, s, z, dx);
                }

                if (right)
                {
                    ElementwiseKernels<T>.Map<HypotSecondDerivative<T>>(xValues, s, z, d);
                }

                break;
            default:
                LogExpDerivatives(operation == ArrayOperation.LogSubExpScalar, xValues, scalar, dz, left, right, dx, d);
                break;
        }

        if (left)
        {
            AddMaskedProduct(x, dz, dx);
        }

        return right ? MaskedSum(dz, d) : T.Zero;
    }

    private T MaskedSum(ReadOnlySpan<T> dz, ReadOnlySpan<T> d)
    {
        Span<T> contribution = Scratch(Contribution, dz.Length);
        ElementwiseKernels<T>.MaskedProduct(dz, d, contribution, overwrite: true);
        return XMath.Sum(new ReadOnlyVectorSpan<T>(contribution));
    }

    private static void MinMaxDerivatives<TLeft, TRight>(ReadOnlySpan<T> x, ReadOnlySpan<T> y, bool left, bool right, Span<T> dx, Span<T> dy)
        where TLeft : IBinaryKernel<T>
        where TRight : IBinaryKernel<T>
    {
        if (left)
        {
            ElementwiseKernels<T>.Map<TLeft>(x, y, dx);
        }

        if (right)
        {
            ElementwiseKernels<T>.Map<TRight>(x, y, dy);
        }
    }

    private static void MinMaxDerivatives<TLeft, TRight>(ReadOnlySpan<T> x, T s, bool left, bool right, Span<T> dx, Span<T> ds)
        where TLeft : IBinaryKernel<T>
        where TRight : IBinaryKernel<T>
    {
        if (left)
        {
            ElementwiseKernels<T>.Map<TLeft>(x, s, dx);
        }

        if (right)
        {
            ElementwiseKernels<T>.Map<TRight>(x, s, ds);
        }
    }

    // Recomputes x^y in double when T is float so XMathExtensions power derivatives receive full-precision values.
    private static void PowDerivatives(
        ReadOnlySpan<T> x, ReadOnlySpan<T> y, ReadOnlySpan<T> z, ReadOnlySpan<T> dz, bool left, bool right, Span<T> dx, Span<T> dy)
    {
        bool broadcast = y.Length != dz.Length;
        for (int i = 0; i < dz.Length; i++)
        {
            if (T.IsZero(dz[i]))
            {
                if (left)
                {
                    dx[i] = T.Zero;
                }

                if (right)
                {
                    dy[i] = T.Zero;
                }

                continue;
            }

            double a = double.CreateTruncating(x[i]), b = double.CreateTruncating(broadcast ? y[0] : y[i]);
            double value = typeof(T) == typeof(double) ? double.CreateTruncating(z[i]) : XMath.Pow<double>(a, b);
            if (left)
            {
                dx[i] = FromDouble(XMathExtensions.PowerBaseDerivative(a, b, value));
            }

            if (right)
            {
                dy[i] = FromDouble(XMathExtensions.PowerExponentDerivative(a, b, value));
            }
        }
    }

    private static void LogExpDerivatives(
        bool subtract, ReadOnlySpan<T> x, ReadOnlySpan<T> y, ReadOnlySpan<T> dz, bool left, bool right, Span<T> dx, Span<T> dy)
    {
        bool broadcast = y.Length != dz.Length;
        for (int i = 0; i < dz.Length; i++)
        {
            if (T.IsZero(dz[i]))
            {
                if (left)
                {
                    dx[i] = T.Zero;
                }

                if (right)
                {
                    dy[i] = T.Zero;
                }

                continue;
            }

            double a = double.CreateTruncating(x[i]), b = double.CreateTruncating(broadcast ? y[0] : y[i]);
            double da, db;
            if (subtract)
            {
                XMathExtensions.LogSubExpDerivatives(a, b, out da, out db);
            }
            else
            {
                XMathExtensions.LogAddExpDerivatives(a, b, out da, out db);
            }

            if (left)
            {
                dx[i] = FromDouble(da);
            }

            if (right)
            {
                dy[i] = FromDouble(db);
            }
        }
    }

    // A transposed view is contiguous in logical column-major order only when it has <= 1 stored column or 1 stored row.
    private bool IsContiguous(in ArrayView view)
    {
        int storageRows = rows[view.Slot];
        return !view.Broadcast && (view.Transposed
            ? view.ColumnCount <= 1 || storageRows == 1
            : view.ColumnCount <= 1 || view.RowCount == storageRows);
    }

    private ReadOnlySpan<T> LogicalValues(in ArrayView view, int index)
    {
        Span<T> values = Values(view.Slot);
        if (IsContiguous(view))
        {
            return values.Slice(Offset(view), view.Length);
        }

        Span<T> copy = Scratch(index, view.Length);
        MatrixView(view).CopyTo(new MatrixSpan<T>(copy, view.Rows, view.Columns));
        return copy;
    }

    private void AddMaskedProduct(in ArrayView target, ReadOnlySpan<T> dy, ReadOnlySpan<T> d)
    {
        Span<T> gradient = BeginGradient(target, out bool overwrite);
        if (IsContiguous(target))
        {
            ElementwiseKernels<T>.MaskedProduct(dy, d, gradient.Slice(Offset(target), target.Length), overwrite);
            return;
        }

        Span<T> contribution = Scratch(Contribution, target.Length);
        ElementwiseKernels<T>.MaskedProduct(dy, d, contribution, overwrite: true);
        Scatter(target, gradient, contribution, overwrite);
    }

    private void AddMaskedScale(in ArrayView target, ReadOnlySpan<T> dy, T d)
    {
        Span<T> gradient = BeginGradient(target, out bool overwrite);
        if (IsContiguous(target))
        {
            ElementwiseKernels<T>.MaskedScale(dy, d, gradient.Slice(Offset(target), target.Length), overwrite);
            return;
        }

        Span<T> contribution = Scratch(Contribution, target.Length);
        ElementwiseKernels<T>.MaskedScale(dy, d, contribution, overwrite: true);
        Scatter(target, gradient, contribution, overwrite);
    }

    public void AddGradient(in ArrayView target, ReadOnlySpan<T> dy, bool negate)
    {
        Span<T> gradient = BeginGradient(target, out bool overwrite);
        if (IsContiguous(target))
        {
            var destination = new VectorSpan<T>(gradient.Slice(Offset(target), target.Length));
            var source = new ReadOnlyVectorSpan<T>(dy);
            if (overwrite)
            {
                if (negate)
                {
                    XMath.Negate(source, destination);
                }
                else
                {
                    source.CopyTo(destination);
                }
            }
            else if (negate)
            {
                XMath.Subtract(destination.AsReadOnly(), source, destination);
            }
            else
            {
                XMath.Add(destination.AsReadOnly(), source, destination);
            }

            return;
        }

        if (target.Broadcast)
        {
            AddReduced(target, gradient, dy, overwrite, negate);
            return;
        }

        MatrixSpan<T> view = WritableMatrixOf(gradient, target);
        var contribution = new ReadOnlyMatrixSpan<T>(dy, target.Rows, target.Columns);
        if (overwrite)
        {
            if (negate)
            {
                XMath.Negate(contribution, view);
            }
            else
            {
                contribution.CopyTo(view);
            }
        }
        else if (negate)
        {
            XMath.Subtract(view.AsReadOnly(), contribution, view);
        }
        else
        {
            XMath.Add(view.AsReadOnly(), contribution, view);
        }
    }

    private void Scatter(in ArrayView target, Span<T> gradient, ReadOnlySpan<T> contribution, bool overwrite)
    {
        if (target.Broadcast)
        {
            AddReduced(target, gradient, contribution, overwrite, negate: false);
            return;
        }

        MatrixSpan<T> view = WritableMatrixOf(gradient, target);
        var source = new ReadOnlyMatrixSpan<T>(contribution, target.Rows, target.Columns);
        if (overwrite)
        {
            source.CopyTo(view);
        }
        else
        {
            XMath.Add(view.AsReadOnly(), source, view);
        }
    }

    private void AddReduced(in ArrayView target, Span<T> gradient, ReadOnlySpan<T> contribution, bool overwrite, bool negate)
    {
        var source = new ReadOnlyMatrixSpan<T>(contribution, target.Rows, target.Columns);
        var destination = new VectorSpan<T>(gradient.Slice(Offset(target), target.RowCount));
        VectorSpan<T> sums = overwrite ? destination : new VectorSpan<T>(Scratch(Reduction, target.RowCount));
        if (target.Transposed)
        {
            XMath.ColumnSums(source, sums);
        }
        else
        {
            XMath.RowSums(source, sums);
        }

        if (overwrite)
        {
            if (negate)
            {
                XMath.Negate(destination.AsReadOnly(), destination);
            }
        }
        else if (negate)
        {
            XMath.Subtract(destination.AsReadOnly(), sums.AsReadOnly(), destination);
        }
        else
        {
            XMath.Add(destination.AsReadOnly(), sums.AsReadOnly(), destination);
        }
    }

    private MatrixSpan<T> WritableMatrixOf(Span<T> storage, scoped in ArrayView view)
    {
        int slot = view.Slot;
        var matrix = new MatrixSpan<T>(storage, rows[slot], columns[slot]);
        if (view.RowCount != rows[slot] || view.ColumnCount != columns[slot])
        {
            matrix = matrix[view.RowStart..(view.RowStart + view.RowCount), view.ColumnStart..(view.ColumnStart + view.ColumnCount)];
        }

        return view.Transposed ? matrix.T : matrix;
    }
}
