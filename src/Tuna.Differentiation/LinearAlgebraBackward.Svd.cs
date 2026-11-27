using Tuna.LinearAlgebra;

namespace Tuna.Differentiation;

internal static partial class LinearAlgebraBackward
{
    // Reserve exponent headroom for two inverse singular values, RHS products, and sums along both axes.
    private static readonly double SvdSmall = Math.ScaleB(1.0, -200);
    private static readonly double SvdLarge = Math.ScaleB(1.0, 200);

    private static void SvdSolve(ArrayStore<double> store, in ArrayRecord record)
    {
        int r = (int)record.Scalar;
        if (r == 0 || !store.Touched(record.Output) || record.Right.Columns == 0)
        {
            return;
        }

        ArrayView a = record.Left, b = record.Right;
        int m = a.Rows, n = a.Columns, k = b.Columns;
        bool needsA = store.RequiresGradient(a.Slot), needsB = store.RequiresGradient(b.Slot);
        var svd = store.Decompositions.SingularValue(record.ScalarInput);
        ReadOnlySpan<double> sigma = svd.Values.AsSpan(0, r);
        var u = new SvdOperand(svd.U![.., ..r]);
        var v = new SvdOperand(svd.V![.., ..r]);
        var g = new SvdOperand(new ReadOnlyMatrixSpan<double>(store.OutputGradient(record.Output), n, k));
        var rhs = new SvdOperand(store.MatrixView(b));
        var x = new SvdOperand(new ReadOnlyMatrixSpan<double>(store.Values(record.Output), n, k));
        bool scaled = NeedsSvdScaling(sigma, u.Values, v.Values, g.Values)
            || (needsA && (NeedsSvdScaling(rhs.Values) || NeedsSvdScaling(x.Values)));

        // H = VᵀG, W = Σ⁻¹H, and the right-hand-side pullback is UW.
        var h = SvdWorkspace(store, ArrayStore<double>.DecompositionFirst, r, k, scaled);
        var divided = SvdWorkspace(store, ArrayStore<double>.DecompositionFifth, r, k, scaled);
        var left = SvdWorkspace(store, ArrayStore<double>.DecompositionThird, m, k, scaled);
        SvdMultiply(v.T, g, h);
        SvdDivide(h, divided, rowSigma: sigma);
        SvdMultiply(u, divided, left);

        if (!needsA)
        {
            store.AddGradient(b, left.RoundedData(), negate: false);
            return;
        }

        SvdBuffer f = default, right = default;
        if (r < m || r < n || scaled)
        {
            f = SvdWorkspace(store, ArrayStore<double>.DecompositionSecond, r, k, scaled);
            right = SvdWorkspace(store, ArrayStore<double>.DecompositionFourth, n, k, scaled);
            SvdMultiply(u.T, rhs, f);
        }

        if (scaled && AllFinite(rhs.Values))
        {
            // The stored X can already have rounded to zero or infinity before the adjoint is applied.
            SvdDivide(f, divided, rowSigma: sigma);
            SvdMultiply(v, divided, right);
            x = right;
        }

        // Golub–Pereyra (1973), using VᵀX = Σ⁻¹UᵀB for the final term.
        var contribution = SvdWorkspace(store, ArrayStore<double>.MatrixContribution, m, n, scaled);
        SvdMultiply(left, x.T, contribution, subtract: true);
        if (needsB)
        {
            store.AddGradient(b, left.RoundedData(), negate: false);
        }

        if (r < m)
        {
            SvdMultiply(u, f, left);
            SvdSubtractFrom(rhs, left);
            SvdDivide(h, divided, rowSigma: sigma, squared: true);
            SvdMultiply(v, divided, right);
            SvdMultiply(left, right.ReadOnly.T, contribution, accumulate: true);
        }

        if (r < n)
        {
            SvdMultiply(v, h, right);
            SvdSubtractFrom(g, right);
            SvdDivide(f, divided, rowSigma: sigma, squared: true);
            SvdMultiply(u, divided, left);
            SvdMultiply(left, right.ReadOnly.T, contribution, accumulate: true);
        }

        store.AddGradient(a, contribution.RoundedData(), negate: false);
    }

    private static void SvdPseudoInverse(ArrayStore<double> store, in ArrayRecord record)
    {
        int r = (int)record.Scalar;
        if (r == 0 || !store.Touched(record.Output))
        {
            return;
        }

        ArrayView a = record.Left;
        int m = a.Rows, n = a.Columns;
        var svd = store.Decompositions.SingularValue(record.ScalarInput);
        ReadOnlySpan<double> sigma = svd.Values.AsSpan(0, r);
        var u = new SvdOperand(svd.U![.., ..r]);
        var v = new SvdOperand(svd.V![.., ..r]);
        var g = new SvdOperand(new ReadOnlyMatrixSpan<double>(store.OutputGradient(record.Output), n, m));
        bool scaled = NeedsSvdScaling(sigma, u.Values, v.Values, g.Values);

        // Reuse H = UᵀGᵀV for both orthogonal-complement projections.
        var projected = SvdWorkspace(store, ArrayStore<double>.DecompositionThird, m, r, scaled);
        var h = SvdWorkspace(store, ArrayStore<double>.DecompositionFirst, r, r, scaled);
        var divided = SvdWorkspace(store, ArrayStore<double>.DecompositionSecond, r, r, scaled);
        var side = SvdWorkspace(store, ArrayStore<double>.DecompositionFourth, m, r, scaled);
        var contribution = SvdWorkspace(store, ArrayStore<double>.MatrixContribution, m, n, scaled);
        SvdMultiply(g.T, v, projected);
        SvdMultiply(u.T, projected, h);
        SvdDivide(h.ReadOnly.T, divided, rowSigma: sigma, columnSigma: sigma);
        SvdMultiply(u, divided, side);
        SvdMultiply(side, v.T, contribution, subtract: true);

        if (r < m)
        {
            SvdMultiply(u, h, projected, subtract: true, accumulate: true);
            SvdDivide(projected, projected, columnSigma: sigma, squared: true);
            SvdMultiply(projected, v.T, contribution, accumulate: true);
        }

        if (r < n)
        {
            var right = SvdWorkspace(store, ArrayStore<double>.DecompositionFourth, n, r, scaled);
            SvdMultiply(g, u, right);
            SvdMultiply(v, h.ReadOnly.T, right, subtract: true, accumulate: true);
            SvdDivide(right, right, columnSigma: sigma, squared: true);
            SvdMultiply(u, right.ReadOnly.T, contribution, accumulate: true);
        }

        store.AddGradient(a, contribution.RoundedData(), negate: false);
    }

    private static bool NeedsSvdScaling(ReadOnlySpan<double> sigma, ReadOnlyMatrixSpan<double> u, ReadOnlyMatrixSpan<double> v,
        ReadOnlyMatrixSpan<double> adjoint) =>
        sigma[0] > SvdLarge || sigma[^1] < SvdSmall || NeedsSvdScaling(adjoint) || NeedsSvdScaling(u) || NeedsSvdScaling(v);

    private static bool NeedsSvdScaling(ReadOnlyMatrixSpan<double> matrix)
    {
        for (int j = 0; j < matrix.Columns; j++)
        {
            for (int i = 0; i < matrix.Rows; i++)
            {
                double value = Math.Abs(matrix[i, j]);
                if (double.IsFinite(value) && (value > SvdLarge || (value != 0 && value < SvdSmall)))
                {
                    return true;
                }
            }
        }

        return false;
    }

    private static bool AllFinite(ReadOnlyMatrixSpan<double> matrix)
    {
        for (int j = 0; j < matrix.Columns; j++)
        {
            for (int i = 0; i < matrix.Rows; i++)
            {
                if (!double.IsFinite(matrix[i, j]))
                {
                    return false;
                }
            }
        }

        return true;
    }

    private static SvdBuffer SvdWorkspace(ArrayStore<double> store, int slot, int rows, int columns, bool scaled)
    {
        int length = checked(rows * columns);
        var values = new MatrixSpan<double>(store.Scratch(slot, length), rows, columns);
        var exponents = scaled ? new MatrixSpan<int>(store.ScratchExponents(slot, length), rows, columns) : default;
        return new(values, exponents);
    }

    private static void SvdMultiply(SvdOperand a, SvdOperand b, SvdBuffer destination, bool subtract = false, bool accumulate = false)
    {
        if (!destination.Scaled)
        {
            if (subtract || accumulate)
            {
                // All operands of the accumulating products are compact factors or work matrices.
                Blas.Gemm(a.Transpose, b.Transpose, a.Values.Rows, b.Values.Columns, a.Values.Columns,
                    subtract ? -1.0 : 1.0, a.Values.Data, a.LeadingDimension, b.Values.Data, b.LeadingDimension,
                    accumulate ? 1.0 : 0.0, destination.Values.Data, destination.Values.Rows);
            }
            else
            {
                XMath.Multiply(a.Values, b.Values, destination.Values);
            }

            return;
        }

        for (int j = 0; j < b.Values.Columns; j++)
        {
            for (int i = 0; i < a.Values.Rows; i++)
            {
                SvdNumber sum = default;
                for (int p = 0; p < a.Values.Columns; p++)
                {
                    sum += a.Get(i, p) * b.Get(p, j);
                }

                sum = subtract ? -sum : sum;
                destination.Set(i, j, accumulate ? destination.Get(i, j) + sum : sum);
            }
        }
    }

    private static void SvdSubtractFrom(SvdOperand minuend, SvdBuffer destination)
    {
        for (int j = 0; j < destination.Values.Columns; j++)
        {
            for (int i = 0; i < destination.Values.Rows; i++)
            {
                if (destination.Scaled)
                {
                    destination.Set(i, j, minuend.Get(i, j) + -destination.Get(i, j));
                }
                else
                {
                    destination.Values[i, j] = minuend.Values[i, j] - destination.Values[i, j];
                }
            }
        }
    }

    private static void SvdDivide(SvdOperand input, SvdBuffer destination, ReadOnlySpan<double> rowSigma = default,
        ReadOnlySpan<double> columnSigma = default, bool squared = false)
    {
        for (int j = 0; j < input.Values.Columns; j++)
        {
            double col = columnSigma.IsEmpty ? 1 : columnSigma[j];
            for (int i = 0; i < input.Values.Rows; i++)
            {
                double row = rowSigma.IsEmpty ? 1 : rowSigma[i];
                if (destination.Scaled)
                {
                    SvdNumber divisor = SvdNumber.FromDouble(row) * SvdNumber.FromDouble(col);
                    destination.Set(i, j, input.Get(i, j) / (squared ? divisor * divisor : divisor));
                }
                else
                {
                    double divisor = row * col;
                    destination.Values[i, j] = input.Values[i, j] / (squared ? divisor * divisor : divisor);
                }
            }
        }
    }

    private readonly ref struct SvdOperand(ReadOnlyMatrixSpan<double> values, ReadOnlyMatrixSpan<int> exponents = default, bool transposed = false)
    {
        public readonly ReadOnlyMatrixSpan<double> Values = values;
        private readonly ReadOnlyMatrixSpan<int> shifts = exponents;

        public SvdOperand T => new(Values.T, shifts.T, !transposed);
        public BlasTranspose Transpose => transposed ? BlasTranspose.Transpose : BlasTranspose.None;
        public int LeadingDimension => transposed ? Values.Columns : Values.Rows;

        public SvdNumber Get(int row, int column) => shifts.IsEmpty
            ? SvdNumber.FromDouble(Values[row, column])
            : new(Values[row, column], shifts[row, column]);
    }

    private readonly ref struct SvdBuffer(MatrixSpan<double> values, MatrixSpan<int> exponents)
    {
        public readonly MatrixSpan<double> Values = values;
        private readonly MatrixSpan<int> shifts = exponents;

        public bool Scaled => !shifts.IsEmpty;
        public SvdOperand ReadOnly => new(Values, shifts);

        public SvdNumber Get(int row, int column) => new(Values[row, column], shifts[row, column]);

        public void Set(int row, int column, SvdNumber value)
        {
            Values[row, column] = value.Mantissa;
            shifts[row, column] = value.Exponent;
        }

        public Span<double> RoundedData()
        {
            var data = Values.Data;
            if (Scaled)
            {
                var exponents = shifts.Data;
                for (int i = 0; i < data.Length; i++)
                {
                    data[i] = Math.ScaleB(data[i], exponents[i]);
                }
            }

            return data;
        }

        public static implicit operator SvdOperand(SvdBuffer buffer) => buffer.ReadOnly;
    }

    // Keep each entry's binary exponent separate through the products so rounding happens at the final adjoint.
    private readonly record struct SvdNumber(double Mantissa, int Exponent)
    {
        public static SvdNumber FromDouble(double value, int exponent = 0)
        {
            if (value == 0 || !double.IsFinite(value))
            {
                return new(value, 0);
            }

            int shift = Math.ILogB(value);
            return new(Math.ScaleB(value, -shift), exponent + shift);
        }

        public static SvdNumber operator -(SvdNumber value) => new(-value.Mantissa, value.Exponent);

        public static SvdNumber operator +(SvdNumber a, SvdNumber b)
        {
            if (a.Mantissa == 0)
            {
                return b;
            }

            if (b.Mantissa == 0)
            {
                return a;
            }

            int exponent = Math.Max(a.Exponent, b.Exponent);
            return FromDouble(Math.ScaleB(a.Mantissa, a.Exponent - exponent) + Math.ScaleB(b.Mantissa, b.Exponent - exponent), exponent);
        }

        public static SvdNumber operator *(SvdNumber a, SvdNumber b) => FromDouble(a.Mantissa * b.Mantissa, a.Exponent + b.Exponent);

        public static SvdNumber operator /(SvdNumber a, SvdNumber b) => FromDouble(a.Mantissa / b.Mantissa, a.Exponent - b.Exponent);
    }
}
