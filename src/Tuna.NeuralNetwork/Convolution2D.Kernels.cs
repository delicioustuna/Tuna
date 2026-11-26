using System.Buffers;
using System.Numerics;

namespace Tuna.NeuralNetwork;

using Tuna.LinearAlgebra;

internal readonly record struct ConvolutionGeometry(
    int Channels, int KernelHeight, int KernelWidth,
    int StrideHeight, int StrideWidth, int DilationHeight, int DilationWidth)
{
    internal int ColumnLength => checked(KernelHeight * KernelWidth * Channels);
}

public sealed partial class Convolution2D<T>
{
    private const int StackKernelHeight = 128;

    internal static unsafe void ForwardCore(ReadOnlyTensorSpan<T> x, ReadOnlyMatrixSpan<T> weights, ReadOnlyVectorSpan<T> bias,
        ConvolutionGeometry shape, Padding2D padding, TensorSpan<T> destination)
    {
        if (destination.IsEmpty)
        {
            return;
        }

        int width = (int)x.Lengths[1], height = (int)x.Lengths[2];
        int outWidth = (int)destination.Lengths[1], outHeight = (int)destination.Lengths[2];
        T[]? packedInput = null, packedOutput = null, packedWeight = null, packedBias = null;
        try
        {
            var input = x;
            if (shape.Channels > 1 && x.Strides[0] != 1 && !x.IsEmpty)
            {
                int length = checked(shape.Channels * width * height);
                packedInput = ArrayPool<T>.Shared.Rent(length);
                var buffer = new TensorSpan<T>(packedInput.AsSpan(0, length), shape.Channels, width, height);
                x.CopyTo(buffer);
                input = buffer.AsReadOnly();
            }

            var target = destination;
            if (weights.Rows > 1 && destination.Strides[0] != 1)
            {
                int length = checked(weights.Rows * outWidth * outHeight);
                packedOutput = ArrayPool<T>.Shared.Rent(length);
                target = new TensorSpan<T>(packedOutput.AsSpan(0, length), weights.Rows, outWidth, outHeight);
            }

            var w = Dense.Columns(weights, ref packedWeight);
            var b = Dense.Contiguous(bias, ref packedBias);
            int positions = outWidth * outHeight;
            int threads = Split.Threads((double)w.Rows * shape.ColumnLength * positions);
            fixed (T* px = input.Data, py = target.Data, pw = &Dense.Ref(w), pb = &Dense.Ref(b))
            {
                var work = new ForwardWork
                {
                    X = px,
                    Y = py,
                    Weights = pw,
                    Bias = bias.Length == 0 ? null : pb,
                    Shape = shape,
                    Padding = padding,
                    Width = width,
                    Height = height,
                    OutWidth = outWidth,
                    OutChannels = w.Rows,
                    InputColumnStride = input.IsEmpty ? 0 : (int)input.Strides[1],
                    InputRowStride = input.IsEmpty ? 0 : (int)input.Strides[2],
                    OutputColumnStride = (int)target.Strides[1],
                    OutputRowStride = (int)target.Strides[2],
                    WeightStride = w.ColumnStride,
                };
                Split.For(positions, Split.Chunks(positions, threads, 8), threads, ref work);
            }

            if (packedOutput is not null)
            {
                target.CopyTo(destination);
            }
        }
        finally
        {
            Dense.Return(packedInput);
            Dense.Return(packedOutput);
            Dense.Return(packedWeight);
            Dense.Return(packedBias);
        }
    }

    private unsafe struct ForwardWork : Split.IRangeWork
    {
        internal T* X, Y, Weights, Bias;
        internal ConvolutionGeometry Shape;
        internal Padding2D Padding;
        internal int Width, Height, OutWidth, OutChannels;
        internal int InputColumnStride, InputRowStride, OutputColumnStride, OutputRowStride, WeightStride;

        public readonly void Invoke(int start, int end)
        {
            int k = Shape.ColumnLength;
            int block = Math.Max(1, Math.Min(end - start, ColumnBufferElements / k));
            var buffer = ArrayPool<T>.Shared.Rent(k * block);
            nint[]? rentedRows = null;
            int[]? rentedStrides = null;
            try
            {
                int kernelHeight = Shape.KernelHeight;
                Span<nint> rows = kernelHeight <= StackKernelHeight ? stackalloc nint[kernelHeight]
                    : (rentedRows = ArrayPool<nint>.Shared.Rent(kernelHeight)).AsSpan(0, kernelHeight);
                Span<int> strides = kernelHeight <= StackKernelHeight ? stackalloc int[kernelHeight]
                    : (rentedStrides = ArrayPool<int>.Shared.Rent(kernelHeight)).AsSpan(0, kernelHeight);
                strides.Fill(InputColumnStride);
                var w = Views.Read(Weights, OutChannels, k, WeightStride);
                var bias = Bias is null ? default : new ReadOnlyVectorSpan<T>(new ReadOnlySpan<T>(Bias, OutChannels));
                fixed (T* columns = buffer)
                fixed (nint* rowPointers = rows)
                fixed (int* rowStrides = strides)
                {
                    for (int p = start; p < end;)
                    {
                        int oh = p / OutWidth, ow0 = p - oh * OutWidth;
                        int count = Math.Min(OutWidth - ow0, Math.Min(end - p, block));
                        SetRowPointers(Shape, (T**)rowPointers, X, InputRowStride, Height, oh, Padding.Top);
                        PackColumns(Shape, (T**)rowPointers, rowStrides, Width, Padding.Left, ow0, ow0 + count, columns);
                        var output = Views.Write(Y + (nint)oh * OutputRowStride + (nint)ow0 * OutputColumnStride,
                            OutChannels, count, OutputColumnStride);
                        EvaluateColumns(w, bias, columns, count, output);
                        p += count;
                    }
                }
            }
            finally
            {
                ArrayPool<T>.Shared.Return(buffer);
                if (rentedRows is not null)
                {
                    ArrayPool<nint>.Shared.Return(rentedRows);
                }

                if (rentedStrides is not null)
                {
                    ArrayPool<int>.Shared.Return(rentedStrides);
                }
            }
        }
    }

    private static unsafe void SetRowPointers(ConvolutionGeometry shape, T** rows, T* x, int stride, int height, int oh, int top)
    {
        for (int i = 0; i < shape.KernelHeight; i++)
        {
            long ih = (long)oh * shape.StrideHeight + (long)i * shape.DilationHeight - top;
            rows[i] = (ulong)ih < (uint)height ? x + (nint)ih * stride : null;
        }
    }

    // Pack (kernel row, kernel column, channel) for each output position.
    private static unsafe void PackColumns(ConvolutionGeometry shape, T** rows, int* strides, int width, int left,
        int ow0, int ow1, T* cols)
    {
        int k = shape.ColumnLength, channels = shape.Channels;
        nint bytes = (nint)channels * sizeof(T);
        for (int ow = ow0; ow < ow1; ow++)
        {
            T* column = cols + (nint)(ow - ow0) * k;
            for (int i = 0; i < shape.KernelHeight; i++)
            {
                T* row = rows[i];
                int ldx = strides[i];
                for (int j = 0; j < shape.KernelWidth; j++)
                {
                    long iw = (long)ow * shape.StrideWidth + (long)j * shape.DilationWidth - left;
                    T* target = column + (nint)(i * shape.KernelWidth + j) * channels;
                    if (row is null || (ulong)iw >= (uint)width)
                    {
                        new Span<T>(target, channels).Clear();
                    }
                    else
                    {
                        Buffer.MemoryCopy(row + (nint)iw * ldx, target, bytes, bytes);
                    }
                }
            }
        }
    }

    private static unsafe void EvaluateColumns(ReadOnlyMatrixSpan<T> w, ReadOnlyVectorSpan<T> bias, T* columns, int count,
        MatrixSpan<T> output)
    {
        XMath.Multiply(w, Views.Read(columns, w.Columns, count, w.Columns), output);
        if (bias.Length != 0)
        {
            XMathExtensions.AddBias(output, bias);
        }
    }

    internal static unsafe void BackwardCore(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> weights, ReadOnlyMatrixSpan<T> dy,
        ConvolutionGeometry shape, Padding2D padding, int width, int height, int outWidth, MatrixSpan<T> dx, MatrixSpan<T> dw)
    {
        if (dy.IsEmpty || (dx.IsEmpty && dw.IsEmpty))
        {
            return;
        }

        int k = shape.ColumnLength;
        int block = Math.Min(outWidth, Math.Max(1, ColumnBufferElements / k));
        int inputRowStride = x.IsEmpty ? 0 : shape.Channels * width;
        var buffer = ArrayPool<T>.Shared.Rent(k * block);
        nint[]? rentedRows = null;
        int[]? rentedStrides = null;
        try
        {
            int kernelHeight = dw.IsEmpty ? 0 : shape.KernelHeight;
            Span<nint> rows = kernelHeight <= StackKernelHeight ? stackalloc nint[kernelHeight]
                : (rentedRows = ArrayPool<nint>.Shared.Rent(kernelHeight)).AsSpan(0, kernelHeight);
            Span<int> strides = kernelHeight <= StackKernelHeight ? stackalloc int[kernelHeight]
                : (rentedStrides = ArrayPool<int>.Shared.Rent(kernelHeight)).AsSpan(0, kernelHeight);
            if (!dw.IsEmpty)
            {
                strides.Fill(shape.Channels);
            }

            fixed (T* input = x.Data, columns = buffer)
            fixed (nint* rowPointers = rows)
            fixed (int* rowStrides = strides)
            {
                for (int p = 0; p < dy.Columns;)
                {
                    int oh = p / outWidth, ow = p - oh * outWidth;
                    int count = Math.Min(outWidth - ow, block);
                    var outputGradient = dy[.., p..(p + count)];
                    var panel = Views.Write(columns, k, count, k);
                    if (!dw.IsEmpty)
                    {
                        SetRowPointers(shape, (T**)rowPointers, input, inputRowStride, height, oh, padding.Top);
                        PackColumns(shape, (T**)rowPointers, rowStrides, width, padding.Left, ow, ow + count, columns);
                        GradientKernels.AddProduct(dw, outputGradient, panel.T.AsReadOnly());
                    }

                    if (!dx.IsEmpty)
                    {
                        XMath.Multiply(weights.T, outputGradient, panel);
                        ScatterColumns(shape, padding, width, height, oh, ow, panel.AsReadOnly(), dx);
                    }

                    p += count;
                }
            }
        }
        finally
        {
            ArrayPool<T>.Shared.Return(buffer);
            if (rentedRows is not null)
            {
                ArrayPool<nint>.Shared.Return(rentedRows);
            }

            if (rentedStrides is not null)
            {
                ArrayPool<int>.Shared.Return(rentedStrides);
            }
        }
    }

    private static void ScatterColumns(ConvolutionGeometry shape, Padding2D padding, int width, int height, int oh, int ow,
        ReadOnlyMatrixSpan<T> columns, MatrixSpan<T> destination)
    {
        for (int p = 0; p < columns.Columns; p++)
        {
            var source = columns.Column(p);
            for (int i = 0; i < shape.KernelHeight; i++)
            {
                long ih = (long)oh * shape.StrideHeight + (long)i * shape.DilationHeight - padding.Top;
                if ((ulong)ih >= (uint)height)
                {
                    continue;
                }

                for (int j = 0; j < shape.KernelWidth; j++)
                {
                    long iw = (long)(ow + p) * shape.StrideWidth + (long)j * shape.DilationWidth - padding.Left;
                    if ((ulong)iw < (uint)width)
                    {
                        int start = (i * shape.KernelWidth + j) * shape.Channels;
                        var target = destination.Column((int)ih * width + (int)iw);
                        XMath.Add(target.AsReadOnly(), source[start..(start + shape.Channels)], target);
                    }
                }
            }
        }
    }
}
