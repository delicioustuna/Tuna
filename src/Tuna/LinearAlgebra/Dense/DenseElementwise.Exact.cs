using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace Tuna.LinearAlgebra;

// 正しく丸められる四則演算・平方根およびビット・選択演算の SIMD 実装（スカラー経路とビット単位で一致）。
// 複数 NaN 入力時のペイロード選択および sNaN の qNaN 化は未規定。指数・対数系は TryMapTranscendental へ委譲する。
internal static partial class DenseElementwise
{
    // 対応する SIMD 実装があれば destination[i] = op.Invoke(x[i]) を計算して true を返す（インプレース可）。
    // 成否は演算子の型と状態のみで決まり、入力データには依存しない。
    internal static bool TryMap<T, TOperator>(ReadOnlySpan<T> x, Span<T> destination, TOperator op) where T : unmanaged
    {
        DenseStorage.RequireLength(x.Length, destination.Length);
        return TryMapLines(in MemoryMarshal.GetReference(x), 0, ref MemoryMarshal.GetReference(destination), 0, x.Length, 1, op);
    }

    // 対応する SIMD 実装があれば destination[i] = op.Invoke(x[i], y[i]) を計算して true を返す。
    internal static bool TryZip<T, TOperator>(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination, TOperator op) where T : unmanaged
    {
        DenseStorage.RequireLength(x.Length, y.Length);
        DenseStorage.RequireLength(x.Length, destination.Length);
        return TryZipLines<T, TOperator>(in MemoryMarshal.GetReference(x), 0, in MemoryMarshal.GetReference(y), 0,
            ref MemoryMarshal.GetReference(destination), 0, x.Length, 1);
    }

    // count 本の連続区間に対し TryMap と同一の結果を計算する（インプレース可）。演算子ディスパッチは 1 回のみ行う。
    internal static bool TryMapLines<T, TOperator>(ref readonly T x, int xStride, ref T destination, int destinationStride, int length, int count, TOperator op)
        where T : unmanaged
    {
        if (typeof(T) == typeof(double))
        {
            ref double a = ref Unsafe.As<T, double>(ref Unsafe.AsRef(in x));
            ref double c = ref Unsafe.As<T, double>(ref destination);
            return TryMapReal(ref a, xStride, ref c, destinationStride, length, count, op)
                || TryMapTranscendental(ref a, xStride, ref c, destinationStride, length, count, op);
        }

        if (typeof(T) == typeof(float))
        {
            ref float a = ref Unsafe.As<T, float>(ref Unsafe.AsRef(in x));
            ref float c = ref Unsafe.As<T, float>(ref destination);
            return TryMapReal(ref a, xStride, ref c, destinationStride, length, count, op)
                || TryMapTranscendental(ref a, xStride, ref c, destinationStride, length, count, op);
        }

        if (typeof(T) == typeof(Complex))
        {
            return TryMapComplex(ref Unsafe.As<T, Complex>(ref Unsafe.AsRef(in x)), xStride,
                ref Unsafe.As<T, Complex>(ref destination), destinationStride, length, count, op);
        }

        return false;
    }

    // 2 入力版の TryMapLines。
    internal static bool TryZipLines<T, TOperator>(ref readonly T x, int xStride, ref readonly T y, int yStride,
        ref T destination, int destinationStride, int length, int count)
        where T : unmanaged
    {
        if (typeof(T) == typeof(double))
        {
            return TryZipReal<double, TOperator>(ref Unsafe.As<T, double>(ref Unsafe.AsRef(in x)), xStride,
                ref Unsafe.As<T, double>(ref Unsafe.AsRef(in y)), yStride, ref Unsafe.As<T, double>(ref destination), destinationStride, length, count);
        }

        if (typeof(T) == typeof(float))
        {
            return TryZipReal<float, TOperator>(ref Unsafe.As<T, float>(ref Unsafe.AsRef(in x)), xStride,
                ref Unsafe.As<T, float>(ref Unsafe.AsRef(in y)), yStride, ref Unsafe.As<T, float>(ref destination), destinationStride, length, count);
        }

        if (typeof(T) == typeof(Complex))
        {
            return TryZipComplex<TOperator>(ref Unsafe.As<T, Complex>(ref Unsafe.AsRef(in x)), xStride,
                ref Unsafe.As<T, Complex>(ref Unsafe.AsRef(in y)), yStride, ref Unsafe.As<T, Complex>(ref destination), destinationStride, length, count);
        }

        return false;
    }

    private static bool TryMapReal<T, TOperator>(ref T x, int xStride, ref T destination, int destinationStride, int length, int count, TOperator op)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (typeof(TOperator) == typeof(StaticUnaryOperator<T, XMath.Operators.Sqrt<T>>))
        {
            MapLines(ref x, xStride, ref destination, destinationStride, length, count, default(SqrtKernel<T>));
        }
        else if (typeof(TOperator) == typeof(StaticUnaryOperator<T, XMath.Operators.Abs<T>>))
        {
            MapLines(ref x, xStride, ref destination, destinationStride, length, count, default(AbsKernel<T>));
        }
        else if (typeof(TOperator) == typeof(StaticUnaryOperator<T, XMath.Operators.Square<T>>))
        {
            MapLines(ref x, xStride, ref destination, destinationStride, length, count, default(SquareKernel<T>));
        }
        else if (typeof(TOperator) == typeof(StaticUnaryOperator<T, XMath.Operators.Reciprocal<T>>))
        {
            MapLines(ref x, xStride, ref destination, destinationStride, length, count, default(ReciprocalKernel<T>));
        }
        else if (typeof(TOperator) == typeof(StaticUnaryOperator<T, XMath.Operators.Relu<T>>))
        {
            MapLines(ref x, xStride, ref destination, destinationStride, length, count, default(ReluKernel<T>));
        }
        else if (typeof(TOperator) == typeof(NegateOperator<T>))
        {
            MapLines(ref x, xStride, ref destination, destinationStride, length, count, default(NegateKernel<T>));
        }
        else if (typeof(TOperator) == typeof(AddScalarOperator<T>))
        {
            MapLines(ref x, xStride, ref destination, destinationStride, length, count,
                new AddScalarKernel<T>(Unsafe.As<TOperator, AddScalarOperator<T>>(ref op).Scalar));
        }
        else if (typeof(TOperator) == typeof(SubtractScalarOperator<T>))
        {
            MapLines(ref x, xStride, ref destination, destinationStride, length, count,
                new SubtractScalarKernel<T>(Unsafe.As<TOperator, SubtractScalarOperator<T>>(ref op).Scalar));
        }
        else if (typeof(TOperator) == typeof(ScaleOperator<T>))
        {
            MapLines(ref x, xStride, ref destination, destinationStride, length, count,
                new ScaleKernel<T>(Unsafe.As<TOperator, ScaleOperator<T>>(ref op).Scalar));
        }
        else if (typeof(TOperator) == typeof(XMath.Operators.DivideScalar<T>))
        {
            MapLines(ref x, xStride, ref destination, destinationStride, length, count,
                new DivideScalarKernel<T>(Unsafe.As<TOperator, XMath.Operators.DivideScalar<T>>(ref op).Y));
        }
        else if (typeof(TOperator) == typeof(XMath.Operators.HypotScalar<T>))
        {
            MapLines(ref x, xStride, ref destination, destinationStride, length, count,
                new HypotScalarKernel<T>(Unsafe.As<TOperator, XMath.Operators.HypotScalar<T>>(ref op).Y));
        }
        else if (typeof(TOperator) == typeof(XMath.Operators.MaxScalar<T>))
        {
            MapLines(ref x, xStride, ref destination, destinationStride, length, count,
                new MaxScalarKernel<T>(Unsafe.As<TOperator, XMath.Operators.MaxScalar<T>>(ref op).Y));
        }
        else if (typeof(TOperator) == typeof(XMath.Operators.MinScalar<T>))
        {
            MapLines(ref x, xStride, ref destination, destinationStride, length, count,
                new MinScalarKernel<T>(Unsafe.As<TOperator, XMath.Operators.MinScalar<T>>(ref op).Y));
        }
        else if (typeof(TOperator) == typeof(XMath.Operators.LeakyRelu<T>))
        {
            MapLines(ref x, xStride, ref destination, destinationStride, length, count,
                new LeakyReluKernel<T>(Unsafe.As<TOperator, XMath.Operators.LeakyRelu<T>>(ref op).Slope));
        }
        else if (typeof(TOperator) == typeof(XMath.Operators.Clamp<T>))
        {
            ref var clamp = ref Unsafe.As<TOperator, XMath.Operators.Clamp<T>>(ref op);
            // min > max はスカラー経路で最初の要素評価時に例外を送出させる
            if (clamp.Min > clamp.Max)
            {
                return false;
            }

            MapLines(ref x, xStride, ref destination, destinationStride, length, count, new ClampKernel<T>(clamp.Min, clamp.Max));
        }
        else
        {
            return false;
        }

        return true;
    }

    private static bool TryZipReal<T, TOperator>(ref T x, int xStride, ref T y, int yStride, ref T destination, int destinationStride, int length, int count)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (typeof(TOperator) == typeof(AddOperator<T>))
        {
            ZipLines<T, AddKernel<T>>(ref x, xStride, ref y, yStride, ref destination, destinationStride, length, count);
        }
        else if (typeof(TOperator) == typeof(SubtractOperator<T>))
        {
            ZipLines<T, SubtractKernel<T>>(ref x, xStride, ref y, yStride, ref destination, destinationStride, length, count);
        }
        else if (typeof(TOperator) == typeof(MultiplyOperator<T>))
        {
            ZipLines<T, MultiplyKernel<T>>(ref x, xStride, ref y, yStride, ref destination, destinationStride, length, count);
        }
        else if (typeof(TOperator) == typeof(StaticBinaryOperator<T, XMath.Operators.Divide<T>>))
        {
            ZipLines<T, DivideKernel<T>>(ref x, xStride, ref y, yStride, ref destination, destinationStride, length, count);
        }
        else if (typeof(TOperator) == typeof(StaticBinaryOperator<T, XMath.Operators.Max<T>>))
        {
            ZipLines<T, MaxKernel<T>>(ref x, xStride, ref y, yStride, ref destination, destinationStride, length, count);
        }
        else if (typeof(TOperator) == typeof(StaticBinaryOperator<T, XMath.Operators.Min<T>>))
        {
            ZipLines<T, MinKernel<T>>(ref x, xStride, ref y, yStride, ref destination, destinationStride, length, count);
        }
        else if (typeof(TOperator) == typeof(StaticBinaryOperator<T, XMath.Operators.Hypot<T>>))
        {
            ZipLines<T, HypotKernel<T>>(ref x, xStride, ref y, yStride, ref destination, destinationStride, length, count);
        }
        else
        {
            return false;
        }

        return true;
    }

    // 単一区間は MapLine、複数区間は MapManyLines へ振り分ける。
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void MapLines<T, TKernel>(ref T x, int xStride, ref T destination, int destinationStride, int length, int count, TKernel kernel)
        where T : unmanaged
        where TKernel : IExactUnaryKernel<T>
    {
        if (count == 1)
        {
            MapLine(ref x, ref destination, length, kernel);
        }
        else
        {
            MapManyLines(ref x, xStride, ref destination, destinationStride, length, count, kernel);
        }
    }

    // 単一区間向け（外側ループのレジスタ圧迫を避けるため MapManyLines と分離し、即時最適化する）。
    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    private static void MapLine<T, TKernel>(ref T x, ref T destination, int length, TKernel kernel)
        where T : unmanaged
        where TKernel : IExactUnaryKernel<T>
    {
        ref T a = ref x;
        ref T c = ref destination;
        int i = 0;
        if (Vector512.IsHardwareAccelerated && Vector512<T>.IsSupported)
        {
            for (int width = Vector512<T>.Count; i <= length - width; i += width)
            {
                kernel.Invoke(Vector512.LoadUnsafe(ref a, (nuint)i)).StoreUnsafe(ref c, (nuint)i);
            }
        }
        else if (Vector256.IsHardwareAccelerated && Vector256<T>.IsSupported)
        {
            for (int width = Vector256<T>.Count; i <= length - width; i += width)
            {
                kernel.Invoke(Vector256.LoadUnsafe(ref a, (nuint)i)).StoreUnsafe(ref c, (nuint)i);
            }
        }
        else if (Vector128.IsHardwareAccelerated && Vector128<T>.IsSupported)
        {
            for (int width = Vector128<T>.Count; i <= length - width; i += width)
            {
                kernel.Invoke(Vector128.LoadUnsafe(ref a, (nuint)i)).StoreUnsafe(ref c, (nuint)i);
            }
        }

        for (; i < length; i++)
        {
            Unsafe.Add(ref c, i) = kernel.Invoke(Unsafe.Add(ref a, i));
        }
    }

    // 複数区間向け（外側ループと SIMD ループを同一メソッドで即時最適化する）。
    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    private static void MapManyLines<T, TKernel>(ref T x, int xStride, ref T destination, int destinationStride, int length, int count, TKernel kernel)
        where T : unmanaged
        where TKernel : IExactUnaryKernel<T>
    {
        for (int k = 0; k < count; k++)
        {
            ref T a = ref Unsafe.Add(ref x, (nint)k * xStride);
            ref T c = ref Unsafe.Add(ref destination, (nint)k * destinationStride);
            int i = 0;
            if (Vector512.IsHardwareAccelerated && Vector512<T>.IsSupported)
            {
                for (int width = Vector512<T>.Count; i <= length - width; i += width)
                {
                    kernel.Invoke(Vector512.LoadUnsafe(ref a, (nuint)i)).StoreUnsafe(ref c, (nuint)i);
                }
            }
            else if (Vector256.IsHardwareAccelerated && Vector256<T>.IsSupported)
            {
                for (int width = Vector256<T>.Count; i <= length - width; i += width)
                {
                    kernel.Invoke(Vector256.LoadUnsafe(ref a, (nuint)i)).StoreUnsafe(ref c, (nuint)i);
                }
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<T>.IsSupported)
            {
                for (int width = Vector128<T>.Count; i <= length - width; i += width)
                {
                    kernel.Invoke(Vector128.LoadUnsafe(ref a, (nuint)i)).StoreUnsafe(ref c, (nuint)i);
                }
            }

            for (; i < length; i++)
            {
                Unsafe.Add(ref c, i) = kernel.Invoke(Unsafe.Add(ref a, i));
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void ZipLines<T, TKernel>(ref T x, int xStride, ref T y, int yStride, ref T destination, int destinationStride, int length, int count)
        where T : unmanaged
        where TKernel : struct, IExactBinaryKernel<T>
    {
        for (int k = 0; k < count; k++)
        {
            ref T a = ref Unsafe.Add(ref x, (nint)k * xStride);
            ref T b = ref Unsafe.Add(ref y, (nint)k * yStride);
            ref T c = ref Unsafe.Add(ref destination, (nint)k * destinationStride);
            int i = 0;
            if (Vector512.IsHardwareAccelerated && Vector512<T>.IsSupported)
            {
                for (int width = Vector512<T>.Count; i <= length - width; i += width)
                {
                    default(TKernel).Invoke(Vector512.LoadUnsafe(ref a, (nuint)i), Vector512.LoadUnsafe(ref b, (nuint)i)).StoreUnsafe(ref c, (nuint)i);
                }
            }
            else if (Vector256.IsHardwareAccelerated && Vector256<T>.IsSupported)
            {
                for (int width = Vector256<T>.Count; i <= length - width; i += width)
                {
                    default(TKernel).Invoke(Vector256.LoadUnsafe(ref a, (nuint)i), Vector256.LoadUnsafe(ref b, (nuint)i)).StoreUnsafe(ref c, (nuint)i);
                }
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<T>.IsSupported)
            {
                for (int width = Vector128<T>.Count; i <= length - width; i += width)
                {
                    default(TKernel).Invoke(Vector128.LoadUnsafe(ref a, (nuint)i), Vector128.LoadUnsafe(ref b, (nuint)i)).StoreUnsafe(ref c, (nuint)i);
                }
            }

            for (; i < length; i++)
            {
                Unsafe.Add(ref c, i) = default(TKernel).Invoke(Unsafe.Add(ref a, i), Unsafe.Add(ref b, i));
            }
        }
    }

    // 単項カーネル（スカラーと各 SIMD 幅でビット単位に一致すること）。
    private interface IExactUnaryKernel<T> where T : unmanaged
    {
        T Invoke(T x);

        Vector128<T> Invoke(Vector128<T> x);

        Vector256<T> Invoke(Vector256<T> x);

        Vector512<T> Invoke(Vector512<T> x);
    }

    // 二項カーネル（スカラーと各 SIMD 幅でビット単位に一致すること）。
    private interface IExactBinaryKernel<T> where T : unmanaged
    {
        T Invoke(T x, T y);

        Vector128<T> Invoke(Vector128<T> x, Vector128<T> y);

        Vector256<T> Invoke(Vector256<T> x, Vector256<T> y);

        Vector512<T> Invoke(Vector512<T> x, Vector512<T> y);
    }

    private readonly struct SqrtKernel<T> : IExactUnaryKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
    {
        public T Invoke(T x) => XMath.Sqrt(x);
        public Vector128<T> Invoke(Vector128<T> x) => Vector128.Sqrt(x);
        public Vector256<T> Invoke(Vector256<T> x) => Vector256.Sqrt(x);
        public Vector512<T> Invoke(Vector512<T> x) => Vector512.Sqrt(x);
    }

    private readonly struct AbsKernel<T> : IExactUnaryKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
    {
        public T Invoke(T x) => XMath.Abs(x);
        public Vector128<T> Invoke(Vector128<T> x) => Vector128.Abs(x);
        public Vector256<T> Invoke(Vector256<T> x) => Vector256.Abs(x);
        public Vector512<T> Invoke(Vector512<T> x) => Vector512.Abs(x);
    }

    private readonly struct SquareKernel<T> : IExactUnaryKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
    {
        public T Invoke(T x) => XMath.Square(x);
        public Vector128<T> Invoke(Vector128<T> x) => x * x;
        public Vector256<T> Invoke(Vector256<T> x) => x * x;
        public Vector512<T> Invoke(Vector512<T> x) => x * x;
    }

    private readonly struct ReciprocalKernel<T> : IExactUnaryKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
    {
        public T Invoke(T x) => XMath.Reciprocal(x);
        public Vector128<T> Invoke(Vector128<T> x) => Vector128<T>.One / x;
        public Vector256<T> Invoke(Vector256<T> x) => Vector256<T>.One / x;
        public Vector512<T> Invoke(Vector512<T> x) => Vector512<T>.One / x;
    }

    // x > 0 または NaN なら x、それ以外（-0 を含む）は +0。
    private readonly struct ReluKernel<T> : IExactUnaryKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
    {
        public T Invoke(T x) => XMath.Relu(x);
        public Vector128<T> Invoke(Vector128<T> x) => Vector128.ConditionalSelect(Vector128.GreaterThan(x, Vector128<T>.Zero) | Vector128.IsNaN(x), x, Vector128<T>.Zero);
        public Vector256<T> Invoke(Vector256<T> x) => Vector256.ConditionalSelect(Vector256.GreaterThan(x, Vector256<T>.Zero) | Vector256.IsNaN(x), x, Vector256<T>.Zero);
        public Vector512<T> Invoke(Vector512<T> x) => Vector512.ConditionalSelect(Vector512.GreaterThan(x, Vector512<T>.Zero) | Vector512.IsNaN(x), x, Vector512<T>.Zero);
    }

    // 符号ビットのみ反転する（-(+0) = -0、NaN のペイロードを保持）。
    private readonly struct NegateKernel<T> : IExactUnaryKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
    {
        public T Invoke(T x) => -x;
        public Vector128<T> Invoke(Vector128<T> x) => -x;
        public Vector256<T> Invoke(Vector256<T> x) => -x;
        public Vector512<T> Invoke(Vector512<T> x) => -x;
    }

    private readonly struct AddScalarKernel<T>(T scalar) : IExactUnaryKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
    {
        public T Invoke(T x) => x + scalar;
        public Vector128<T> Invoke(Vector128<T> x) => x + Vector128.Create(scalar);
        public Vector256<T> Invoke(Vector256<T> x) => x + Vector256.Create(scalar);
        public Vector512<T> Invoke(Vector512<T> x) => x + Vector512.Create(scalar);
    }

    private readonly struct SubtractScalarKernel<T>(T scalar) : IExactUnaryKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
    {
        public T Invoke(T x) => x - scalar;
        public Vector128<T> Invoke(Vector128<T> x) => x - Vector128.Create(scalar);
        public Vector256<T> Invoke(Vector256<T> x) => x - Vector256.Create(scalar);
        public Vector512<T> Invoke(Vector512<T> x) => x - Vector512.Create(scalar);
    }

    private readonly struct ScaleKernel<T>(T scalar) : IExactUnaryKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
    {
        public T Invoke(T x) => x * scalar;
        public Vector128<T> Invoke(Vector128<T> x) => x * Vector128.Create(scalar);
        public Vector256<T> Invoke(Vector256<T> x) => x * Vector256.Create(scalar);
        public Vector512<T> Invoke(Vector512<T> x) => x * Vector512.Create(scalar);
    }

    private readonly struct DivideScalarKernel<T>(T y) : IExactUnaryKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
    {
        public T Invoke(T x) => XMath.Divide(x, y);
        public Vector128<T> Invoke(Vector128<T> x) => x / Vector128.Create(y);
        public Vector256<T> Invoke(Vector256<T> x) => x / Vector256.Create(y);
        public Vector512<T> Invoke(Vector512<T> x) => x / Vector512.Create(y);
    }

    private readonly struct MaxScalarKernel<T>(T y) : IExactUnaryKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
    {
        public T Invoke(T x) => XMath.Max(x, y);
        public Vector128<T> Invoke(Vector128<T> x) => Vector128.Max(x, Vector128.Create(y));
        public Vector256<T> Invoke(Vector256<T> x) => Vector256.Max(x, Vector256.Create(y));
        public Vector512<T> Invoke(Vector512<T> x) => Vector512.Max(x, Vector512.Create(y));
    }

    private readonly struct MinScalarKernel<T>(T y) : IExactUnaryKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
    {
        public T Invoke(T x) => XMath.Min(x, y);
        public Vector128<T> Invoke(Vector128<T> x) => Vector128.Min(x, Vector128.Create(y));
        public Vector256<T> Invoke(Vector256<T> x) => Vector256.Min(x, Vector256.Create(y));
        public Vector512<T> Invoke(Vector512<T> x) => Vector512.Min(x, Vector512.Create(y));
    }

    private readonly struct LeakyReluKernel<T>(T slope) : IExactUnaryKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
    {
        public T Invoke(T x) => XMath.LeakyRelu(x, slope);
        public Vector128<T> Invoke(Vector128<T> x) => Vector128.ConditionalSelect(Vector128.GreaterThan(x, Vector128<T>.Zero), x, x * Vector128.Create(slope));
        public Vector256<T> Invoke(Vector256<T> x) => Vector256.ConditionalSelect(Vector256.GreaterThan(x, Vector256<T>.Zero), x, x * Vector256.Create(slope));
        public Vector512<T> Invoke(Vector512<T> x) => Vector512.ConditionalSelect(Vector512.GreaterThan(x, Vector512<T>.Zero), x, x * Vector512.Create(slope));
    }

    // .NET 10 の Clamp と同じ NaN ペイロード保持規則を再現するため、Vector.Max/Min ではなく比較と選択で実装する（min <= max 前提）。
    private readonly struct ClampKernel<T>(T min, T max) : IExactUnaryKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
    {
        public T Invoke(T x) => XMath.Clamp(x, min, max);

        public Vector128<T> Invoke(Vector128<T> x)
        {
            var low = Vector128.Create(min);
            var high = Vector128.Create(max);
            var m = Vector128.ConditionalSelect(SelectMaximum(x, low), x, low);
            return Vector128.ConditionalSelect(SelectMinimum(m, high), m, high);
        }

        public Vector256<T> Invoke(Vector256<T> x)
        {
            var low = Vector256.Create(min);
            var high = Vector256.Create(max);
            var m = Vector256.ConditionalSelect(SelectMaximum(x, low), x, low);
            return Vector256.ConditionalSelect(SelectMinimum(m, high), m, high);
        }

        public Vector512<T> Invoke(Vector512<T> x)
        {
            var low = Vector512.Create(min);
            var high = Vector512.Create(max);
            var m = Vector512.ConditionalSelect(SelectMaximum(x, low), x, low);
            return Vector512.ConditionalSelect(SelectMinimum(m, high), m, high);
        }
    }

    // max(a, b) で a を選ぶ条件: a != b なら a が NaN または b < a、a == b なら b が負（-0）。
    // min(a, b) で a を選ぶ条件: a != b なら a が NaN または a < b、a == b なら a が負（-0）。
    private static Vector128<T> SelectMaximum<T>(Vector128<T> a, Vector128<T> b) where T : unmanaged, IFloatingPointIeee754<T> =>
        Vector128.ConditionalSelect(~Vector128.Equals(a, b), Vector128.IsNaN(a) | Vector128.LessThan(b, a), Vector128.IsNegative(b));

    private static Vector256<T> SelectMaximum<T>(Vector256<T> a, Vector256<T> b) where T : unmanaged, IFloatingPointIeee754<T> =>
        Vector256.ConditionalSelect(~Vector256.Equals(a, b), Vector256.IsNaN(a) | Vector256.LessThan(b, a), Vector256.IsNegative(b));

    private static Vector512<T> SelectMaximum<T>(Vector512<T> a, Vector512<T> b) where T : unmanaged, IFloatingPointIeee754<T> =>
        Vector512.ConditionalSelect(~Vector512.Equals(a, b), Vector512.IsNaN(a) | Vector512.LessThan(b, a), Vector512.IsNegative(b));

    private static Vector128<T> SelectMinimum<T>(Vector128<T> a, Vector128<T> b) where T : unmanaged, IFloatingPointIeee754<T> =>
        Vector128.ConditionalSelect(~Vector128.Equals(a, b), Vector128.IsNaN(a) | Vector128.LessThan(a, b), Vector128.IsNegative(a));

    private static Vector256<T> SelectMinimum<T>(Vector256<T> a, Vector256<T> b) where T : unmanaged, IFloatingPointIeee754<T> =>
        Vector256.ConditionalSelect(~Vector256.Equals(a, b), Vector256.IsNaN(a) | Vector256.LessThan(a, b), Vector256.IsNegative(a));

    private static Vector512<T> SelectMinimum<T>(Vector512<T> a, Vector512<T> b) where T : unmanaged, IFloatingPointIeee754<T> =>
        Vector512.ConditionalSelect(~Vector512.Equals(a, b), Vector512.IsNaN(a) | Vector512.LessThan(a, b), Vector512.IsNegative(a));

    private readonly struct AddKernel<T> : IExactBinaryKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
    {
        public T Invoke(T x, T y) => x + y;
        public Vector128<T> Invoke(Vector128<T> x, Vector128<T> y) => x + y;
        public Vector256<T> Invoke(Vector256<T> x, Vector256<T> y) => x + y;
        public Vector512<T> Invoke(Vector512<T> x, Vector512<T> y) => x + y;
    }

    private readonly struct SubtractKernel<T> : IExactBinaryKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
    {
        public T Invoke(T x, T y) => x - y;
        public Vector128<T> Invoke(Vector128<T> x, Vector128<T> y) => x - y;
        public Vector256<T> Invoke(Vector256<T> x, Vector256<T> y) => x - y;
        public Vector512<T> Invoke(Vector512<T> x, Vector512<T> y) => x - y;
    }

    private readonly struct MultiplyKernel<T> : IExactBinaryKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
    {
        public T Invoke(T x, T y) => x * y;
        public Vector128<T> Invoke(Vector128<T> x, Vector128<T> y) => x * y;
        public Vector256<T> Invoke(Vector256<T> x, Vector256<T> y) => x * y;
        public Vector512<T> Invoke(Vector512<T> x, Vector512<T> y) => x * y;
    }

    private readonly struct DivideKernel<T> : IExactBinaryKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
    {
        public T Invoke(T x, T y) => XMath.Divide(x, y);
        public Vector128<T> Invoke(Vector128<T> x, Vector128<T> y) => x / y;
        public Vector256<T> Invoke(Vector256<T> x, Vector256<T> y) => x / y;
        public Vector512<T> Invoke(Vector512<T> x, Vector512<T> y) => x / y;
    }

    private readonly struct MaxKernel<T> : IExactBinaryKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
    {
        public T Invoke(T x, T y) => XMath.Max(x, y);
        public Vector128<T> Invoke(Vector128<T> x, Vector128<T> y) => Vector128.Max(x, y);
        public Vector256<T> Invoke(Vector256<T> x, Vector256<T> y) => Vector256.Max(x, y);
        public Vector512<T> Invoke(Vector512<T> x, Vector512<T> y) => Vector512.Max(x, y);
    }

    private readonly struct MinKernel<T> : IExactBinaryKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
    {
        public T Invoke(T x, T y) => XMath.Min(x, y);
        public Vector128<T> Invoke(Vector128<T> x, Vector128<T> y) => Vector128.Min(x, y);
        public Vector256<T> Invoke(Vector256<T> x, Vector256<T> y) => Vector256.Min(x, y);
        public Vector512<T> Invoke(Vector512<T> x, Vector512<T> y) => Vector512.Min(x, y);
    }
}
