using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace Tuna.LinearAlgebra;

internal static partial class DenseElementwise
{
    // count 本の連続区間とスカラーの二項演算。出力先は値の側と完全一致（インプレース）でもよい。未対応なら何も書き込まず false を返す。
    internal static bool TryZipScalarLines<T, TOperator>(ref readonly T values, int valueStride, ref readonly T scalars, int scalarStride,
        ref T destination, int destinationStride, int length, int count, bool scalarOnLeft)
        where T : unmanaged
    {
        if (typeof(T) == typeof(double))
        {
            return TryZipScalarLinesReal<double, TOperator>(ref Unsafe.As<T, double>(ref Unsafe.AsRef(in values)), valueStride,
                ref Unsafe.As<T, double>(ref Unsafe.AsRef(in scalars)), scalarStride,
                ref Unsafe.As<T, double>(ref destination), destinationStride, length, count, scalarOnLeft);
        }

        if (typeof(T) == typeof(float))
        {
            return TryZipScalarLinesReal<float, TOperator>(ref Unsafe.As<T, float>(ref Unsafe.AsRef(in values)), valueStride,
                ref Unsafe.As<T, float>(ref Unsafe.AsRef(in scalars)), scalarStride,
                ref Unsafe.As<T, float>(ref destination), destinationStride, length, count, scalarOnLeft);
        }

        return false;
    }

    private static bool TryZipScalarLinesReal<T, TOperator>(ref T values, int valueStride, ref T scalars, int scalarStride,
        ref T destination, int destinationStride, int length, int count, bool scalarOnLeft)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (typeof(TOperator) == typeof(AddOperator<T>))
        {
            ZipScalarLines<T, AddKernel<T>>(ref values, valueStride, ref scalars, scalarStride, ref destination, destinationStride, length, count, scalarOnLeft);
        }
        else if (typeof(TOperator) == typeof(SubtractOperator<T>))
        {
            ZipScalarLines<T, SubtractKernel<T>>(ref values, valueStride, ref scalars, scalarStride, ref destination, destinationStride, length, count, scalarOnLeft);
        }
        else if (typeof(TOperator) == typeof(MultiplyOperator<T>))
        {
            ZipScalarLines<T, MultiplyKernel<T>>(ref values, valueStride, ref scalars, scalarStride, ref destination, destinationStride, length, count, scalarOnLeft);
        }
        else if (typeof(TOperator) == typeof(StaticBinaryOperator<T, XMath.Operators.Divide<T>>))
        {
            ZipScalarLines<T, DivideKernel<T>>(ref values, valueStride, ref scalars, scalarStride, ref destination, destinationStride, length, count, scalarOnLeft);
        }
        else if (typeof(TOperator) == typeof(StaticBinaryOperator<T, XMath.Operators.Hypot<T>>))
        {
            ZipScalarLines<T, HypotKernel<T>>(ref values, valueStride, ref scalars, scalarStride, ref destination, destinationStride, length, count, scalarOnLeft);
        }
        else if (typeof(TOperator) == typeof(StaticBinaryOperator<T, XMath.Operators.Min<T>>))
        {
            ZipScalarLines<T, MinKernel<T>>(ref values, valueStride, ref scalars, scalarStride, ref destination, destinationStride, length, count, scalarOnLeft);
        }
        else if (typeof(TOperator) == typeof(StaticBinaryOperator<T, XMath.Operators.Max<T>>))
        {
            ZipScalarLines<T, MaxKernel<T>>(ref values, valueStride, ref scalars, scalarStride, ref destination, destinationStride, length, count, scalarOnLeft);
        }
        else
        {
            return false;
        }

        return true;
    }

    // スカラーが左のときは引数を入れ替えたカーネルに差し替え、内側ループから左右の分岐を排除する。
    private static void ZipScalarLines<T, TKernel>(ref T values, int valueStride, ref T scalars, int scalarStride,
        ref T destination, int destinationStride, int length, int count, bool scalarOnLeft)
        where T : unmanaged
        where TKernel : struct, IExactBinaryKernel<T>
    {
        if (scalarOnLeft)
        {
            ZipScalarLinesCore<T, SwappedKernel<T, TKernel>>(ref values, valueStride, ref scalars, scalarStride,
                ref destination, destinationStride, length, count);
        }
        else
        {
            ZipScalarLinesCore<T, TKernel>(ref values, valueStride, ref scalars, scalarStride, ref destination, destinationStride, length, count);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void ZipScalarLinesCore<T, TKernel>(ref T values, int valueStride, ref T scalars, int scalarStride,
        ref T destination, int destinationStride, int length, int count)
        where T : unmanaged
        where TKernel : struct, IExactBinaryKernel<T>
    {
        for (int k = 0; k < count; k++)
        {
            ref T a = ref Unsafe.Add(ref values, (nint)k * valueStride);
            ref T c = ref Unsafe.Add(ref destination, (nint)k * destinationStride);
            T scalar = Unsafe.Add(ref scalars, (nint)k * scalarStride);
            int i = 0;
            if (Vector512.IsHardwareAccelerated && Vector512<T>.IsSupported)
            {
                var y = Vector512.Create(scalar);
                for (int width = Vector512<T>.Count; i <= length - width; i += width)
                {
                    default(TKernel).Invoke(Vector512.LoadUnsafe(ref a, (nuint)i), y).StoreUnsafe(ref c, (nuint)i);
                }
            }
            else if (Vector256.IsHardwareAccelerated && Vector256<T>.IsSupported)
            {
                var y = Vector256.Create(scalar);
                for (int width = Vector256<T>.Count; i <= length - width; i += width)
                {
                    default(TKernel).Invoke(Vector256.LoadUnsafe(ref a, (nuint)i), y).StoreUnsafe(ref c, (nuint)i);
                }
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<T>.IsSupported)
            {
                var y = Vector128.Create(scalar);
                for (int width = Vector128<T>.Count; i <= length - width; i += width)
                {
                    default(TKernel).Invoke(Vector128.LoadUnsafe(ref a, (nuint)i), y).StoreUnsafe(ref c, (nuint)i);
                }
            }

            for (; i < length; i++)
            {
                Unsafe.Add(ref c, i) = default(TKernel).Invoke(Unsafe.Add(ref a, i), scalar);
            }
        }
    }

    private readonly struct SwappedKernel<T, TKernel> : IExactBinaryKernel<T>
        where T : unmanaged
        where TKernel : struct, IExactBinaryKernel<T>
    {
        public T Invoke(T x, T y) => default(TKernel).Invoke(y, x);
        public Vector128<T> Invoke(Vector128<T> x, Vector128<T> y) => default(TKernel).Invoke(y, x);
        public Vector256<T> Invoke(Vector256<T> x, Vector256<T> y) => default(TKernel).Invoke(y, x);
        public Vector512<T> Invoke(Vector512<T> x, Vector512<T> y) => default(TKernel).Invoke(y, x);
    }
}
