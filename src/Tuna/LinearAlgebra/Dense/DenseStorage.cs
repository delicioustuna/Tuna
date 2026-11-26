using System.Numerics;

namespace Tuna.LinearAlgebra;

// 公開コンテナの形状検証と要素型変換（演算カーネル非依存）。
internal static class DenseStorage
{
    internal static int Length(int rows, int columns)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(rows);
        ArgumentOutOfRangeException.ThrowIfNegative(columns);
        return checked(rows * columns);
    }

    internal static int Length(int d0, int d1, int d2)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(d0);
        ArgumentOutOfRangeException.ThrowIfNegative(d1);
        ArgumentOutOfRangeException.ThrowIfNegative(d2);
        return d0 == 0 || d1 == 0 || d2 == 0 ? 0 : checked(checked(d0 * d1) * d2);
    }

    // 列優先バッチの行列間隔（行列が 1 つ以下なら未使用のため 0）。
    internal static int MatrixStride(int rows, int columns, int count) =>
        count > 1 && rows > 0 && columns > 0 ? checked(rows * columns) : 0;

    internal static void RequireLength(int expected, int actual)
    {
        if (expected != actual)
        {
            throw new ArgumentException("The element count must match the shape.");
        }
    }

    internal static void CheckIndex(int index, int length, string name)
    {
        if ((uint)index >= (uint)length)
        {
            throw new ArgumentOutOfRangeException(name);
        }
    }

    internal static T[] Allocate<T>(int length) where T : unmanaged, INumberBase<T>
    {
        var data = new T[length];
        // INumberBase does not require default(T) to be the additive identity.
        if (T.Zero != default(T))
        {
            data.AsSpan().Fill(T.Zero);
        }

        return data;
    }

    internal static TTo[] ConvertChecked<T, TTo>(ReadOnlySpan<T> source)
        where T : unmanaged, INumberBase<T>
        where TTo : unmanaged, INumberBase<TTo>
    {
        var result = new TTo[source.Length];
        for (int i = 0; i < source.Length; i++)
        {
            result[i] = TTo.CreateChecked(source[i]);
        }

        return result;
    }
}
