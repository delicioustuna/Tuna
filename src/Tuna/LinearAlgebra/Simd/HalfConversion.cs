using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace Tuna.LinearAlgebra;

// Half <-> float 変換の SIMD 実装（.NET 標準のキャストと NaN ペイロードを含めビット単位で一致）。
// AVX-SSE 遷移ペナルティを避けるため、ベクトル処理とスカラー端数処理を別メソッドに分離する。
internal static class HalfConversion
{
    internal static void Widen(ReadOnlySpan<Half> source, Span<float> destination)
    {
        DenseStorage.RequireLength(source.Length, destination.Length);
        int done = WidenVectors(source, destination);
        WidenScalars(source[done..], destination[done..]);
    }

    internal static void Narrow(ReadOnlySpan<float> source, Span<Half> destination)
    {
        DenseStorage.RequireLength(source.Length, destination.Length);
        int done = NarrowVectors(source, destination);
        NarrowScalars(source[done..], destination[done..]);
    }

    private static void WidenScalars(ReadOnlySpan<Half> source, Span<float> destination)
    {
        for (int i = 0; i < source.Length; i++)
        {
            destination[i] = (float)source[i];
        }
    }

    private static void NarrowScalars(ReadOnlySpan<float> source, Span<Half> destination)
    {
        for (int i = 0; i < source.Length; i++)
        {
            destination[i] = (Half)source[i];
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static int WidenVectors(ReadOnlySpan<Half> source, Span<float> destination)
    {
        ref ushort a = ref Unsafe.As<Half, ushort>(ref MemoryMarshal.GetReference(source));
        ref uint c = ref Unsafe.As<float, uint>(ref MemoryMarshal.GetReference(destination));
        int n = source.Length, i = 0;
        if (Vector512.IsHardwareAccelerated)
        {
            for (int width = Vector512<ushort>.Count; i <= n - width; i += width)
            {
                var (lower, upper) = Vector512.Widen(Vector512.LoadUnsafe(ref a, (nuint)i));
                SingleBits(lower).StoreUnsafe(ref c, (nuint)i);
                SingleBits(upper).StoreUnsafe(ref c, (nuint)(i + width / 2));
            }
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            for (int width = Vector256<ushort>.Count; i <= n - width; i += width)
            {
                var (lower, upper) = Vector256.Widen(Vector256.LoadUnsafe(ref a, (nuint)i));
                SingleBits(lower).StoreUnsafe(ref c, (nuint)i);
                SingleBits(upper).StoreUnsafe(ref c, (nuint)(i + width / 2));
            }
        }
        else if (Vector128.IsHardwareAccelerated)
        {
            for (int width = Vector128<ushort>.Count; i <= n - width; i += width)
            {
                var (lower, upper) = Vector128.Widen(Vector128.LoadUnsafe(ref a, (nuint)i));
                SingleBits(lower).StoreUnsafe(ref c, (nuint)i);
                SingleBits(upper).StoreUnsafe(ref c, (nuint)(i + width / 2));
            }
        }

        return i;
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static int NarrowVectors(ReadOnlySpan<float> source, Span<Half> destination)
    {
        ref float a = ref MemoryMarshal.GetReference(source);
        ref ushort c = ref Unsafe.As<Half, ushort>(ref MemoryMarshal.GetReference(destination));
        int n = source.Length, i = 0;
        if (Vector512.IsHardwareAccelerated)
        {
            for (int width = Vector512<ushort>.Count; i <= n - width; i += width)
            {
                var lower = HalfBits(Vector512.LoadUnsafe(ref a, (nuint)i));
                var upper = HalfBits(Vector512.LoadUnsafe(ref a, (nuint)(i + width / 2)));
                Vector512.Narrow(lower, upper).StoreUnsafe(ref c, (nuint)i);
            }
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            for (int width = Vector256<ushort>.Count; i <= n - width; i += width)
            {
                var lower = HalfBits(Vector256.LoadUnsafe(ref a, (nuint)i));
                var upper = HalfBits(Vector256.LoadUnsafe(ref a, (nuint)(i + width / 2)));
                Vector256.Narrow(lower, upper).StoreUnsafe(ref c, (nuint)i);
            }
        }
        else if (Vector128.IsHardwareAccelerated)
        {
            for (int width = Vector128<ushort>.Count; i <= n - width; i += width)
            {
                var lower = HalfBits(Vector128.LoadUnsafe(ref a, (nuint)i));
                var upper = HalfBits(Vector128.LoadUnsafe(ref a, (nuint)(i + width / 2)));
                Vector128.Narrow(lower, upper).StoreUnsafe(ref c, (nuint)i);
            }
        }

        return i;
    }

    // 下位 16 ビットに Half のビット列を置いた h から float のビット列へ
    private static Vector512<uint> SingleBits(Vector512<uint> h)
    {
        var sign = (h << 16) & Vector512.Create(0x8000_0000u);
        var exponent = h & Vector512.Create(0x7C00u);
        // 非正規化数（と 0）なら 2^-14、それ以外は 0
        var lowerBound = Vector512.Equals(exponent, Vector512<uint>.Zero) & Vector512.Create(0x3880_0000u);
        // 指数に足す数（∞・NaN は指数が最大のまま残るよう 2 倍）
        var offset = Vector512.Create(0x3800_0000u) | lowerBound;
        offset = Vector512.ConditionalSelect(Vector512.Equals(exponent, Vector512.Create(0x7C00u)), offset << 1, offset);
        var bits = ((h << 13) & Vector512.Create(0x0FFF_E000u)) + offset;
        return (bits.AsSingle() - lowerBound.AsSingle()).AsUInt32() | sign;
    }

    private static Vector256<uint> SingleBits(Vector256<uint> h)
    {
        var sign = (h << 16) & Vector256.Create(0x8000_0000u);
        var exponent = h & Vector256.Create(0x7C00u);
        var lowerBound = Vector256.Equals(exponent, Vector256<uint>.Zero) & Vector256.Create(0x3880_0000u);
        var offset = Vector256.Create(0x3800_0000u) | lowerBound;
        offset = Vector256.ConditionalSelect(Vector256.Equals(exponent, Vector256.Create(0x7C00u)), offset << 1, offset);
        var bits = ((h << 13) & Vector256.Create(0x0FFF_E000u)) + offset;
        return (bits.AsSingle() - lowerBound.AsSingle()).AsUInt32() | sign;
    }

    private static Vector128<uint> SingleBits(Vector128<uint> h)
    {
        var sign = (h << 16) & Vector128.Create(0x8000_0000u);
        var exponent = h & Vector128.Create(0x7C00u);
        var lowerBound = Vector128.Equals(exponent, Vector128<uint>.Zero) & Vector128.Create(0x3880_0000u);
        var offset = Vector128.Create(0x3800_0000u) | lowerBound;
        offset = Vector128.ConditionalSelect(Vector128.Equals(exponent, Vector128.Create(0x7C00u)), offset << 1, offset);
        var bits = ((h << 13) & Vector128.Create(0x0FFF_E000u)) + offset;
        return (bits.AsSingle() - lowerBound.AsSingle()).AsUInt32() | sign;
    }

    // float から、下位 16 ビットに Half のビット列を置いた値へ（上位 16 ビットは Narrow で捨てる）
    private static Vector512<uint> HalfBits(Vector512<float> value)
    {
        var sign = (value.AsUInt32() & Vector512.Create(0x8000_0000u)) >> 16;
        // NaN なら 0、それ以外は ~0
        var real = Vector512.Equals(value, value).AsUInt32();
        // Half で ∞ になる大きさは 65520 にそろえ、2^-14 未満は非正規化数の丸めの位置（2^-24）に合わせる
        var magnitude = Vector512.Min(Vector512.Create(65520.0f), Vector512.Abs(value));
        var exponent = Vector512.Max(magnitude, Vector512.Create(0x3880_0000u).AsSingle()).AsUInt32() & Vector512.Create(0x7F80_0000u);
        // Half の仮数の最下位の位置に 1 を置いた数を足して丸める
        var bits = (magnitude + (exponent + Vector512.Create(0x0680_0000u)).AsSingle()).AsUInt32() - Vector512.Create(0x3F00_0000u);
        var nanExponent = ~real & Vector512.Create(0x7C00u);
        // 仮数の繰り上がりは指数に入る。NaN は仮数を消してから指数を最大にする
        bits = ((bits & real) + (bits >> 13)) & ~nanExponent;
        return bits | nanExponent | sign;
    }

    private static Vector256<uint> HalfBits(Vector256<float> value)
    {
        var sign = (value.AsUInt32() & Vector256.Create(0x8000_0000u)) >> 16;
        var real = Vector256.Equals(value, value).AsUInt32();
        var magnitude = Vector256.Min(Vector256.Create(65520.0f), Vector256.Abs(value));
        var exponent = Vector256.Max(magnitude, Vector256.Create(0x3880_0000u).AsSingle()).AsUInt32() & Vector256.Create(0x7F80_0000u);
        var bits = (magnitude + (exponent + Vector256.Create(0x0680_0000u)).AsSingle()).AsUInt32() - Vector256.Create(0x3F00_0000u);
        var nanExponent = ~real & Vector256.Create(0x7C00u);
        bits = ((bits & real) + (bits >> 13)) & ~nanExponent;
        return bits | nanExponent | sign;
    }

    private static Vector128<uint> HalfBits(Vector128<float> value)
    {
        var sign = (value.AsUInt32() & Vector128.Create(0x8000_0000u)) >> 16;
        var real = Vector128.Equals(value, value).AsUInt32();
        var magnitude = Vector128.Min(Vector128.Create(65520.0f), Vector128.Abs(value));
        var exponent = Vector128.Max(magnitude, Vector128.Create(0x3880_0000u).AsSingle()).AsUInt32() & Vector128.Create(0x7F80_0000u);
        var bits = (magnitude + (exponent + Vector128.Create(0x0680_0000u)).AsSingle()).AsUInt32() - Vector128.Create(0x3F00_0000u);
        var nanExponent = ~real & Vector128.Create(0x7C00u);
        bits = ((bits & real) + (bits >> 13)) & ~nanExponent;
        return bits | nanExponent | sign;
    }
}
