using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace Tuna.LinearAlgebra;

// 複素数の絶対値 |z|（Complex.Magnitude）の SIMD 実装。
// .NET 10 の double.Hypot（AMD AOCL 由来、FMA 未使用）と同じ演算順序をレーンごとの選択で再現し、ビット単位で一致させる。
internal static partial class DenseElementwise
{
    private const ulong HypotScaleBits = 600UL << 52;           // 2^±600 の倍率を指数部に加減する
    private const double HypotDown = 4.149515568880993E+180;    // 2^600（縮小した値を掛け戻す）
    private const double HypotUp = 2.409919865102884E-181;      // 2^-600
    private const double HypotSubnormalBias = 9.232978617785736E-128; // 2^-422（非正規化数を正規化数に変換する際の補正）
    private const ulong HypotHeadMask = 0xFFFFFFFFF8000000;     // 仮数の上位 26 ビット（2 乗が丸めなしで求まる）

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void Magnitude(ReadOnlySpan<Complex> x, Span<double> destination)
    {
        DenseStorage.RequireLength(x.Length, destination.Length);
        ref double a = ref MemoryMarshal.GetReference(MemoryMarshal.Cast<Complex, double>(x));
        ref double c = ref MemoryMarshal.GetReference(destination);
        int n = x.Length;
        int i = 0;
        if (Vector512.IsHardwareAccelerated)
        {
            var order = Vector512.Create(0L, 2, 4, 6, 1, 3, 5, 7);
            for (int width = Vector512<double>.Count; i <= n - width; i += width)
            {
                // 実部と虚部が交互に並ぶので、それぞれを1本のベクトルに集める
                var first = Vector512.Shuffle(Vector512.LoadUnsafe(ref a, (nuint)(2 * i)), order);
                var second = Vector512.Shuffle(Vector512.LoadUnsafe(ref a, (nuint)(2 * i + width)), order);
                Hypot(Vector512.Create(first.GetLower(), second.GetLower()), Vector512.Create(first.GetUpper(), second.GetUpper())).StoreUnsafe(ref c, (nuint)i);
            }
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            var order = Vector256.Create(0L, 2, 1, 3);
            for (int width = Vector256<double>.Count; i <= n - width; i += width)
            {
                var first = Vector256.Shuffle(Vector256.LoadUnsafe(ref a, (nuint)(2 * i)), order);
                var second = Vector256.Shuffle(Vector256.LoadUnsafe(ref a, (nuint)(2 * i + width)), order);
                Hypot(Vector256.Create(first.GetLower(), second.GetLower()), Vector256.Create(first.GetUpper(), second.GetUpper())).StoreUnsafe(ref c, (nuint)i);
            }
        }
        else if (Vector128.IsHardwareAccelerated)
        {
            for (int width = Vector128<double>.Count; i <= n - width; i += width)
            {
                var first = Vector128.LoadUnsafe(ref a, (nuint)(2 * i));
                var second = Vector128.LoadUnsafe(ref a, (nuint)(2 * i + width));
                Hypot(Vector128.Create(first.ToScalar(), second.ToScalar()), Vector128.Create(first.GetElement(1), second.GetElement(1))).StoreUnsafe(ref c, (nuint)i);
            }
        }

        for (; i < n; i++)
        {
            destination[i] = double.Hypot(x[i].Real, x[i].Imaginary);
        }
    }

    // double.Hypot と同じ式。各ステップの対応は Vector256 版のコメントを参照
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static Vector128<double> Hypot(Vector128<double> x, Vector128<double> y)
    {
        var ax = Vector128.Abs(x);
        var ay = Vector128.Abs(y);
        var xBits = ax.AsUInt64();
        var yBits = ay.AsUInt64();
        var xExponent = (xBits >> 52) & Vector128.Create(0x7FFUL);
        var yExponent = (yBits >> 52) & Vector128.Create(0x7FFUL);
        var difference = xExponent.AsInt64() - yExponent.AsInt64();
        var large = Vector128.GreaterThan(xExponent, Vector128.Create(1523UL)) | Vector128.GreaterThan(yExponent, Vector128.Create(1523UL));
        var small = ~large & (Vector128.LessThan(xExponent, Vector128.Create(523UL)) | Vector128.LessThan(yExponent, Vector128.Create(523UL)));
        var shift = Vector128.Create(HypotScaleBits);
        xBits = Vector128.ConditionalSelect(large, xBits - shift, Vector128.ConditionalSelect(small, xBits + shift, xBits));
        yBits = Vector128.ConditionalSelect(large, yBits - shift, Vector128.ConditionalSelect(small, yBits + shift, yBits));
        var factor = Vector128.ConditionalSelect(large.AsDouble(), Vector128.Create(HypotDown), Vector128.ConditionalSelect(small.AsDouble(), Vector128.Create(HypotUp), Vector128<double>.One));
        var implicitBit = Vector128.Create(1UL << 52);
        var bias = Vector128.Create(HypotSubnormalBias);
        xBits = Vector128.ConditionalSelect(small & Vector128.Equals(xExponent, Vector128<ulong>.Zero), ((xBits + implicitBit).AsDouble() - bias).AsUInt64(), xBits);
        yBits = Vector128.ConditionalSelect(small & Vector128.Equals(yExponent, Vector128<ulong>.Zero), ((yBits + implicitBit).AsDouble() - bias).AsUInt64(), yBits);
        var swap = Vector128.LessThan(xBits.AsDouble(), yBits.AsDouble()).AsUInt64();
        var hiBits = Vector128.ConditionalSelect(swap, yBits, xBits);
        var loBits = Vector128.ConditionalSelect(swap, xBits, yBits);
        var hi = hiBits.AsDouble();
        var lo = loBits.AsDouble();
        var hiHead = (hiBits & Vector128.Create(HypotHeadMask)).AsDouble();
        var loHead = (loBits & Vector128.Create(HypotHeadMask)).AsDouble();
        var hiTail = hi - hiHead;
        var loTail = lo - loHead;
        var hh = hi * hi;
        var ll = lo * lo;
        var sum = hh + ll;
        var tail = hh - sum + ll;
        tail += hiHead * hiHead - hh;
        tail += Vector128.Create(2.0) * hiHead * hiTail;
        tail += hiTail * hiTail;
        var tailSameExponent = tail + (loHead * loHead - ll);
        tailSameExponent += Vector128.Create(2.0) * loHead * loTail;
        tailSameExponent += loTail * loTail;
        tail = Vector128.ConditionalSelect(Vector128.Equals(difference, Vector128<long>.Zero).AsDouble(), tailSameExponent, tail);
        var result = Vector128.Sqrt(sum + tail) * factor;
        var near = Vector128.LessThanOrEqual(difference, Vector128.Create(54L)) & Vector128.GreaterThanOrEqual(difference, Vector128.Create(-54L));
        result = Vector128.ConditionalSelect(near.AsDouble(), result, ax + ay);
        result = Vector128.ConditionalSelect(Vector128.Equals(ay, Vector128<double>.Zero), ax, result);
        result = Vector128.ConditionalSelect(Vector128.Equals(ax, Vector128<double>.Zero), ay, result);
        var infinity = Vector128.Create(double.PositiveInfinity);
        var finite = Vector128.LessThan(ax, infinity) & Vector128.LessThan(ay, infinity);
        var infinite = Vector128.Equals(ax, infinity) | Vector128.Equals(ay, infinity);
        return Vector128.ConditionalSelect(finite, result, Vector128.ConditionalSelect(infinite, infinity, Vector128.Create(double.NaN)));
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static Vector256<double> Hypot(Vector256<double> x, Vector256<double> y)
    {
        var ax = Vector256.Abs(x);
        var ay = Vector256.Abs(y);
        var xBits = ax.AsUInt64();
        var yBits = ay.AsUInt64();
        var xExponent = (xBits >> 52) & Vector256.Create(0x7FFUL);
        var yExponent = (yBits >> 52) & Vector256.Create(0x7FFUL);
        var difference = xExponent.AsInt64() - yExponent.AsInt64();
        // 指数の差が 54 以内なら、過大または過小な組を 2^∓600 倍してオーバーフローやアンダーフローを回避する
        var large = Vector256.GreaterThan(xExponent, Vector256.Create(1523UL)) | Vector256.GreaterThan(yExponent, Vector256.Create(1523UL));
        var small = ~large & (Vector256.LessThan(xExponent, Vector256.Create(523UL)) | Vector256.LessThan(yExponent, Vector256.Create(523UL)));
        var shift = Vector256.Create(HypotScaleBits);
        xBits = Vector256.ConditionalSelect(large, xBits - shift, Vector256.ConditionalSelect(small, xBits + shift, xBits));
        yBits = Vector256.ConditionalSelect(large, yBits - shift, Vector256.ConditionalSelect(small, yBits + shift, yBits));
        var factor = Vector256.ConditionalSelect(large.AsDouble(), Vector256.Create(HypotDown), Vector256.ConditionalSelect(small.AsDouble(), Vector256.Create(HypotUp), Vector256<double>.One));
        // 非正規化数は、暗黙の 1 を足した正規化数から 2^-422 を引いて、2^600 倍した値に直す
        var implicitBit = Vector256.Create(1UL << 52);
        var bias = Vector256.Create(HypotSubnormalBias);
        xBits = Vector256.ConditionalSelect(small & Vector256.Equals(xExponent, Vector256<ulong>.Zero), ((xBits + implicitBit).AsDouble() - bias).AsUInt64(), xBits);
        yBits = Vector256.ConditionalSelect(small & Vector256.Equals(yExponent, Vector256<ulong>.Zero), ((yBits + implicitBit).AsDouble() - bias).AsUInt64(), yBits);
        // 大きいほうを hi にする
        var swap = Vector256.LessThan(xBits.AsDouble(), yBits.AsDouble()).AsUInt64();
        var hiBits = Vector256.ConditionalSelect(swap, yBits, xBits);
        var loBits = Vector256.ConditionalSelect(swap, xBits, yBits);
        var hi = hiBits.AsDouble();
        var lo = loBits.AsDouble();
        // hi² + lo² を上位と下位に分けて、丸め誤差を下位に集める
        var hiHead = (hiBits & Vector256.Create(HypotHeadMask)).AsDouble();
        var loHead = (loBits & Vector256.Create(HypotHeadMask)).AsDouble();
        var hiTail = hi - hiHead;
        var loTail = lo - loHead;
        var hh = hi * hi;
        var ll = lo * lo;
        var sum = hh + ll;
        var tail = hh - sum + ll;
        tail += hiHead * hiHead - hh;
        tail += Vector256.Create(2.0) * hiHead * hiTail;
        tail += hiTail * hiTail;
        // 指数が等しいときだけ lo² の丸め誤差も足す
        var tailSameExponent = tail + (loHead * loHead - ll);
        tailSameExponent += Vector256.Create(2.0) * loHead * loTail;
        tailSameExponent += loTail * loTail;
        tail = Vector256.ConditionalSelect(Vector256.Equals(difference, Vector256<long>.Zero).AsDouble(), tailSameExponent, tail);
        var result = Vector256.Sqrt(sum + tail) * factor;
        // 指数の差が 54 を超えれば |x| + |y|、片方が 0 ならもう一方、∞ を含めば ∞、それ以外の NaN を含めば NaN
        var near = Vector256.LessThanOrEqual(difference, Vector256.Create(54L)) & Vector256.GreaterThanOrEqual(difference, Vector256.Create(-54L));
        result = Vector256.ConditionalSelect(near.AsDouble(), result, ax + ay);
        result = Vector256.ConditionalSelect(Vector256.Equals(ay, Vector256<double>.Zero), ax, result);
        result = Vector256.ConditionalSelect(Vector256.Equals(ax, Vector256<double>.Zero), ay, result);
        var infinity = Vector256.Create(double.PositiveInfinity);
        var finite = Vector256.LessThan(ax, infinity) & Vector256.LessThan(ay, infinity);
        var infinite = Vector256.Equals(ax, infinity) | Vector256.Equals(ay, infinity);
        return Vector256.ConditionalSelect(finite, result, Vector256.ConditionalSelect(infinite, infinity, Vector256.Create(double.NaN)));
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static Vector512<double> Hypot(Vector512<double> x, Vector512<double> y)
    {
        var ax = Vector512.Abs(x);
        var ay = Vector512.Abs(y);
        var xBits = ax.AsUInt64();
        var yBits = ay.AsUInt64();
        var xExponent = (xBits >> 52) & Vector512.Create(0x7FFUL);
        var yExponent = (yBits >> 52) & Vector512.Create(0x7FFUL);
        var difference = xExponent.AsInt64() - yExponent.AsInt64();
        var large = Vector512.GreaterThan(xExponent, Vector512.Create(1523UL)) | Vector512.GreaterThan(yExponent, Vector512.Create(1523UL));
        var small = ~large & (Vector512.LessThan(xExponent, Vector512.Create(523UL)) | Vector512.LessThan(yExponent, Vector512.Create(523UL)));
        var shift = Vector512.Create(HypotScaleBits);
        xBits = Vector512.ConditionalSelect(large, xBits - shift, Vector512.ConditionalSelect(small, xBits + shift, xBits));
        yBits = Vector512.ConditionalSelect(large, yBits - shift, Vector512.ConditionalSelect(small, yBits + shift, yBits));
        var factor = Vector512.ConditionalSelect(large.AsDouble(), Vector512.Create(HypotDown), Vector512.ConditionalSelect(small.AsDouble(), Vector512.Create(HypotUp), Vector512<double>.One));
        var implicitBit = Vector512.Create(1UL << 52);
        var bias = Vector512.Create(HypotSubnormalBias);
        xBits = Vector512.ConditionalSelect(small & Vector512.Equals(xExponent, Vector512<ulong>.Zero), ((xBits + implicitBit).AsDouble() - bias).AsUInt64(), xBits);
        yBits = Vector512.ConditionalSelect(small & Vector512.Equals(yExponent, Vector512<ulong>.Zero), ((yBits + implicitBit).AsDouble() - bias).AsUInt64(), yBits);
        var swap = Vector512.LessThan(xBits.AsDouble(), yBits.AsDouble()).AsUInt64();
        var hiBits = Vector512.ConditionalSelect(swap, yBits, xBits);
        var loBits = Vector512.ConditionalSelect(swap, xBits, yBits);
        var hi = hiBits.AsDouble();
        var lo = loBits.AsDouble();
        var hiHead = (hiBits & Vector512.Create(HypotHeadMask)).AsDouble();
        var loHead = (loBits & Vector512.Create(HypotHeadMask)).AsDouble();
        var hiTail = hi - hiHead;
        var loTail = lo - loHead;
        var hh = hi * hi;
        var ll = lo * lo;
        var sum = hh + ll;
        var tail = hh - sum + ll;
        tail += hiHead * hiHead - hh;
        tail += Vector512.Create(2.0) * hiHead * hiTail;
        tail += hiTail * hiTail;
        var tailSameExponent = tail + (loHead * loHead - ll);
        tailSameExponent += Vector512.Create(2.0) * loHead * loTail;
        tailSameExponent += loTail * loTail;
        tail = Vector512.ConditionalSelect(Vector512.Equals(difference, Vector512<long>.Zero).AsDouble(), tailSameExponent, tail);
        var result = Vector512.Sqrt(sum + tail) * factor;
        var near = Vector512.LessThanOrEqual(difference, Vector512.Create(54L)) & Vector512.GreaterThanOrEqual(difference, Vector512.Create(-54L));
        result = Vector512.ConditionalSelect(near.AsDouble(), result, ax + ay);
        result = Vector512.ConditionalSelect(Vector512.Equals(ay, Vector512<double>.Zero), ax, result);
        result = Vector512.ConditionalSelect(Vector512.Equals(ax, Vector512<double>.Zero), ay, result);
        var infinity = Vector512.Create(double.PositiveInfinity);
        var finite = Vector512.LessThan(ax, infinity) & Vector512.LessThan(ay, infinity);
        var infinite = Vector512.Equals(ax, infinity) | Vector512.Equals(ay, infinity);
        return Vector512.ConditionalSelect(finite, result, Vector512.ConditionalSelect(infinite, infinity, Vector512.Create(double.NaN)));
    }
}
