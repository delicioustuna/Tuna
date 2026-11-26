using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Tuna.LinearAlgebra;

// ステートレス演算子をステートフル演算子インターフェースへ適合させるアダプタ群。
internal readonly struct StaticUnaryOperator<T, TOperator> : IStatefulUnaryOperator<T> where TOperator : IUnaryOperator<T>
{
    public T Invoke(T x) => TOperator.Invoke(x);
}

internal readonly struct StaticBinaryOperator<T, TOperator> : IStatefulBinaryOperator<T> where TOperator : IBinaryOperator<T>
{
    public T Invoke(T x, T y) => TOperator.Invoke(x, y);
}

internal readonly struct StaticTernaryOperator<T, TOperator> : IStatefulTernaryOperator<T> where TOperator : ITernaryOperator<T>
{
    public T Invoke(T x, T y, T z) => TOperator.Invoke(x, y, z);
}

internal readonly struct StaticPairwiseOperator<T, TOperator> : IStatefulPairwiseOperator<T>
    where T : unmanaged, INumberBase<T>
    where TOperator : IPairwiseOperator<T>
{
    public T Invoke(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y) => TOperator.Invoke(x, y);
}

internal readonly struct IdentityOperator<T> : IStatefulUnaryOperator<T>
{
    public T Invoke(T x) => x;
}

internal readonly struct NegateOperator<T> : IStatefulUnaryOperator<T> where T : INumberBase<T>
{
    public T Invoke(T x) => unchecked(-x);
}

// e^x（float / double は DenseElementwise.Exp を呼び、全レイアウトでビット一致させる）。
internal readonly struct ExpOperator<T> : IStatefulUnaryOperator<T> where T : IExponentialFunctions<T>
{
    public T Invoke(T x) => DenseElementwise.Exp(x);
}

internal readonly struct AddScalarOperator<T>(T scalar) : IStatefulUnaryOperator<T> where T : INumberBase<T>
{
    internal T Scalar => scalar;

    public T Invoke(T x) => unchecked(x + scalar);
}

internal readonly struct SubtractScalarOperator<T>(T scalar) : IStatefulUnaryOperator<T> where T : INumberBase<T>
{
    internal T Scalar => scalar;

    public T Invoke(T x) => unchecked(x - scalar);
}

internal readonly struct ScaleOperator<T>(T scalar) : IStatefulUnaryOperator<T> where T : INumberBase<T>
{
    internal T Scalar => scalar;

    public T Invoke(T x) => unchecked(x * scalar);
}

// 軟判定しきい値 sign(x) * max(|x| - threshold, 0)（threshold >= 0 前提）。
internal readonly struct SoftThresholdOperator<T>(T threshold) : IStatefulUnaryOperator<T> where T : INumber<T>
{
    // x > threshold >= 0 なら x - threshold はオーバーフローしない。AggressiveOptimization ループの端数処理へ確実に展開させる。
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public T Invoke(T x) =>
        x > threshold ? x - threshold
        : T.IsNegative(x) && x + threshold < T.Zero ? x + threshold
        : T.IsNaN(x) ? x
        : T.Zero;
}

internal readonly struct AddOperator<T> : IStatefulBinaryOperator<T> where T : INumberBase<T>
{
    public T Invoke(T x, T y) => unchecked(x + y);
}

internal readonly struct SubtractOperator<T> : IStatefulBinaryOperator<T> where T : INumberBase<T>
{
    public T Invoke(T x, T y) => unchecked(x - y);
}

internal readonly struct MultiplyOperator<T> : IStatefulBinaryOperator<T> where T : INumberBase<T>
{
    public T Invoke(T x, T y) => unchecked(x * y);
}

internal readonly struct FirstOperator<T> : IBinaryOperator<T>
{
    public static T Invoke(T x, T parameter) => x;
}

// x^2（複素数でも共役を取らない）。
internal readonly struct SquareOperator<T> : IBinaryOperator<T> where T : INumberBase<T>
{
    public static T Invoke(T x, T parameter) => unchecked(x * x);
}

internal readonly struct DeviationOperator<T> : IBinaryOperator<T> where T : INumberBase<T>
{
    public static T Invoke(T x, T mean) => unchecked(x - mean);
}

internal readonly struct SquaredDeviationOperator<T> : IBinaryOperator<T> where T : INumberBase<T>
{
    public static T Invoke(T x, T mean)
    {
        T d = unchecked(x - mean);
        return unchecked(d * d);
    }
}

internal readonly struct ScaledOperator<T> : IBinaryOperator<T> where T : INumberBase<T>
{
    public static T Invoke(T x, T scale) => unchecked(x * scale);
}

internal readonly struct AbsoluteOperator<T> : IBinaryOperator<T> where T : INumberBase<T>
{
    public static T Invoke(T x, T parameter) => T.Abs(x);
}

// exp(x - shift)（x == shift の項は 0 とし、一致個数は別途カウントする）。
internal readonly struct ShiftedExpOperator<T> : IBinaryOperator<T> where T : IExponentialFunctions<T>
{
    public static T Invoke(T x, T shift)
    {
        T difference = x - shift;
        return difference == T.Zero ? T.Zero : DenseElementwise.Exp(difference);
    }
}

// (x * 2^k)^2（2 のべき乗スケーリングにより丸めを増やさない）。
internal readonly struct ScaledSquareOperator<T> : IBinaryOperator<T> where T : IFloatingPointIeee754<T>
{
    public static T Invoke(T x, T exponent)
    {
        T s = T.ScaleB(x, int.CreateTruncating(exponent));
        return s * s;
    }
}

// (x * s)^2（s が正規化数の 2 のべき乗なら ScaledSquareOperator とビット一致）。
internal readonly struct ScaledSquareByFactorOperator<T> : IBinaryOperator<T> where T : INumberBase<T>
{
    public static T Invoke(T x, T factor)
    {
        T s = unchecked(x * factor);
        return unchecked(s * s);
    }
}

// NaN を +0 としてスキップする総和項（部分和は +0 開始のため -0 符号に影響しない）。
internal readonly struct IgnoreNaNOperator<T> : IBinaryOperator<T> where T : INumberBase<T>
{
    public static T Invoke(T x, T parameter) => T.IsNaN(x) ? T.Zero : x;
}

internal readonly struct ScaledIgnoreNaNOperator<T> : IBinaryOperator<T> where T : INumberBase<T>
{
    public static T Invoke(T x, T scale) => T.IsNaN(x) ? T.Zero : unchecked(x * scale);
}

// decimal など ±0 の順序を区別しない型でも -0 < +0 となるよう補正する補助。
internal static class SignedZeroOrder
{
    private static bool IsOrdered<T>() =>
        typeof(T) == typeof(double) || typeof(T) == typeof(float) || typeof(T) == typeof(Half) || typeof(T) == typeof(NFloat);

    internal static T Larger<T>(T x, T y, T result) where T : INumber<T> =>
        IsOrdered<T>() || !T.IsZero(result) || x != y ? result : T.IsNegative(x) ? y : x;

    internal static T Smaller<T>(T x, T y, T result) where T : INumber<T> =>
        IsOrdered<T>() || !T.IsZero(result) || x != y ? result : T.IsNegative(x) ? x : y;
}

internal readonly struct MaximumNumberOperator<T> : IBinaryOperator<T> where T : INumber<T>
{
    public static T Invoke(T x, T y) => SignedZeroOrder.Larger(x, y, T.MaxNumber(x, y));
}

internal readonly struct MinimumNumberOperator<T> : IBinaryOperator<T> where T : INumber<T>
{
    public static T Invoke(T x, T y) => SignedZeroOrder.Smaller(x, y, T.MinNumber(x, y));
}

internal readonly struct MaximumOperator<T> : IBinaryOperator<T> where T : INumber<T>
{
    public static T Invoke(T x, T y) => SignedZeroOrder.Larger(x, y, T.Max(x, y));
}

internal readonly struct MinimumOperator<T> : IBinaryOperator<T> where T : INumber<T>
{
    public static T Invoke(T x, T y) => SignedZeroOrder.Smaller(x, y, T.Min(x, y));
}
