using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace Tuna.LinearAlgebra;

// 連続領域に対する縮約（総和・極値・検索・ソフトマックス）の SIMD 実装（float / double）。
// 総和はブロック並列転置により StridedKernels.PairwiseSum と同一の木構造で加算し、ISA によらずビット単位で一致させる。
// AVX-SSE 遷移ペナルティを避けるため、ベクトル処理とスカラー関数呼び出しループはメソッドを分離する。
internal static partial class DenseReductions
{
    private const int Block = StridedKernels.SumBlockSize;

    // 超越関数の項を求める作業領域の要素数（ブロックの倍数）
    private const int TermChunk = 16 * Block;

    // 行ごとの和で、一度に扱う行数（作業領域をキャッシュに収める。行どうしは独立なので結果は変わらない）
    private const int RowTile = 512;

    // ---------- エントリポイント ----------

    // 連続な float / double で対応カーネルがあれば PairwiseSum とビット一致する総和を計算して true を返す。
    internal static bool TrySum<T, TOperator>(ReadOnlyVectorSpan<T> x, T parameter, out T sum) where T : unmanaged, INumberBase<T>
    {
        sum = T.Zero;
        if (!IsContiguous(x))
        {
            return false;
        }

        if (typeof(T) == typeof(double))
        {
            bool done = TrySumReal<double, TOperator>(in As<T, double>(in x.Reference), x.Length, Unsafe.BitCast<T, double>(parameter), out double result);
            sum = Unsafe.BitCast<double, T>(result);
            return done;
        }

        if (typeof(T) == typeof(float))
        {
            bool done = TrySumReal<float, TOperator>(in As<T, float>(in x.Reference), x.Length, Unsafe.BitCast<T, float>(parameter), out float result);
            sum = Unsafe.BitCast<float, T>(result);
            return done;
        }

        return false;
    }

    // 連続な float / double の内積を PairwiseSum とビット一致する順序で計算する。
    internal static bool TryDot<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y, out T sum) where T : unmanaged, INumberBase<T>
    {
        sum = T.Zero;
        if (!IsContiguous(x) || !IsContiguous(y))
        {
            return false;
        }

        if (typeof(T) == typeof(double))
        {
            sum = Unsafe.BitCast<double, T>(SumProduct(in As<T, double>(in x.Reference), in As<T, double>(in y.Reference), x.Length));
            return true;
        }

        if (typeof(T) == typeof(float))
        {
            sum = Unsafe.BitCast<float, T>(SumProduct(in As<T, float>(in x.Reference), in As<T, float>(in y.Reference), x.Length));
            return true;
        }

        return false;
    }

    // Σ exp(x[i] - shift)（x[i] == shift の項は 0）を PairwiseSum とビット一致する順序で計算し、x[i] == shift の要素数を count に返す（shift は有限値）。
    internal static bool TrySumOfShiftedExp<T>(ReadOnlyVectorSpan<T> x, T shift, out T sum, out int count) where T : unmanaged, INumberBase<T>
    {
        sum = T.Zero;
        count = 0;
        if (!IsContiguous(x))
        {
            return false;
        }

        if (typeof(T) == typeof(double))
        {
            sum = Unsafe.BitCast<double, T>(SumOfShiftedExp(MemoryMarshal.CreateReadOnlySpan(in As<T, double>(in x.Reference), x.Length), Unsafe.BitCast<T, double>(shift), out count));
            return true;
        }

        if (typeof(T) == typeof(float))
        {
            sum = Unsafe.BitCast<float, T>(SumOfShiftedExp(MemoryMarshal.CreateReadOnlySpan(in As<T, float>(in x.Reference), x.Length), Unsafe.BitCast<T, float>(shift), out count));
            return true;
        }

        return false;
    }

    // 連続な float / double の非空ベクトルに対し、MaximumOperator / MinimumOperator の極値を計算する。
    internal static bool TryExtremum<T, TOperator>(ReadOnlyVectorSpan<T> x, out T result) where T : unmanaged, INumberBase<T>
    {
        result = T.Zero;
        if (!IsContiguous(x) || x.Length == 0)
        {
            return false;
        }

        if (typeof(T) == typeof(double))
        {
            bool done = TryExtremumReal<double, TOperator>(MemoryMarshal.CreateReadOnlySpan(in As<T, double>(in x.Reference), x.Length), out double value);
            result = Unsafe.BitCast<double, T>(value);
            return done;
        }

        if (typeof(T) == typeof(float))
        {
            bool done = TryExtremumReal<float, TOperator>(MemoryMarshal.CreateReadOnlySpan(in As<T, float>(in x.Reference), x.Length), out float value);
            result = Unsafe.BitCast<float, T>(value);
            return done;
        }

        return false;
    }

    // 最大値ノルム max |x[i]| を T.Max の順次適用とビット一致する結果で計算する（空なら 0、NaN を含む場合は NaN）。
    internal static bool TryNormInfinity<T>(ReadOnlyVectorSpan<T> x, out T result) where T : unmanaged, INumberBase<T>
    {
        result = T.Zero;
        if (!IsContiguous(x))
        {
            return false;
        }

        if (typeof(T) == typeof(double))
        {
            result = Unsafe.BitCast<double, T>(Reduce<double, NormInfinityKernel<double>>(MemoryMarshal.CreateReadOnlySpan(in As<T, double>(in x.Reference), x.Length), 0.0));
            return true;
        }

        if (typeof(T) == typeof(float))
        {
            result = Unsafe.BitCast<float, T>(Reduce<float, NormInfinityKernel<float>>(MemoryMarshal.CreateReadOnlySpan(in As<T, float>(in x.Reference), x.Length), 0f));
            return true;
        }

        return false;
    }

    // 絶対値の最大値を StridedKernels.LargestMagnitude と同じ規則で計算する（NaN は符号ビットのみ消去、複数 NaN 時の選択は未規定）。
    internal static bool TryLargestMagnitude<T>(ReadOnlyVectorSpan<T> x, out T result) where T : unmanaged, INumberBase<T>
    {
        result = T.Zero;
        if (!IsContiguous(x))
        {
            return false;
        }

        if (typeof(T) == typeof(double))
        {
            result = Unsafe.BitCast<double, T>(Reduce<double, LargestMagnitudeKernel<double>>(MemoryMarshal.CreateReadOnlySpan(in As<T, double>(in x.Reference), x.Length), 0.0));
            return true;
        }

        if (typeof(T) == typeof(float))
        {
            result = Unsafe.BitCast<float, T>(Reduce<float, LargestMagnitudeKernel<float>>(MemoryMarshal.CreateReadOnlySpan(in As<T, float>(in x.Reference), x.Length), 0f));
            return true;
        }

        return false;
    }

    // 各列が連続な float / double 行列について、各行の総和を RowPairwiseSums とビット一致する順序で計算する（FirstOperator<T> では parameters は空可）。
    internal static bool TryRowSums<T, TOperator>(ReadOnlyMatrixSpan<T> a, ReadOnlySpan<T> parameters, Span<T> destination) where T : unmanaged, INumberBase<T>
    {
        if (!IsColumnContiguous(a))
        {
            return false;
        }

        if (typeof(T) == typeof(double))
        {
            return TryRowSumsReal<double, TOperator>(a.Rows, a.Columns, in As<T, double>(in a.Reference), a.ColumnStride,
                MemoryMarshal.Cast<T, double>(parameters), MemoryMarshal.Cast<T, double>(destination));
        }

        if (typeof(T) == typeof(float))
        {
            return TryRowSumsReal<float, TOperator>(a.Rows, a.Columns, in As<T, float>(in a.Reference), a.ColumnStride,
                MemoryMarshal.Cast<T, float>(parameters), MemoryMarshal.Cast<T, float>(destination));
        }

        return false;
    }

    // 各行の Σ_j exp(a[i, j] - shifts[i])（a[i, j] == shifts[i] の項は 0）を RowPairwiseSums とビット一致する順序で計算し、
    // a[i, j] == shifts[i] の要素数を counts に加算する（shifts が非有限の行の counts は参照しないこと）。
    internal static bool TryRowSumsOfShiftedExp<T>(ReadOnlyMatrixSpan<T> a, ReadOnlySpan<T> shifts, Span<T> destination, Span<int> counts) where T : unmanaged, INumberBase<T>
    {
        if (!IsColumnContiguous(a))
        {
            return false;
        }

        if (typeof(T) == typeof(double))
        {
            RowSums<double, ElementKernel<double>>(a.Rows, a.Columns, in As<T, double>(in a.Reference), a.ColumnStride,
                MemoryMarshal.Cast<T, double>(shifts), MemoryMarshal.Cast<T, double>(destination), exponentiate: true, counts);
            return true;
        }

        if (typeof(T) == typeof(float))
        {
            RowSums<float, ElementKernel<float>>(a.Rows, a.Columns, in As<T, float>(in a.Reference), a.ColumnStride,
                MemoryMarshal.Cast<T, float>(shifts), MemoryMarshal.Cast<T, float>(destination), exponentiate: true, counts);
            return true;
        }

        return false;
    }

    // 各列が連続な float / double 行列の各行の最大値または最小値を求める（列数 >= 1 前提）。
    internal static bool TryRowExtrema<T, TOperator>(ReadOnlyMatrixSpan<T> a, Span<T> destination) where T : unmanaged, INumberBase<T>
    {
        if (!IsColumnContiguous(a) || a.Columns == 0)
        {
            return false;
        }

        if (typeof(T) == typeof(double))
        {
            return TryRowExtremaReal<double, TOperator>(a.Rows, a.Columns, in As<T, double>(in a.Reference), a.ColumnStride, MemoryMarshal.Cast<T, double>(destination));
        }

        if (typeof(T) == typeof(float))
        {
            return TryRowExtremaReal<float, TOperator>(a.Rows, a.Columns, in As<T, float>(in a.Reference), a.ColumnStride, MemoryMarshal.Cast<T, float>(destination));
        }

        return false;
    }

    // 連続な float / double ベクトルから value と一致する最初のインデックスを検索する（NaN は最初の NaN、0 は符号一致、未検出時は -1）。
    internal static bool TryIndexOf<T>(ReadOnlyVectorSpan<T> x, T value, out int index) where T : unmanaged, INumberBase<T>
    {
        index = -1;
        if (!IsContiguous(x))
        {
            return false;
        }

        if (typeof(T) == typeof(double))
        {
            var span = MemoryMarshal.CreateReadOnlySpan(in As<T, double>(in x.Reference), x.Length);
            index = T.IsNaN(value) ? IndexOfNaN(span) : MemoryMarshal.Cast<double, long>(span).IndexOf(Unsafe.BitCast<T, long>(value));
            return true;
        }

        if (typeof(T) == typeof(float))
        {
            var span = MemoryMarshal.CreateReadOnlySpan(in As<T, float>(in x.Reference), x.Length);
            index = T.IsNaN(value) ? IndexOfNaN(span) : MemoryMarshal.Cast<float, int>(span).IndexOf(Unsafe.BitCast<T, int>(value));
            return true;
        }

        return false;
    }

    // 連続な float / double ベクトル内の NaN 要素数を求める。
    internal static bool TryCountNaN<T>(ReadOnlyVectorSpan<T> x, out int count) where T : unmanaged, INumberBase<T>
    {
        count = 0;
        if (!IsContiguous(x))
        {
            return false;
        }

        if (typeof(T) == typeof(double))
        {
            count = CountNaN(MemoryMarshal.CreateReadOnlySpan(in As<T, double>(in x.Reference), x.Length));
            return true;
        }

        if (typeof(T) == typeof(float))
        {
            count = CountNaN(MemoryMarshal.CreateReadOnlySpan(in As<T, float>(in x.Reference), x.Length));
            return true;
        }

        return false;
    }

    // 連続な float / double ベクトルのソフトマックスを ColumnSoftmax の 1 列分とビット一致する順序で計算する（インプレース可）。
    internal static bool TrySoftmax<T>(ReadOnlyVectorSpan<T> x, VectorSpan<T> destination) where T : unmanaged, INumberBase<T>
    {
        if (!IsContiguous(x) || !IsContiguous(destination.AsReadOnly()) || x.Length == 0)
        {
            return false;
        }

        if (typeof(T) == typeof(double))
        {
            Softmax(MemoryMarshal.CreateReadOnlySpan(in As<T, double>(in x.Reference), x.Length),
                MemoryMarshal.CreateSpan(ref Unsafe.As<T, double>(ref Unsafe.AsRef(in destination.AsReadOnly().Reference)), x.Length));
            return true;
        }

        if (typeof(T) == typeof(float))
        {
            Softmax(MemoryMarshal.CreateReadOnlySpan(in As<T, float>(in x.Reference), x.Length),
                MemoryMarshal.CreateSpan(ref Unsafe.As<T, float>(ref Unsafe.AsRef(in destination.AsReadOnly().Reference)), x.Length));
            return true;
        }

        return false;
    }

    // destination[i] = (x[i] - first) - second を各減算で丸めつつ 1 パスで計算する（インプレース可）。
    internal static bool TrySubtract<T>(ReadOnlyVectorSpan<T> x, T first, T second, VectorSpan<T> destination) where T : unmanaged, INumberBase<T>
    {
        if (!IsContiguous(x) || !IsContiguous(destination.AsReadOnly()))
        {
            return false;
        }

        if (typeof(T) == typeof(double))
        {
            SubtractTwiceInto(MemoryMarshal.CreateReadOnlySpan(in As<T, double>(in x.Reference), x.Length), Unsafe.BitCast<T, double>(first), Unsafe.BitCast<T, double>(second),
                MemoryMarshal.CreateSpan(ref Unsafe.As<T, double>(ref Unsafe.AsRef(in destination.AsReadOnly().Reference)), x.Length));
            return true;
        }

        if (typeof(T) == typeof(float))
        {
            SubtractTwiceInto(MemoryMarshal.CreateReadOnlySpan(in As<T, float>(in x.Reference), x.Length), Unsafe.BitCast<T, float>(first), Unsafe.BitCast<T, float>(second),
                MemoryMarshal.CreateSpan(ref Unsafe.As<T, float>(ref Unsafe.AsRef(in destination.AsReadOnly().Reference)), x.Length));
            return true;
        }

        return false;
    }

    // 各行から shifts[i] を減算する（exponentiate が true ならその結果の指数をとる、インプレース可）。
    internal static bool TrySubtractRows<T>(ReadOnlyMatrixSpan<T> x, ReadOnlySpan<T> shifts, MatrixSpan<T> destination, bool exponentiate) where T : unmanaged, INumberBase<T> =>
        TrySubtractRows(x, shifts, ReadOnlySpan<T>.Empty, destination, exponentiate);

    // destination[i, j] = (x[i, j] - first[i]) - second[i] を各減算で丸めつつ 1 パスで計算する（インプレース可）。
    internal static bool TrySubtractRows<T>(ReadOnlyMatrixSpan<T> x, ReadOnlySpan<T> first, ReadOnlySpan<T> second, MatrixSpan<T> destination) where T : unmanaged, INumberBase<T> =>
        TrySubtractRows(x, first, second, destination, exponentiate: false);

    // second が空なら x - first（exponentiate なら、その指数）、空でなければ (x - first) - second
    private static bool TrySubtractRows<T>(ReadOnlyMatrixSpan<T> x, ReadOnlySpan<T> first, ReadOnlySpan<T> second, MatrixSpan<T> destination, bool exponentiate)
        where T : unmanaged, INumberBase<T>
    {
        var output = destination.AsReadOnly();
        if (!IsColumnContiguous(x) || !IsColumnContiguous(output))
        {
            return false;
        }

        if (typeof(T) == typeof(double))
        {
            SubtractRows(x.Rows, x.Columns, in As<T, double>(in x.Reference), x.ColumnStride, MemoryMarshal.Cast<T, double>(first), MemoryMarshal.Cast<T, double>(second),
                ref Unsafe.As<T, double>(ref Unsafe.AsRef(in output.Reference)), output.ColumnStride, exponentiate);
            return true;
        }

        if (typeof(T) == typeof(float))
        {
            SubtractRows(x.Rows, x.Columns, in As<T, float>(in x.Reference), x.ColumnStride, MemoryMarshal.Cast<T, float>(first), MemoryMarshal.Cast<T, float>(second),
                ref Unsafe.As<T, float>(ref Unsafe.AsRef(in output.Reference)), output.ColumnStride, exponentiate);
            return true;
        }

        return false;
    }

    // destination[i, j] /= divisors[i] を列連続の float / double 行列に対して計算する。
    internal static bool TryDivideRows<T>(MatrixSpan<T> destination, ReadOnlySpan<T> divisors) where T : unmanaged, INumberBase<T>
    {
        var output = destination.AsReadOnly();
        if (!IsColumnContiguous(output))
        {
            return false;
        }

        if (typeof(T) == typeof(double))
        {
            DivideRows(output.Rows, output.Columns, ref Unsafe.As<T, double>(ref Unsafe.AsRef(in output.Reference)), output.ColumnStride, MemoryMarshal.Cast<T, double>(divisors));
            return true;
        }

        if (typeof(T) == typeof(float))
        {
            DivideRows(output.Rows, output.Columns, ref Unsafe.As<T, float>(ref Unsafe.AsRef(in output.Reference)), output.ColumnStride, MemoryMarshal.Cast<T, float>(divisors));
            return true;
        }

        return false;
    }

    // 長さ 1 以下のビューはストライドが 0 に正規化されている
    private static bool IsContiguous<T>(ReadOnlyVectorSpan<T> x) where T : unmanaged, INumberBase<T> => x.Length <= 1 || x.Stride == 1;

    private static bool IsColumnContiguous<T>(ReadOnlyMatrixSpan<T> a) where T : unmanaged, INumberBase<T> => a.Rows <= 1 || a.RowStride == 1;

    private static ref readonly TTo As<TFrom, TTo>(ref readonly TFrom source) => ref Unsafe.As<TFrom, TTo>(ref Unsafe.AsRef(in source));
}
