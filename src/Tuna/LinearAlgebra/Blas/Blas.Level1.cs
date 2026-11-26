using System.Numerics;

namespace Tuna.LinearAlgebra;

/// <summary>
/// Provides Basic Linear Algebra Subprograms (BLAS) Level 1, 2, and 3 routines operating directly on <see cref="ReadOnlySpan{T}"/> and <see cref="Span{T}"/> buffers.
/// <para><see cref="ReadOnlySpan{T}"/> および <see cref="Span{T}"/> に対して直接動作する BLAS（Level 1 / 2 / 3）基本線形代数ルーチンを提供します。</para>
/// </summary>
/// <remarks>
/// <para>Matrix operands use column-major storage with explicit leading dimensions (<c>A[i, j] = a[i + j * lda]</c>).</para>
/// <para>行列引数はリーディングディメンションを指定する列優先レイアウト（<c>A[i, j] = a[i + j * lda]</c>）で扱います。</para>
/// </remarks>
public static partial class Blas
{
    // ---- float ----

    /// <summary>
    /// Computes the dot product <c>Σ x[i]·y[i]</c> (BLAS <c>sdot</c>).
    /// <para>内積 <c>Σ x[i]·y[i]</c> を計算します（BLAS <c>sdot</c>）。</para>
    /// </summary>
    /// <param name="x">The first input vector. / 第 1 入力ベクトル。</param>
    /// <param name="y">The second input vector. / 第 2 入力ベクトル。</param>
    /// <returns>The dot product <c>Σ x[i]·y[i]</c>. / 内積 <c>Σ x[i]·y[i]</c>。</returns>
    /// <exception cref="ArgumentException"><paramref name="x"/> and <paramref name="y"/> have different lengths. / <paramref name="x"/> と <paramref name="y"/> の長さが異なる場合。</exception>
    public static float Dot(ReadOnlySpan<float> x, ReadOnlySpan<float> y)
    {
        BlasCheck.SameLength(x.Length, y.Length);
        return ParallelLevel1<float>.Dot(x, y, conjugateX: false);
    }

    /// <summary>
    /// Computes <c>y ← α·x + y</c> in place (BLAS <c>saxpy</c>).
    /// <para>スカラー倍加算 <c>y ← α·x + y</c> をインプレースで計算します（BLAS <c>saxpy</c>）。</para>
    /// </summary>
    /// <param name="alpha">The scalar multiplier α. / スカラー係数 α。</param>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="y">The input/output vector updated in place. / インプレースで更新される入出力ベクトル。</param>
    /// <exception cref="ArgumentException"><paramref name="x"/> and <paramref name="y"/> have different lengths. / <paramref name="x"/> と <paramref name="y"/> の長さが異なる場合。</exception>
    public static void Axpy(float alpha, ReadOnlySpan<float> x, Span<float> y)
    {
        BlasCheck.SameLength(x.Length, y.Length);
        if (alpha != 0)
        {
            ParallelLevel1<float>.Axpy(alpha, x, y);
        }
    }

    /// <summary>
    /// Scales a vector in place: <c>x ← α·x</c> (BLAS <c>sscal</c>).
    /// <para>ベクトルをインプレースでスカラー倍します（<c>x ← α·x</c>、BLAS <c>sscal</c>）。</para>
    /// </summary>
    /// <param name="alpha">The scalar multiplier α. / スカラー係数 α。</param>
    /// <param name="x">The vector scaled in place. / インプレースでスケーリングされるベクトル。</param>
    public static void Scale(float alpha, Span<float> x) => ParallelLevel1<float>.Scale(alpha, x);

    /// <summary>
    /// Computes the Euclidean norm <c>‖x‖₂</c> (BLAS <c>snrm2</c>).
    /// <para>ユークリッドノルム <c>‖x‖₂</c> を計算します（BLAS <c>snrm2</c>）。</para>
    /// </summary>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <returns>The Euclidean norm <c>‖x‖₂</c>. / ユークリッドノルム <c>‖x‖₂</c>。</returns>
    public static float Norm2(ReadOnlySpan<float> x) => ParallelLevel1Norms.Norm2(x);

    /// <summary>
    /// Computes the sum of absolute values <c>Σ |x[i]|</c> (BLAS <c>sasum</c>).
    /// <para>絶対値の和 <c>Σ |x[i]|</c> を計算します（BLAS <c>sasum</c>）。</para>
    /// </summary>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <returns>The L1 norm <c>Σ |x[i]|</c>. / L1 ノルム <c>Σ |x[i]|</c>。</returns>
    public static float SumAbs(ReadOnlySpan<float> x) => ParallelLevel1<float>.SumAbs(x);

    /// <summary>
    /// Finds the zero-based index of the first element with maximum absolute value (BLAS <c>isamax</c>).
    /// <para>絶対値が最大となる最初の要素の 0 から始まるインデックスを返します（BLAS <c>isamax</c>）。</para>
    /// </summary>
    /// <param name="x">The vector to search. / 検索対象のベクトル。</param>
    /// <returns>The zero-based index of the first element with maximum magnitude, the index of the first <see cref="float.NaN"/> if any, or -1 if <paramref name="x"/> is empty. / 絶対値最大の最初の要素の 0 始まりインデックス。<see cref="float.NaN"/> を含む場合は最初の NaN のインデックス、空の場合は -1。</returns>
    public static int IndexOfMaxAbs(ReadOnlySpan<float> x) => RealVectorOps.IndexOfMaxAbs(x);

    /// <summary>
    /// Swaps the elements of <paramref name="x"/> and <paramref name="y"/> in place (BLAS <c>sswap</c>).
    /// <para><paramref name="x"/> と <paramref name="y"/> の要素をインプレースで交換します（BLAS <c>sswap</c>）。</para>
    /// </summary>
    /// <param name="x">The first vector. / 第 1 ベクトル。</param>
    /// <param name="y">The second vector. / 第 2 ベクトル。</param>
    /// <exception cref="ArgumentException"><paramref name="x"/> and <paramref name="y"/> have different lengths. / <paramref name="x"/> と <paramref name="y"/> の長さが異なる場合。</exception>
    public static void Swap(Span<float> x, Span<float> y)
    {
        BlasCheck.SameLength(x.Length, y.Length);
        ParallelLevel1<float>.Swap(x, y);
    }

    /// <summary>
    /// Copies elements from <paramref name="x"/> to <paramref name="y"/>: <c>y ← x</c> (BLAS <c>scopy</c>).
    /// <para><paramref name="x"/> の要素を <paramref name="y"/> へコピーします（<c>y ← x</c>、BLAS <c>scopy</c>）。</para>
    /// </summary>
    /// <param name="x">The source vector. / コピー元ベクトル。</param>
    /// <param name="y">The destination vector. / コピー先ベクトル。</param>
    /// <exception cref="ArgumentException"><paramref name="x"/> and <paramref name="y"/> have different lengths. / <paramref name="x"/> と <paramref name="y"/> の長さが異なる場合。</exception>
    public static void Copy(ReadOnlySpan<float> x, Span<float> y)
    {
        BlasCheck.SameLength(x.Length, y.Length);
        ParallelLevel1<float>.Copy(x, y);
    }

    /// <summary>
    /// Applies a plane rotation in place: <c>(x, y) ← (c·x + s·y, c·y - s·x)</c> (BLAS <c>srot</c>).
    /// <para>平面回転 <c>(x, y) ← (c·x + s·y, c·y - s·x)</c> をインプレースで適用します（BLAS <c>srot</c>）。</para>
    /// </summary>
    /// <param name="x">The first vector, updated in place. / インプレースで更新される第 1 ベクトル。</param>
    /// <param name="y">The second vector, updated in place. / インプレースで更新される第 2 ベクトル。</param>
    /// <param name="c">The cosine rotation parameter. / 回転の余弦係数 c。</param>
    /// <param name="s">The sine rotation parameter. / 回転の正弦係数 s。</param>
    /// <exception cref="ArgumentException"><paramref name="x"/> and <paramref name="y"/> have different lengths. / <paramref name="x"/> と <paramref name="y"/> の長さが異なる場合。</exception>
    public static void Rotate(Span<float> x, Span<float> y, float c, float s)
    {
        BlasCheck.SameLength(x.Length, y.Length);
        ParallelLevel1<float>.Rotate(x, y, c, s);
    }

    // ---- double ----

    /// <summary>
    /// Computes the dot product <c>Σ x[i]·y[i]</c> (BLAS <c>ddot</c>).
    /// <para>内積 <c>Σ x[i]·y[i]</c> を計算します（BLAS <c>ddot</c>）。</para>
    /// </summary>
    /// <param name="x">The first input vector. / 第 1 入力ベクトル。</param>
    /// <param name="y">The second input vector. / 第 2 入力ベクトル。</param>
    /// <returns>The dot product <c>Σ x[i]·y[i]</c>. / 内積 <c>Σ x[i]·y[i]</c>。</returns>
    /// <exception cref="ArgumentException"><paramref name="x"/> and <paramref name="y"/> have different lengths. / <paramref name="x"/> と <paramref name="y"/> の長さが異なる場合。</exception>
    public static double Dot(ReadOnlySpan<double> x, ReadOnlySpan<double> y)
    {
        BlasCheck.SameLength(x.Length, y.Length);
        return ParallelLevel1<double>.Dot(x, y, conjugateX: false);
    }

    /// <summary>
    /// Computes <c>y ← α·x + y</c> in place (BLAS <c>daxpy</c>).
    /// <para>スカラー倍加算 <c>y ← α·x + y</c> をインプレースで計算します（BLAS <c>daxpy</c>）。</para>
    /// </summary>
    /// <param name="alpha">The scalar multiplier α. / スカラー係数 α。</param>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="y">The input/output vector updated in place. / インプレースで更新される入出力ベクトル。</param>
    /// <exception cref="ArgumentException"><paramref name="x"/> and <paramref name="y"/> have different lengths. / <paramref name="x"/> と <paramref name="y"/> の長さが異なる場合。</exception>
    public static void Axpy(double alpha, ReadOnlySpan<double> x, Span<double> y)
    {
        BlasCheck.SameLength(x.Length, y.Length);
        if (alpha != 0)
        {
            ParallelLevel1<double>.Axpy(alpha, x, y);
        }
    }

    /// <summary>
    /// Scales a vector in place: <c>x ← α·x</c> (BLAS <c>dscal</c>).
    /// <para>ベクトルをインプレースでスカラー倍します（<c>x ← α·x</c>、BLAS <c>dscal</c>）。</para>
    /// </summary>
    /// <param name="alpha">The scalar multiplier α. / スカラー係数 α。</param>
    /// <param name="x">The vector scaled in place. / インプレースでスケーリングされるベクトル。</param>
    public static void Scale(double alpha, Span<double> x) => ParallelLevel1<double>.Scale(alpha, x);

    /// <summary>
    /// Computes the Euclidean norm <c>‖x‖₂</c> (BLAS <c>dnrm2</c>).
    /// <para>ユークリッドノルム <c>‖x‖₂</c> を計算します（BLAS <c>dnrm2</c>）。</para>
    /// </summary>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <returns>The Euclidean norm <c>‖x‖₂</c>. / ユークリッドノルム <c>‖x‖₂</c>。</returns>
    public static double Norm2(ReadOnlySpan<double> x) => ParallelLevel1Norms.Norm2(x);

    /// <summary>
    /// Computes the sum of absolute values <c>Σ |x[i]|</c> (BLAS <c>dasum</c>).
    /// <para>絶対値の和 <c>Σ |x[i]|</c> を計算します（BLAS <c>dasum</c>）。</para>
    /// </summary>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <returns>The L1 norm <c>Σ |x[i]|</c>. / L1 ノルム <c>Σ |x[i]|</c>。</returns>
    public static double SumAbs(ReadOnlySpan<double> x) => ParallelLevel1<double>.SumAbs(x);

    /// <summary>
    /// Finds the zero-based index of the first element with maximum absolute value (BLAS <c>idamax</c>).
    /// <para>絶対値が最大となる最初の要素の 0 から始まるインデックスを返します（BLAS <c>idamax</c>）。</para>
    /// </summary>
    /// <param name="x">The vector to search. / 検索対象のベクトル。</param>
    /// <returns>The zero-based index of the first element with maximum magnitude, the index of the first <see cref="double.NaN"/> if any, or -1 if <paramref name="x"/> is empty. / 絶対値最大の最初の要素の 0 始まりインデックス。<see cref="double.NaN"/> を含む場合は最初の NaN のインデックス、空の場合は -1。</returns>
    public static int IndexOfMaxAbs(ReadOnlySpan<double> x) => RealVectorOps.IndexOfMaxAbs(x);

    /// <summary>
    /// Swaps the elements of <paramref name="x"/> and <paramref name="y"/> in place (BLAS <c>dswap</c>).
    /// <para><paramref name="x"/> と <paramref name="y"/> の要素をインプレースで交換します（BLAS <c>dswap</c>）。</para>
    /// </summary>
    /// <param name="x">The first vector. / 第 1 ベクトル。</param>
    /// <param name="y">The second vector. / 第 2 ベクトル。</param>
    /// <exception cref="ArgumentException"><paramref name="x"/> and <paramref name="y"/> have different lengths. / <paramref name="x"/> と <paramref name="y"/> の長さが異なる場合。</exception>
    public static void Swap(Span<double> x, Span<double> y)
    {
        BlasCheck.SameLength(x.Length, y.Length);
        ParallelLevel1<double>.Swap(x, y);
    }

    /// <summary>
    /// Copies elements from <paramref name="x"/> to <paramref name="y"/>: <c>y ← x</c> (BLAS <c>dcopy</c>).
    /// <para><paramref name="x"/> の要素を <paramref name="y"/> へコピーします（<c>y ← x</c>、BLAS <c>dcopy</c>）。</para>
    /// </summary>
    /// <param name="x">The source vector. / コピー元ベクトル。</param>
    /// <param name="y">The destination vector. / コピー先ベクトル。</param>
    /// <exception cref="ArgumentException"><paramref name="x"/> and <paramref name="y"/> have different lengths. / <paramref name="x"/> と <paramref name="y"/> の長さが異なる場合。</exception>
    public static void Copy(ReadOnlySpan<double> x, Span<double> y)
    {
        BlasCheck.SameLength(x.Length, y.Length);
        ParallelLevel1<double>.Copy(x, y);
    }

    /// <summary>
    /// Applies a plane rotation in place: <c>(x, y) ← (c·x + s·y, c·y - s·x)</c> (BLAS <c>drot</c>).
    /// <para>平面回転 <c>(x, y) ← (c·x + s·y, c·y - s·x)</c> をインプレースで適用します（BLAS <c>drot</c>）。</para>
    /// </summary>
    /// <param name="x">The first vector, updated in place. / インプレースで更新される第 1 ベクトル。</param>
    /// <param name="y">The second vector, updated in place. / インプレースで更新される第 2 ベクトル。</param>
    /// <param name="c">The cosine rotation parameter. / 回転の余弦係数 c。</param>
    /// <param name="s">The sine rotation parameter. / 回転の正弦係数 s。</param>
    /// <exception cref="ArgumentException"><paramref name="x"/> and <paramref name="y"/> have different lengths. / <paramref name="x"/> と <paramref name="y"/> の長さが異なる場合。</exception>
    public static void Rotate(Span<double> x, Span<double> y, double c, double s)
    {
        BlasCheck.SameLength(x.Length, y.Length);
        ParallelLevel1<double>.Rotate(x, y, c, s);
    }

    // ---- Complex ----

    /// <summary>
    /// Computes the unconjugated complex dot product <c>Σ x[i]·y[i]</c> (BLAS <c>zdotu</c>).
    /// <para>共役を取らない複素内積 <c>Σ x[i]·y[i]</c> を計算します（BLAS <c>zdotu</c>）。</para>
    /// </summary>
    /// <param name="x">The first input vector. / 第 1 入力ベクトル。</param>
    /// <param name="y">The second input vector. / 第 2 入力ベクトル。</param>
    /// <returns>The unconjugated dot product <c>Σ x[i]·y[i]</c>. / 共役を取らない内積 <c>Σ x[i]·y[i]</c>。</returns>
    /// <exception cref="ArgumentException"><paramref name="x"/> and <paramref name="y"/> have different lengths. / <paramref name="x"/> と <paramref name="y"/> の長さが異なる場合。</exception>
    public static Complex Dot(ReadOnlySpan<Complex> x, ReadOnlySpan<Complex> y)
    {
        BlasCheck.SameLength(x.Length, y.Length);
        return ParallelLevel1<Complex>.Dot(x, y, conjugateX: false);
    }

    /// <summary>
    /// Computes the conjugate complex dot product <c>Σ conj(x[i])·y[i]</c> (BLAS <c>zdotc</c>).
    /// <para>共役内積 <c>Σ conj(x[i])·y[i]</c> を計算します（BLAS <c>zdotc</c>）。</para>
    /// </summary>
    /// <param name="x">The first input vector (conjugated). / 第 1 入力ベクトル（共役を取ります）。</param>
    /// <param name="y">The second input vector. / 第 2 入力ベクトル。</param>
    /// <returns>The conjugate dot product <c>Σ conj(x[i])·y[i]</c>. / 共役内積 <c>Σ conj(x[i])·y[i]</c>。</returns>
    /// <exception cref="ArgumentException"><paramref name="x"/> and <paramref name="y"/> have different lengths. / <paramref name="x"/> と <paramref name="y"/> の長さが異なる場合。</exception>
    public static Complex DotConjugate(ReadOnlySpan<Complex> x, ReadOnlySpan<Complex> y)
    {
        BlasCheck.SameLength(x.Length, y.Length);
        return ParallelLevel1<Complex>.Dot(x, y, conjugateX: true);
    }

    /// <summary>
    /// Computes <c>y ← α·x + y</c> in place (BLAS <c>zaxpy</c>).
    /// <para>スカラー倍加算 <c>y ← α·x + y</c> をインプレースで計算します（BLAS <c>zaxpy</c>）。</para>
    /// </summary>
    /// <param name="alpha">The complex scalar multiplier α. / 複素スカラー係数 α。</param>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="y">The input/output vector updated in place. / インプレースで更新される入出力ベクトル。</param>
    /// <exception cref="ArgumentException"><paramref name="x"/> and <paramref name="y"/> have different lengths. / <paramref name="x"/> と <paramref name="y"/> の長さが異なる場合。</exception>
    public static void Axpy(Complex alpha, ReadOnlySpan<Complex> x, Span<Complex> y)
    {
        BlasCheck.SameLength(x.Length, y.Length);
        if (alpha != Complex.Zero)
        {
            ParallelLevel1<Complex>.Axpy(alpha, x, y);
        }
    }

    /// <summary>
    /// Scales a complex vector by a complex scalar in place: <c>x ← α·x</c> (BLAS <c>zscal</c>).
    /// <para>複素ベクトルを複素スカラーでインプレースにスカラー倍します（<c>x ← α·x</c>、BLAS <c>zscal</c>）。</para>
    /// </summary>
    /// <param name="alpha">The complex scalar multiplier α. / 複素スカラー係数 α。</param>
    /// <param name="x">The vector scaled in place. / インプレースでスケーリングされるベクトル。</param>
    public static void Scale(Complex alpha, Span<Complex> x) => ParallelLevel1<Complex>.Scale(alpha, x);

    /// <summary>
    /// Scales a complex vector by a real scalar in place: <c>x ← α·x</c> (BLAS <c>zdscal</c>).
    /// <para>複素ベクトルを実数スカラーでインプレースにスカラー倍します（<c>x ← α·x</c>、BLAS <c>zdscal</c>）。</para>
    /// </summary>
    /// <param name="alpha">The real scalar multiplier α. / 実数スカラー係数 α。</param>
    /// <param name="x">The vector scaled in place. / インプレースでスケーリングされるベクトル。</param>
    public static void Scale(double alpha, Span<Complex> x) => ParallelLevel1<double>.Scale(alpha, BlasCheck.AsDoubles(x));

    /// <summary>
    /// Computes the Euclidean norm <c>‖x‖₂</c> of a complex vector (BLAS <c>dznrm2</c>).
    /// <para>複素ベクトルのユークリッドノルム <c>‖x‖₂</c> を計算します（BLAS <c>dznrm2</c>）。</para>
    /// </summary>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <returns>The real-valued Euclidean norm <c>‖x‖₂</c>. / 実数値のユークリッドノルム <c>‖x‖₂</c>。</returns>
    public static double Norm2(ReadOnlySpan<Complex> x) => ParallelLevel1Norms.Norm2(x);

    /// <summary>
    /// Computes the sum of real and imaginary magnitudes <c>Σ (|Re x[i]| + |Im x[i]|)</c> (BLAS <c>dzasum</c>).
    /// <para>実部と虚部の絶対値の和 <c>Σ (|Re x[i]| + |Im x[i]|)</c> を計算します（BLAS <c>dzasum</c>）。</para>
    /// </summary>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <returns>The sum <c>Σ (|Re x[i]| + |Im x[i]|)</c>. / <c>Σ (|Re x[i]| + |Im x[i]|)</c> の値。</returns>
    public static double SumAbs(ReadOnlySpan<Complex> x) => ParallelLevel1<double>.SumAbs(BlasCheck.AsDoubles(x));

    /// <summary>
    /// Finds the zero-based index of the first element with maximum <c>|Re| + |Im|</c> value (BLAS <c>izamax</c>).
    /// <para><c>|Re| + |Im|</c> が最大となる最初の要素の 0 から始まるインデックスを返します（BLAS <c>izamax</c>）。</para>
    /// </summary>
    /// <param name="x">The vector to search. / 検索対象のベクトル。</param>
    /// <returns>The zero-based index of the first element with maximum <c>|Re| + |Im|</c>, the index of the first element containing NaN if any, or -1 if <paramref name="x"/> is empty. / <c>|Re| + |Im|</c> が最大の最初の要素の 0 始まりインデックス。NaN を含む要素がある場合はその最初のインデックス、空の場合は -1。</returns>
    public static int IndexOfMaxAbs(ReadOnlySpan<Complex> x) => ComplexVectorOps.IndexOfMaxAbs(x);

    /// <summary>
    /// Swaps the elements of <paramref name="x"/> and <paramref name="y"/> in place (BLAS <c>zswap</c>).
    /// <para><paramref name="x"/> と <paramref name="y"/> の要素をインプレースで交換します（BLAS <c>zswap</c>）。</para>
    /// </summary>
    /// <param name="x">The first vector. / 第 1 ベクトル。</param>
    /// <param name="y">The second vector. / 第 2 ベクトル。</param>
    /// <exception cref="ArgumentException"><paramref name="x"/> and <paramref name="y"/> have different lengths. / <paramref name="x"/> と <paramref name="y"/> の長さが異なる場合。</exception>
    public static void Swap(Span<Complex> x, Span<Complex> y)
    {
        BlasCheck.SameLength(x.Length, y.Length);
        ParallelLevel1<Complex>.Swap(x, y);
    }

    /// <summary>
    /// Copies elements from <paramref name="x"/> to <paramref name="y"/>: <c>y ← x</c> (BLAS <c>zcopy</c>).
    /// <para><paramref name="x"/> の要素を <paramref name="y"/> へコピーします（<c>y ← x</c>、BLAS <c>zcopy</c>）。</para>
    /// </summary>
    /// <param name="x">The source vector. / コピー元ベクトル。</param>
    /// <param name="y">The destination vector. / コピー先ベクトル。</param>
    /// <exception cref="ArgumentException"><paramref name="x"/> and <paramref name="y"/> have different lengths. / <paramref name="x"/> と <paramref name="y"/> の長さが異なる場合。</exception>
    public static void Copy(ReadOnlySpan<Complex> x, Span<Complex> y)
    {
        BlasCheck.SameLength(x.Length, y.Length);
        ParallelLevel1<Complex>.Copy(x, y);
    }

    /// <summary>
    /// Applies a real plane rotation to complex vectors in place: <c>(x, y) ← (c·x + s·y, c·y - s·x)</c> (BLAS <c>zdrot</c>).
    /// <para>複素ベクトルに実平面回転 <c>(x, y) ← (c·x + s·y, c·y - s·x)</c> をインプレースで適用します（BLAS <c>zdrot</c>）。</para>
    /// </summary>
    /// <param name="x">The first vector, updated in place. / インプレースで更新される第 1 ベクトル。</param>
    /// <param name="y">The second vector, updated in place. / インプレースで更新される第 2 ベクトル。</param>
    /// <param name="c">The real cosine rotation parameter. / 回転の実余弦係数 c。</param>
    /// <param name="s">The real sine rotation parameter. / 回転の実正弦係数 s。</param>
    /// <exception cref="ArgumentException"><paramref name="x"/> and <paramref name="y"/> have different lengths. / <paramref name="x"/> と <paramref name="y"/> の長さが異なる場合。</exception>
    public static void Rotate(Span<Complex> x, Span<Complex> y, double c, double s)
    {
        BlasCheck.SameLength(x.Length, y.Length);
        ParallelLevel1<double>.Rotate(BlasCheck.AsDoubles(x), BlasCheck.AsDoubles(y), c, s);
    }
}
