using System.Numerics;

namespace Tuna.LinearAlgebra;

public static partial class Blas
{
    // ---- Level 2 (float) ----

    /// <summary>
    /// Computes the matrix-vector product <c>y ← α·op(A)·x + β·y</c> (BLAS <c>sgemv</c>).
    /// <para>行列ベクトル積 <c>y ← α·op(A)·x + β·y</c> を計算します（BLAS <c>sgemv</c>）。</para>
    /// </summary>
    /// <param name="trans">The transpose operation applied to <c>A</c>. / <c>A</c> に適用する転置操作。</param>
    /// <param name="m">The number of rows of <c>A</c>. / <c>A</c> の行数。</param>
    /// <param name="n">The number of columns of <c>A</c>. / <c>A</c> の列数。</param>
    /// <param name="alpha">The scalar multiplier α. / スカラー係数 α。</param>
    /// <param name="a">The column-major <c>m × n</c> matrix <c>A</c>. / 列優先の <c>m × n</c> 行列 <c>A</c>。</param>
    /// <param name="lda">The leading dimension of <c>A</c> (at least <paramref name="m"/>). / <c>A</c> のリーディングディメンション（<paramref name="m"/> 以上）。</param>
    /// <param name="x">The input vector of length at least <paramref name="n"/> (or <paramref name="m"/> when transposed). / 長さ <paramref name="n"/> 以上（転置時は <paramref name="m"/> 以上）の入力ベクトル。</param>
    /// <param name="beta">The scalar multiplier β applied to <paramref name="y"/>; if 0, the initial contents of <paramref name="y"/> are not read. / <paramref name="y"/> に掛ける係数 β（0 の場合は <paramref name="y"/> の元の値を読み取りません）。</param>
    /// <param name="y">The input/output vector of length at least <paramref name="m"/> (or <paramref name="n"/> when transposed), updated in place. / 長さ <paramref name="m"/> 以上（転置時は <paramref name="n"/> 以上）の入出力ベクトル（インプレースで更新）。</param>
    /// <exception cref="ArgumentException">Dimensions, leading dimension, or span lengths are invalid. / サイズ・リーディングディメンション・スパン長が不正な場合。</exception>
    public static void Gemv(BlasTranspose trans, int m, int n, float alpha, ReadOnlySpan<float> a, int lda, ReadOnlySpan<float> x, float beta, Span<float> y)
    {
        BlasCheck.Matrix(m, n, a.Length, lda);
        bool notrans = trans == BlasTranspose.None;
        BlasCheck.Vector(x.Length, notrans ? n : m, nameof(x));
        BlasCheck.Vector(y.Length, notrans ? m : n, nameof(y));
        ParallelLevel2<float>.Gemv(trans, m, n, alpha, a, lda, x, beta, y);
    }

    /// <summary>
    /// Performs the rank-1 update <c>A ← α·x·yᵀ + A</c> in place (BLAS <c>sger</c>).
    /// <para>ランク 1 更新 <c>A ← α·x·yᵀ + A</c> をインプレースで実行します（BLAS <c>sger</c>）。</para>
    /// </summary>
    /// <param name="m">The number of rows of <c>A</c>. / <c>A</c> の行数。</param>
    /// <param name="n">The number of columns of <c>A</c>. / <c>A</c> の列数。</param>
    /// <param name="alpha">The scalar multiplier α. / スカラー係数 α。</param>
    /// <param name="x">The input vector of length at least <paramref name="m"/>. / 長さ <paramref name="m"/> 以上の入力ベクトル。</param>
    /// <param name="y">The input vector of length at least <paramref name="n"/>. / 長さ <paramref name="n"/> 以上の入力ベクトル。</param>
    /// <param name="a">The column-major <c>m × n</c> matrix <c>A</c>, updated in place. / インプレースで更新される列優先の <c>m × n</c> 行列 <c>A</c>。</param>
    /// <param name="lda">The leading dimension of <c>A</c> (at least <paramref name="m"/>). / <c>A</c> のリーディングディメンション（<paramref name="m"/> 以上）。</param>
    /// <exception cref="ArgumentException">Dimensions, leading dimension, or span lengths are invalid. / サイズ・リーディングディメンション・スパン長が不正な場合。</exception>
    public static void Ger(int m, int n, float alpha, ReadOnlySpan<float> x, ReadOnlySpan<float> y, Span<float> a, int lda)
    {
        BlasCheck.Matrix(m, n, a.Length, lda);
        BlasCheck.Vector(x.Length, m, nameof(x));
        BlasCheck.Vector(y.Length, n, nameof(y));
        ParallelLevel2<float>.Ger(m, n, alpha, x, y, a, lda, conjugateY: false);
    }

    /// <summary>
    /// Computes the symmetric matrix-vector product <c>y ← α·A·x + β·y</c> (BLAS <c>ssymv</c>).
    /// <para>対称行列ベクトル積 <c>y ← α·A·x + β·y</c> を計算します（BLAS <c>ssymv</c>）。</para>
    /// </summary>
    /// <param name="uplo">The triangular part of <c>A</c> to reference. / <c>A</c> の参照する三角部分。</param>
    /// <param name="n">The order of the symmetric matrix <c>A</c>. / 対称行列 <c>A</c> の次数。</param>
    /// <param name="alpha">The scalar multiplier α. / スカラー係数 α。</param>
    /// <param name="a">The column-major <c>n × n</c> symmetric matrix <c>A</c> (only the <paramref name="uplo"/> triangle is read). / 列優先の <c>n × n</c> 対称行列 <c>A</c>（<paramref name="uplo"/> 側のみ参照）。</param>
    /// <param name="lda">The leading dimension of <c>A</c> (at least <paramref name="n"/>). / <c>A</c> のリーディングディメンション（<paramref name="n"/> 以上）。</param>
    /// <param name="x">The input vector of length at least <paramref name="n"/>. / 長さ <paramref name="n"/> 以上の入力ベクトル。</param>
    /// <param name="beta">The scalar multiplier β applied to <paramref name="y"/>. / <paramref name="y"/> に掛ける係数 β。</param>
    /// <param name="y">The input/output vector of length at least <paramref name="n"/>, updated in place. / 長さ <paramref name="n"/> 以上の入出力ベクトル（インプレースで更新）。</param>
    /// <exception cref="ArgumentException">Dimensions, leading dimension, or span lengths are invalid. / サイズ・リーディングディメンション・スパン長が不正な場合。</exception>
    public static void Symv(BlasTriangle uplo, int n, float alpha, ReadOnlySpan<float> a, int lda, ReadOnlySpan<float> x, float beta, Span<float> y)
    {
        BlasCheck.Matrix(n, n, a.Length, lda);
        BlasCheck.Vector(x.Length, n, nameof(x));
        BlasCheck.Vector(y.Length, n, nameof(y));
        ParallelLevel2<float>.Symv(uplo, n, alpha, a, lda, x, beta, y, hermitian: false);
    }

    /// <summary>
    /// Performs the symmetric rank-1 update <c>A ← α·x·xᵀ + A</c> in place (BLAS <c>ssyr</c>).
    /// <para>対称ランク 1 更新 <c>A ← α·x·xᵀ + A</c> をインプレースで実行します（BLAS <c>ssyr</c>）。</para>
    /// </summary>
    /// <param name="uplo">The triangular part of <c>A</c> to update. / 更新する <c>A</c> の三角部分。</param>
    /// <param name="n">The order of the symmetric matrix <c>A</c>. / 対称行列 <c>A</c> の次数。</param>
    /// <param name="alpha">The scalar multiplier α. / スカラー係数 α。</param>
    /// <param name="x">The input vector of length at least <paramref name="n"/>. / 長さ <paramref name="n"/> 以上の入力ベクトル。</param>
    /// <param name="a">The column-major <c>n × n</c> symmetric matrix <c>A</c> (only the <paramref name="uplo"/> triangle is updated). / 列優先の <c>n × n</c> 対称行列 <c>A</c>（<paramref name="uplo"/> 側のみ更新）。</param>
    /// <param name="lda">The leading dimension of <c>A</c> (at least <paramref name="n"/>). / <c>A</c> のリーディングディメンション（<paramref name="n"/> 以上）。</param>
    /// <exception cref="ArgumentException">Dimensions, leading dimension, or span lengths are invalid. / サイズ・リーディングディメンション・スパン長が不正な場合。</exception>
    public static void Syr(BlasTriangle uplo, int n, float alpha, ReadOnlySpan<float> x, Span<float> a, int lda)
    {
        BlasCheck.Matrix(n, n, a.Length, lda);
        BlasCheck.Vector(x.Length, n, nameof(x));
        ParallelLevel2<float>.Syr(uplo, n, alpha, x, a, lda, hermitian: false);
    }

    /// <summary>
    /// Performs the symmetric rank-2 update <c>A ← α·x·yᵀ + α·y·xᵀ + A</c> in place (BLAS <c>ssyr2</c>).
    /// <para>対称ランク 2 更新 <c>A ← α·x·yᵀ + α·y·xᵀ + A</c> をインプレースで実行します（BLAS <c>ssyr2</c>）。</para>
    /// </summary>
    /// <param name="uplo">The triangular part of <c>A</c> to update. / 更新する <c>A</c> の三角部分。</param>
    /// <param name="n">The order of the symmetric matrix <c>A</c>. / 対称行列 <c>A</c> の次数。</param>
    /// <param name="alpha">The scalar multiplier α. / スカラー係数 α。</param>
    /// <param name="x">The first input vector of length at least <paramref name="n"/>. / 長さ <paramref name="n"/> 以上の第 1 入力ベクトル。</param>
    /// <param name="y">The second input vector of length at least <paramref name="n"/>. / 長さ <paramref name="n"/> 以上の第 2 入力ベクトル。</param>
    /// <param name="a">The column-major <c>n × n</c> symmetric matrix <c>A</c> (only the <paramref name="uplo"/> triangle is updated). / 列優先の <c>n × n</c> 対称行列 <c>A</c>（<paramref name="uplo"/> 側のみ更新）。</param>
    /// <param name="lda">The leading dimension of <c>A</c> (at least <paramref name="n"/>). / <c>A</c> のリーディングディメンション（<paramref name="n"/> 以上）。</param>
    /// <exception cref="ArgumentException">Dimensions, leading dimension, or span lengths are invalid. / サイズ・リーディングディメンション・スパン長が不正な場合。</exception>
    public static void Syr2(BlasTriangle uplo, int n, float alpha, ReadOnlySpan<float> x, ReadOnlySpan<float> y, Span<float> a, int lda)
    {
        BlasCheck.Matrix(n, n, a.Length, lda);
        BlasCheck.Vector(x.Length, n, nameof(x));
        BlasCheck.Vector(y.Length, n, nameof(y));
        ParallelLevel2<float>.Syr2(uplo, n, alpha, x, y, a, lda, hermitian: false);
    }

    /// <summary>
    /// Computes the triangular matrix-vector product <c>x ← op(A)·x</c> in place (BLAS <c>strmv</c>).
    /// <para>三角行列ベクトル積 <c>x ← op(A)·x</c> をインプレースで計算します（BLAS <c>strmv</c>）。</para>
    /// </summary>
    /// <param name="uplo">Specifies whether <c>A</c> is upper or lower triangular. / <c>A</c> が上三角か下三角かを指定します。</param>
    /// <param name="trans">The transpose operation applied to <c>A</c>. / <c>A</c> に適用する転置操作。</param>
    /// <param name="diag">Specifies whether <c>A</c> has unit diagonal elements. / <c>A</c> の対角要素を 1 とみなすかどうか。</param>
    /// <param name="n">The order of the triangular matrix <c>A</c>. / 三角行列 <c>A</c> の次数。</param>
    /// <param name="a">The column-major <c>n × n</c> triangular matrix <c>A</c> (only the <paramref name="uplo"/> triangle is read). / 列優先の <c>n × n</c> 三角行列 <c>A</c>（<paramref name="uplo"/> 側のみ参照）。</param>
    /// <param name="lda">The leading dimension of <c>A</c> (at least <paramref name="n"/>). / <c>A</c> のリーディングディメンション（<paramref name="n"/> 以上）。</param>
    /// <param name="x">The input/output vector of length at least <paramref name="n"/>, overwritten with <c>op(A)·x</c>. / 長さ <paramref name="n"/> 以上の入出力ベクトル（<c>op(A)·x</c> で上書き）。</param>
    /// <exception cref="ArgumentException">Dimensions, leading dimension, or span lengths are invalid. / サイズ・リーディングディメンション・スパン長が不正な場合。</exception>
    public static void Trmv(BlasTriangle uplo, BlasTranspose trans, BlasDiagonal diag, int n, ReadOnlySpan<float> a, int lda, Span<float> x)
    {
        BlasCheck.Matrix(n, n, a.Length, lda);
        BlasCheck.Vector(x.Length, n, nameof(x));
        ParallelLevel2<float>.Trmv(uplo, trans, diag, n, a, lda, x);
    }

    /// <summary>
    /// Solves the triangular linear system <c>op(A)·x = b</c> in place (BLAS <c>strsv</c>).
    /// <para>三角連立一次方程式 <c>op(A)·x = b</c> をインプレースで解きます（BLAS <c>strsv</c>）。</para>
    /// </summary>
    /// <param name="uplo">Specifies whether <c>A</c> is upper or lower triangular. / <c>A</c> が上三角か下三角かを指定します。</param>
    /// <param name="trans">The transpose operation applied to <c>A</c>. / <c>A</c> に適用する転置操作。</param>
    /// <param name="diag">Specifies whether <c>A</c> has unit diagonal elements. / <c>A</c> の対角要素を 1 とみなすかどうか。</param>
    /// <param name="n">The order of the triangular matrix <c>A</c>. / 三角行列 <c>A</c> の次数。</param>
    /// <param name="a">The column-major <c>n × n</c> triangular matrix <c>A</c> (only the <paramref name="uplo"/> triangle is read). / 列優先の <c>n × n</c> 三角行列 <c>A</c>（<paramref name="uplo"/> 側のみ参照）。</param>
    /// <param name="lda">The leading dimension of <c>A</c> (at least <paramref name="n"/>). / <c>A</c> のリーディングディメンション（<paramref name="n"/> 以上）。</param>
    /// <param name="x">On entry, the right-hand side <c>b</c> of length at least <paramref name="n"/>; on exit, overwritten with the solution <c>x</c>. / 入力時は長さ <paramref name="n"/> 以上の右辺ベクトル <c>b</c>、出力時は解 <c>x</c> で上書きされます。</param>
    /// <remarks>
    /// <para>Singularity is not checked; zero diagonal entries produce infinities or NaN.</para>
    /// <para>特異性の検査は行わないため、対角要素が 0 の場合は無限大または NaN が生じます。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">Dimensions, leading dimension, or span lengths are invalid. / サイズ・リーディングディメンション・スパン長が不正な場合。</exception>
    public static void Trsv(BlasTriangle uplo, BlasTranspose trans, BlasDiagonal diag, int n, ReadOnlySpan<float> a, int lda, Span<float> x)
    {
        BlasCheck.Matrix(n, n, a.Length, lda);
        BlasCheck.Vector(x.Length, n, nameof(x));
        ParallelLevel2<float>.Trsv(uplo, trans, diag, n, a, lda, x);
    }

    // ---- Level 2 (double) ----

    /// <summary>
    /// Computes the matrix-vector product <c>y ← α·op(A)·x + β·y</c> (BLAS <c>dgemv</c>).
    /// <para>行列ベクトル積 <c>y ← α·op(A)·x + β·y</c> を計算します（BLAS <c>dgemv</c>）。</para>
    /// </summary>
    /// <param name="trans">The transpose operation applied to <c>A</c>. / <c>A</c> に適用する転置操作。</param>
    /// <param name="m">The number of rows of <c>A</c>. / <c>A</c> の行数。</param>
    /// <param name="n">The number of columns of <c>A</c>. / <c>A</c> の列数。</param>
    /// <param name="alpha">The scalar multiplier α. / スカラー係数 α。</param>
    /// <param name="a">The column-major <c>m × n</c> matrix <c>A</c>. / 列優先の <c>m × n</c> 行列 <c>A</c>。</param>
    /// <param name="lda">The leading dimension of <c>A</c> (at least <paramref name="m"/>). / <c>A</c> のリーディングディメンション（<paramref name="m"/> 以上）。</param>
    /// <param name="x">The input vector of length at least <paramref name="n"/> (or <paramref name="m"/> when transposed). / 長さ <paramref name="n"/> 以上（転置時は <paramref name="m"/> 以上）の入力ベクトル。</param>
    /// <param name="beta">The scalar multiplier β applied to <paramref name="y"/>; if 0, the initial contents of <paramref name="y"/> are not read. / <paramref name="y"/> に掛ける係数 β（0 の場合は <paramref name="y"/> の元の値を読み取りません）。</param>
    /// <param name="y">The input/output vector of length at least <paramref name="m"/> (or <paramref name="n"/> when transposed), updated in place. / 長さ <paramref name="m"/> 以上（転置時は <paramref name="n"/> 以上）の入出力ベクトル（インプレースで更新）。</param>
    /// <exception cref="ArgumentException">Dimensions, leading dimension, or span lengths are invalid. / サイズ・リーディングディメンション・スパン長が不正な場合。</exception>
    public static void Gemv(BlasTranspose trans, int m, int n, double alpha, ReadOnlySpan<double> a, int lda, ReadOnlySpan<double> x, double beta, Span<double> y)
    {
        BlasCheck.Matrix(m, n, a.Length, lda);
        bool notrans = trans == BlasTranspose.None;
        BlasCheck.Vector(x.Length, notrans ? n : m, nameof(x));
        BlasCheck.Vector(y.Length, notrans ? m : n, nameof(y));
        ParallelLevel2<double>.Gemv(trans, m, n, alpha, a, lda, x, beta, y);
    }

    /// <summary>
    /// Performs the rank-1 update <c>A ← α·x·yᵀ + A</c> in place (BLAS <c>dger</c>).
    /// <para>ランク 1 更新 <c>A ← α·x·yᵀ + A</c> をインプレースで実行します（BLAS <c>dger</c>）。</para>
    /// </summary>
    /// <param name="m">The number of rows of <c>A</c>. / <c>A</c> の行数。</param>
    /// <param name="n">The number of columns of <c>A</c>. / <c>A</c> の列数。</param>
    /// <param name="alpha">The scalar multiplier α. / スカラー係数 α。</param>
    /// <param name="x">The input vector of length at least <paramref name="m"/>. / 長さ <paramref name="m"/> 以上の入力ベクトル。</param>
    /// <param name="y">The input vector of length at least <paramref name="n"/>. / 長さ <paramref name="n"/> 以上の入力ベクトル。</param>
    /// <param name="a">The column-major <c>m × n</c> matrix <c>A</c>, updated in place. / インプレースで更新される列優先の <c>m × n</c> 行列 <c>A</c>。</param>
    /// <param name="lda">The leading dimension of <c>A</c> (at least <paramref name="m"/>). / <c>A</c> のリーディングディメンション（<paramref name="m"/> 以上）。</param>
    /// <exception cref="ArgumentException">Dimensions, leading dimension, or span lengths are invalid. / サイズ・リーディングディメンション・スパン長が不正な場合。</exception>
    public static void Ger(int m, int n, double alpha, ReadOnlySpan<double> x, ReadOnlySpan<double> y, Span<double> a, int lda)
    {
        BlasCheck.Matrix(m, n, a.Length, lda);
        BlasCheck.Vector(x.Length, m, nameof(x));
        BlasCheck.Vector(y.Length, n, nameof(y));
        ParallelLevel2<double>.Ger(m, n, alpha, x, y, a, lda, conjugateY: false);
    }

    /// <summary>
    /// Computes the symmetric matrix-vector product <c>y ← α·A·x + β·y</c> (BLAS <c>dsymv</c>).
    /// <para>対称行列ベクトル積 <c>y ← α·A·x + β·y</c> を計算します（BLAS <c>dsymv</c>）。</para>
    /// </summary>
    /// <param name="uplo">The triangular part of <c>A</c> to reference. / <c>A</c> の参照する三角部分。</param>
    /// <param name="n">The order of the symmetric matrix <c>A</c>. / 対称行列 <c>A</c> の次数。</param>
    /// <param name="alpha">The scalar multiplier α. / スカラー係数 α。</param>
    /// <param name="a">The column-major <c>n × n</c> symmetric matrix <c>A</c> (only the <paramref name="uplo"/> triangle is read). / 列優先の <c>n × n</c> 対称行列 <c>A</c>（<paramref name="uplo"/> 側のみ参照）。</param>
    /// <param name="lda">The leading dimension of <c>A</c> (at least <paramref name="n"/>). / <c>A</c> のリーディングディメンション（<paramref name="n"/> 以上）。</param>
    /// <param name="x">The input vector of length at least <paramref name="n"/>. / 長さ <paramref name="n"/> 以上の入力ベクトル。</param>
    /// <param name="beta">The scalar multiplier β applied to <paramref name="y"/>. / <paramref name="y"/> に掛ける係数 β。</param>
    /// <param name="y">The input/output vector of length at least <paramref name="n"/>, updated in place. / 長さ <paramref name="n"/> 以上の入出力ベクトル（インプレースで更新）。</param>
    /// <exception cref="ArgumentException">Dimensions, leading dimension, or span lengths are invalid. / サイズ・リーディングディメンション・スパン長が不正な場合。</exception>
    public static void Symv(BlasTriangle uplo, int n, double alpha, ReadOnlySpan<double> a, int lda, ReadOnlySpan<double> x, double beta, Span<double> y)
    {
        BlasCheck.Matrix(n, n, a.Length, lda);
        BlasCheck.Vector(x.Length, n, nameof(x));
        BlasCheck.Vector(y.Length, n, nameof(y));
        ParallelLevel2<double>.Symv(uplo, n, alpha, a, lda, x, beta, y, hermitian: false);
    }

    /// <summary>
    /// Performs the symmetric rank-1 update <c>A ← α·x·xᵀ + A</c> in place (BLAS <c>dsyr</c>).
    /// <para>対称ランク 1 更新 <c>A ← α·x·xᵀ + A</c> をインプレースで実行します（BLAS <c>dsyr</c>）。</para>
    /// </summary>
    /// <param name="uplo">The triangular part of <c>A</c> to update. / 更新する <c>A</c> の三角部分。</param>
    /// <param name="n">The order of the symmetric matrix <c>A</c>. / 対称行列 <c>A</c> の次数。</param>
    /// <param name="alpha">The scalar multiplier α. / スカラー係数 α。</param>
    /// <param name="x">The input vector of length at least <paramref name="n"/>. / 長さ <paramref name="n"/> 以上の入力ベクトル。</param>
    /// <param name="a">The column-major <c>n × n</c> symmetric matrix <c>A</c> (only the <paramref name="uplo"/> triangle is updated). / 列優先の <c>n × n</c> 対称行列 <c>A</c>（<paramref name="uplo"/> 側のみ更新）。</param>
    /// <param name="lda">The leading dimension of <c>A</c> (at least <paramref name="n"/>). / <c>A</c> のリーディングディメンション（<paramref name="n"/> 以上）。</param>
    /// <exception cref="ArgumentException">Dimensions, leading dimension, or span lengths are invalid. / サイズ・リーディングディメンション・スパン長が不正な場合。</exception>
    public static void Syr(BlasTriangle uplo, int n, double alpha, ReadOnlySpan<double> x, Span<double> a, int lda)
    {
        BlasCheck.Matrix(n, n, a.Length, lda);
        BlasCheck.Vector(x.Length, n, nameof(x));
        ParallelLevel2<double>.Syr(uplo, n, alpha, x, a, lda, hermitian: false);
    }

    /// <summary>
    /// Performs the symmetric rank-2 update <c>A ← α·x·yᵀ + α·y·xᵀ + A</c> in place (BLAS <c>dsyr2</c>).
    /// <para>対称ランク 2 更新 <c>A ← α·x·yᵀ + α·y·xᵀ + A</c> をインプレースで実行します（BLAS <c>dsyr2</c>）。</para>
    /// </summary>
    /// <param name="uplo">The triangular part of <c>A</c> to update. / 更新する <c>A</c> の三角部分。</param>
    /// <param name="n">The order of the symmetric matrix <c>A</c>. / 対称行列 <c>A</c> の次数。</param>
    /// <param name="alpha">The scalar multiplier α. / スカラー係数 α。</param>
    /// <param name="x">The first input vector of length at least <paramref name="n"/>. / 長さ <paramref name="n"/> 以上の第 1 入力ベクトル。</param>
    /// <param name="y">The second input vector of length at least <paramref name="n"/>. / 長さ <paramref name="n"/> 以上の第 2 入力ベクトル。</param>
    /// <param name="a">The column-major <c>n × n</c> symmetric matrix <c>A</c> (only the <paramref name="uplo"/> triangle is updated). / 列優先の <c>n × n</c> 対称行列 <c>A</c>（<paramref name="uplo"/> 側のみ更新）。</param>
    /// <param name="lda">The leading dimension of <c>A</c> (at least <paramref name="n"/>). / <c>A</c> のリーディングディメンション（<paramref name="n"/> 以上）。</param>
    /// <exception cref="ArgumentException">Dimensions, leading dimension, or span lengths are invalid. / サイズ・リーディングディメンション・スパン長が不正な場合。</exception>
    public static void Syr2(BlasTriangle uplo, int n, double alpha, ReadOnlySpan<double> x, ReadOnlySpan<double> y, Span<double> a, int lda)
    {
        BlasCheck.Matrix(n, n, a.Length, lda);
        BlasCheck.Vector(x.Length, n, nameof(x));
        BlasCheck.Vector(y.Length, n, nameof(y));
        ParallelLevel2<double>.Syr2(uplo, n, alpha, x, y, a, lda, hermitian: false);
    }

    /// <summary>
    /// Computes the triangular matrix-vector product <c>x ← op(A)·x</c> in place (BLAS <c>dtrmv</c>).
    /// <para>三角行列ベクトル積 <c>x ← op(A)·x</c> をインプレースで計算します（BLAS <c>dtrmv</c>）。</para>
    /// </summary>
    /// <param name="uplo">Specifies whether <c>A</c> is upper or lower triangular. / <c>A</c> が上三角か下三角かを指定します。</param>
    /// <param name="trans">The transpose operation applied to <c>A</c>. / <c>A</c> に適用する転置操作。</param>
    /// <param name="diag">Specifies whether <c>A</c> has unit diagonal elements. / <c>A</c> の対角要素を 1 とみなすかどうか。</param>
    /// <param name="n">The order of the triangular matrix <c>A</c>. / 三角行列 <c>A</c> の次数。</param>
    /// <param name="a">The column-major <c>n × n</c> triangular matrix <c>A</c> (only the <paramref name="uplo"/> triangle is read). / 列優先の <c>n × n</c> 三角行列 <c>A</c>（<paramref name="uplo"/> 側のみ参照）。</param>
    /// <param name="lda">The leading dimension of <c>A</c> (at least <paramref name="n"/>). / <c>A</c> のリーディングディメンション（<paramref name="n"/> 以上）。</param>
    /// <param name="x">The input/output vector of length at least <paramref name="n"/>, overwritten with <c>op(A)·x</c>. / 長さ <paramref name="n"/> 以上の入出力ベクトル（<c>op(A)·x</c> で上書き）。</param>
    /// <exception cref="ArgumentException">Dimensions, leading dimension, or span lengths are invalid. / サイズ・リーディングディメンション・スパン長が不正な場合。</exception>
    public static void Trmv(BlasTriangle uplo, BlasTranspose trans, BlasDiagonal diag, int n, ReadOnlySpan<double> a, int lda, Span<double> x)
    {
        BlasCheck.Matrix(n, n, a.Length, lda);
        BlasCheck.Vector(x.Length, n, nameof(x));
        ParallelLevel2<double>.Trmv(uplo, trans, diag, n, a, lda, x);
    }

    /// <summary>
    /// Solves the triangular linear system <c>op(A)·x = b</c> in place (BLAS <c>dtrsv</c>).
    /// <para>三角連立一次方程式 <c>op(A)·x = b</c> をインプレースで解きます（BLAS <c>dtrsv</c>）。</para>
    /// </summary>
    /// <param name="uplo">Specifies whether <c>A</c> is upper or lower triangular. / <c>A</c> が上三角か下三角かを指定します。</param>
    /// <param name="trans">The transpose operation applied to <c>A</c>. / <c>A</c> に適用する転置操作。</param>
    /// <param name="diag">Specifies whether <c>A</c> has unit diagonal elements. / <c>A</c> の対角要素を 1 とみなすかどうか。</param>
    /// <param name="n">The order of the triangular matrix <c>A</c>. / 三角行列 <c>A</c> の次数。</param>
    /// <param name="a">The column-major <c>n × n</c> triangular matrix <c>A</c> (only the <paramref name="uplo"/> triangle is read). / 列優先の <c>n × n</c> 三角行列 <c>A</c>（<paramref name="uplo"/> 側のみ参照）。</param>
    /// <param name="lda">The leading dimension of <c>A</c> (at least <paramref name="n"/>). / <c>A</c> のリーディングディメンション（<paramref name="n"/> 以上）。</param>
    /// <param name="x">On entry, the right-hand side <c>b</c> of length at least <paramref name="n"/>; on exit, overwritten with the solution <c>x</c>. / 入力時は長さ <paramref name="n"/> 以上の右辺ベクトル <c>b</c>、出力時は解 <c>x</c> で上書きされます。</param>
    /// <remarks>
    /// <para>Singularity is not checked; zero diagonal entries produce infinities or NaN.</para>
    /// <para>特異性の検査は行わないため、対角要素が 0 の場合は無限大または NaN が生じます。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">Dimensions, leading dimension, or span lengths are invalid. / サイズ・リーディングディメンション・スパン長が不正な場合。</exception>
    public static void Trsv(BlasTriangle uplo, BlasTranspose trans, BlasDiagonal diag, int n, ReadOnlySpan<double> a, int lda, Span<double> x)
    {
        BlasCheck.Matrix(n, n, a.Length, lda);
        BlasCheck.Vector(x.Length, n, nameof(x));
        ParallelLevel2<double>.Trsv(uplo, trans, diag, n, a, lda, x);
    }

    // ---- Level 2 (Complex) ----

    /// <summary>
    /// Computes the complex matrix-vector product <c>y ← α·op(A)·x + β·y</c> (BLAS <c>zgemv</c>).
    /// <para>複素行列ベクトル積 <c>y ← α·op(A)·x + β·y</c> を計算します（BLAS <c>zgemv</c>）。</para>
    /// </summary>
    /// <param name="trans">The transpose or conjugate-transpose operation applied to <c>A</c>. / <c>A</c> に適用する転置または共役転置操作。</param>
    /// <param name="m">The number of rows of <c>A</c>. / <c>A</c> の行数。</param>
    /// <param name="n">The number of columns of <c>A</c>. / <c>A</c> の列数。</param>
    /// <param name="alpha">The complex scalar multiplier α. / 複素スカラー係数 α。</param>
    /// <param name="a">The column-major <c>m × n</c> matrix <c>A</c>. / 列優先の <c>m × n</c> 行列 <c>A</c>。</param>
    /// <param name="lda">The leading dimension of <c>A</c> (at least <paramref name="m"/>). / <c>A</c> のリーディングディメンション（<paramref name="m"/> 以上）。</param>
    /// <param name="x">The input vector of length at least <paramref name="n"/> (or <paramref name="m"/> when transposed). / 長さ <paramref name="n"/> 以上（転置時は <paramref name="m"/> 以上）の入力ベクトル。</param>
    /// <param name="beta">The complex scalar multiplier β applied to <paramref name="y"/>; if 0, the initial contents of <paramref name="y"/> are not read. / <paramref name="y"/> に掛ける係数 β（0 の場合は <paramref name="y"/> の元の値を読み取りません）。</param>
    /// <param name="y">The input/output vector of length at least <paramref name="m"/> (or <paramref name="n"/> when transposed), updated in place. / 長さ <paramref name="m"/> 以上（転置時は <paramref name="n"/> 以上）の入出力ベクトル（インプレースで更新）。</param>
    /// <exception cref="ArgumentException">Dimensions, leading dimension, or span lengths are invalid. / サイズ・リーディングディメンション・スパン長が不正な場合。</exception>
    public static void Gemv(BlasTranspose trans, int m, int n, Complex alpha, ReadOnlySpan<Complex> a, int lda, ReadOnlySpan<Complex> x, Complex beta, Span<Complex> y)
    {
        BlasCheck.Matrix(m, n, a.Length, lda);
        bool notrans = trans == BlasTranspose.None;
        BlasCheck.Vector(x.Length, notrans ? n : m, nameof(x));
        BlasCheck.Vector(y.Length, notrans ? m : n, nameof(y));
        ParallelLevel2<Complex>.Gemv(trans, m, n, alpha, a, lda, x, beta, y);
    }

    /// <summary>
    /// Performs the unconjugated complex rank-1 update <c>A ← α·x·yᵀ + A</c> in place (BLAS <c>zgeru</c>).
    /// <para>共役を取らない複素ランク 1 更新 <c>A ← α·x·yᵀ + A</c> をインプレースで実行します（BLAS <c>zgeru</c>）。</para>
    /// </summary>
    /// <param name="m">The number of rows of <c>A</c>. / <c>A</c> の行数。</param>
    /// <param name="n">The number of columns of <c>A</c>. / <c>A</c> の列数。</param>
    /// <param name="alpha">The complex scalar multiplier α. / 複素スカラー係数 α。</param>
    /// <param name="x">The input vector of length at least <paramref name="m"/>. / 長さ <paramref name="m"/> 以上の入力ベクトル。</param>
    /// <param name="y">The input vector of length at least <paramref name="n"/>. / 長さ <paramref name="n"/> 以上の入力ベクトル。</param>
    /// <param name="a">The column-major <c>m × n</c> matrix <c>A</c>, updated in place. / インプレースで更新される列優先の <c>m × n</c> 行列 <c>A</c>。</param>
    /// <param name="lda">The leading dimension of <c>A</c> (at least <paramref name="m"/>). / <c>A</c> のリーディングディメンション（<paramref name="m"/> 以上）。</param>
    /// <exception cref="ArgumentException">Dimensions, leading dimension, or span lengths are invalid. / サイズ・リーディングディメンション・スパン長が不正な場合。</exception>
    public static void Ger(int m, int n, Complex alpha, ReadOnlySpan<Complex> x, ReadOnlySpan<Complex> y, Span<Complex> a, int lda)
    {
        BlasCheck.Matrix(m, n, a.Length, lda);
        BlasCheck.Vector(x.Length, m, nameof(x));
        BlasCheck.Vector(y.Length, n, nameof(y));
        ParallelLevel2<Complex>.Ger(m, n, alpha, x, y, a, lda, conjugateY: false);
    }

    /// <summary>
    /// Performs the conjugate complex rank-1 update <c>A ← α·x·yᴴ + A</c> in place (BLAS <c>zgerc</c>).
    /// <para>共役を取る複素ランク 1 更新 <c>A ← α·x·yᴴ + A</c> をインプレースで実行します（BLAS <c>zgerc</c>）。</para>
    /// </summary>
    /// <param name="m">The number of rows of <c>A</c>. / <c>A</c> の行数。</param>
    /// <param name="n">The number of columns of <c>A</c>. / <c>A</c> の列数。</param>
    /// <param name="alpha">The complex scalar multiplier α. / 複素スカラー係数 α。</param>
    /// <param name="x">The input vector of length at least <paramref name="m"/>. / 長さ <paramref name="m"/> 以上の入力ベクトル。</param>
    /// <param name="y">The input vector of length at least <paramref name="n"/> (conjugated). / 長さ <paramref name="n"/> 以上の入力ベクトル（共役を取ります）。</param>
    /// <param name="a">The column-major <c>m × n</c> matrix <c>A</c>, updated in place. / インプレースで更新される列優先の <c>m × n</c> 行列 <c>A</c>。</param>
    /// <param name="lda">The leading dimension of <c>A</c> (at least <paramref name="m"/>). / <c>A</c> のリーディングディメンション（<paramref name="m"/> 以上）。</param>
    /// <exception cref="ArgumentException">Dimensions, leading dimension, or span lengths are invalid. / サイズ・リーディングディメンション・スパン長が不正な場合。</exception>
    public static void GerConjugate(int m, int n, Complex alpha, ReadOnlySpan<Complex> x, ReadOnlySpan<Complex> y, Span<Complex> a, int lda)
    {
        BlasCheck.Matrix(m, n, a.Length, lda);
        BlasCheck.Vector(x.Length, m, nameof(x));
        BlasCheck.Vector(y.Length, n, nameof(y));
        ParallelLevel2<Complex>.Ger(m, n, alpha, x, y, a, lda, conjugateY: true);
    }

    /// <summary>
    /// Computes the complex symmetric matrix-vector product <c>y ← α·A·x + β·y</c> (LAPACK <c>zsymv</c>).
    /// <para>複素対称行列ベクトル積 <c>y ← α·A·x + β·y</c> を計算します（LAPACK <c>zsymv</c>）。</para>
    /// </summary>
    /// <param name="uplo">The triangular part of <c>A</c> to reference. / <c>A</c> の参照する三角部分。</param>
    /// <param name="n">The order of the symmetric matrix <c>A</c>. / 対称行列 <c>A</c> の次数。</param>
    /// <param name="alpha">The complex scalar multiplier α. / 複素スカラー係数 α。</param>
    /// <param name="a">The column-major <c>n × n</c> complex symmetric matrix <c>A</c> (only the <paramref name="uplo"/> triangle is read). / 列優先の <c>n × n</c> 複素対称行列 <c>A</c>（<paramref name="uplo"/> 側のみ参照）。</param>
    /// <param name="lda">The leading dimension of <c>A</c> (at least <paramref name="n"/>). / <c>A</c> のリーディングディメンション（<paramref name="n"/> 以上）。</param>
    /// <param name="x">The input vector of length at least <paramref name="n"/>. / 長さ <paramref name="n"/> 以上の入力ベクトル。</param>
    /// <param name="beta">The complex scalar multiplier β applied to <paramref name="y"/>. / <paramref name="y"/> に掛ける係数 β。</param>
    /// <param name="y">The input/output vector of length at least <paramref name="n"/>, updated in place. / 長さ <paramref name="n"/> 以上の入出力ベクトル（インプレースで更新）。</param>
    /// <exception cref="ArgumentException">Dimensions, leading dimension, or span lengths are invalid. / サイズ・リーディングディメンション・スパン長が不正な場合。</exception>
    public static void Symv(BlasTriangle uplo, int n, Complex alpha, ReadOnlySpan<Complex> a, int lda, ReadOnlySpan<Complex> x, Complex beta, Span<Complex> y)
    {
        BlasCheck.Matrix(n, n, a.Length, lda);
        BlasCheck.Vector(x.Length, n, nameof(x));
        BlasCheck.Vector(y.Length, n, nameof(y));
        ParallelLevel2<Complex>.Symv(uplo, n, alpha, a, lda, x, beta, y, hermitian: false);
    }

    /// <summary>
    /// Computes the Hermitian matrix-vector product <c>y ← α·A·x + β·y</c> (BLAS <c>zhemv</c>).
    /// <para>エルミート行列ベクトル積 <c>y ← α·A·x + β·y</c> を計算します（BLAS <c>zhemv</c>）。</para>
    /// </summary>
    /// <param name="uplo">The triangular part of <c>A</c> to reference. / <c>A</c> の参照する三角部分。</param>
    /// <param name="n">The order of the Hermitian matrix <c>A</c>. / エルミート行列 <c>A</c> の次数。</param>
    /// <param name="alpha">The complex scalar multiplier α. / 複素スカラー係数 α。</param>
    /// <param name="a">The column-major <c>n × n</c> Hermitian matrix <c>A</c> (only the <paramref name="uplo"/> triangle is read; imaginary parts on the diagonal are assumed 0). / 列優先の <c>n × n</c> エルミート行列 <c>A</c>（<paramref name="uplo"/> 側のみ参照し、対角の虚部は 0 とみなします）。</param>
    /// <param name="lda">The leading dimension of <c>A</c> (at least <paramref name="n"/>). / <c>A</c> のリーディングディメンション（<paramref name="n"/> 以上）。</param>
    /// <param name="x">The input vector of length at least <paramref name="n"/>. / 長さ <paramref name="n"/> 以上の入力ベクトル。</param>
    /// <param name="beta">The complex scalar multiplier β applied to <paramref name="y"/>. / <paramref name="y"/> に掛ける係数 β。</param>
    /// <param name="y">The input/output vector of length at least <paramref name="n"/>, updated in place. / 長さ <paramref name="n"/> 以上の入出力ベクトル（インプレースで更新）。</param>
    /// <exception cref="ArgumentException">Dimensions, leading dimension, or span lengths are invalid. / サイズ・リーディングディメンション・スパン長が不正な場合。</exception>
    public static void Hemv(BlasTriangle uplo, int n, Complex alpha, ReadOnlySpan<Complex> a, int lda, ReadOnlySpan<Complex> x, Complex beta, Span<Complex> y)
    {
        BlasCheck.Matrix(n, n, a.Length, lda);
        BlasCheck.Vector(x.Length, n, nameof(x));
        BlasCheck.Vector(y.Length, n, nameof(y));
        ParallelLevel2<Complex>.Symv(uplo, n, alpha, a, lda, x, beta, y, hermitian: true);
    }

    /// <summary>
    /// Performs the Hermitian rank-1 update <c>A ← α·x·xᴴ + A</c> in place (BLAS <c>zher</c>).
    /// <para>エルミートランク 1 更新 <c>A ← α·x·xᴴ + A</c> をインプレースで実行します（BLAS <c>zher</c>）。</para>
    /// </summary>
    /// <param name="uplo">The triangular part of <c>A</c> to update. / 更新する <c>A</c> の三角部分。</param>
    /// <param name="n">The order of the Hermitian matrix <c>A</c>. / エルミート行列 <c>A</c> の次数。</param>
    /// <param name="alpha">The real scalar multiplier α. / 実スカラー係数 α。</param>
    /// <param name="x">The input vector of length at least <paramref name="n"/>. / 長さ <paramref name="n"/> 以上の入力ベクトル。</param>
    /// <param name="a">The column-major <c>n × n</c> Hermitian matrix <c>A</c> (only the <paramref name="uplo"/> triangle is updated and diagonal imaginary parts are set to 0). / 列優先の <c>n × n</c> エルミート行列 <c>A</c>（<paramref name="uplo"/> 側のみ更新し、対角の虚部を 0 に設定）。</param>
    /// <param name="lda">The leading dimension of <c>A</c> (at least <paramref name="n"/>). / <c>A</c> のリーディングディメンション（<paramref name="n"/> 以上）。</param>
    /// <exception cref="ArgumentException">Dimensions, leading dimension, or span lengths are invalid. / サイズ・リーディングディメンション・スパン長が不正な場合。</exception>
    public static void Her(BlasTriangle uplo, int n, double alpha, ReadOnlySpan<Complex> x, Span<Complex> a, int lda)
    {
        BlasCheck.Matrix(n, n, a.Length, lda);
        BlasCheck.Vector(x.Length, n, nameof(x));
        ParallelLevel2<Complex>.Syr(uplo, n, alpha, x, a, lda, hermitian: true);
    }

    /// <summary>
    /// Performs the Hermitian rank-2 update <c>A ← α·x·yᴴ + conj(α)·y·xᴴ + A</c> in place (BLAS <c>zher2</c>).
    /// <para>エルミートランク 2 更新 <c>A ← α·x·yᴴ + conj(α)·y·xᴴ + A</c> をインプレースで実行します（BLAS <c>zher2</c>）。</para>
    /// </summary>
    /// <param name="uplo">The triangular part of <c>A</c> to update. / 更新する <c>A</c> の三角部分。</param>
    /// <param name="n">The order of the Hermitian matrix <c>A</c>. / エルミート行列 <c>A</c> の次数。</param>
    /// <param name="alpha">The complex scalar multiplier α. / 複素スカラー係数 α。</param>
    /// <param name="x">The first input vector of length at least <paramref name="n"/>. / 長さ <paramref name="n"/> 以上の第 1 入力ベクトル。</param>
    /// <param name="y">The second input vector of length at least <paramref name="n"/>. / 長さ <paramref name="n"/> 以上の第 2 入力ベクトル。</param>
    /// <param name="a">The column-major <c>n × n</c> Hermitian matrix <c>A</c> (only the <paramref name="uplo"/> triangle is updated and diagonal imaginary parts are set to 0). / 列優先の <c>n × n</c> エルミート行列 <c>A</c>（<paramref name="uplo"/> 側のみ更新し、対角の虚部を 0 に設定）。</param>
    /// <param name="lda">The leading dimension of <c>A</c> (at least <paramref name="n"/>). / <c>A</c> のリーディングディメンション（<paramref name="n"/> 以上）。</param>
    /// <exception cref="ArgumentException">Dimensions, leading dimension, or span lengths are invalid. / サイズ・リーディングディメンション・スパン長が不正な場合。</exception>
    public static void Her2(BlasTriangle uplo, int n, Complex alpha, ReadOnlySpan<Complex> x, ReadOnlySpan<Complex> y, Span<Complex> a, int lda)
    {
        BlasCheck.Matrix(n, n, a.Length, lda);
        BlasCheck.Vector(x.Length, n, nameof(x));
        BlasCheck.Vector(y.Length, n, nameof(y));
        ParallelLevel2<Complex>.Syr2(uplo, n, alpha, x, y, a, lda, hermitian: true);
    }

    /// <summary>
    /// Computes the complex triangular matrix-vector product <c>x ← op(A)·x</c> in place (BLAS <c>ztrmv</c>).
    /// <para>複素三角行列ベクトル積 <c>x ← op(A)·x</c> をインプレースで計算します（BLAS <c>ztrmv</c>）。</para>
    /// </summary>
    /// <param name="uplo">Specifies whether <c>A</c> is upper or lower triangular. / <c>A</c> が上三角か下三角かを指定します。</param>
    /// <param name="trans">The transpose or conjugate-transpose operation applied to <c>A</c>. / <c>A</c> に適用する転置または共役転置操作。</param>
    /// <param name="diag">Specifies whether <c>A</c> has unit diagonal elements. / <c>A</c> の対角要素を 1 とみなすかどうか。</param>
    /// <param name="n">The order of the triangular matrix <c>A</c>. / 三角行列 <c>A</c> の次数。</param>
    /// <param name="a">The column-major <c>n × n</c> triangular matrix <c>A</c> (only the <paramref name="uplo"/> triangle is read). / 列優先の <c>n × n</c> 三角行列 <c>A</c>（<paramref name="uplo"/> 側のみ参照）。</param>
    /// <param name="lda">The leading dimension of <c>A</c> (at least <paramref name="n"/>). / <c>A</c> のリーディングディメンション（<paramref name="n"/> 以上）。</param>
    /// <param name="x">The input/output vector of length at least <paramref name="n"/>, overwritten with <c>op(A)·x</c>. / 長さ <paramref name="n"/> 以上の入出力ベクトル（<c>op(A)·x</c> で上書き）。</param>
    /// <exception cref="ArgumentException">Dimensions, leading dimension, or span lengths are invalid. / サイズ・リーディングディメンション・スパン長が不正な場合。</exception>
    public static void Trmv(BlasTriangle uplo, BlasTranspose trans, BlasDiagonal diag, int n, ReadOnlySpan<Complex> a, int lda, Span<Complex> x)
    {
        BlasCheck.Matrix(n, n, a.Length, lda);
        BlasCheck.Vector(x.Length, n, nameof(x));
        ParallelLevel2<Complex>.Trmv(uplo, trans, diag, n, a, lda, x);
    }

    /// <summary>
    /// Solves the complex triangular linear system <c>op(A)·x = b</c> in place (BLAS <c>ztrsv</c>).
    /// <para>複素三角連立一次方程式 <c>op(A)·x = b</c> をインプレースで解きます（BLAS <c>ztrsv</c>）。</para>
    /// </summary>
    /// <param name="uplo">Specifies whether <c>A</c> is upper or lower triangular. / <c>A</c> が上三角か下三角かを指定します。</param>
    /// <param name="trans">The transpose or conjugate-transpose operation applied to <c>A</c>. / <c>A</c> に適用する転置または共役転置操作。</param>
    /// <param name="diag">Specifies whether <c>A</c> has unit diagonal elements. / <c>A</c> の対角要素を 1 とみなすかどうか。</param>
    /// <param name="n">The order of the triangular matrix <c>A</c>. / 三角行列 <c>A</c> の次数。</param>
    /// <param name="a">The column-major <c>n × n</c> triangular matrix <c>A</c> (only the <paramref name="uplo"/> triangle is read). / 列優先の <c>n × n</c> 三角行列 <c>A</c>（<paramref name="uplo"/> 側のみ参照）。</param>
    /// <param name="lda">The leading dimension of <c>A</c> (at least <paramref name="n"/>). / <c>A</c> のリーディングディメンション（<paramref name="n"/> 以上）。</param>
    /// <param name="x">On entry, the right-hand side <c>b</c> of length at least <paramref name="n"/>; on exit, overwritten with the solution <c>x</c>. / 入力時は長さ <paramref name="n"/> 以上の右辺ベクトル <c>b</c>、出力時は解 <c>x</c> で上書きされます。</param>
    /// <remarks>
    /// <para>Singularity is not checked; zero diagonal entries produce infinities or NaN.</para>
    /// <para>特異性の検査は行わないため、対角要素が 0 の場合は無限大または NaN が生じます。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">Dimensions, leading dimension, or span lengths are invalid. / サイズ・リーディングディメンション・スパン長が不正な場合。</exception>
    public static void Trsv(BlasTriangle uplo, BlasTranspose trans, BlasDiagonal diag, int n, ReadOnlySpan<Complex> a, int lda, Span<Complex> x)
    {
        BlasCheck.Matrix(n, n, a.Length, lda);
        BlasCheck.Vector(x.Length, n, nameof(x));
        ParallelLevel2<Complex>.Trsv(uplo, trans, diag, n, a, lda, x);
    }

    // ---- Level 2 (binary integer) ----

    /// <summary>
    /// Computes the integer matrix-vector product <c>y ← α·op(A)·x + β·y</c>.
    /// <para>整数型の行列ベクトル積 <c>y ← α·op(A)·x + β·y</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The binary integer element type. / 2 進整数要素型。</typeparam>
    /// <param name="trans">The transpose operation applied to <c>A</c>. / <c>A</c> に適用する転置操作。</param>
    /// <param name="m">The number of rows of <c>A</c>. / <c>A</c> の行数。</param>
    /// <param name="n">The number of columns of <c>A</c>. / <c>A</c> の列数。</param>
    /// <param name="alpha">The scalar multiplier α. / スカラー係数 α。</param>
    /// <param name="a">The column-major <c>m × n</c> matrix <c>A</c>. / 列優先の <c>m × n</c> 行列 <c>A</c>。</param>
    /// <param name="lda">The leading dimension of <c>A</c> (at least <paramref name="m"/>). / <c>A</c> のリーディングディメンション（<paramref name="m"/> 以上）。</param>
    /// <param name="x">The input vector of length at least <paramref name="n"/> (or <paramref name="m"/> when transposed). / 長さ <paramref name="n"/> 以上（転置時は <paramref name="m"/> 以上）の入力ベクトル。</param>
    /// <param name="beta">The scalar multiplier β applied to <paramref name="y"/>; if 0, the initial contents of <paramref name="y"/> are not read. / <paramref name="y"/> に掛ける係数 β（0 の場合は <paramref name="y"/> の元の値を読み取りません）。</param>
    /// <param name="y">The input/output vector of length at least <paramref name="m"/> (or <paramref name="n"/> when transposed), updated in place. / 長さ <paramref name="m"/> 以上（転置時は <paramref name="n"/> 以上）の入出力ベクトル（インプレースで更新）。</param>
    /// <exception cref="ArgumentException">Dimensions, leading dimension, or span lengths are invalid. / サイズ・リーディングディメンション・スパン長が不正な場合。</exception>
    public static void Gemv<T>(BlasTranspose trans, int m, int n, T alpha, ReadOnlySpan<T> a, int lda, ReadOnlySpan<T> x, T beta, Span<T> y)
        where T : unmanaged, IBinaryInteger<T>
    {
        BlasCheck.Matrix(m, n, a.Length, lda);
        bool notrans = trans == BlasTranspose.None;
        BlasCheck.Vector(x.Length, notrans ? n : m, nameof(x));
        BlasCheck.Vector(y.Length, notrans ? m : n, nameof(y));
        ParallelLevel2<T>.Gemv(trans, m, n, alpha, a, lda, x, beta, y);
    }

    /// <summary>
    /// Performs the integer rank-1 update <c>A ← α·x·yᵀ + A</c> in place.
    /// <para>整数型のランク 1 更新 <c>A ← α·x·yᵀ + A</c> をインプレースで実行します。</para>
    /// </summary>
    /// <typeparam name="T">The binary integer element type. / 2 進整数要素型。</typeparam>
    /// <param name="m">The number of rows of <c>A</c>. / <c>A</c> の行数。</param>
    /// <param name="n">The number of columns of <c>A</c>. / <c>A</c> の列数。</param>
    /// <param name="alpha">The scalar multiplier α. / スカラー係数 α。</param>
    /// <param name="x">The input vector of length at least <paramref name="m"/>. / 長さ <paramref name="m"/> 以上の入力ベクトル。</param>
    /// <param name="y">The input vector of length at least <paramref name="n"/>. / 長さ <paramref name="n"/> 以上の入力ベクトル。</param>
    /// <param name="a">The column-major <c>m × n</c> matrix <c>A</c>, updated in place. / インプレースで更新される列優先の <c>m × n</c> 行列 <c>A</c>。</param>
    /// <param name="lda">The leading dimension of <c>A</c> (at least <paramref name="m"/>). / <c>A</c> のリーディングディメンション（<paramref name="m"/> 以上）。</param>
    /// <exception cref="ArgumentException">Dimensions, leading dimension, or span lengths are invalid. / サイズ・リーディングディメンション・スパン長が不正な場合。</exception>
    public static void Ger<T>(int m, int n, T alpha, ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> a, int lda)
        where T : unmanaged, IBinaryInteger<T>
    {
        BlasCheck.Matrix(m, n, a.Length, lda);
        BlasCheck.Vector(x.Length, m, nameof(x));
        BlasCheck.Vector(y.Length, n, nameof(y));
        ParallelLevel2<T>.Ger(m, n, alpha, x, y, a, lda, conjugateY: false);
    }

    /// <summary>
    /// Computes the integer symmetric matrix-vector product <c>y ← α·A·x + β·y</c>.
    /// <para>整数型の対称行列ベクトル積 <c>y ← α·A·x + β·y</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The binary integer element type. / 2 進整数要素型。</typeparam>
    /// <param name="uplo">The triangular part of <c>A</c> to reference. / <c>A</c> の参照する三角部分。</param>
    /// <param name="n">The order of the symmetric matrix <c>A</c>. / 対称行列 <c>A</c> の次数。</param>
    /// <param name="alpha">The scalar multiplier α. / スカラー係数 α。</param>
    /// <param name="a">The column-major <c>n × n</c> symmetric matrix <c>A</c> (only the <paramref name="uplo"/> triangle is read). / 列優先の <c>n × n</c> 対称行列 <c>A</c>（<paramref name="uplo"/> 側のみ参照）。</param>
    /// <param name="lda">The leading dimension of <c>A</c> (at least <paramref name="n"/>). / <c>A</c> のリーディングディメンション（<paramref name="n"/> 以上）。</param>
    /// <param name="x">The input vector of length at least <paramref name="n"/>. / 長さ <paramref name="n"/> 以上の入力ベクトル。</param>
    /// <param name="beta">The scalar multiplier β applied to <paramref name="y"/>. / <paramref name="y"/> に掛ける係数 β。</param>
    /// <param name="y">The input/output vector of length at least <paramref name="n"/>, updated in place. / 長さ <paramref name="n"/> 以上の入出力ベクトル（インプレースで更新）。</param>
    /// <exception cref="ArgumentException">Dimensions, leading dimension, or span lengths are invalid. / サイズ・リーディングディメンション・スパン長が不正な場合。</exception>
    public static void Symv<T>(BlasTriangle uplo, int n, T alpha, ReadOnlySpan<T> a, int lda, ReadOnlySpan<T> x, T beta, Span<T> y)
        where T : unmanaged, IBinaryInteger<T>
    {
        BlasCheck.Matrix(n, n, a.Length, lda);
        BlasCheck.Vector(x.Length, n, nameof(x));
        BlasCheck.Vector(y.Length, n, nameof(y));
        ParallelLevel2<T>.Symv(uplo, n, alpha, a, lda, x, beta, y, hermitian: false);
    }

    /// <summary>
    /// Performs the integer symmetric rank-1 update <c>A ← α·x·xᵀ + A</c> in place.
    /// <para>整数型の対称ランク 1 更新 <c>A ← α·x·xᵀ + A</c> をインプレースで実行します。</para>
    /// </summary>
    /// <typeparam name="T">The binary integer element type. / 2 進整数要素型。</typeparam>
    /// <param name="uplo">The triangular part of <c>A</c> to update. / 更新する <c>A</c> の三角部分。</param>
    /// <param name="n">The order of the symmetric matrix <c>A</c>. / 対称行列 <c>A</c> の次数。</param>
    /// <param name="alpha">The scalar multiplier α. / スカラー係数 α。</param>
    /// <param name="x">The input vector of length at least <paramref name="n"/>. / 長さ <paramref name="n"/> 以上の入力ベクトル。</param>
    /// <param name="a">The column-major <c>n × n</c> symmetric matrix <c>A</c> (only the <paramref name="uplo"/> triangle is updated). / 列優先の <c>n × n</c> 対称行列 <c>A</c>（<paramref name="uplo"/> 側のみ更新）。</param>
    /// <param name="lda">The leading dimension of <c>A</c> (at least <paramref name="n"/>). / <c>A</c> のリーディングディメンション（<paramref name="n"/> 以上）。</param>
    /// <exception cref="ArgumentException">Dimensions, leading dimension, or span lengths are invalid. / サイズ・リーディングディメンション・スパン長が不正な場合。</exception>
    public static void Syr<T>(BlasTriangle uplo, int n, T alpha, ReadOnlySpan<T> x, Span<T> a, int lda)
        where T : unmanaged, IBinaryInteger<T>
    {
        BlasCheck.Matrix(n, n, a.Length, lda);
        BlasCheck.Vector(x.Length, n, nameof(x));
        ParallelLevel2<T>.Syr(uplo, n, alpha, x, a, lda, hermitian: false);
    }

    /// <summary>
    /// Performs the integer symmetric rank-2 update <c>A ← α·x·yᵀ + α·y·xᵀ + A</c> in place.
    /// <para>整数型の対称ランク 2 更新 <c>A ← α·x·yᵀ + α·y·xᵀ + A</c> をインプレースで実行します。</para>
    /// </summary>
    /// <typeparam name="T">The binary integer element type. / 2 進整数要素型。</typeparam>
    /// <param name="uplo">The triangular part of <c>A</c> to update. / 更新する <c>A</c> の三角部分。</param>
    /// <param name="n">The order of the symmetric matrix <c>A</c>. / 対称行列 <c>A</c> の次数。</param>
    /// <param name="alpha">The scalar multiplier α. / スカラー係数 α。</param>
    /// <param name="x">The first input vector of length at least <paramref name="n"/>. / 長さ <paramref name="n"/> 以上の第 1 入力ベクトル。</param>
    /// <param name="y">The second input vector of length at least <paramref name="n"/>. / 長さ <paramref name="n"/> 以上の第 2 入力ベクトル。</param>
    /// <param name="a">The column-major <c>n × n</c> symmetric matrix <c>A</c> (only the <paramref name="uplo"/> triangle is updated). / 列優先の <c>n × n</c> 対称行列 <c>A</c>（<paramref name="uplo"/> 側のみ更新）。</param>
    /// <param name="lda">The leading dimension of <c>A</c> (at least <paramref name="n"/>). / <c>A</c> のリーディングディメンション（<paramref name="n"/> 以上）。</param>
    /// <exception cref="ArgumentException">Dimensions, leading dimension, or span lengths are invalid. / サイズ・リーディングディメンション・スパン長が不正な場合。</exception>
    public static void Syr2<T>(BlasTriangle uplo, int n, T alpha, ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> a, int lda)
        where T : unmanaged, IBinaryInteger<T>
    {
        BlasCheck.Matrix(n, n, a.Length, lda);
        BlasCheck.Vector(x.Length, n, nameof(x));
        BlasCheck.Vector(y.Length, n, nameof(y));
        ParallelLevel2<T>.Syr2(uplo, n, alpha, x, y, a, lda, hermitian: false);
    }

    /// <summary>
    /// Computes the integer triangular matrix-vector product <c>x ← op(A)·x</c> in place.
    /// <para>整数型の三角行列ベクトル積 <c>x ← op(A)·x</c> をインプレースで計算します。</para>
    /// </summary>
    /// <typeparam name="T">The binary integer element type. / 2 進整数要素型。</typeparam>
    /// <param name="uplo">Specifies whether <c>A</c> is upper or lower triangular. / <c>A</c> が上三角か下三角かを指定します。</param>
    /// <param name="trans">The transpose operation applied to <c>A</c>. / <c>A</c> に適用する転置操作。</param>
    /// <param name="diag">Specifies whether <c>A</c> has unit diagonal elements. / <c>A</c> の対角要素を 1 とみなすかどうか。</param>
    /// <param name="n">The order of the triangular matrix <c>A</c>. / 三角行列 <c>A</c> の次数。</param>
    /// <param name="a">The column-major <c>n × n</c> triangular matrix <c>A</c> (only the <paramref name="uplo"/> triangle is read). / 列優先の <c>n × n</c> 三角行列 <c>A</c>（<paramref name="uplo"/> 側のみ参照）。</param>
    /// <param name="lda">The leading dimension of <c>A</c> (at least <paramref name="n"/>). / <c>A</c> のリーディングディメンション（<paramref name="n"/> 以上）。</param>
    /// <param name="x">The input/output vector of length at least <paramref name="n"/>, overwritten with <c>op(A)·x</c>. / 長さ <paramref name="n"/> 以上の入出力ベクトル（<c>op(A)·x</c> で上書き）。</param>
    /// <exception cref="ArgumentException">Dimensions, leading dimension, or span lengths are invalid. / サイズ・リーディングディメンション・スパン長が不正な場合。</exception>
    public static void Trmv<T>(BlasTriangle uplo, BlasTranspose trans, BlasDiagonal diag, int n, ReadOnlySpan<T> a, int lda, Span<T> x)
        where T : unmanaged, IBinaryInteger<T>
    {
        BlasCheck.Matrix(n, n, a.Length, lda);
        BlasCheck.Vector(x.Length, n, nameof(x));
        ParallelLevel2<T>.Trmv(uplo, trans, diag, n, a, lda, x);
    }
}
