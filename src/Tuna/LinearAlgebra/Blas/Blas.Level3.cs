namespace Tuna.LinearAlgebra;

public static partial class Blas
{
    // ---- Level 3 (float) ----

    /// <summary>
    /// Computes the general matrix-matrix product <c>C ← α·op(A)·op(B) + β·C</c> (BLAS <c>sgemm</c>).
    /// <para>一般行列積 <c>C ← α·op(A)·op(B) + β·C</c> を計算します（BLAS <c>sgemm</c>）。</para>
    /// </summary>
    /// <param name="transA">The transpose operation applied to <c>A</c>. / <c>A</c> に適用する転置操作。</param>
    /// <param name="transB">The transpose operation applied to <c>B</c>. / <c>B</c> に適用する転置操作。</param>
    /// <param name="m">The number of rows of <c>op(A)</c> and <c>C</c>. / <c>op(A)</c> および <c>C</c> の行数。</param>
    /// <param name="n">The number of columns of <c>op(B)</c> and <c>C</c>. / <c>op(B)</c> および <c>C</c> の列数。</param>
    /// <param name="k">The number of columns of <c>op(A)</c> and rows of <c>op(B)</c>. / <c>op(A)</c> の列数かつ <c>op(B)</c> の行数。</param>
    /// <param name="alpha">The scalar multiplier α. / スカラー係数 α。</param>
    /// <param name="a">The column-major matrix <c>A</c> (<c>m × k</c>, or <c>k × m</c> when transposed). / 列優先の行列 <c>A</c>（転置なしは <c>m × k</c>、転置ありは <c>k × m</c>）。</param>
    /// <param name="lda">The leading dimension of <c>A</c> (at least the row count of <c>A</c>). / <c>A</c> のリーディングディメンション（<c>A</c> の行数以上）。</param>
    /// <param name="b">The column-major matrix <c>B</c> (<c>k × n</c>, or <c>n × k</c> when transposed). / 列優先の行列 <c>B</c>（転置なしは <c>k × n</c>、転置ありは <c>n × k</c>）。</param>
    /// <param name="ldb">The leading dimension of <c>B</c> (at least the row count of <c>B</c>). / <c>B</c> のリーディングディメンション（<c>B</c> の行数以上）。</param>
    /// <param name="beta">The scalar multiplier β applied to <c>C</c>. / <c>C</c> に掛けるスカラー係数 β。</param>
    /// <param name="c">The column-major output matrix <c>C</c> (<c>m × n</c>), overwritten in place; padding elements beyond <paramref name="m"/> in each column are untouched. / 列優先の出力行列 <c>C</c>（<c>m × n</c>）。インプレースで上書きされ、各列の <paramref name="m"/> 行目以降の余白は読み書きされません。</param>
    /// <param name="ldc">The leading dimension of <c>C</c> (at least <paramref name="m"/>). / <c>C</c> のリーディングディメンション（<paramref name="m"/> 以上）。</param>
    /// <remarks>
    /// <para><paramref name="c"/> must not share any elements with <paramref name="a"/> or <paramref name="b"/> (disjoint submatrices within the same backing array are allowed); <paramref name="a"/> and <paramref name="b"/> may overlap. When <paramref name="alpha"/> or <paramref name="k"/> is 0, <paramref name="a"/> and <paramref name="b"/> are not read. When <paramref name="beta"/> is 0, the initial contents of <paramref name="c"/> are not read. Results are bit-for-bit identical regardless of thread count.</para>
    /// <para><paramref name="c"/> は <paramref name="a"/> および <paramref name="b"/> と要素を共有してはなりません（同じ配列内の互いに素な部分行列であれば可）。<paramref name="a"/> と <paramref name="b"/> は重複可能です。<paramref name="alpha"/> または <paramref name="k"/> が 0 の場合は <paramref name="a"/>・<paramref name="b"/> を読み取らず、<paramref name="beta"/> が 0 の場合は <paramref name="c"/> の元の値を読み取りません。計算結果はスレッド数によらずビット単位で同一です。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">Matrix dimensions or leading dimensions are invalid, or <paramref name="c"/> aliases elements of <paramref name="a"/> or <paramref name="b"/>. / 行列のサイズやリーディングディメンションが不正、または <paramref name="c"/> が <paramref name="a"/> や <paramref name="b"/> と要素を共有している場合。</exception>
    public static void Gemm(
        BlasTranspose transA, BlasTranspose transB, int m, int n, int k,
        float alpha, ReadOnlySpan<float> a, int lda, ReadOnlySpan<float> b, int ldb, float beta, Span<float> c, int ldc)
    {
        bool ta = transA != BlasTranspose.None, tb = transB != BlasTranspose.None;
        BlasCheck.MatrixProduct(ta, tb, m, n, k, a, lda, b, ldb, c, ldc);
        ParallelLevel2<float>.ScaleOrClear(beta, m, n, c, ldc);
        GemmSingle.Multiply(ta, tb, m, n, k, alpha, a, lda, b, ldb, c, ldc);
    }

    // ---- Level 3 (double) ----

    /// <summary>
    /// Computes the general matrix-matrix product <c>C ← α·op(A)·op(B) + β·C</c> (BLAS <c>dgemm</c>).
    /// <para>一般行列積 <c>C ← α·op(A)·op(B) + β·C</c> を計算します（BLAS <c>dgemm</c>）。</para>
    /// </summary>
    /// <param name="transA">The transpose operation applied to <c>A</c>. / <c>A</c> に適用する転置操作。</param>
    /// <param name="transB">The transpose operation applied to <c>B</c>. / <c>B</c> に適用する転置操作。</param>
    /// <param name="m">The number of rows of <c>op(A)</c> and <c>C</c>. / <c>op(A)</c> および <c>C</c> の行数。</param>
    /// <param name="n">The number of columns of <c>op(B)</c> and <c>C</c>. / <c>op(B)</c> および <c>C</c> の列数。</param>
    /// <param name="k">The number of columns of <c>op(A)</c> and rows of <c>op(B)</c>. / <c>op(A)</c> の列数かつ <c>op(B)</c> の行数。</param>
    /// <param name="alpha">The scalar multiplier α. / スカラー係数 α。</param>
    /// <param name="a">The column-major matrix <c>A</c> (<c>m × k</c>, or <c>k × m</c> when transposed). / 列優先の行列 <c>A</c>（転置なしは <c>m × k</c>、転置ありは <c>k × m</c>）。</param>
    /// <param name="lda">The leading dimension of <c>A</c> (at least the row count of <c>A</c>). / <c>A</c> のリーディングディメンション（<c>A</c> の行数以上）。</param>
    /// <param name="b">The column-major matrix <c>B</c> (<c>k × n</c>, or <c>n × k</c> when transposed). / 列優先の行列 <c>B</c>（転置なしは <c>k × n</c>、転置ありは <c>n × k</c>）。</param>
    /// <param name="ldb">The leading dimension of <c>B</c> (at least the row count of <c>B</c>). / <c>B</c> のリーディングディメンション（<c>B</c> の行数以上）。</param>
    /// <param name="beta">The scalar multiplier β applied to <c>C</c>. / <c>C</c> に掛けるスカラー係数 β。</param>
    /// <param name="c">The column-major output matrix <c>C</c> (<c>m × n</c>), overwritten in place; padding elements beyond <paramref name="m"/> in each column are untouched. / 列優先の出力行列 <c>C</c>（<c>m × n</c>）。インプレースで上書きされ、各列の <paramref name="m"/> 行目以降の余白は読み書きされません。</param>
    /// <param name="ldc">The leading dimension of <c>C</c> (at least <paramref name="m"/>). / <c>C</c> のリーディングディメンション（<paramref name="m"/> 以上）。</param>
    /// <remarks>
    /// <para><paramref name="c"/> must not share any elements with <paramref name="a"/> or <paramref name="b"/> (disjoint submatrices within the same backing array are allowed); <paramref name="a"/> and <paramref name="b"/> may overlap. When <paramref name="alpha"/> or <paramref name="k"/> is 0, <paramref name="a"/> and <paramref name="b"/> are not read. When <paramref name="beta"/> is 0, the initial contents of <paramref name="c"/> are not read. Results are bit-for-bit identical regardless of thread count.</para>
    /// <para><paramref name="c"/> は <paramref name="a"/> および <paramref name="b"/> と要素を共有してはなりません（同じ配列内の互いに素な部分行列であれば可）。<paramref name="a"/> と <paramref name="b"/> は重複可能です。<paramref name="alpha"/> または <paramref name="k"/> が 0 の場合は <paramref name="a"/>・<paramref name="b"/> を読み取らず、<paramref name="beta"/> が 0 の場合は <paramref name="c"/> の元の値を読み取りません。計算結果はスレッド数によらずビット単位で同一です。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">Matrix dimensions or leading dimensions are invalid, or <paramref name="c"/> aliases elements of <paramref name="a"/> or <paramref name="b"/>. / 行列のサイズやリーディングディメンションが不正、または <paramref name="c"/> が <paramref name="a"/> や <paramref name="b"/> と要素を共有している場合。</exception>
    public static void Gemm(
        BlasTranspose transA, BlasTranspose transB, int m, int n, int k,
        double alpha, ReadOnlySpan<double> a, int lda, ReadOnlySpan<double> b, int ldb, double beta, Span<double> c, int ldc)
    {
        bool ta = transA != BlasTranspose.None, tb = transB != BlasTranspose.None;
        BlasCheck.MatrixProduct(ta, tb, m, n, k, a, lda, b, ldb, c, ldc);
        ParallelLevel2<double>.ScaleOrClear(beta, m, n, c, ldc);

        // Qualify with namespace because the internal Gemm type shares this method name.
        LinearAlgebra.Gemm.Multiply(ta, tb, m, n, k, alpha, a, lda, b, ldb, c, ldc);
    }
}
