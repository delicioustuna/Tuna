namespace Tuna.LinearAlgebra;

/// <summary>
/// Represents the LU decomposition with partial pivoting, <c>PA = LU</c>, of a square matrix.
/// <para>正方行列の部分ピボット付き LU 分解 <c>PA = LU</c> を表します。</para>
/// </summary>
/// <remarks>
/// <para>
/// <c>L</c> is a unit lower triangular matrix, <c>U</c> is an upper triangular matrix, and <c>P</c> is a row permutation.
/// Even if the matrix is singular (an exact zero pivot is encountered), factorization runs to completion without throwing, sets <see cref="IsSingular"/> to <see langword="true"/>, and yields a determinant of 0. Calling <see cref="Solve(ReadOnlyVectorSpan{double})"/> or other solve/inverse methods on a singular decomposition throws <see cref="InvalidOperationException"/>.
/// Near-zero pivots (ill-conditioned matrices) are not flagged as singular; use <see cref="EstimateReciprocalConditionNumber"/> to check conditioning and <see cref="Refine(ReadOnlyMatrixSpan{double}, ReadOnlyVectorSpan{double}, VectorSpan{double}, int)"/> to improve solution accuracy when needed.
/// </para>
/// <para>
/// <c>L</c> は対角成分が 1 の下三角行列、<c>U</c> は上三角行列、<c>P</c> は行の置換です。
/// 行列が特異（ピボットが厳密に 0）であっても例外を投げずに最後まで分解を実行し、<see cref="IsSingular"/> を <see langword="true"/> に設定して行列式を 0 とします。特異な分解に対して <see cref="Solve(ReadOnlyVectorSpan{double})"/> などの求解・逆行列メソッドを呼び出すと <see cref="InvalidOperationException"/> がスローされます。
/// 0 に近いピボット（悪条件）は特異としては検出しません。条件の良し悪しは <see cref="EstimateReciprocalConditionNumber"/> で確認し、必要に応じて <see cref="Refine(ReadOnlyMatrixSpan{double}, ReadOnlyVectorSpan{double}, VectorSpan{double}, int)"/> で解の精度を改良してください。
/// </para>
/// </remarks>
/// <seealso href="https://www.netlib.org/lapack/explore-html/d3/d6a/group__getrf.html"/>
public sealed class LuDecomposition
{
    private readonly double[] factors;
    private readonly int[] pivots;
    private double norm;
    private Matrix<double>? l;
    private Matrix<double>? u;
    private int[]? permutation;

    private LuDecomposition(int order, double[] factors, int[] pivots, int info, double norm)
    {
        Order = order;
        this.factors = factors;
        this.pivots = pivots;
        this.norm = norm;
        IsSingular = info != 0;
    }

    /// <summary>
    /// Gets the order <c>n</c> of the factored matrix.
    /// <para>分解された行列の次数 <c>n</c> を取得します。</para>
    /// </summary>
    public int Order { get; }

    /// <summary>
    /// Gets the unit lower triangular factor <c>L</c> (<c>n × n</c>).
    /// <para>対角成分が 1 の下三角因子 <c>L</c>（<c>n × n</c>）を取得します。</para>
    /// </summary>
    /// <remarks>
    /// <para>
    /// Returns a copy materialized lazily from the factor storage on first access.
    /// After <see cref="Recompute"/>, subsequent accesses materialize a new matrix; previously obtained instances still hold the prior factorization.
    /// </para>
    /// <para>
    /// 初回アクセス時に内部の因子ストレージからコピーを生成して返します。
    /// <see cref="Recompute"/> 後のアクセスでは新しい行列を生成しますが、取得済みのインスタンスは再分解前の内容を保持します。
    /// </para>
    /// </remarks>
    public Matrix<double> L => l ??= FormL();

    /// <summary>
    /// Gets the upper triangular factor <c>U</c> (<c>n × n</c>).
    /// <para>上三角因子 <c>U</c>（<c>n × n</c>）を取得します。</para>
    /// </summary>
    /// <remarks>
    /// <para>
    /// Returns a copy materialized lazily from the factor storage on first access.
    /// After <see cref="Recompute"/>, subsequent accesses materialize a new matrix; previously obtained instances still hold the prior factorization.
    /// </para>
    /// <para>
    /// 初回アクセス時に内部の因子ストレージからコピーを生成して返します。
    /// <see cref="Recompute"/> 後のアクセスでは新しい行列を生成しますが、取得済みのインスタンスは再分解前の内容を保持します。
    /// </para>
    /// </remarks>
    public Matrix<double> U => u ??= FormU();

    /// <summary>
    /// Gets the row permutation <c>P</c> as a new index array.
    /// <para>行の置換 <c>P</c> を新しいインデックス配列として取得します。</para>
    /// </summary>
    /// <value>A newly allocated permutation array where row <c>i</c> of <c>PA</c> is row <c>Permutation[i]</c> of <c>A</c>. / <c>PA</c> の第 <c>i</c> 行が <c>A</c> の第 <c>Permutation[i]</c> 行に対応する新しい配列。</value>
    public int[] Permutation => (int[])(permutation ??= FormPermutation()).Clone();

    /// <summary>
    /// Gets a value indicating whether an exact zero pivot was encountered during factorization.
    /// <para>分解中に厳密に 0 のピボットが発生した（特異である）かどうかを示す値を取得します。</para>
    /// </summary>
    public bool IsSingular { get; private set; }

    /// <summary>
    /// Gets the determinant <c>det A = (det P)·Π uᵢᵢ</c>.
    /// <para>行列式 <c>det A = (det P)·Π uᵢᵢ</c> を取得します。</para>
    /// </summary>
    /// <remarks>
    /// <para>For large matrices, the determinant may overflow or underflow; prefer <see cref="LogAbsDeterminant"/> and <see cref="DeterminantSign"/>.</para>
    /// <para>大きな行列ではオーバーフローやアンダーフローが発生する可能性があるため、<see cref="LogAbsDeterminant"/> と <see cref="DeterminantSign"/> の使用を推奨します。</para>
    /// </remarks>
    public double Determinant
    {
        get
        {
            double det = DeterminantSign;
            int n = Order;
            for (int i = 0; i < n; i++)
            {
                det *= Math.Abs(factors[i + i * n]);
            }

            return det;
        }
    }

    /// <summary>
    /// Gets the natural logarithm of the absolute determinant, <c>ln |det A| = Σ ln |uᵢᵢ|</c>.
    /// <para>行列式の絶対値の自然対数 <c>ln |det A| = Σ ln |uᵢᵢ|</c> を取得します。</para>
    /// </summary>
    /// <value>The log absolute determinant, or <see cref="double.NegativeInfinity"/> if the matrix is singular. / 行列式の絶対値の自然対数。特異な場合は <see cref="double.NegativeInfinity"/>。</value>
    public double LogAbsDeterminant
    {
        get
        {
            double sum = 0;
            int n = Order;
            for (int i = 0; i < n; i++)
            {
                sum += Math.Log(Math.Abs(factors[i + i * n]));
            }

            return sum;
        }
    }

    /// <summary>
    /// Gets the sign of the determinant.
    /// <para>行列式の符号を取得します。</para>
    /// </summary>
    /// <value><c>1</c> or <c>-1</c>, or <c>0</c> if the matrix is singular. / <c>1</c> または <c>-1</c>。特異な場合は <c>0</c>。</value>
    public int DeterminantSign
    {
        get
        {
            if (IsSingular)
            {
                return 0;
            }

            int sign = 1;
            int n = Order;
            for (int i = 0; i < n; i++)
            {
                if (pivots[i] != i)
                {
                    sign = -sign;
                }

                if (factors[i + i * n] < 0)
                {
                    sign = -sign;
                }
            }

            return sign;
        }
    }

    /// <summary>
    /// Computes the LU decomposition with partial pivoting of the specified square matrix.
    /// <para>指定した正方行列の部分ピボット付き LU 分解を計算します。</para>
    /// </summary>
    /// <param name="a">The square matrix to factor. Not modified when <paramref name="overwriteInput"/> is <see langword="false"/>. / 分解対象の正方行列。<paramref name="overwriteInput"/> が <see langword="false"/> の場合は変更されません。</param>
    /// <param name="overwriteInput"><see langword="true"/> to store the factors in <paramref name="a"/>'s backing array in place without allocating another matrix (destroying its contents); <see langword="false"/> to copy <paramref name="a"/> first. When <see langword="true"/>, do not modify <paramref name="a"/> while the decomposition is in use. / <see langword="true"/> の場合、新たな行列を確保せず <paramref name="a"/> の内部配列に因子をインプレースで格納します（元の内容は破壊されます）。<see langword="true"/> を指定した場合、分解の使用中は <paramref name="a"/> を変更しないでください。</param>
    /// <returns>The LU decomposition (returned even if the matrix is singular). / LU 分解の結果（特異な場合も返されます）。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="a"/> is <see langword="null"/>. / <paramref name="a"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="a"/> is not square. / <paramref name="a"/> が正方行列でない場合。</exception>
    public static LuDecomposition Compute(Matrix<double> a, bool overwriteInput = false)
    {
        ArgumentNullException.ThrowIfNull(a);
        if (!a.IsSquare)
        {
            throw new ArgumentException("The matrix must be square.", nameof(a));
        }

        int n = a.Rows;
        var factors = overwriteInput ? a.Data : (double[])a.Data.Clone();
        var pivots = new int[n];
        double norm = ConditionKernels.Norm1(n, factors, n);
        int info = LuKernels.Factor(n, n, factors, n, pivots);
        return new LuDecomposition(n, factors, pivots, info, norm);
    }

    /// <summary>
    /// Computes the LU decomposition with partial pivoting of the specified square matrix view.
    /// <para>指定した正方行列ビューの部分ピボット付き LU 分解を計算します。</para>
    /// </summary>
    /// <param name="a">The square matrix view to factor (not modified). / 分解対象の正方行列ビュー（変更されません）。</param>
    /// <returns>The LU decomposition (returned even if the matrix is singular). / LU 分解の結果（特異な場合も返されます）。</returns>
    /// <exception cref="ArgumentException"><paramref name="a"/> is not square. / <paramref name="a"/> が正方行列でない場合。</exception>
    public static LuDecomposition Compute(ReadOnlyMatrixSpan<double> a)
    {
        if (a.Rows != a.Columns)
        {
            throw new ArgumentException("The matrix must be square.", nameof(a));
        }

        int n = a.Rows;
        var factors = GC.AllocateUninitializedArray<double>(checked(n * n));
        StridedKernels.CopyIntoFactorStorage(a, factors);
        var pivots = new int[n];
        double norm = ConditionKernels.Norm1(n, factors, n);
        int info = LuKernels.Factor(n, n, factors, n, pivots);
        return new LuDecomposition(n, factors, pivots, info, norm);
    }

    /// <summary>
    /// Recomputes the LU decomposition for another square matrix <paramref name="a"/> of the same order, overwriting the existing factor storage in place without allocating.
    /// <para>同じ次数の別の正方行列 <paramref name="a"/> に対して LU 分解を再計算し、既存の因子ストレージを新規アロケーションなしでインプレース更新します。</para>
    /// </summary>
    /// <param name="a">A square matrix or matrix view of the same order (not modified unless it aliases the decomposition's factor storage from <c>overwriteInput</c>, in which case it is factored in place without copying). / 同じ次数の正方行列または行列ビュー。<c>overwriteInput</c> で渡した因子ストレージ自身を指す場合はコピーせずインプレースで分解します。</param>
    /// <remarks>
    /// <para>
    /// Reuses the decomposition instance, the <c>n × n</c> factor buffer, and the pivot array across repeated factorizations of the same order (such as optimization iterations or tape replay).
    /// Singular matrices are factored normally and update <see cref="IsSingular"/>.
    /// Previously obtained <see cref="L"/>, <see cref="U"/>, and <see cref="Permutation"/> instances remain unchanged; subsequent property accesses materialize fresh copies.
    /// If the decomposition was created with <c>overwriteInput</c>, the original input matrix's backing array is overwritten.
    /// Must not be called concurrently with any other operation on the same decomposition.
    /// </para>
    /// <para>
    /// 最適化の反復や自動微分の再実行など、同じ次数の行列を繰り返し分解する用途で分解オブジェクト、<c>n × n</c> の因子バッファ、およびピボット配列を再利用します。
    /// 特異な行列でも分解を実行し、<see cref="IsSingular"/> を新しい結果に合わせて更新します。
    /// 取得済みの <see cref="L"/>、<see cref="U"/>、<see cref="Permutation"/> はコピーのため再分解前の内容を保持し、次回アクセス時に新しく生成されます。
    /// <c>overwriteInput</c> を指定して作成した分解では、元の入力行列の内部配列が上書きされます。
    /// 同じ分解インスタンスに対する他の呼び出しと同時に実行しないでください。
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="a"/> is not square, does not have order <see cref="Order"/>, or partially overlaps the factor storage. / <paramref name="a"/> が正方行列でない、次数が <see cref="Order"/> と一致しない、または因子ストレージと部分的に重なる場合。</exception>
    public void Recompute(ReadOnlyMatrixSpan<double> a)
    {
        int n = Order;
        if (a.Rows != n || a.Columns != n)
        {
            throw new ArgumentException("The matrix must be square with the same order as the decomposition.", nameof(a));
        }

        StridedKernels.CopyIntoFactorStorage(a, factors);
        norm = ConditionKernels.Norm1(n, factors, n);
        IsSingular = LuKernels.Factor(n, n, factors, n, pivots) != 0;
        l = null;
        u = null;
        permutation = null;
    }

    /// <summary>
    /// Estimates the reciprocal 1-norm condition number <c>1 / (‖A‖₁‖A⁻¹‖₁)</c> in <c>O(n²)</c> time.
    /// <para>1-ノルム条件数の逆数 <c>1 / (‖A‖₁‖A⁻¹‖₁)</c> を <c>O(n²)</c> で推定します。</para>
    /// </summary>
    /// <returns>The estimated reciprocal 1-norm condition number. / 1-ノルム条件数の逆数の推定値。</returns>
    /// <remarks>
    /// <para>
    /// Values closer to 0 indicate worse conditioning; relative solution error can grow up to roughly <c>ε / rcond</c> (<c>ε ≈ 1.1×10⁻¹⁶</c>).
    /// Because <c>‖A⁻¹‖₁</c> is estimated from below using the Hager–Higham 1-norm estimator, the returned value is an upper bound on the true reciprocal condition number and is typically within a factor of 3.
    /// Returns <c>0</c> if <see cref="IsSingular"/> is <see langword="true"/>, <c>1</c> if <see cref="Order"/> is 0, and <see cref="double.NaN"/> if <c>A</c> contains <see cref="double.NaN"/>.
    /// </para>
    /// <para>
    /// 0 に近いほど悪条件であり、解の相対誤差はおおよそ <c>ε / rcond</c>（<c>ε ≈ 1.1×10⁻¹⁶</c>）程度まで拡大し得ます。
    /// Hager–Higham の 1-ノルム推定法により <c>‖A⁻¹‖₁</c> を下から推定するため、戻り値は真の条件数逆数以上となり、通常は真値の 3 倍以内に収まります。
    /// <see cref="IsSingular"/> が <see langword="true"/> の場合は <c>0</c>、<see cref="Order"/> が 0 の場合は <c>1</c>、<c>A</c> が <see cref="double.NaN"/> を含む場合は <see cref="double.NaN"/> を返します。
    /// </para>
    /// </remarks>
    public double EstimateReciprocalConditionNumber()
    {
        if (IsSingular)
        {
            return 0;
        }

        int n = Order;
        return ConditionKernels.ReciprocalCondition(n, norm, (x, transpose) =>
        {
            // (PA)⁻¹ = U⁻¹L⁻¹ is a column permutation of A⁻¹, so its 1-norm is identical (no permutation needed, matching LAPACK dgecon).
            if (!transpose)
            {
                ParallelLevel2<double>.Trsv(BlasTriangle.Lower, BlasTranspose.None, BlasDiagonal.Unit, n, factors, n, x);
                ParallelLevel2<double>.Trsv(BlasTriangle.Upper, BlasTranspose.None, BlasDiagonal.NonUnit, n, factors, n, x);
            }
            else
            {
                ParallelLevel2<double>.Trsv(BlasTriangle.Upper, BlasTranspose.Transpose, BlasDiagonal.NonUnit, n, factors, n, x);
                ParallelLevel2<double>.Trsv(BlasTriangle.Lower, BlasTranspose.Transpose, BlasDiagonal.Unit, n, factors, n, x);
            }
        });
    }

    /// <summary>
    /// Solves the linear system <c>Ax = b</c>.
    /// <para>連立一次方程式 <c>Ax = b</c> を解きます。</para>
    /// </summary>
    /// <param name="b">The right-hand side vector. / 右辺ベクトル。</param>
    /// <returns>A newly allocated vector containing the solution <c>x</c>. / 解 <c>x</c> を格納した新しいベクトル。</returns>
    /// <exception cref="ArgumentException">The length of <paramref name="b"/> does not equal <see cref="Order"/>. / <paramref name="b"/> の長さが <see cref="Order"/> と一致しない場合。</exception>
    /// <exception cref="InvalidOperationException"><see cref="IsSingular"/> is <see langword="true"/>. / <see cref="IsSingular"/> が <see langword="true"/> の場合。</exception>
    public Vector<double> Solve(ReadOnlyVectorSpan<double> b)
    {
        var x = Vector<double>.Uninitialized(b.Length);
        Solve(b, x);
        return x;
    }

    /// <summary>
    /// Solves the linear system <c>Ax = b</c> and writes the solution to <paramref name="destination"/>.
    /// <para>連立一次方程式 <c>Ax = b</c> を解き、解を <paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="b">The right-hand side vector. / 右辺ベクトル。</param>
    /// <param name="destination">The destination vector. May identically alias <paramref name="b"/> for in-place solving, but must not partially overlap <paramref name="b"/> or overlap the factor storage. / 出力先ベクトル。<paramref name="b"/> と完全に一致する領域ならインプレースで解けますが、部分的な重なりや因子ストレージとの重なりは不可です。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException">The length does not match <see cref="Order"/>, or <paramref name="destination"/> overlaps invalidly. / 長さが <see cref="Order"/> と一致しない、または出力先が不正に重なる場合。</exception>
    /// <exception cref="InvalidOperationException"><see cref="IsSingular"/> is <see langword="true"/>. / <see cref="IsSingular"/> が <see langword="true"/> の場合。</exception>
    public VectorSpan<double> Solve(ReadOnlyVectorSpan<double> b, VectorSpan<double> destination)
    {
        Solve(b.AsMatrix(), destination.AsMatrix(), vector: true);
        return destination;
    }

    /// <summary>
    /// Solves the linear system <c>AX = B</c> with multiple right-hand sides.
    /// <para>複数右辺の連立一次方程式 <c>AX = B</c> を解きます。</para>
    /// </summary>
    /// <param name="b">The right-hand side matrix. / 右辺行列。</param>
    /// <returns>A newly allocated matrix containing the solutions <c>X</c>. / 解 <c>X</c> を格納した新しい行列。</returns>
    /// <exception cref="ArgumentException">The number of rows in <paramref name="b"/> does not equal <see cref="Order"/>. / <paramref name="b"/> の行数が <see cref="Order"/> と一致しない場合。</exception>
    /// <exception cref="InvalidOperationException"><see cref="IsSingular"/> is <see langword="true"/>. / <see cref="IsSingular"/> が <see langword="true"/> の場合。</exception>
    public Matrix<double> Solve(ReadOnlyMatrixSpan<double> b)
    {
        var x = Matrix<double>.Uninitialized(b.Rows, b.Columns);
        Solve(b, x);
        return x;
    }

    /// <summary>
    /// Solves the linear system <c>AX = B</c> with multiple right-hand sides and writes the solutions to <paramref name="destination"/>.
    /// <para>複数右辺の連立一次方程式 <c>AX = B</c> を解き、解を <paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="b">The right-hand side matrix. / 右辺行列。</param>
    /// <param name="destination">The destination matrix. May identically alias <paramref name="b"/> for in-place solving, but must not partially overlap <paramref name="b"/> or overlap the factor storage. / 出力先行列。<paramref name="b"/> と完全に一致する領域ならインプレースで解けますが、部分的な重なりや因子ストレージとの重なりは不可です。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException">The shapes do not match, or <paramref name="destination"/> overlaps invalidly. / 形状が一致しない、または出力先が不正に重なる場合。</exception>
    /// <exception cref="InvalidOperationException"><see cref="IsSingular"/> is <see langword="true"/>. / <see cref="IsSingular"/> が <see langword="true"/> の場合。</exception>
    public MatrixSpan<double> Solve(ReadOnlyMatrixSpan<double> b, MatrixSpan<double> destination)
    {
        Solve(b, destination, vector: false);
        return destination;
    }

    /// <summary>
    /// Solves the transposed linear system <c>Aᵀx = b</c> using the existing factors.
    /// <para>既存の因子を用いて転置連立一次方程式 <c>Aᵀx = b</c> を解きます。</para>
    /// </summary>
    /// <param name="b">The right-hand side vector. / 右辺ベクトル。</param>
    /// <returns>A newly allocated vector containing the solution <c>x</c>. / 解 <c>x</c> を格納した新しいベクトル。</returns>
    /// <exception cref="ArgumentException">The length of <paramref name="b"/> does not equal <see cref="Order"/>. / <paramref name="b"/> の長さが <see cref="Order"/> と一致しない場合。</exception>
    /// <exception cref="InvalidOperationException"><see cref="IsSingular"/> is <see langword="true"/>. / <see cref="IsSingular"/> が <see langword="true"/> の場合。</exception>
    public Vector<double> SolveTransposed(ReadOnlyVectorSpan<double> b)
    {
        var x = Vector<double>.Uninitialized(b.Length);
        SolveTransposed(b, x);
        return x;
    }

    /// <summary>
    /// Solves the transposed linear system <c>Aᵀx = b</c> and writes the solution to <paramref name="destination"/>.
    /// <para>転置連立一次方程式 <c>Aᵀx = b</c> を解き、解を <paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="b">The right-hand side vector. / 右辺ベクトル。</param>
    /// <param name="destination">The destination vector. May identically alias <paramref name="b"/> for in-place solving, but must not partially overlap <paramref name="b"/> or overlap the factor storage. / 出力先ベクトル。<paramref name="b"/> と完全に一致する領域ならインプレースで解けますが、部分的な重なりや因子ストレージとの重なりは不可です。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException">The length does not match <see cref="Order"/>, or <paramref name="destination"/> overlaps invalidly. / 長さが <see cref="Order"/> と一致しない、または出力先が不正に重なる場合。</exception>
    /// <exception cref="InvalidOperationException"><see cref="IsSingular"/> is <see langword="true"/>. / <see cref="IsSingular"/> が <see langword="true"/> の場合。</exception>
    public VectorSpan<double> SolveTransposed(ReadOnlyVectorSpan<double> b, VectorSpan<double> destination)
    {
        Solve(b.AsMatrix(), destination.AsMatrix(), vector: true, transpose: true);
        return destination;
    }

    /// <summary>
    /// Solves the transposed linear system <c>AᵀX = B</c> with multiple right-hand sides.
    /// <para>複数右辺の転置連立一次方程式 <c>AᵀX = B</c> を解きます。</para>
    /// </summary>
    /// <param name="b">The right-hand side matrix. / 右辺行列。</param>
    /// <returns>A newly allocated matrix containing the solutions <c>X</c>. / 解 <c>X</c> を格納した新しい行列。</returns>
    /// <exception cref="ArgumentException">The number of rows in <paramref name="b"/> does not equal <see cref="Order"/>. / <paramref name="b"/> の行数が <see cref="Order"/> と一致しない場合。</exception>
    /// <exception cref="InvalidOperationException"><see cref="IsSingular"/> is <see langword="true"/>. / <see cref="IsSingular"/> が <see langword="true"/> の場合。</exception>
    public Matrix<double> SolveTransposed(ReadOnlyMatrixSpan<double> b)
    {
        var x = Matrix<double>.Uninitialized(b.Rows, b.Columns);
        SolveTransposed(b, x);
        return x;
    }

    /// <summary>
    /// Solves the transposed linear system <c>AᵀX = B</c> with multiple right-hand sides and writes the solutions to <paramref name="destination"/>.
    /// <para>複数右辺の転置連立一次方程式 <c>AᵀX = B</c> を解き、解を <paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="b">The right-hand side matrix. / 右辺行列。</param>
    /// <param name="destination">The destination matrix. May identically alias <paramref name="b"/> for in-place solving, but must not partially overlap <paramref name="b"/> or overlap the factor storage. / 出力先行列。<paramref name="b"/> と完全に一致する領域ならインプレースで解けますが、部分的な重なりや因子ストレージとの重なりは不可です。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException">The shapes do not match, or <paramref name="destination"/> overlaps invalidly. / 形状が一致しない、または出力先が不正に重なる場合。</exception>
    /// <exception cref="InvalidOperationException"><see cref="IsSingular"/> is <see langword="true"/>. / <see cref="IsSingular"/> が <see langword="true"/> の場合。</exception>
    public MatrixSpan<double> SolveTransposed(ReadOnlyMatrixSpan<double> b, MatrixSpan<double> destination)
    {
        Solve(b, destination, vector: false, transpose: true);
        return destination;
    }

    // LAPACK dgerfs equivalent; computes residual b - Ax in extended precision.
    /// <summary>
    /// Improves the solution <paramref name="x"/> of <c>Ax = b</c> in place using iterative refinement.
    /// <para>反復改良を用いて連立一次方程式 <c>Ax = b</c> の解 <paramref name="x"/> をインプレースで高精度化します。</para>
    /// </summary>
    /// <param name="a">The original square matrix (<c>n × n</c>). / 分解前の元の正方行列（<c>n × n</c>）。</param>
    /// <param name="b">The right-hand side vector. / 右辺ベクトル。</param>
    /// <param name="x">The initial solution vector, overwritten in place with the refined solution; must not overlap <paramref name="a"/> or <paramref name="b"/>. / 初期解ベクトル。改良後の解でインプレース上書きされます（<paramref name="a"/>・<paramref name="b"/> と重なってはなりません）。</param>
    /// <param name="maxIterations">The maximum number of refinement iterations (0 computes only the backward error without refining). / 反復改良の最大回数（0 の場合は改良を行わず後退誤差のみを返します）。</param>
    /// <returns>The componentwise backward error <c>maxᵢ |b - Ax|ᵢ / (|A||x| + |b|)ᵢ</c>. / 成分ごとの後退誤差 <c>maxᵢ |b - Ax|ᵢ / (|A||x| + |b|)ᵢ</c>。</returns>
    /// <remarks>
    /// <para>Intended for ill-conditioned matrices when extra accuracy beyond <see cref="Solve(ReadOnlyVectorSpan{double})"/> is needed (<c>O(n²)</c> per iteration). Requires preserving the original matrix <paramref name="a"/>.</para>
    /// <para>悪条件の行列で <see cref="Solve(ReadOnlyVectorSpan{double})"/> 以上の精度が必要な場合に使用します（1 回あたり <c>O(n²)</c>）。元の行列 <paramref name="a"/> が必要です。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">The shapes do not match, or <paramref name="x"/> overlaps <paramref name="a"/>, <paramref name="b"/>, or the factor storage. / 形状が一致しない、または <paramref name="x"/> が <paramref name="a"/>・<paramref name="b"/>・因子ストレージと重なる場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxIterations"/> is negative. / <paramref name="maxIterations"/> が負の場合。</exception>
    /// <exception cref="InvalidOperationException"><see cref="IsSingular"/> is <see langword="true"/>. / <see cref="IsSingular"/> が <see langword="true"/> の場合。</exception>
    public double Refine(
        ReadOnlyMatrixSpan<double> a, ReadOnlyVectorSpan<double> b, VectorSpan<double> x, int maxIterations = ConditionKernels.DefaultRefinementIterations) =>
        Refine(a, b.AsMatrix(), x.AsMatrix(), maxIterations);

    /// <summary>
    /// Improves the solutions <paramref name="x"/> of <c>AX = B</c> column by column in place using iterative refinement.
    /// <para>反復改良を用いて複数右辺の連立一次方程式 <c>AX = B</c> の解 <paramref name="x"/> を列ごとにインプレースで高精度化します。</para>
    /// </summary>
    /// <param name="a">The original square matrix (<c>n × n</c>). / 分解前の元の正方行列（<c>n × n</c>）。</param>
    /// <param name="b">The right-hand side matrix. / 右辺行列。</param>
    /// <param name="x">The initial solution matrix, overwritten in place with the refined solutions; must not overlap <paramref name="a"/> or <paramref name="b"/>. / 初期解行列。改良後の解でインプレース上書きされます（<paramref name="a"/>・<paramref name="b"/> と重なってはなりません）。</param>
    /// <param name="maxIterations">The maximum number of refinement iterations per column (0 computes only the backward error). / 各列の反復改良の最大回数（0 の場合は後退誤差のみを返します）。</param>
    /// <returns>The maximum componentwise backward error across all columns. / 全列における成分ごとの後退誤差の最大値。</returns>
    /// <exception cref="ArgumentException">The shapes do not match, or <paramref name="x"/> overlaps <paramref name="a"/>, <paramref name="b"/>, or the factor storage. / 形状が一致しない、または <paramref name="x"/> が <paramref name="a"/>・<paramref name="b"/>・因子ストレージと重なる場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxIterations"/> is negative. / <paramref name="maxIterations"/> が負の場合。</exception>
    /// <exception cref="InvalidOperationException"><see cref="IsSingular"/> is <see langword="true"/>. / <see cref="IsSingular"/> が <see langword="true"/> の場合。</exception>
    public double Refine(
        ReadOnlyMatrixSpan<double> a, ReadOnlyMatrixSpan<double> b, MatrixSpan<double> x, int maxIterations = ConditionKernels.DefaultRefinementIterations)
    {
        int n = Order;
        StridedKernels.RequireRefinementArguments(n, a, b, x, maxIterations, new MatrixSpan<double>(factors, n, n));
        if (IsSingular)
        {
            throw new InvalidOperationException("The matrix is singular.");
        }

        return StridedKernels.Refine(a, symmetricLower: false, b, x, maxIterations, (r, _) =>
            LuKernels.Solve(n, factors, n, pivots, 1, r, n, vector: true));
    }

    /// <summary>
    /// Computes the inverse matrix <c>A⁻¹</c>.
    /// <para>逆行列 <c>A⁻¹</c> を計算します。</para>
    /// </summary>
    /// <returns>A newly allocated matrix containing <c>A⁻¹</c>. / <c>A⁻¹</c> を格納した新しい行列。</returns>
    /// <remarks>
    /// <para>When solving linear systems, prefer <see cref="Solve(ReadOnlyMatrixSpan{double})"/> over forming the inverse explicitly.</para>
    /// <para>連立一次方程式を解くだけであれば、明示的に逆行列を求めるよりも <see cref="Solve(ReadOnlyMatrixSpan{double})"/> の方が高速かつ高精度です。</para>
    /// </remarks>
    /// <exception cref="InvalidOperationException"><see cref="IsSingular"/> is <see langword="true"/>. / <see cref="IsSingular"/> が <see langword="true"/> の場合。</exception>
    public Matrix<double> Inverse()
    {
        var x = Matrix<double>.Identity(Order);
        Solve(x, x);
        return x;
    }

    /// <summary>
    /// Computes the inverse matrix <c>A⁻¹</c> and writes it to <paramref name="destination"/>.
    /// <para>逆行列 <c>A⁻¹</c> を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="destination">The <c>n × n</c> destination matrix, which must not overlap the factor storage. / <c>n × n</c> の出力先行列（因子ストレージと重なってはなりません）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException">The shape does not match, or <paramref name="destination"/> overlaps the factor storage. / 形状が一致しない、または <paramref name="destination"/> が因子ストレージと重なる場合。</exception>
    /// <exception cref="InvalidOperationException"><see cref="IsSingular"/> is <see langword="true"/>. / <see cref="IsSingular"/> が <see langword="true"/> の場合。</exception>
    public MatrixSpan<double> Inverse(MatrixSpan<double> destination)
    {
        int n = Order;
        if (destination.Rows != n || destination.Columns != n)
        {
            throw new ArgumentException("Destination shape must agree.", nameof(destination));
        }

        StridedKernels.RequireDisjoint(new ReadOnlyMatrixSpan<double>(factors, n, n), destination);

        if (IsSingular)
        {
            throw new InvalidOperationException("The matrix is singular.");
        }

        destination.Fill(0);
        destination.Diagonal().Fill(1);
        StridedKernels.SolveFactoredInPlace(n, factors, pivots, destination, vector: false);
        return destination;
    }

    private void Solve(ReadOnlyMatrixSpan<double> b, MatrixSpan<double> x, bool vector, bool transpose = false)
    {
        int n = Order;
        if (b.Rows != n)
        {
            throw new ArgumentException("The right-hand side must have as many rows as the order of the matrix.", nameof(b));
        }

        if (x.Rows != n || x.Columns != b.Columns)
        {
            throw new ArgumentException("Destination shape must agree.", "destination");
        }

        // overwriteInput で分解した場合、因子の領域は利用者の行列なので、出力先に渡されうる。右辺を写す前に拒む
        StridedKernels.RequireDisjoint(new ReadOnlyMatrixSpan<double>(factors, n, n), x);
        if (IsSingular)
        {
            throw new InvalidOperationException("The matrix is singular.");
        }

        StridedKernels.CopyRightHandSide(b, x);
        StridedKernels.SolveFactoredInPlace(n, factors, pivots, x, vector, transpose);
    }

    private Matrix<double> FormL()
    {
        int n = Order;
        var result = new Matrix<double>(n, n);
        var data = result.Data;
        for (int j = 0; j < n; j++)
        {
            data[j + j * n] = 1;
            factors.AsSpan(j + 1 + j * n, n - j - 1).CopyTo(data.AsSpan(j + 1 + j * n));
        }

        return result;
    }

    private Matrix<double> FormU()
    {
        int n = Order;
        var result = new Matrix<double>(n, n);
        var data = result.Data;
        for (int j = 0; j < n; j++)
        {
            factors.AsSpan(j * n, j + 1).CopyTo(data.AsSpan(j * n));
        }

        return result;
    }

    private int[] FormPermutation()
    {
        int n = Order;
        var result = new int[n];
        for (int i = 0; i < n; i++)
        {
            result[i] = i;
        }

        for (int i = 0; i < n; i++)
        {
            int p = pivots[i];
            if (p != i)
            {
                (result[i], result[p]) = (result[p], result[i]);
            }
        }

        return result;
    }
}
