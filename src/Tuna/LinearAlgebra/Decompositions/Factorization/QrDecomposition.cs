namespace Tuna.LinearAlgebra;

/// <summary>
/// Represents the thin QR decomposition <c>A = QR</c> of a real matrix.
/// <para>実行列の薄い QR 分解 <c>A = QR</c> を表します。</para>
/// </summary>
/// <remarks>
/// <para>For an <c>m × n</c> matrix <c>A</c> with <c>k = min(m, n)</c>, <see cref="Q"/> is an <c>m × k</c> matrix with orthonormal columns and <see cref="R"/> is a <c>k × n</c> upper trapezoidal matrix. Diagonal entries of <see cref="R"/> are not guaranteed to be positive.</para>
/// <para><c>m × n</c> 行列 <c>A</c> に対し <c>k = min(m, n)</c> として、<see cref="Q"/> は正規直交列を持つ <c>m × k</c> 行列、<see cref="R"/> は <c>k × n</c> の上台形行列になります（薄い QR 分解）。<see cref="R"/> の対角成分は正とは限りません。</para>
/// </remarks>
/// <seealso href="https://www.netlib.org/lapack/explore-html/d0/da1/group__geqrf.html"/>
public sealed class QrDecomposition
{
    private readonly double[] factors;
    private readonly double[] tau;
    private Matrix<double>? q;
    private Matrix<double>? r;

    private QrDecomposition(int rows, int columns, double[] factors, double[] tau)
    {
        Rows = rows;
        Columns = columns;
        this.factors = factors;
        this.tau = tau;
    }

    /// <summary>
    /// Gets the number of rows <c>m</c> of the original matrix.
    /// <para>元の行列の行数 <c>m</c> を取得します。</para>
    /// </summary>
    public int Rows { get; }

    /// <summary>
    /// Gets the number of columns <c>n</c> of the original matrix.
    /// <para>元の行列の列数 <c>n</c> を取得します。</para>
    /// </summary>
    public int Columns { get; }

    /// <summary>
    /// Gets the <c>m × min(m, n)</c> matrix <c>Q</c> with orthonormal columns.
    /// <para>正規直交列を持つ <c>m × min(m, n)</c> 行列 <c>Q</c> を取得します。</para>
    /// </summary>
    /// <remarks>
    /// <para>When created via <see cref="ComputeFactors(Matrix{double}, bool)"/>, <see cref="Q"/> is materialized lazily on first access. Once materialized, subsequent calls to <see cref="Recompute"/> overwrite the same matrix instance in place.</para>
    /// <para><see cref="ComputeFactors(Matrix{double}, bool)"/> で作成した場合、初回アクセス時に生成されます。一度生成された後は、<see cref="Recompute"/> を呼ぶと同じ行列インスタンスが新しい分解結果で上書きされます。</para>
    /// </remarks>
    public Matrix<double> Q => q ??= FormQ();

    /// <summary>
    /// Gets the <c>min(m, n) × n</c> upper trapezoidal factor <c>R</c>.
    /// <para><c>min(m, n) × n</c> の上台形行列 <c>R</c> を取得します。</para>
    /// </summary>
    /// <remarks>
    /// <para>When created via <see cref="ComputeFactors(Matrix{double}, bool)"/>, <see cref="R"/> is materialized lazily on first access. Once materialized, subsequent calls to <see cref="Recompute"/> overwrite the same matrix instance in place.</para>
    /// <para><see cref="ComputeFactors(Matrix{double}, bool)"/> で作成した場合、初回アクセス時に生成されます。一度生成された後は、<see cref="Recompute"/> を呼ぶと同じ行列インスタンスが新しい分解結果で上書きされます。</para>
    /// </remarks>
    public Matrix<double> R => r ??= FormR();

    /// <summary>
    /// Computes the QR decomposition of <paramref name="a"/> and immediately materializes <see cref="Q"/> and <see cref="R"/>.
    /// <para><paramref name="a"/> の QR 分解を計算し、<see cref="Q"/> および <see cref="R"/> を直ちに生成します。</para>
    /// </summary>
    /// <param name="a">The input matrix (unmodified when <paramref name="overwriteInput"/> is <see langword="false"/>). / 入力行列（<paramref name="overwriteInput"/> が <see langword="false"/> の場合は変更されません）。</param>
    /// <param name="overwriteInput"><see langword="true"/> to reuse <paramref name="a"/>'s backing array as factor storage (destroying its contents); <see langword="false"/> to copy <paramref name="a"/>. / <paramref name="a"/> の格納領域を因子領域として再利用する（元の内容は破壊されます）場合は <see langword="true"/>、コピーする場合は <see langword="false"/>。</param>
    /// <returns>The computed QR decomposition. / QR 分解結果。</returns>
    public static QrDecomposition Compute(Matrix<double> a, bool overwriteInput = false)
    {
        var result = ComputeFactors(a, overwriteInput);
        result.q = result.FormQ();
        result.r = result.FormR();
        return result;
    }

    /// <summary>
    /// Computes the QR decomposition of the matrix view <paramref name="a"/> and immediately materializes <see cref="Q"/> and <see cref="R"/>.
    /// <para>行列ビュー <paramref name="a"/> の QR 分解を計算し、<see cref="Q"/> および <see cref="R"/> を直ちに生成します。</para>
    /// </summary>
    /// <param name="a">The input matrix view (not modified). / 入力行列ビュー（変更されません）。</param>
    /// <returns>The computed QR decomposition. / QR 分解結果。</returns>
    public static QrDecomposition Compute(ReadOnlyMatrixSpan<double> a)
    {
        var result = ComputeFactors(a);
        result.q = result.FormQ();
        result.r = result.FormR();
        return result;
    }

    /// <summary>
    /// Computes only the compact Householder factorization, deferring materialization of <see cref="Q"/> and <see cref="R"/> until accessed.
    /// <para>コンパクトなハウスホルダー因子のみを計算し、<see cref="Q"/> と <see cref="R"/> の生成を初回アクセス時まで遅延します。</para>
    /// </summary>
    /// <param name="a">The input matrix (unmodified when <paramref name="overwriteInput"/> is <see langword="false"/>). / 入力行列（<paramref name="overwriteInput"/> が <see langword="false"/> の場合は変更されません）。</param>
    /// <param name="overwriteInput"><see langword="true"/> to reuse <paramref name="a"/>'s backing array as factor storage (do not modify <paramref name="a"/> while this decomposition is in use); <see langword="false"/> to copy <paramref name="a"/>. / <paramref name="a"/> の格納領域を因子領域として再利用する（この分解を使用している間は <paramref name="a"/> を変更しないでください）場合は <see langword="true"/>。</param>
    /// <returns>The computed QR decomposition. / QR 分解結果。</returns>
    public static QrDecomposition ComputeFactors(Matrix<double> a, bool overwriteInput = false)
    {
        int m = a.Rows;
        int n = a.Columns;
        var result = new QrDecomposition(m, n, overwriteInput ? a.Data : (double[])a.Data.Clone(), new double[Math.Min(m, n)]);
        result.Factor();
        return result;
    }

    /// <summary>
    /// Computes only the compact Householder factorization of the matrix view <paramref name="a"/>, deferring materialization of <see cref="Q"/> and <see cref="R"/> until accessed.
    /// <para>行列ビュー <paramref name="a"/> のコンパクトなハウスホルダー因子のみを計算し、<see cref="Q"/> と <see cref="R"/> の生成を初回アクセス時まで遅延します。</para>
    /// </summary>
    /// <param name="a">The input matrix view (not modified). / 入力行列ビュー（変更されません）。</param>
    /// <returns>The computed QR decomposition. / QR 分解結果。</returns>
    public static QrDecomposition ComputeFactors(ReadOnlyMatrixSpan<double> a)
    {
        int m = a.Rows;
        int n = a.Columns;
        var factors = GC.AllocateUninitializedArray<double>(checked(m * n));
        StridedKernels.CopyIntoFactorStorage(a, factors);
        var result = new QrDecomposition(m, n, factors, new double[Math.Min(m, n)]);
        result.Factor();
        return result;
    }

    /// <summary>
    /// Recomputes the QR decomposition in place for another matrix <paramref name="a"/> of the same dimensions, reusing existing buffers without heap allocation.
    /// <para>同じ形状の別の行列 <paramref name="a"/> を再分解し、既存の領域とオブジェクトを再利用してインプレースで更新します（追加のヒープアロケーションなし）。</para>
    /// </summary>
    /// <param name="a">A matrix or view with the same <see cref="Rows"/> and <see cref="Columns"/>. May alias the factor storage (when created with <c>overwriteInput</c>) or already-materialized <see cref="Q"/>/<see cref="R"/>; otherwise unmodified. / <see cref="Rows"/>・<see cref="Columns"/> が一致する行列またはビュー。<c>overwriteInput</c> の因子領域や生成済みの <see cref="Q"/>・<see cref="R"/> と同一領域でも構いません（それ以外の場合は変更されません）。</param>
    /// <remarks>
    /// <para>Produces bit-for-bit identical results to <see cref="Compute(Matrix{double}, bool)"/> / <see cref="ComputeFactors(Matrix{double}, bool)"/>. If <see cref="Q"/> or <see cref="R"/> has already been materialized, those existing <see cref="Matrix{TElement}"/> instances are overwritten with the new factors; otherwise they remain lazy. Do not call concurrently with other operations on the same instance.</para>
    /// <para><see cref="Compute(Matrix{double}, bool)"/> および <see cref="ComputeFactors(Matrix{double}, bool)"/> とビット単位で同じ結果になります。すでに <see cref="Q"/> や <see cref="R"/> が生成されている場合は既存の行列インスタンスが新しい値で上書きされます（未生成の場合は遅延生成のまま維持されます）。同一インスタンスに対する他の操作と同時に呼び出さないでください。</para>
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="a"/> does not match <see cref="Rows"/> and <see cref="Columns"/>, or partially overlaps the internal factor storage. / <paramref name="a"/> の形状が一致しない、または内部の因子領域と部分的に重なっている場合。</exception>
    public void Recompute(ReadOnlyMatrixSpan<double> a)
    {
        if (a.Rows != Rows || a.Columns != Columns)
        {
            throw new ArgumentException("The matrix must have the same numbers of rows and columns as the decomposition.", nameof(a));
        }

        StridedKernels.CopyIntoFactorStorage(a, factors.AsSpan(0, Rows * Columns));
        Array.Clear(tau);
        Factor();
        if (q is not null)
        {
            FormQ(q.Data);
        }

        if (r is not null)
        {
            FormR(r.Data);
        }
    }

    private void Factor()
    {
        int m = Rows;
        int n = Columns;

        // Run unscaled first so tiny subnormal entries in normal inputs do not lose precision.
        // Only snapshot the input when a column is large enough to risk overflow, and retry with
        // per-column power-of-2 scaling only if non-finite entries actually appear.
        int limit = LargeColumnExponent(m);
        if (!HasLargeColumn(m, n, factors, limit))
        {
            QrKernels.Factor(m, n, factors, m, tau);
            return;
        }

        double[] original = WorkArrayPool<double>.Rent(m * n);
        double[] columnScales = WorkArrayPool<double>.Rent(n);
        try
        {
            factors.AsSpan(0, m * n).CopyTo(original);
            QrKernels.Factor(m, n, factors, m, tau);
            if (IsFinite(factors.AsSpan(0, m * n), tau))
            {
                return;
            }

            // Per-column power-of-2 scaling commutes with rounding (absent subnormals), leaving
            // Householder reflectors (v, tau) and unscaled columns unchanged while avoiding zeroing small columns.
            original.AsSpan(0, m * n).CopyTo(factors);
            Array.Clear(tau);
            ScaleLargeColumns(m, n, factors, limit, columnScales);
            QrKernels.Factor(m, n, factors, m, tau);
            int k = Math.Min(m, n);
            for (int j = 0; j < n; j++)
            {
                if (columnScales[j] != 1)
                {
                    Level1Kernels.Scale(1.0 / columnScales[j], factors.AsSpan(j * m, Math.Min(j + 1, k)));
                }
            }
        }
        finally
        {
            WorkArrayPool<double>.Return(original);
            WorkArrayPool<double>.Return(columnScales);
        }
    }

    // Columns with max |a_ij| < 2^(1013 - ceil(log2 m)) cannot overflow during Householder updates (2^10 headroom).
    private static int LargeColumnExponent(int m) =>
        1013 - (m <= 1 ? 0 : 32 - System.Numerics.BitOperations.LeadingZeroCount((uint)(m - 1)));

    private static bool HasLargeColumn(int m, int n, ReadOnlySpan<double> a, int limit)
    {
        double max = 0;
        foreach (double v in a[..(m * n)])
        {
            max = Math.Max(max, Math.Abs(v));
        }

        return double.IsFinite(max) && max >= Math.ScaleB(1.0, limit);
    }

    private static bool IsFinite(ReadOnlySpan<double> factors, ReadOnlySpan<double> tau)
    {
        foreach (double v in factors)
        {
            if (!double.IsFinite(v))
            {
                return false;
            }
        }

        foreach (double v in tau)
        {
            if (!double.IsFinite(v))
            {
                return false;
            }
        }

        return true;
    }

    private static void ScaleLargeColumns(int m, int n, Span<double> a, int limit, Span<double> scales)
    {
        for (int j = 0; j < n; j++)
        {
            var column = a.Slice(j * m, m);
            double max = 0;
            foreach (double v in column)
            {
                max = Math.Max(max, Math.Abs(v));
            }

            scales[j] = max < Math.ScaleB(1.0, limit) ? 1 : Math.ScaleB(1.0, limit - 1 - Math.ILogB(max));
            if (scales[j] != 1)
            {
                Level1Kernels.Scale(scales[j], column);
            }
        }
    }

    /// <summary>
    /// Computes the linear least-squares solution <c>argmin ‖Ax - b‖₂</c> assuming <c>Rows &gt;= Columns</c> and full column rank.
    /// <para><c>Rows &gt;= Columns</c> かつ列フルランクのもとで最小二乗解 <c>argmin ‖Ax - b‖₂</c> を計算します。</para>
    /// </summary>
    /// <param name="b">The right-hand side vector of length <see cref="Rows"/>. / 長さ <see cref="Rows"/> の右辺ベクトル。</param>
    /// <returns>A new array of length <see cref="Columns"/> containing the least-squares solution. / 最小二乗解を格納した長さ <see cref="Columns"/> の新しい配列。</returns>
    /// <exception cref="InvalidOperationException"><see cref="Rows"/> is less than <see cref="Columns"/>. / <see cref="Rows"/> が <see cref="Columns"/> 未満の場合。</exception>
    /// <exception cref="ArgumentException">The length of <paramref name="b"/> does not equal <see cref="Rows"/>. / <paramref name="b"/> の長さが <see cref="Rows"/> と一致しない場合。</exception>
    public double[] Solve(ReadOnlySpan<double> b)
    {
        var x = new double[Columns];
        Solve(b, x);
        return x;
    }

    /// <summary>
    /// Computes the linear least-squares solution <c>argmin ‖Ax - b‖₂</c> and writes it into <paramref name="destination"/>.
    /// <para>最小二乗解 <c>argmin ‖Ax - b‖₂</c> を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="b">The right-hand side vector of length <see cref="Rows"/>. / 長さ <see cref="Rows"/> の右辺ベクトル。</param>
    /// <param name="destination">The destination span of length <see cref="Columns"/> (may overlap <paramref name="b"/>). / 長さ <see cref="Columns"/> の出力先スパン（<paramref name="b"/> と重複可）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="InvalidOperationException"><see cref="Rows"/> is less than <see cref="Columns"/>. / <see cref="Rows"/> が <see cref="Columns"/> 未満の場合。</exception>
    /// <exception cref="ArgumentException">The length of <paramref name="b"/> does not equal <see cref="Rows"/>, or the length of <paramref name="destination"/> does not equal <see cref="Columns"/>. / <paramref name="b"/> または <paramref name="destination"/> の長さが一致しない場合。</exception>
    public Span<double> Solve(ReadOnlySpan<double> b, Span<double> destination)
    {
        int m = Rows;
        int n = Columns;
        if (m < n)
        {
            throw new InvalidOperationException("Least-squares solve requires rows >= columns.");
        }

        if (b.Length != m)
        {
            throw new ArgumentException("The length must equal the number of rows.", nameof(b));
        }

        if (destination.Length != n)
        {
            throw new ArgumentException("The length must equal the number of columns.", nameof(destination));
        }

        double[] work = WorkArrayPool<double>.Rent(m);
        try
        {
            var y = work.AsSpan(0, m);
            b.CopyTo(y);
            SolveColumn(y, destination);
        }
        finally
        {
            WorkArrayPool<double>.Return(work);
        }

        return destination;
    }

    /// <summary>
    /// Computes the linear least-squares solution <c>argmin ‖Ax - b‖₂</c> assuming <c>Rows &gt;= Columns</c> and full column rank.
    /// <para><c>Rows &gt;= Columns</c> かつ列フルランクのもとで最小二乗解 <c>argmin ‖Ax - b‖₂</c> を計算します。</para>
    /// </summary>
    /// <param name="b">The right-hand side vector of length <see cref="Rows"/>. / 長さ <see cref="Rows"/> の右辺ベクトル。</param>
    /// <returns>A newly allocated vector of length <see cref="Columns"/> containing the least-squares solution. / 最小二乗解を格納した長さ <see cref="Columns"/> の新しいベクトル。</returns>
    /// <exception cref="InvalidOperationException"><see cref="Rows"/> is less than <see cref="Columns"/>. / <see cref="Rows"/> が <see cref="Columns"/> 未満の場合。</exception>
    /// <exception cref="ArgumentException">The length of <paramref name="b"/> does not equal <see cref="Rows"/>. / <paramref name="b"/> の長さが <see cref="Rows"/> と一致しない場合。</exception>
    public Vector<double> Solve(ReadOnlyVectorSpan<double> b)
    {
        var x = Vector<double>.Uninitialized(Columns);
        Solve(b, x);
        return x;
    }

    /// <summary>
    /// Computes the linear least-squares solution <c>argmin ‖Ax - b‖₂</c> and writes it into <paramref name="destination"/>.
    /// <para>最小二乗解 <c>argmin ‖Ax - b‖₂</c> を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="b">The right-hand side vector of length <see cref="Rows"/>. / 長さ <see cref="Rows"/> の右辺ベクトル。</param>
    /// <param name="destination">The destination vector of length <see cref="Columns"/> (may overlap <paramref name="b"/>, but must not overlap the factor storage). / 長さ <see cref="Columns"/> の出力先ベクトル（<paramref name="b"/> との重複は可能ですが、因子領域との重複は不可です）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="InvalidOperationException"><see cref="Rows"/> is less than <see cref="Columns"/>. / <see cref="Rows"/> が <see cref="Columns"/> 未満の場合。</exception>
    /// <exception cref="ArgumentException">The lengths do not match <see cref="Rows"/> and <see cref="Columns"/>, or <paramref name="destination"/> overlaps the factor storage. / 長さが一致しない、または <paramref name="destination"/> が因子領域と重なっている場合。</exception>
    public VectorSpan<double> Solve(ReadOnlyVectorSpan<double> b, VectorSpan<double> destination)
    {
        Solve(b.AsMatrix(), destination.AsMatrix());
        return destination;
    }

    /// <summary>
    /// Computes the linear least-squares solutions <c>argmin ‖AX - B‖_F</c> for multiple right-hand sides assuming <c>Rows &gt;= Columns</c> and full column rank.
    /// <para><c>Rows &gt;= Columns</c> かつ列フルランクのもとで、複数右辺に対する最小二乗解 <c>argmin ‖AX - B‖_F</c> を計算します。</para>
    /// </summary>
    /// <param name="b">The right-hand side matrix with <see cref="Rows"/> rows. / 行数が <see cref="Rows"/> の右辺行列。</param>
    /// <returns>A newly allocated matrix with <see cref="Columns"/> rows containing the least-squares solutions. / 最小二乗解を格納した行数 <see cref="Columns"/> の新しい行列。</returns>
    /// <exception cref="InvalidOperationException"><see cref="Rows"/> is less than <see cref="Columns"/>. / <see cref="Rows"/> が <see cref="Columns"/> 未満の場合。</exception>
    /// <exception cref="ArgumentException">The number of rows in <paramref name="b"/> does not equal <see cref="Rows"/>. / <paramref name="b"/> の行数が <see cref="Rows"/> と一致しない場合。</exception>
    public Matrix<double> Solve(ReadOnlyMatrixSpan<double> b)
    {
        var x = Matrix<double>.Uninitialized(Columns, b.Columns);
        Solve(b, x);
        return x;
    }

    /// <summary>
    /// Computes the linear least-squares solutions <c>argmin ‖AX - B‖_F</c> for multiple right-hand sides and writes them into <paramref name="destination"/>.
    /// <para>複数右辺に対する最小二乗解 <c>argmin ‖AX - B‖_F</c> を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="b">The right-hand side matrix with <see cref="Rows"/> rows. / 行数が <see cref="Rows"/> の右辺行列。</param>
    /// <param name="destination">The destination matrix with <see cref="Columns"/> rows and the same number of columns as <paramref name="b"/> (may overlap <paramref name="b"/>, but must not overlap the factor storage). / 行数 <see cref="Columns"/>、列数が <paramref name="b"/> と一致する出力先行列（<paramref name="b"/> との重複は可能ですが、因子領域との重複は不可です）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="InvalidOperationException"><see cref="Rows"/> is less than <see cref="Columns"/>. / <see cref="Rows"/> が <see cref="Columns"/> 未満の場合。</exception>
    /// <exception cref="ArgumentException">The shapes do not match, or <paramref name="destination"/> overlaps the factor storage. / 形状が一致しない、または <paramref name="destination"/> が因子領域と重なっている場合。</exception>
    public MatrixSpan<double> Solve(ReadOnlyMatrixSpan<double> b, MatrixSpan<double> destination)
    {
        int m = Rows;
        int n = Columns;
        if (m < n)
        {
            throw new InvalidOperationException("Least-squares solve requires rows >= columns.");
        }

        if (b.Rows != m)
        {
            throw new ArgumentException("The number of rows must equal the number of rows of the decomposition.", nameof(b));
        }

        if (destination.Rows != n || destination.Columns != b.Columns)
        {
            throw new ArgumentException("Destination shape must agree.", nameof(destination));
        }

        StridedKernels.RequireDisjoint(new ReadOnlyMatrixSpan<double>(factors.AsSpan(0, m * n), m, n), destination);
        int nrhs = b.Columns;
        if (n == 0 || nrhs == 0)
        {
            return destination;
        }

        ReadOnlyMatrixSpan<double> source = b;
        if (nrhs > 1 && StridedKernels.Overlaps(b, destination, out _))
        {
            source = b.ToMatrix();
        }

        double[] yBuffer = WorkArrayPool<double>.Rent(m);
        double[] xBuffer = WorkArrayPool<double>.Rent(n);
        try
        {
            var y = yBuffer.AsSpan(0, m);
            var x = xBuffer.AsSpan(0, n);
            for (int col = 0; col < nrhs; col++)
            {
                source.Column(col).CopyTo(new VectorSpan<double>(y));

                var dstCol = destination.Column(col);
                if (dstCol.Length <= 1 || dstCol.Strides[0] == 1)
                {
                    SolveColumn(y, dstCol.Data.Slice(0, n));
                }
                else
                {
                    SolveColumn(y, x);
                    new ReadOnlyVectorSpan<double>(x).CopyTo(dstCol);
                }
            }
        }
        finally
        {
            WorkArrayPool<double>.Return(yBuffer);
            WorkArrayPool<double>.Return(xBuffer);
        }

        return destination;
    }

    private void SolveColumn(Span<double> y, Span<double> destination)
    {
        int m = Rows;
        int n = Columns;

        // Apply y = Q^T b using stored Householder reflectors.
        for (int i = 0; i < n; i++)
        {
            double t = tau[i];
            if (t == 0)
            {
                continue;
            }

            var v = factors.AsSpan(i + 1 + i * m, m - i - 1);
            double w = t * (y[i] + Level1Kernels.Dot(v, y.Slice(i + 1)));
            y[i] -= w;
            Level1Kernels.Axpy(-w, v, y.Slice(i + 1));
        }

        // Back-substitute Rx = y[0:n].
        for (int i = n - 1; i >= 0; i--)
        {
            double sum = y[i];
            for (int j = i + 1; j < n; j++)
            {
                sum -= factors[i + j * m] * destination[j];
            }

            destination[i] = sum / factors[i + i * m];
        }
    }

    private Matrix<double> FormQ()
    {
        int k = Math.Min(Rows, Columns);
        var qData = new double[Rows * k];
        FormQ(qData);
        return Matrix<double>.Wrap(Rows, k, qData);
    }

    // Rebuilds Q from the identity matrix, so prior contents of qData are irrelevant.
    private void FormQ(Span<double> qData)
    {
        int k = Math.Min(Rows, Columns);
        QrKernels.FormQ(Rows, k, k, factors, Rows, tau, qData, Rows);
    }

    private Matrix<double> FormR()
    {
        int k = Math.Min(Rows, Columns);
        var rData = new double[k * Columns];
        FormR(rData);
        return Matrix<double>.Wrap(k, Columns, rData);
    }

    private void FormR(Span<double> rData)
    {
        int k = Math.Min(Rows, Columns);
        for (int j = 0; j < Columns; j++)
        {
            int len = Math.Min(j + 1, k);
            factors.AsSpan(j * Rows, len).CopyTo(rData.Slice(j * k, len));
            rData.Slice(j * k + len, k - len).Clear();
        }
    }
}
