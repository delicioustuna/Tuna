using Tuna.LinearAlgebra;

namespace Tuna.Differentiation;

public static partial class XMathExtensions
{
    extension(XMath)
    {
        /// <summary>
        /// Solves the linear system <c>Ax = b</c> using partially pivoted LU decomposition.
        /// <para>部分ピボット付き LU 分解を用いて連立一次方程式 <c>Ax = b</c> を解きます。</para>
        /// </summary>
        /// <param name="a">The square coefficient matrix node. / 正方係数行列ノード。</param>
        /// <param name="b">The right-hand side vector node. / 右辺のベクトルノード。</param>
        /// <returns>The solution vector node <c>x</c>. / 解のベクトルノード <c>x</c>。</returns>
        /// <remarks>
        /// <para>The local derivatives accumulate <c>db += u</c> and <c>dA -= u·xᵀ</c>, where <c>u = A⁻ᵀ·dx</c> is solved using the forward LU factorization. Use <see cref="LuDecompositionNode"/> to reuse one factorization across multiple solves or with <c>ln |det A|</c>.</para>
        /// <para>局所微分は順伝播の分解から <c>u = A⁻ᵀ·dx</c> を求め、<c>db += u</c>、<c>dA -= u·xᵀ</c> とします。同一係数で複数回解く場合や <c>ln |det A|</c> も必要な場合は <see cref="LuDecompositionNode"/> で分解を再利用してください。</para>
        /// </remarks>
        /// <exception cref="ArgumentException">Thrown if <paramref name="a"/> is not square, dimensions do not match, or <paramref name="a"/> is singular (exact zero pivot). / 正方行列でない、形状が一致しない、または特異（ピボットが厳密に 0）な場合。</exception>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<double> Solve(MatrixNode<double> a, VectorNode<double> b)
        {
            Tape tape = Owner(a.Tape, a.Generation, b.Tape, b.Generation);
            RequireSystemShape(a.Rows, a.Columns, b.Length);
            EnsureCanRecordSolve(tape, a.View, b.View);
            int index = TakeFactoredLu(tape, a);
            return SolveLu(tape, index, a.View, b, transposed: false, release: true);
        }

        /// <summary>
        /// Solves the linear system <c>AX = B</c> for multiple right-hand sides using partially pivoted LU decomposition.
        /// <para>部分ピボット付き LU 分解を用いて複数右辺の連立一次方程式 <c>AX = B</c> を解きます。</para>
        /// </summary>
        /// <param name="a">The square coefficient matrix node. / 正方係数行列ノード。</param>
        /// <param name="b">The right-hand side matrix node. / 右辺の行列ノード。</param>
        /// <returns>The solution matrix node <c>X</c>. / 解の行列ノード <c>X</c>。</returns>
        /// <remarks>
        /// <para>The local derivatives accumulate <c>dB += U</c> and <c>dA -= U·Xᵀ</c>, where <c>U = A⁻ᵀ·dX</c>.</para>
        /// <para>局所微分は <c>U = A⁻ᵀ·dX</c> として <c>dB += U</c>、<c>dA -= U·Xᵀ</c> とします。</para>
        /// </remarks>
        /// <exception cref="ArgumentException">Thrown if <paramref name="a"/> is not square, dimensions do not match, or <paramref name="a"/> is singular. / 正方行列でない、形状が一致しない、または特異な場合。</exception>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<double> Solve(MatrixNode<double> a, MatrixNode<double> b)
        {
            Tape tape = Owner(a.Tape, a.Generation, b.Tape, b.Generation);
            RequireSystemShape(a.Rows, a.Columns, b.Rows);
            EnsureCanRecordSolve(tape, a.View, b.View);
            int index = TakeFactoredLu(tape, a);
            return SolveLu(tape, index, a.View, b, transposed: false, release: true);
        }

        /// <summary>
        /// Solves the lower-triangular linear system <c>Lx = b</c>.
        /// <para>下三角行列を係数とする連立一次方程式 <c>Lx = b</c> を解きます。</para>
        /// </summary>
        /// <param name="a">The coefficient matrix node (only the lower triangle including the diagonal is read). / 係数行列ノード（対角を含む下三角部分のみ参照）。</param>
        /// <param name="b">The right-hand side vector node. / 右辺のベクトルノード。</param>
        /// <returns>The solution vector node <c>x</c>. / 解のベクトルノード <c>x</c>。</returns>
        /// <remarks>
        /// <para>The local derivatives accumulate <c>db += u</c> and <c>dL -= tril(u·xᵀ)</c>, where <c>u = L⁻ᵀ·dx</c>; strictly upper-triangular gradients remain 0. Zero diagonal entries are not checked and may produce <c>±∞</c> or <c>NaN</c>.</para>
        /// <para>局所微分は <c>u = L⁻ᵀ·dx</c> として <c>db += u</c>、<c>dL -= tril(u·xᵀ)</c> とし、上三角部分の勾配は 0 となります。対角の 0 は検査しないため、値および勾配に <c>±∞</c> や <c>NaN</c> が含まれる場合があります。</para>
        /// </remarks>
        /// <exception cref="ArgumentException">Thrown if <paramref name="a"/> is not square or dimensions do not match. / 正方行列でない、または形状が一致しない場合。</exception>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<double> SolveLowerTriangular(MatrixNode<double> a, VectorNode<double> b)
            => SolveTriangularCore(a, b, lower: true);

        /// <summary>
        /// Solves the lower-triangular linear system <c>LX = B</c> for multiple right-hand sides.
        /// <para>下三角行列を係数とする複数右辺の連立一次方程式 <c>LX = B</c> を解きます。</para>
        /// </summary>
        /// <param name="a">The coefficient matrix node (only the lower triangle including the diagonal is read). / 係数行列ノード（対角を含む下三角部分のみ参照）。</param>
        /// <param name="b">The right-hand side matrix node. / 右辺の行列ノード。</param>
        /// <returns>The solution matrix node <c>X</c>. / 解の行列ノード <c>X</c>。</returns>
        /// <remarks>
        /// <para>The local derivatives accumulate <c>dB += U</c> and <c>dL -= tril(U·Xᵀ)</c>, where <c>U = L⁻ᵀ·dX</c>; strictly upper-triangular gradients remain 0.</para>
        /// <para>局所微分は <c>U = L⁻ᵀ·dX</c> として <c>dB += U</c>、<c>dL -= tril(U·Xᵀ)</c> とし、上三角部分の勾配は 0 となります。</para>
        /// </remarks>
        /// <exception cref="ArgumentException">Thrown if <paramref name="a"/> is not square or dimensions do not match. / 正方行列でない、または形状が一致しない場合。</exception>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<double> SolveLowerTriangular(MatrixNode<double> a, MatrixNode<double> b)
            => SolveTriangularCore(a, b, lower: true);

        /// <summary>
        /// Solves the upper-triangular linear system <c>Ux = b</c>.
        /// <para>上三角行列を係数とする連立一次方程式 <c>Ux = b</c> を解きます。</para>
        /// </summary>
        /// <param name="a">The coefficient matrix node (only the upper triangle including the diagonal is read; pass <c>L.T</c> to solve with a transposed Cholesky factor). / 係数行列ノード（対角を含む上三角部分のみ参照。コレスキー因子 <c>L</c> の転置で解く場合は <c>L.T</c> を渡します）。</param>
        /// <param name="b">The right-hand side vector node. / 右辺のベクトルノード。</param>
        /// <returns>The solution vector node <c>x</c>. / 解のベクトルノード <c>x</c>。</returns>
        /// <remarks>
        /// <para>The local derivatives accumulate <c>db += u</c> and <c>dU -= triu(u·xᵀ)</c>, where <c>u = U⁻ᵀ·dx</c>; strictly lower-triangular gradients remain 0.</para>
        /// <para>局所微分は <c>u = U⁻ᵀ·dx</c> として <c>db += u</c>、<c>dU -= triu(u·xᵀ)</c> とし、下三角部分の勾配は 0 となります。</para>
        /// </remarks>
        /// <exception cref="ArgumentException">Thrown if <paramref name="a"/> is not square or dimensions do not match. / 正方行列でない、または形状が一致しない場合。</exception>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<double> SolveUpperTriangular(MatrixNode<double> a, VectorNode<double> b)
            => SolveTriangularCore(a, b, lower: false);

        /// <summary>
        /// Solves the upper-triangular linear system <c>UX = B</c> for multiple right-hand sides.
        /// <para>上三角行列を係数とする複数右辺の連立一次方程式 <c>UX = B</c> を解きます。</para>
        /// </summary>
        /// <param name="a">The coefficient matrix node (only the upper triangle including the diagonal is read). / 係数行列ノード（対角を含む上三角部分のみ参照）。</param>
        /// <param name="b">The right-hand side matrix node. / 右辺の行列ノード。</param>
        /// <returns>The solution matrix node <c>X</c>. / 解の行列ノード <c>X</c>。</returns>
        /// <remarks>
        /// <para>The local derivatives accumulate <c>dB += W</c> and <c>dU -= triu(W·Xᵀ)</c>, where <c>W = U⁻ᵀ·dX</c>; strictly lower-triangular gradients remain 0.</para>
        /// <para>局所微分は <c>W = U⁻ᵀ·dX</c> として <c>dB += W</c>、<c>dU -= triu(W·Xᵀ)</c> とし、下三角部分の勾配は 0 となります。</para>
        /// </remarks>
        /// <exception cref="ArgumentException">Thrown if <paramref name="a"/> is not square or dimensions do not match. / 正方行列でない、または形状が一致しない場合。</exception>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<double> SolveUpperTriangular(MatrixNode<double> a, MatrixNode<double> b)
            => SolveTriangularCore(a, b, lower: false);
    }

    internal static void RequireSystemShape(int rows, int columns, int rightRows)
    {
        if (rows != columns)
        {
            throw new ArgumentException("The coefficient matrix must be square.", "a");
        }

        if (rightRows != rows)
        {
            throw new ArgumentException("The right-hand side must have as many rows as the coefficient matrix.", "b");
        }
    }

    // Check tape state before borrowing an LU instance so exceptions do not leak a pool entry.
    private static void EnsureCanRecordSolve(Tape tape, in ArrayView a, in ArrayView b)
    {
        var store = tape.Store<double>();
        tape.EnsureCanApply(store.RequiresGradient(a.Slot) || store.RequiresGradient(b.Slot));
    }

    private static int TakeFactoredLu(Tape tape, MatrixNode<double> a)
    {
        var store = tape.Store<double>();
        var pool = store.Decompositions;
        int index = pool.TakeLu(a.Rows);
        var lu = pool.Lu(index);
        lu.Recompute(store.MatrixView(a.View));
        if (lu.IsSingular)
        {
            pool.ReleaseLast(index);
            throw new ArgumentException("The matrix is singular.", nameof(a));
        }

        return index;
    }

    // When release is true (one-shot XMath.Solve), return the pooled LU immediately if no gradient is recorded.
    internal static VectorNode<double> SolveLu(Tape tape, int index, in ArrayView a, VectorNode<double> b, bool transposed, bool release)
    {
        var store = tape.Store<double>();
        var pool = store.Decompositions;
        bool record = store.RequiresGradient(a.Slot) || store.RequiresGradient(b.View.Slot);
        tape.EnsureCanApply(record);

        var lu = pool.Lu(index);
        int n = b.Length;
        int slot = store.AddPooled(n, 1, record);
        if (transposed)
        {
            lu.SolveTransposed(store.VectorView(b.View), store.WritableVector(slot));
        }
        else
        {
            lu.Solve(store.VectorView(b.View), store.WritableVector(slot));
        }

        if (record)
        {
            var operation = transposed ? ArrayOperation.LuSolveTransposedVector : ArrayOperation.LuSolveVector;
            tape.AppendArrayRecord<double>(operation, slot, a, b.View, 0, index);
        }
        else if (release)
        {
            pool.ReleaseLast(index);
        }

        return new(tape, ArrayView.Whole(slot, n, 1));
    }

    internal static MatrixNode<double> SolveLu(Tape tape, int index, in ArrayView a, MatrixNode<double> b, bool transposed, bool release)
    {
        var store = tape.Store<double>();
        var pool = store.Decompositions;
        bool record = store.RequiresGradient(a.Slot) || store.RequiresGradient(b.View.Slot);
        tape.EnsureCanApply(record);

        var lu = pool.Lu(index);
        int n = b.Rows, k = b.Columns;
        int slot = store.AddPooled(n, k, record);
        if (transposed)
        {
            lu.SolveTransposed(store.MatrixView(b.View), store.WritableMatrix(slot));
        }
        else
        {
            lu.Solve(store.MatrixView(b.View), store.WritableMatrix(slot));
        }

        if (record)
        {
            var operation = transposed ? ArrayOperation.LuSolveTransposedMatrix : ArrayOperation.LuSolveMatrix;
            tape.AppendArrayRecord<double>(operation, slot, a, b.View, 0, index);
        }
        else if (release)
        {
            pool.ReleaseLast(index);
        }

        return new(tape, ArrayView.Whole(slot, n, k));
    }

    // Materialize broadcast views into tape scratch to avoid per-call allocations inside core triangular solves.
    private static ReadOnlyMatrixSpan<double> Coefficient(ArrayStore<double> store, in ArrayView a)
        => store.CoefficientView(a, ArrayStore<double>.MatrixContribution, transpose: false);

    private static VectorNode<double> SolveTriangularCore(MatrixNode<double> a, VectorNode<double> b, bool lower)
    {
        Tape tape = Owner(a.Tape, a.Generation, b.Tape, b.Generation);
        RequireSystemShape(a.Rows, a.Columns, b.Length);
        var store = tape.Store<double>();
        bool record = store.RequiresGradient(a.View.Slot) || store.RequiresGradient(b.View.Slot);
        tape.EnsureCanApply(record);

        int n = b.Length;
        int slot = store.AddPooled(n, 1, record);
        if (lower)
        {
            XMath.SolveLowerTriangular(Coefficient(store, a.View), store.VectorView(b.View), store.WritableVector(slot));
        }
        else
        {
            XMath.SolveUpperTriangular(Coefficient(store, a.View), store.VectorView(b.View), store.WritableVector(slot));
        }

        if (record)
        {
            var operation = lower ? ArrayOperation.SolveLowerTriangularVector : ArrayOperation.SolveUpperTriangularVector;
            tape.AppendArrayRecord<double>(operation, slot, a.View, b.View);
        }

        return new(tape, ArrayView.Whole(slot, n, 1));
    }

    private static MatrixNode<double> SolveTriangularCore(MatrixNode<double> a, MatrixNode<double> b, bool lower)
    {
        Tape tape = Owner(a.Tape, a.Generation, b.Tape, b.Generation);
        RequireSystemShape(a.Rows, a.Columns, b.Rows);
        var store = tape.Store<double>();
        bool record = store.RequiresGradient(a.View.Slot) || store.RequiresGradient(b.View.Slot);
        tape.EnsureCanApply(record);

        int n = b.Rows, k = b.Columns;
        int slot = store.AddPooled(n, k, record);
        if (lower)
        {
            XMath.SolveLowerTriangular(Coefficient(store, a.View), store.MatrixView(b.View), store.WritableMatrix(slot));
        }
        else
        {
            XMath.SolveUpperTriangular(Coefficient(store, a.View), store.MatrixView(b.View), store.WritableMatrix(slot));
        }

        if (record)
        {
            var operation = lower ? ArrayOperation.SolveLowerTriangularMatrix : ArrayOperation.SolveUpperTriangularMatrix;
            tape.AppendArrayRecord<double>(operation, slot, a.View, b.View);
        }

        return new(tape, ArrayView.Whole(slot, n, k));
    }
}
