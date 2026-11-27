using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using Tuna.LinearAlgebra;

namespace Tuna.Differentiation;

/// <summary>
/// Represents a Cholesky decomposition <c>A = LLᵀ</c> of a symmetric positive-definite matrix recorded on a <see cref="Differentiation.Tape"/>.
/// <para><see cref="Differentiation.Tape"/> に記録された対称正定値行列のコレスキー分解 <c>A = LLᵀ</c> を表すノードです。</para>
/// </summary>
/// <remarks>
/// <para>
/// Created from a matrix node via <see cref="Compute"/> or <see cref="TryCompute"/>. The factor <see cref="L"/>, solves (<see cref="Solve(VectorNode{double})"/>, <see cref="Solve(MatrixNode{double})"/>), <see cref="LogDeterminant"/>, and <see cref="InverseQuadraticForm"/> all share the same factorization, whose values match <see cref="CholeskyDecomposition"/> bit-for-bit.
/// The underlying decomposition is pooled per matrix order by the tape and reused across recordings.
/// </para>
/// <para>
/// Only the lower triangle (including the diagonal) of the input matrix is read, and gradients are accumulated exclusively into the lower triangle (upper-triangular gradients remain 0). Each strictly lower-triangular gradient <c>dA[i, j]</c> (<c>i &gt; j</c>) represents the derivative with respect to perturbing both <c>(i, j)</c> and <c>(j, i)</c> symmetrically. Nodes become invalid after <see cref="Tape.Reset"/>.
/// </para>
/// <para>
/// <see cref="Compute"/> または <see cref="TryCompute"/> で行列ノードから作成します。因子 <see cref="L"/>、連立方程式の求解（<see cref="Solve(VectorNode{double})"/>・<see cref="Solve(MatrixNode{double})"/>）、<see cref="LogDeterminant"/>、<see cref="InverseQuadraticForm"/> は同一の分解を共有し、値は <see cref="CholeskyDecomposition"/> とビット単位で一致します。
/// 分解オブジェクトはテープが次数ごとにプールし、次回以降の記録で再利用します。
/// </para>
/// <para>
/// 入力行列は下三角部分（対角を含む）のみを参照し、勾配も下三角部分にのみ加算します（上三角部分の勾配は 0）。下三角の非対角要素 <c>dA[i, j]</c>（<c>i &gt; j</c>）の勾配は、<c>(i, j)</c> と <c>(j, i)</c> を対称に変化させたときの微分に対応します。<see cref="Tape.Reset"/> 後はノードを使用できません。
/// </para>
/// </remarks>
public readonly struct CholeskyDecompositionNode
{
    internal readonly Tape? Tape;
    internal readonly int Generation;
    internal readonly int Index;

    private CholeskyDecompositionNode(Tape tape, int index)
    {
        Tape = tape;
        Generation = tape.Generation;
        Index = index;
    }

    /// <summary>
    /// Gets the order <c>n</c> of the factored matrix.
    /// <para>分解対象の行列の次数 <c>n</c> を取得します。</para>
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized or belongs to a stale tape generation. / ノードが未初期化、またはテープの世代が古い場合。</exception>
    public int Order => Decomposition(Owner()).L.Rows;

    /// <summary>
    /// Gets the lower-triangular Cholesky factor <c>L</c> with positive diagonal entries.
    /// <para>対角成分が正の下三角因子 <c>L</c> の行列ノードを取得します。</para>
    /// </summary>
    /// <value>A matrix node matching <see cref="CholeskyDecomposition.L"/> bit-for-bit (strictly upper triangle is 0). / <see cref="CholeskyDecomposition.L"/> とビット単位で一致する行列ノード（上三角部分は 0）。</value>
    /// <remarks>
    /// <para>The local derivative accumulates <c>dA += Φ(S + Sᵀ)</c> where <c>S = L⁻ᵀ Φ(Lᵀ·tril(dL)) L⁻¹</c> and <c>Φ</c> takes the lower triangle with halved diagonal; the strictly upper triangle of <c>dL</c> is ignored.</para>
    /// <para>局所微分は <c>S = L⁻ᵀ Φ(Lᵀ·tril(dL)) L⁻¹</c>（<c>Φ</c> は下三角を取り対角を半分にする操作）として <c>dA += Φ(S + Sᵀ)</c> を加算します。<c>dL</c> の上三角部分は参照しません。</para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized or belongs to a stale tape generation. / ノードが未初期化、またはテープの世代が古い場合。</exception>
    public MatrixNode<double> L
    {
        get
        {
            Tape tape = Owner();
            int n = Decomposition(tape).L.Rows;
            return new(tape, ArrayView.Whole(tape.Store<double>().Decompositions.Info(Index).Factor, n, n));
        }
    }

    /// <summary>
    /// Gets the natural logarithm of the determinant, <c>ln det A = 2 Σ ln lᵢᵢ</c>.
    /// <para>行列式の自然対数 <c>ln det A = 2 Σ ln lᵢᵢ</c> を取得します。</para>
    /// </summary>
    /// <value>A scalar node matching <see cref="CholeskyDecomposition.LogDeterminant"/> bit-for-bit. / <see cref="CholeskyDecomposition.LogDeterminant"/> とビット単位で一致するスカラーノード。</value>
    /// <remarks>
    /// <para>The local derivative is <c>A⁻¹</c>, accumulated into the lower triangle with off-diagonal entries doubled to <c>2A⁻¹[i, j]</c>.</para>
    /// <para>局所微分は <c>A⁻¹</c> とし、下三角部分に加算します（非対角要素は対称な 2 要素分の <c>2A⁻¹[i, j]</c>）。</para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized or belongs to a stale tape generation. / ノードが未初期化、またはテープの世代が古い場合。</exception>
    public ScalarNode LogDeterminant
    {
        get
        {
            Tape tape = Owner();
            ref DecompositionInfo info = ref tape.Store<double>().Decompositions.Info(Index);
            return info.LogDeterminantIndex < 0 ? new(info.LogDeterminant) : new(info.LogDeterminant, tape, info.LogDeterminantIndex, Generation);
        }
    }

    /// <summary>
    /// Computes the Cholesky decomposition of a symmetric positive-definite matrix node.
    /// <para>対称正定値行列ノードのコレスキー分解を計算します。</para>
    /// </summary>
    /// <param name="a">The symmetric positive-definite matrix node (lower triangle is read). / 対称正定値行列のノード（下三角部分のみ参照）。</param>
    /// <returns>The Cholesky decomposition node. / コレスキー分解ノード。</returns>
    /// <remarks>
    /// <para>When <paramref name="a"/> requires a gradient, records for <see cref="L"/> and <see cref="LogDeterminant"/> are appended immediately and skipped during backpropagation if unused.</para>
    /// <para><paramref name="a"/> が勾配を必要とする場合、<see cref="L"/> と <see cref="LogDeterminant"/> はこの時点で記録されます（未使用であれば逆伝播時にスキップされます）。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if <paramref name="a"/> is not square or is not numerically positive definite (nothing is recorded on the tape). / 正方行列でない、または数値的に正定値でない場合（テープには何も記録されません）。</exception>
    /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
    public static CholeskyDecompositionNode Compute(MatrixNode<double> a)
    {
        if (!TryCompute(a, out var result))
        {
            throw new ArgumentException("The matrix is not positive definite.", nameof(a));
        }

        return result;
    }

    /// <summary>
    /// Attempts to compute the Cholesky decomposition of a symmetric matrix node.
    /// <para>対称行列ノードのコレスキー分解を試みます。</para>
    /// </summary>
    /// <param name="a">The symmetric matrix node (lower triangle is read). / 対称行列のノード（下三角部分のみ参照）。</param>
    /// <param name="result">When this method returns <see langword="true"/>, contains the decomposition node; otherwise, <see langword="default"/>. / 分解に成功した場合は分解ノード、失敗した場合は既定値。</param>
    /// <returns><see langword="true"/> if <paramref name="a"/> is numerically positive definite; otherwise, <see langword="false"/> (nothing is recorded on the tape). / 数値的に正定値で分解できた場合は <see langword="true"/>、それ以外は <see langword="false"/>（失敗時はテープに記録しません）。</returns>
    /// <exception cref="ArgumentException">Thrown if <paramref name="a"/> is not square. / 正方行列でない場合。</exception>
    /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
    public static bool TryCompute(MatrixNode<double> a, out CholeskyDecompositionNode result)
    {
        Tape tape = a.Owner();
        var store = tape.Store<double>();
        int index = TryTakeFactor(tape, store, a, out bool record);
        if (index < 0)
        {
            result = default;
            return false;
        }

        var pool = store.Decompositions;
        var cholesky = pool.Cholesky(index);
        ref DecompositionInfo info = ref pool.Info(index);
        info.Input = a.View;
        info.RequiresGradient = record;
        info.Factor = store.AddFactor(cholesky.L, record);
        info.Second = -1;
        info.Third = -1;
        info.LogDeterminant = cholesky.LogDeterminant;
        info.LogDeterminantIndex = -1;
        if (record)
        {
            tape.AppendArrayRecord<double>(ArrayOperation.Cholesky, info.Factor, a.View, default, 0, index);
            info.LogDeterminantIndex = RecordLogDeterminant(tape, a.View, index);
        }

        result = new(tape, index);
        return true;
    }

    internal static ScalarNode ComputeLogDeterminant(MatrixNode<double> a)
    {
        Tape tape = a.Owner();
        var store = tape.Store<double>();
        int index = TryTakeFactor(tape, store, a, out bool record);
        if (index < 0)
        {
            throw new ArgumentException("The matrix is not positive definite.", nameof(a));
        }

        var pool = store.Decompositions;
        double value = pool.Cholesky(index).LogDeterminant;
        if (record)
        {
            int output = RecordLogDeterminant(tape, a.View, index);
            return new(value, tape, output, tape.Generation);
        }

        pool.ReleaseLast(index);
        return new(value);
    }

    private static int TryTakeFactor(Tape tape, ArrayStore<double> store, MatrixNode<double> a, out bool record)
    {
        if (a.Rows != a.Columns)
        {
            throw new ArgumentException("The matrix must be square.", nameof(a));
        }

        record = store.RequiresGradient(a.View.Slot);
        tape.EnsureCanApply(record);
        var pool = store.Decompositions;
        int index = pool.TakeCholesky(a.Rows);
        bool success = false;
        try
        {
            success = pool.Cholesky(index).TryRecompute(store.MatrixView(a.View));
            return success ? index : -1;
        }
        finally
        {
            if (!success)
            {
                pool.ReleaseLast(index);
            }
        }
    }

    private static int RecordLogDeterminant(Tape tape, in ArrayView a, int index)
    {
        int output = tape.AppendScalarOutput();
        tape.AppendArrayRecord<double>(ArrayOperation.CholeskyLogDeterminant, output, a, default, 0, index);
        return output;
    }

    /// <summary>
    /// Solves the linear system <c>Ax = b</c> for a single right-hand side vector.
    /// <para>単一右辺の連立一次方程式 <c>Ax = b</c> を解きます。</para>
    /// </summary>
    /// <param name="b">The right-hand side vector node. / 右辺のベクトルノード。</param>
    /// <returns>The solution vector node <c>x</c>. / 解のベクトルノード <c>x</c>。</returns>
    /// <remarks>
    /// <para>The local derivatives accumulate <c>db += u</c> and <c>dA -= u·xᵀ</c> (symmetrized into the lower triangle), where <c>u = A⁻¹·dx</c>.</para>
    /// <para>局所微分は <c>u = A⁻¹·dx</c> として <c>db += u</c>、<c>dA -= u·xᵀ</c>（下三角部分に対称化して加算）とします。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the length of <paramref name="b"/> does not match <see cref="Order"/>. / <paramref name="b"/> の長さが <see cref="Order"/> と一致しない場合。</exception>
    /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
    public VectorNode<double> Solve(VectorNode<double> b)
    {
        Tape tape = XMathExtensions.Owner(Owner(), Generation, b.Tape, b.Generation);
        var store = tape.Store<double>();
        var pool = store.Decompositions;
        var cholesky = pool.Cholesky(Index);
        int n = cholesky.L.Rows;
        RequireRows(n, b.Length, nameof(b));
        ref DecompositionInfo info = ref pool.Info(Index);
        bool record = info.RequiresGradient || store.RequiresGradient(b.View.Slot);
        tape.EnsureCanApply(record);

        int slot = store.AddPooled(n, 1, record);
        cholesky.Solve(store.VectorView(b.View), store.WritableVector(slot));
        if (record)
        {
            tape.AppendArrayRecord<double>(ArrayOperation.CholeskySolveVector, slot, info.Input, b.View, 0, Index);
        }

        return new(tape, ArrayView.Whole(slot, n, 1));
    }

    /// <summary>
    /// Solves the linear system <c>AX = B</c> for multiple right-hand sides.
    /// <para>複数右辺の連立一次方程式 <c>AX = B</c> を解きます。</para>
    /// </summary>
    /// <param name="b">The right-hand side matrix node. / 右辺の行列ノード。</param>
    /// <returns>The solution matrix node <c>X</c>. / 解の行列ノード <c>X</c>。</returns>
    /// <remarks>
    /// <para>The local derivatives accumulate <c>dB += U</c> and <c>dA -= U·Xᵀ</c> (symmetrized into the lower triangle), where <c>U = A⁻¹·dX</c>.</para>
    /// <para>局所微分は <c>U = A⁻¹·dX</c> として <c>dB += U</c>、<c>dA -= U·Xᵀ</c>（下三角部分に対称化して加算）とします。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the row count of <paramref name="b"/> does not match <see cref="Order"/>. / <paramref name="b"/> の行数が <see cref="Order"/> と一致しない場合。</exception>
    /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
    public MatrixNode<double> Solve(MatrixNode<double> b)
    {
        Tape tape = XMathExtensions.Owner(Owner(), Generation, b.Tape, b.Generation);
        var store = tape.Store<double>();
        var pool = store.Decompositions;
        var cholesky = pool.Cholesky(Index);
        int n = cholesky.L.Rows, k = b.Columns;
        RequireRows(n, b.Rows, nameof(b));
        ref DecompositionInfo info = ref pool.Info(Index);
        bool record = info.RequiresGradient || store.RequiresGradient(b.View.Slot);
        tape.EnsureCanApply(record);

        int slot = store.AddPooled(n, k, record);
        cholesky.Solve(store.MatrixView(b.View), store.WritableMatrix(slot));
        if (record)
        {
            tape.AppendArrayRecord<double>(ArrayOperation.CholeskySolveMatrix, slot, info.Input, b.View, 0, Index);
        }

        return new(tape, ArrayView.Whole(slot, n, k));
    }

    /// <summary>
    /// Computes the inverse quadratic form <c>xᵀA⁻¹x = ‖L⁻¹x‖²</c>.
    /// <para>二次形式 <c>xᵀA⁻¹x = ‖L⁻¹x‖²</c> を計算します。</para>
    /// </summary>
    /// <param name="x">The input vector node. / 入力ベクトルノード。</param>
    /// <returns>A non-negative scalar node holding <c>xᵀA⁻¹x</c>. / 二次形式の値を持つ非負のスカラーノード。</returns>
    /// <remarks>
    /// <para>Useful for squared Mahalanobis distances in multivariate normal models. The local derivatives accumulate <c>dx += 2w</c> and <c>dA -= w·wᵀ</c> (symmetrized into the lower triangle), where <c>w = A⁻¹x</c>.</para>
    /// <para>多変量正規分布のマハラノビス距離の 2 乗などに利用できます。局所微分は <c>w = A⁻¹x</c> として <c>dx += 2w</c>、<c>dA -= w·wᵀ</c>（下三角部分に対称化して加算）とします。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the length of <paramref name="x"/> does not match <see cref="Order"/>. / <paramref name="x"/> の長さが <see cref="Order"/> と一致しない場合。</exception>
    /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
    public ScalarNode InverseQuadraticForm(VectorNode<double> x)
    {
        Tape tape = XMathExtensions.Owner(Owner(), Generation, x.Tape, x.Generation);
        var store = tape.Store<double>();
        var pool = store.Decompositions;
        var cholesky = pool.Cholesky(Index);
        int n = cholesky.L.Rows;
        RequireRows(n, x.Length, nameof(x));
        ref DecompositionInfo info = ref pool.Info(Index);
        bool record = info.RequiresGradient || store.RequiresGradient(x.View.Slot);
        tape.EnsureCanApply(record);

        // Scratch buffers are only live during Backward, so borrowing one during forward is safe.
        var z = new VectorSpan<double>(store.Scratch(ArrayStore<double>.SolveAdjoint, n));
        XMath.SolveLowerTriangular(cholesky.L, store.VectorView(x.View), z);
        double value = XMath.SumOfSquares(z.AsReadOnly());
        if (!record)
        {
            return new(value);
        }

        int output = tape.AppendScalarOutput();
        tape.AppendArrayRecord<double>(ArrayOperation.CholeskyInverseQuadraticForm, output, info.Input, x.View, 0, Index);
        return new(value, tape, output, tape.Generation);
    }

    private static void RequireRows(int order, int rows, string parameterName)
    {
        if (rows != order)
        {
            throw new ArgumentException(parameterName == "x"
                ? "The vector must have as many elements as the order of the matrix."
                : "The right-hand side must have as many rows as the order of the matrix.", parameterName);
        }
    }

    private CholeskyDecomposition Decomposition(Tape tape) => tape.Store<double>().Decompositions.Cholesky(Index);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal Tape Owner()
    {
        Tape? tape = Tape;
        if (tape is null)
        {
            ThrowUninitialized();
        }

        tape.ValidateGeneration(Generation);
        return tape;
    }

    [DoesNotReturn]
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void ThrowUninitialized() => throw new InvalidOperationException("The decomposition node is not initialized. Create it with Compute.");
}
