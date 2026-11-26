using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using Tuna.LinearAlgebra;

namespace Tuna.Differentiation;

/// <summary>
/// Represents a partially pivoted LU decomposition <c>PA = LU</c> recorded on a <see cref="Differentiation.Tape"/>.
/// <para><see cref="Differentiation.Tape"/> に記録された部分ピボット付き LU 分解 <c>PA = LU</c> を表すノードです。</para>
/// </summary>
/// <remarks>
/// <para>
/// Created from a square matrix node via <see cref="Compute"/> so multiple linear systems (<see cref="Solve(VectorNode{double})"/>, <see cref="SolveTransposed(VectorNode{double})"/>, and their matrix overloads) and <see cref="LogAbsDeterminant"/> can share a single factorization.
/// Values match <see cref="LuDecomposition"/> bit-for-bit, and the underlying decomposition is pooled per order by the tape for reuse across recordings.
/// Gradients are accumulated directly into the input matrix using analytic solve and log-determinant rules rather than differentiating through the factorization. Nodes become invalid after <see cref="Tape.Reset"/>.
/// </para>
/// <para>
/// <see cref="Compute"/> で正方行列ノードから作成し、単一の分解を複数の求解（<see cref="Solve(VectorNode{double})"/>・<see cref="SolveTransposed(VectorNode{double})"/> およびその行列オーバーロード）と <see cref="LogAbsDeterminant"/> で共有します。
/// 値は <see cref="LuDecomposition"/> とビット単位で一致し、分解オブジェクトはテープが次数ごとにプールして再利用します。
/// 勾配は分解過程をたどらず、求解と対数行列式の微分式から元の行列へ直接加算します。<see cref="Tape.Reset"/> 後はノードを使用できません。
/// </para>
/// </remarks>
public readonly struct LuDecompositionNode
{
    internal readonly Tape? Tape;
    internal readonly int Generation;
    internal readonly int Index;

    private LuDecompositionNode(Tape tape, int index)
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
    public int Order => Decomposition(Owner()).Order;

    /// <summary>
    /// Gets a value indicating whether an exact zero pivot was encountered during factorization.
    /// <para>分解の途中でピボットが厳密に 0 になった（特異である）かどうかを取得します。</para>
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized or belongs to a stale tape generation. / ノードが未初期化、またはテープの世代が古い場合。</exception>
    public bool IsSingular => Decomposition(Owner()).IsSingular;

    /// <summary>
    /// Gets the sign of the determinant (<c>1</c>, <c>-1</c>, or <c>0</c> if singular; non-differentiable).
    /// <para>行列式の符号（<c>1</c>、<c>-1</c>、特異の場合は <c>0</c>。微分対象外）を取得します。</para>
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized or belongs to a stale tape generation. / ノードが未初期化、またはテープの世代が古い場合。</exception>
    public int DeterminantSign => Decomposition(Owner()).DeterminantSign;

    /// <summary>
    /// Gets the natural logarithm of the absolute value of the determinant, <c>ln |det A|</c>.
    /// <para>行列式の絶対値の自然対数 <c>ln |det A|</c> を取得します。</para>
    /// </summary>
    /// <value>A scalar node matching <see cref="LuDecomposition.LogAbsDeterminant"/> bit-for-bit (<c>-∞</c> if singular). / <see cref="LuDecomposition.LogAbsDeterminant"/> とビット単位で一致するスカラーノード（特異の場合は <c>-∞</c>）。</value>
    /// <remarks>
    /// <para>The local derivative is <c>A⁻ᵀ</c>, or <c>NaN</c> when the matrix is singular.</para>
    /// <para>局所微分は <c>A⁻ᵀ</c> とし、特異な場合は <c>NaN</c> とします。</para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized or belongs to a stale tape generation. / ノードが未初期化、またはテープの世代が古い場合。</exception>
    public ScalarNode LogAbsDeterminant
    {
        get
        {
            Tape tape = Owner();
            ref DecompositionInfo info = ref tape.Store<double>().Decompositions.Info(Index);
            return info.LogDeterminantIndex < 0 ? new(info.LogDeterminant) : new(info.LogDeterminant, tape, info.LogDeterminantIndex, Generation);
        }
    }

    /// <summary>
    /// Computes the LU decomposition of a square matrix node.
    /// <para>正方行列ノードの LU 分解を計算します。</para>
    /// </summary>
    /// <param name="a">The square matrix node. / 正方行列ノード。</param>
    /// <returns>The LU decomposition node (returned even when the matrix is singular). / LU 分解ノード（特異行列の場合も返されます）。</returns>
    /// <remarks>
    /// <para>When <paramref name="a"/> requires a gradient, <see cref="LogAbsDeterminant"/> is recorded immediately and skipped during backpropagation if unused.</para>
    /// <para><paramref name="a"/> が勾配を必要とする場合、<see cref="LogAbsDeterminant"/> はこの時点で記録されます（未使用であれば逆伝播時にスキップされます）。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if <paramref name="a"/> is not square. / 正方行列でない場合。</exception>
    /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
    public static LuDecompositionNode Compute(MatrixNode<double> a)
    {
        Tape tape = a.Owner();
        if (a.Rows != a.Columns)
        {
            throw new ArgumentException("The matrix must be square.", nameof(a));
        }

        var store = tape.Store<double>();
        bool record = store.RequiresGradient(a.View.Slot);
        tape.EnsureCanApply(record);

        var pool = store.Decompositions;
        int index = pool.TakeLu(a.Rows);
        var lu = pool.Lu(index);
        lu.Recompute(store.MatrixView(a.View));
        ref DecompositionInfo info = ref pool.Info(index);
        info.Input = a.View;
        info.Factor = -1;
        info.Second = -1;
        info.Third = -1;
        info.RequiresGradient = record;
        info.LogDeterminant = lu.LogAbsDeterminant;
        info.LogDeterminantIndex = -1;
        if (record)
        {
            info.LogDeterminantIndex = tape.AppendScalarOutput();
            tape.AppendArrayRecord<double>(ArrayOperation.LuLogAbsDeterminant, info.LogDeterminantIndex, a.View, default, 0, index);
        }

        return new(tape, index);
    }

    /// <summary>
    /// Solves the linear system <c>Ax = b</c> for a single right-hand side vector.
    /// <para>単一右辺の連立一次方程式 <c>Ax = b</c> を解きます。</para>
    /// </summary>
    /// <param name="b">The right-hand side vector node. / 右辺のベクトルノード。</param>
    /// <returns>The solution vector node <c>x</c>. / 解のベクトルノード <c>x</c>。</returns>
    /// <remarks>
    /// <para>The local derivatives accumulate <c>db += u</c> and <c>dA -= u·xᵀ</c>, where <c>u = A⁻ᵀ·dx</c>.</para>
    /// <para>局所微分は <c>u = A⁻ᵀ·dx</c> として <c>db += u</c>、<c>dA -= u·xᵀ</c> とします。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the length of <paramref name="b"/> does not match <see cref="Order"/>. / <paramref name="b"/> の長さが <see cref="Order"/> と一致しない場合。</exception>
    /// <exception cref="InvalidOperationException">Thrown if the matrix is singular, either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / 行列が特異である、ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
    public VectorNode<double> Solve(VectorNode<double> b) => SolveCore(b, transposed: false);

    /// <summary>
    /// Solves the linear system <c>AX = B</c> for multiple right-hand sides.
    /// <para>複数右辺の連立一次方程式 <c>AX = B</c> を解きます。</para>
    /// </summary>
    /// <param name="b">The right-hand side matrix node. / 右辺の行列ノード。</param>
    /// <returns>The solution matrix node <c>X</c>. / 解の行列ノード <c>X</c>。</returns>
    /// <remarks>
    /// <para>The local derivatives accumulate <c>dB += U</c> and <c>dA -= U·Xᵀ</c>, where <c>U = A⁻ᵀ·dX</c>.</para>
    /// <para>局所微分は <c>U = A⁻ᵀ·dX</c> として <c>dB += U</c>、<c>dA -= U·Xᵀ</c> とします。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the row count of <paramref name="b"/> does not match <see cref="Order"/>. / <paramref name="b"/> の行数が <see cref="Order"/> と一致しない場合。</exception>
    /// <exception cref="InvalidOperationException">Thrown if the matrix is singular, either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / 行列が特異である、ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
    public MatrixNode<double> Solve(MatrixNode<double> b) => SolveCore(b, transposed: false);

    /// <summary>
    /// Solves the transposed linear system <c>Aᵀx = b</c> for a single right-hand side vector.
    /// <para>転置された単一右辺の連立一次方程式 <c>Aᵀx = b</c> を解きます。</para>
    /// </summary>
    /// <param name="b">The right-hand side vector node. / 右辺のベクトルノード。</param>
    /// <returns>The solution vector node <c>x</c>. / 解のベクトルノード <c>x</c>。</returns>
    /// <remarks>
    /// <para>The local derivatives accumulate <c>db += u</c> and <c>dA -= x·uᵀ</c>, where <c>u = A⁻¹·dx</c>.</para>
    /// <para>局所微分は <c>u = A⁻¹·dx</c> として <c>db += u</c>、<c>dA -= x·uᵀ</c> とします。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the length of <paramref name="b"/> does not match <see cref="Order"/>. / <paramref name="b"/> の長さが <see cref="Order"/> と一致しない場合。</exception>
    /// <exception cref="InvalidOperationException">Thrown if the matrix is singular, either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / 行列が特異である、ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
    public VectorNode<double> SolveTransposed(VectorNode<double> b) => SolveCore(b, transposed: true);

    /// <summary>
    /// Solves the transposed linear system <c>AᵀX = B</c> for multiple right-hand sides.
    /// <para>転置された複数右辺の連立一次方程式 <c>AᵀX = B</c> を解きます。</para>
    /// </summary>
    /// <param name="b">The right-hand side matrix node. / 右辺の行列ノード。</param>
    /// <returns>The solution matrix node <c>X</c>. / 解の行列ノード <c>X</c>。</returns>
    /// <remarks>
    /// <para>The local derivatives accumulate <c>dB += U</c> and <c>dA -= X·Uᵀ</c>, where <c>U = A⁻¹·dX</c>.</para>
    /// <para>局所微分は <c>U = A⁻¹·dX</c> として <c>dB += U</c>、<c>dA -= X·Uᵀ</c> とします。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the row count of <paramref name="b"/> does not match <see cref="Order"/>. / <paramref name="b"/> の行数が <see cref="Order"/> と一致しない場合。</exception>
    /// <exception cref="InvalidOperationException">Thrown if the matrix is singular, either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / 行列が特異である、ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
    public MatrixNode<double> SolveTransposed(MatrixNode<double> b) => SolveCore(b, transposed: true);

    private VectorNode<double> SolveCore(VectorNode<double> b, bool transposed)
    {
        Tape tape = XMathExtensions.Owner(Owner(), Generation, b.Tape, b.Generation);
        var pool = tape.Store<double>().Decompositions;
        RequireSolvable(pool, b.Length);
        return XMathExtensions.SolveLu(tape, Index, pool.Info(Index).Input, b, transposed, release: false);
    }

    private MatrixNode<double> SolveCore(MatrixNode<double> b, bool transposed)
    {
        Tape tape = XMathExtensions.Owner(Owner(), Generation, b.Tape, b.Generation);
        var pool = tape.Store<double>().Decompositions;
        RequireSolvable(pool, b.Rows);
        return XMathExtensions.SolveLu(tape, Index, pool.Info(Index).Input, b, transposed, release: false);
    }

    // Validate shape before singularity to match LuDecomposition.Solve exception order.
    private void RequireSolvable(DecompositionPool pool, int rows)
    {
        var lu = pool.Lu(Index);
        if (rows != lu.Order)
        {
            throw new ArgumentException("The right-hand side must have as many rows as the order of the matrix.", "b");
        }

        if (lu.IsSingular)
        {
            throw new InvalidOperationException("The matrix is singular.");
        }
    }

    private static LuDecomposition Decomposition(Tape tape, int index) => tape.Store<double>().Decompositions.Lu(index);

    private LuDecomposition Decomposition(Tape tape) => Decomposition(tape, Index);

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
    internal static void ThrowUninitialized() => throw new InvalidOperationException("The decomposition node is not initialized. Create it with Compute.");
}
