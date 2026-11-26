using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using Tuna.LinearAlgebra;

namespace Tuna.Differentiation;

/// <summary>
/// Represents an eigendecomposition <c>A = VΛVᵀ</c> of a real symmetric matrix recorded on a <see cref="Differentiation.Tape"/>.
/// <para><see cref="Differentiation.Tape"/> に記録された実対称行列の固有値分解 <c>A = VΛVᵀ</c> を表すノードです。</para>
/// </summary>
/// <remarks>
/// <para>
/// Created from a square symmetric matrix node via <see cref="Compute"/>: <see cref="Values"/> holds eigenvalues in ascending order, and <see cref="Vectors"/> holds the corresponding orthonormal eigenvectors as columns of <c>V</c>.
/// Values match <see cref="SymmetricEigenDecomposition.Compute(Matrix{double}, bool, bool)"/> (with eigenvectors) bit-for-bit, and the underlying decomposition is pooled per matrix order by the tape for reuse across recordings.
/// </para>
/// <para>
/// Only the lower triangle (including the diagonal) of the input matrix is read, and gradients are accumulated exclusively into the lower triangle (upper-triangular gradients remain 0), with each strictly lower-triangular gradient reflecting symmetric perturbation of <c>(i, j)</c> and <c>(j, i)</c>.
/// Gradients through <see cref="Vectors"/> require distinct eigenvalues; for repeated eigenvalues <c>λᵢ = λⱼ</c> where <c>(VᵀV̄)ᵢⱼ ≠ 0</c>, gradients become non-finite even for basis-invariant expressions. Gradients depending only on <see cref="Values"/> are always finite. Nodes become invalid after <see cref="Tape.Reset"/>.
/// </para>
/// <para>
/// <see cref="Compute"/> で正方対称行列ノードから作成します。<see cref="Values"/> は昇順の固有値、<see cref="Vectors"/> は対応する正規直交固有ベクトルを列に持つ直交行列 <c>V</c> を表します。
/// 値は固有ベクトルを含む <see cref="SymmetricEigenDecomposition.Compute(Matrix{double}, bool, bool)"/> とビット単位で一致し、分解オブジェクトはテープが次数ごとにプールして再利用します。
/// </para>
/// <para>
/// 入力行列は下三角部分（対角を含む）のみを参照し、勾配も下三角部分にのみ加算します（上三角部分の勾配は 0。非対角要素の勾配は <c>(i, j)</c> と <c>(j, i)</c> の対称な変化に対する微分に対応します）。
/// <see cref="Vectors"/> を経由する勾配は固有値が互いに異なるときに定義され、重複固有値 <c>λᵢ = λⱼ</c> で <c>(VᵀV̄)ᵢⱼ ≠ 0</c> の場合は基底非依存の量であっても有限でない値になります。<see cref="Values"/> のみに依存する勾配は常に有限です。<see cref="Tape.Reset"/> 後はノードを使用できません。
/// </para>
/// </remarks>
public readonly struct SymmetricEigenDecompositionNode
{
    internal readonly Tape? Tape;
    internal readonly int Generation;
    internal readonly int Index;

    private SymmetricEigenDecompositionNode(Tape tape, int index)
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
    public int Order => Decomposition(Owner()).Values.Length;

    /// <summary>
    /// Gets the eigenvalues in ascending order.
    /// <para>昇順に並んだ固有値のベクトルノードを取得します。</para>
    /// </summary>
    /// <value>A vector node matching <see cref="SymmetricEigenDecomposition.Values"/> bit-for-bit. / <see cref="SymmetricEigenDecomposition.Values"/> とビット単位で一致するベクトルノード。</value>
    /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized or belongs to a stale tape generation. / ノードが未初期化、またはテープの世代が古い場合。</exception>
    public VectorNode<double> Values
    {
        get
        {
            Tape tape = Owner();
            int n = Decomposition(tape).Values.Length;
            return new(tape, ArrayView.Whole(tape.Store<double>().Decompositions.Info(Index).Factor, n, 1));
        }
    }

    /// <summary>
    /// Gets the orthogonal matrix <c>V</c> whose columns are the eigenvectors.
    /// <para>固有ベクトルを列とする直交行列 <c>V</c> のノードを取得します。</para>
    /// </summary>
    /// <value>A matrix node matching <see cref="SymmetricEigenDecomposition.Vectors"/> bit-for-bit. / <see cref="SymmetricEigenDecomposition.Vectors"/> とビット単位で一致する行列ノード。</value>
    /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized or belongs to a stale tape generation. / ノードが未初期化、またはテープの世代が古い場合。</exception>
    public MatrixNode<double> Vectors
    {
        get
        {
            Tape tape = Owner();
            int n = Decomposition(tape).Values.Length;
            return new(tape, ArrayView.Whole(tape.Store<double>().Decompositions.Info(Index).Second, n, n));
        }
    }

    /// <summary>
    /// Computes the eigendecomposition of a real symmetric matrix node.
    /// <para>実対称行列ノードの固有値分解を計算します。</para>
    /// </summary>
    /// <param name="a">The symmetric matrix node (lower triangle is read). / 対称行列ノード（下三角部分のみ参照）。</param>
    /// <returns>The symmetric eigendecomposition node. / 対称固有値分解ノード。</returns>
    /// <remarks>
    /// <para>
    /// Eigenvectors are always computed because they are required even for eigenvalue gradients. When <paramref name="a"/> requires a gradient, <see cref="Values"/> and <see cref="Vectors"/> are recorded together in a single record (skipped during backpropagation if neither is used).
    /// Backpropagation forms <c>G = VKVᵀ</c> with <c>C = VᵀV̄</c> and <c>K = diag(λ̄) + F∘C</c> (<c>F[i, j] = 1/(λⱼ - λᵢ)</c> for <c>i ≠ j</c>, 0 when <c>C[i, j] = 0</c>), then folds <c>G[i, j] + G[j, i]</c> into the strictly lower triangle and <c>G[i, i]</c> onto the diagonal.
    /// </para>
    /// <para>
    /// 固有値の勾配計算にも必要なため、固有ベクトルは常に計算されます。<paramref name="a"/> が勾配を必要とする場合、<see cref="Values"/> と <see cref="Vectors"/> は 1 つの記録にまとめて追加されます（どちらも未使用なら逆伝播時にスキップされます）。
    /// 局所微分は <c>C = VᵀV̄</c>、<c>K = diag(λ̄) + F∘C</c>（<c>i ≠ j</c> で <c>F[i, j] = 1/(λⱼ - λᵢ)</c>、<c>C[i, j] = 0</c> の要素は 0）として <c>G = VKVᵀ</c> を求め、下三角部分の非対角に <c>G[i, j] + G[j, i]</c>、対角に <c>G[i, i]</c> を加算します。
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if <paramref name="a"/> is not square. / 正方行列でない場合。</exception>
    /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
    public static SymmetricEigenDecompositionNode Compute(MatrixNode<double> a)
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
        int index = pool.TakeSymmetricEigen(a.Rows);
        var eigen = pool.SymmetricEigen(index);
        eigen.Recompute(store.MatrixView(a.View));
        ref DecompositionInfo info = ref pool.Info(index);
        info.Input = a.View;
        info.RequiresGradient = record;
        info.Factor = store.AddFactor(eigen.Values, record);
        info.Second = store.AddFactor(eigen.Vectors!, record);
        info.Third = -1;
        info.LogDeterminantIndex = -1;
        if (record)
        {
            tape.AppendArrayRecord<double>(ArrayOperation.SymmetricEigen, info.Factor, a.View, default, 0, index);
        }

        return new(tape, index);
    }

    private SymmetricEigenDecomposition Decomposition(Tape tape) => tape.Store<double>().Decompositions.SymmetricEigen(Index);

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
