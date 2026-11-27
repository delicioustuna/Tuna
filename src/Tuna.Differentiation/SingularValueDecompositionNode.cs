using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using Tuna.LinearAlgebra;

namespace Tuna.Differentiation;

/// <summary>
/// Represents a thin singular value decomposition <c>A = UΣVᵀ</c> recorded on a <see cref="Differentiation.Tape"/>.
/// <para><see cref="Differentiation.Tape"/> に記録された薄い特異値分解 <c>A = UΣVᵀ</c> を表すノードです。</para>
/// </summary>
/// <remarks>
/// <para>
/// Created from an <c>m × n</c> matrix node via <see cref="Compute"/> with <c>k = min(m, n)</c>: <see cref="Values"/> contains non-negative singular values in descending order, while <see cref="U"/> (<c>m × k</c>) and <see cref="V"/> (<c>n × k</c>) have orthonormal columns.
/// Values match <see cref="SingularValueDecomposition.Compute(Matrix{double}, bool, bool)"/> (with vectors) bit-for-bit, and the underlying decomposition is pooled per matrix shape by the tape for reuse across recordings.
/// </para>
/// <para>
/// Singular vector signs and bases for repeated singular subspaces follow the core decomposition. Gradients flowing through <see cref="U"/> or <see cref="V"/> require distinct nonzero singular values (repeated singular values, or zero singular values when <c>m ≠ n</c>, can produce non-finite gradients); gradients depending only on <see cref="Values"/> (<c>U·diag(σ̄)·Vᵀ</c>) are always finite. Nodes become invalid after <see cref="Tape.Reset"/>.
/// </para>
/// <para>
/// <see cref="Compute"/> で <c>m × n</c> 行列ノードから作成します。<c>k = min(m, n)</c> として、<see cref="Values"/> は降順・非負の特異値、<see cref="U"/>（<c>m × k</c>）と <see cref="V"/>（<c>n × k</c>）は正規直交列を持ちます。
/// 値は特異ベクトルを含む <see cref="SingularValueDecomposition.Compute(Matrix{double}, bool, bool)"/> とビット単位で一致し、分解オブジェクトはテープが形状ごとにプールして再利用します。
/// </para>
/// <para>
/// 特異ベクトルの符号（<c>uᵢ</c> と <c>vᵢ</c> の同時反転）および重複特異値の特異空間の基底はコアの分解に従います。<see cref="U"/>・<see cref="V"/> を経由する勾配は特異値が互いに異なり非ゼロのときに定義され、重複特異値や（<c>m ≠ n</c> での）0 の特異値では有限でない値になり得ます。<see cref="Values"/> のみに依存する勾配 <c>U·diag(σ̄)·Vᵀ</c> は常に有限です。<see cref="Tape.Reset"/> 後はノードを使用できません。
/// </para>
/// </remarks>
public readonly struct SingularValueDecompositionNode
{
    internal readonly Tape? Tape;
    internal readonly int Generation;
    internal readonly int Index;

    private SingularValueDecompositionNode(Tape tape, int index)
    {
        Tape = tape;
        Generation = tape.Generation;
        Index = index;
    }

    /// <summary>
    /// Gets the row count <c>m</c> of the factored matrix.
    /// <para>元の行列の行数 <c>m</c> を取得します。</para>
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized or belongs to a stale tape generation. / ノードが未初期化、またはテープの世代が古い場合。</exception>
    public int Rows => Decomposition(Owner()).U!.Rows;

    /// <summary>
    /// Gets the column count <c>n</c> of the factored matrix.
    /// <para>元の行列の列数 <c>n</c> を取得します。</para>
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized or belongs to a stale tape generation. / ノードが未初期化、またはテープの世代が古い場合。</exception>
    public int Columns => Decomposition(Owner()).V!.Rows;

    /// <summary>
    /// Gets the non-negative singular values in descending order.
    /// <para>降順かつ非負の特異値のベクトルノードを取得します。</para>
    /// </summary>
    /// <value>A vector node matching <see cref="SingularValueDecomposition.Values"/> bit-for-bit. / <see cref="SingularValueDecomposition.Values"/> とビット単位で一致するベクトルノード。</value>
    /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized or belongs to a stale tape generation. / ノードが未初期化、またはテープの世代が古い場合。</exception>
    public VectorNode<double> Values
    {
        get
        {
            Tape tape = Owner();
            int k = Decomposition(tape).Values.Length;
            return new(tape, ArrayView.Whole(tape.Store<double>().Decompositions.Info(Index).Factor, k, 1));
        }
    }

    /// <summary>
    /// Gets the <c>m × min(m, n)</c> matrix <c>U</c> whose columns are the left singular vectors.
    /// <para>左特異ベクトルを列とする <c>m × min(m, n)</c> 行列 <c>U</c> のノードを取得します。</para>
    /// </summary>
    /// <value>A matrix node matching <see cref="SingularValueDecomposition.U"/> bit-for-bit. / <see cref="SingularValueDecomposition.U"/> とビット単位で一致する行列ノード。</value>
    /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized or belongs to a stale tape generation. / ノードが未初期化、またはテープの世代が古い場合。</exception>
    public MatrixNode<double> U
    {
        get
        {
            Tape tape = Owner();
            var u = Decomposition(tape).U!;
            return new(tape, ArrayView.Whole(tape.Store<double>().Decompositions.Info(Index).Second, u.Rows, u.Columns));
        }
    }

    /// <summary>
    /// Gets the <c>n × min(m, n)</c> matrix <c>V</c> whose columns are the right singular vectors.
    /// <para>右特異ベクトルを列とする <c>n × min(m, n)</c> 行列 <c>V</c> のノードを取得します。</para>
    /// </summary>
    /// <value>A matrix node matching <see cref="SingularValueDecomposition.V"/> bit-for-bit. / <see cref="SingularValueDecomposition.V"/> とビット単位で一致する行列ノード。</value>
    /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized or belongs to a stale tape generation. / ノードが未初期化、またはテープの世代が古い場合。</exception>
    public MatrixNode<double> V
    {
        get
        {
            Tape tape = Owner();
            var v = Decomposition(tape).V!;
            return new(tape, ArrayView.Whole(tape.Store<double>().Decompositions.Info(Index).Third, v.Rows, v.Columns));
        }
    }

    /// <summary>
    /// Returns the minimum-norm least-squares solution for a vector right-hand side.
    /// <para>ベクトル右辺に対して、最小二乗解のうちノルムが最小の解を返します。</para>
    /// </summary>
    /// <param name="b">A right-hand side vector node with length <see cref="Rows"/>. / 長さが <see cref="Rows"/> の右辺ベクトルノード。</param>
    /// <param name="tolerance">The relative cutoff tolerance, as in <see cref="SingularValueDecomposition.Solve(ReadOnlyVectorSpan{double}, double?)"/>. / <see cref="SingularValueDecomposition.Solve(ReadOnlyVectorSpan{double}, double?)"/> と同じカットオフの相対許容誤差。</param>
    /// <returns>The solution vector node with length <see cref="Columns"/>. / 長さが <see cref="Columns"/> の解ベクトルノード。</returns>
    /// <remarks>
    /// <para>The value matches the core solve bit-for-bit. Backpropagation uses the Golub–Pereyra fixed-rank derivative at the truncated matrix, using the rank retained in the forward pass. The value can be discontinuous when a singular value crosses the cutoff.</para>
    /// <para>値はコアの求解結果とビット単位で一致します。順伝播で保持したランクを使い、切り捨て後の行列に対して Golub–Pereyra の定ランクの微分を計算します。特異値がカットオフをまたぐ点では値が不連続になり得ます。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">The length of <paramref name="b"/> does not match <see cref="Rows"/>. / <paramref name="b"/> の長さが <see cref="Rows"/> と一致しない場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="tolerance"/> is negative or non-finite. / <paramref name="tolerance"/> が負または有限でない場合。</exception>
    /// <exception cref="InvalidOperationException">The singular spectrum is invalid, a node is uninitialized or stale, the nodes belong to different tapes, or the tape has finished backpropagation. / 特異値が無効、ノードが未初期化または世代が古い、ノードが異なるテープに属する、あるいは逆伝播済みのテープへ記録しようとした場合。</exception>
    public VectorNode<double> Solve(VectorNode<double> b, double? tolerance = null)
    {
        Tape tape = XMathExtensions.Owner(Owner(), Generation, b.Tape, b.Generation);
        return new(tape, SolveCore(tape, b.View, tolerance));
    }

    /// <summary>
    /// Returns the minimum-norm least-squares solutions for the columns of a matrix right-hand side.
    /// <para>右辺行列の列ごとに、最小二乗解のうちノルムが最小の解を返します。</para>
    /// </summary>
    /// <param name="b">A right-hand side matrix node with <see cref="Rows"/> rows. / 行数が <see cref="Rows"/> の右辺行列ノード。</param>
    /// <param name="tolerance">The relative cutoff tolerance, as in <see cref="Solve(VectorNode{double}, double?)"/>. / <see cref="Solve(VectorNode{double}, double?)"/> と同じカットオフの相対許容誤差。</param>
    /// <returns>The solution matrix node, with <see cref="Columns"/> rows and the same number of columns as <paramref name="b"/>. / 行数が <see cref="Columns"/> で、<paramref name="b"/> と同じ列数を持つ解行列ノード。</returns>
    /// <remarks>
    /// <para>The fixed-rank derivative and cutoff behavior are the same as for <see cref="Solve(VectorNode{double}, double?)"/>.</para>
    /// <para>定ランクの微分とカットオフでの挙動は、<see cref="Solve(VectorNode{double}, double?)"/> と同じです。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">The row count of <paramref name="b"/> does not match <see cref="Rows"/>. / <paramref name="b"/> の行数が <see cref="Rows"/> と一致しない場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="tolerance"/> is negative or non-finite. / <paramref name="tolerance"/> が負または有限でない場合。</exception>
    /// <exception cref="InvalidOperationException">The singular spectrum is invalid, a node is uninitialized or stale, the nodes belong to different tapes, or the tape has finished backpropagation. / 特異値が無効、ノードが未初期化または世代が古い、ノードが異なるテープに属する、あるいは逆伝播済みのテープへ記録しようとした場合。</exception>
    public MatrixNode<double> Solve(MatrixNode<double> b, double? tolerance = null)
    {
        Tape tape = XMathExtensions.Owner(Owner(), Generation, b.Tape, b.Generation);
        return new(tape, SolveCore(tape, b.View, tolerance));
    }

    /// <summary>
    /// Returns the Moore–Penrose pseudoinverse computed with the retained singular values.
    /// <para>保持した特異値を使ってムーア・ペンローズ疑似逆行列を求めます。</para>
    /// </summary>
    /// <param name="tolerance">The relative cutoff tolerance, as in <see cref="Solve(VectorNode{double}, double?)"/>. / <see cref="Solve(VectorNode{double}, double?)"/> と同じカットオフの相対許容誤差。</param>
    /// <returns>The <c>n × m</c> pseudoinverse matrix node for an <c>m × n</c> input. / 入力が <c>m × n</c> の場合の <c>n × m</c> の疑似逆行列ノード。</returns>
    /// <remarks>
    /// <para>The value matches the core pseudoinverse bit-for-bit. Backpropagation uses the Golub–Pereyra fixed-rank derivative at the truncated matrix, including repeated retained singular values. The value can be discontinuous at the cutoff.</para>
    /// <para>値はコアの疑似逆行列とビット単位で一致します。逆伝播では切り捨て後の行列に対して Golub–Pereyra の定ランクの微分を使います。保持した特異値が重複する場合にも適用できます。カットオフの境界では値が不連続になり得ます。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="tolerance"/> is negative or non-finite. / <paramref name="tolerance"/> が負または有限でない場合。</exception>
    /// <exception cref="InvalidOperationException">The singular spectrum is invalid, the node is uninitialized or stale, or the tape has finished backpropagation. / 特異値が無効、ノードが未初期化または世代が古い、あるいは逆伝播済みのテープへ記録しようとした場合。</exception>
    public MatrixNode<double> PseudoInverse(double? tolerance = null)
    {
        Tape tape = Owner();
        var store = tape.Store<double>();
        var pool = store.Decompositions;
        var info = pool.Info(Index);
        tape.EnsureCanApply(info.RequiresGradient);

        int m = info.Input.Rows, n = info.Input.Columns;
        int slot = store.AddPooled(n, m, info.RequiresGradient);
        var svd = pool.SingularValue(Index);
        try
        {
            svd.PseudoInverse(store.WritableMatrix(slot), tolerance);
            if (info.RequiresGradient)
            {
                tape.AppendArrayRecord<double>(ArrayOperation.SvdPseudoInverse, slot, info.Input, default, RetainedRank(svd, tolerance), Index);
            }
        }
        catch
        {
            store.DiscardOperationSlots(slot);
            throw;
        }

        return new(tape, ArrayView.Whole(slot, n, m));
    }

    private ArrayView SolveCore(Tape tape, in ArrayView b, double? tolerance)
    {
        var store = tape.Store<double>();
        var pool = store.Decompositions;
        var info = pool.Info(Index);
        if (b.Rows != info.Input.Rows)
        {
            throw new ArgumentException("The right-hand side must have as many rows as the original matrix.", nameof(b));
        }

        bool record = info.RequiresGradient || store.RequiresGradient(b.Slot);
        tape.EnsureCanApply(record);
        int n = info.Input.Columns;
        int slot = store.AddPooled(n, b.Columns, record);
        var svd = pool.SingularValue(Index);
        try
        {
            svd.Solve(store.MatrixView(b), store.WritableMatrix(slot), tolerance);
            if (record)
            {
                tape.AppendArrayRecord<double>(ArrayOperation.SvdSolve, slot, info.Input, b, RetainedRank(svd, tolerance), Index);
            }
        }
        catch
        {
            store.DiscardOperationSlots(slot);
            throw;
        }

        return ArrayView.Whole(slot, n, b.Columns);
    }

    private static int RetainedRank(SingularValueDecomposition svd, double? tolerance)
    {
        var values = svd.Values;
        if (tolerance is not { } relativeTolerance || values.Length == 0)
        {
            return svd.Rank;
        }

        // Match the core cutoff, including the rescaling that avoids a rounded subnormal threshold.
        double cutoff = relativeTolerance * values[0];
        int shift = double.IsSubnormal(cutoff) ? 54 : 0;
        if (shift != 0)
        {
            cutoff = relativeTolerance * Math.ScaleB(values[0], shift);
        }

        int rank = 0;
        while (rank < values.Length && (shift == 0 ? values[rank] : Math.ScaleB(values[rank], shift)) > cutoff)
        {
            rank++;
        }

        return rank;
    }

    /// <summary>
    /// Computes the thin singular value decomposition of a matrix node.
    /// <para>行列ノードの薄い特異値分解を計算します。</para>
    /// </summary>
    /// <param name="a">The input matrix node. / 入力行列ノード。</param>
    /// <returns>The singular value decomposition node. / 特異値分解ノード。</returns>
    /// <remarks>
    /// <para>
    /// Singular vectors are always computed because they are required even for singular-value gradients. When <paramref name="a"/> requires a gradient, <see cref="Values"/>, <see cref="U"/>, and <see cref="V"/> are recorded together in a single record (skipped during backpropagation if none are used).
    /// Backpropagation follows Townsend (2016): with <c>P = UᵀŪ</c>, <c>W = VᵀV̄</c>, and <c>F[i, j] = 1/(σⱼ² - σᵢ²)</c> (<c>i ≠ j</c>, 0 where the numerator is 0), <c>Ā = U[(F∘(P - Pᵀ))Σ + diag(σ̄) + Σ(F∘(W - Wᵀ))]Vᵀ + (I - UUᵀ)ŪΣ⁻¹Vᵀ + UΣ⁻¹V̄ᵀ(I - VVᵀ)</c> (omitting the second term when <c>m = k</c> and the third when <c>n = k</c>).
    /// </para>
    /// <para>
    /// 特異値の勾配計算にも必要なため、特異ベクトルは常に計算されます。<paramref name="a"/> が勾配を必要とする場合、<see cref="Values"/>・<see cref="U"/>・<see cref="V"/> は 1 つの記録にまとめて追加されます（いずれも未使用なら逆伝播時にスキップされます）。
    /// 局所微分は Townsend (2016) の式に従い、<c>P = UᵀŪ</c>、<c>W = VᵀV̄</c>、<c>F[i, j] = 1/(σⱼ² - σᵢ²)</c>（<c>i ≠ j</c>、分子が 0 の要素は 0）として <c>Ā = U[(F∘(P - Pᵀ))Σ + diag(σ̄) + Σ(F∘(W - Wᵀ))]Vᵀ + (I - UUᵀ)ŪΣ⁻¹Vᵀ + UΣ⁻¹V̄ᵀ(I - VVᵀ)</c> を計算します（<c>m = k</c> では第 2 項、<c>n = k</c> では第 3 項が 0 のため省略します）。
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
    public static SingularValueDecompositionNode Compute(MatrixNode<double> a)
    {
        Tape tape = a.Owner();
        var store = tape.Store<double>();
        bool record = store.RequiresGradient(a.View.Slot);
        tape.EnsureCanApply(record);

        var pool = store.Decompositions;
        int index = pool.TakeSingularValue(a.Rows, a.Columns);
        var svd = pool.SingularValue(index);
        svd.Recompute(store.MatrixView(a.View));
        ref DecompositionInfo info = ref pool.Info(index);
        info.Input = a.View;
        info.RequiresGradient = record;
        info.Factor = store.AddFactor(svd.Values, record);
        info.Second = store.AddFactor(svd.U!, record);
        info.Third = store.AddFactor(svd.V!, record);
        info.LogDeterminantIndex = -1;
        if (record)
        {
            tape.AppendArrayRecord<double>(ArrayOperation.SingularValueDecomposition, info.Factor, a.View, default, 0, index);
        }

        return new(tape, index);
    }

    private SingularValueDecomposition Decomposition(Tape tape) => tape.Store<double>().Decompositions.SingularValue(Index);

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
