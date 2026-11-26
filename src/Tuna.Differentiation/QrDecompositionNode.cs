using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using Tuna.LinearAlgebra;

namespace Tuna.Differentiation;

/// <summary>
/// Represents a thin QR decomposition <c>A = QR</c> recorded on a <see cref="Differentiation.Tape"/>.
/// <para><see cref="Differentiation.Tape"/> に記録された薄い QR 分解 <c>A = QR</c> を表すノードです。</para>
/// </summary>
/// <remarks>
/// <para>
/// Created from an <c>m × n</c> matrix node via <see cref="Compute"/> with <c>k = min(m, n)</c>: <see cref="Q"/> is <c>m × k</c> with orthonormal columns, and <see cref="R"/> is <c>k × n</c> upper trapezoidal.
/// Values match <see cref="QrDecomposition.Q"/> and <see cref="QrDecomposition.R"/> from <see cref="QrDecomposition.Compute(Matrix{double}, bool)"/> bit-for-bit; diagonal signs of <c>R</c> follow the core decomposition and need not be positive.
/// The underlying decomposition is pooled per matrix shape by the tape for reuse across recordings. Gradients are well-defined when the leading <c>k</c> columns of <c>A</c> are linearly independent (nonzero diagonal of <c>R</c>); otherwise gradients may be non-finite. Nodes become invalid after <see cref="Tape.Reset"/>.
/// </para>
/// <para>
/// <see cref="Compute"/> で <c>m × n</c> 行列ノードから作成します。<c>k = min(m, n)</c> として、<see cref="Q"/> は正規直交列を持つ <c>m × k</c> 行列、<see cref="R"/> は <c>k × n</c> の上台形行列になります。
/// 値は <see cref="QrDecomposition.Compute(Matrix{double}, bool)"/> が返す <see cref="QrDecomposition.Q"/>・<see cref="QrDecomposition.R"/> とビット単位で一致します。<c>R</c> の対角成分の符号はコアの分解に従い、正とは限りません（局所微分はこの符号のもとで計算されます）。
/// 分解オブジェクトはテープが形状ごとにプールして再利用します。勾配は <c>R</c> の対角成分がすべて非ゼロ（<c>A</c> の先頭 <c>k</c> 列が線形独立）のときに定義され、対角に 0 がある場合は有限でない値になり得ます。<see cref="Tape.Reset"/> 後はノードを使用できません。
/// </para>
/// </remarks>
public readonly struct QrDecompositionNode
{
    internal readonly Tape? Tape;
    internal readonly int Generation;
    internal readonly int Index;

    private QrDecompositionNode(Tape tape, int index)
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
    public int Rows => Decomposition(Owner()).Rows;

    /// <summary>
    /// Gets the column count <c>n</c> of the factored matrix.
    /// <para>元の行列の列数 <c>n</c> を取得します。</para>
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized or belongs to a stale tape generation. / ノードが未初期化、またはテープの世代が古い場合。</exception>
    public int Columns => Decomposition(Owner()).Columns;

    /// <summary>
    /// Gets the <c>m × min(m, n)</c> matrix <c>Q</c> with orthonormal columns.
    /// <para>正規直交列を持つ <c>m × min(m, n)</c> 行列 <c>Q</c> のノードを取得します。</para>
    /// </summary>
    /// <value>A matrix node matching <see cref="QrDecomposition.Q"/> bit-for-bit. / <see cref="QrDecomposition.Q"/> とビット単位で一致する行列ノード。</value>
    /// <remarks>
    /// <para>Backpropagated jointly with <see cref="R"/>; see <see cref="Compute"/> for the derivative formula.</para>
    /// <para><see cref="R"/> とまとめて逆伝播されます（微分式は <see cref="Compute"/> を参照してください）。</para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized or belongs to a stale tape generation. / ノードが未初期化、またはテープの世代が古い場合。</exception>
    public MatrixNode<double> Q
    {
        get
        {
            Tape tape = Owner();
            var qr = Decomposition(tape);
            return new(tape, ArrayView.Whole(tape.Store<double>().Decompositions.Info(Index).Factor, qr.Rows, Math.Min(qr.Rows, qr.Columns)));
        }
    }

    /// <summary>
    /// Gets the <c>min(m, n) × n</c> upper-trapezoidal matrix <c>R</c>.
    /// <para><c>min(m, n) × n</c> の上台形行列 <c>R</c> のノードを取得します。</para>
    /// </summary>
    /// <value>A matrix node matching <see cref="QrDecomposition.R"/> bit-for-bit (strictly lower triangle is 0). / <see cref="QrDecomposition.R"/> とビット単位で一致する行列ノード（対角より下は 0）。</value>
    /// <remarks>
    /// <para>Entries below the diagonal are constant zeros and their adjoints are ignored.</para>
    /// <para>対角より下の要素は定数 0 であり、その随伴は使用しません。</para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized or belongs to a stale tape generation. / ノードが未初期化、またはテープの世代が古い場合。</exception>
    public MatrixNode<double> R
    {
        get
        {
            Tape tape = Owner();
            var qr = Decomposition(tape);
            return new(tape, ArrayView.Whole(tape.Store<double>().Decompositions.Info(Index).Second, Math.Min(qr.Rows, qr.Columns), qr.Columns));
        }
    }

    /// <summary>
    /// Computes the thin QR decomposition of a matrix node.
    /// <para>行列ノードの薄い QR 分解を計算します。</para>
    /// </summary>
    /// <param name="a">The input matrix node. / 入力行列ノード。</param>
    /// <returns>The QR decomposition node. / QR 分解ノード。</returns>
    /// <remarks>
    /// <para>
    /// When <paramref name="a"/> requires a gradient, <see cref="Q"/> and <see cref="R"/> are recorded together in a single record (skipped during backpropagation if neither is used).
    /// For <c>m ≥ n</c>, <c>Ā = (Q̄ + Q·copyltu(M))R⁻ᵀ</c> with <c>M = R·triu(R̄)ᵀ - Q̄ᵀQ</c>, where <c>copyltu(M)</c> symmetrizes <c>M</c> by copying its lower triangle to its upper triangle.
    /// For <c>m &lt; n</c>, partitioning <c>A = [X | Y]</c> and <c>R = [U | V]</c> (<c>U</c> is <c>m × m</c>) yields <c>Ȳ = QV̄</c> and computes <c>X̄</c> with the square formula using <c>Q̄ + YV̄ᵀ</c>.
    /// </para>
    /// <para>
    /// <paramref name="a"/> が勾配を必要とする場合、<see cref="Q"/> と <see cref="R"/> は 1 つの記録にまとめて追加されます（どちらも未使用なら逆伝播時にスキップされます）。
    /// 局所微分は、<c>m ≥ n</c> では <c>M = R·triu(R̄)ᵀ - Q̄ᵀQ</c> として <c>Ā = (Q̄ + Q·copyltu(M))R⁻ᵀ</c>（<c>copyltu(M)</c> は下三角部分を上三角部分へ複製した対称行列）となります。
    /// <c>m &lt; n</c> では <c>A = [X | Y]</c>、<c>R = [U | V]</c>（<c>U</c> は <c>m × m</c>）と分割し、<c>Ȳ = QV̄</c>、<c>X̄</c> は <c>Q̄</c> を <c>Q̄ + YV̄ᵀ</c> に置き換えた正方行列の式で求めます。
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
    public static QrDecompositionNode Compute(MatrixNode<double> a)
    {
        Tape tape = a.Owner();
        var store = tape.Store<double>();
        bool record = store.RequiresGradient(a.View.Slot);
        tape.EnsureCanApply(record);

        var pool = store.Decompositions;
        int index = pool.TakeQr(a.Rows, a.Columns);
        var qr = pool.Qr(index);
        qr.Recompute(store.MatrixView(a.View));
        ref DecompositionInfo info = ref pool.Info(index);
        info.Input = a.View;
        info.RequiresGradient = record;
        info.Factor = store.AddFactor(qr.Q, record);
        info.Second = store.AddFactor(qr.R, record);
        info.Third = -1;
        info.LogDeterminantIndex = -1;
        if (record)
        {
            tape.AppendArrayRecord<double>(ArrayOperation.Qr, info.Factor, a.View, default, 0, index);
        }

        return new(tape, index);
    }

    private QrDecomposition Decomposition(Tape tape) => tape.Store<double>().Decompositions.Qr(Index);

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
