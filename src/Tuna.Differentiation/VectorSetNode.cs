using System.Numerics;
using Tuna.LinearAlgebra;

namespace Tuna.Differentiation;

/// <summary>
/// Represents a shared vector or a matrix containing one vector per point.
/// <para>全点で共通のベクトル、または点ごとのベクトルを格納する行列を表します。</para>
/// </summary>
/// <typeparam name="T">The element type, either <see langword="float"/> or <see langword="double"/>. / 要素型。<see langword="float"/> または <see langword="double"/>。</typeparam>
/// <remarks>
/// <para>When a matrix is supplied, its columns correspond to points in order.</para>
/// <para>行列を指定する場合、各列は入力順に各点に対応します。</para>
/// </remarks>
public readonly struct VectorSetNode<T> where T : unmanaged, IFloatingPointIeee754<T>
{
    private readonly VectorNode<T> vector;
    private readonly MatrixNode<T> matrix;
    private readonly bool perPoint;

    private VectorSetNode(VectorNode<T> vector) => this.vector = vector;

    private VectorSetNode(MatrixNode<T> matrix)
    {
        this.matrix = matrix;
        perPoint = true;
    }

    /// <summary>
    /// Uses a vector node for all points.
    /// <para>全点で共通のベクトルノードを使います。</para>
    /// </summary>
    /// <param name="value">The shared vector node. / 共通のベクトルノード。</param>
    public static implicit operator VectorSetNode<T>(VectorNode<T> value) => new(value);

    /// <summary>
    /// Uses one column of a matrix node per point.
    /// <para>各点に行列ノードの列を 1 つずつ対応させます。</para>
    /// </summary>
    /// <param name="value">The matrix node with one column per point. / 点ごとのベクトルを列に格納する行列ノード。</param>
    public static implicit operator VectorSetNode<T>(MatrixNode<T> value) => new(value);

    internal BoundVectorSet<T> Bind(int length, scoped ref Tape? tape, string parameterName)
    {
        if (!perPoint)
        {
            var shared = PointwiseInput.Bind(vector, PointwiseInputKind.SharedArray, ref tape);
            return new(vector.Value, default, shared, isShared: true);
        }

        var input = PointwiseInput.Bind(matrix, PointwiseInputKind.PointColumns, ref tape);
        if (matrix.Columns != length)
        {
            throw new ArgumentException("The matrix column count must match the number of points.", parameterName);
        }

        return new(default, matrix.Value, input, isShared: false);
    }
}

internal readonly ref struct BoundVectorSet<T>(ReadOnlyVectorSpan<T> shared, ReadOnlyMatrixSpan<T> columns, PointwiseInput input, bool isShared)
    where T : unmanaged, IFloatingPointIeee754<T>
{
    private readonly ReadOnlyMatrixSpan<T> perPoint = columns;

    internal ReadOnlyVectorSpan<T> SharedValue { get; } = shared;
    internal bool IsShared { get; } = isShared;
    internal int VectorLength => IsShared ? SharedValue.Length : perPoint.Rows;
    internal PointwiseInput Input { get; } = input;
    internal ReadOnlyVectorSpan<T> this[int index] => IsShared ? SharedValue : perPoint.Column(index);
}
