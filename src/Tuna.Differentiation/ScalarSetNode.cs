using System.Numerics;
using Tuna.LinearAlgebra;

namespace Tuna.Differentiation;

/// <summary>
/// Represents a shared scalar or a vector containing one scalar per point.
/// <para>全点で共通のスカラー、または点ごとのスカラーを格納するベクトルを表します。</para>
/// </summary>
/// <typeparam name="T">The element type, either <see langword="float"/> or <see langword="double"/>. / 要素型。<see langword="float"/> または <see langword="double"/>。</typeparam>
/// <remarks>
/// <para>The default value represents a shared zero. For <see langword="float"/> inputs, a shared <see cref="ScalarNode"/> is rounded to <see langword="float"/>.</para>
/// <para>既定値は全点で共通の定数 0 です。要素型が <see langword="float"/> の場合、共通の <see cref="ScalarNode"/> の値は <see langword="float"/> に丸めます。</para>
/// </remarks>
public readonly struct ScalarSetNode<T> where T : unmanaged, IFloatingPointIeee754<T>
{
    private readonly ScalarNode scalar;
    private readonly VectorNode<T> vector;
    private readonly bool perPoint;

    private ScalarSetNode(ScalarNode scalar) => this.scalar = scalar;

    private ScalarSetNode(VectorNode<T> vector)
    {
        this.vector = vector;
        perPoint = true;
    }

    /// <summary>
    /// Uses a constant for all points.
    /// <para>全点で共通の定数を使います。</para>
    /// </summary>
    /// <param name="value">The constant value. / 定数値。</param>
    public static implicit operator ScalarSetNode<T>(T value) => new((ScalarNode)double.CreateTruncating(value));

    /// <summary>
    /// Uses a scalar node for all points.
    /// <para>全点で共通のスカラーノードを使います。</para>
    /// </summary>
    /// <param name="value">The shared scalar node. / 共通のスカラーノード。</param>
    public static implicit operator ScalarSetNode<T>(ScalarNode value) => new(value);

    /// <summary>
    /// Uses one vector element per point.
    /// <para>各点にベクトルの要素を 1 つずつ対応させます。</para>
    /// </summary>
    /// <param name="value">The vector node with one element per point. / 点ごとのスカラーを格納するベクトルノード。</param>
    public static implicit operator ScalarSetNode<T>(VectorNode<T> value) => new(value);

    internal BoundScalarSet<T> Bind(int length, scoped ref Tape? tape, string parameterName)
    {
        TapeElementTypes.Require<T>();
        if (!perPoint)
        {
            return new(default, ArrayStore<T>.FromDouble(scalar.Value), PointwiseInput.Bind(scalar, ref tape));
        }

        var input = PointwiseInput.Bind(vector, PointwiseInputKind.PointVector, ref tape);
        if (vector.Length != length)
        {
            throw new ArgumentException("The vector length must match the number of points.", parameterName);
        }

        return new(vector.Value, null, input);
    }
}

internal readonly ref struct BoundScalarSet<T>(ReadOnlyVectorSpan<T> vector, T? sharedValue, PointwiseInput input)
    where T : unmanaged, IFloatingPointIeee754<T>
{
    private readonly ReadOnlyVectorSpan<T> values = vector;

    internal T? SharedValue { get; } = sharedValue;
    internal PointwiseInput Input { get; } = input;
    internal T this[int index] => SharedValue ?? values[index];
}
