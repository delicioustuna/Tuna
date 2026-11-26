using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;

namespace Tuna.Differentiation;

/// <summary>
/// Represents a scalar node for reverse-mode automatic differentiation. A node without an associated <see cref="Tuna.Differentiation.Tape"/> is an unrecorded constant.
/// <para>逆モード自動微分のスカラーノードを表します。<see cref="Tuna.Differentiation.Tape"/> に属さない値は記録されない定数として扱われます。</para>
/// </summary>
/// <remarks>
/// <para>
/// Nodes created before <see cref="Tuna.Differentiation.Tape.Reset"/> become invalid and cannot be used. Comparison operators compare values only and do not propagate gradients.
/// </para>
/// <para>
/// <see cref="Tuna.Differentiation.Tape.Reset"/> より前に作成したノードは無効になり使用できません。比較演算子は値のみを比較し、勾配は伝播しません。
/// </para>
/// </remarks>
public readonly struct ScalarNode
{
    private readonly double value;
    internal readonly Tape? Tape;
    internal readonly int Index;
    internal readonly int Generation;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal ScalarNode(double value, Tape? tape = null, int index = 0, int generation = 0)
    {
        this.value = value;
        Tape = tape;
        Index = index;
        Generation = generation;
    }

    /// <summary>
    /// Gets the scalar value of the node.
    /// <para>ノードのスカラー値を取得します。</para>
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown if the node was created before <see cref="Tuna.Differentiation.Tape.Reset"/>. / <see cref="Tuna.Differentiation.Tape.Reset"/> より前に作成したノードの場合。</exception>
    public double Value
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            Validate();
            return value;
        }
    }

    /// <summary>
    /// Gets a value indicating whether this node depends on any differentiable leaf.
    /// <para>このノードが微分対象の葉に依存しているかどうかを示す値を取得します。</para>
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown if the node was created before <see cref="Tuna.Differentiation.Tape.Reset"/>. / <see cref="Tuna.Differentiation.Tape.Reset"/> より前に作成したノードの場合。</exception>
    public bool RequiresGradient
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            Validate();
            return Tape is not null;
        }
    }

    /// <summary>
    /// Implicitly converts a <see langword="double"/> value to an unrecorded constant <see cref="ScalarNode"/>.
    /// <para><see langword="double"/> 値を記録されない定数 <see cref="ScalarNode"/> に暗黙的に変換します。</para>
    /// </summary>
    /// <param name="value">The constant value. / 定数値。</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static implicit operator ScalarNode(double value) => new(value);

    /// <summary>
    /// Records a custom scalar operation with user-supplied local derivatives.
    /// <para>ユーザー定義のスカラー演算と局所微分をテープに記録します。</para>
    /// </summary>
    /// <param name="value">The output value of the operation. / 演算結果の値。</param>
    /// <param name="inputs">Pairs of input nodes and their corresponding local partial derivatives. / 入力ノードと対応する局所偏微分の組。</param>
    /// <returns>A scalar node with the specified value, or an unrecorded constant if all inputs are constants. / 指定した値を持つスカラーノード。すべての入力が定数の場合は定数ノード。</returns>
    /// <remarks>
    /// <para>
    /// If all inputs are constants, nothing is recorded on the tape. Passing the same input multiple times accumulates each corresponding local derivative.
    /// The caller is responsible for the correctness and domain validity of the supplied derivatives. The <paramref name="inputs"/> span is not retained.
    /// </para>
    /// <para>
    /// すべての入力が定数の場合はテープへ記録しません。同じ入力を複数回渡すと、それぞれの局所微分が加算されます。
    /// 局所微分の正当性と定義域の整合性は呼び出し側が保証します。<paramref name="inputs"/> のスパンは保持されません。
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown if any input belongs to an earlier generation or a different tape, or if backpropagation has already completed on the tape. / 世代が古い、異なるテープに属している、または逆伝播完了後のテープに記録しようとした場合。</exception>
    public static ScalarNode Record(double value, ReadOnlySpan<(ScalarNode Input, double Derivative)> inputs)
    {
        Tape? tape = null;
        foreach (var (input, _) in inputs)
        {
            tape = IncludeInputTape(tape, input);
        }

        return tape is null ? new(value) : tape.Record(value, inputs);
    }

    internal static ScalarNode Record(double value, ReadOnlySpan<ScalarNode> inputs, ReadOnlySpan<double> localDerivatives)
    {
        if (inputs.Length != localDerivatives.Length)
        {
            throw new ArgumentException("Input and derivative lengths must match.", nameof(localDerivatives));
        }

        Tape? tape = null;
        foreach (var input in inputs)
        {
            tape = IncludeInputTape(tape, input);
        }

        return tape is null ? new(value) : tape.Record(value, inputs, localDerivatives);
    }

    private static Tape? IncludeInputTape(Tape? tape, ScalarNode input)
    {
        input.Validate();
        if (input.Tape is { } owner)
        {
            if (tape is not null && tape != owner)
            {
                ThrowDifferentTapes();
            }

            return owner;
        }

        return tape;
    }

    /// <summary>
    /// Adds two scalar nodes.
    /// <para>2 つのスカラーノードの和を計算します。</para>
    /// </summary>
    /// <param name="left">The left operand. / 左辺。</param>
    /// <param name="right">The right operand. / 右辺。</param>
    /// <returns>The sum <c>left + right</c>. / 和 <c>left + right</c>。</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ScalarNode operator +(ScalarNode left, ScalarNode right)
        => Binary(left.value + right.value, left, 1, right, 1);

    /// <summary>
    /// Subtracts one scalar node from another.
    /// <para>2 つのスカラーノードの差を計算します。</para>
    /// </summary>
    /// <param name="left">The left operand. / 左辺。</param>
    /// <param name="right">The right operand. / 右辺。</param>
    /// <returns>The difference <c>left - right</c>. / 差 <c>left - right</c>。</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ScalarNode operator -(ScalarNode left, ScalarNode right)
        => Binary(left.value - right.value, left, 1, right, -1);

    /// <summary>
    /// Multiplies two scalar nodes.
    /// <para>2 つのスカラーノードの積を計算します。</para>
    /// </summary>
    /// <param name="left">The left operand. / 左辺。</param>
    /// <param name="right">The right operand. / 右辺。</param>
    /// <returns>The product <c>left * right</c>. / 積 <c>left * right</c>。</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ScalarNode operator *(ScalarNode left, ScalarNode right)
        => Binary(left.value * right.value, left, right.value, right, left.value);

    /// <summary>
    /// Divides one scalar node by another.
    /// <para>2 つのスカラーノードの商を計算します。</para>
    /// </summary>
    /// <param name="left">The dividend. / 被除数。</param>
    /// <param name="right">The divisor. / 除数。</param>
    /// <returns>The quotient <c>left / right</c>. / 商 <c>left / right</c>。</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ScalarNode operator /(ScalarNode left, ScalarNode right)
    {
        double result = left.value / right.value;
        if (left.Tape is null && right.Tape is null)
        {
            return new(result);
        }

        // Skip partial derivatives for constant operands; compute -a/b^2 with separated exponents to avoid overflow/underflow in b^2.
        double dl = left.Tape is null ? 0 : 1 / right.value;
        double dr = right.Tape is null ? 0 : DenominatorDerivative(left.value, right.value);
        return Binary(result, left, dl, right, dr);
    }

    /// <summary>
    /// Negates a scalar node.
    /// <para>スカラーノードの符号を反転します。</para>
    /// </summary>
    /// <param name="operand">The operand to negate. / 符号を反転するノード。</param>
    /// <returns>The negated value <c>-operand</c>. / 符号を反転した値 <c>-operand</c>。</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ScalarNode operator -(ScalarNode operand)
        => Unary(-operand.value, operand, -1);

    /// <summary>
    /// Determines whether the value of <paramref name="left"/> is less than that of <paramref name="right"/>.
    /// <para><paramref name="left"/> の値が <paramref name="right"/> の値より小さいかどうかを判定します。</para>
    /// </summary>
    /// <param name="left">The left operand. / 左辺。</param>
    /// <param name="right">The right operand. / 右辺。</param>
    /// <returns><see langword="true"/> if <paramref name="left"/> is less than <paramref name="right"/>; otherwise, <see langword="false"/>. / 条件を満たす場合は <see langword="true"/>、それ以外は <see langword="false"/>。</returns>
    /// <remarks>
    /// <para>Compares values only and does not propagate gradients.</para>
    /// <para>値のみを比較し、勾配は伝播しません。</para>
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool operator <(ScalarNode left, ScalarNode right) => left.Value < right.Value;

    /// <summary>
    /// Determines whether the value of <paramref name="left"/> is greater than that of <paramref name="right"/>.
    /// <para><paramref name="left"/> の値が <paramref name="right"/> の値より大きいかどうかを判定します。</para>
    /// </summary>
    /// <param name="left">The left operand. / 左辺。</param>
    /// <param name="right">The right operand. / 右辺。</param>
    /// <returns><see langword="true"/> if <paramref name="left"/> is greater than <paramref name="right"/>; otherwise, <see langword="false"/>. / 条件を満たす場合は <see langword="true"/>、それ以外は <see langword="false"/>。</returns>
    /// <remarks>
    /// <para>Compares values only and does not propagate gradients.</para>
    /// <para>値のみを比較し、勾配は伝播しません。</para>
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool operator >(ScalarNode left, ScalarNode right) => left.Value > right.Value;

    /// <summary>
    /// Determines whether the value of <paramref name="left"/> is less than or equal to that of <paramref name="right"/>.
    /// <para><paramref name="left"/> の値が <paramref name="right"/> の値以下かどうかを判定します。</para>
    /// </summary>
    /// <param name="left">The left operand. / 左辺。</param>
    /// <param name="right">The right operand. / 右辺。</param>
    /// <returns><see langword="true"/> if <paramref name="left"/> is less than or equal to <paramref name="right"/>; otherwise, <see langword="false"/>. / 条件を満たす場合は <see langword="true"/>、それ以外は <see langword="false"/>。</returns>
    /// <remarks>
    /// <para>Compares values only and does not propagate gradients.</para>
    /// <para>値のみを比較し、勾配は伝播しません。</para>
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool operator <=(ScalarNode left, ScalarNode right) => left.Value <= right.Value;

    /// <summary>
    /// Determines whether the value of <paramref name="left"/> is greater than or equal to that of <paramref name="right"/>.
    /// <para><paramref name="left"/> の値が <paramref name="right"/> の値以上かどうかを判定します。</para>
    /// </summary>
    /// <param name="left">The left operand. / 左辺。</param>
    /// <param name="right">The right operand. / 右辺。</param>
    /// <returns><see langword="true"/> if <paramref name="left"/> is greater than or equal to <paramref name="right"/>; otherwise, <see langword="false"/>. / 条件を満たす場合は <see langword="true"/>、それ以外は <see langword="false"/>。</returns>
    /// <remarks>
    /// <para>Compares values only and does not propagate gradients.</para>
    /// <para>値のみを比較し、勾配は伝播しません。</para>
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool operator >=(ScalarNode left, ScalarNode right) => left.Value >= right.Value;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal void Validate() => Tape?.ValidateGeneration(Generation);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static ScalarNode Unary(double value, ScalarNode input, double derivative)
    {
        Tape? tape = input.Tape;
        if (tape is null)
        {
            return new(value);
        }

        tape.ValidateGeneration(input.Generation);
        int index = tape.Append(input.Index, derivative, -1, 0);
        return new(value, tape, index, tape.Generation);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static ScalarNode Binary(double value, ScalarNode left, double dl, ScalarNode right, double dr)
    {
        Tape? leftTape = left.Tape, rightTape = right.Tape;
        if (leftTape is null && rightTape is null)
        {
            return new(value);
        }

        leftTape?.ValidateGeneration(left.Generation);
        rightTape?.ValidateGeneration(right.Generation);
        if (leftTape is not null && rightTape is not null && leftTape != rightTape)
        {
            ThrowDifferentTapes();
        }

        Tape tape = (leftTape ?? rightTape)!;
        int index = tape.Append(leftTape is null ? -1 : left.Index, dl, rightTape is null ? -1 : right.Index, dr);
        return new(value, tape, index, tape.Generation);
    }

    // Computes -a/b^2 for division (shared with array division) without intermediate overflow/underflow in b^2.
    internal static double DenominatorDerivative(double numerator, double denominator)
    {
        if (denominator == 0 || !double.IsFinite(denominator) || !double.IsFinite(numerator))
        {
            return -(numerator / denominator) / denominator;
        }

        if (numerator == 0)
        {
            // Since b^2 > 0, preserve signed zero: -(±0)/b^2 = ∓0.
            return -numerator;
        }

        // Factor out binary exponents: -a/b^2 = -(n/d^2) * 2^(e_a - 2*e_b) with |n|, |d| in [1, 2).
        int numeratorExponent = Math.ILogB(numerator);
        int denominatorExponent = Math.ILogB(denominator);
        double normalizedNumerator = Math.ScaleB(numerator, -numeratorExponent);
        double normalizedDenominator = Math.ScaleB(denominator, -denominatorExponent);
        return Math.ScaleB(-normalizedNumerator / (normalizedDenominator * normalizedDenominator), numeratorExponent - 2 * denominatorExponent);
    }

    [DoesNotReturn]
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void ThrowDifferentTapes() => throw new InvalidOperationException("Nodes must belong to the same tape.");
}
