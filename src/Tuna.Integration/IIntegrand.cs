using Tuna.LinearAlgebra;

namespace Tuna.Integration;

/// <summary>
/// Represents a one-dimensional integrand function.
/// <para>1変数の被積分関数を表します。</para>
/// </summary>
/// <remarks>
/// <para>Implementations must be deterministic and return the same value for the same input. Implementing this interface on a value type (<see langword="struct"/>) allows the JIT compiler to inline evaluations when passed to generic overloads.</para>
/// <para>実装は決定的であり、同じ入力に対して常に同じ値を返す必要があります。値型（<see langword="struct"/>）で実装してジェネリック版オーバーロードに渡すと、JIT コンパイラが評価呼び出しをインライン化できます。</para>
/// </remarks>
public interface IIntegrand
{
    /// <summary>
    /// Evaluates the integrand at the specified point.
    /// <para>指定した点で被積分関数を評価します。</para>
    /// </summary>
    /// <param name="x">The point at which to evaluate the integrand. / 評価点。</param>
    /// <returns>The value of the integrand at <paramref name="x"/>. / <paramref name="x"/> における被積分関数の値。</returns>
    double Evaluate(double x);
}

/// <summary>
/// Represents a multivariate integrand function.
/// <para>多変数の被積分関数を表します。</para>
/// </summary>
/// <remarks>
/// <para>Implementations must be deterministic and return the same value for the same input. Implementing this interface on a value type (<see langword="struct"/>) allows the JIT compiler to inline evaluations when passed to generic overloads.</para>
/// <para>実装は決定的であり、同じ入力に対して常に同じ値を返す必要があります。値型（<see langword="struct"/>）で実装してジェネリック版オーバーロードに渡すと、JIT コンパイラが評価呼び出しをインライン化できます。</para>
/// </remarks>
public interface IMultivariateIntegrand
{
    /// <summary>
    /// Evaluates the integrand at the specified point.
    /// <para>指定した点で被積分関数を評価します。</para>
    /// </summary>
    /// <param name="x">The point at which to evaluate the integrand. The span is valid only during this call and must be copied if retained. / 評価点。この呼び出しの間だけ有効なビューであるため、保持する場合はコピーしてください。</param>
    /// <returns>The value of the integrand at <paramref name="x"/>. / <paramref name="x"/> における被積分関数の値。</returns>
    double Evaluate(ReadOnlyVectorSpan<double> x);
}

internal readonly struct DelegateIntegrand(Func<double, double> function) : IIntegrand
{
    public double Evaluate(double x) => function(x);
}

internal readonly struct DelegateMultivariateIntegrand(Func<ReadOnlyVectorSpan<double>, double> function) : IMultivariateIntegrand
{
    public double Evaluate(ReadOnlyVectorSpan<double> x) => function(x);
}
