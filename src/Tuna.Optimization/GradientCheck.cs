using Tuna.LinearAlgebra;

namespace Tuna.Optimization;

/// <summary>
/// Provides utilities for verifying analytic gradients against finite-difference approximations.
/// <para>解析勾配を有限差分近似と照合して検証するためのユーティリティを提供します。</para>
/// </summary>
public static class GradientCheck
{
    /// <summary>
    /// Compares the analytic gradient of an objective function against a central finite-difference approximation at the specified point.
    /// <para>指定した点で目的関数の解析勾配と中心差分近似を比較します。</para>
    /// </summary>
    /// <typeparam name="TObjective">The differentiable objective function type. / 微分可能な目的関数の型。</typeparam>
    /// <param name="objective">The differentiable objective function to check. / 検証対象の微分可能な目的関数。</param>
    /// <param name="x">The point at which to compare gradients (length at least 1). / 勾配を比較する点（長さ 1 以上）。</param>
    /// <returns>The maximum scaled componentwise error <c>|gᵢ − dᵢ| / max(1, |gᵢ|, |dᵢ|)</c>, or <see cref="double.NaN"/> if any component error is <see cref="double.NaN"/>. / 各成分の正規化誤差 <c>|gᵢ − dᵢ| / max(1, |gᵢ|, |dᵢ|)</c> の最大値。いずれかの成分誤差が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="x"/> is empty. / <paramref name="x"/> が空の場合。</exception>
    /// <exception cref="ArgumentNullException"><paramref name="objective"/> is <see langword="null"/>. / <paramref name="objective"/> が <see langword="null"/> の場合。</exception>
    public static double MaximumError<TObjective>(TObjective objective, ReadOnlySpan<double> x)
        where TObjective : IDifferentiableObjectiveFunction
    {
        ArgumentNullException.ThrowIfNull(objective);
        if (x.IsEmpty)
        {
            throw new ArgumentException("The point must not be empty.", nameof(x));
        }

        var analytic = new double[x.Length];
        var numeric = new double[x.Length];
        objective.Evaluate(x, analytic);
        new FiniteDifferenceGradient(objective, x.Length).Evaluate(x, numeric);
        double worst = 0;
        for (int i = 0; i < x.Length; i++)
        {
            double error = Math.Abs(analytic[i] - numeric[i]) / Math.Max(1.0, Math.Max(Math.Abs(analytic[i]), Math.Abs(numeric[i])));
            worst = double.IsNaN(error) ? double.NaN : Math.Max(worst, error);
            if (double.IsNaN(worst))
            {
                break;
            }
        }

        return worst;
    }

    /// <summary>
    /// Compares the analytic gradient of a delegate objective function against a central finite-difference approximation at the specified point.
    /// <para>指定した点でデリゲート目的関数の解析勾配と中心差分近似を比較します。</para>
    /// </summary>
    /// <param name="objective">The objective delegate returning the objective value and writing the gradient to its second parameter. / 目的関数値を返し、第2引数に勾配を書き込むデリゲート。</param>
    /// <param name="x">The point at which to compare gradients (length at least 1). / 勾配を比較する点（長さ 1 以上）。</param>
    /// <returns>The maximum scaled componentwise error <c>|gᵢ − dᵢ| / max(1, |gᵢ|, |dᵢ|)</c>, or <see cref="double.NaN"/> if any component error is <see cref="double.NaN"/>. / 各成分の正規化誤差 <c>|gᵢ − dᵢ| / max(1, |gᵢ|, |dᵢ|)</c> の最大値。いずれかの成分誤差が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="x"/> is empty. / <paramref name="x"/> が空の場合。</exception>
    /// <exception cref="ArgumentNullException"><paramref name="objective"/> is <see langword="null"/>. / <paramref name="objective"/> が <see langword="null"/> の場合。</exception>
    public static double MaximumError(Func<ReadOnlyVectorSpan<double>, VectorSpan<double>, double> objective, ReadOnlySpan<double> x)
    {
        ArgumentNullException.ThrowIfNull(objective);
        return MaximumError(new DelegateObjective(objective), x);
    }
}

internal sealed class DelegateObjective(Func<ReadOnlyVectorSpan<double>, VectorSpan<double>, double> function) : IDifferentiableObjectiveFunction
{
    public double Evaluate(ReadOnlyVectorSpan<double> x, VectorSpan<double> gradient) => function(x, gradient);

    public double Evaluate(ReadOnlyVectorSpan<double> x) => function(x, new double[x.Length]);
}
