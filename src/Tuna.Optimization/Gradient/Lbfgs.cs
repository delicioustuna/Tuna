using Tuna.LinearAlgebra;

namespace Tuna.Optimization.Gradient;

/// <summary>
/// Minimizes an unconstrained differentiable objective function using the L-BFGS method.
/// <para>L-BFGS 法を用いて無制約の微分可能な目的関数を最小化します。</para>
/// </summary>
/// <remarks>
/// <para>Uses two-loop recursion over the most recent curvature pairs with Moré–Thuente strong Wolfe line search. See <see cref="LineSearchMinimizer"/> for the evaluation workflow and termination criteria.</para>
/// <para>直近の曲率ペアに対する2重ループ再帰と、強 Wolfe 条件に基づく Moré–Thuente 直線探索を使用します。試行点の評価手順および終了条件については <see cref="LineSearchMinimizer"/> を参照してください。</para>
/// </remarks>
public sealed class Lbfgs : LineSearchMinimizer
{
    private readonly CurvaturePairs pairs;
    private readonly double[] alpha;

    /// <summary>
    /// Initializes a new L-BFGS minimizer for the specified problem dimension.
    /// <para>指定した次元に対する L-BFGS オプティマイザーを初期化します。</para>
    /// </summary>
    /// <param name="dimension">The number of variables (at least 1). / 変数の数（1 以上）。</param>
    /// <param name="options">Optional solver configuration, or <see langword="null"/> for defaults. / オプション設定。<see langword="null"/> の場合は既定値を使用します。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="dimension"/> is less than 1, or an option value in <paramref name="options"/> is out of range. / <paramref name="dimension"/> が 1 未満、または <paramref name="options"/> の設定値が有効範囲外の場合。</exception>
    public Lbfgs(int dimension, LbfgsOptions? options = null)
        : base(dimension, options ??= new LbfgsOptions())
    {
        pairs = new CurvaturePairs(dimension, options.HistorySize, trackProducts: false);
        alpha = new double[options.HistorySize];
    }

    /// <summary>
    /// Gets the maximum number of curvature pairs retained in history.
    /// <para>履歴に保持する曲率ペアの最大数 <c>m</c> を取得します。</para>
    /// </summary>
    public int HistorySize => pairs.Capacity;

    /// <summary>
    /// Gets the number of curvature pairs currently stored in history.
    /// <para>現在履歴に保持している曲率ペアの数を取得します。</para>
    /// </summary>
    public int HistoryCount => pairs.Count;

    private protected override bool HasHistory => pairs.Count > 0;

    private protected override void ClearHistory() => pairs.Clear();

    private protected override void UpdateHistory(ReadOnlySpan<double> s, ReadOnlySpan<double> y)
    {
        double sy = Blas.Dot(s, y);
        double yy = Blas.Dot(y, y);
        if (sy > NumericConstants.MachineEpsilon * yy && double.IsFinite(sy) && double.IsFinite(yy))
        {
            pairs.Add(s, y, sy, yy);
        }
    }

    private protected override bool ComputeDirection(Span<double> d)
    {
        // 2重ループの再帰で q = H∇f を作り、d = −q
        g.CopyTo(d);
        int m = pairs.Count;
        for (int k = m - 1; k >= 0; k--)
        {
            alpha[k] = Blas.Dot(pairs.S(k), d) / pairs.SY(k);
            Blas.Axpy(-alpha[k], pairs.Y(k), d);
        }

        Blas.Scale(pairs.LatestScaling, d);
        for (int k = 0; k < m; k++)
        {
            double beta = Blas.Dot(pairs.Y(k), d) / pairs.SY(k);
            Blas.Axpy(alpha[k] - beta, pairs.S(k), d);
        }

        Blas.Scale(-1.0, d);
        foreach (double v in d)
        {
            if (!double.IsFinite(v))
            {
                return false;
            }
        }

        return true;
    }
}
