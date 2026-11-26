namespace Tuna.Integration.Internal;

// 軸ごとの15点 Kronrod 則と7点 Gauss 則の直積（15^d 点、各軸次数 23 まで厳密）。
// 誤差見積もりは |K − G|（下限 50ε·V·Σ|wₖ f(xₖ)|）、軸スコアは軸 i のみ Gauss 則にした直積との差 |K − Mᵢ|。
internal sealed class ProductGaussKronrod : ICubatureRule
{
    private const int Nodes = GaussKronrod.Points;

    private readonly int d;
    private readonly double[] point;
    private readonly int[] index;
    private readonly double[] axisError;
    private readonly double[] axisSum;

    // Gauss 重み / Kronrod 重み（非 Gauss 節点は 0）
    private static readonly double[] Ratio = CreateRatio();

    internal ProductGaussKronrod(int dimension)
    {
        d = dimension;
        point = new double[dimension];
        index = new int[dimension];
        axisError = new double[dimension];
        axisSum = new double[dimension];
        long points = 1;
        for (int i = 0; i < dimension; i++)
        {
            points *= Nodes;
        }

        Points = points;
    }

    public long Points { get; }

    public ReadOnlySpan<double> AxisScores => axisError;

    public bool Apply<TIntegrand>(ref TIntegrand f, ReadOnlySpan<double> lower, ReadOnlySpan<double> upper, out RuleEstimate estimate)
        where TIntegrand : IBoxIntegrand
    {
        var nodes = GaussKronrod.AllNodes;
        var kronrod = GaussKronrod.AllKronrodWeights;
        var gauss = GaussKronrod.AllGaussWeights;
        Span<double> center = stackalloc double[d];
        Span<double> half = stackalloc double[d];
        double scale = 1;
        for (int i = 0; i < d; i++)
        {
            center[i] = 0.5 * lower[i] + 0.5 * upper[i];
            half[i] = 0.5 * upper[i] - 0.5 * lower[i];
            scale *= half[i];
            index[i] = 0;
            point[i] = center[i] + half[i] * nodes[0];
            axisSum[i] = 0;
        }

        double sumK = 0, sumG = 0, sumAbs = 0;
        bool finite = true;
        for (long n = 0; n < Points; n++)
        {
            double v = f.Evaluate(point);
            finite &= double.IsFinite(v);
            double wk = 1, wg = 1;
            for (int i = 0; i < d; i++)
            {
                wk *= kronrod[index[i]];
                wg *= gauss[index[i]];
            }

            double weighted = wk * v;
            sumK += weighted;
            sumG += wg * v;
            sumAbs += wk * Math.Abs(v);
            for (int i = 0; i < d; i++)
            {
                axisSum[i] += Ratio[index[i]] * weighted;
            }

            // 添字を1つ進める（第0軸が最も速く回る）
            for (int i = 0; i < d; i++)
            {
                if (++index[i] < Nodes)
                {
                    point[i] = center[i] + half[i] * nodes[index[i]];
                    break;
                }

                index[i] = 0;
                point[i] = center[i] + half[i] * nodes[0];
            }
        }

        if (!finite)
        {
            estimate = default;
            return false;
        }

        double value = scale * sumK;
        double error = Math.Abs(scale * (sumK - sumG));
        for (int i = 0; i < d; i++)
        {
            axisError[i] = Math.Abs(scale * (sumK - axisSum[i]));
        }

        double resabs = Math.Abs(scale) * sumAbs;
        double roundoff = 50 * GaussKronrod.MachineEpsilon * resabs;
        bool atRoundoff = false;
        if (resabs > GaussKronrod.MinNormal / (50 * GaussKronrod.MachineEpsilon) && error <= roundoff)
        {
            error = roundoff;
            atRoundoff = true;
        }

        estimate = new RuleEstimate(value, error, atRoundoff);
        return double.IsFinite(value) && double.IsFinite(error);
    }

    private static double[] CreateRatio()
    {
        var ratio = new double[Nodes];
        for (int j = 0; j < Nodes; j++)
        {
            ratio[j] = GaussKronrod.AllGaussWeights[j] / GaussKronrod.AllKronrodWeights[j];
        }

        return ratio;
    }
}

// Cubature の領域評価に用いる求積則インターフェース
internal interface ICubatureRule
{
    long Points { get; }

    // 直前の Apply における各軸の分割スコア（値が大きい軸を優先的に分割する）
    ReadOnlySpan<double> AxisScores { get; }

    bool Apply<TIntegrand>(ref TIntegrand f, ReadOnlySpan<double> lower, ReadOnlySpan<double> upper, out RuleEstimate estimate)
        where TIntegrand : IBoxIntegrand;
}
