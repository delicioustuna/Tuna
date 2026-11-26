namespace Tuna.Integration.Internal;

// 1区間に求積則を適用した結果。AtRoundoff は誤差見積もりが丸め誤差下限で決まっており、分割しても減らないことを示す。
internal readonly record struct RuleEstimate(double Value, double Error, bool AtRoundoff);

// 7点 Gauss–Legendre 則と、それを含む15点 Kronrod 則（Kronrod 1965、区間 [-1, 1]、次数 23 まで厳密）。
// 節点・重みは 60 桁精度で計算して binary64 に丸めた定数。
// 誤差見積もりは QUADPACK（Piessens et al. 1983）の式 I|f − f̄|·min(1, (200|K − G| / I|f − f̄|)^1.5) を用い、
// 丸め誤差下限 50ε·I|f| を適用する。
internal static class GaussKronrod
{
    internal const int Points = 15;

    internal const double MachineEpsilon = 2.220446049250313e-16;

    // 正規化数の最小値 2^-1022
    internal const double MinNormal = 2.2250738585072014e-308;

    // 正側の節点（降順、末尾が中心 0）。奇数インデックス（1, 3, 5, 7）が Gauss 節点。
    private static ReadOnlySpan<double> Nodes =>
    [
        0.9914553711208126, 0.9491079123427585, 0.8648644233597691, 0.7415311855993945,
        0.5860872354676911, 0.4058451513773972, 0.20778495500789848, 0.0,
    ];

    private static ReadOnlySpan<double> KronrodWeights =>
    [
        0.022935322010529224, 0.06309209262997856, 0.10479001032225019, 0.14065325971552592,
        0.1690047266392679, 0.19035057806478542, 0.20443294007529889, 0.20948214108472782,
    ];

    // Nodes[1], Nodes[3], Nodes[5], Nodes[7] に対応する Gauss 重み
    private static ReadOnlySpan<double> GaussWeights => [0.1294849661688697, 0.27970539148927664, 0.3818300505051189, 0.4179591836734694];

    // 昇順に展開した15点の節点と重み（AllGaussWeights は非 Gauss 節点が 0）
    internal static readonly double[] AllNodes = new double[Points];

    internal static readonly double[] AllKronrodWeights = new double[Points];

    internal static readonly double[] AllGaussWeights = new double[Points];

    static GaussKronrod()
    {
        for (int i = 0; i < 8; i++)
        {
            double gauss = (i & 1) != 0 ? GaussWeights[i >> 1] : 0;
            AllNodes[i] = -Nodes[i];
            AllNodes[Points - 1 - i] = Nodes[i];
            AllKronrodWeights[i] = AllKronrodWeights[Points - 1 - i] = KronrodWeights[i];
            AllGaussWeights[i] = AllGaussWeights[Points - 1 - i] = gauss;
        }
    }

    internal static bool Apply<TIntegrand>(ref TIntegrand f, double t0, double t1, out RuleEstimate estimate)
        where TIntegrand : IIntegrand
    {
        double center = 0.5 * t0 + 0.5 * t1;
        double half = 0.5 * t1 - 0.5 * t0;
        Span<double> lower = stackalloc double[7];
        Span<double> upper = stackalloc double[7];
        var nodes = Nodes;
        var kronrod = KronrodWeights;
        var gauss = GaussWeights;

        double fc = f.Evaluate(center);
        bool finite = double.IsFinite(fc);
        double resultK = kronrod[7] * fc;
        double resultG = gauss[3] * fc;
        double absK = kronrod[7] * Math.Abs(fc);
        for (int i = 0; i < 7; i++)
        {
            double dx = half * nodes[i];
            double fl = f.Evaluate(center - dx);
            double fu = f.Evaluate(center + dx);
            finite &= double.IsFinite(fl) & double.IsFinite(fu);
            lower[i] = fl;
            upper[i] = fu;
            double pair = fl + fu;
            resultK += kronrod[i] * pair;
            absK += kronrod[i] * (Math.Abs(fl) + Math.Abs(fu));
            if ((i & 1) != 0)
            {
                resultG += gauss[i >> 1] * pair;
            }
        }

        if (!finite)
        {
            estimate = default;
            return false;
        }

        // 平均 f̄ = K/2 からのずれの積分
        double mean = 0.5 * resultK;
        double ascK = kronrod[7] * Math.Abs(fc - mean);
        for (int i = 0; i < 7; i++)
        {
            ascK += kronrod[i] * (Math.Abs(lower[i] - mean) + Math.Abs(upper[i] - mean));
        }

        double scale = Math.Abs(half);
        double value = resultK * half;
        double resabs = absK * scale;
        double resasc = ascK * scale;
        double error = Math.Abs((resultK - resultG) * half);
        if (resasc != 0 && error != 0)
        {
            error = resasc * Math.Min(1, Math.Pow(200 * error / resasc, 1.5));
        }

        // 和の丸め誤差（絶対値の積分が非正規化数に近いときは使わない）
        double roundoff = 50 * MachineEpsilon * resabs;
        bool atRoundoff = false;
        if (resabs > MinNormal / (50 * MachineEpsilon) && error <= roundoff)
        {
            error = roundoff;
            atRoundoff = true;
        }

        estimate = new RuleEstimate(value, error, atRoundoff);
        return double.IsFinite(value) && double.IsFinite(error);
    }
}
