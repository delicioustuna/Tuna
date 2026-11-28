namespace Tuna.Statistics.Distributions;

internal interface IQuadratureIntegrand
{
    int Count { get; }

    double Density(double x);

    // 大きな値と微小な mass の積がオーバーフローしないよう、mass 側から順に乗算して values に格納する
    void Evaluate(double x, double density, double mass, Span<double> values);
}

// tanh-sinh（有限区間）および exp-sinh（半無限区間）による二重指数変換求積。
// 発散しうる非ゼロ有限端点付近では丸めによる端点距離の相対誤差を抑えるため、局所的な冪則 f ∝ d^s で密度を補正・外挿する。
internal static class DoubleExponentialQuadrature
{
    internal const int MaxComponents = 5;

    // 刻み半減ごとに誤差はおおよそ 2 乗で縮小するため、前段との相対差がこの値以下なら ε 未満とみなす
    private const double Tolerance = 1e-10;

    // 閉じた端では距離比がこれ未満で打ち切り、開いた端ではここから先で寄与判定を行う
    private const double NegligibleDistance = 1e-30;

    // 開いた端・無限遠側で点の寄与が絶対値積分のこの割合以下なら打ち切る
    private const double NegligibleContribution = 1e-25;

    // 端点 0 で冪外挿を開始する距離 2^-969（これより近い点は非正規化数に近く分布計算の精度が低下しうる）
    private const double ZeroEndSpacing = 2.004168360008973e-292;

    private const int MinLevel = 3;
    private const int MaxLevel = 10;

    internal static void Finite<T>(ref T integrand, double a, double b, double supportMinimum, double supportMaximum, Span<double> sums)
        where T : struct, IQuadratureIntegrand
    {
        var nodes = new FiniteNodes(a, b, supportMinimum, supportMaximum);
        Integrate(ref integrand, ref nodes, sums);
    }

    internal static void Finite<T>(ref T integrand, double a, double b, Span<double> sums)
        where T : struct, IQuadratureIntegrand =>
        Finite(ref integrand, a, b, double.NegativeInfinity, double.PositiveInfinity, sums);

    // 半無限区間 [a, ∞) (direction = 1) または (-∞, a] (direction = -1) の exp-sinh 積分
    internal static void SemiInfinite<T>(ref T integrand, double a, double scale, int direction, Span<double> sums)
        where T : struct, IQuadratureIntegrand
    {
        var nodes = new SemiInfiniteNodes(a, scale, direction);
        Integrate(ref integrand, ref nodes, sums);
    }

    private interface INodes
    {
        // 側を打ち切る場合は false を返す。tail が true の点では密度 0 または寄与無視で打ち切る
        bool Node<T>(ref T integrand, double t, out double x, out double weight, out double density, out bool tail)
            where T : struct, IQuadratureIntegrand;
    }

    private struct FiniteNodes(double a, double b, double supportMinimum, double supportMaximum) : INodes
    {
        private readonly double _half = 0.5 * b - 0.5 * a;
        private SupportEnd _lowerEnd = new(supportMinimum, 1), _upperEnd = new(supportMaximum, -1);

        public bool Node<T>(ref T integrand, double t, out double x, out double weight, out double density, out bool tail)
            where T : struct, IQuadratureIntegrand
        {
            // 端からの距離 half·(1 - tanh((π/2) sinh |t|)) と重みを、e = exp(-π sinh |t|) で桁落ちなく求める
            double e = Math.Exp(-Math.PI * Math.Sinh(Math.Abs(t)));
            double denominator = 1 + e;
            double distance = _half * (2 * e / denominator);
            weight = _half * XMath.Constants.PiOver2 * Math.Cosh(t) * (4 * e / (denominator * denominator));
            bool upper = t >= 0;
            double end = upper ? b : a;
            bool open = t != 0 && end == (upper ? supportMaximum : supportMinimum);
            tail = open && e < NegligibleDistance;
            density = 0;

            // 丸める前の点は x + error（端の近くでは正確な和）
            x = upper ? end - distance : end + distance;
            double error = upper ? (end - x) - distance : distance - (x - end);
            if (!(weight > 0 && distance > 0) || !open && e < NegligibleDistance)
            {
                return false;
            }

            // 近い方の台の端からの距離で密度を直す
            double lowerDistance = (x - supportMinimum) + error, upperDistance = (supportMaximum - x) - error;
            density = lowerDistance <= upperDistance
                ? _lowerEnd.Density(ref integrand, x, lowerDistance)
                : _upperEnd.Density(ref integrand, x, upperDistance);
            return density < double.PositiveInfinity;
        }
    }

    // 密度の台の端点付近で丸め前の距離に対応する密度を局所冪則 f ∝ d^s で補正・外挿する
    private struct SupportEnd
    {
        private readonly double _end;

        // 台の内側方向 (±1)
        private readonly double _direction;

        // 端での double の間隔（0 の端では ZeroEndSpacing）。これより近い点は外挿する
        private readonly double _spacing;

        // 外挿用に初回のみ計算する基準点密度と冪指数
        private bool _initialized;
        private double _density;
        private double _exponent;

        public SupportEnd(double end, double direction)
        {
            _end = end;
            _direction = direction;
            _spacing = !double.IsFinite(end) ? 0
                : end == 0 ? ZeroEndSpacing
                : Math.Abs((direction > 0 ? Math.BitIncrement(end) : Math.BitDecrement(end)) - end);
        }

        // 丸められた点 x と丸め前の端点距離 distance から、丸め前の距離での密度を推定する
        public double Density<T>(ref T integrand, double x, double distance)
            where T : struct, IQuadratureIntegrand
        {
            if (!(distance >= 2 * _spacing))
            {
                return Extrapolate(ref integrand, distance);
            }

            // 端点付近では Sterbenz の補題により rounded は正確。端から遠ければ丸めの影響は ε 程度のためそのまま返す
            double density = integrand.Density(x);
            double rounded = _direction * (x - _end);
            if (rounded == distance || !(rounded < 0.5 * Math.Abs(_end)) || !(density > 0 && density < double.PositiveInfinity))
            {
                return density;
            }

            // 丸めた点と端との中点の密度から冪指数を推定し、丸め前の距離へ補正する
            double between = _end + _direction * (0.5 * rounded);
            double middle = _direction * (between - _end);
            double inner = integrand.Density(between);
            if (!(inner > 0 && inner < double.PositiveInfinity && middle > 0 && middle < rounded))
            {
                return density;
            }

            double exponent = (Math.Log(inner) - Math.Log(density)) / Math.Log(middle / rounded);
            return density * Math.Exp(exponent * DistributionMath.Log1p((distance - rounded) / rounded));
        }

        // 間隔の 1 倍と 2 倍の点（いずれも double で厳密表現可能）の密度から求めた冪則で外挿する
        private double Extrapolate<T>(ref T integrand, double distance)
            where T : struct, IQuadratureIntegrand
        {
            if (!_initialized)
            {
                _initialized = true;
                _density = integrand.Density(_end + _direction * _spacing);
                double far = integrand.Density(_end + _direction * (2 * _spacing));
                _exponent = (Math.Log(far) - Math.Log(_density)) / XMath.Constants.Ln2;
            }

            return _density > 0 && double.IsFinite(_exponent) ? _density * Math.Exp(_exponent * Math.Log(distance / _spacing)) : 0;
        }
    }

    private readonly struct SemiInfiniteNodes(double a, double scale, int direction) : INodes
    {
        public bool Node<T>(ref T integrand, double t, out double x, out double weight, out double density, out bool tail)
            where T : struct, IQuadratureIntegrand
        {
            double ratio = Math.Exp(XMath.Constants.PiOver2 * Math.Sinh(t));
            double offset = scale * ratio;
            x = a + direction * offset;
            weight = offset * XMath.Constants.PiOver2 * Math.Cosh(t);

            // 無限遠の側は、密度が 0 になるか寄与が無視できたら止める。有限の端の側は密度が有限なので、端からの距離が十分小さくなったら止める
            tail = t > 0;
            density = 0;
            if (!(ratio >= NegligibleDistance && double.IsFinite(x) && weight > 0 && double.IsFinite(weight)))
            {
                return false;
            }

            density = integrand.Density(x);
            return true;
        }
    }

    private static void Integrate<T, TNodes>(ref T integrand, ref TNodes nodes, Span<double> sums)
        where T : struct, IQuadratureIntegrand
        where TNodes : struct, INodes
    {
        int count = integrand.Count;
        Span<double> values = stackalloc double[MaxComponents];
        Span<double> total = stackalloc double[MaxComponents];
        Span<double> compensation = stackalloc double[MaxComponents];
        Span<double> absolute = stackalloc double[MaxComponents];
        Span<double> previous = stackalloc double[MaxComponents];
        total.Clear();
        compensation.Clear();
        absolute.Clear();

        // 刻み 1 の点（t = 0 と整数）
        double h = 1;
        AddNode(ref integrand, ref nodes, 0, total, compensation, absolute, values, count);
        for (int side = -1; side <= 1; side += 2)
        {
            for (int k = 1; AddNode(ref integrand, ref nodes, side * k, total, compensation, absolute, values, count); k++)
            {
            }
        }

        for (int level = 1; level <= MaxLevel; level++)
        {
            total[..count].CopyTo(previous);
            h *= 0.5;

            // 新しい点は h の奇数倍。前の段の和は刻みが倍なので、同じ重みの和として足し合わせ、最後に h を掛ける
            for (int side = -1; side <= 1; side += 2)
            {
                for (int k = 1; AddNode(ref integrand, ref nodes, side * k * h, total, compensation, absolute, values, count); k += 2)
                {
                }
            }

            if (level >= MinLevel && Converged(total, previous, absolute, count))
            {
                break;
            }
        }

        // total は刻み 1 の重みの和なので、最後の刻みを掛ける（各段で和を半分にする代わり）
        for (int j = 0; j < count; j++)
        {
            sums[j] = (total[j] + compensation[j]) * h;
        }
    }

    // Neumaier の補償付き加算で 1 点を追加する。その側を打ち切る場合は false を返す
    private static bool AddNode<T, TNodes>(ref T integrand, ref TNodes nodes, double t, Span<double> total, Span<double> compensation,
        Span<double> absolute, Span<double> values, int count)
        where T : struct, IQuadratureIntegrand
        where TNodes : struct, INodes
    {
        if (!nodes.Node(ref integrand, t, out double x, out double weight, out double density, out bool tail))
        {
            return false;
        }

        if (!(density > 0))
        {
            // 裾では密度 0 で打ち切り、それ以外は点をスキップして継続する
            return !tail;
        }

        integrand.Evaluate(x, density, weight * density, values);
        bool negligible = tail;
        for (int j = 0; j < count; j++)
        {
            double value = values[j], sum = total[j] + value;
            compensation[j] += Math.Abs(total[j]) >= Math.Abs(value) ? (total[j] - sum) + value : (value - sum) + total[j];
            total[j] = sum;
            absolute[j] += Math.Abs(values[j]);
            if (!(Math.Abs(values[j]) <= NegligibleContribution * absolute[j]))
            {
                negligible = false;
            }
        }

        return !negligible;
    }

    // total は刻み 1 の重み和であり段ごとに点数が倍になるため、前段の和を 2 倍して比較する
    private static bool Converged(Span<double> total, Span<double> previous, Span<double> absolute, int count)
    {
        for (int j = 0; j < count; j++)
        {
            if (!(Math.Abs(total[j] - 2 * previous[j]) <= Tolerance * absolute[j]))
            {
                return false;
            }
        }

        return true;
    }
}
