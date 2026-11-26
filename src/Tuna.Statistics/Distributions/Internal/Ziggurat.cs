using Tuna.RandomSource;

namespace Tuna.Statistics.Distributions;

// 64 ビット乱数から層番号（下位 8 ビット）、符号（第 8 ビット）、層内一様乱数（上位 53 ビット）を独立に取り、Doornik (2005) の相関を避ける。
// 裾は正規分布で Marsaglia (1964) の棄却法、指数分布で無記憶性を用いる。
internal static class Ziggurat
{
    private const int Layers = 256;
    private const double Unit53 = 1.0 / (1UL << 53);

    private static readonly Table NormalTable = Table.Build(
        x => Math.Exp(-0.5 * x * x),
        y => Math.Sqrt(-2 * Math.Log(y)),
        r => Math.Sqrt(Math.PI / 2) * XMath.Special.Erfc(r / Math.Sqrt(2)),
        3, 4.5);

    private static readonly Table ExponentialTable = Table.Build(
        x => Math.Exp(-x),
        y => -Math.Log(y),
        r => Math.Exp(-r),
        6, 9);

    internal static double StandardNormal(IRandomSource random)
    {
        Table table = NormalTable;
        while (true)
        {
            ulong bits = random.NextUInt64();
            int i = (int)(bits & 0xFF);
            bool negative = (bits & 0x100) != 0;
            double x = (bits >> 11) * Unit53 * table.X[i];
            if (x < table.X[i + 1])
            {
                return negative ? -x : x;
            }

            if (i == 0)
            {
                // 裾 x > r: 指数分布2つによる棄却法（Marsaglia 1964）
                double a, b;
                do
                {
                    a = StandardExponential(random) / table.R;
                    b = StandardExponential(random);
                }
                while (b + b < a * a);

                x = table.R + a;
                return negative ? -x : x;
            }

            double y = table.F[i] + random.NextDouble() * (table.F[i + 1] - table.F[i]);
            if (y < Math.Exp(-0.5 * x * x))
            {
                return negative ? -x : x;
            }
        }
    }

    internal static double StandardExponential(IRandomSource random)
    {
        Table table = ExponentialTable;
        double offset = 0;
        while (true)
        {
            ulong bits = random.NextUInt64();
            int i = (int)(bits & 0xFF);
            double x = (bits >> 11) * Unit53 * table.X[i];
            if (x < table.X[i + 1])
            {
                return offset + x;
            }

            if (i == 0)
            {
                // 裾 x > r: 無記憶性により r + (指数分布)
                offset += table.R;
                continue;
            }

            double y = table.F[i] + random.NextDouble() * (table.F[i + 1] - table.F[i]);
            if (y < Math.Exp(-x))
            {
                return offset + x;
            }
        }
    }

    // 層の右端 X[i]（X[0] は最下層の等面積長方形幅、X[1] = r、X[256] = 0）と密度 F[i] = f(X[i])
    private sealed class Table
    {
        internal readonly double[] X = new double[Layers + 1];
        internal readonly double[] F = new double[Layers + 1];
        internal double R;

        // 非正規化密度 f（f(0) = 1 で単調減少）と裾面積 ∫_r^∞ f から最下層右端 r を二分法で決定し表を構築する
        internal static Table Build(Func<double, double> density, Func<double, double> inverseDensity,
            Func<double, double> tail, double lower, double upper)
        {
            for (int iteration = 0; iteration < 200; iteration++)
            {
                double middle = 0.5 * (lower + upper);
                if (middle == lower || middle == upper)
                {
                    break;
                }

                if (TopExcess(middle, density, inverseDensity, tail) > 0)
                {
                    lower = middle;
                }
                else
                {
                    upper = middle;
                }
            }

            double r = upper;
            double area = r * density(r) + tail(r);
            var table = new Table { R = r };
            table.X[0] = area / density(r);
            table.X[1] = r;
            for (int i = 1; i < Layers - 1; i++)
            {
                table.X[i + 1] = inverseDensity(density(table.X[i]) + area / table.X[i]);
            }

            table.X[Layers] = 0;
            for (int i = 1; i < Layers; i++)
            {
                table.F[i] = density(table.X[i]);
            }

            table.F[Layers] = 1;
            return table;
        }

        // 層を積み上げた最上層の上端と f(0) = 1 の差（正なら超過）
        private static double TopExcess(double r, Func<double, double> density, Func<double, double> inverseDensity,
            Func<double, double> tail)
        {
            double area = r * density(r) + tail(r);
            double x = r;
            for (int i = 1; i < Layers - 1; i++)
            {
                double y = density(x) + area / x;
                if (y >= 1)
                {
                    return 1;
                }

                x = inverseDensity(y);
            }

            return density(x) + area / x - 1;
        }
    }
}
