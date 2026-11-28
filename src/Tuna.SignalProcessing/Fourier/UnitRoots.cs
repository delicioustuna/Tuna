using System.Numerics;

namespace Tuna.SignalProcessing;

// 1 の冪根 exp(-2πi·k/n) を整数演算で八分円 [0, π/4] に還元して評価する。
// 角 2πk/n の丸め誤差や漸化式の誤差蓄積を避け、n/8 の倍数での厳密値と共役対称性を保つ。
internal static class UnitRoots
{
    internal static Complex Root(long k, long n)
    {
        k %= n;
        if (k < 0)
        {
            k += n;
        }

        // 2πk/n = (π/4)·(o + r/n), o ∈ [0, 7], 0 ≤ r < n
        long eightK = 8 * k;
        long o = eightK / n;
        long r = eightK - o * n;

        double c, s;
        if ((o & 1) == 0)
        {
            if (r == 0)
            {
                (c, s) = (1, 0);
            }
            else
            {
                (s, c) = Math.SinCos(Math.PI / 4 * ((double)r / n));
            }
        }
        else
        {
            // cos(π/2 − φ′) = sin(φ′) を利用して [0, π/4] で評価する
            if (r == 0)
            {
                (c, s) = (XMath.Constants.OneOverSqrt2, XMath.Constants.OneOverSqrt2);
            }
            else
            {
                (c, s) = Math.SinCos(Math.PI / 4 * ((double)(n - r) / n));
            }
        }

        (c, s) = (o >> 1) switch
        {
            0 => (c, s),
            1 => (-s, c),
            2 => (-c, -s),
            _ => (s, -c),
        };

        // -0.0 を +0.0 に正規化する
        return new Complex(c + 0.0, 0.0 - s);
    }
}
