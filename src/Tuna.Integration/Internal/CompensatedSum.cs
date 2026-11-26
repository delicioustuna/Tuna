namespace Tuna.Integration.Internal;

// Neumaier 補償和。区間の積分値や誤差見積もりの加減算で丸め誤差を抑える。
internal struct CompensatedSum
{
    private double sum;
    private double compensation;

    // 和が ±∞ のとき補正項は ∞ − ∞ で NaN になるため、和をそのまま返す。
    public readonly double Value => double.IsFinite(sum) ? sum + compensation : sum;

    public void Add(double x)
    {
        double t = sum + x;
        compensation += Math.Abs(sum) >= Math.Abs(x) ? (sum - t) + x : (x - t) + sum;
        sum = t;
    }
}
