namespace Tuna.Statistics.Distributions;

internal struct CompensatedSum
{
    private double sum;
    private double correction;

    internal readonly double Sum => sum;
    internal readonly double Correction => correction;
    internal readonly double Value => sum + correction;

    internal void Add(double value)
    {
        double next = sum + value;
        correction += DifferenceError(sum, -value, next);
        sum = next;
    }

    internal void AddDifference(double x, double y)
    {
        double difference = x - y;
        Add(difference);
        Add(DifferenceError(x, y, difference));
    }

    // Fast2Sum。差を計算したときの丸め誤差を、絶対値が大きい引数を起点に復元する。
    internal static double DifferenceError(double x, double y, double difference) =>
        Math.Abs(x) >= Math.Abs(y) ? (x - difference) - y : x - (y + difference);
}
