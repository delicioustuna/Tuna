using Tuna;

namespace Tuna.Tests;

// 有限参照値は mpmath 1.4.1、100 桁以上で生成し binary64 に丸めた。
// 入力も binary64 の値を正確に高精度へ変換（十進文字列の値とは区別）。
// 参照値生成: https://mpmath.org/doc/current/functions/index.html
[Trait("SpecialFunction", "Polygamma")]
public class PolygammaTests
{
    private readonly Xunit.Abstractions.ITestOutputHelper output;

    public PolygammaTests(Xunit.Abstractions.ITestOutputHelper output) => this.output = output;

    // 負の反射領域の回帰テスト（参照値は mpmath 1.4.1 の 160/240 桁で評価）。
    // 偶数階・負の半整数は厳密な反射恒等式 ψ^(n)(x) = ψ^(n)(1 - x) でも確認する。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(8, -2.3, 2047472533.0226316)]
    [InlineData(9, -2.3, 61466921303.20306)]
    [InlineData(170, -0.5, -5.612762894906297e276)]
    [InlineData(172, -0.5, -7.33700365622151e280)]
    [InlineData(172, -2.5, -1.6052211731871423e217)]
    public void Polygamma_NegativeReflectionRegression_MatchesReference(int n, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.Polygamma(n, x), relativeTolerance: 5e-15);
    }

    // 「NaNでない」だけでなく、真のoverflowと符号を確認する。
    // 半整数の偶数階は正側の反射値、-2.3の巨大階数は最近接極の項が正に発散。
    [Trait("Category", "AuditRegression")]
    [Theory]
    [InlineData(171, -0.5, double.PositiveInfinity)]
    [InlineData(172, -0.25, double.PositiveInfinity)]
    [InlineData(172, -0.75, double.NegativeInfinity)]
    [InlineData(1000, -0.5, double.NegativeInfinity)]
    [InlineData(60000, -0.5, double.NegativeInfinity)]
    [InlineData(60000, -2.3, double.PositiveInfinity)]
    [InlineData(60001, -2.3, double.PositiveInfinity)]
    public void Polygamma_NegativeHighOrderOverflow_HasCorrectSign(int n, double x, double expected)
    {
        Assert.Equal(expected, XMath.Special.Polygamma(n, x));
    }

    // 数msのハード制約はCI負荷で揺れるため、秒単位への退行を検出する余裕ある上限。
    // 実測値もTRXへ残す。ウォームアップ後の9回の中央値で一時的な停止を除く。
    [Trait("Category", "Performance")]
    [Fact]
    public void Polygamma_Order60000_AvoidsSecondsLongRegression()
    {
        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(double.PositiveInfinity, XMath.Special.Polygamma(60000, -2.3));
        }

        double[] milliseconds = new double[9];
        for (int i = 0; i < milliseconds.Length; i++)
        {
            long start = System.Diagnostics.Stopwatch.GetTimestamp();
            double actual = XMath.Special.Polygamma(60000, -2.3);
            milliseconds[i] = System.Diagnostics.Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            Assert.Equal(double.PositiveInfinity, actual);
        }

        output.WriteLine("Polygamma(60000, -2.3) milliseconds: " +
            string.Join(", ", milliseconds.Select(v => v.ToString("F6", System.Globalization.CultureInfo.InvariantCulture))));
        Array.Sort(milliseconds);
        double median = milliseconds[milliseconds.Length / 2];
        output.WriteLine($"Median: {median:F6} ms; regression ceiling: 250 ms");
        Assert.True(median < 250, $"Median {median:F6} ms exceeds the 250 ms regression ceiling");
    }

    // https://dlmf.nist.gov/5.15; https://dlmf.nist.gov/5.2
    [Theory]
    // 半整数の既知値。
    [InlineData(0.5, -1.9635100260214235)]
    // 整数の既知値。
    [InlineData(1.0, -0.5772156649015329)]
    // 負側反射公式。
    [InlineData(-0.5, 0.03648997397857652)]
    // 負側の複数の極を越える反射公式。
    [InlineData(-3.25, 4.46627587235028)]
    // 負整数極のすぐ右。
    [InlineData(-1.9999999999999998, -4503599627370495.0)]
    // 負整数極のすぐ左。
    [InlineData(-2.0000000000000004, 2251799813685249.0)]
    // 漸近展開へ切替直前。
    [InlineData(9.999999999999998, 2.251752589066721)]
    // 漸近展開の開始点。
    [InlineData(10.0, 2.251752589066721)]
    // 大引数の漸近領域。
    [InlineData(10000000000.0, 23.025850929890456)]
    // 正の零点と両隣。専用Taylor展開の相対精度を確認。
    [InlineData(1.461632144968362, -3.072790566546293e-16)]
    // 正の零点と両隣。専用Taylor展開の相対精度を確認。
    [InlineData(1.4616321449683622, -9.241265521729427e-17)]
    // 正の零点と両隣。専用Taylor展開の相対精度を確認。
    [InlineData(1.4616321449683625, 1.2245374622004068e-16)]
    // 零点まわりのTaylor領域の外。
    [InlineData(1.2616311449683622, -0.21361631179163493)]
    // 零点まわりのTaylor領域の端近く。
    [InlineData(1.6616311449683623, 0.17765922791220598)]
    public void Digamma_MatchesReference(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.Digamma(x), 5e-13, 0);
    }

    // https://dlmf.nist.gov/5.15; https://dlmf.nist.gov/5.2
    [Theory]
    // 半整数の既知値。
    [InlineData(0.5, 4.934802200544679)]
    // 整数の既知値。
    [InlineData(1.0, 1.6449340668482264)]
    // 負側反射公式。
    [InlineData(-0.5, 8.934802200544679)]
    // 負側の複数の極を越える反射公式。
    [InlineData(-3.25, 19.474085068082154)]
    // 負整数極のすぐ右。
    [InlineData(-1.9999999999999998, 2.028240960365167e+31)]
    // 負整数極のすぐ左。
    [InlineData(-2.0000000000000004, 5.070602400912918e+30)]
    // 漸近展開へ切替直前。
    [InlineData(9.999999999999998, 0.10516633568168576)]
    // 漸近展開の開始点。
    [InlineData(10.0, 0.10516633568168575)]
    // 大引数の漸近領域。
    [InlineData(10000000000.0, 1.00000000005e-10)]
    public void Trigamma_MatchesReference(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.Trigamma(x), 5e-13, 0);
    }

    // https://dlmf.nist.gov/5.15; https://dlmf.nist.gov/5.2
    [Theory]
    // 0階はDigamma。
    [InlineData(0, 0.5, -1.9635100260214235)]
    // 1階はTrigamma。
    [InlineData(1, 0.5, 4.934802200544679)]
    // 偶数階は正のxで負。
    [InlineData(2, 0.5, -16.82879664423432)]
    // 奇数階は正。
    [InlineData(3, 0.5, 97.40909103400244)]
    // cot の高階導関数を用いる反射。
    [InlineData(2, -0.25, 122.69736678366236)]
    // 反射の奇数階。
    [InlineData(3, -0.25, 1555.7633125348507)]
    // 高階反射多項式。
    [InlineData(10, -3.25, 15220204740667.846)]
    // n! ζ(n+1) の高階。
    [InlineData(20, 1.0, -2.4329031685078615e+18)]
    // 階乗テーブル末尾でも有限。
    [InlineData(170, 200.0, -4.23554280011251e-87)]
    // 階乗が溢れるが結果は有限なスケーリング評価。
    [InlineData(171, 200.0, 3.607958931214698e-87)]
    public void Polygamma_MatchesReference(int n, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.Polygamma(n, x), 2e-12, 0);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/5.15; https://dlmf.nist.gov/5.2
    [Theory]
    // +0 の片側極。
    [InlineData(0.0, double.NegativeInfinity)]
    // -0 の片側極。
    [InlineData(-0.0, double.PositiveInfinity)]
    // 整数極には符号なし。
    [InlineData(-1.0, double.NaN)]
    // 対数増大。
    [InlineData(double.PositiveInfinity, double.PositiveInfinity)]
    // 負の無限大。
    [InlineData(double.NegativeInfinity, double.NaN)]
    // NaN。
    [InlineData(double.NaN, double.NaN)]
    public void Digamma_SpecialValues(double x, double expected)
    {
        double actual = XMath.Special.Digamma(x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/5.15; https://dlmf.nist.gov/5.2
    [Theory]
    // 二位の極。
    [InlineData(0.0, double.PositiveInfinity)]
    // 負の整数でも正の二位の極。
    [InlineData(-1.0, double.PositiveInfinity)]
    // 大引数で減衰。
    [InlineData(double.PositiveInfinity, 0.0)]
    // 負の無限大。
    [InlineData(double.NegativeInfinity, double.NaN)]
    // NaN。
    [InlineData(double.NaN, double.NaN)]
    public void Trigamma_SpecialValues(double x, double expected)
    {
        double actual = XMath.Special.Trigamma(x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/5.15; https://dlmf.nist.gov/5.2
    [Theory]
    // 負の階数は未定義。
    [InlineData(-1, 1.0, double.NaN)]
    // 偶数階の極は符号なし。
    [InlineData(2, -1.0, double.NaN)]
    // 奇数階の極は正。
    [InlineData(3, -1.0, double.PositiveInfinity)]
    // 大引数の減衰。
    [InlineData(2, double.PositiveInfinity, 0.0)]
    // 負の無限大。
    [InlineData(2, double.NegativeInfinity, double.NaN)]
    // NaN。
    [InlineData(2, double.NaN, double.NaN)]
    public void Polygamma_SpecialValues(int n, double x, double expected)
    {
        double actual = XMath.Special.Polygamma(n, x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // 170!の表境界、逆数累乗の512項ブロック境界、2の冪の両隣、非零subnormal。
    // 参照値はpolygammaを160桁・220桁で計算。https://dlmf.nist.gov/5.15
    [Theory]
    [InlineData(169, 200.0, 5.001703057472369e-87)]
    [InlineData(170, 200.0, -4.23554280011251e-87)]
    [InlineData(171, 200.0, 3.607958931214698e-87)]
    [InlineData(172, 200.0, -3.0914370658031885e-87)]
    [InlineData(510, 200.0, -2.1508986995093267e-15)]
    [InlineData(511, 200.0, 5.49322362987085e-15)]
    [InlineData(512, 200.0, -1.4056741373166739e-14)]
    [InlineData(513, 200.0, 3.604046776985236e-14)]
    [InlineData(1022, 500.0, -5.341577183102888e-128)]
    [InlineData(1023, 500.0, 1.0925618697160816e-127)]
    [InlineData(1024, 500.0, -2.236903196607818e-127)]
    [InlineData(1025, 500.0, 4.584294872972134e-127)]
    [InlineData(2000, 1000.0, -3.8355185180233863e-268)]
    [InlineData(2001, 1000.0, 7.673672008661624e-268)]
    [InlineData(170, 4000.0, -1.946843301215111e-308)]
    [InlineData(171, 4000.0, 8.27511104833823e-310)]
    [InlineData(171, 127.99999999999999, 6.113899168092384e-54)]
    [InlineData(171, 128.0, 6.113899168092268e-54)]
    [InlineData(171, 128.00000000000003, 6.113899168092035e-54)]
    public void Polygamma_ScaledEvaluationBoundaries_MatchReference(int n, double x, double expected)
    {
        double actual = XMath.Special.Polygamma(n, x);
        Assert.True(actual != 0, $"A nonzero result was lost at n={n}, x={x:R}");
        SpecialFunctionAssert.Close(expected, actual, 2e-12, double.Epsilon);
    }

    [Theory]
    [InlineData(171, double.PositiveInfinity)]
    [InlineData(172, double.NegativeInfinity)]
    public void Polygamma_FinalOverflowHasCorrectSign(int n, double expected)
    {
        Assert.Equal(expected, XMath.Special.Polygamma(n, 1));
    }

    [Fact]
    public void Polygamma_FinalUnderflowPreservesParitySign()
    {
        Assert.Equal(long.MinValue, BitConverter.DoubleToInt64Bits(XMath.Special.Polygamma(2, double.MaxValue)));
        Assert.Equal(0L, BitConverter.DoubleToInt64Bits(XMath.Special.Polygamma(3, double.MaxValue)));
        Assert.True(double.IsNaN(XMath.Special.Polygamma(int.MinValue, 1)));
    }

    // 探索で得た補強点。mpmath 1.4.1で独立に計算し、複数精度でbinary64丸めを確認。
    // 入力・期待値・許容誤差をここに固定する。出典と選定方針は README.md を参照。

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(1.0617311644323756e-7, -9418580.661323896)]
    [InlineData(1384.7460604131504, 7.232910931435963)]
    [InlineData(1.8966877923176462e-8, -52723490.6555725)]
    [InlineData(-21.253941432442673, 6.14462256178266)]
    // 前回の探索で最大相対誤差となった入力。
    [InlineData(-8.709630328754312, -0.21029061516846945)]
    [InlineData(-29.17497567525023, 8.51784419181285)]
    public void Digamma_ExploredRelativeValues(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.Digamma(x), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(1.0617311644323756e-7, 88709650800765.16)]
    [InlineData(0.0036377156904871056, 75570.4305792438)]
    [InlineData(1384.7460604131504, 0.0007224148835296422)]
    [InlineData(1.8966877923176462e-8, 2779766406042596.0)]
    [InlineData(-21.253941432442673, 19.216272570807114)]
    [InlineData(-29.17497567525023, 36.12709121101158)]
    public void Trigamma_ExploredRelativeValues(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.Trigamma(x), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(2, 0.4418360439252094, -24.104125331605385)]
    [InlineData(11, 4.1214670185270705, 1.8016298059083962)]
    [InlineData(22, 0.2747459122570079, -9.022940255973881e+33)]
    [InlineData(11, 266.8488524524185, 7.581763513073322e-21)]
    [InlineData(34, 710.189438263978, -1.0053546818750982e-60)]
    [InlineData(39, 97.91003395765331, 1.4454626965776578e-33)]
    public void Polygamma_ExploredRelativeValues(int n, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.Polygamma(n, x), relativeTolerance: 2e-12);
    }

    // 偶数階で x が負の半整数に近いときの桁落ち回避を検証する（(300, -10.5) は厳密な半整数）。
    // 参照値は mpmath 1.4.1 の psi を binary64 入力のまま (300 + 16n)/(500 + 16n) bit で評価して一致を確認。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(10, -10.499999999999392, 0.19864302922959345)]
    [InlineData(10, -3.5000000000005675, -0.4537801257286397)]
    [InlineData(10, -0.4999999, -9408.98537453762)]
    [InlineData(20, -0.4999999, 4.285770085202432e+19)]
    [InlineData(26, -0.4999999999949048, 2.977903128275437e+25)]
    [InlineData(52, -0.5000000000000278, -4.274883284060063e+72)]
    [InlineData(300, -10.5, -1.6434371399311987e+295)]
    public void Polygamma_NearNegativeHalfInteger_AvoidsCancellation(int n, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.Polygamma(n, x), relativeTolerance: 2e-12);
    }
}
