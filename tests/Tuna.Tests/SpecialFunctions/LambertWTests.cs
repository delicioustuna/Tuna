using Tuna;

namespace Tuna.Tests;

// 有限参照値は mpmath 1.4.1、100 桁以上で生成し binary64 に丸めた。
// 入力も binary64 の値を正確に高精度へ変換（十進文字列の値とは区別）。
// 参照値生成: https://mpmath.org/doc/current/functions/index.html
[Trait("SpecialFunction", "LambertW")]
public class LambertWTests
{

    // https://dlmf.nist.gov/4.13
    [Theory]
    // 分岐点-1/eのすぐ内側。binary64入力の厳密なWを参照。
    [InlineData(-0.3678794411714423, -0.9999999846957459)]
    // 分岐点から2ulp内側。
    [InlineData(-0.3678794411714422, -0.9999999768481082)]
    // 分岐点級数の通常域。
    [InlineData(-0.3, -0.4894022271802149)]
    // q=1+ex=0.5 の初期値切替直前。
    [InlineData(-0.1839397205857212, -0.2319609529865345)]
    // 初期値切替直後。
    [InlineData(-0.18393972058572114, -0.2319609529865344)]
    // 0に近い負側で枝の違い。
    [InlineData(-1e-20, -1e-20)]
    // 最小subnormalの負入力。
    [InlineData(-5e-324, -5e-324)]
    // W0の原点。
    [InlineData(0.0, 0.0)]
    // Ω定数の既知値。
    [InlineData(1.0, 0.5671432904097838)]
    // 対数形式へ切替直前。
    [InlineData(2.7182818284590446, 0.9999999999999999)]
    // 対数形式へ切替直後。
    [InlineData(2.7182818284590455, 1.0)]
    // 最大付近の引数でも途中のexpを溢れさせない。
    [InlineData(1e+308, 702.6413620341068)]
    public void LambertW0_MatchesReference(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.LambertW0(x), 5e-13, double.Epsilon);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/4.13
    [Theory]
    // 数学的分岐点の丸め値を-1へ拡張する実装規約。
    [InlineData(-0.36787944117144233, -1.0)]
    // 分岐点より充分外側（丸め許容の外）。
    [InlineData(-0.367879441171443, double.NaN)]
    // NaN。
    [InlineData(double.NaN, double.NaN)]
    // 正の無限大。
    [InlineData(double.PositiveInfinity, double.PositiveInfinity)]
    // 負の無限大。
    [InlineData(double.NegativeInfinity, double.NaN)]
    public void LambertW0_SpecialValues(double x, double expected)
    {
        double actual = XMath.Special.LambertW0(x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // https://dlmf.nist.gov/4.13
    [Theory]
    // 分岐点-1/eのすぐ内側。binary64入力の厳密なWを参照。
    [InlineData(-0.3678794411714423, -1.0000000153042543)]
    // 分岐点から2ulp内側。
    [InlineData(-0.3678794411714422, -1.0000000231518922)]
    // 分岐点級数の通常域。
    [InlineData(-0.3, -1.7813370234216277)]
    // q=1+ex=0.5 の初期値切替直前。
    [InlineData(-0.1839397205857212, -2.6783469900166605)]
    // 初期値切替直後。
    [InlineData(-0.18393972058572114, -2.678346990016661)]
    // 0に近い負側で枝の違い。
    [InlineData(-1e-20, -49.962984276674476)]
    // 最小subnormalの負入力。
    [InlineData(-5e-324, -751.0615595398791)]
    public void LambertWm1_MatchesReference(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.LambertWm1(x), 5e-13, double.Epsilon);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/4.13
    [Theory]
    // 数学的分岐点の丸め値を-1へ拡張する実装規約。
    [InlineData(-0.36787944117144233, -1.0)]
    // 分岐点より充分外側（丸め許容の外）。
    [InlineData(-0.367879441171443, double.NaN)]
    // NaN。
    [InlineData(double.NaN, double.NaN)]
    // 下枝の0端点。
    [InlineData(0.0, double.NegativeInfinity)]
    // 下枝は正入力を拒否。
    [InlineData(1e-20, double.NaN)]
    // 下枝は正の無限大を拒否。
    [InlineData(double.PositiveInfinity, double.NaN)]
    public void LambertWm1_SpecialValues(double x, double expected)
    {
        double actual = XMath.Special.LambertWm1(x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // 主枝は原点の符号を保存する。https://dlmf.nist.gov/4.13
    [Fact]
    public void LambertW0_PreservesNegativeZero()
    {
        Assert.Equal(long.MinValue, BitConverter.DoubleToInt64Bits(XMath.Special.LambertW0(-0.0)));
    }

    // 最小非正規数に絶対許容誤差epsilonを使うだけだと、誤った0も通ってしまう。
    // W0(x)=x+O(x²)で、この2入力の正しい丸め値は入力そのもの。
    // https://dlmf.nist.gov/4.13
    [Fact]
    public void LambertW0_SmallestSubnormalIsNotRoundedToZero()
    {
        Assert.Equal(double.Epsilon, XMath.Special.LambertW0(double.Epsilon));
        Assert.Equal(-double.Epsilon, XMath.Special.LambertW0(-double.Epsilon));
        Assert.True(double.IsNaN(XMath.Special.LambertWm1(double.Epsilon)));
    }

    // 探索で得た補強点。mpmath 1.4.1で独立に計算し、複数精度でbinary64丸めを確認。
    // 入力・期待値・許容誤差をここに固定する。出典と選定方針は README.md を参照。

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(9.952863543907355e+245, 560.1030872355277)]
    [InlineData(6.710533544343361e+253, 578.0979636485291)]
    [InlineData(1.5538698471460177e+55, 122.2766425699223)]
    // 前回の探索で最大相対誤差となった入力。
    [InlineData(2.106511803071556e+239, 544.7625206733488)]
    [InlineData(2.7203055327805275e-229, 2.7203055327805275e-229)]
    [InlineData(-0.22392954375569293, -0.30326041009834165)]
    [InlineData(-0.26188113759976883, -0.384779365277143)]
    public void LambertW0_ExploredRelativeValues(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.LambertW0(x), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(-1.013325371747581e-62, -147.74250935261972)]
    [InlineData(-9.824193127970466e-206, -478.217747293807)]
    [InlineData(-1.6815259483812373e-131, -306.84528917496334)]
    [InlineData(-1.6942833045875347e-266, -618.3874901818909)]
    [InlineData(-1.5022787380720117e-76, -179.7812246465184)]
    [InlineData(-2.3384472195949926e-266, -618.0647408192618)]
    public void LambertWm1_ExploredRelativeValues(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.LambertWm1(x), relativeTolerance: 2e-12);
    }
}
