using System.Numerics;

using Tuna;

namespace Tuna.Tests;

// 有限参照値は mpmath 1.4.1、100 桁以上で生成し binary64 に丸めた。
// 入力も binary64 の値を正確に高精度へ変換（十進文字列の値とは区別）。
// 参照値生成: https://mpmath.org/doc/current/functions/index.html
[Trait("SpecialFunction", "Combinatorics")]
public class CombinatoricsTests
{

    // https://dlmf.nist.gov/5.4; https://dlmf.nist.gov/26.3; https://dlmf.nist.gov/24.2
    [Theory]
    // 空積・既知値、およびテーブル末尾/漸近式への切替を確認。
    [InlineData(0, 1.0)]
    // 空積・既知値、およびテーブル末尾/漸近式への切替を確認。
    [InlineData(1, 1.0)]
    // 空積・既知値、およびテーブル末尾/漸近式への切替を確認。
    [InlineData(20, 2.43290200817664e+18)]
    // 空積・既知値、およびテーブル末尾/漸近式への切替を確認。
    [InlineData(169, 4.269068009004705e+304)]
    // 空積・既知値、およびテーブル末尾/漸近式への切替を確認。
    [InlineData(170, 7.257415615307999e+306)]
    public void Factorial_MatchesReference(int n, double expected)
    {
        // 正しく丸めた値という API の保証を確認するため、許容誤差は0。
        SpecialFunctionAssert.Close(expected, XMath.Special.Factorial(n), 0, 0);
    }

    // https://dlmf.nist.gov/5.4; https://dlmf.nist.gov/26.3; https://dlmf.nist.gov/24.2
    [Theory]
    // 空積・既知値、およびテーブル末尾/漸近式への切替を確認。
    [InlineData(0, 0.0)]
    // 空積・既知値、およびテーブル末尾/漸近式への切替を確認。
    [InlineData(1, 0.0)]
    // 空積・既知値、およびテーブル末尾/漸近式への切替を確認。
    [InlineData(170, 706.5730622457874)]
    // 空積・既知値、およびテーブル末尾/漸近式への切替を確認。
    [InlineData(171, 711.71472580229)]
    // 空積・既知値、およびテーブル末尾/漸近式への切替を確認。
    [InlineData(10000, 82108.92783681436)]
    public void LogFactorial_MatchesReference(int n, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.LogFactorial(n), 5e-13, 0);
    }

    // https://dlmf.nist.gov/5.4; https://dlmf.nist.gov/26.3; https://dlmf.nist.gov/24.2
    [Theory]
    // 空積・既知値、およびテーブル末尾/漸近式への切替を確認。
    [InlineData(-1, 1.0)]
    // 空積・既知値、およびテーブル末尾/漸近式への切替を確認。
    [InlineData(0, 1.0)]
    // 空積・既知値、およびテーブル末尾/漸近式への切替を確認。
    [InlineData(1, 1.0)]
    // 空積・既知値、およびテーブル末尾/漸近式への切替を確認。
    [InlineData(20, 3715891200.0)]
    // 空積・既知値、およびテーブル末尾/漸近式への切替を確認。
    [InlineData(299, 3.753274111571926e+306)]
    // 空積・既知値、およびテーブル末尾/漸近式への切替を確認。
    [InlineData(300, 8.154414069380594e+307)]
    public void DoubleFactorial_MatchesReference(int n, double expected)
    {
        // 整数の二重階乗を正しく binary64 へ丸める保証を確認。
        SpecialFunctionAssert.Close(expected, XMath.Special.DoubleFactorial(n), 0, 0);
    }

    // https://dlmf.nist.gov/5.4; https://dlmf.nist.gov/26.3; https://dlmf.nist.gov/24.2
    [Theory]
    // 空積・既知値、およびテーブル末尾/漸近式への切替を確認。
    [InlineData(0, 1.0)]
    // 空積・既知値、およびテーブル末尾/漸近式への切替を確認。
    [InlineData(1, -0.5)]
    // 空積・既知値、およびテーブル末尾/漸近式への切替を確認。
    [InlineData(2, 0.16666666666666666)]
    // 空積・既知値、およびテーブル末尾/漸近式への切替を確認。
    [InlineData(3, 0.0)]
    // 空積・既知値、およびテーブル末尾/漸近式への切替を確認。
    [InlineData(58, 2.3865427499683627e+32)]
    // 空積・既知値、およびテーブル末尾/漸近式への切替を確認。
    [InlineData(60, -2.1399949257225335e+34)]
    // 空積・既知値、およびテーブル末尾/漸近式への切替を確認。
    [InlineData(62, 2.0500975723478097e+36)]
    // 空積・既知値、およびテーブル末尾/漸近式への切替を確認。
    [InlineData(100, -2.8382249570693707e+78)]
    public void BernoulliNumber_MatchesReference(int n, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.BernoulliNumber(n), 2e-12, 0);
    }

    // https://dlmf.nist.gov/5.4; https://dlmf.nist.gov/26.3; https://dlmf.nist.gov/24.2
    [Theory]
    // 整数で厳密な組合せ。
    [InlineData(5.0, 2.0, 10.0)]
    // 階乗の積が溢れても二項係数は有限。
    [InlineData(170.0, 85.0, 9.144841845131556e+49)]
    // 階乗テーブルの外側。
    [InlineData(171.0, 85.0, 1.8183348319970882e+50)]
    // 非常に大きいが有限な係数。
    [InlineData(1000.0, 500.0, 2.7028824094543655e+299)]
    // 負の上引数で一般化された積。
    [InlineData(-3.0, 2.0, 6.0)]
    // 次数の偶奇による符号。
    [InlineData(-3.0, 3.0, -10.0)]
    // 実数次数のGamma定義。
    [InlineData(2.5, 1.25, 2.588892485704221)]
    public void Binomial_MatchesReference(double n, double k, double expected)
    {
        double actual = XMath.Special.Binomial(n, k);
        if (n == Math.Truncate(n) && k == Math.Truncate(k) && Math.Abs(expected) <= 9007199254740992.0)
        {
            // 計画書の「整数値は 2^53 まで厳密」を、相対許容誤差で隠さず確認する。
            Assert.Equal(expected, actual);
        }
        else
        {
            SpecialFunctionAssert.Close(expected, actual, 2e-12, 0);
        }
    }

    // https://dlmf.nist.gov/5.4; https://dlmf.nist.gov/26.3; https://dlmf.nist.gov/24.2
    [Theory]
    // 整数で厳密な組合せ。
    [InlineData(5.0, 2.0, 2.302585092994046)]
    // 階乗の積が溢れても二項係数は有限。
    [InlineData(170.0, 85.0, 115.0398595442661)]
    // 階乗テーブルの外側。
    [InlineData(171.0, 85.0, 115.72717580451526)]
    // 非常に大きいが有限な係数。
    [InlineData(1000.0, 500.0, 689.4672615678512)]
    // 負の上引数で一般化された積。
    [InlineData(-3.0, 2.0, 1.791759469228055)]
    // 次数の偶奇による符号。
    [InlineData(-3.0, 3.0, 2.302585092994046)]
    // 実数次数のGamma定義。
    [InlineData(2.5, 1.25, 0.951230172562281)]
    // 係数自体が溢れるが対数は有限。
    [InlineData(2000.0, 1000.0, 1382.2679935374802)]
    public void LogBinomial_MatchesReference(double n, double k, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.LogBinomial(n, k), 5e-13, 0);
    }

    // https://dlmf.nist.gov/5.4; https://dlmf.nist.gov/26.3; https://dlmf.nist.gov/24.2
    [Theory]
    // H0=0。
    [InlineData(0.0, 0.0)]
    // 原点近傍でDigammaとの加減による桁落ちを避ける。
    [InlineData(1e-20, 1.6449340668482264e-20)]
    // 原点の負側。
    [InlineData(-1e-20, -1.6449340668482264e-20)]
    // 原点級数の切替。
    [InlineData(0.25, 0.3497621315252675)]
    // 実数への拡張。
    [InlineData(0.5, 0.6137056388801094)]
    // 整数の直接和の末尾。
    [InlineData(64.0, 4.743890903705769)]
    // Digammaへ切替。
    [InlineData(65.0, 4.759275519090385)]
    // 負の非整数への拡張。
    [InlineData(-0.5, -1.3862943611198906)]
    // 大引数。
    [InlineData(10000000000.0, 23.603066594891988)]
    public void HarmonicNumber_MatchesReference(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.HarmonicNumber(x), 5e-13, 0);
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/5.4; https://dlmf.nist.gov/26.3; https://dlmf.nist.gov/24.2
    [Theory]
    // 定義域外の整数。
    [InlineData(-2, double.NaN)]
    // binary64 の有限範囲を超える最初の値。
    [InlineData(171, double.PositiveInfinity)]
    public void Factorial_SpecialValues(int n, double expected)
    {
        double actual = XMath.Special.Factorial(n);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/5.4; https://dlmf.nist.gov/26.3; https://dlmf.nist.gov/24.2
    [Theory]
    // 定義域外の整数。
    [InlineData(-2, double.NaN)]
    // binary64 の有限範囲を超える最初の値。
    [InlineData(301, double.PositiveInfinity)]
    public void DoubleFactorial_SpecialValues(int n, double expected)
    {
        double actual = XMath.Special.DoubleFactorial(n);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/5.4; https://dlmf.nist.gov/26.3; https://dlmf.nist.gov/24.2
    [Theory]
    // 負の階乗は定義域外。
    [InlineData(-1, double.NaN)]
    public void LogFactorial_SpecialValues(int n, double expected)
    {
        double actual = XMath.Special.LogFactorial(n);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/5.4; https://dlmf.nist.gov/26.3; https://dlmf.nist.gov/24.2
    [Theory]
    // 負の添字。
    [InlineData(-1, double.NaN)]
    // B1=-1/2 の規約。
    [InlineData(1, -0.5)]
    // B1以外の奇数は零。
    [InlineData(101, 0.0)]
    public void BernoulliNumber_SpecialValues(int n, double expected)
    {
        double actual = XMath.Special.BernoulliNumber(n);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/5.4; https://dlmf.nist.gov/26.3; https://dlmf.nist.gov/24.2
    [Theory]
    // k>n の整数で0。
    [InlineData(5.0, 6.0, 0.0)]
    // 負のkで0。
    [InlineData(5.0, -1.0, 0.0)]
    // n NaN。
    [InlineData(double.NaN, 2.0, double.NaN)]
    // k NaN。
    [InlineData(5.0, double.NaN, double.NaN)]
    public void Binomial_SpecialValues(double n, double k, double expected)
    {
        double actual = XMath.Special.Binomial(n, k);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/5.4; https://dlmf.nist.gov/26.3; https://dlmf.nist.gov/24.2
    [Theory]
    // k>n の整数で0。
    [InlineData(5.0, 6.0, double.NegativeInfinity)]
    // 負のkで0。
    [InlineData(5.0, -1.0, double.NegativeInfinity)]
    // n NaN。
    [InlineData(double.NaN, 2.0, double.NaN)]
    // k NaN。
    [InlineData(5.0, double.NaN, double.NaN)]
    public void LogBinomial_SpecialValues(double n, double k, double expected)
    {
        double actual = XMath.Special.LogBinomial(n, k);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // 極・端点・定義域外と NaN の規約。https://dlmf.nist.gov/5.4; https://dlmf.nist.gov/26.3; https://dlmf.nist.gov/24.2
    [Theory]
    // x=-1 はDigamma(+0)の極。
    [InlineData(-1.0, double.NegativeInfinity)]
    // 負の整数極の符号なし。
    [InlineData(-2.0, double.NaN)]
    // 正の無限大。
    [InlineData(double.PositiveInfinity, double.PositiveInfinity)]
    // NaN。
    [InlineData(double.NaN, double.NaN)]
    public void HarmonicNumber_SpecialValues(double x, double expected)
    {
        double actual = XMath.Special.HarmonicNumber(x);
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual));
        }
        else
        {
            Assert.Equal(expected, actual);
        }
    }

    // 0の直前とintの端。負の添字を表へ渡したり、整数の符号反転で溢れない。
    [Fact]
    public void Combinatorics_IntegerIndexBoundaries()
    {
        Assert.True(double.IsNaN(XMath.Special.Factorial(-1)));
        Assert.True(double.IsNaN(XMath.Special.Factorial(int.MinValue)));
        Assert.True(double.IsNaN(XMath.Special.LogFactorial(int.MinValue)));
        Assert.True(double.IsNaN(XMath.Special.DoubleFactorial(int.MinValue)));
        Assert.True(double.IsNaN(XMath.Special.BernoulliNumber(int.MinValue)));
        Assert.Equal(double.PositiveInfinity, XMath.Special.Factorial(int.MaxValue));
        Assert.Equal(double.PositiveInfinity, XMath.Special.DoubleFactorial(int.MaxValue));
        Assert.Equal(0, XMath.Special.BernoulliNumber(int.MaxValue));
    }

    // R05: 全段階をBigIntegerで計算。結果 <=2^53 は double に厳密に表現できる。
    [Trait("Category", "ReviewRegression")]
    [Theory]
    [InlineData(5, 2)]
    [InlineData(56, 24)]
    [InlineData(57, 23)]
    public void Binomial_ExactlyRepresentableInteger_IsExact(int n, int k)
    {
        BigInteger expected = BigInteger.One;
        for (int j = 1; j <= k; j++)
        {
            expected = expected * (n - j + 1) / j;
        }

        Assert.True(expected <= new BigInteger(9007199254740992L));
        Assert.Equal((double)expected, XMath.Special.Binomial(n, k));
    }

    // 探索で得た補強点。mpmath 1.4.1で独立に計算し、複数精度でbinary64丸めを確認。
    // 入力・期待値・許容誤差をここに固定する。出典と選定方針は README.md を参照。

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(73, 4.4701154615126844e+105)]
    [InlineData(68, 2.4800355424368305e+96)]
    [InlineData(128, 3.856204823625804e+215)]
    [InlineData(66, 5.443449390774431e+92)]
    [InlineData(63, 1.98260831540444e+87)]
    [InlineData(169, 4.269068009004705e+304)]
    public void Factorial_ExploredExactValues(int n, double expected)
    {
        Assert.Equal(expected, XMath.Special.Factorial(n));
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(73, 243.2688490029827)]
    [InlineData(68, 221.95644181913033)]
    [InlineData(128, 496.40547848721764)]
    [InlineData(66, 213.53224149456327)]
    [InlineData(63, 201.00931639928152)]
    [InlineData(169, 701.437263808737)]
    public void LogFactorial_ExploredRelativeValues(int n, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.LogFactorial(n), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(236, 1.5567007266178815e+230)]
    [InlineData(197, 3.349903854308017e+184)]
    [InlineData(169, 6.659313574418617e+152)]
    [InlineData(252, 2.0180218145843514e+249)]
    [InlineData(28, 1428329123020800.0)]
    [InlineData(8, 384.0)]
    public void DoubleFactorial_ExploredExactValues(int n, double expected)
    {
        Assert.Equal(expected, XMath.Special.DoubleFactorial(n));
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(20, 6, 38760.0)]
    [InlineData(73, 16, 5271759063474612.0)]
    [InlineData(103, 102, 103.0)]
    [InlineData(90, 2, 4005.0)]
    [InlineData(21, 1, 21.0)]
    [InlineData(8, 1, 8.0)]
    public void Binomial_ExploredExactValues(double n, double k, double expected)
    {
        Assert.Equal(expected, XMath.Special.Binomial(n, k));
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(20, 6, 10.565144066004702)]
    [InlineData(98, 42, 64.41440698053293)]
    [InlineData(103, 102, 4.634728988229636)]
    [InlineData(90, 2, 8.29529885950246)]
    [InlineData(21, 1, 3.044522437723423)]
    [InlineData(8, 1, 2.0794415416798357)]
    public void LogBinomial_ExploredRelativeValues(double n, double k, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.LogBinomial(n, k), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(98, 42, 9.436729507885199e+27)]
    [InlineData(96, 19, 5.615132224289221e+19)]
    public void Binomial_ExploredRelativeValues(double n, double k, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.Binomial(n, k), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(196, -3.661423368368119e+209)]
    [InlineData(90, 4.3328896986641194e+66)]
    [InlineData(86, 1.1011910323627977e+62)]
    [InlineData(12, -0.2531135531135531)]
    [InlineData(130, 2.2301817894241627e+116)]
    // 前回の探索で最大相対誤差となった入力。
    [InlineData(230, 4.068580607643398e+261)]
    public void BernoulliNumber_ExploredRelativeValues(int n, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.BernoulliNumber(n), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(2985.7169643912985, 8.578978305579799)]
    [InlineData(0.07552340122971968, 0.11780956499543757)]
    [InlineData(2.407046759469035, 1.6491873959543517)]
    [InlineData(35873.96076188285, 11.064996586855209)]
    [InlineData(411.01047501943276, 6.597050386443737)]
    [InlineData(7.880923488117104e-7, 1.2963592057983662e-06)]
    public void HarmonicNumber_ExploredRelativeValues(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.HarmonicNumber(x), relativeTolerance: 2e-12);
    }

    // 実数の n, k で B(n-k+1, k+1) が非正規化数や 0 になる領域、および負の k（または k > n + 1）での ∞ × 0 を回避できることを検証する（参照値は mpmath）。
    [Trait("Category", "AuditRegression")]
    [Theory]
    [InlineData(667105649.7585777, 40.0, 1.1379919067968059e+305)]
    [InlineData(3499790004.8364396, 33.38707399353931, 1.3112075395179835e+281)]
    [InlineData(4370525.202528875, 47.385612966066056, 4.0340740493561857e+254)]
    [InlineData(200.5, -2.5, 7.273958310196762e-07)]
    [InlineData(0.5, -300.5, 5.4086264676045976e-05)]
    [InlineData(0.5, 300.7, 3.183873590734657e-05)]
    [InlineData(10.25, 13.5, -0.0001628035970644788)]
    public void Binomial_RealArgumentsWithExtremeGammas_MatchesReference(double n, double k, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.Binomial(n, k), 2e-12);
    }

    [Trait("Category", "AuditRegression")]
    [Theory]
    [InlineData(200.5, -2.5, -14.133795035754906)]
    [InlineData(0.5, -300.5, -9.824930292035388)]
    [InlineData(0.5, 300.7, -10.354826899178645)]
    [InlineData(667105649.7585777, 40.0, 702.4177185870849)]
    [InlineData(16970281.823288906, -3.4869422615132617, -58.00601293215368)]
    public void LogBinomial_RealArgumentsWithExtremeGammas_MatchesReference(double n, double k, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.LogBinomial(n, k), 2e-12);
    }

    // C(n, 0) = 1 より ln C(n, 0) = 0。整数の k は Binomial の値の対数をとる。
    [Trait("Category", "AuditRegression")]
    [Theory]
    [InlineData(7.0)]
    [InlineData(0.5)]
    [InlineData(1e10)]
    [InlineData(-3.25)]
    public void LogBinomial_ZeroLowerIndex_IsZero(double n)
    {
        Assert.Equal(0.0, XMath.Special.LogBinomial(n, 0));
        Assert.Equal(Math.Log(Math.Abs(XMath.Special.Binomial(n, 7))), XMath.Special.LogBinomial(n, 7));
    }

    // n - k が小さな非負整数のときは対称性 C(n, k) = C(n, n - k) を利用し、丸めで整数になっただけの差（11 - (-2 + 2⁻⁵¹)）では切り替えない（参照値は mpmath）。
    [Trait("Category", "AuditRegression")]
    [Fact]
    public void Binomial_IntegralComplement_UsesSymmetry()
    {
        double n = -1 + Math.ScaleB(1, -48);
        Assert.Equal(n, XMath.Special.Binomial(n, n - 1));
        Assert.Equal(Math.Log(-n), XMath.Special.LogBinomial(n, n - 1));

        double k = -2 + Math.ScaleB(1, -51);
        SpecialFunctionAssert.Close(-2.8467257041670708085e-18, XMath.Special.Binomial(11, k), 1e-14);
        SpecialFunctionAssert.Close(-40.400362215806746858, XMath.Special.LogBinomial(11, k), 1e-15);
    }

    // 負の整数 n に対する反射公式 C(n, k) = (-1)^k C(k - n - 1, k) の検証。
    [Trait("Category", "AuditRegression")]
    [Theory]
    [InlineData(-1.0, 21.0, -1.0)]
    [InlineData(-1.0, 22.0, 1.0)]
    [InlineData(-2.0, 21.0, -22.0)]
    [InlineData(-3.0, 30.0, 496.0)]
    [InlineData(-5.0, 25.0, -23751.0)]
    [InlineData(-5.0, 3.0, -35.0)]
    [InlineData(-3.0, -5.0, 6.0)]   // k ≤ n < 0 は C(n, n - k) = C(-3, 2)
    [InlineData(-3.0, -2.0, 0.0)]
    public void Binomial_NegativeIntegerUpperIndex_IsExact(double n, double k, double expected)
    {
        Assert.Equal(expected, XMath.Special.Binomial(n, k));
        Assert.Equal(expected == 0 ? double.NegativeInfinity : Math.Log(Math.Abs(expected)), XMath.Special.LogBinomial(n, k));
    }

    // n < -1 の非整数で Γ の比と 1/Γ がオーバーフローまたは 0 になる領域を反射公式で評価する（参照値は mpmath 300 桁）。
    [Trait("Category", "AuditRegression")]
    [Theory]
    [InlineData(-151.0343082041763, 42.0, 4.4598450666836940254e+42, 98.203687932766771033)]
    [InlineData(-285.50318766282516, 57.0, -4.4043019969516776278e+65, 151.15061283445380318)]
    [InlineData(-325.6493859240642, 0.034585637969816035, 1.1690967538043654032, 0.15623144536470392983)]
    [InlineData(-700.3257308068358, 4.079938427700974, 16943498845.289814465, 23.553150048248704954)]
    [InlineData(-185.8854477214161, 393.14254263875335, 1.9329502437712651976e+155, 357.55973687355360137)]
    public void Binomial_NegativeRealUpperIndexWithExtremeGammas_MatchesReference(double n, double k, double expected, double expectedLog)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.Binomial(n, k), 1e-13);
        SpecialFunctionAssert.Close(expectedLog, XMath.Special.LogBinomial(n, k), 1e-15, 1e-13);
    }

    // 両方の引数が負の非整数でも反射公式で正の引数のベータ関数に帰着して評価する（対称性 C(n, k) = C(n, n - k) を確認、参照値は mpmath 300 桁）。
    [Trait("Category", "AuditRegression")]
    [Theory]
    [InlineData(-6993.559647587758, -83.72029444035743, -3.0341801706905292632e-199, -457.10449224277236177)]
    [InlineData(-3.5, -123456789.5, 50957864953413361642.0, 45.377530787760470464)]
    [InlineData(-3.5, 123456786.0, 50957864953413361642.0, 45.377530787760470464)]
    [InlineData(-1000000.5, -3.25, 1.2830277218476111419e-20, -45.802479167428925017)]
    public void Binomial_NegativeArgumentsWithExtremeGammas_MatchReference(double n, double k, double expected, double expectedLog)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.Binomial(n, k), 1e-13);
        SpecialFunctionAssert.Close(expectedLog, XMath.Special.LogBinomial(n, k), 1e-15);
    }

    // 負の整数 n と巨大な k で、補数 m = -n - 1 の丸め落ちと符号の偶奇を正確に保つことを確認する（参照値は厳密な整数）。
    [Trait("Category", "AuditRegression")]
    [Theory]
    [InlineData(-1.0, 9007199254740992.0, 1.0, 0.0)]
    [InlineData(-1.0, 9007199254740994.0, 1.0, 0.0)]
    [InlineData(-2.0, 1e16, 1e16, 36.841361487904734)]
    [InlineData(-3.0, 1e16, 5.0000000000000015e31, 72.989575795249516879)]
    [InlineData(-5.0, 1000000000000001.0, -4.166666666666725e58, 134.97705174929480942)]
    [InlineData(-2.0, -1e16, 1e16, 36.841361487904730844)]
    [InlineData(-4.0, -9007199254740992.0, 1.2179180311090849539e47, 108.41864223980324853)]
    public void Binomial_NegativeIntegerUpperIndexWithHugeLowerIndex_KeepsSmallComplement(double n, double k, double expected, double expectedLog)
    {
        SpecialFunctionAssert.WithinUlps(expected, XMath.Special.Binomial(n, k), 1);
        if (expectedLog == 0)
        {
            Assert.Equal(0.0, XMath.Special.LogBinomial(n, k));
        }
        else
        {
            SpecialFunctionAssert.WithinUlps(expectedLog, XMath.Special.LogBinomial(n, k), 2);
        }
    }

    // 丸めた n - k + 1 が 0 以下の整数でも実数としては Γ の極でない点（k は 0.75・40.75 の次の double）の精度検証（参照値は mpmath 300 桁）。
    [Trait("Category", "AuditRegression")]
    [Theory]
    [InlineData(-0.25, 0.7500000000000001, -1.4802973661668752517e-16, -36.449118497225320563)]
    [InlineData(-2.25, 0.7500000000000001, -9.4739031434680031886e-16, -34.592820506859694224)]
    [InlineData(-30.25, 40.75000000000001, -1389824.3699375444281, 14.144687944560799269)]
    [InlineData(-30.25, 40.750000000000014, -2779648.7398750994414, 14.837835125120748387)]
    public void Binomial_RoundedDifferenceAtGammaPole_IsNotPole(double n, double k, double expected, double expectedLog)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.Binomial(n, k), 1e-14);
        SpecialFunctionAssert.Close(expectedLog, XMath.Special.LogBinomial(n, k), 1e-15);
    }

    // 反射公式の係数 sin(π(n - k))/sin(πn) が小さく、C(k - n - 1, k) 単独ではオーバーフローする領域を対数合成で評価する（参照値は mpmath 300 桁）。
    [Trait("Category", "AuditRegression")]
    [Theory]
    [InlineData(-400.375, 400.62499999999983, 1.0893520832753056079e+227, 522.77239921013891902)]
    [InlineData(-400.375, 400.62500000000017, -1.0893520832755623249e+227, 522.77239921013915468)]
    public void Binomial_SmallReflectionFactor_DoesNotOverflowEarly(double n, double k, double expected, double expectedLog)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.Binomial(n, k), 1e-13);
        SpecialFunctionAssert.Close(expectedLog, XMath.Special.LogBinomial(n, k), 1e-15);
    }

    // k 項の積が非正規化数・0 になっても、仮数と指数を分けて掛けて対数は有限（C(±ε, 2) は 0 に丸まるが ln |C| = ln ε - ln 2 + ln(1 ∓ ε)）
    [Trait("Category", "AuditRegression")]
    [Theory]
    [InlineData(double.Epsilon, 2.0, -745.13321910194120762)]
    [InlineData(-double.Epsilon, 2.0, -745.13321910194120762)]
    [InlineData(-1e-310, 5.0, -715.41081674058826548)]
    [InlineData(1e300, 20.0, 13773.17494150352062)]
    public void LogBinomial_UnderflowingOrOverflowingProduct_IsFinite(double n, double k, double expected)
    {
        SpecialFunctionAssert.WithinUlps(expected, XMath.Special.LogBinomial(n, k), 2);
    }

    // 正の引数へ反射した二項係数は double.MaxValue を超えるが、小さい正弦の比を掛けた最終結果は有限。
    // 両側で符号も確認する（参照値は入力を binary64 から正確に読み、mpmath 180 桁・360 桁で一致を確認）。
    [Trait("Category", "ReviewRegression")]
    [Theory]
    [InlineData(520.7499999999995, 5.881392921805950351995730e299)]
    [InlineData(520.7500000000005, -5.881392921809647790074871e299)]
    public void Binomial_ReflectedProduct_DoesNotOverflowBeforeSmallFactor(double k, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.Binomial(-520.25, k), 2e-12);
    }

    // k は 0.5・520.75 の次に大きい double。実数としての n - k + 1 は極ではないが、単純な double の引き算では整数に丸まる。
    [Trait("Category", "ReviewRegression")]
    [Theory]
    [InlineData(-0.5, 0.5000000000000001, -2.220446049250312929557143e-16, -36.04365338911715615783108)]
    [InlineData(-520.25, 520.7500000000001, -1.470348230452065312698799e299, 688.8584420694199420782095)]
    public void Binomial_RoundedDifferenceNearGammaPole_IsNotPole(double n, double k, double expected, double expectedLog)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.Binomial(n, k), 2e-12);
        SpecialFunctionAssert.Close(expectedLog, XMath.Special.LogBinomial(n, k), 2e-12);
    }

    // 負の非整数の n, k でも対称性 C(n, k) = C(n, n - k) を保つ（n - k = 999999999999998 は丸めなしで表せる）。
    [Trait("Category", "ReviewRegression")]
    [Fact]
    public void Binomial_NegativeFractionalArguments_PreserveSymmetry()
    {
        const double n = -1.5;
        const double k = -999999999999999.5;
        Assert.Equal(999999999999998.0, n - k);
        SpecialFunctionAssert.Close(35682482.32305539998922788, XMath.Special.Binomial(n, n - k), 2e-12);
        SpecialFunctionAssert.Close(35682482.32305539998922788, XMath.Special.Binomial(n, k), 2e-12);
        SpecialFunctionAssert.Close(17.39017043509058722748045, XMath.Special.LogBinomial(n, n - k), 2e-12);
        SpecialFunctionAssert.Close(17.39017043509058722748045, XMath.Special.LogBinomial(n, k), 2e-12);
    }

    // C(n, n - 1) = C(n, 1) = n。対数が 0 に近くても、大きな対数同士の引き算で桁落ちしないこと。
    [Trait("Category", "ReviewRegression")]
    [Theory]
    [InlineData(28)]
    [InlineData(48)]
    [InlineData(50)]
    public void LogBinomial_NearMinusOne_PreservesSymmetry(int power)
    {
        double n = -1 + Math.ScaleB(1, -power);
        double expected = Math.Log(-n);
        SpecialFunctionAssert.WithinUlps(expected, XMath.Special.LogBinomial(n, 1), 8);
        SpecialFunctionAssert.WithinUlps(expected, XMath.Special.LogBinomial(n, n - 1), 8);
    }
}
