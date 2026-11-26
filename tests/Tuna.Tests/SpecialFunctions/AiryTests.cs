using Tuna;

namespace Tuna.Tests;

// 有限参照値は mpmath 1.4.1、100 桁以上で生成し binary64 に丸めた。
// 入力も binary64 の値を正確に高精度へ変換（十進文字列の値とは区別）。
// 参照値生成: https://mpmath.org/doc/current/functions/index.html
[Trait("SpecialFunction", "Airy")]
public class AiryTests
{
    // 値だけを求める経路と4値を同時に求める経路の接続。位相補正・指数の分離も保つ。
    [Theory]
    [InlineData(-1000000)]
    [InlineData(-12)]
    [InlineData(-3)]
    [InlineData(-1.0000000000000002)]
    [InlineData(-1)]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(1.0000000000000002)]
    [InlineData(3)]
    [InlineData(12)]
    [InlineData(104)]
    [InlineData(104.4)]
    [InlineData(105)]
    [InlineData(200)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void Airy_IndividualValues_AgreeWithJointEvaluation(double x)
    {
        var joint = XMath.Special.Airy(x);
        Assert.Equal(joint.Ai, XMath.Special.AiryAi(x));
        Assert.Equal(joint.Bi, XMath.Special.AiryBi(x));
    }

    // https://dlmf.nist.gov/9.2; https://dlmf.nist.gov/9.6; https://dlmf.nist.gov/9.7
    [Theory]
    // 原点の既知値で正規化と導関数の符号を確認。DLMF 9.2(ii)。
    [InlineData(0.0, 0.3550280538878172)]
    // Maclaurin 展開の負側切替点そのもの。
    [InlineData(-1.0, 0.5355608832923521)]
    // 負側切替点の直外で Bessel 表現との接続を確認。
    [InlineData(-1.0000000000000002, 0.5355608832923521)]
    // Maclaurin 展開の正側切替点そのもの。
    [InlineData(1.0, 0.13529241631288141)]
    // 正側切替点の直外で修正 Bessel 表現との接続を確認。
    [InlineData(1.0000000000000002, 0.1352924163128814)]
    // 負の引数で振動する領域と導関数の符号を確認。DLMF 9.7。
    [InlineData(-10.0, 0.04024123848644319)]
    // Ai の減衰と Bi の増大を同時に確認。DLMF 9.7。
    [InlineData(10.0, 1.1047532552898686e-10)]
    // 指数因子の分離が必要な大きい正の引数。DLMF 9.7。
    [InlineData(100.0, 2.6344821520881846e-291)]
    // 非常に小さい Ai と大きい Bi が共に有限な領域。DLMF 9.7。
    [InlineData(103.0, 1.9562320229339223e-304)]
    // Ai は subnormal、Bi は有限。指数因子の分離と underflow を確認。DLMF 9.7。
    [InlineData(104.0, 7.448752158292227e-309)]
    public void AiryAi_MatchesReference(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.AiryAi(x), 2e-12, double.Epsilon);
    }

    // https://dlmf.nist.gov/9.2; https://dlmf.nist.gov/9.6; https://dlmf.nist.gov/9.7
    [Theory]
    // 原点の既知値で正規化と導関数の符号を確認。DLMF 9.2(ii)。
    [InlineData(0.0, -0.2588194037928068)]
    // Maclaurin 展開の負側切替点そのもの。
    [InlineData(-1.0, -0.01016056711664521)]
    // 負側切替点の直外で Bessel 表現との接続を確認。
    [InlineData(-1.0000000000000002, -0.01016056711664509)]
    // Maclaurin 展開の正側切替点そのもの。
    [InlineData(1.0, -0.1591474412967932)]
    // 正側切替点の直外で修正 Bessel 表現との接続を確認。
    [InlineData(1.0000000000000002, -0.15914744129679317)]
    // 負の引数で振動する領域と導関数の符号を確認。DLMF 9.7。
    [InlineData(-10.0, 0.99626504413279)]
    // Ai の減衰と Bi の増大を同時に確認。DLMF 9.7。
    [InlineData(10.0, -3.5206336767389237e-10)]
    // 指数因子の分離が必要な大きい正の引数。DLMF 9.7。
    [InlineData(100.0, -2.6351403616044097e-290)]
    // 非常に小さい Ai と大きい Bi が共に有限な領域。DLMF 9.7。
    [InlineData(103.0, -1.985833197807815e-303)]
    public void AiryAiPrime_MatchesReference(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.AiryAiPrime(x), 2e-12, double.Epsilon);
    }

    // https://dlmf.nist.gov/9.2; https://dlmf.nist.gov/9.6; https://dlmf.nist.gov/9.7
    [Theory]
    // 原点の既知値で正規化と導関数の符号を確認。DLMF 9.2(ii)。
    [InlineData(0.0, 0.6149266274460007)]
    // Maclaurin 展開の負側切替点そのもの。
    [InlineData(-1.0, 0.1039973894969446)]
    // 負側切替点の直外で Bessel 表現との接続を確認。
    [InlineData(-1.0000000000000002, 0.10399738949694448)]
    // Maclaurin 展開の正側切替点そのもの。
    [InlineData(1.0, 1.2074235949528713)]
    // 正側切替点の直外で修正 Bessel 表現との接続を確認。
    [InlineData(1.0000000000000002, 1.2074235949528715)]
    // 負の引数で振動する領域と導関数の符号を確認。DLMF 9.7。
    [InlineData(-10.0, -0.3146798296438386)]
    // Ai の減衰と Bi の増大を同時に確認。DLMF 9.7。
    [InlineData(10.0, 455641153.54822516)]
    // 指数因子の分離が必要な大きい正の引数。DLMF 9.7。
    [InlineData(100.0, 6.041223996670201e+288)]
    // 非常に小さい Ai と大きい Bi が共に有限な領域。DLMF 9.7。
    [InlineData(103.0, 8.01643392902487e+301)]
    public void AiryBi_MatchesReference(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.AiryBi(x), 2e-12, double.Epsilon);
    }

    // https://dlmf.nist.gov/9.2; https://dlmf.nist.gov/9.6; https://dlmf.nist.gov/9.7
    [Theory]
    // 原点の既知値で正規化と導関数の符号を確認。DLMF 9.2(ii)。
    [InlineData(0.0, 0.4482883573538264)]
    // Maclaurin 展開の負側切替点そのもの。
    [InlineData(-1.0, 0.5923756264227924)]
    // 負側切替点の直外で Bessel 表現との接続を確認。
    [InlineData(-1.0000000000000002, 0.5923756264227924)]
    // Maclaurin 展開の正側切替点そのもの。
    [InlineData(1.0, 0.9324359333927756)]
    // 正側切替点の直外で修正 Bessel 表現との接続を確認。
    [InlineData(1.0000000000000002, 0.9324359333927759)]
    // 負の引数で振動する領域と導関数の符号を確認。DLMF 9.7。
    [InlineData(-10.0, 0.11941411339990923)]
    // Ai の減衰と Bi の増大を同時に確認。DLMF 9.7。
    [InlineData(10.0, 1429236134.4828658)]
    // 指数因子の分離が必要な大きい正の引数。DLMF 9.7。
    [InlineData(100.0, 6.039712745310603e+289)]
    // 非常に小さい Ai と大きい Bi が共に有限な領域。DLMF 9.7。
    [InlineData(103.0, 8.133844967032322e+302)]
    public void AiryBiPrime_MatchesReference(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.AiryBiPrime(x), 2e-12, double.Epsilon);
    }

    // https://dlmf.nist.gov/9.9
    [Theory]
    // 最初の零点の左隣。桁落ちするため絶対誤差で比較。
    [InlineData(-2.3381074104597674, -2.839669667936813e-16)]
    // 最初の零点を binary64 へ丸めた位置（厳密な零点ではない）。
    [InlineData(-2.338107410459767, 2.743319340666283e-17)]
    // 最初の零点の右隣。左右の残差を比較。
    [InlineData(-2.3381074104597666, 3.38833353607007e-16)]
    public void AiryAi_NearZero_MatchesReference(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.AiryAi(x), 5e-13, 2e-14);
    }

    // https://dlmf.nist.gov/9.9
    [Theory]
    // 最初の零点の左隣。桁落ちするため絶対誤差で比較。
    [InlineData(-1.0187929716474713, 1.0992802669238472e-16)]
    // 最初の零点を binary64 へ丸めた位置（厳密な零点ではない）。
    [InlineData(-1.018792971647471, -1.1246873724687218e-17)]
    // 最初の零点の右隣。左右の残差を比較。
    [InlineData(-1.0187929716474708, -1.3242177414175914e-16)]
    public void AiryAiPrime_NearZero_MatchesReference(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.AiryAiPrime(x), 5e-13, 2e-14);
    }

    // https://dlmf.nist.gov/9.9
    [Theory]
    // 最初の零点の左隣。桁落ちするため絶対誤差で比較。
    [InlineData(-1.1737132227091283, -1.9964674330776938e-16)]
    // 最初の零点を binary64 へ丸めた位置（厳密な零点ではない）。
    [InlineData(-1.173713222709128, -6.5985241890579e-17)]
    // 最初の零点の右隣。左右の残差を比較。
    [InlineData(-1.1737132227091278, 6.767625952661137e-17)]
    public void AiryBi_NearZero_MatchesReference(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.AiryBi(x), 5e-13, 2e-14);
    }

    // https://dlmf.nist.gov/9.9
    [Theory]
    // 最初の零点の左隣。桁落ちするため絶対誤差で比較。
    [InlineData(-2.2944396826141236, -3.3305086672782557e-16)]
    // 最初の零点を binary64 へ丸めた位置（厳密な零点ではない）。
    [InlineData(-2.294439682614123, 1.3050830085951108e-16)]
    // 最初の零点の右隣。左右の残差を比較。
    [InlineData(-2.2944396826141227, 5.940674684468477e-16)]
    public void AiryBiPrime_NearZero_MatchesReference(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.AiryBiPrime(x), 5e-13, 2e-14);
    }

    // 同時計算 API も既知の初期値で独立に確認。https://dlmf.nist.gov/9.2#ii
    [Fact]
    public void Airy_AtOrigin()
    {
        var a = XMath.Special.Airy(0);
        SpecialFunctionAssert.Close(0.3550280538878172, a.Ai);
        SpecialFunctionAssert.Close(-0.2588194037928068, a.AiPrime);
        SpecialFunctionAssert.Close(0.6149266274460007, a.Bi);
        SpecialFunctionAssert.Close(0.4482883573538264, a.BiPrime);
    }

    // W(Ai,Bi)=1/π は導関数の符号・正規化に敏感。https://dlmf.nist.gov/9.2#iv
    [Theory]
    [InlineData(-10.0)]
    [InlineData(-1.0)]
    [InlineData(0.0)]
    [InlineData(1.0)]
    [InlineData(10.0)]
    public void Airy_Wronskian(double x)
    {
        var a = XMath.Special.Airy(x);
        SpecialFunctionAssert.Close(1 / Math.PI, a.Ai * a.BiPrime - a.AiPrime * a.Bi, 2e-12);
    }

    // NaN と無限大は公開 API の拡張規約。負側の導関数には極限がない。https://dlmf.nist.gov/9.7
    [Fact]
    public void Airy_NonFiniteInputs()
    {
        var nan = XMath.Special.Airy(double.NaN);
        Assert.True(double.IsNaN(nan.Ai) && double.IsNaN(nan.AiPrime) && double.IsNaN(nan.Bi) && double.IsNaN(nan.BiPrime));
        var positive = XMath.Special.Airy(double.PositiveInfinity);
        Assert.Equal(0, positive.Ai);
        Assert.Equal(0, positive.AiPrime);
        Assert.Equal(double.PositiveInfinity, positive.Bi);
        Assert.Equal(double.PositiveInfinity, positive.BiPrime);
        var negative = XMath.Special.Airy(double.NegativeInfinity);
        Assert.Equal(0, negative.Ai);
        Assert.Equal(0, negative.Bi);
        Assert.True(double.IsNaN(negative.AiPrime) && double.IsNaN(negative.BiPrime));
        Assert.True(double.IsNaN(XMath.Special.AiryAi(double.NaN)));
        Assert.True(double.IsNaN(XMath.Special.AiryAiPrime(double.NaN)));
        Assert.True(double.IsNaN(XMath.Special.AiryBi(double.NaN)));
        Assert.True(double.IsNaN(XMath.Special.AiryBiPrime(double.NaN)));
    }

    // R10: Bi と Bi' は正の大引数で正に発散する。https://dlmf.nist.gov/9.7
    [Trait("Category", "ReviewRegression")]
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AiryBi_LargeFiniteArgument_HasPositiveOverflow(bool derivative)
    {
        double actual = derivative ? XMath.Special.AiryBiPrime(1e300) : XMath.Special.AiryBi(1e300);
        Assert.Equal(double.PositiveInfinity, actual);
    }

    // 探索で得た補強点。mpmath 1.4.1で独立に計算し、複数精度でbinary64丸めを確認。
    // 入力・期待値・許容誤差をここに固定する。出典と選定方針は README.md を参照。

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(58.74810967908596, 4.32688589646773e-132)]
    [InlineData(68.72568633545424, 1.0806432911700604e-166)]
    [InlineData(45.462180646962594, 1.9311579744826938e-90)]
    [InlineData(-78.14045522731223, 0.09695700372818877)]
    [InlineData(80.57835412585729, 3.569270439642528e-211)]
    [InlineData(-54.00996651486022, 0.20772541757589016)]
    public void AiryAi_ExploredRelativeValues(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.AiryAi(x), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(58.74810967908596, 4.798970443727584e+129)]
    [InlineData(68.72568633545424, 1.7765540958253302e+164)]
    [InlineData(45.462180646962594, 1.2223003056988672e+88)]
    [InlineData(-78.14045522731223, -0.1631208713638583)]
    [InlineData(80.57835412585729, 4.967428331477947e+208)]
    [InlineData(-54.00996651486022, 0.012751474187617592)]
    public void AiryBi_ExploredRelativeValues(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.AiryBi(x), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(58.74810967908596, -3.318280622903909e-131)]
    [InlineData(68.72568633545424, -8.962562977844837e-166)]
    [InlineData(-74.18314411946564, -1.442511709689944)]
    [InlineData(45.462180646962594, -1.3031555900370317e-89)]
    [InlineData(-78.14045522731223, 1.4422517075395713)]
    [InlineData(80.57835412585729, -3.2050779886692836e-210)]
    // 前回の探索で最大相対誤差となった入力。
    [InlineData(-54.00996651486022, -0.09275104688301052)]
    public void AiryAiPrime_ExploredRelativeValues(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.AiryAiPrime(x), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(58.74810967908596, 3.676236990223087e+130)]
    [InlineData(68.72568633545424, 1.472133383974605e+165)]
    [InlineData(-74.18314411946564, -0.8128671104741096)]
    [InlineData(45.462180646962594, 8.234703772426705e+88)]
    [InlineData(-78.14045522731223, 0.8565500967495157)]
    [InlineData(80.57835412585729, 4.45749171487839e+209)]
    public void AiryBiPrime_ExploredRelativeValues(double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.AiryBiPrime(x), relativeTolerance: 2e-12);
    }

    // 負の大きな x における位相精度（|x| ≤ 1e12）の検証。零点付近でも判定できるよう、誤差は振幅（Ai・Bi は |x|^(-1/4)/√π、導関数は |x|^(1/4)/√π）に対して測る。
    // 参照値は mpmath（72桁・132桁で丸め一致）。
    [Trait("Category", "AuditRegression")]
    [Theory]
    [InlineData(-300.5, -0.11758430476170201, -1.1676370226247388, 0.06735180354409188, -2.038260355503009)]
    [InlineData(-3000.5, 0.016307855744202925, -4.078973005308891, 0.07446533716956553, 0.8932986817158136)]
    [InlineData(-100000.5, 0.01809646982749932, -8.240769948627987, 0.026059537705434812, 5.722620597987425)]
    [InlineData(-10000000.25, -0.005170307470086836, -27.189399674205667, 0.008598043010796587, -16.349948013016686)]
    // ζ の倍々精度の下位 ζ_lo が 1 を超える範囲（|x| ≳ 1e11）
    [InlineData(-1e8, -0.005554128800056995, 9.912829519132075, -0.000991282951914596, -55.54128800057242)]
    [InlineData(-1e10, 0.00017362064481528185, -177.56561416929327, 0.0017756561416929328, 17.36206448152823)]
    [InlineData(-1e11, 0.00024493432278036264, 307.6672585137824, -0.0009729292983633836, 77.45503371368117)]
    [InlineData(-1e12, -0.00042921836070697696, 366.1713874892549, -0.0003661713874892549, -429.21836070697697)]
    public void Airy_LargeNegativeArgument_MatchesReferenceRelativeToAmplitude(double x, double ai, double aiPrime, double bi, double biPrime)
    {
        double amplitude = Math.Pow(-x, -0.25) / Math.Sqrt(Math.PI);
        double primeAmplitude = Math.Pow(-x, 0.25) / Math.Sqrt(Math.PI);
        var value = XMath.Special.Airy(x);
        SpecialFunctionAssert.Close(ai, value.Ai, 0, 4e-15 * amplitude);
        SpecialFunctionAssert.Close(aiPrime, value.AiPrime, 0, 4e-15 * primeAmplitude);
        SpecialFunctionAssert.Close(bi, value.Bi, 0, 4e-15 * amplitude);
        SpecialFunctionAssert.Close(biPrime, value.BiPrime, 0, 4e-15 * primeAmplitude);
    }

    // 振幅 Ai² + Bi² = 1/(π√|x|)・(1 + O(ζ⁻²))、Ai'² + Bi'² = √|x|/π・(1 + O(ζ⁻²)) は位相によらず保たれ、値が発散しない。
    // 倍々精度の位相計算がオーバーフローしない範囲で確認する。
    [Trait("Category", "AuditRegression")]
    [Theory]
    [InlineData(-1e13)]
    [InlineData(-1e15)]
    [InlineData(-1e20)]
    [InlineData(-1e50)]
    [InlineData(-1e100)]
    [InlineData(-1e200)]
    [InlineData(-1e205)]
    [InlineData(-2.0e205)]
    // NaN になる最初の double の、0 に近い側の隣接値。
    [InlineData(-2.0065826040452472e205)]
    public void Airy_HugeNegativeArgument_KeepsAmplitude(double x)
    {
        var value = XMath.Special.Airy(x);
        double root = Math.Sqrt(-x);
        Assert.Equal(1.0, (value.Ai * value.Ai + value.Bi * value.Bi) * Math.PI * root, 1e-13);
        Assert.Equal(1.0, (value.AiPrime * value.AiPrime + value.BiPrime * value.BiPrime) * Math.PI / root, 1e-13);
    }

    // 2|x|^(3/2) がオーバーフローして位相を決められない有限の負数では、同時計算・個別の API とも NaN。
    [Trait("Category", "AuditRegression")]
    [Theory]
    [InlineData(-2.0065826040452475e205)]
    [InlineData(-2.01e205)]
    [InlineData(-1e206)]
    [InlineData(-1e250)]
    [InlineData(-1e300)]
    [InlineData(-double.MaxValue)]
    public void Airy_NegativeArgumentWithOverflowingPhase_ReturnsNaN(double x)
    {
        var value = XMath.Special.Airy(x);
        Assert.True(double.IsNaN(value.Ai));
        Assert.True(double.IsNaN(value.AiPrime));
        Assert.True(double.IsNaN(value.Bi));
        Assert.True(double.IsNaN(value.BiPrime));
        Assert.True(double.IsNaN(XMath.Special.AiryAi(x)));
        Assert.True(double.IsNaN(XMath.Special.AiryAiPrime(x)));
        Assert.True(double.IsNaN(XMath.Special.AiryBi(x)));
        Assert.True(double.IsNaN(XMath.Special.AiryBiPrime(x)));
    }

    // exp(ζ) が先にオーバーフローする境界（Bi は 104.3・104.4 でも有限、104.5 でオーバーフロー）。参照値は 100/180 桁で一致を確認。
    [Trait("Category", "AuditRegression")]
    [Trait("Category", "CoreAudit20261102")]
    [Theory]
    [InlineData(104.0, 2.095173527033602e+306)]
    [InlineData(104.26599999999999, 3.160797897360514e+307)]
    [InlineData(104.266, 3.1607978973609726e+307)]
    [InlineData(104.26600000000002, 3.160797897361431e+307)]
    [InlineData(104.3, 4.472500738060502e+307)]
    [InlineData(104.4, 1.241898624243905e+308)]
    [InlineData(104.5, double.PositiveInfinity)]
    public void AiryBi_FiniteOverflowBoundary_FixedReferences(double x, double expected)
    {
        double actual = XMath.Special.AiryBi(x);
        if (double.IsInfinity(expected) || expected == 0)
        {
            Assert.Equal(expected, actual);
        }
        else
        {
            SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 2e-12,
                absoluteTolerance: Math.Abs(expected) < 2.2250738585072014e-308 ? 2 * double.Epsilon : 0);
        }

        double simultaneous = XMath.Special.Airy(x).Bi;
        if (double.IsInfinity(expected))
        {
            Assert.Equal(expected, simultaneous);
        }
        else
        {
            SpecialFunctionAssert.Close(expected, simultaneous, relativeTolerance: 2e-12);
        }
    }

    // Bi と Bi′ のオーバーフロー境界の違いを同時 API と個別 API の両方で検証する（参照値は 100/180 桁で一致を確認）。
    [Trait("Category", "AuditRegression")]
    [Trait("Category", "CoreAudit20261102")]
    [Theory]
    [InlineData(104.0, 2.1361621950432753e+307)]
    [InlineData(104.26599999999999, double.PositiveInfinity)]
    [InlineData(104.266, double.PositiveInfinity)]
    [InlineData(104.26600000000002, double.PositiveInfinity)]
    [InlineData(104.3, double.PositiveInfinity)]
    [InlineData(104.4, double.PositiveInfinity)]
    [InlineData(104.5, double.PositiveInfinity)]
    public void AiryBiPrime_FiniteOverflowBoundary_FixedReferences(double x, double expected)
    {
        double actual = XMath.Special.AiryBiPrime(x);
        if (double.IsInfinity(expected) || expected == 0)
        {
            Assert.Equal(expected, actual);
        }
        else
        {
            SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 2e-12,
                absoluteTolerance: Math.Abs(expected) < 2.2250738585072014e-308 ? 2 * double.Epsilon : 0);
        }

        double simultaneous = XMath.Special.Airy(x).BiPrime;
        if (double.IsInfinity(expected))
        {
            Assert.Equal(expected, simultaneous);
        }
        else
        {
            SpecialFunctionAssert.Close(expected, simultaneous, relativeTolerance: 2e-12);
        }
    }
}
