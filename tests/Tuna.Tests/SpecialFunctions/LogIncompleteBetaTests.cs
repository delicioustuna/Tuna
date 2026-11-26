namespace Tuna.Tests;

[Trait("SpecialFunction", "LogIncompleteBeta")]
public class LogIncompleteBetaTests
{
    [Theory]
    [InlineData(2, 3, 0, double.NegativeInfinity, 0)]
    [InlineData(2, 3, 1, 0, double.NegativeInfinity)]
    [InlineData(0, 3, 0.5, double.NaN, double.NaN)]
    [InlineData(-1, 3, 0.5, double.NaN, double.NaN)]
    [InlineData(2, 0, 0.5, double.NaN, double.NaN)]
    [InlineData(2, -1, 0.5, double.NaN, double.NaN)]
    [InlineData(double.PositiveInfinity, 3, 0.5, double.NaN, double.NaN)]
    [InlineData(2, double.PositiveInfinity, 0.5, double.NaN, double.NaN)]
    [InlineData(double.PositiveInfinity, 3, 0, double.NaN, double.NaN)]
    [InlineData(2, double.PositiveInfinity, 1, double.NaN, double.NaN)]
    [InlineData(2, 3, -double.Epsilon, double.NaN, double.NaN)]
    [InlineData(2, 3, 1.0000000000000002, double.NaN, double.NaN)]
    [InlineData(2, 3, double.PositiveInfinity, double.NaN, double.NaN)]
    [InlineData(2, 3, double.NegativeInfinity, double.NaN, double.NaN)]
    [InlineData(double.NaN, 3, 0.5, double.NaN, double.NaN)]
    [InlineData(2, double.NaN, 0.5, double.NaN, double.NaN)]
    [InlineData(2, 3, double.NaN, double.NaN, double.NaN)]
    public void EndpointsAndDomain(double a, double b, double x, double expectedP, double expectedQ)
    {
        Assert.Equal(expectedP, XMath.Special.LogRegularizedIncompleteBeta(a, b, x));
        Assert.Equal(expectedQ, XMath.Special.LogRegularizedIncompleteBetaComplement(a, b, x));
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(1e-20, 0.5)]
    [InlineData(0.1, 5)]
    [InlineData(2, 20)]
    [InlineData(10000, 20000)]
    public void LogProbabilitiesAreBoundedAndMonotone(double a, double b)
    {
        double previousP = double.NegativeInfinity, previousQ = 0;
        for (int i = -60; i <= 60; i++)
        {
            double x = 1 / (1 + Math.Exp(-i * 0.5));
            double p = XMath.Special.LogRegularizedIncompleteBeta(a, b, x);
            double q = XMath.Special.LogRegularizedIncompleteBetaComplement(a, b, x);
            Assert.True(double.IsFinite(p) && double.IsFinite(q));
            Assert.InRange(p, previousP, 0);
            Assert.InRange(q, double.NegativeInfinity, previousQ);
            double maximum = Math.Max(p, q), minimum = Math.Min(p, q);
            Assert.InRange(Math.Abs(maximum + Math.Log(1 + Math.Exp(minimum - maximum))), 0, 5e-15);
            previousP = p;
            previousQ = q;
        }
    }

    // 探索で得た補強点。mpmath 1.4.1で独立に計算し、複数精度でbinary64丸めを確認。
    // 入力・期待値・許容誤差をここに固定する。出典と選定方針は README.md を参照。

    [Trait("Category", "Accuracy")]
    [Theory]
    // 対数スケールで調べた入力の代表点
    [InlineData(0.01, 0.1, double.Epsilon, -7.538186783894081, 2e-12)]
    [InlineData(0.01, 0.1, 1e-300, -7.001541343662405, 2e-12)]
    [InlineData(2, 3, double.Epsilon, -1487.0883843735344, 2e-12)]
    [InlineData(10, 20, 0.3, -1.010590199672137, 2e-12)]
    [InlineData(100, 200, double.Epsilon, -74256.47817004913, 1.1641532182693481e-10)]
    [InlineData(100, 200, 0.3, -2.2178481989357404, 2e-12)]
    // 極小形状パラメータを反転した入力
    [InlineData(1, double.Epsilon, 0.99, -742.9128922955733, 2e-12)]
    [InlineData(1, double.Epsilon, 0.9, -743.6060394761333, 2e-12)]
    [InlineData(1, double.Epsilon, 0.7, -744.2544451625189, 2e-12)]
    [InlineData(0.5, double.Epsilon, 0.99, -742.6505740589502, 2e-12)]
    [InlineData(0.5, double.Epsilon, 0.9, -743.1489421980798, 2e-12)]
    [InlineData(0.5, double.Epsilon, 0.7, -743.5563580013808, 2e-12)]
    [InlineData(7, double.Epsilon, 0.99, -743.6450783255144, 2e-12)]
    [InlineData(7, double.Epsilon, 0.9, -745.3974748122806, 2e-12)]
    [InlineData(7, double.Epsilon, 0.010000000000000009, -778.6133836313574, 2e-12)]
    [InlineData(1, 1e-323, 0.99, -742.2197451150134, 2e-12)]
    [InlineData(1, 1e-323, 0.9, -742.9128922955733, 2e-12)]
    [InlineData(1, 1e-323, 0.7, -743.561297981959, 2e-12)]
    [InlineData(0.5, 1e-323, 0.99, -741.9574268783903, 2e-12)]
    [InlineData(0.5, 1e-323, 0.9, -742.4557950175199, 2e-12)]
    [InlineData(0.5, 1e-323, 0.7, -742.8632108208209, 2e-12)]
    [InlineData(7, 1e-323, 0.99, -742.9519311449544, 2e-12)]
    [InlineData(7, 1e-323, 0.9, -744.7043276317207, 2e-12)]
    [InlineData(1, 8e-323, 0.99, -740.1403035733335, 2e-12)]
    [InlineData(1, 8e-323, 0.9, -740.8334507538935, 2e-12)]
    [InlineData(1, 8e-323, 0.7, -741.4818564402791, 2e-12)]
    [InlineData(0.5, 8e-323, 0.99, -739.8779853367104, 2e-12)]
    [InlineData(0.5, 8e-323, 0.9, -740.37635347584, 2e-12)]
    [InlineData(0.5, 8e-323, 0.7, -740.7837692791411, 2e-12)]
    [InlineData(7, 8e-323, 0.99, -740.8724896032746, 2e-12)]
    [InlineData(7, 8e-323, 0.9, -742.6248860900408, 2e-12)]
    [InlineData(1, 1e-320, 0.99, -735.300061265166, 2e-12)]
    [InlineData(1, 1e-320, 0.9, -735.9932084457259, 2e-12)]
    [InlineData(1, 1e-320, 0.7, -736.6416141321115, 2e-12)]
    [InlineData(0.5, 1e-320, 0.99, -735.0377430285429, 2e-12)]
    [InlineData(0.5, 1e-320, 0.9, -735.5361111676724, 2e-12)]
    [InlineData(0.5, 1e-320, 0.7, -735.9435269709735, 2e-12)]
    [InlineData(7, 1e-320, 0.99, -736.032247295107, 2e-12)]
    [InlineData(7, 1e-320, 0.9, -737.7846437818733, 2e-12)]
    [InlineData(0.5, 1e-20, 0.99, -44.26220399744984, 2e-12)]
    [InlineData(7, 1e-20, 0.010000000000000009, -80.22501356985701, 2e-12)]
    // 確率のアンダーフローと1近傍の微小対数
    [InlineData(1, 1000, 1e-20, -39.14394658089878, 2e-12)]
    [InlineData(1000, 1, 1e-20, -46051.701859880915, 5.820766091346741e-11)]
    [InlineData(1000, 1, 0.99, -10.05033585350145, 2e-12)]
    [InlineData(2, 3000, 1e-20, -76.78348248723115, 2e-12)]
    [InlineData(3000, 2, 1e-20, -138147.0988787973, 2.3283064365386963e-10)]
    [InlineData(3000, 2, 0.99, -26.717020356019205, 2e-12)]
    // 連分数の向きの切替近傍
    [InlineData(10, 0.01, 0.9159034138218151, -5.8295973081395305, 2e-12)]
    [InlineData(10, 0.01, 0.9159034138218152, -5.829597308139528, 2e-12)]
    [InlineData(10, 0.01, 0.9159034138218153, -5.829597308139526, 2e-12)]
    // 形状パラメータ1と10の近傍
    [InlineData(20, 0.9999999999999999, 0.7, -7.133498878774649, 2e-12)]
    [InlineData(20, 1, 0.7, -7.133498878774649, 2e-12)]
    [InlineData(20, 1.0000000000000002, 0.7, -7.133498878774648, 2e-12)]
    [InlineData(9.999999999999998, 20, 0.3, -1.0105901996721365, 2e-12)]
    [InlineData(10.000000000000002, 20, 0.3, -1.0105901996721374, 2e-12)]
    // 中心付近と漸近展開の閾値
    [InlineData(599.9999999999999, 299.99999999999994, 0.6509619050856152, -1.8403191691449483, 2e-12)]
    [InlineData(600, 300, 0.6509619050856151, -1.840319169144959, 2e-12)]
    [InlineData(600.0000000000001, 300.00000000000006, 0.6509619050856151, -1.8403191691449592, 2e-12)]
    // 極端なパラメータ比
    [InlineData(2, 100000000000000000000.0, 1e-170, -691.4686750787737, 2e-12)]
    // 片方の形状が1の閉形式
    [InlineData(1, 100000000000000000000.0, 1e-300, -644.7238260383328, 2e-12)]
    [InlineData(1, 1e+100, 1e-300, -460.51701859880916, 2e-12)]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(44.66447225354958, 0.00003101320880962474, 0.6935564823937079, -29.387588640767806, 2e-12)]
    [InlineData(0.0000010446942834803446, 1.0304840868608479e-9, 0.4431008210006121, -6.922437041208124, 2e-12)]
    [InlineData(0.000005360261942088599, 0.000003087879628177505, 0.3944407679893276, -1.0064641316309828, 2e-12)]
    [InlineData(0.0001451896509012912, 1.3458845620918837e-9, 0.7821871900193117, -11.5885683374814, 2e-12)]
    [InlineData(1.0490485674755796, 3.210506416568144e-8, 0.13180399609380256, -19.35646696439343, 2e-12)]
    [InlineData(2.163151112687556, 153.78484275640287, 3.3614986150159407e-267, -1317.2096041165191, 2e-12)]
    public void LogRegularizedIncompleteBeta_ExploredAbsoluteValues(
        double a, double b, double x, double expected, double absoluteTolerance)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.LogRegularizedIncompleteBeta(a, b, x), relativeTolerance: 0, absoluteTolerance: absoluteTolerance);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 対数スケールで調べた入力の代表点
    [InlineData(0.01, 0.1, double.Epsilon, -0.0005325037895850892)]
    [InlineData(0.01, 0.1, 1e-300, -0.000910892261126496)]
    [InlineData(0.01, 0.1, 1e-20, -0.8544256590249395)]
    [InlineData(1, 1, 1e-300, -1e-300)]
    [InlineData(1, 1, 0.5, -0.6931471805599453)]
    [InlineData(100, 200, 0.3, -0.11523473418689843)]
    // 極小形状パラメータを反転した入力
    [InlineData(1, 1e-300, 0.99, -4.6051701859880905e-300)]
    [InlineData(0.5, 1e-300, 0.99, -5.986445692252761e-300)]
    [InlineData(7, 1e-300, 0.30000000000000004, -4.244393873237896e-305)]
    [InlineData(1, 1e-20, 0.99, -4.6051701859880903e-20)]
    [InlineData(0.5, 1e-20, 0.99, -5.986445692252761e-20)]
    [InlineData(7, 1e-20, 0.010000000000000009, -1.4411835488575674e-35)]
    // 両形状が極小: 対称性と逆数オーバーフロー境界
    [InlineData(double.Epsilon, double.Epsilon, 0.3, -0.6931471805599453)]
    [InlineData(double.Epsilon, double.Epsilon, 0.5, -0.6931471805599453)]
    [InlineData(double.Epsilon, double.Epsilon, 0.7, -0.6931471805599453)]
    [InlineData(1e-310, 1e-310, 0.3, -0.6931471805599453)]
    [InlineData(1e-310, 1e-310, 0.5, -0.6931471805599453)]
    [InlineData(1e-310, 1e-310, 0.7, -0.6931471805599453)]
    [InlineData(1e-300, 1e-300, 0.3, -0.6931471805599453)]
    [InlineData(1e-300, 1e-300, 0.5, -0.6931471805599453)]
    [InlineData(1e-300, 1e-300, 0.7, -0.6931471805599453)]
    [InlineData(1e-20, 1e-20, 0.3, -0.6931471805599453)]
    [InlineData(1e-20, 1e-20, 0.5, -0.6931471805599453)]
    [InlineData(1e-20, 1e-20, 0.7, -0.6931471805599453)]
    [InlineData(5.562684646268e-309, 5.562684646268e-309, 0.3, -0.6931471805599453)]
    [InlineData(5.562684646268e-309, 5.562684646268e-309, 0.5, -0.6931471805599453)]
    [InlineData(5.562684646268e-309, 5.562684646268e-309, 0.7, -0.6931471805599453)]
    [InlineData(5.562684646268003e-309, 5.562684646268003e-309, 0.3, -0.6931471805599453)]
    [InlineData(5.562684646268003e-309, 5.562684646268003e-309, 0.5, -0.6931471805599453)]
    [InlineData(5.562684646268003e-309, 5.562684646268003e-309, 0.7, -0.6931471805599453)]
    [InlineData(5.56268464626801e-309, 5.56268464626801e-309, 0.3, -0.6931471805599453)]
    [InlineData(5.56268464626801e-309, 5.56268464626801e-309, 0.5, -0.6931471805599453)]
    [InlineData(5.56268464626801e-309, 5.56268464626801e-309, 0.7, -0.6931471805599453)]
    [InlineData(1.1125369292536e-308, 1.1125369292536e-308, 0.3, -0.6931471805599453)]
    [InlineData(1.1125369292536e-308, 1.1125369292536e-308, 0.5, -0.6931471805599453)]
    [InlineData(1.1125369292536e-308, 1.1125369292536e-308, 0.7, -0.6931471805599453)]
    [InlineData(1.1125369292536007e-308, 1.1125369292536007e-308, 0.3, -0.6931471805599453)]
    [InlineData(1.1125369292536007e-308, 1.1125369292536007e-308, 0.5, -0.6931471805599453)]
    [InlineData(1.1125369292536007e-308, 1.1125369292536007e-308, 0.7, -0.6931471805599453)]
    [InlineData(1.112536929253601e-308, 1.112536929253601e-308, 0.3, -0.6931471805599453)]
    [InlineData(1.112536929253601e-308, 1.112536929253601e-308, 0.5, -0.6931471805599453)]
    [InlineData(1.112536929253601e-308, 1.112536929253601e-308, 0.7, -0.6931471805599453)]
    // 確率のアンダーフローと1近傍の微小対数
    [InlineData(1, 1000, 1e-20, -9.999999999999999e-18)]
    [InlineData(1000, 1, 0.99, -4.317217931578046e-05)]
    [InlineData(2, 3000, 1e-20, -4.5014999999999995e-34)]
    [InlineData(3000, 2, 0.99, -2.4942816200548864e-12)]
    // 連分数の向きの切替近傍
    [InlineData(10, 0.01, 0.9159034138218151, -0.0029435884669273796)]
    [InlineData(10, 0.01, 0.9159034138218152, -0.0029435884669273857)]
    [InlineData(10, 0.01, 0.9159034138218153, -0.0029435884669273913)]
    [InlineData(10, 20, 0.34374999999999994, -0.8314703968131769)]
    [InlineData(10, 20, 0.34375, -0.8314703968131775)]
    [InlineData(10, 20, 0.34375000000000006, -0.8314703968131781)]
    // 形状パラメータ1と10の近傍
    [InlineData(20, 0.9999999999999999, 0.7, -0.0007982411727061897)]
    [InlineData(20, 1, 0.7, -0.0007982411727061899)]
    [InlineData(20, 1.0000000000000002, 0.7, -0.0007982411727061904)]
    [InlineData(9.999999999999998, 20, 0.3, -0.4525631324423893)]
    [InlineData(10.000000000000002, 20, 0.3, -0.45256313244238877)]
    // 中心付近と漸近展開の閾値
    [InlineData(599.9999999999999, 299.99999999999994, 0.6509619050856152, -0.17288630261742938)]
    [InlineData(599.9999999999999, 299.99999999999994, 0.6666666666666667, -0.6868969187335309)]
    [InlineData(600, 300, 0.6509619050856151, -0.17288630261742735)]
    [InlineData(600, 300, 0.6666666666666666, -0.6868969187335253)]
    [InlineData(600.0000000000001, 300.00000000000006, 0.6509619050856151, -0.17288630261742732)]
    [InlineData(600.0000000000001, 300.00000000000006, 0.6666666666666666, -0.6868969187335253)]
    // 極端なパラメータ比
    [InlineData(2, 100000000000000000000.0, 1e-170, -5e-301)]
    // 片方の形状が1の閉形式
    [InlineData(1, 100000000000000000000.0, 1e-300, -1.0000000000000001e-280)]
    [InlineData(1, 1e+100, 1e-300, -1e-200)]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(44.66447225354958, 0.00003101320880962474, 0.6935564823937079, -1.7263641821174207e-13)]
    [InlineData(0.0000010446942834803446, 1.0304840868608479e-9, 0.4431008210006121, -0.000985911340226549)]
    [InlineData(0.000005360261942088599, 0.000003087879628177505, 0.3944407679893276, -0.4549323172667588)]
    [InlineData(0.28026973671865224, 0.014887579185175331, 4.397702072399194e-257, -7.1811446930162e-74)]
    [InlineData(0.0001451896509012912, 1.3458845620918837e-9, 0.7821871900193117, -9.271515552507115e-06)]
    [InlineData(7.53447350773006e-12, 7.582550216847525e-12, 1.7568589852996916e-155, -0.6963325483191795)]
    public void LogRegularizedIncompleteBetaComplement_ExploredRelativeValues(double a, double b, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.LogRegularizedIncompleteBetaComplement(a, b, x), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 対数スケールで調べた入力の代表点
    [InlineData(0.01, 0.1, 1e-20, -0.5543030832790771)]
    [InlineData(0.5, 0.5, 0.3, -0.9969312110207781)]
    [InlineData(2, 3, 0.9999999999999999, -5.473822126268816e-48)]
    [InlineData(100, 200, 0.9, -4.3273159031619185e-124)]
    [InlineData(1e-20, 0.5, double.Epsilon, -7.45826366282501e-18)]
    [InlineData(1e-20, 7, 0.9999999999999999, -2.970117075898543e-133)]
    // 極小形状パラメータ（既知の失敗例を含む）
    [InlineData(1e-300, 1, 0.01, -4.605170185988091e-300)]
    [InlineData(1e-300, 0.5, 0.01, -5.986445692252762e-300)]
    [InlineData(1e-300, 7, 0.3, -3.1392304325936e-302)]
    [InlineData(1e-300, 7, 0.7, -4.244393873237896e-305)]
    [InlineData(1e-20, 1, 0.01, -4.605170185988091e-20)]
    [InlineData(1e-20, 7, 0.99, -1.4411835488575674e-35)]
    // 両形状が極小: 対称性と逆数オーバーフロー境界
    [InlineData(double.Epsilon, double.Epsilon, 0.3, -0.6931471805599453)]
    [InlineData(double.Epsilon, double.Epsilon, 0.5, -0.6931471805599453)]
    [InlineData(double.Epsilon, double.Epsilon, 0.7, -0.6931471805599453)]
    [InlineData(1e-310, 1e-310, 0.3, -0.6931471805599453)]
    [InlineData(1e-310, 1e-310, 0.5, -0.6931471805599453)]
    [InlineData(1e-310, 1e-310, 0.7, -0.6931471805599453)]
    [InlineData(1e-300, 1e-300, 0.3, -0.6931471805599453)]
    [InlineData(1e-300, 1e-300, 0.5, -0.6931471805599453)]
    [InlineData(1e-300, 1e-300, 0.7, -0.6931471805599453)]
    [InlineData(1e-20, 1e-20, 0.3, -0.6931471805599453)]
    [InlineData(1e-20, 1e-20, 0.5, -0.6931471805599453)]
    [InlineData(1e-20, 1e-20, 0.7, -0.6931471805599453)]
    [InlineData(5.562684646268e-309, 5.562684646268e-309, 0.3, -0.6931471805599453)]
    [InlineData(5.562684646268e-309, 5.562684646268e-309, 0.5, -0.6931471805599453)]
    [InlineData(5.562684646268e-309, 5.562684646268e-309, 0.7, -0.6931471805599453)]
    [InlineData(5.562684646268003e-309, 5.562684646268003e-309, 0.3, -0.6931471805599453)]
    [InlineData(5.562684646268003e-309, 5.562684646268003e-309, 0.5, -0.6931471805599453)]
    [InlineData(5.562684646268003e-309, 5.562684646268003e-309, 0.7, -0.6931471805599453)]
    [InlineData(5.56268464626801e-309, 5.56268464626801e-309, 0.3, -0.6931471805599453)]
    [InlineData(5.56268464626801e-309, 5.56268464626801e-309, 0.5, -0.6931471805599453)]
    [InlineData(5.56268464626801e-309, 5.56268464626801e-309, 0.7, -0.6931471805599453)]
    [InlineData(1.1125369292536e-308, 1.1125369292536e-308, 0.3, -0.6931471805599453)]
    [InlineData(1.1125369292536e-308, 1.1125369292536e-308, 0.5, -0.6931471805599453)]
    [InlineData(1.1125369292536e-308, 1.1125369292536e-308, 0.7, -0.6931471805599453)]
    [InlineData(1.1125369292536007e-308, 1.1125369292536007e-308, 0.3, -0.6931471805599453)]
    [InlineData(1.1125369292536007e-308, 1.1125369292536007e-308, 0.5, -0.6931471805599453)]
    [InlineData(1.1125369292536007e-308, 1.1125369292536007e-308, 0.7, -0.6931471805599453)]
    [InlineData(1.112536929253601e-308, 1.112536929253601e-308, 0.3, -0.6931471805599453)]
    [InlineData(1.112536929253601e-308, 1.112536929253601e-308, 0.5, -0.6931471805599453)]
    [InlineData(1.112536929253601e-308, 1.112536929253601e-308, 0.7, -0.6931471805599453)]
    // 確率のアンダーフローと1近傍の微小対数
    [InlineData(1, 1000, 0.01, -4.317217931578084e-05)]
    [InlineData(1000, 1, 0.9999999999999999, -1.1102230246251565e-13)]
    [InlineData(2, 3000, 0.01, -2.4942816200549498e-12)]
    [InlineData(3000, 2, 0.9999999999999999, -5.548527132580619e-26)]
    // 連分数の向きの切替近傍
    [InlineData(0.5, 7, 0.15789473684210523, -0.1363823937546361)]
    [InlineData(0.5, 7, 0.15789473684210525, -0.13638239375463607)]
    [InlineData(0.5, 7, 0.15789473684210528, -0.136382393754636)]
    [InlineData(0.01, 10, 0.08409658617818483, -0.0029435884669273857)]
    [InlineData(0.01, 10, 0.08409658617818484, -0.002943588466927385)]
    [InlineData(0.01, 10, 0.08409658617818486, -0.002943588466927384)]
    [InlineData(10, 20, 0.34374999999999994, -0.5716529811202423)]
    [InlineData(10, 20, 0.34375, -0.5716529811202419)]
    [InlineData(10, 20, 0.34375000000000006, -0.5716529811202414)]
    // 形状パラメータ1と10の近傍
    [InlineData(0.9999999999999999, 20, 0.3, -0.000798241172706191)]
    [InlineData(1, 20, 0.3, -0.0007982411727061912)]
    [InlineData(1.0000000000000002, 20, 0.3, -0.0007982411727061917)]
    [InlineData(20, 9.999999999999998, 0.7, -0.4525631324423897)]
    [InlineData(20, 10, 0.7, -0.45256313244238944)]
    [InlineData(20, 10.000000000000002, 0.7, -0.45256313244238916)]
    // 中心付近と漸近展開の閾値
    [InlineData(599.9999999999999, 299.99999999999994, 0.6666666666666667, -0.6994367539958938)]
    [InlineData(599.9999999999999, 299.99999999999994, 0.6823714282477182, -0.1728874156439083)]
    [InlineData(600, 300, 0.6666666666666666, -0.6994367539958994)]
    [InlineData(600, 300, 0.6823714282477181, -0.17288741564391036)]
    [InlineData(600.0000000000001, 300.00000000000006, 0.6666666666666666, -0.6994367539958994)]
    [InlineData(600.0000000000001, 300.00000000000006, 0.6823714282477181, -0.17288741564391033)]
    // 中心の漸近展開への切替近傍
    [InlineData(300, 300, 0.5499999999999999, -0.0070765660436638966)]
    [InlineData(300, 300, 0.55, -0.007076566043663789)]
    [InlineData(300, 300, 0.5500000000000002, -0.007076566043663681)]
    [InlineData(900, 300, 0.7874999999999999, -0.000965217269483431)]
    [InlineData(900, 300, 0.7875, -0.0009652172694833997)]
    [InlineData(900, 300, 0.7875000000000001, -0.0009652172694833684)]
    [InlineData(1000, 1000, 0.5473276378744922, -1.1071427863522181e-05)]
    [InlineData(1000, 1000, 0.5473276378744923, -1.1071427863521686e-05)]
    [InlineData(1000, 1000, 0.5473276378744925, -1.1071427863521193e-05)]
    // 極端なパラメータ比
    [InlineData(0.5, 100000000000000000000.0, 1e-20, -0.17114331524104098)]
    [InlineData(0.01, 100000000000000000000.0, 1e-22, -0.040460160488512475)]
    [InlineData(0.5, 1e+300, 5e-301, -0.38171514630212605)]
    // 片方の形状が1の閉形式
    [InlineData(1, 1e+300, 1e-300, -0.4586751453870819)]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(4.871821101018633e-12, 4.686195774603045e-10, 0.051307607537208866, -0.010342442999362735)]
    [InlineData(8.292927223885911e-11, 94.0665225863748, 0.36579826569383267, -5.895706411273727e-31)]
    [InlineData(8.173047212685199e-15, 59.36960654307052, 3.656119463216737e-258, -4.8067161727579e-12)]
    [InlineData(3.115971194925685e-13, 8.569723704635028e-11, 0.11734088463802028, -0.003629428585675328)]
    [InlineData(1.2429794905154313e-9, 1.1501463725252846e-9, 0.6677542226307293, -0.7327111882826058)]
    [InlineData(0.00013186728950743523, 0.06056104680399929, 7.133840450444654e-73, -0.02406877985780795)]
    public void LogRegularizedIncompleteBeta_ExploredRelativeValues(double a, double b, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.LogRegularizedIncompleteBeta(a, b, x), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 対数スケールで調べた入力の代表点
    [InlineData(0.01, 0.1, 0.01, -2.0369733245769925, 2e-12)]
    [InlineData(2, 3, 0.5, -1.1631508098056809, 2e-12)]
    [InlineData(10, 20, 0.9999999999999999, -718.616416367209, 2e-12)]
    [InlineData(100, 200, 0.9999999999999999, -7160.52423902699, 7.275957614183426e-12)]
    [InlineData(1e-20, 0.5, double.Epsilon, -39.43720903974085, 2e-12)]
    [InlineData(1e-20, 7, 0.9999999999999999, -305.15521599667596, 2e-12)]
    // 極小形状パラメータ（既知の失敗例を含む）
    [InlineData(double.Epsilon, 1, 0.01, -742.9128922955733, 2e-12)]
    [InlineData(double.Epsilon, 1, 0.1, -743.6060394761333, 2e-12)]
    [InlineData(double.Epsilon, 1, 0.3, -744.2544451625189, 2e-12)]
    [InlineData(double.Epsilon, 0.5, 0.01, -742.6505740589502, 2e-12)]
    [InlineData(double.Epsilon, 0.5, 0.1, -743.1489421980798, 2e-12)]
    [InlineData(double.Epsilon, 0.5, 0.3, -743.5563580013808, 2e-12)]
    [InlineData(double.Epsilon, 7, 0.01, -743.6450783255144, 2e-12)]
    [InlineData(double.Epsilon, 7, 0.1, -745.3974748122806, 2e-12)]
    [InlineData(double.Epsilon, 7, 0.99, -778.6133836313574, 2e-12)]
    [InlineData(1e-323, 1, 0.01, -742.2197451150134, 2e-12)]
    [InlineData(1e-323, 1, 0.1, -742.9128922955733, 2e-12)]
    [InlineData(1e-323, 1, 0.3, -743.561297981959, 2e-12)]
    [InlineData(1e-323, 0.5, 0.01, -741.9574268783903, 2e-12)]
    [InlineData(1e-323, 0.5, 0.1, -742.4557950175199, 2e-12)]
    [InlineData(1e-323, 0.5, 0.3, -742.8632108208209, 2e-12)]
    [InlineData(1e-323, 7, 0.01, -742.9519311449544, 2e-12)]
    [InlineData(1e-323, 7, 0.1, -744.7043276317207, 2e-12)]
    [InlineData(8e-323, 1, 0.01, -740.1403035733335, 2e-12)]
    [InlineData(8e-323, 1, 0.1, -740.8334507538935, 2e-12)]
    [InlineData(8e-323, 1, 0.3, -741.4818564402791, 2e-12)]
    [InlineData(8e-323, 0.5, 0.01, -739.8779853367104, 2e-12)]
    [InlineData(8e-323, 0.5, 0.1, -740.37635347584, 2e-12)]
    [InlineData(8e-323, 0.5, 0.3, -740.7837692791411, 2e-12)]
    [InlineData(8e-323, 7, 0.01, -740.8724896032746, 2e-12)]
    [InlineData(8e-323, 7, 0.1, -742.6248860900408, 2e-12)]
    [InlineData(1e-320, 1, 0.01, -735.300061265166, 2e-12)]
    [InlineData(1e-320, 1, 0.1, -735.9932084457259, 2e-12)]
    [InlineData(1e-320, 1, 0.3, -736.6416141321115, 2e-12)]
    [InlineData(1e-320, 0.5, 0.01, -735.0377430285429, 2e-12)]
    [InlineData(1e-320, 0.5, 0.1, -735.5361111676724, 2e-12)]
    [InlineData(1e-320, 0.5, 0.3, -735.9435269709735, 2e-12)]
    [InlineData(1e-320, 7, 0.01, -736.032247295107, 2e-12)]
    [InlineData(1e-320, 7, 0.1, -737.7846437818733, 2e-12)]
    [InlineData(1e-20, 1, 0.01, -44.52452223407301, 2e-12)]
    [InlineData(1e-20, 7, 0.99, -80.22501356985701, 2e-12)]
    // 確率のアンダーフローと1近傍の微小対数
    [InlineData(1, 1000, 0.01, -10.05033585350144, 2e-12)]
    [InlineData(1000, 1, 0.9999999999999999, -29.82904529069502, 2e-12)]
    [InlineData(2, 3000, 0.99, -13807.513904088808, 1.4551915228366852e-11)]
    [InlineData(2, 3000, 0.9999999999999999, -110202.39500818586, 1.1641532182693481e-10)]
    [InlineData(3000, 2, 0.9999999999999999, -58.153679906823754, 2e-12)]
    [InlineData(300, 600, 0.9999999999999999, -21473.88462392004, 2.9103830456733704e-11)]
    // 連分数の向きの切替近傍
    [InlineData(0.5, 7, 0.15789473684210523, -2.059708930362889, 2e-12)]
    [InlineData(0.5, 7, 0.15789473684210525, -2.059708930362889, 2e-12)]
    [InlineData(0.5, 7, 0.15789473684210528, -2.0597089303628895, 2e-12)]
    [InlineData(0.01, 10, 0.08409658617818483, -5.829597308139528, 2e-12)]
    [InlineData(0.01, 10, 0.08409658617818484, -5.829597308139528, 2e-12)]
    [InlineData(0.01, 10, 0.08409658617818486, -5.829597308139529, 2e-12)]
    // 形状パラメータ1と10の近傍
    [InlineData(0.9999999999999999, 20, 0.3, -7.133498878774647, 2e-12)]
    [InlineData(1, 20, 0.3, -7.133498878774647, 2e-12)]
    [InlineData(1.0000000000000002, 20, 0.3, -7.133498878774646, 2e-12)]
    [InlineData(20, 9.999999999999998, 0.7, -1.0105901996721358, 2e-12)]
    [InlineData(20, 10, 0.7, -1.0105901996721363, 2e-12)]
    [InlineData(20, 10.000000000000002, 0.7, -1.0105901996721367, 2e-12)]
    // 中心付近と漸近展開の閾値
    [InlineData(599.9999999999999, 299.99999999999994, 0.6823714282477182, -1.8403132717411264, 2e-12)]
    [InlineData(600, 300, 0.6823714282477181, -1.8403132717411155, 2e-12)]
    [InlineData(600.0000000000001, 300.00000000000006, 0.6823714282477181, -1.8403132717411157, 2e-12)]
    // 中心の漸近展開への切替近傍
    [InlineData(300, 300, 0.5499999999999999, -4.954502707467381, 2e-12)]
    [InlineData(300, 300, 0.55, -4.954502707467396, 2e-12)]
    [InlineData(300, 300, 0.5500000000000002, -4.954502707467412, 2e-12)]
    [InlineData(900, 300, 0.7874999999999999, -6.943639902060147, 2e-12)]
    [InlineData(900, 300, 0.7875, -6.94363990206018, 2e-12)]
    [InlineData(900, 300, 0.7875000000000001, -6.9436399020602115, 2e-12)]
    [InlineData(1000, 1000, 0.5473276378744922, -11.411148370309045, 2e-12)]
    [InlineData(1000, 1000, 0.5473276378744923, -11.411148370309089, 2e-12)]
    [InlineData(1000, 1000, 0.5473276378744925, -11.411148370309133, 2e-12)]
    // 極端なパラメータ比
    [InlineData(0.5, 100000000000000000000.0, 1e-20, -1.8496055099332482, 2e-12)]
    [InlineData(0.01, 100000000000000000000.0, 1e-22, -3.227599352470411, 2e-12)]
    [InlineData(0.5, 1e+300, 5e-301, -1.1478744644493182, 2e-12)]
    // 片方の形状が1の閉形式
    [InlineData(1, 100000000000000000000.0, 0.5, -6.931471805599453e+19, 65536)]
    [InlineData(1, 100000000000000000000.0, 0.9999999999999999, -3.67368005696771e+21, 4194304)]
    [InlineData(1, 1e+100, 0.5, -6.931471805599453e+99, 7.770675568902916e+84)]
    [InlineData(1, 1e+100, 0.9999999999999999, -3.6736800569677104e+101, 4.9732323640978664e+86)]
    [InlineData(1, 1e+300, 1e-300, -1.0, 2e-12)]
    [InlineData(1, 1e+300, 0.5, -6.931471805599454e+299, 1.1896135267822265e+285)]
    [InlineData(1, 1e+300, 0.9999999999999999, -3.6736800569677104e+301, 3.806763285703125e+286)]
    [InlineData(1, 1.7976931348623157e+308, 1e-300, -179769313.48623157, 2.384185791015625e-7)]
    [InlineData(1, 1.7976931348623157e+308, 0.5, -1.2460659279417838e+308, 1.596672247627776e+293)]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(4.871821101018633e-12, 4.686195774603045e-10, 0.051307607537208866, -4.5766659355272274, 2e-12)]
    [InlineData(2.7827253541038584e-10, 2.8510639218750617e-9, 0.27399169463311124, -2.4199752871335467, 2e-12)]
    [InlineData(8.292927223885911e-11, 94.0665225863748, 0.36579826569383267, -69.60591352372826, 2e-12)]
    [InlineData(8.173047212685199e-15, 59.36960654307052, 3.656119463216737e-258, -26.061006973331104, 2e-12)]
    [InlineData(3.115971194925685e-13, 8.569723704635028e-11, 0.11734088463802028, -5.620494222939074, 2e-12)]
    [InlineData(0.00013186728950743523, 0.06056104680399929, 7.133840450444654e-73, -3.7388499721686737, 2e-12)]
    public void LogRegularizedIncompleteBetaComplement_ExploredAbsoluteValues(
        double a, double b, double x, double expected, double absoluteTolerance)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.LogRegularizedIncompleteBetaComplement(a, b, x), relativeTolerance: 0, absoluteTolerance: absoluteTolerance);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 対数スケールで調べた入力の代表点
    [InlineData(1, 1, double.Epsilon, -5e-324)]
    // 極小形状パラメータを反転した入力
    [InlineData(1, double.Epsilon, 0.99, -2.5e-323)]
    [InlineData(1, double.Epsilon, 0.7, -5e-324)]
    [InlineData(0.5, double.Epsilon, 0.99, -3e-323)]
    [InlineData(7, 8e-323, 0.7, -5e-324)]
    [InlineData(0.5, 1e-310, 0.99, -5.98644569225275e-310)]
    [InlineData(7, 1e-300, 0.010000000000000009, -1.44118355e-315)]
    public void LogRegularizedIncompleteBetaComplement_ExploredSubnormalValues(double a, double b, double x, double expected)
    {
        double actual = XMath.Special.LogRegularizedIncompleteBetaComplement(a, b, x);
        Assert.True(actual == 0 || Math.Sign(actual) == Math.Sign(expected));
        SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 0, absoluteTolerance: 2 * double.Epsilon);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 対数スケールで調べた入力の代表点
    [InlineData(2, 3, double.Epsilon, -0.0)]
    [InlineData(2, 3, 1e-300, -0.0)]
    [InlineData(10, 20, double.Epsilon, -0.0)]
    [InlineData(10, 20, 1e-300, -0.0)]
    [InlineData(100, 200, double.Epsilon, -0.0)]
    [InlineData(100, 200, 1e-20, -0.0)]
    // 極小形状パラメータを反転した入力
    [InlineData(1, double.Epsilon, 0.30000000000000004, -0.0)]
    [InlineData(1, double.Epsilon, 0.010000000000000009, -0.0)]
    [InlineData(0.5, double.Epsilon, 0.010000000000000009, -0.0)]
    [InlineData(7, double.Epsilon, 0.9, -0.0)]
    [InlineData(7, 1e-323, 0.7, -0.0)]
    [InlineData(7, 1e-310, 0.010000000000000009, -0.0)]
    // 確率のアンダーフローと1近傍の微小対数
    [InlineData(1000, 1, 1e-20, -0.0)]
    [InlineData(1000, 1, 0.01, -0.0)]
    [InlineData(3000, 2, 1e-20, -0.0)]
    [InlineData(3000, 2, 0.01, -0.0)]
    [InlineData(300, 600, 1e-20, -0.0)]
    [InlineData(300, 600, 0.01, -0.0)]
    // 片方の形状が1の閉形式
    [InlineData(1, 1.7976931348623157e+308, 0.9999999999999999, double.NegativeInfinity)]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(2.163151112687556, 153.78484275640287, 3.3614986150159407e-267, -0.0)]
    public void LogRegularizedIncompleteBetaComplement_ExploredExactValues(double a, double b, double x, double expected)
    {
        Assert.Equal(expected, XMath.Special.LogRegularizedIncompleteBetaComplement(a, b, x));
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 対数スケールで調べた入力の代表点
    [InlineData(10, 20, 0.9999999999999999, -8.10691774616e-313)]
    // 極小形状パラメータ（既知の失敗例を含む）
    [InlineData(double.Epsilon, 1, 0.01, -2.5e-323)]
    [InlineData(double.Epsilon, 1, 0.3, -5e-324)]
    [InlineData(double.Epsilon, 0.5, 0.01, -3e-323)]
    [InlineData(8e-323, 7, 0.3, -5e-324)]
    [InlineData(1e-310, 0.5, 0.01, -5.98644569225275e-310)]
    [InlineData(1e-300, 7, 0.99, -1.44118355e-315)]
    // 確率のアンダーフローと1近傍の微小対数
    [InlineData(100, 200, 0.99, -5.15e-320)]
    public void LogRegularizedIncompleteBeta_ExploredSubnormalValues(double a, double b, double x, double expected)
    {
        double actual = XMath.Special.LogRegularizedIncompleteBeta(a, b, x);
        Assert.True(actual == 0 || Math.Sign(actual) == Math.Sign(expected));
        SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 0, absoluteTolerance: 2 * double.Epsilon);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 対数スケールで調べた入力の代表点
    [InlineData(100, 200, 0.9999999999999999, -0.0)]
    // 極小形状パラメータ（既知の失敗例を含む）
    [InlineData(double.Epsilon, 1, 0.7, -0.0)]
    [InlineData(double.Epsilon, 1, 0.99, -0.0)]
    [InlineData(double.Epsilon, 0.5, 0.99, -0.0)]
    [InlineData(double.Epsilon, 7, 0.1, -0.0)]
    [InlineData(1e-323, 7, 0.3, -0.0)]
    [InlineData(1e-310, 7, 0.99, -0.0)]
    // 確率のアンダーフローと1近傍の微小対数
    [InlineData(1, 1000, 0.99, -0.0)]
    [InlineData(1, 1000, 0.9999999999999999, -0.0)]
    [InlineData(2, 3000, 0.99, -0.0)]
    [InlineData(2, 3000, 0.9999999999999999, -0.0)]
    [InlineData(300, 600, 0.99, -0.0)]
    [InlineData(300, 600, 0.9999999999999999, -0.0)]
    // 片方の形状が1の閉形式
    [InlineData(1, 100000000000000000000.0, 0.5, -0.0)]
    [InlineData(1, 100000000000000000000.0, 0.9999999999999999, -0.0)]
    [InlineData(1, 1e+100, 0.5, -0.0)]
    [InlineData(1, 1e+100, 0.9999999999999999, -0.0)]
    [InlineData(1, 1e+300, 0.5, -0.0)]
    [InlineData(1, 1e+300, 0.9999999999999999, -0.0)]
    [InlineData(1, 1.7976931348623157e+308, 1e-300, -0.0)]
    [InlineData(1, 1.7976931348623157e+308, 0.5, -0.0)]
    [InlineData(1, 1.7976931348623157e+308, 0.9999999999999999, -0.0)]
    public void LogRegularizedIncompleteBeta_ExploredExactValues(double a, double b, double x, double expected)
    {
        Assert.Equal(expected, XMath.Special.LogRegularizedIncompleteBeta(a, b, x));
    }
}
