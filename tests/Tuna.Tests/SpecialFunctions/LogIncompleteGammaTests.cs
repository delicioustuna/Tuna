namespace Tuna.Tests;

// Public API contract, including the limiting conventions shared with P and Q.
[Trait("SpecialFunction", "LogIncompleteGamma")]
public class LogIncompleteGammaTests
{
    [Theory]
    [InlineData(2, 0, double.NegativeInfinity, 0)]
    [InlineData(2, double.PositiveInfinity, 0, double.NegativeInfinity)]
    [InlineData(double.PositiveInfinity, 2, double.NegativeInfinity, 0)]
    [InlineData(double.PositiveInfinity, 0, double.NegativeInfinity, 0)]
    [InlineData(double.PositiveInfinity, double.PositiveInfinity, double.NaN, double.NaN)]
    [InlineData(0, 1, double.NaN, double.NaN)]
    [InlineData(-1, 1, double.NaN, double.NaN)]
    [InlineData(double.NegativeInfinity, 1, double.NaN, double.NaN)]
    [InlineData(2, -double.Epsilon, double.NaN, double.NaN)]
    [InlineData(2, double.NegativeInfinity, double.NaN, double.NaN)]
    [InlineData(double.NaN, 1, double.NaN, double.NaN)]
    [InlineData(2, double.NaN, double.NaN, double.NaN)]
    public void EndpointsAndDomain(double a, double x, double expectedP, double expectedQ)
    {
        Assert.Equal(expectedP, XMath.Special.LogRegularizedGammaP(a, x));
        Assert.Equal(expectedQ, XMath.Special.LogRegularizedGammaQ(a, x));
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(1e-20)]
    [InlineData(0.1)]
    [InlineData(2)]
    [InlineData(10000)]
    public void LogProbabilitiesAreBoundedAndMonotone(double a)
    {
        double previousP = double.NegativeInfinity, previousQ = 0;
        for (int i = 0; i <= 120; i++)
        {
            double x = Math.Exp(-700 + i * 708.0 / 120);
            double p = XMath.Special.LogRegularizedGammaP(a, x);
            double q = XMath.Special.LogRegularizedGammaQ(a, x);
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
    [InlineData(0.01, 1e-300, -6.902064971036068, 2e-12)]
    [InlineData(0.1, 1e-300, -69.02768034856153, 2e-12)]
    [InlineData(0.5, 0.1, -1.0634020471545287, 2e-12)]
    [InlineData(10, 1e-20, -475.62143117188464, 2e-12)]
    [InlineData(10000, 1e-300, -6989864.206818951, 7.450580596923828e-9)]
    [InlineData(10000, 1000, -14031.269698821252, 1.4551915228366852e-11)]
    // 確率のアンダーフローと1近傍の微小対数
    [InlineData(0.1, double.Epsilon, -74.3941347508783, 2e-12)]
    [InlineData(0.1, 1e-310, -71.33026544155558, 2e-12)]
    [InlineData(0.1, 1e-100, -22.975978488680617, 2e-12)]
    [InlineData(1, double.Epsilon, -744.4400719213812, 2e-12)]
    [InlineData(10, double.Epsilon, -7459.505131786888, 7.275957614183426e-12)]
    [InlineData(10, 1e-100, -2317.689505567121, 3.637978807091713e-12)]
    // 小引数の分岐近傍
    [InlineData(3.5, 0.49999999999999994, -5.264599558278468, 2e-12)]
    [InlineData(3.5, 0.5, -5.2645995582784675, 2e-12)]
    [InlineData(3.5, 0.5000000000000001, -5.264599558278467, 2e-12)]
    [InlineData(3.5, 1.0999999999999999, -2.95548348013829, 2e-12)]
    [InlineData(3.5, 1.1, -2.955483480138289, 2e-12)]
    [InlineData(3.5, 1.1000000000000003, -2.9554834801382888, 2e-12)]
    // 中心付近と漸近展開の閾値
    [InlineData(9999.999999999998, 9199.999999999998, -36.82835357444092, 2e-12)]
    [InlineData(9999.999999999998, 9899.999999999998, -1.8410472463668297, 2e-12)]
    [InlineData(10000, 9200, -36.82835357444091, 2e-12)]
    [InlineData(10000, 9900, -1.8410472463668295, 2e-12)]
    [InlineData(10000.000000000002, 9200.000000000002, -36.82835357444091, 2e-12)]
    [InlineData(10000.000000000002, 9900.000000000002, -1.8410472463668293, 2e-12)]
    [InlineData(100000, 97470.1778718653, -35.56321194798062, 2e-12)]
    [InlineData(100000, 99683.77223398317, -1.8410241926340902, 2e-12)]
    // 中心の漸近展開への切替近傍
    [InlineData(10000, 6999.999999999999, -571.0703596493532, 2e-12)]
    [InlineData(10000, 7000, -571.0703596493529, 2e-12)]
    [InlineData(10000, 7000.000000000001, -571.0703596493524, 2e-12)]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(5.366102937593265, 3.3774541193585636e-86, -1061.5009375383881, 2e-12)]
    [InlineData(6.640393439826551, 1.0731996853762981e-255, -3906.308056134681, 3.637978807091713e-12)]
    [InlineData(0.06514094524330558, 1.0328324561911851e-10, -1.4636048382557112, 2e-12)]
    [InlineData(0.007305470870617144, 4.333609383416088e-294, -4.930625846440507, 2e-12)]
    [InlineData(1.9932271247181217, 3.595330574142313e-39, -177.1297287852924, 2e-12)]
    [InlineData(8674.55774537032, 1.316237382718235e-88, -1825313.3591974778, 1.862645149230957e-9)]
    public void LogRegularizedGammaP_ExploredAbsoluteValues(double a, double x, double expected, double absoluteTolerance)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.LogRegularizedGammaP(a, x), relativeTolerance: 0, absoluteTolerance: absoluteTolerance);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 対数スケールで調べた入力の代表点
    [InlineData(0.01, 1e-300, -0.0010062125906395688)]
    [InlineData(0.1, 1e-300, -1.0511370061117737e-30)]
    [InlineData(1, 1e-300, -1e-300)]
    [InlineData(2, 0.1, -0.00468982019567514)]
    [InlineData(10, 10, -0.781039568496278)]
    [InlineData(100, 100, -0.720104893025474)]
    // 確率のアンダーフローと1近傍の微小対数
    [InlineData(0.1, double.Epsilon, -4.9095059615996955e-33)]
    [InlineData(0.1, 1e-310, -1.0511370061117734e-31)]
    [InlineData(0.1, 1e-100, -1.0511370061670209e-10)]
    [InlineData(0.5, double.Epsilon, -2.508114666398235e-162)]
    [InlineData(0.5, 1e-310, -1.1283791670955108e-155)]
    [InlineData(1, 1e-100, -1e-100)]
    // 小引数の分岐近傍
    [InlineData(3.5, 0.49999999999999994, -0.005184881782251553)]
    [InlineData(3.5, 0.5, -0.005184881782251555)]
    [InlineData(3.5, 0.5000000000000001, -0.005184881782251559)]
    [InlineData(3.5, 1.0999999999999999, -0.053457199905096835)]
    [InlineData(3.5, 1.1, -0.05345719990509687)]
    [InlineData(3.5, 1.1000000000000003, -0.0534571999050969)]
    // 中心付近と漸近展開の閾値
    [InlineData(9999.999999999998, 9199.999999999998, -1.013092884401965e-16)]
    [InlineData(9999.999999999998, 9899.999999999998, -0.1727489513614159)]
    [InlineData(9999.999999999998, 9999.999999999998, -0.69581034030382)]
    [InlineData(10000, 9200, -1.0130928844019717e-16)]
    [InlineData(10000, 9900, -0.17274895136141594)]
    [InlineData(10000, 10000, -0.69581034030382)]
    [InlineData(10000.000000000002, 9200.000000000002, -1.0130928844019784e-16)]
    [InlineData(10000.000000000002, 9900.000000000002, -0.17274895136141596)]
    [InlineData(10000.000000000002, 10000.000000000002, -0.69581034030382)]
    [InlineData(100000, 97470.1778718653, -3.589990441364625e-16)]
    [InlineData(100000, 99683.77223398317, -0.17275329860985755)]
    [InlineData(100000, 100000, -0.6939885786567972)]
    // 中心の漸近展開への切替近傍
    [InlineData(10000, 6999.999999999999, -9.711672437702064e-249)]
    [InlineData(10000, 7000, -9.711672437705851e-249)]
    [InlineData(10000, 7000.000000000001, -9.71167243770964e-249)]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(0.06514094524330558, 1.0328324561911851e-10, -0.26318539144437686)]
    [InlineData(0.007305470870617144, 4.333609383416088e-294, -0.007248186770161805)]
    [InlineData(1.9932271247181217, 3.595330574142313e-39, -1.1845031159882023e-77)]
    [InlineData(0.032653375488717724, 4.201464483375956e-237, -1.946779767198195e-08)]
    public void LogRegularizedGammaQ_ExploredRelativeValues(double a, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.LogRegularizedGammaQ(a, x), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 対数スケールで調べた入力の代表点
    [InlineData(0.01, 1e-20, -0.4548267106527395)]
    [InlineData(0.01, 0.1, -0.018301776508399554)]
    [InlineData(0.01, 1, -0.0022186941057126645)]
    [InlineData(0.5, 10, -7.744246417642965e-06)]
    [InlineData(100, 100, -0.6668971505852894)]
    [InlineData(100, 1000, -6.035827529631278e-294)]
    // 極小形状パラメータ（既知の失敗例を含む）
    [InlineData(1e-300, 0.01, -4.037929576538114e-300)]
    [InlineData(1e-300, 0.5, -5.597735947761608e-301)]
    [InlineData(1e-300, 1.1, -1.8599090453604013e-301)]
    [InlineData(9.999999999999998e-21, 1.1, -1.859909045360401e-21)]
    [InlineData(1.0000000000000001e-20, 0.01, -4.037929576538114e-20)]
    [InlineData(1.0000000000000001e-20, 20, -9.835525290649882e-31)]
    // 確率のアンダーフローと1近傍の微小対数
    [InlineData(0.1, 700, -2.846928310475215e-308)]
    [InlineData(0.5, 700, -2.1010145162642176e-306)]
    [InlineData(1, 700, -9.85967654375977e-305)]
    [InlineData(10, 700, -1.1106924650193637e-284)]
    [InlineData(10, 745, -5.565775267462702e-304)]
    // 小引数の分岐近傍
    [InlineData(0.3, 0.49999999999999994, -0.2060261378987249)]
    [InlineData(0.5, 0.49999999999999994, -0.3817151463021261)]
    [InlineData(0.3, 0.5, -0.20602613789872487)]
    [InlineData(0.5, 0.5, -0.38171514630212605)]
    [InlineData(0.3, 0.5000000000000001, -0.20602613789872484)]
    [InlineData(0.5, 0.5000000000000001, -0.381715146302126)]
    [InlineData(0.3, 1.0999999999999999, -0.07580715562374173)]
    [InlineData(0.5, 1.0999999999999999, -0.14851246497219894)]
    [InlineData(0.3, 1.1, -0.0758071556237417)]
    [InlineData(0.5, 1.1, -0.14851246497219892)]
    [InlineData(0.3, 1.1000000000000003, -0.07580715562374167)]
    [InlineData(0.5, 1.1000000000000003, -0.14851246497219886)]
    // 前因子と連分数の切替近傍
    [InlineData(9.999999999999998, 10.999999999999996, -0.41628944412807606)]
    [InlineData(9.999999999999998, 10.999999999999998, -0.4162894441280757)]
    [InlineData(9.999999999999998, 11, -0.41628944412807545)]
    [InlineData(10, 10.999999999999998, -0.41628944412807606)]
    [InlineData(10, 11, -0.4162894441280758)]
    [InlineData(10, 11.000000000000002, -0.41628944412807545)]
    [InlineData(10.000000000000002, 11, -0.41628944412807606)]
    [InlineData(10.000000000000002, 11.000000000000002, -0.4162894441280758)]
    [InlineData(10.000000000000002, 11.000000000000004, -0.4162894441280755)]
    [InlineData(3.5, 4.499999999999999, -0.2912297533651846)]
    [InlineData(3.5, 4.5, -0.2912297533651844)]
    [InlineData(3.5, 4.500000000000001, -0.29122975336518425)]
    // 小さい形状の分岐近傍
    [InlineData(0.17371779276130073, 0.1, -0.3371796192453092)]
    [InlineData(0.17371779276130075, 0.1, -0.33717961924530926)]
    [InlineData(0.17371779276130078, 0.1, -0.3371796192453093)]
    [InlineData(0.37499999999999994, 0.5, -0.2685722359286824)]
    [InlineData(0.375, 0.5, -0.26857223592868246)]
    [InlineData(0.37500000000000006, 0.5, -0.26857223592868246)]
    // 中心付近と漸近展開の閾値
    [InlineData(9999.999999999998, 9999.999999999998, -0.6904910944019723)]
    [InlineData(9999.999999999998, 10099.999999999998, -0.1727490195367757)]
    [InlineData(9999.999999999998, 10799.999999999998, -3.109819962168833e-15)]
    [InlineData(10000, 10000, -0.6904910944019723)]
    [InlineData(10000, 10100, -0.17274901953677574)]
    [InlineData(10000, 10800, -3.1098199621688497e-15)]
    [InlineData(10000.000000000002, 10000.000000000002, -0.6904910944019723)]
    [InlineData(10000.000000000002, 10100.000000000002, -0.17274901953677577)]
    [InlineData(10000.000000000002, 10800.000000000002, -3.1098199621688663e-15)]
    [InlineData(100000, 100000, -0.6923064898187249)]
    [InlineData(100000, 100316.22776601683, -0.17275330076565168)]
    [InlineData(100000, 102529.8221281347, -1.0561434323529749e-15)]
    // 中心の漸近展開への切替近傍
    [InlineData(10000, 12999.999999999998, -4.712347177487624e-166)]
    [InlineData(10000, 13000, -4.712347177485643e-166)]
    [InlineData(10000, 13000.000000000002, -4.712347177483662e-166)]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(2.077540955398398e-12, 2.4542276280087156e-71, -3.365793226211344e-10)]
    [InlineData(2.9117490308372614e-8, 1.0898316877590093e-141, -9.434103524708622e-06)]
    [InlineData(1.3223540935724265e-7, 1.9292968182680833e-188, -5.707962950787395e-05)]
    [InlineData(2.16138597386859e-15, 2.477257279694589e-199, -9.871699581254556e-13)]
    [InlineData(1.741254767133025e-13, 7.230725150450989e-261, -1.0420002051302264e-10)]
    [InlineData(2.8132689228784736e-7, 1.539616949340005e-132, -8.522305406899672e-05)]
    public void LogRegularizedGammaP_ExploredRelativeValues(double a, double x, double expected)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.LogRegularizedGammaP(a, x), relativeTolerance: 2e-12);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 対数スケールで調べた入力の代表点
    [InlineData(0.01, 1e-20, -1.0066474817041988, 2e-12)]
    [InlineData(0.01, 0.1, -4.00989407871855, 2e-12)]
    [InlineData(0.01, 1000, -1011.4391461284306, 2e-12)]
    [InlineData(0.5, 10, -11.768564260156875, 2e-12)]
    [InlineData(1, 1, -1.0, 2e-12)]
    [InlineData(100, 1000, -675.1623043733719, 2e-12)]
    // 極小形状パラメータ（既知の失敗例を含む）
    [InlineData(double.Epsilon, 0.01, -743.0443398418264, 2e-12)]
    [InlineData(double.Epsilon, 0.5, -745.020294793426, 2e-12)]
    [InlineData(double.Epsilon, 1, -745.9570038803834, 2e-12)]
    [InlineData(double.Epsilon, 700, -1450.9925777791889, 2e-12)]
    [InlineData(9.999999999999998e-21, 0.01, -44.655969780326075, 2e-12)]
    [InlineData(1.0000000000000001e-20, 700, -752.6042077176886, 2e-12)]
    // 確率のアンダーフローと1近傍の微小対数
    [InlineData(0.1, 700, -708.1499680149125, 2e-12)]
    [InlineData(0.1, 745, -753.2059641595331, 2e-12)]
    [InlineData(0.1, 10000, -10010.542108973466, 1.4551915228366852e-11)]
    [InlineData(1, 700, -700.0, 2e-12)]
    [InlineData(10, 700, -653.8291827471337, 2e-12)]
    [InlineData(10, 10000, -9929.907863817198, 1.4551915228366852e-11)]
    // 小引数の分岐近傍
    [InlineData(0.3, 0.49999999999999994, -1.6809973139705547, 2e-12)]
    [InlineData(0.5, 0.49999999999999994, -1.1478744644493182, 2e-12)]
    [InlineData(0.3, 0.5, -1.680997313970555, 2e-12)]
    [InlineData(0.5, 0.5, -1.1478744644493182, 2e-12)]
    [InlineData(0.3, 0.5000000000000001, -1.6809973139705552, 2e-12)]
    [InlineData(0.5, 0.5000000000000001, -1.1478744644493184, 2e-12)]
    [InlineData(0.3, 1.0999999999999999, -2.617226731830011, 2e-12)]
    [InlineData(0.5, 1.0999999999999999, -1.9804237883833065, 2e-12)]
    [InlineData(0.3, 1.1, -2.6172267318300113, 2e-12)]
    [InlineData(0.5, 1.1, -1.980423788383307, 2e-12)]
    [InlineData(0.3, 1.1000000000000003, -2.6172267318300118, 2e-12)]
    [InlineData(0.5, 1.1000000000000003, -1.9804237883833071, 2e-12)]
    // 前因子と連分数の切替近傍
    [InlineData(9.999999999999998, 10.999999999999996, -1.0773088984752501, 2e-12)]
    [InlineData(9.999999999999998, 10.999999999999998, -1.0773088984752508, 2e-12)]
    [InlineData(9.999999999999998, 11, -1.0773088984752512, 2e-12)]
    [InlineData(10, 10.999999999999998, -1.0773088984752501, 2e-12)]
    [InlineData(10, 11, -1.0773088984752508, 2e-12)]
    [InlineData(10, 11.000000000000002, -1.0773088984752512, 2e-12)]
    [InlineData(10.000000000000002, 11, -1.0773088984752501, 2e-12)]
    [InlineData(10.000000000000002, 11.000000000000002, -1.0773088984752506, 2e-12)]
    [InlineData(10.000000000000002, 11.000000000000004, -1.0773088984752512, 2e-12)]
    [InlineData(3.5, 4.499999999999999, -1.3757262152267575, 2e-12)]
    [InlineData(3.5, 4.5, -1.375726215226758, 2e-12)]
    [InlineData(3.5, 4.500000000000001, -1.3757262152267584, 2e-12)]
    // 小さい形状の分岐近傍
    [InlineData(0.17371779276130073, 0.1, -1.2509966980498128, 2e-12)]
    [InlineData(0.17371779276130075, 0.1, -1.2509966980498128, 2e-12)]
    [InlineData(0.17371779276130078, 0.1, -1.2509966980498126, 2e-12)]
    [InlineData(0.37499999999999994, 0.5, -1.4459178283112144, 2e-12)]
    [InlineData(0.375, 0.5, -1.4459178283112142, 2e-12)]
    [InlineData(0.37500000000000006, 0.5, -1.4459178283112142, 2e-12)]
    // 中心付近と漸近展開の閾値
    [InlineData(9999.999999999998, 10099.999999999998, -1.84104688482372, 2e-12)]
    [InlineData(9999.999999999998, 10799.999999999998, -33.4042115603731, 2e-12)]
    [InlineData(10000, 10100, -1.84104688482372, 2e-12)]
    [InlineData(10000, 10800, -33.40421156037309, 2e-12)]
    [InlineData(10000.000000000002, 10100.000000000002, -1.8410468848237198, 2e-12)]
    [InlineData(10000.000000000002, 10800.000000000002, -33.40421156037309, 2e-12)]
    [InlineData(100000, 100316.22776601683, -1.8410241812019348, 2e-12)]
    [InlineData(100000, 102529.8221281347, -34.48415239275831, 2e-12)]
    // 中心の漸近展開への切替近傍
    [InlineData(10000, 12999.999999999998, -380.67893931394366, 2e-12)]
    [InlineData(10000, 13000, -380.67893931394406, 2e-12)]
    [InlineData(10000, 13000.000000000002, -380.67893931394445, 2e-12)]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(2.077540955398398e-12, 2.4542276280087156e-71, -21.812187266348445, 2e-12)]
    [InlineData(2.9117490308372614e-8, 1.0898316877590093e-141, -11.571184116635248, 2e-12)]
    [InlineData(1.3223540935724265e-7, 1.9292968182680833e-188, -9.771091795807736, 2e-12)]
    [InlineData(2.16138597386859e-15, 2.477257279694589e-199, -27.643934173619517, 2e-12)]
    [InlineData(1.741254767133025e-13, 7.230725150450989e-261, -22.98470878979938, 2e-12)]
    [InlineData(2.8132689228784736e-7, 1.539616949340005e-132, -9.370281184291374, 2e-12)]
    // 指数分布の閉形式で確認できる巨大な対数
    [InlineData(1, 100000000000000000000.0, -1e+20, 131072)]
    [InlineData(1, 1e+100, -1e+100, 1.5541351137805833e+85)]
    [InlineData(1, 1e+300, -1e+300, 1.1896135267822265e+285)]
    [InlineData(1, 1.7976931348623157e+308, -1.7976931348623157e+308, 1.596672247627776e+293)]
    public void LogRegularizedGammaQ_ExploredAbsoluteValues(double a, double x, double expected, double absoluteTolerance)
    {
        SpecialFunctionAssert.Close(expected, XMath.Special.LogRegularizedGammaQ(a, x), relativeTolerance: 0, absoluteTolerance: absoluteTolerance);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 対数スケールで調べた入力の代表点
    [InlineData(0.01, 1000, -0.0)]
    [InlineData(0.1, 1000, -0.0)]
    [InlineData(0.5, 1000, -0.0)]
    [InlineData(1, 1000, -0.0)]
    [InlineData(2, 1000, -0.0)]
    [InlineData(10, 1000, -0.0)]
    // 極小形状パラメータ（既知の失敗例を含む）
    [InlineData(double.Epsilon, 1, -0.0)]
    [InlineData(double.Epsilon, 1.1, -0.0)]
    [InlineData(double.Epsilon, 20, -0.0)]
    [InlineData(double.Epsilon, 700, -0.0)]
    [InlineData(1e-300, 700, -0.0)]
    [InlineData(1.0000000000000001e-20, 700, -0.0)]
    // 確率のアンダーフローと1近傍の微小対数
    [InlineData(0.1, 745, -0.0)]
    [InlineData(0.1, 10000, -0.0)]
    [InlineData(0.5, 745, -0.0)]
    [InlineData(0.5, 10000, -0.0)]
    [InlineData(1, 10000, -0.0)]
    [InlineData(10, 10000, -0.0)]
    // 指数分布の閉形式で確認できる巨大な対数
    [InlineData(1, 100000000000000000000.0, -0.0)]
    [InlineData(1, 1e+100, -0.0)]
    [InlineData(1, 1e+300, -0.0)]
    [InlineData(1, 1.7976931348623157e+308, -0.0)]
    public void LogRegularizedGammaP_ExploredExactValues(double a, double x, double expected)
    {
        Assert.Equal(expected, XMath.Special.LogRegularizedGammaP(a, x));
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 対数スケールで調べた入力の代表点
    [InlineData(2, 1e-300, -0.0)]
    [InlineData(10, 1e-300, -0.0)]
    [InlineData(100, 1e-300, -0.0)]
    [InlineData(100, 1e-20, -0.0)]
    [InlineData(10000, 1e-20, -0.0)]
    [InlineData(10000, 1000, -0.0)]
    // 確率のアンダーフローと1近傍の微小対数
    [InlineData(10, double.Epsilon, -0.0)]
    [InlineData(10, 1e-310, -0.0)]
    [InlineData(10, 1e-100, -0.0)]
    // 固定seed探索: 入力と出力の大小を代表する点
    [InlineData(5.366102937593265, 3.3774541193585636e-86, -0.0)]
    [InlineData(6.640393439826551, 1.0731996853762981e-255, -0.0)]
    [InlineData(1798.2057605443872, 2.5461701757279944e-14, -0.0)]
    [InlineData(1044.5358688088377, 3.588105444055958e-55, -0.0)]
    [InlineData(8674.55774537032, 1.316237382718235e-88, -0.0)]
    public void LogRegularizedGammaQ_ExploredExactValues(double a, double x, double expected)
    {
        Assert.Equal(expected, XMath.Special.LogRegularizedGammaQ(a, x));
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 極小形状パラメータ（既知の失敗例を含む）
    [InlineData(double.Epsilon, 0.01, -2e-323)]
    [InlineData(double.Epsilon, 0.5, -5e-324)]
    [InlineData(1e-300, 20, -9.83552529065e-311)]
    // 確率のアンダーフローと1近傍の微小対数
    [InlineData(1, 745, -5e-324)]
    public void LogRegularizedGammaP_ExploredSubnormalValues(double a, double x, double expected)
    {
        double actual = XMath.Special.LogRegularizedGammaP(a, x);
        Assert.True(actual == 0 || Math.Sign(actual) == Math.Sign(expected));
        SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 0, absoluteTolerance: 2 * double.Epsilon);
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    // 確率のアンダーフローと1近傍の微小対数
    [InlineData(1, double.Epsilon, -5e-324)]
    [InlineData(1, 1e-310, -1e-310)]
    public void LogRegularizedGammaQ_ExploredSubnormalValues(double a, double x, double expected)
    {
        double actual = XMath.Special.LogRegularizedGammaQ(a, x);
        Assert.True(actual == 0 || Math.Sign(actual) == Math.Sign(expected));
        SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 0, absoluteTolerance: 2 * double.Epsilon);
    }
}
