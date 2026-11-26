namespace Tuna.Tests;

[Trait("SpecialFunction", "IncompleteGammaShapeDerivative")]
public class IncompleteGammaShapeDerivativeTests
{
    [Theory]
    [InlineData(2, 0, double.NegativeInfinity, 0)]
    [InlineData(2, double.PositiveInfinity, -0.0, double.PositiveInfinity)]
    [InlineData(double.PositiveInfinity, 1, double.NegativeInfinity, 0)]
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
        AssertDerivative(expectedP, XMath.Special.LogRegularizedGammaPShapeDerivative(a, x));
        AssertDerivative(expectedQ, XMath.Special.LogRegularizedGammaQShapeDerivative(a, x));
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(1e100)]
    [InlineData(1e200)]
    [InlineData(double.MaxValue)]
    public void VeryLargeShapeUsesTheCentralLimit(double a)
    {
        double scale = Math.Sqrt(2 / Math.PI) / Math.Sqrt(a);
        AssertDerivative(-scale, XMath.Special.LogRegularizedGammaPShapeDerivative(a, a));
        AssertDerivative(scale, XMath.Special.LogRegularizedGammaQShapeDerivative(a, a));
    }

    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(1e150, 1e149, -2.302585092994046, 0)]
    [InlineData(1e150, 2e150, -0.0, 0.6931471805599453)]
    [InlineData(1e308, 1.4e308, -0.0, 0.3364722366212129)]
    public void VeryLargeShapeUsesTheLogRatioInTheTails(double a, double x, double expectedP, double expectedQ)
    {
        AssertDerivative(expectedP, XMath.Special.LogRegularizedGammaPShapeDerivative(a, x));
        AssertDerivative(expectedQ, XMath.Special.LogRegularizedGammaQShapeDerivative(a, x));
    }

    // mpmath 1.4.1 の参照値。通常域、分岐境界、非正規数と確率が 0 に丸まる裾を含む。
    // tools/SpecialFunctionReferences/generate_incomplete_gamma_shape_derivatives.py で生成。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(0.1, 0.1, -1.959977067134353, 9.405619139746607)]
    [InlineData(0.4, 0.1, -2.291935518591947, 1.7734839943174883)]
    [InlineData(0.5, 0.5, -0.9480156935648195, 2.039643618286225)]
    [InlineData(3.0, 2.5, -0.5417020218846704, 0.45441595699998744)]
    [InlineData(3.0, 5.0, -0.1298830193458, 0.9120818030807539)]
    [InlineData(10.0, 10.0, -0.234683262680543, 0.2778042550845187)]
    [InlineData(1.0, 1.0, -0.6829863458878442, 1.1735630272247268)]
    [InlineData(1.0, 2.0, -0.25538850138172997, 1.6316914623497007)]
    [InlineData(2.0, 3.0, -0.24651931567280672, 0.9913488886334721)]
    [InlineData(20.0, 21.0, -0.13856888618637242, 0.22204082203445955)]
    [InlineData(100.0, 101.0, -0.0715004599635879, 0.0884187241398769)]
    [InlineData(15.999999999999998, 16.0, -0.1880099593673906, 0.21480100404433852)]
    [InlineData(16.0, 16.0, -0.18800995936739065, 0.21480100404433844)]
    [InlineData(16.000000000000004, 16.0, -0.1880099593673908, 0.21480100404433827)]
    [InlineData(0.4, 0.49999999999999994, -0.8791059958096599, 2.6096667384437398)]
    [InlineData(0.4, 0.5, -0.8791059958096598, 2.6096667384437398)]
    [InlineData(0.4, 0.5000000000000001, -0.8791059958096595, 2.6096667384437398)]
    [InlineData(0.4, 1.0999999999999999, -0.3636110598521935, 3.1281591238832536)]
    [InlineData(0.4, 1.1, -0.3636110598521934, 3.128159123883254)]
    [InlineData(0.4, 1.1000000000000003, -0.3636110598521933, 3.128159123883254)]
    [InlineData(3.0, 3.9999999999999996, -0.2339521344200499, 0.7486135375522739)]
    [InlineData(3.0, 4.0, -0.23395213442004983, 0.7486135375522739)]
    [InlineData(3.0, 4.000000000000001, -0.2339521344200497, 0.7486135375522741)]
    [InlineData(1e-20, 0.1, -1.8229239584193906, 1e+20)]
    [InlineData(9.999999999999998e-21, 0.1, -1.8229239584193906, 1.0000000000000002e+20)]
    [InlineData(1.0000000000000001e-20, 0.1, -1.8229239584193906, 9.999999999999998e+19)]
    [InlineData(1e-20, 100.0, -3.683597761682032e-46, 1e+20)]
    [InlineData(1e-100, 2.0, -0.04890051070806112, 1e+100)]
    [InlineData(1e-300, 710.0, -6.29577363674e-312, 9.999999999999999e+299)]
    [InlineData(1e-310, 0.1, -1.8229239584193906, double.PositiveInfinity)]
    [InlineData(double.Epsilon, double.Epsilon, -743.8628562564797, double.PositiveInfinity)]
    [InlineData(double.Epsilon, 3.0, -0.013048381094197037, double.PositiveInfinity)]
    [InlineData(0.1, double.Epsilon, -744.0163169809701, 3.652752543745522e-30)]
    [InlineData(1.0, double.Epsilon, -744.8628562564797, 3.68e-321)]
    [InlineData(2.0, 1e-155, -357.82347374917555, 1.7891173687458777e-308)]
    [InlineData(10.0, 1e-100, -232.61026188847129, 0.0)]
    [InlineData(1.0, 740.0, -3.01e-321, 7.185215381216265)]
    [InlineData(2.0, 800.0, -0.0, 6.263077390623632)]
    [InlineData(20.0, 800.0, -1.65369998494121e-309, 3.7153664306918492)]
    [InlineData(20.0, 900.0, -0.0, 3.833004504163919)]
    [InlineData(0.018074874357378274, 703.1742252909537, -7.51609873004177e-309, 62.43031210066157)]
    [InlineData(20000.0, 15154.282258895793, -0.2776302263184811, 1.216900516260107e-308)]
    [InlineData(20000.0, 25780.15285321662, -1.64233414169661e-308, 0.25407039305696105)]
    [InlineData(9999.999999999998, 10000.0, -0.00795774754081316, 0.008000189536441016)]
    [InlineData(10000.0, 10000.0, -0.007957747540813274, 0.0080001895364409)]
    [InlineData(10000.0, 9999.0, -0.008021582321822734, 0.007936693416497066)]
    [InlineData(10000.0, 10001.0, -0.007894137529958007, 0.008063896929034246)]
    [InlineData(10000.0, 11000.0, -1.6308824174316845e-23, 0.09633927300615691)]
    [InlineData(10000.0, 9200.0, -0.08454806652618439, 8.565504458762194e-18)]
    [InlineData(10000.0, 10900.0, -9.594265823978333e-20, 0.08731069677901519)]
    [InlineData(10000.0, 6999.999999999999, -0.35695776018131053, 3.4666568409766965e-249)]
    [InlineData(10000.0, 7000.0, -0.35695776018131037, 3.4666568409780476e-249)]
    [InlineData(10000.0, 7000.000000000001, -0.35695776018131026, 3.466656840979399e-249)]
    [InlineData(10000.0, 12999.999999999998, -1.2381533972053652e-166, 0.26274664208113024)]
    [InlineData(10000.0, 13000.0, -1.2381533972048453e-166, 0.2627466420811304)]
    [InlineData(10000.0, 13000.000000000002, -1.2381533972043255e-166, 0.2627466420811305)]
    [InlineData(1000000.0, 1000000.0, -0.0007976724771073847, 0.0007980968903567098)]
    [InlineData(1000000.0, 1004000.0, -1.3643895750979707e-07, 0.004218044753205066)]
    [InlineData(1000000.0, 996000.0, -0.004233212295575216, 1.312637841731833e-07)]
    [InlineData(1e+30, 9.999999999999633e+29, -3.6759668115768824e-14, 4.06889505669033e-309)]
    [InlineData(1e+30, 1.0000000000000368e+30, -4.06889505682462e-309, 3.6759668115767473e-14)]
    [InlineData(1e+31, 9.999999999999884e+30, -1.1605379355483638e-14, 1.175790016910328e-308)]
    [InlineData(1e+31, 1.0000000000000116e+31, -1.175790016922539e-308, 1.1605379355483504e-14)]
    [InlineData(1.0517121179454555e-06, 3.3621969126596476e-26, -58.077403879573076, 950801.5147752841)]
    [InlineData(11.5129637163737, 1.1667926804523423, -2.3400476783204125, 3.3554535618348635e-08)]
    [InlineData(0.022224236130130286, 7.790941447805949e-14, -29.64198925124416, 31.81676459418579)]
    [InlineData(118.6245679013424, 202.78824476536872, -3.270085673005771e-11, 0.5516934235025442)]
    [InlineData(88393.2636644859, 24997.01514556811, -1.2630494573442206, 0.0)]
    [InlineData(5.791364575948579e-06, 1.4291049708097084e-05, -10.578685263494991, 172665.66539575666)]
    [InlineData(12.271239908291502, 24.468908323088385, -0.0019167767830827471, 0.7964512563123577)]
    [InlineData(1.6187363946204597e-05, 0.0003230476196353002, -7.460844789402037, 61772.96084570133)]
    [InlineData(0.06080852357454892, 3.21024999931217e-30, -67.42979948448382, 1.139291746638378)]
    [InlineData(0.253681025898513, 2.1575349843984267e-29, -65.78294623164342, 3.882588650006733e-06)]
    [InlineData(0.02795060462051987, 9.523455312314873e-14, -29.45027872999082, 23.075707886974836)]
    [InlineData(2.2957903307788938e-06, 2.3014557008751982e-29, -65.3642139721893, 435547.17784685263)]
    [InlineData(7.483592668904743, 9.197132670776665e-06, -13.67465937588008, 2.0585430075792699e-41)]
    [InlineData(1220.6490551987993, 180.73855146812858, -1.9106381151619147, 0.0)]
    [InlineData(25887.77078468182, 76666.87663289708, -0.0, 1.0857380784504798)]
    [InlineData(519.3674706004546, 1740.9532019809674, -1.805615504673207e-260, 1.2113561162243174)]
    public void MatchesMpmathReference(double a, double x, double expectedP, double expectedQ)
    {
        AssertDerivative(expectedP, XMath.Special.LogRegularizedGammaPShapeDerivative(a, x));
        AssertDerivative(expectedQ, XMath.Special.LogRegularizedGammaQShapeDerivative(a, x));
    }

    private static void AssertDerivative(double expected, double actual)
    {
        if (expected == 0)
        {
            Assert.Equal(BitConverter.DoubleToInt64Bits(expected), BitConverter.DoubleToInt64Bits(actual));
        }
        else if (!double.IsFinite(expected))
        {
            Assert.Equal(expected, actual);
        }
        else if (double.IsSubnormal(expected))
        {
            Assert.Equal(double.IsNegative(expected), double.IsNegative(actual));
            SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 0, absoluteTolerance: 2 * double.Epsilon);
        }
        else
        {
            SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 5e-13);
        }
    }
}
