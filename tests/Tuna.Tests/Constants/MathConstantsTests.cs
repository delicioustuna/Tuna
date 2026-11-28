using System.Globalization;
using System.Numerics;
using System.Reflection;
using System.Runtime.InteropServices;

namespace Tuna.Tests;

/// <summary>数学定数（XMath.Constants・XMath.Constants&lt;T&gt;）と内部定数（NumericConstants）の値の検証</summary>
public class MathConstantsTests
{
    private const BindingFlags PublicStatic = BindingFlags.Public | BindingFlags.Static;
    private const BindingFlags InternalStatic = BindingFlags.NonPublic | BindingFlags.Static;

    public static TheoryData<string, string> References()
    {
        var data = new TheoryData<string, string>();
        foreach (var (name, value) in MathConstantReferences.Values)
        {
            data.Add(name, value);
        }

        return data;
    }

    [Fact]
    public void ReferenceTable_CoversEveryPublicConstant()
    {
        var expected = MathConstantReferences.Values.Select(r => r.Name).Order(StringComparer.Ordinal).ToArray();

        var doubles = typeof(XMath.Constants).GetFields(PublicStatic);
        Assert.All(doubles, f => Assert.True(f.IsLiteral, $"{f.Name} は const にする"));
        Assert.Equal(expected, doubles.Select(f => f.Name).Order(StringComparer.Ordinal));

        foreach (var type in new[] { typeof(XMath.Constants<double>), typeof(XMath.Constants<float>), typeof(XMath.Constants<Half>), typeof(XMath.Constants<NFloat>) })
        {
            var fields = type.GetFields(PublicStatic);
            Assert.All(fields, f => Assert.True(f.IsInitOnly, $"{type}.{f.Name} は static readonly にする"));
            Assert.Equal(expected, fields.Select(f => f.Name).Order(StringComparer.Ordinal));
        }
    }

    [Theory]
    [MemberData(nameof(References))]
    public void Double_IsNearestToReference(string name, string reference)
    {
        double expected = double.Parse(reference, CultureInfo.InvariantCulture);
        double actual = (double)typeof(XMath.Constants).GetField(name, PublicStatic)!.GetRawConstantValue()!;
        Assert.Equal(BitConverter.DoubleToInt64Bits(expected), BitConverter.DoubleToInt64Bits(actual));
        Assert.Equal(BitConverter.DoubleToInt64Bits(expected), BitConverter.DoubleToInt64Bits(Generic<double>(name)));
    }

    [Theory]
    [MemberData(nameof(References))]
    public void Single_IsNearestToReferenceWithoutDoubleRounding(string name, string reference)
    {
        float expected = float.Parse(reference, CultureInfo.InvariantCulture);
        Assert.Equal(BitConverter.SingleToInt32Bits(expected), BitConverter.SingleToInt32Bits(Generic<float>(name)));
    }

    [Theory]
    [MemberData(nameof(References))]
    public void Half_IsNearestToReferenceWithoutDoubleRounding(string name, string reference)
    {
        Half expected = Half.Parse(reference, CultureInfo.InvariantCulture);
        Assert.Equal(BitConverter.HalfToUInt16Bits(expected), BitConverter.HalfToUInt16Bits(Generic<Half>(name)));
    }

    [Theory]
    [MemberData(nameof(References))]
    public void NFloat_MatchesPlatformSize(string name, string reference)
    {
        NFloat expected = NFloat.Size == sizeof(double)
            ? new NFloat(double.Parse(reference, CultureInfo.InvariantCulture))
            : new NFloat(float.Parse(reference, CultureInfo.InvariantCulture));
        Assert.Equal(expected, Generic<NFloat>(name));
    }

    [Fact]
    public void BclConstants_Agree()
    {
        Assert.Equal(Math.PI, XMath.Constants.Pi);
        Assert.Equal(Math.Tau, XMath.Constants.Tau);
        Assert.Equal(Math.E, XMath.Constants.E);
        Assert.Equal(MathF.PI, XMath.Constants<float>.Pi);
        Assert.Equal(MathF.Tau, XMath.Constants<float>.Tau);
        Assert.Equal(MathF.E, XMath.Constants<float>.E);
        Assert.Equal(Half.Pi, XMath.Constants<Half>.Pi);
        Assert.Equal(Half.Tau, XMath.Constants<Half>.Tau);
        Assert.Equal(Half.E, XMath.Constants<Half>.E);
        Assert.Equal(NFloat.Pi, XMath.Constants<NFloat>.Pi);
        Assert.Equal(XMath.Constants.EulerGamma, XMath.Special.EulerGamma);
    }

    // 平方根と四則演算は IEEE 754 で正しく丸められるため、倍精度の式との一致・不一致は実行環境によらない。
    [Fact]
    public void DoublePrecisionExpressions_MatchOrDifferInLastPlace()
    {
        Assert.Equal(XMath.Constants.Sqrt2, Math.Sqrt(2));
        Assert.Equal(XMath.Constants.Sqrt3, Math.Sqrt(3));
        Assert.Equal(XMath.Constants.OneOverSqrt2, Math.Sqrt(0.5));
        Assert.Equal(XMath.Constants.OneOverSqrtPi, 1 / Math.Sqrt(Math.PI));
        Assert.Equal(XMath.Constants.SqrtTwoOverPi, Math.Sqrt(2 / Math.PI));

        Assert.Equal(Math.BitDecrement(XMath.Constants.OneOverSqrt2), 1 / Math.Sqrt(2));
        Assert.Equal(Math.BitDecrement(XMath.Constants.SqrtPi), Math.Sqrt(Math.PI));
        Assert.Equal(Math.BitDecrement(XMath.Constants.SqrtTwoPi), Math.Sqrt(2 * Math.PI));
        Assert.Equal(Math.BitDecrement(XMath.Constants.SqrtPiOver2), Math.Sqrt(Math.PI / 2));
    }

    // 二次方程式の根を、丸めの両側の中点で多項式の符号が反転することで（mpmath を使わず）厳密に確かめる。
    [Theory]
    [InlineData(nameof(XMath.Constants.Sqrt2), 1, 0, -2)]
    [InlineData(nameof(XMath.Constants.Sqrt3), 1, 0, -3)]
    [InlineData(nameof(XMath.Constants.OneOverSqrt2), 2, 0, -1)]
    [InlineData(nameof(XMath.Constants.GoldenRatio), 1, -1, -1)]
    public void QuadraticRoots_AreCorrectlyRoundedInEveryFormat(string name, int a2, int a1, int a0)
    {
        AssertBracketsRoot(Generic<double>(name), 53, a2, a1, a0);
        AssertBracketsRoot(Generic<float>(name), 24, a2, a1, a0);
        AssertBracketsRoot((double)Generic<Half>(name), 11, a2, a1, a0);
    }

    [Theory]
    [MemberData(nameof(DoubleDoubleNames))]
    public void DoubleDoubleLowPart_IsNearestToResidual(string name)
    {
        string reference = MathConstantReferences.Values.Single(r => r.Name == name).Value;
        double hi = (double)typeof(XMath.Constants).GetField(name, PublicStatic)!.GetRawConstantValue()!;
        double lo = Internal(name + "Lo");

        Assert.True(Math.Abs(lo) <= 0.5 * Math.ScaleB(1.0, Math.ILogB(hi) - 52), $"{name}Lo は {name} の 1/2 ulp 以内にする");

        // lo の 1 ulp を単位とした整数で、参照値と hi + lo の差が 1/2 ulp 未満であることを確かめる。
        int unit = Math.ILogB(lo) - 52;
        var (numerator, digits) = ParseDecimal(reference);
        BigInteger scaledReference = numerator << -unit;
        BigInteger scaledSum = (ToScaledInteger(hi, unit) + ToScaledInteger(lo, unit)) * BigInteger.Pow(10, digits);
        Assert.True(2 * BigInteger.Abs(scaledReference - scaledSum) < BigInteger.Pow(10, digits), $"{name}Lo は残差を正しく丸めた値にする");
    }

    public static TheoryData<string> DoubleDoubleNames() => new(MathConstantReferences.DoubleDoubleNames);

    [Fact]
    public void MachineConstants_HaveExpectedPowersOfTwo()
    {
        Assert.Equal(Math.BitIncrement(1.0) - 1.0, Internal("MachineEpsilon"));
        Assert.Equal(Math.ScaleB(1.0, -52), Internal("MachineEpsilon"));
        Assert.Equal(Math.ScaleB(1.0, -53), Internal("UnitRoundoff"));
        Assert.Equal(Math.Sqrt(Internal("MachineEpsilon")), Internal("SqrtMachineEpsilon"));
        Assert.Equal(BitConverter.Int64BitsToDouble(0x0010_0000_0000_0000), Internal("MinNormal"));
        Assert.True(double.IsNormal(Internal("MinNormal")));
        Assert.False(double.IsNormal(Math.BitDecrement(Internal("MinNormal"))));
    }

    [Fact]
    public void LnTau_IsLogarithmOfDoubleTauAndOneUlpBelowLnTwoPi()
    {
        double lnTau = Internal("LnTau");
        Assert.Equal(double.Parse(MathConstantReferences.LnTau, CultureInfo.InvariantCulture), lnTau);
        Assert.Equal(Math.BitDecrement(XMath.Constants.LnTwoPi), lnTau);
    }

    private static T Generic<T>(string name)
        where T : IFloatingPointIeee754<T> =>
        (T)typeof(XMath.Constants<T>).GetField(name, PublicStatic)!.GetValue(null)!;

    private static double Internal(string name)
    {
        var type = typeof(XMath).Assembly.GetType("Tuna.NumericConstants", throwOnError: true)!;
        return (double)type.GetField(name, InternalStatic)!.GetRawConstantValue()!;
    }

    // value を p ビットの仮数で丸めた値とみなし、両側の中点 value ± ulp/2 で a2 x² + a1 x + a0 の符号が負から正へ変わることを確かめる。
    private static void AssertBracketsRoot(double value, int precision, int a2, int a1, int a0)
    {
        int unit = Math.ILogB(value) - precision;
        BigInteger scaled = ToScaledInteger(value, unit);
        Assert.True(Quadratic(scaled - 1, unit, a2, a1, a0).Sign < 0, $"{value:R}（{precision} ビット）の下側の中点");
        Assert.True(Quadratic(scaled + 1, unit, a2, a1, a0).Sign > 0, $"{value:R}（{precision} ビット）の上側の中点");
    }

    // x = m·2^unit（unit < 0）での (a2 x² + a1 x + a0)·2^(-2·unit)
    private static BigInteger Quadratic(BigInteger m, int unit, int a2, int a1, int a0) =>
        a2 * m * m + ((a1 * m) << -unit) + (new BigInteger(a0) << (-2 * unit));

    // value / 2^unit（整数になる unit を渡す）
    private static BigInteger ToScaledInteger(double value, int unit)
    {
        double scaled = Math.ScaleB(value, -unit);
        Assert.Equal(Math.Round(scaled), scaled);
        return new BigInteger(scaled);
    }

    // "1.234" → (1234, 3)
    private static (BigInteger Numerator, int Digits) ParseDecimal(string text)
    {
        bool negative = text.StartsWith('-');
        string body = negative ? text[1..] : text;
        int point = body.IndexOf('.');
        string digits = point < 0 ? body : body.Remove(point, 1);
        int fractionDigits = point < 0 ? 0 : body.Length - point - 1;
        var value = BigInteger.Parse(digits, CultureInfo.InvariantCulture);
        return (negative ? -value : value, fractionDigits);
    }
}
