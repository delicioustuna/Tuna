using System.Numerics;
using Tuna;

namespace Tuna.Tests;

// mpmath 1.4.1による独立参照値。binary64入力を正確に変換し、
// 90/160桁（正規分布は490/560桁）でbinary64への丸め一致を確認。
// 生成式と選定理由: tools/SpecialFunctionReferences/generate_remaining_accuracy_references.py
// 通常テストはC#の固定値と対象APIの直接呼び出しだけで完結する。
[Trait("SpecialFunction", "AiryZeros")]
public class AiryZerosTests
{

    // 4種類の独立零点・漸近初期値への切替・大きい零点番号。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(1, -2.338107410459767)]
    [InlineData(2, -4.08794944413097)]
    [InlineData(5, -7.944133587120853)]
    [InlineData(20, -20.537332907677566)]
    [InlineData(59, -42.47824759730839)]
    [InlineData(60, -42.95891113021656)]
    [InlineData(61, -43.43690049989685)]
    [InlineData(1000, -281.03151961252155)]
    [InlineData(100000, -6055.639744320186)]
    public void AiryAiZero_MatchesIndependentReferences(int k, double expected)
    {
        double actual = XMath.Special.AiryAiZero(k);
        if (expected == 0 || double.IsInfinity(expected))
        {
            Assert.Equal(expected, actual);
        }
        else
        {
            SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 2e-12);
            if (Math.Abs(expected) < 2.2250738585072014e-308)
            {
                Assert.Equal(Math.Sign(expected), Math.Sign(actual));
            }
        }
    }

    // 4種類の独立零点・漸近初期値への切替・大きい零点番号。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(1, -1.018792971647471)]
    [InlineData(2, -3.2481975821798366)]
    [InlineData(5, -7.37217725504777)]
    [InlineData(20, -20.188631509463374)]
    [InlineData(59, -42.23675395695976)]
    [InlineData(60, -42.71878039026198)]
    [InlineData(61, -43.19810240513271)]
    [InlineData(1000, -280.9378080358935)]
    [InlineData(100000, -6055.619558780269)]
    public void AiryAiPrimeZero_MatchesIndependentReferences(int k, double expected)
    {
        double actual = XMath.Special.AiryAiPrimeZero(k);
        if (expected == 0 || double.IsInfinity(expected))
        {
            Assert.Equal(expected, actual);
        }
        else
        {
            SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 2e-12);
            if (Math.Abs(expected) < 2.2250738585072014e-308)
            {
                Assert.Equal(Math.Sign(expected), Math.Sign(actual));
            }
        }
    }

    // 4種類の独立零点・漸近初期値への切替・大きい零点番号。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(1, -1.173713222709128)]
    [InlineData(2, -3.271093302836353)]
    [InlineData(5, -7.376762079367763)]
    [InlineData(20, -20.189244785396202)]
    [InlineData(59, -42.23689409345657)]
    [InlineData(60, -42.71891738216254)]
    [InlineData(61, -43.19823637387693)]
    [InlineData(1000, -280.93781120341527)]
    [InlineData(100000, -6055.619558787086)]
    public void AiryBiZero_MatchesIndependentReferences(int k, double expected)
    {
        double actual = XMath.Special.AiryBiZero(k);
        if (expected == 0 || double.IsInfinity(expected))
        {
            Assert.Equal(expected, actual);
        }
        else
        {
            SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 2e-12);
            if (Math.Abs(expected) < 2.2250738585072014e-308)
            {
                Assert.Equal(Math.Sign(expected), Math.Sign(actual));
            }
        }
    }

    // 4種類の独立零点・漸近初期値への切替・大きい零点番号。
    [Trait("Category", "Accuracy")]
    [Theory]
    [InlineData(1, -2.294439682614123)]
    [InlineData(2, -4.073155089071828)]
    [InlineData(5, -7.940178689168579)]
    [InlineData(20, -20.536740241453273)]
    [InlineData(59, -42.478109048720626)]
    [InlineData(60, -42.95877566465105)]
    [InlineData(61, -43.436767999268724)]
    [InlineData(1000, -281.03151644711187)]
    [InlineData(100000, -6055.639744313368)]
    public void AiryBiPrimeZero_MatchesIndependentReferences(int k, double expected)
    {
        double actual = XMath.Special.AiryBiPrimeZero(k);
        if (expected == 0 || double.IsInfinity(expected))
        {
            Assert.Equal(expected, actual);
        }
        else
        {
            SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 2e-12);
            if (Math.Abs(expected) < 2.2250738585072014e-308)
            {
                Assert.Equal(Math.Sign(expected), Math.Sign(actual));
            }
        }
    }

    [Fact]
    public void InvalidIndicesAndZeroOrdering()
    {
        Assert.True(double.IsNaN(XMath.Special.AiryAiZero(0)));
        Assert.True(double.IsNaN(XMath.Special.AiryAiPrimeZero(-1)));
        Assert.True(double.IsNaN(XMath.Special.AiryBiZero(0)));
        Assert.True(double.IsNaN(XMath.Special.AiryBiPrimeZero(-1)));
        Assert.True(XMath.Special.AiryAiZero(2) < XMath.Special.AiryAiZero(1));
        Assert.True(XMath.Special.AiryBiZero(2) < XMath.Special.AiryBiZero(1));
        SpecialFunctionAssert.Close(0, XMath.Special.AiryAi(XMath.Special.AiryAiZero(1)), relativeTolerance: 0, absoluteTolerance: 2e-14);
        SpecialFunctionAssert.Close(0, XMath.Special.AiryBiPrime(XMath.Special.AiryBiPrimeZero(1)), relativeTolerance: 0, absoluteTolerance: 2e-14);
    }
}
