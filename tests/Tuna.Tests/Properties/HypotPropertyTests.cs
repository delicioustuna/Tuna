using System.Numerics;
using CsCheck;

namespace Tuna.Tests.Properties;

[Trait("Category", "Property")]
public class HypotPropertyTests
{
    [Fact]
    public void HypotSatisfiesTheEuclideanNormProperties()
    {
        CheckProperties<double>(900);
        CheckProperties<float>(80);
    }

    private static void CheckProperties<T>(int maxExponent) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var coordinate = Gen.Int[-1000, 1000];
        Gen.Select(coordinate, coordinate, coordinate, coordinate, Gen.Int[-maxExponent, maxExponent]).Sample((a, b, c, d, exponent) =>
        {
            T x = T.CreateChecked(a), y = T.CreateChecked(b);
            T u = T.CreateChecked(c), v = T.CreateChecked(d);
            T norm = XMath.Hypot(x, y);
            Assert.InRange(XMath.Hypot(y, x), T.BitDecrement(norm), T.BitIncrement(norm));
            Assert.Equal(norm, XMath.Hypot(-x, y));
            Assert.Equal(norm, XMath.Hypot(x, -y));
            Assert.True(norm >= T.Max(T.Abs(x), T.Abs(y)));

            T scaled = T.ScaleB(norm, exponent);
            Assert.InRange(XMath.Hypot(T.ScaleB(x, exponent), T.ScaleB(y, exponent)), T.BitDecrement(scaled), T.BitIncrement(scaled));

            T sum = norm + XMath.Hypot(u, v);
            Assert.True(XMath.Hypot(x + u, y + v) <= T.BitIncrement(sum));
        }, iter: 1000, seed: "176001760017");
    }
}
