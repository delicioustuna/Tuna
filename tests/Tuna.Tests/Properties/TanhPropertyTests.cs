using System.Numerics;
using CsCheck;

namespace Tuna.Tests.Properties;

[Trait("Category", "Property")]
public class TanhPropertyTests
{
    [Fact]
    public void TanhIsOddBoundedAndNondecreasing()
    {
        CheckProperties<double>();
        CheckProperties<float>();
    }

    private static void CheckProperties<T>() where T : unmanaged, IFloatingPointIeee754<T>
    {
        var coordinate = Gen.Int[-24_000, 24_000];
        Gen.Select(coordinate, coordinate).Sample((a, b) =>
        {
            T x = T.ScaleB(T.CreateChecked(a), -10);
            T y = T.ScaleB(T.CreateChecked(b), -10);
            T tx = XMath.Tanh(x);
            T ty = XMath.Tanh(y);
            Assert.Equal(-tx, XMath.Tanh(-x));
            Assert.InRange(tx, -T.One, T.One);
            Assert.True(a <= b ? tx <= ty : tx >= ty);
        }, iter: 1_000, seed: "176001760017");
    }
}
