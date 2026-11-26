using Tuna.LinearAlgebra;
using Tuna.SignalProcessing;

namespace Tuna.Tests.SignalProcessing;

/// <summary>
/// 拡張の畳み込み・相関を、コアの入口 <see cref="XMath"/> の静的拡張メンバー（C# 14）として呼べることを確かめる。
/// このファイルは <c>using static</c> を使わず、利用者が <c>XMath.Convolve(...)</c> と型名で書く場合と同じ名前解決を通す。
/// </summary>
[Trait("SignalProcessing", "Extension")]
public class XMathExtensionTests
{
    private static readonly double[] Signal = [1, 2, 3, 4, 5, 6, 7, 8];
    private static readonly double[] Kernel = [2, -1, 0.5];

    [Fact]
    public void QualifiedCallsMatchTheProvider()
    {
        var x = new Vector<double>(Signal);
        var y = new Vector<double>(Kernel);
        foreach (var method in new[] { ConvolutionMethod.Auto, ConvolutionMethod.Direct, ConvolutionMethod.Fourier })
        {
            foreach (var mode in new[] { ConvolutionMode.Full, ConvolutionMode.Same, ConvolutionMode.Valid })
            {
                Assert.Equal(XMathExtensions.Convolve(x, y, mode, method).ToArray(), XMath.Convolve(x, y, mode, method).ToArray());
                Assert.Equal(XMathExtensions.Correlate(x, y, mode, method).ToArray(), XMath.Correlate(x, y, mode, method).ToArray());
            }

            Assert.Equal(XMathExtensions.AutoCorrelate(x, 3, method).ToArray(), XMath.AutoCorrelate(x, 3, method).ToArray());
        }

        Assert.Equal(new double[] { 2, 3, 4.5, 6, 7.5, 9, 10.5, 12, -4.5, 4 }, XMath.Convolve(x, y).ToArray());
    }

    [Fact]
    public void GeneratedOverloadsInferTheElementType()
    {
        // 生成したオーバーロード（コンテナー・書き込み可能なビュー・読み取り専用ビューの混在）も拡張ブロックに入っている
        var x = new Vector<double>(Signal);
        var m = Matrix<double>.FromColumnMajor(2, 4, Signal);
        VectorSpan<double> row = m.Row(1);
        ReadOnlyVectorSpan<double> kernel = new Vector<double>(Kernel);
        var expected = XMath.Convolve(row.ToVector(), kernel.ToVector()).ToArray();
        Assert.Equal(expected, XMath.Convolve(row, kernel).ToArray());
        Assert.Equal(expected, XMath.Convolve(m.Row(1), new Vector<double>(Kernel)).ToArray());
        Assert.Equal(XMath.Convolve(x, x).ToArray(), XMath.Convolve<double>(x, x).ToArray());
    }

    [Fact]
    public void ContainerDestinationIsReturned()
    {
        var x = new Vector<double>(Signal);
        var y = new Vector<double>(Kernel);
        var destination = new Vector<double>(Signal.Length + Kernel.Length - 1);
        Vector<double> returned = XMath.Convolve(x, y, destination);
        Assert.Same(destination, returned);
        Assert.Equal(XMath.Convolve(x, y).ToArray(), destination.ToArray());

        var columns = new Matrix<double>(3, 2);
        var draws = Matrix<double>.FromColumnMajor(4, 2, Signal);
        Matrix<double> sameColumns = XMath.ColumnAutoCorrelations(draws, 2, columns);
        Assert.Same(columns, sameColumns);
        Assert.Equal(XMath.AutoCorrelate(draws.Column(1).ToVector(), 2).ToArray(), columns.Column(1).ToArray());
    }

    [Fact]
    public void CoreDoesNotReferenceTheExtension()
    {
        // コアの入口に Convolve の実体はなく、コアのアセンブリはこの拡張を参照しない（依存は拡張からコアへの一方向）
        Assert.DoesNotContain(typeof(XMath).GetMethods(), m => m.Name is "Convolve" or "Correlate" or "AutoCorrelate");
        Assert.DoesNotContain(typeof(XMath).Assembly.GetReferencedAssemblies(), a => a.Name == typeof(XMathExtensions).Assembly.GetName().Name);
        Assert.Equal("Tuna.SignalProcessing", typeof(Fourier).Assembly.GetName().Name);
        Assert.Equal("Tuna.SignalProcessing", typeof(Wavelets).Assembly.GetName().Name);
        // 公開する型の名前空間はアセンブリ名から始める
        Assert.All(typeof(XMathExtensions).Assembly.GetExportedTypes(), t => Assert.Equal("Tuna.SignalProcessing", t.Namespace));
    }
}
