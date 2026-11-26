using Tuna.LinearAlgebra;
using Tuna.SignalProcessing;
using static Tuna.Tests.SignalProcessing.WaveletAssertions;

namespace Tuna.Tests.SignalProcessing;

/// <summary>ウェーブレットパケット分解、最良基底、節の組からの再構成</summary>
[Trait("Wavelets", "Packets")]
public class WaveletPacketTests
{
    [Fact]
    public void PacketsMatchPyWavelets()
    {
        foreach (var (name, boundary, levels, x, nodes) in WaveletExtensionReferences.Packets)
        {
            var w = DiscreteWavelet.FromName(name);
            var tree = levels < 0 ? Wavelets.DecomposePackets(x, w, boundary) : Wavelets.DecomposePackets(x, w, levels, boundary);
            Assert.Equal(nodes.Length - 1, tree.Levels);
            Assert.Equal(x.Length, tree.Length);
            for (int level = 0; level <= tree.Levels; level++)
            {
                var matrix = tree.Level(level);
                Assert.Equal(1 << level, matrix.Rows);
                for (int i = 0; i < matrix.Rows; i++)
                {
                    var expected = nodes[level][i];
                    Close(expected, tree.Node(level, i), Tolerance(name, MaxAbs(expected)), $"{name} {boundary} node ({level}, {i})");
                }
            }
        }
    }

    [Fact]
    public void FrequencyOrderMatchesPyWavelets()
    {
        var expected = WaveletExtensionReferences.FrequencyOrder;
        for (int p = 0; p < expected.Length; p++)
        {
            Assert.Equal(expected[p], WaveletPacketTree.NaturalIndex(p));
            Assert.Equal(p, WaveletPacketTree.FrequencyIndex(expected[p]));
        }

        for (int i = 0; i < 4096; i++)
        {
            Assert.Equal(i, WaveletPacketTree.FrequencyIndex(WaveletPacketTree.NaturalIndex(i)));
            Assert.Equal(i, WaveletPacketTree.NaturalIndex(WaveletPacketTree.FrequencyIndex(i)));
        }

        Assert.Equal(int.MaxValue, WaveletPacketTree.FrequencyIndex(WaveletPacketTree.NaturalIndex(int.MaxValue)));
        Assert.Throws<ArgumentOutOfRangeException>(() => WaveletPacketTree.NaturalIndex(-1));

        var tree = Wavelets.DecomposePackets(RandomSignal(64, 1), DiscreteWavelet.Daubechies(2), 3);
        var byFrequency = tree.LevelByFrequency(3);
        for (int p = 0; p < 8; p++)
        {
            Identical(tree.Node(3, WaveletPacketTree.NaturalIndex(p)), byFrequency.Row(p), $"frequency row {p}");
        }
    }

    [Fact]
    public void PacketReconstructionsMatchPyWavelets()
    {
        foreach (var (name, boundary, levels, x, basis, scales, expected) in WaveletExtensionReferences.PacketReconstructions)
        {
            var w = DiscreteWavelet.FromName(name);
            var tree = Wavelets.DecomposePackets(x, w, levels, boundary);
            for (int k = 0; k < basis.Length; k++)
            {
                XMath.InPlace.Multiply(tree.Node(basis[k].Level, basis[k].Index), scales[k]);
            }

            // PyWavelets の reconstruct は元の長さにそろえる
            Close(expected, Wavelets.Reconstruct(tree, basis), Tolerance(name, MaxAbs(expected)), $"{name} {boundary}");
        }
    }

    [Theory]
    [InlineData("db4", WaveletBoundary.Symmetric, 77)]
    [InlineData("sym3", WaveletBoundary.Periodization, 64)]
    [InlineData("bior3.3", WaveletBoundary.Zero, 45)]
    [InlineData("coif1", WaveletBoundary.Reflect, 50)]
    [InlineData("haar", WaveletBoundary.Constant, 31)]
    [InlineData("db2", WaveletBoundary.Periodic, 40)]
    public void EveryLevelAndTheBestBasisRoundTrip(string name, WaveletBoundary boundary, int n)
    {
        var w = DiscreteWavelet.FromName(name);
        var x = RandomSignal(n, n);
        var tree = Wavelets.DecomposePackets(x, w, 3, boundary);
        for (int level = 0; level <= 3; level++)
        {
            Close(x, Wavelets.Reconstruct(tree, level), 1e-12, $"{name} level {level}");
        }

        Close(x, Wavelets.Reconstruct(tree, tree.BestBasis()), 1e-12, $"{name} best basis");
    }

    [Fact]
    public void BestBasisMinimizesTheCostOverAllBases()
    {
        // 深さ 3 の木の基底（26 通り）をすべて調べる
        var x = new double[96];
        for (int t = 0; t < x.Length; t++)
        {
            x[t] = Math.Sin(0.9 * t) + 0.3 * Math.Cos(2.7 * t) + (t > 40 && t < 44 ? 1.5 : 0);
        }

        var tree = Wavelets.DecomposePackets(x, DiscreteWavelet.Symlet(4), 3, WaveletBoundary.Periodization);
        foreach (var cost in new Func<ReadOnlyVectorSpan<double>, double>[]
                 {
                     WaveletPacketTree.ShannonEntropy,
                     static c =>
                     {
                         double s = 0;
                         for (int i = 0; i < c.Length; i++)
                         {
                             s += Math.Abs(c[i]);
                         }

                         return s;
                     },
                 })
        {
            var best = tree.BestBasis(cost);
            double bestCost = best.Sum(node => cost(tree.Node(node.Level, node.Index)));
            double minimum = double.PositiveInfinity;
            int count = 0;
            foreach (var basis in AllBases(0, 0, 3))
            {
                minimum = Math.Min(minimum, basis.Sum(node => cost(tree.Node(node.Level, node.Index))));
                count++;
            }

            Assert.Equal(26, count);
            Assert.True(bestCost <= minimum + 1e-12 * Math.Abs(minimum), $"{bestCost:R} > {minimum:R}");

            // 深さ優先の順（区間の左端の昇順）に並ぶ
            var starts = best.Select(node => node.Index << (3 - node.Level)).ToArray();
            Assert.Equal(starts.OrderBy(s => s).ToArray(), starts);
        }
    }

    [Fact]
    public void BestBasisPrefersTheParentOnTies()
    {
        var tree = Wavelets.DecomposePackets(RandomSignal(32, 4), DiscreteWavelet.Haar, 2);
        Assert.Equal([(0, 0)], tree.BestBasis(static _ => 0.0));
        Assert.Equal([(2, 0), (2, 1), (2, 2), (2, 3)], tree.BestBasis(static c => c.Length == 32 ? 100.0 : c.Length == 16 ? 10.0 : 1.0));
        Assert.Equal(-(0.25 * Math.Log(0.25) + 4 * Math.Log(4)), WaveletPacketTree.ShannonEntropy(new double[] { 0.5, 0, -2 }), 1e-15);
    }

    [Fact]
    public void NodesAreViewsOfTheTree()
    {
        var x = RandomSignal(40, 6);
        var w = DiscreteWavelet.Daubechies(3);
        var tree = Wavelets.DecomposePackets(x, w, 2);
        tree.Node(2, 3).Fill(0);
        Assert.Equal(0, tree.Level(2)[3, 5]);

        // 組に入っていない節の値は使わない
        tree.Level(1).Row(1).Fill(double.NaN);
        var y = Wavelets.Reconstruct(tree, [(1, 0), (2, 2), (2, 3)]);
        Assert.All(ToArray(y), v => Assert.True(double.IsFinite(v)));
        Assert.Throws<ArgumentOutOfRangeException>(() => tree.Node(2, 4));
        Assert.Throws<ArgumentOutOfRangeException>(() => tree.Level(3));
    }

    [Fact]
    public void ReconstructRejectsInvalidBases()
    {
        var tree = Wavelets.DecomposePackets(RandomSignal(64, 8), DiscreteWavelet.Daubechies(2), 3);
        Assert.Throws<ArgumentException>(() => Wavelets.Reconstruct(tree, [(1, 0), (1, 0), (1, 1)]));
        Assert.Throws<ArgumentException>(() => Wavelets.Reconstruct(tree, [(1, 0), (2, 0), (1, 1)]));
        Assert.Throws<ArgumentException>(() => Wavelets.Reconstruct(tree, [(1, 0), (2, 2)]));
        Assert.Throws<ArgumentException>(() => Wavelets.Reconstruct(tree, Array.Empty<(int, int)>()));
        Assert.Throws<ArgumentOutOfRangeException>(() => Wavelets.Reconstruct(tree, [(1, 0), (1, 2)]));
        Assert.Throws<ArgumentOutOfRangeException>(() => Wavelets.Reconstruct(tree, [(4, 0)]));
        Assert.Throws<ArgumentOutOfRangeException>(() => Wavelets.Reconstruct(tree, 4));
    }

    [Fact]
    public void DecomposePacketsValidatesArguments()
    {
        var w = DiscreteWavelet.Daubechies(2);
        Assert.Throws<ArgumentException>(() => Wavelets.DecomposePackets(Array.Empty<double>(), w));
        Assert.Throws<ArgumentOutOfRangeException>(() => Wavelets.DecomposePackets(new double[8], w, -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => Wavelets.DecomposePackets(new double[8], w, 31));
        Assert.Throws<ArgumentOutOfRangeException>(() => Wavelets.DecomposePackets(new double[2], DiscreteWavelet.Haar, 2, WaveletBoundary.Reflect));

        // 既定の段数は MaxLevel（短い信号は根だけ）
        Assert.Equal(Wavelets.MaxLevel(100, w), Wavelets.DecomposePackets(new double[100], w).Levels);
        var root = Wavelets.DecomposePackets(new double[] { 1, 2 }, w);
        Assert.Equal(0, root.Levels);
        Assert.Equal([(0, 0)], root.BestBasis());
        Close([1, 2], Wavelets.Reconstruct(root, 0), 0, "root");
    }

    private static IEnumerable<List<(int Level, int Index)>> AllBases(int level, int index, int depth)
    {
        yield return [(level, index)];
        if (level == depth)
        {
            yield break;
        }

        foreach (var left in AllBases(level + 1, 2 * index, depth))
        {
            foreach (var right in AllBases(level + 1, 2 * index + 1, depth))
            {
                yield return [.. left, .. right];
            }
        }
    }
}
