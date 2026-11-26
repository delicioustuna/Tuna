using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.Tests.Differentiation;

/// <summary>連立一次方程式・三角求解・LU 分解・Cholesky 分解の記録の値（コアとのビット一致）と勾配（中心差分）</summary>
public class LinearAlgebraTests
{
    // 次数の組。Cholesky の逆伝播のブロック（64）の境目と、ブロックを3つ以上に分ける次数を含める。
    public static TheoryData<int> Orders() => [1, 2, 5, 31, 64, 65, 130];

    [Theory]
    [MemberData(nameof(Orders))]
    public void SolveValuesMatchXMathBitForBit(int n)
    {
        var random = new Random(n);
        var storage = RandomMatrix(random, n + 2, n + 3);
        var b = RandomVector(random, n);
        var bs = RandomMatrix(random, n, 4);
        var tape = new Tape();
        foreach (bool transposed in new[] { false, true })
        {
            // 部分行列（列の間隔が行数と違う）と、その転置のビュー。
            ReadOnlyMatrixSpan<double> part = ((ReadOnlyMatrixSpan<double>)storage)[1..(n + 1), 2..(n + 2)];
            if (transposed)
            {
                part = part.T;
            }

            var expectedVector = XMath.Solve(part, b);
            var expectedMatrix = XMath.Solve(part, bs);
            foreach (var source in new[] { tape.Leaf(storage), tape.Constant(storage) })
            {
                var a = transposed ? source[1..(n + 1), 2..(n + 2)].T : source[1..(n + 1), 2..(n + 2)];
                AssertBits(expectedVector.AsSpan(), XMath.Solve(a, tape.Constant(b)).Value.ToVector().AsSpan());
                AssertBits(expectedVector.AsSpan(), XMath.Solve(a, tape.Leaf(b)).Value.ToVector().AsSpan());
                AssertBits(expectedMatrix.AsSpan(), XMath.Solve(a, tape.Constant(bs)).Value.ToMatrix().AsSpan());
                AssertBits(expectedMatrix.AsSpan(), XMath.Solve(a, tape.Leaf(bs)).Value.ToMatrix().AsSpan());
            }
        }
    }

    [Theory]
    [MemberData(nameof(Orders))]
    public void LuNodeValuesMatchLuDecompositionBitForBit(int n)
    {
        var random = new Random(n + 100);
        var storage = RandomMatrix(random, n, n);
        var b = RandomVector(random, n);
        var bs = RandomMatrix(random, n, 3);
        foreach (bool transposed in new[] { false, true })
        {
            var expected = LuDecomposition.Compute(transposed ? storage.T.ToMatrix() : storage);
            var tape = new Tape();
            foreach (var source in new[] { tape.Leaf(storage), tape.Constant(storage) })
            {
                var lu = LuDecompositionNode.Compute(transposed ? source.T : source);
                Assert.Equal(n, lu.Order);
                Assert.False(lu.IsSingular);
                Assert.Equal(expected.DeterminantSign, lu.DeterminantSign);
                DifferentiationAssert.EqualBits(expected.LogAbsDeterminant, lu.LogAbsDeterminant.Value);
                AssertBits(expected.Solve(b).AsSpan(), lu.Solve(tape.Constant(b)).Value.ToVector().AsSpan());
                AssertBits(expected.SolveTransposed(b).AsSpan(), lu.SolveTransposed(tape.Leaf(b)).Value.ToVector().AsSpan());
                AssertBits(expected.Solve(bs).AsSpan(), lu.Solve(tape.Leaf(bs)).Value.ToMatrix().AsSpan());
                AssertBits(expected.SolveTransposed(bs).AsSpan(), lu.SolveTransposed(tape.Constant(bs)).Value.ToMatrix().AsSpan());
            }
        }
    }

    [Theory]
    [MemberData(nameof(Orders))]
    public void TriangularSolveValuesMatchXMathBitForBit(int n)
    {
        var random = new Random(n + 200);
        var storage = WellConditionedTriangles(random, n);
        var b = RandomVector(random, n);
        var bs = RandomMatrix(random, n, 5);
        var tape = new Tape();
        foreach (bool transposed in new[] { false, true })
        {
            ReadOnlyMatrixSpan<double> view = transposed ? storage.T : storage;
            foreach (var source in new[] { tape.Leaf(storage), tape.Constant(storage) })
            {
                var a = transposed ? source.T : source;
                AssertBits(XMath.SolveLowerTriangular(view, b).AsSpan(), XMath.SolveLowerTriangular(a, tape.Leaf(b)).Value.ToVector().AsSpan());
                AssertBits(XMath.SolveUpperTriangular(view, b).AsSpan(), XMath.SolveUpperTriangular(a, tape.Constant(b)).Value.ToVector().AsSpan());
                AssertBits(XMath.SolveLowerTriangular(view, bs).AsSpan(), XMath.SolveLowerTriangular(a, tape.Constant(bs)).Value.ToMatrix().AsSpan());
                AssertBits(XMath.SolveUpperTriangular(view, bs).AsSpan(), XMath.SolveUpperTriangular(a, tape.Leaf(bs)).Value.ToMatrix().AsSpan());
            }
        }
    }

    [Theory]
    [MemberData(nameof(Orders))]
    public void CholeskyValuesMatchCholeskyDecompositionBitForBit(int n)
    {
        var random = new Random(n + 300);
        var storage = SymmetricPositiveDefinite(random, n, garbleUpper: true);
        var b = RandomVector(random, n);
        var bs = RandomMatrix(random, n, 3);
        var expected = CholeskyDecomposition.Compute(storage);
        var z = XMath.SolveLowerTriangular(expected.L, b);
        // 転置のビューでは、格納の上三角部分が論理的な下三角部分になる。
        var transposedStorage = storage.T.ToMatrix();
        foreach (bool transposed in new[] { false, true })
        {
            var tape = new Tape();
            var stored = transposed ? transposedStorage : storage;
            foreach (var source in new[] { tape.Leaf(stored), tape.Constant(stored) })
            {
                var cholesky = CholeskyDecompositionNode.Compute(transposed ? source.T : source);
                Assert.Equal(n, cholesky.Order);
                AssertBits(expected.L.AsSpan(), cholesky.L.Value.ToMatrix().AsSpan());
                DifferentiationAssert.EqualBits(expected.LogDeterminant, cholesky.LogDeterminant.Value);
                AssertBits(expected.Solve(b).AsSpan(), cholesky.Solve(tape.Leaf(b)).Value.ToVector().AsSpan());
                AssertBits(expected.Solve(bs).AsSpan(), cholesky.Solve(tape.Constant(bs)).Value.ToMatrix().AsSpan());
                DifferentiationAssert.EqualBits(XMath.SumOfSquares(z), cholesky.InverseQuadraticForm(tape.Leaf(b)).Value);
                Assert.True(CholeskyDecompositionNode.TryCompute(transposed ? source.T : source, out var again));
                AssertBits(expected.L.AsSpan(), again.L.Value.ToMatrix().AsSpan());
            }
        }
    }

    [Theory]
    [InlineData(1, 1)]
    [InlineData(4, 1)]
    [InlineData(4, 3)]
    [InlineData(9, 2)]
    public void SolveGradientsMatchCentralDifferences(int n, int k)
    {
        var random = new Random(n * 10 + k);
        var a = RandomMatrix(random, n, n);
        Shift(a, 1.5);
        var bs = RandomMatrix(random, n, k);
        var b = RandomVector(random, n);
        var g = RandomMatrix(random, n, k);
        var h = RandomVector(random, n);
        AssertGradients([a, bs], [b],
            (tape, m, v) => Pair(tape, XMath.Solve(m[0], m[1]), g) + XMath.Dot(tape.Constant(h), XMath.Solve(m[0], v[0])),
            1e-7);
        // 転置と部分行列のビュー。勾配は格納の要素に届く。
        var storage = RandomMatrix(random, n + 2, n + 1);
        Shift(storage, 1.5, rowOffset: 1, columnOffset: 1, n);
        var right = RandomMatrix(random, k, n);
        AssertGradients([storage, right], [],
            (tape, m, v) => Pair(tape, XMath.Solve(m[0][1..(n + 1), 1..].T, m[1].T), g),
            1e-7);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(8)]
    public void LuNodeGradientsMatchCentralDifferences(int n)
    {
        var random = new Random(n + 400);
        var a = RandomMatrix(random, n, n);
        Shift(a, 1.2);
        // 行を入れ替えて行列式を負にし、符号を持つ ln |det A| も確かめる。
        if (n > 1)
        {
            for (int j = 0; j < n; j++)
            {
                (a[0, j], a[1, j]) = (a[1, j], a[0, j]);
            }
        }

        var bs = RandomMatrix(random, n, 2);
        var b = RandomVector(random, n);
        var g1 = RandomMatrix(random, n, 2);
        var g2 = RandomMatrix(random, n, 2);
        var h1 = RandomVector(random, n);
        var h2 = RandomVector(random, n);
        // 同じ分解で4通りに解き、行列式も使う。右辺 b は2回使う。
        AssertGradients([a, bs], [b], (tape, m, v) =>
        {
            var lu = LuDecompositionNode.Compute(m[0]);
            return Pair(tape, lu.Solve(m[1]), g1) + Pair(tape, lu.SolveTransposed(m[1]), g2)
                + XMath.Dot(tape.Constant(h1), lu.Solve(v[0])) + XMath.Dot(tape.Constant(h2), lu.SolveTransposed(v[0]))
                + 0.7 * lu.LogAbsDeterminant;
        }, 1e-7);
        // 転置のビューの行列式。
        AssertGradients([a], [], (tape, m, v) => LuDecompositionNode.Compute(m[0].T).LogAbsDeterminant * 1.3, 1e-7);
    }

    [Theory]
    [InlineData(1, 1)]
    [InlineData(5, 1)]
    [InlineData(5, 3)]
    [InlineData(10, 4)]
    public void TriangularSolveGradientsReachOnlyTheReadTriangle(int n, int k)
    {
        var random = new Random(n * 7 + k);
        var a = WellConditionedTriangles(random, n);
        var bs = RandomMatrix(random, n, k);
        var b = RandomVector(random, n);
        var g = RandomMatrix(random, n, k);
        var h = RandomVector(random, n);
        // 同じ格納を下三角・上三角の両方として使う。読まない側の要素の中心差分は 0 なので、勾配が読む三角部分だけに届くことも確かめられる。
        AssertGradients([a, bs], [b], (tape, m, v) =>
            Pair(tape, XMath.SolveLowerTriangular(m[0], m[1]), g) + XMath.Dot(tape.Constant(h), XMath.SolveUpperTriangular(m[0], v[0])),
            1e-7);
        AssertGradients([a, bs], [b], (tape, m, v) =>
            Pair(tape, XMath.SolveUpperTriangular(m[0].T, m[1]), g) + XMath.Dot(tape.Constant(h), XMath.SolveLowerTriangular(m[0].T, v[0])),
            1e-7);
        var tapeOnly = new Tape();
        var leaf = tapeOnly.Leaf(a);
        tapeOnly.Backward(Pair(tapeOnly, XMath.SolveLowerTriangular(leaf, tapeOnly.Constant(bs)), g));
        var gradient = tapeOnly.GetGradient(leaf);
        for (int j = 1; j < n; j++)
        {
            for (int i = 0; i < j; i++)
            {
                Assert.Equal(0, gradient[i, j]);
            }
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(6)]
    [InlineData(65)]
    public void CholeskyGradientsMatchCentralDifferences(int n)
    {
        var random = new Random(n + 500);
        // 上三角部分は読まないので値を乱しておく（中心差分は 0、勾配も 0）。
        var a = SymmetricPositiveDefinite(random, n, garbleUpper: true);
        var bs = RandomMatrix(random, n, 3);
        var b = RandomVector(random, n);
        var x = RandomVector(random, n);
        var gl = RandomMatrix(random, n, n);
        var g = RandomMatrix(random, n, 3);
        var h = RandomVector(random, n);
        AssertGradients([a, bs], [b, x], (tape, m, v) =>
        {
            var cholesky = CholeskyDecompositionNode.Compute(m[0]);
            return Pair(tape, cholesky.L, gl) + 0.9 * cholesky.LogDeterminant + Pair(tape, cholesky.Solve(m[1]), g)
                + XMath.Dot(tape.Constant(h), cholesky.Solve(v[0])) + 0.6 * cholesky.InverseQuadraticForm(v[1]);
        }, n > 10 ? 1e-6 : 1e-7);
        // 各規則を単独で（ほかの寄与と打ち消し合っていないことを確かめる）。
        if (n <= 6)
        {
            AssertGradients([a], [], (tape, m, v) => Pair(tape, CholeskyDecompositionNode.Compute(m[0]).L, gl), 1e-7);
            AssertGradients([a], [], (tape, m, v) => CholeskyDecompositionNode.Compute(m[0]).LogDeterminant * 1.0, 1e-7);
            AssertGradients([a], [x], (tape, m, v) => CholeskyDecompositionNode.Compute(m[0]).InverseQuadraticForm(v[0]) * 1.0, 1e-7);
            AssertGradients([a.T.ToMatrix()], [b], (tape, m, v) =>
                XMath.Dot(tape.Constant(h), CholeskyDecompositionNode.Compute(m[0].T).Solve(v[0])), 1e-7);
        }

        // 因子 L のノードを係数・行列積・転置に使う（L の勾配は三角求解・行列積から届き、上三角部分の随伴は使わない）。
        AssertGradients([a], [b], (tape, m, v) =>
        {
            var l = CholeskyDecompositionNode.Compute(m[0]).L;
            var z = XMath.SolveLowerTriangular(l, v[0]);
            return XMath.Dot(tape.Constant(h), XMath.SolveUpperTriangular(l.T, z)) + XMath.Dot(tape.Constant(h), l * v[0])
                + XMath.Dot(tape.Constant(h), l.T * z);
        }, n > 10 ? 1e-6 : 1e-7);
    }

    [Theory]
    [InlineData(129)]
    [InlineData(140)]
    [InlineData(200)]
    public void BlockedCholeskyGradientsMatchDirectionalDifferences(int n)
    {
        // 3つ以上のブロックに分かれる次数では、要素ごとの中心差分の代わりに無作為な方向の微分と比べる。
        var random = new Random(n + 600);
        var a = SymmetricPositiveDefinite(random, n, garbleUpper: false);
        var gl = RandomMatrix(random, n, n);
        var g = RandomMatrix(random, n, 2);
        var bs = RandomMatrix(random, n, 2);
        var x = RandomVector(random, n);
        ScalarNode Build(Tape tape, MatrixNode<double> m)
        {
            var cholesky = CholeskyDecompositionNode.Compute(m);
            return Pair(tape, cholesky.L, gl) + cholesky.LogDeterminant + Pair(tape, cholesky.Solve(tape.Constant(bs)), g)
                + cholesky.InverseQuadraticForm(tape.Constant(x));
        }

        AssertDirectionalDerivatives(a, Build, random, 1e-7);
        AssertDirectionalDerivatives(a, (tape, m) => Pair(tape, CholeskyDecompositionNode.Compute(m).L, gl), random, 1e-7);
        AssertDirectionalDerivatives(a, (tape, m) => Pair(tape, XMath.Solve(m, tape.Constant(bs)), g), random, 1e-7);
        AssertDirectionalDerivatives(a, (tape, m) => LuDecompositionNode.Compute(m).LogAbsDeterminant * 1.0, random, 1e-7);
    }

    [Theory]
    [InlineData(5)]
    [InlineData(6)]
    public void GradientsOfIllConditionedMatricesMatchCentralDifferences(int n)
    {
        // Hilbert 行列（条件数は n = 5 で約 5e5、n = 6 で約 1.5e7）。刻みを小さくして、刻みの中で正定値のままにする。
        var a = Hilbert(n);
        var random = new Random(n + 700);
        var b = RandomVector(random, n);
        var g = RandomMatrix(random, n, n);
        var h = RandomVector(random, n);
        double step = Math.ScaleB(1, -34);
        AssertGradients([a], [b], (tape, m, v) =>
        {
            var cholesky = CholeskyDecompositionNode.Compute(m[0]);
            return Pair(tape, cholesky.L, g) + cholesky.LogDeterminant + XMath.Dot(tape.Constant(h), cholesky.Solve(v[0]))
                + cholesky.InverseQuadraticForm(v[0]);
        }, 1e-5, step);
        AssertGradients([a], [b], (tape, m, v) =>
        {
            var lu = LuDecompositionNode.Compute(m[0]);
            return XMath.Dot(tape.Constant(h), XMath.Solve(m[0], v[0])) + XMath.Dot(tape.Constant(h), lu.SolveTransposed(v[0])) + lu.LogAbsDeterminant;
        }, 1e-5, step);
    }

    [Fact]
    public void DecompositionsOfTheSameOrderKeepSeparateFactors()
    {
        // 同じ次数の分解を2つ交互に使い、同じ入力を分解と行列積の両方に使う。
        var random = new Random(800);
        const int N = 7;
        var a1 = SymmetricPositiveDefinite(random, N, garbleUpper: false);
        var a2 = SymmetricPositiveDefinite(random, N, garbleUpper: false);
        var b = RandomVector(random, N);
        var g = RandomVector(random, N);
        ScalarNode Build(Tape tape, MatrixNode<double>[] m, VectorNode<double>[] v)
        {
            var first = CholeskyDecompositionNode.Compute(m[0]);
            var second = CholeskyDecompositionNode.Compute(m[1]);
            var lu1 = LuDecompositionNode.Compute(m[0]);
            var lu2 = LuDecompositionNode.Compute(m[1]);
            var x1 = first.Solve(v[0]);
            var x2 = second.Solve(x1);
            var x3 = first.Solve(m[1] * x2);
            var y = lu2.Solve(lu1.SolveTransposed(x3));
            return XMath.Dot(tape.Constant(g), y) + first.LogDeterminant - second.LogDeterminant + lu1.LogAbsDeterminant
                + second.InverseQuadraticForm(x3) + first.InverseQuadraticForm(v[0]);
        }

        AssertGradients([a1, a2], [b], Build, 1e-7);

        // テープを使い回しても、前の記録の分解は値に影響しない（同じ次数の分解を前の記録と逆の順に貸す）。
        var tape = new Tape();
        string expected = Fingerprint(tape, a1, a2, b, Build);
        for (int i = 0; i < 3; i++)
        {
            Fingerprint(tape, a2, a1, b, Build);
            Assert.Equal(expected, Fingerprint(tape, a1, a2, b, Build));
        }

        Assert.Equal(expected, Fingerprint(new Tape(), a1, a2, b, Build));
    }

    [Fact]
    public void FailedCholeskyRecordsNothingAndTheTapeRecovers()
    {
        var random = new Random(900);
        var spd = SymmetricPositiveDefinite(random, 6, garbleUpper: false);
        var indefinite = spd.Clone();
        indefinite[3, 3] = -1;
        var b = RandomVector(random, 6);
        var tape = new Tape();
        var good = tape.Leaf(spd);
        var bad = tape.Leaf(indefinite);
        Assert.False(CholeskyDecompositionNode.TryCompute(bad, out var failed));
        Assert.Throws<InvalidOperationException>(() => _ = failed.Order);
        DifferentiationAssert.Throws<ArgumentException>(() => CholeskyDecompositionNode.Compute(bad), "Compute of an indefinite matrix");
        // 失敗の後も同じ記録で分解でき、勾配は失敗がなかった場合と同じビット。
        var cholesky = CholeskyDecompositionNode.Compute(good);
        var loss = cholesky.InverseQuadraticForm(tape.Constant(b)) + cholesky.LogDeterminant;
        tape.Backward(loss);
        var reference = new Tape();
        var referenceLeaf = reference.Leaf(spd);
        var referenceCholesky = CholeskyDecompositionNode.Compute(referenceLeaf);
        reference.Backward(referenceCholesky.InverseQuadraticForm(reference.Constant(b)) + referenceCholesky.LogDeterminant);
        AssertBits(reference.GetGradient(referenceLeaf).ToMatrix().AsSpan(), tape.GetGradient(good).ToMatrix().AsSpan());
        Assert.Equal(0, XMath.NormInfinity(tape.GetGradient(bad).ToMatrix().AsVector()));

        tape.Reset();
        Assert.Throws<InvalidOperationException>(() => _ = cholesky.Order);
        var again = tape.Leaf(indefinite);
        Assert.False(CholeskyDecompositionNode.TryCompute(again, out _));
        tape.Reset();
        var fixedLeaf = tape.Leaf(spd);
        Assert.True(CholeskyDecompositionNode.TryCompute(fixedLeaf, out var recovered));
        tape.Backward(recovered.InverseQuadraticForm(tape.Constant(b)) + recovered.LogDeterminant);
        AssertBits(reference.GetGradient(referenceLeaf).ToMatrix().AsSpan(), tape.GetGradient(fixedLeaf).ToMatrix().AsSpan());
    }

    [Theory]
    [InlineData(1, 1)]
    [InlineData(6, 1)]
    [InlineData(6, 3)]
    public void BroadcastTriangularCoefficientsMatchXMathAndCentralDifferences(int n, int k)
    {
        // ブロードキャストの係数 L[i, j] = v[i]（テープの一時領域に写して解く）。値はコアが同じビューで解いた値と同じビット。
        var random = new Random(n * 11 + k);
        var v = RandomVector(random, n);
        for (int i = 0; i < n; i++)
        {
            v[i] = 2 + v[i];
        }

        var bs = RandomMatrix(random, n, k);
        var b = RandomVector(random, n);
        var g = RandomMatrix(random, n, k);
        var h = RandomVector(random, n);
        // コアは BLAS の配置に合わないビューを連続した行列に写して解くので、写した行列（とその転置）で解いた値と比べる。
        var explicitMatrix = new Matrix<double>(n, n);
        var explicitTransposed = new Matrix<double>(n, n);
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < n; i++)
            {
                explicitMatrix[i, j] = v[i];
                explicitTransposed[i, j] = v[j];
            }
        }

        var tape = new Tape();
        var coefficient = tape.Leaf(v).Broadcast(n);
        AssertBits(XMath.SolveLowerTriangular(explicitMatrix, b).AsSpan(), XMath.SolveLowerTriangular(coefficient, tape.Constant(b)).Value.ToVector().AsSpan());
        AssertBits(XMath.SolveUpperTriangular(explicitTransposed, b).AsSpan(), XMath.SolveUpperTriangular(coefficient.T, tape.Leaf(b)).Value.ToVector().AsSpan());
        AssertBits(XMath.SolveLowerTriangular(explicitMatrix, bs).AsSpan(), XMath.SolveLowerTriangular(coefficient, tape.Leaf(bs)).Value.ToMatrix().AsSpan());
        AssertBits(XMath.SolveUpperTriangular(explicitTransposed, bs).AsSpan(), XMath.SolveUpperTriangular(coefficient.T, tape.Constant(bs)).Value.ToMatrix().AsSpan());
        AssertGradients([bs], [v, b], (t, m, w) =>
            Pair(t, XMath.SolveLowerTriangular(w[0].Broadcast(n), m[0]), g) + XMath.Dot(t.Constant(h), XMath.SolveUpperTriangular(w[0].Broadcast(n).T, w[1]))
                + XMath.Dot(t.Constant(h), XMath.SolveLowerTriangular(w[0].Broadcast(n), w[1])) + Pair(t, XMath.SolveUpperTriangular(w[0].Broadcast(n).T, m[0]), g),
            1e-7);
    }

    [Fact]
    public void LargeAdjointsKeepFiniteGradientsFinite()
    {
        // 随伴が大きくても、最後の勾配が有限なら有限にする（随伴を先に掛け、2 倍は後にする）。
        var tape = new Tape();
        var identity = new Matrix<double>(2, 2);
        identity[0, 0] = 1;
        identity[1, 1] = 1;
        var a = tape.Leaf(identity);
        var x = tape.Leaf(new LinearAlgebra.Vector<double>([1e-100, 0.0]));
        var cholesky = CholeskyDecompositionNode.Compute(a);
        tape.Backward(1e308 * cholesky.InverseQuadraticForm(x));
        // dx = 2dq·A⁻¹x、dA = −dq·Φ₂(wwᵀ)（w = A⁻¹x = x）。
        Assert.Equal([2 * (1e308 * 1e-100), 0.0], tape.GetGradient(x).ToVector().AsSpan().ToArray());
        Assert.Equal([1e-100 * 1e-100 * -1e308, 0.0, 0.0, 0.0], tape.GetGradient(a).ToMatrix().AsSpan().ToArray());

        tape.Reset();
        a = tape.Leaf(identity);
        tape.Backward(1e308 * CholeskyDecompositionNode.Compute(a).LogDeterminant);
        Assert.Equal([1e308, 0.0, 0.0, 1e308], tape.GetGradient(a).ToMatrix().AsSpan().ToArray());

        // 1 次の L = √A（A = 0.25）の勾配 1/(2√A) = 1 に 1e308 を掛ける。対角の S + Sᵀ を経るとオーバーフローする。
        tape.Reset();
        var quarter = new Matrix<double>(1, 1);
        quarter[0, 0] = 0.25;
        var b = tape.Leaf(quarter);
        tape.Backward(1e308 * XMath.Sum(XMath.ColumnSums(CholeskyDecompositionNode.Compute(b).L)));
        Assert.Equal([1e308], tape.GetGradient(b).ToMatrix().AsSpan().ToArray());
    }

    [Fact]
    public void SingularMatricesFollowTheCoreContracts()
    {
        var singular = new Matrix<double>(3, 3);
        singular[0, 0] = 1;
        singular[1, 1] = 2;
        singular[2, 0] = 1;   // 3列目が 0
        var b = new LinearAlgebra.Vector<double>([1.0, 2.0, 3.0]);
        var tape = new Tape();
        var a = tape.Leaf(singular);
        DifferentiationAssert.Throws<ArgumentException>(() => XMath.Solve(a, tape.Constant(b)), "XMath.Solve of a singular matrix");
        var lu = LuDecompositionNode.Compute(a);
        Assert.True(lu.IsSingular);
        Assert.Equal(0, lu.DeterminantSign);
        Assert.Equal(double.NegativeInfinity, lu.LogAbsDeterminant.Value);
        DifferentiationAssert.Throws<InvalidOperationException>(() => lu.Solve(tape.Constant(b)), "Solve with a singular decomposition");
        DifferentiationAssert.Throws<InvalidOperationException>(() => lu.SolveTransposed(tape.Constant(b)), "SolveTransposed with a singular decomposition");
        // 特異な行列の ln |det A| の勾配は NaN。
        tape.Backward(lu.LogAbsDeterminant * 1.0);
        Assert.All(tape.GetGradient(a).ToMatrix().AsSpan().ToArray(), x => Assert.True(double.IsNaN(x)));
    }

    [Fact]
    public void InvalidArgumentsAreRejected()
    {
        var tape = new Tape();
        var rectangular = tape.Leaf(new Matrix<double>(3, 2));
        var square = tape.Leaf(SymmetricPositiveDefinite(new Random(1), 3, garbleUpper: false));
        var shortVector = tape.Constant(new LinearAlgebra.Vector<double>(2));
        var shortMatrix = tape.Constant(new Matrix<double>(2, 2));
        var singular = tape.Leaf(new Matrix<double>(3, 3));
        DifferentiationAssert.Throws<ArgumentException>(() => XMath.Solve(rectangular, shortVector), "Solve of a rectangular matrix");
        DifferentiationAssert.Throws<ArgumentException>(() => XMath.Solve(square, shortVector), "Solve with a short right-hand side");
        DifferentiationAssert.Throws<ArgumentException>(() => XMath.Solve(square, shortMatrix), "Solve with a short right-hand matrix");
        DifferentiationAssert.Throws<ArgumentException>(() => XMath.SolveLowerTriangular(rectangular, shortVector), "SolveLowerTriangular of a rectangular matrix");
        DifferentiationAssert.Throws<ArgumentException>(() => XMath.SolveUpperTriangular(square, shortMatrix), "SolveUpperTriangular with a short right-hand side");
        DifferentiationAssert.Throws<ArgumentException>(() => LuDecompositionNode.Compute(rectangular), "LU of a rectangular matrix");
        DifferentiationAssert.Throws<ArgumentException>(() => CholeskyDecompositionNode.Compute(rectangular), "Cholesky of a rectangular matrix");
        var lu = LuDecompositionNode.Compute(square);
        var cholesky = CholeskyDecompositionNode.Compute(square);
        DifferentiationAssert.Throws<ArgumentException>(() => lu.Solve(shortVector), "LU solve with a short right-hand side");
        DifferentiationAssert.Throws<ArgumentException>(() => lu.SolveTransposed(shortMatrix), "LU transposed solve with a short right-hand side");
        DifferentiationAssert.Throws<ArgumentException>(() => cholesky.Solve(shortMatrix), "Cholesky solve with a short right-hand side");
        DifferentiationAssert.Throws<ArgumentException>(() => cholesky.InverseQuadraticForm(shortVector), "quadratic form with a short vector");

        // 初期化していないノード、別のテープの右辺、古い世代、逆伝播を終えたテープへの記録。
        DifferentiationAssert.Throws<InvalidOperationException>(() => _ = default(LuDecompositionNode).Order, "an uninitialized LU node");
        DifferentiationAssert.Throws<InvalidOperationException>(() => default(CholeskyDecompositionNode).Solve(shortVector), "an uninitialized Cholesky node");
        var other = new Tape();
        var foreign = other.Constant(new LinearAlgebra.Vector<double>(3));
        DifferentiationAssert.Throws<InvalidOperationException>(() => lu.Solve(foreign), "a right-hand side of another tape");
        DifferentiationAssert.Throws<InvalidOperationException>(() => cholesky.Solve(foreign), "a right-hand side of another tape");
        DifferentiationAssert.Throws<InvalidOperationException>(() => XMath.Solve(square, foreign), "a right-hand side of another tape");
        var b = tape.Constant(new LinearAlgebra.Vector<double>([1.0, 2.0, 3.0]));
        tape.Backward(cholesky.InverseQuadraticForm(b));
        DifferentiationAssert.Throws<InvalidOperationException>(() => cholesky.Solve(b), "recording after the backward pass");
        DifferentiationAssert.Throws<InvalidOperationException>(() => LuDecompositionNode.Compute(square), "recording after the backward pass");
        DifferentiationAssert.Throws<InvalidOperationException>(() => XMath.Solve(square, b), "recording after the backward pass");
        // 特異な係数でも、状態の検査が分解より先（借りた分解が返らない経路を作らない）。
        DifferentiationAssert.Throws<InvalidOperationException>(() => XMath.Solve(singular, b), "recording after the backward pass");
        tape.Reset();
        DifferentiationAssert.Throws<InvalidOperationException>(() => _ = lu.LogAbsDeterminant, "a decomposition before Reset");
        DifferentiationAssert.Throws<InvalidOperationException>(() => _ = cholesky.L, "a decomposition before Reset");
    }

    [Fact]
    public void ConstantInputsRecordNothing()
    {
        var random = new Random(1000);
        var a = SymmetricPositiveDefinite(random, 5, garbleUpper: false);
        var b = RandomVector(random, 5);
        var tape = new Tape();
        var constant = tape.Constant(a);
        var cholesky = CholeskyDecompositionNode.Compute(constant);
        var lu = LuDecompositionNode.Compute(constant);
        Assert.False(cholesky.L.RequiresGradient);
        Assert.False(cholesky.LogDeterminant.RequiresGradient);
        Assert.False(lu.LogAbsDeterminant.RequiresGradient);
        Assert.False(cholesky.Solve(tape.Constant(b)).RequiresGradient);
        Assert.False(XMath.Solve(constant, tape.Constant(b)).RequiresGradient);
        Assert.False(cholesky.InverseQuadraticForm(tape.Constant(b)).RequiresGradient);
        // 右辺だけが勾配を要る場合も解ける（係数の勾配は求めない）。
        var leaf = tape.Leaf(b);
        var loss = cholesky.InverseQuadraticForm(leaf) + XMath.Sum(lu.Solve(leaf)) + XMath.Sum(XMath.Solve(constant, leaf));
        tape.Backward(loss);
        var expected = XMath.Add(XMath.Multiply(CholeskyDecomposition.Compute(a).Solve(b), 2.0), XMath.Multiply(LuDecomposition.Compute(a).SolveTransposed(Ones(5)), 2.0));
        var actual = tape.GetGradient(leaf).ToVector();
        for (int i = 0; i < 5; i++)
        {
            DifferentiationAssert.Close(expected[i], actual[i], 1e-13, 1e-13, $"db[{i}]");
        }
    }

    // build を葉で記録して逆伝播した勾配を、格納の各要素の中心差分（Richardson 補外）と比べる。
    // 中心差分は同じ build を定数で記録した値で求める（値はコアとビット一致することを別に確かめている）。
    internal static void AssertGradients(
        Matrix<double>[] matrices, LinearAlgebra.Vector<double>[] vectors,
        Func<Tape, MatrixNode<double>[], VectorNode<double>[], ScalarNode> build, double relative, double? step = null)
    {
        var tape = new Tape();
        var matrixLeaves = Array.ConvertAll(matrices, m => tape.Leaf(m));
        var vectorLeaves = Array.ConvertAll(vectors, v => tape.Leaf(v));
        tape.Backward(build(tape, matrixLeaves, vectorLeaves));
        var matrixGradients = Array.ConvertAll(matrixLeaves, l => tape.GetGradient(l).ToMatrix());
        var vectorGradients = Array.ConvertAll(vectorLeaves, l => tape.GetGradient(l).ToVector());
        double scale = 1e-300;
        foreach (var gradient in matrixGradients)
        {
            scale = Math.Max(scale, XMath.NormInfinity(gradient.AsVector()));
        }

        foreach (var gradient in vectorGradients)
        {
            scale = Math.Max(scale, XMath.NormInfinity(gradient));
        }

        var evaluation = new Tape();
        double Value()
        {
            evaluation.Reset();
            return build(evaluation, Array.ConvertAll(matrices, m => evaluation.Constant(m)), Array.ConvertAll(vectors, v => evaluation.Constant(v))).Value;
        }

        for (int m = 0; m < matrices.Length; m++)
        {
            var matrix = matrices[m];
            for (int j = 0; j < matrix.Columns; j++)
            {
                for (int i = 0; i < matrix.Rows; i++)
                {
                    double saved = matrix[i, j];
                    double numeric = DifferentiationAssert.Richardson(t =>
                    {
                        matrix[i, j] = t;
                        double value = Value();
                        matrix[i, j] = saved;
                        return value;
                    }, saved, step);
                    DifferentiationAssert.Close(numeric, matrixGradients[m][i, j], relative, relative * scale, $"matrix {m} [{i}, {j}]");
                }
            }
        }

        for (int v = 0; v < vectors.Length; v++)
        {
            var vector = vectors[v];
            for (int i = 0; i < vector.Length; i++)
            {
                double saved = vector[i];
                double numeric = DifferentiationAssert.Richardson(t =>
                {
                    vector[i] = t;
                    double value = Value();
                    vector[i] = saved;
                    return value;
                }, saved, step);
                DifferentiationAssert.Close(numeric, vectorGradients[v][i], relative, relative * scale, $"vector {v} [{i}]");
            }
        }
    }

    // 無作為な方向 E（3本）について、t ↦ f(A + tE) の t = 0 での中心差分（刻み step、既定は 2⁻¹⁰）を ⟨∇f, E⟩ と比べる。
    internal static void AssertDirectionalDerivatives(Matrix<double> a, Func<Tape, MatrixNode<double>, ScalarNode> build, Random random, double relative, double? step = null)
    {
        var tape = new Tape();
        var leaf = tape.Leaf(a);
        tape.Backward(build(tape, leaf));
        var gradient = tape.GetGradient(leaf).ToMatrix();
        var evaluation = new Tape();
        var shifted = new Matrix<double>(a.Rows, a.Columns);
        for (int d = 0; d < 3; d++)
        {
            var direction = RandomMatrix(random, a.Rows, a.Columns);
            double analytic = XMath.Dot(gradient.AsVector(), direction.AsVector());
            double numeric = DifferentiationAssert.Richardson(t =>
            {
                XMath.Add(a, XMath.Multiply(direction, t), shifted);
                evaluation.Reset();
                return build(evaluation, evaluation.Constant(shifted)).Value;
            }, 0, step ?? Math.ScaleB(1, -10));
            double size = XMath.Dot(XMath.Abs(gradient.AsVector()), XMath.Abs(direction.AsVector()));
            DifferentiationAssert.Close(analytic, numeric, relative, relative * size, $"direction {d}");
        }
    }

    // f = Σ G∘X
    internal static ScalarNode Pair(Tape tape, MatrixNode<double> x, Matrix<double> g) => XMath.Sum(XMath.ColumnSums(XMath.Hadamard(tape.Constant(g), x)));

    private static string Fingerprint(Tape tape, Matrix<double> first, Matrix<double> second, LinearAlgebra.Vector<double> b,
        Func<Tape, MatrixNode<double>[], VectorNode<double>[], ScalarNode> build)
    {
        tape.Reset();
        var leaves = new[] { tape.Leaf(first), tape.Leaf(second) };
        var vector = tape.Leaf(b);
        var loss = build(tape, leaves, [vector]);
        tape.Backward(loss);
        return BitConverter.DoubleToInt64Bits(loss.Value).ToString("X16") + ";"
            + Convert.ToHexString(System.Runtime.InteropServices.MemoryMarshal.AsBytes(tape.GetGradient(leaves[0]).ToMatrix().AsSpan())) + ";"
            + Convert.ToHexString(System.Runtime.InteropServices.MemoryMarshal.AsBytes(tape.GetGradient(leaves[1]).ToMatrix().AsSpan())) + ";"
            + Convert.ToHexString(System.Runtime.InteropServices.MemoryMarshal.AsBytes(tape.GetGradient(vector).ToVector().AsSpan()));
    }

    private static void AssertBits(ReadOnlySpan<double> expected, ReadOnlySpan<double> actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            DifferentiationAssert.EqualBits(expected[i], actual[i], $"[{i}]");
        }
    }

    internal static Matrix<double> RandomMatrix(Random random, int rows, int columns) => ArrayProductTests.RandomMatrix<double>(random, rows, columns);

    internal static LinearAlgebra.Vector<double> RandomVector(Random random, int length) => ArrayProductTests.RandomVector<double>(random, length);

    // A = MMᵀ/n + I（固有値は 1 以上）。garbleUpper なら、コアが読まない上三角部分を無関係な値にする。
    internal static Matrix<double> SymmetricPositiveDefinite(Random random, int n, bool garbleUpper)
    {
        var m = RandomMatrix(random, n, n);
        var a = XMath.Multiply(m, m.T);
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < n; i++)
            {
                a[i, j] = a[i, j] / n + (i == j ? 1 : 0);
            }
        }

        if (garbleUpper)
        {
            for (int j = 1; j < n; j++)
            {
                for (int i = 0; i < j; i++)
                {
                    a[i, j] = 3 * random.NextDouble() - 1.5;
                }
            }
        }

        return a;
    }

    // 下三角・上三角のどちらとして読んでも条件のよい正方行列（対角を大きくし、対角の外を小さくする）。
    private static Matrix<double> WellConditionedTriangles(Random random, int n)
    {
        var a = RandomMatrix(random, n, n);
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < n; i++)
            {
                a[i, j] = i == j ? 2 + a[i, j] : a[i, j] / Math.Max(1, n / 4.0);
            }
        }

        return a;
    }

    private static Matrix<double> Hilbert(int n)
    {
        var a = new Matrix<double>(n, n);
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < n; i++)
            {
                a[i, j] = 1.0 / (i + j + 1);
            }
        }

        return a;
    }

    private static void Shift(Matrix<double> a, double value, int rowOffset = 0, int columnOffset = 0, int? order = null)
    {
        int n = order ?? Math.Min(a.Rows, a.Columns);
        for (int i = 0; i < n; i++)
        {
            a[rowOffset + i, columnOffset + i] += value;
        }
    }

    private static LinearAlgebra.Vector<double> Ones(int n)
    {
        var v = new LinearAlgebra.Vector<double>(n);
        v.AsSpan().Fill(1);
        return v;
    }
}
