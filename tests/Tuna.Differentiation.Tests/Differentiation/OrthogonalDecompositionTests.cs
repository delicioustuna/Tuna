using Tuna.Differentiation;
using Tuna.LinearAlgebra;
using static Tuna.Tests.Differentiation.LinearAlgebraTests;

namespace Tuna.Tests.Differentiation;

/// <summary>QR 分解・対称固有値分解・特異値分解の記録の値（コアとのビット一致）と勾配（中心差分）</summary>
/// <remarks>
/// 固有ベクトル・特異ベクトルは符号（と重なった値の空間の基底）をコアが選ぶので、勾配は符号によらない量（要素の2乗、UVᵀ、射影）で確かめる。
/// QR 分解はハウスホルダー変換の符号の選び方が入力の近くで変わらないので、Q と R をそのまま使う。
/// </remarks>
public class OrthogonalDecompositionTests
{
    // 行数・列数の組。正方・縦長・横長と、縦長の特異値分解が先に QR 分解する比（m ≥ 1.6n）を含める。
    public static TheoryData<int, int> Shapes() => new()
    {
        { 1, 1 }, { 3, 1 }, { 1, 3 }, { 5, 5 }, { 7, 4 }, { 4, 7 }, { 10, 4 }, { 4, 10 }, { 33, 33 }, { 70, 40 }, { 40, 70 },
    };

    // 対称固有値分解の次数。n > 25 は分割統治法の経路に入る。
    public static TheoryData<int> Orders() => [1, 2, 5, 25, 26, 70];

    [Theory]
    [MemberData(nameof(Shapes))]
    public void QrValuesMatchQrDecompositionBitForBit(int m, int n)
    {
        var random = new Random(m * 100 + n);
        var storage = RandomMatrix(random, m + 2, n + 3);
        foreach (bool transposed in new[] { false, true })
        {
            var tape = new Tape();
            // 部分行列（列の間隔が行数と違う）か、その転置のビュー。
            var source = transposed ? storage.T.ToMatrix() : storage;
            ReadOnlyMatrixSpan<double> view = transposed
                ? ((ReadOnlyMatrixSpan<double>)source)[2..(n + 2), 1..(m + 1)].T
                : ((ReadOnlyMatrixSpan<double>)source)[1..(m + 1), 2..(n + 2)];
            var expected = QrDecomposition.Compute(view.ToMatrix());
            foreach (var leaf in new[] { tape.Leaf(source), tape.Constant(source) })
            {
                var a = transposed ? leaf[2..(n + 2), 1..(m + 1)].T : leaf[1..(m + 1), 2..(n + 2)];
                var qr = QrDecompositionNode.Compute(a);
                Assert.Equal(m, qr.Rows);
                Assert.Equal(n, qr.Columns);
                AssertBits(expected.Q.AsSpan(), qr.Q.Value.ToMatrix().AsSpan());
                AssertBits(expected.R.AsSpan(), qr.R.Value.ToMatrix().AsSpan());
            }
        }
    }

    [Theory]
    [MemberData(nameof(Orders))]
    public void SymmetricEigenValuesMatchSymmetricEigenDecompositionBitForBit(int n)
    {
        var random = new Random(n + 1100);
        var storage = SymmetricMatrix(random, n, garbleUpper: true);
        var expected = SymmetricEigenDecomposition.Compute(storage);
        // 転置のビューでは、格納の上三角部分が論理的な下三角部分になる。
        var transposedStorage = storage.T.ToMatrix();
        foreach (bool transposed in new[] { false, true })
        {
            var tape = new Tape();
            var stored = transposed ? transposedStorage : storage;
            foreach (var source in new[] { tape.Leaf(stored), tape.Constant(stored) })
            {
                var eigen = SymmetricEigenDecompositionNode.Compute(transposed ? source.T : source);
                Assert.Equal(n, eigen.Order);
                AssertBits(expected.Values, eigen.Values.Value.ToVector().AsSpan());
                AssertBits(expected.Vectors!.AsSpan(), eigen.Vectors.Value.ToMatrix().AsSpan());
            }
        }
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void SingularValuesMatchSingularValueDecompositionBitForBit(int m, int n)
    {
        var random = new Random(m * 100 + n + 1200);
        var storage = RandomMatrix(random, m + 1, n + 2);
        var tape = new Tape();
        foreach (bool transposed in new[] { false, true })
        {
            var source = transposed ? storage.T.ToMatrix() : storage;
            ReadOnlyMatrixSpan<double> view = transposed
                ? ((ReadOnlyMatrixSpan<double>)source)[2..(n + 2), 1..(m + 1)].T
                : ((ReadOnlyMatrixSpan<double>)source)[1..(m + 1), 2..(n + 2)];
            var expected = SingularValueDecomposition.Compute(view.ToMatrix());
            foreach (var leaf in new[] { tape.Leaf(source), tape.Constant(source) })
            {
                var a = transposed ? leaf[2..(n + 2), 1..(m + 1)].T : leaf[1..(m + 1), 2..(n + 2)];
                var svd = SingularValueDecompositionNode.Compute(a);
                Assert.Equal(m, svd.Rows);
                Assert.Equal(n, svd.Columns);
                AssertBits(expected.Values, svd.Values.Value.ToVector().AsSpan());
                AssertBits(expected.U!.AsSpan(), svd.U.Value.ToMatrix().AsSpan());
                AssertBits(expected.V!.AsSpan(), svd.V.Value.ToMatrix().AsSpan());
            }
        }
    }

    [Theory]
    [InlineData(1, 1)]
    [InlineData(4, 4)]
    [InlineData(6, 3)]
    [InlineData(3, 6)]
    [InlineData(9, 2)]
    [InlineData(2, 9)]
    [InlineData(1, 4)]
    [InlineData(4, 1)]
    [InlineData(12, 9)]
    public void QrGradientsMatchCentralDifferences(int m, int n)
    {
        var random = new Random(m * 10 + n + 1300);
        var a = RandomMatrix(random, m, n);
        int k = Math.Min(m, n);
        var gq = RandomMatrix(random, m, k);
        var gr = RandomMatrix(random, k, n);
        var gp = RandomMatrix(random, m, n);
        var h = RandomVector(random, k);
        AssertGradients([a], [], (tape, x, v) =>
        {
            var qr = QrDecompositionNode.Compute(x[0]);
            return Pair(tape, qr.Q, gq) + Pair(tape, qr.R, gr);
        }, 1e-7);
        // 片方の結果だけ（届かなかった結果の随伴を 0 とする経路）と、Q・R を行列積・転置・ベクトルとの積に使う場合。
        AssertGradients([a], [], (tape, x, v) => Pair(tape, QrDecompositionNode.Compute(x[0]).Q, gq), 1e-7);
        AssertGradients([a], [], (tape, x, v) => Pair(tape, QrDecompositionNode.Compute(x[0]).R, gr), 1e-7);
        AssertGradients([a], [], (tape, x, v) =>
        {
            var qr = QrDecompositionNode.Compute(x[0]);
            return Pair(tape, qr.Q * qr.R, gp) + XMath.Dot(tape.Constant(h), qr.Q.T * XMath.ColumnSums(x[0].T)) + XMath.Sum(XMath.ColumnSums(qr.R));
        }, 1e-7);
        // 転置と部分行列のビュー。勾配は格納の要素に届く。
        var storage = RandomMatrix(random, n + 2, m + 1);
        AssertGradients([storage], [], (tape, x, v) =>
        {
            var qr = QrDecompositionNode.Compute(x[0][1..(n + 1), 1..].T);
            return Pair(tape, qr.Q, gq) + Pair(tape, qr.R, gr);
        }, 1e-7);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(5)]
    [InlineData(27)]
    public void SymmetricEigenGradientsMatchCentralDifferences(int n)
    {
        var random = new Random(n + 1400);
        // 上三角部分は読まないので値を乱しておく（中心差分は 0、勾配も 0）。
        var a = SymmetricMatrix(random, n, garbleUpper: true);
        var gv = RandomMatrix(random, n, n);
        var c = RandomVector(random, n);
        var b = RandomVector(random, n);
        double relative = n > 10 ? 1e-6 : 1e-7;
        // 固有ベクトルは要素の2乗（列の符号によらない）と、bᵀvᵢ の2乗の重み付きの和で使う。
        AssertGradients([a], [], (tape, x, v) =>
        {
            var eigen = SymmetricEigenDecompositionNode.Compute(x[0]);
            var vectors = eigen.Vectors;
            var projection = vectors.T * tape.Constant(b);
            return XMath.Dot(tape.Constant(c), eigen.Values) + Pair(tape, XMath.Hadamard(vectors, vectors), gv)
                + XMath.Dot(tape.Constant(c), XMath.Hadamard(projection, projection));
        }, relative);
        AssertGradients([a], [], (tape, x, v) => XMath.Dot(tape.Constant(c), SymmetricEigenDecompositionNode.Compute(x[0]).Values), relative);
        AssertGradients([a], [], (tape, x, v) =>
        {
            var vectors = SymmetricEigenDecompositionNode.Compute(x[0]).Vectors;
            return Pair(tape, XMath.Hadamard(vectors, vectors), gv);
        }, relative);
        // 転置のビュー（格納の上三角部分を読む）。
        AssertGradients([a.T.ToMatrix()], [], (tape, x, v) =>
        {
            var eigen = SymmetricEigenDecompositionNode.Compute(x[0].T);
            return XMath.Dot(tape.Constant(c), eigen.Values) + Pair(tape, XMath.Hadamard(eigen.Vectors, eigen.Vectors), gv);
        }, relative);
    }

    [Theory]
    [InlineData(1, 1)]
    [InlineData(3, 1)]
    [InlineData(1, 3)]
    [InlineData(5, 5)]
    [InlineData(7, 4)]
    [InlineData(4, 7)]
    [InlineData(10, 4)]
    [InlineData(4, 10)]
    [InlineData(14, 11)]
    public void SvdGradientsMatchCentralDifferences(int m, int n)
    {
        var random = new Random(m * 10 + n + 1500);
        var a = RandomMatrix(random, m, n);
        int k = Math.Min(m, n);
        var gu = RandomMatrix(random, m, k);
        var gv = RandomMatrix(random, n, k);
        var gp = RandomMatrix(random, m, n);
        var c = RandomVector(random, k);
        double relative = m * n > 100 ? 1e-6 : 1e-7;
        // 特異ベクトルは要素の2乗と UVᵀ（uᵢ と vᵢ を一緒に反転しても変わらない）で使う。
        AssertGradients([a], [], (tape, x, v) =>
        {
            var svd = SingularValueDecompositionNode.Compute(x[0]);
            var (u, w) = (svd.U, svd.V);
            return XMath.Dot(tape.Constant(c), svd.Values) + Pair(tape, XMath.Hadamard(u, u), gu) + Pair(tape, XMath.Hadamard(w, w), gv)
                + Pair(tape, u * w.T, gp);
        }, relative);
        // 1つの結果だけ（m ≠ n では射影の項の片方だけを使う）。
        AssertGradients([a], [], (tape, x, v) => XMath.Dot(tape.Constant(c), SingularValueDecompositionNode.Compute(x[0]).Values), relative);
        AssertGradients([a], [], (tape, x, v) =>
        {
            var u = SingularValueDecompositionNode.Compute(x[0]).U;
            return Pair(tape, XMath.Hadamard(u, u), gu);
        }, relative);
        AssertGradients([a], [], (tape, x, v) =>
        {
            var w = SingularValueDecompositionNode.Compute(x[0]).V;
            return Pair(tape, XMath.Hadamard(w, w), gv);
        }, relative);
        // 転置と部分行列のビュー。
        var storage = RandomMatrix(random, n + 1, m + 2);
        AssertGradients([storage], [], (tape, x, v) =>
        {
            var svd = SingularValueDecompositionNode.Compute(x[0][1.., 2..].T);
            return XMath.Dot(tape.Constant(c), svd.Values) + Pair(tape, svd.U * svd.V.T, gp);
        }, relative);
    }

    [Theory]
    [InlineData(130, 100)]
    [InlineData(100, 130)]
    [InlineData(200, 70)]
    public void LargeDecompositionGradientsMatchDirectionalDifferences(int m, int n)
    {
        // ブロック化した経路に入る大きさでは、要素ごとの中心差分の代わりに無作為な方向の微分と比べる。
        // 特異値・固有値の間隔は次数とともに狭くなるので、刻みを小さくして、刻みの中で値の並びが入れ替わらないようにする。
        double step = Math.ScaleB(1, -24);
        var random = new Random(m + n + 1600);
        var a = RandomMatrix(random, m, n);
        int k = Math.Min(m, n);
        var gq = RandomMatrix(random, m, k);
        var gr = RandomMatrix(random, k, n);
        var gu = RandomMatrix(random, m, k);
        var gp = RandomMatrix(random, m, n);
        var c = RandomVector(random, k);
        AssertDirectionalDerivatives(a, (tape, x) =>
        {
            var qr = QrDecompositionNode.Compute(x);
            return Pair(tape, qr.Q, gq) + Pair(tape, qr.R, gr);
        }, random, 1e-6, step);
        AssertDirectionalDerivatives(a, (tape, x) =>
        {
            var svd = SingularValueDecompositionNode.Compute(x);
            return XMath.Dot(tape.Constant(c), svd.Values) + Pair(tape, XMath.Hadamard(svd.U, svd.U), gu) + Pair(tape, svd.U * svd.V.T, gp);
        }, random, 1e-6, step);
        var symmetric = SymmetricMatrix(random, m, garbleUpper: true);
        var gv = RandomMatrix(random, m, m);
        var weights = RandomVector(random, m);
        AssertDirectionalDerivatives(symmetric, (tape, x) =>
        {
            var eigen = SymmetricEigenDecompositionNode.Compute(x);
            return XMath.Dot(tape.Constant(weights), eigen.Values) + Pair(tape, XMath.Hadamard(eigen.Vectors, eigen.Vectors), gv);
        }, random, 1e-6, step);
    }

    [Theory]
    [InlineData(5, 5)]
    [InlineData(7, 4)]
    [InlineData(4, 7)]
    public void GradientsOfIllConditionedMatricesMatchCentralDifferences(int m, int n)
    {
        // Hilbert 行列とその長方形版（5 次の条件数は約 5e5）。QR 分解の逆伝播は R⁻ᵀ を、固有ベクトル・特異ベクトルの逆伝播は値の差の逆数を含む。
        // 刻みは最小の特異値・固有値の間隔より十分小さくし、刻みの中で値の並びが入れ替わらないようにする。
        var a = Hilbert(m, n);
        var random = new Random(m * 10 + n + 1650);
        int k = Math.Min(m, n);
        var gq = RandomMatrix(random, m, k);
        var gr = RandomMatrix(random, k, n);
        var gu = RandomMatrix(random, m, k);
        var gp = RandomMatrix(random, m, n);
        var c = RandomVector(random, k);
        double step = Math.ScaleB(1, -26);
        AssertGradients([a], [], (tape, x, v) =>
        {
            var qr = QrDecompositionNode.Compute(x[0]);
            return Pair(tape, qr.Q, gq) + Pair(tape, qr.R, gr);
        }, 1e-5, step);
        AssertGradients([a], [], (tape, x, v) =>
        {
            var svd = SingularValueDecompositionNode.Compute(x[0]);
            return XMath.Dot(tape.Constant(c), svd.Values) + Pair(tape, XMath.Hadamard(svd.U, svd.U), gu) + Pair(tape, svd.U * svd.V.T, gp);
        }, 1e-5, step);
        if (m == n)
        {
            var gv = RandomMatrix(random, n, n);
            AssertGradients([a], [], (tape, x, v) =>
            {
                var eigen = SymmetricEigenDecompositionNode.Compute(x[0]);
                return XMath.Dot(tape.Constant(c), eigen.Values) + Pair(tape, XMath.Hadamard(eigen.Vectors, eigen.Vectors), gv);
            }, 1e-5, step);
        }
    }

    [Fact]
    public void RepeatedEigenvaluesGiveFiniteGradientsOfTheValues()
    {
        // 単位行列の固有値は重なる。固有値だけを使う勾配は V·diag(λ̄)·Vᵀ で有限（ここでは単位行列の下三角部分）。
        var tape = new Tape();
        var a = tape.Leaf(Identity(4));
        var eigen = SymmetricEigenDecompositionNode.Compute(a);
        tape.Backward(XMath.Sum(eigen.Values));
        var gradient = tape.GetGradient(a).ToMatrix();
        for (int j = 0; j < 4; j++)
        {
            for (int i = 0; i < 4; i++)
            {
                DifferentiationAssert.Close(i == j ? 1 : 0, gradient[i, j], 1e-14, 1e-14, $"[{i}, {j}]");
            }
        }

        // 特異値も同じ（直交行列の特異値はすべて 1）。
        tape.Reset();
        var q = tape.Leaf(Identity(3));
        var svd = SingularValueDecompositionNode.Compute(q);
        tape.Backward(XMath.Sum(svd.Values));
        Assert.All(tape.GetGradient(q).ToMatrix().AsSpan().ToArray(), x => Assert.True(double.IsFinite(x)));
    }

    // A = diag(2s, s)、損失 c·(UVᵀ)[1, 0] の勾配は A[1, 0] が c/(3s)、A[0, 1] が −c/(3s)、対角が 0。
    // 逆伝播の (σⱼ − σᵢ)(σⱼ + σᵢ) は s = 1e-160 で非正規化数、1e-200 で 0、1e160・1e200 でオーバーフローする（#136 のレビュー）。
    // 随伴が極端なとき（c = 1.7e308、1e-180 など）は分子の積がオーバーフロー・アンダーフローする（2回目のレビュー）。
    [Theory]
    [InlineData(1e-200, 1.0)]
    [InlineData(1e-160, 1.0)]
    [InlineData(1.0, 1.0)]
    [InlineData(1e160, 1.0)]
    [InlineData(1e200, 1.0)]
    [InlineData(1e200, 1.7e308)]
    [InlineData(1.0, 1.7e308)]
    [InlineData(1e-150, 1e-180)]
    [InlineData(1e-200, 1e-200)]
    [InlineData(1e-200, 1e100)]
    public void SvdGradientsOfTinyAndHugeSingularValuesStayAccurate(double s, double c)
    {
        var tape = new Tape();
        var matrix = new Matrix<double>(2, 2);
        matrix[0, 0] = 2 * s;
        matrix[1, 1] = s;
        var a = tape.Leaf(matrix);
        var svd = SingularValueDecompositionNode.Compute(a);
        var pick = new Matrix<double>(2, 2);
        pick[1, 0] = c;
        tape.Backward(Pair(tape, svd.U * svd.V.T, pick));
        var gradient = tape.GetGradient(a).ToMatrix();
        double expected = c / (3 * s);
        DifferentiationAssert.Close(expected, gradient[1, 0], 1e-14, 0, "[1, 0]");
        DifferentiationAssert.Close(-expected, gradient[0, 1], 1e-14, 0, "[0, 1]");
        Assert.Equal(0, gradient[0, 0]);
        Assert.Equal(0, gradient[1, 1]);
    }

    // A = diag(a, b)（a > b > 0）、損失 c₁·U[0, 0]·U[1, 0] + c₂·V[0, 0]·V[1, 0] の勾配は、A[0, 1] が (c₁b + c₂a)/(a² − b²)、
    // A[1, 0] が (c₁a + c₂b)/(a² − b²)、対角は 0（期待値は mpmath で求めた）。随伴 d = Pᵢⱼ − Pⱼᵢ と e = Wᵢⱼ − Wⱼᵢ を
    // 大きい方の指数の 2 のべきでまとめて割ると、大きさの違う小さい方が 0 に丸まった（#136 のレビュー。2つ目と4つ目の組）。
    [Theory]
    [InlineData(2.0, 1.0, 1.0, 1.0, 1.0, 1.0)]
    [InlineData(1e155, 1e-200, 1e185, 1e-145, 9.999999999999999e-301, 1e30)]
    [InlineData(1e155, 1e-200, 1e-145, 1e185, 1e30, 9.999999999999999e-301)]
    [InlineData(1e200, 1e-150, 1e-100, 1e250, 9.999999999999999e49, 2e-300)]
    [InlineData(2e-200, 1e-200, 1e-300, 1e100, 6.666666666666667e299, 3.3333333333333335e299)]
    [InlineData(2.0, 1.0, 1e308, 1e308, 1e308, 1e308)]
    public void SvdGradientsOfAdjointsOfDifferentScalesStayAccurate(double a, double b, double c1, double c2, double expected01, double expected10)
    {
        var tape = new Tape();
        var matrix = new Matrix<double>(2, 2);
        matrix[0, 0] = a;
        matrix[1, 1] = b;
        var leaf = tape.Leaf(matrix);
        var svd = SingularValueDecompositionNode.Compute(leaf);
        var first = new LinearAlgebra.Vector<double>(2);
        first[0] = 1;
        var u = svd.U * tape.Constant(first);
        var v = svd.V * tape.Constant(first);
        tape.Backward(c1 * XMath.Dot(u[..1], u[1..]) + c2 * XMath.Dot(v[..1], v[1..]));
        var gradient = tape.GetGradient(leaf).ToMatrix();
        DifferentiationAssert.Close(expected01, gradient[0, 1], 1e-14, 0, "[0, 1]");
        DifferentiationAssert.Close(expected10, gradient[1, 0], 1e-14, 0, "[1, 0]");
        Assert.Equal(0, gradient[0, 0]);
        Assert.Equal(0, gradient[1, 1]);
    }

    // A = diag(2s, s)、Q = UVᵀ、損失 c·(Q[1, 0] − Q[0, 1]) の勾配は、A[1, 0] が 2c/(3s)、A[0, 1] がその符号違い、対角は 0。
    // c = 1e308 では、有限な2つの随伴の差 Pᵢⱼ − Pⱼᵢ（と Wᵢⱼ − Wⱼᵢ）がオーバーフローして NaN になった（#136 のレビュー）。
    [Theory]
    [InlineData(1.0, 1.0)]
    [InlineData(1e-200, 1e-200)]
    [InlineData(1.0, 1e308)]
    [InlineData(1e200, 1e308)]
    public void SvdGradientsOfAdjointsWhoseDifferenceOverflowsStayAccurate(double s, double c)
    {
        var tape = new Tape();
        var matrix = new Matrix<double>(2, 2);
        matrix[0, 0] = 2 * s;
        matrix[1, 1] = s;
        var a = tape.Leaf(matrix);
        var svd = SingularValueDecompositionNode.Compute(a);
        var pick = new Matrix<double>(2, 2);
        pick[1, 0] = c;
        pick[0, 1] = -c;
        tape.Backward(Pair(tape, svd.U * svd.V.T, pick));
        var gradient = tape.GetGradient(a).ToMatrix();
        double expected = c / (3 * s) * 2;
        DifferentiationAssert.Close(expected, gradient[1, 0], 1e-14, 0, "[1, 0]");
        DifferentiationAssert.Close(-expected, gradient[0, 1], 1e-14, 0, "[0, 1]");
        Assert.Equal(0, gradient[0, 0]);
        Assert.Equal(0, gradient[1, 1]);
    }

    // A = diag(−s, s)（下三角部分を読む）、損失 V[0, 0]·V[1, 0] の勾配は A[1, 0] が 1/(λ₀ − λ₁) = −1/(2s)、ほかは 0。
    // s = 1e308 では有限な2つの固有値の差がオーバーフローする（#136 のレビュー。勾配は非正規化数の約 −5e-309）。
    [Theory]
    [InlineData(1.0)]
    [InlineData(1e300)]
    [InlineData(1e308)]
    [InlineData(double.MaxValue)]
    public void SymmetricEigenGradientsOfEigenvaluesFarApartStayAccurate(double s)
    {
        var tape = new Tape();
        var matrix = new Matrix<double>(2, 2);
        matrix[0, 0] = -s;
        matrix[1, 1] = s;
        var a = tape.Leaf(matrix);
        var eigen = SymmetricEigenDecompositionNode.Compute(a);
        var first = new LinearAlgebra.Vector<double>(2);
        first[0] = 1;
        var column = eigen.Vectors * tape.Constant(first);
        tape.Backward(XMath.Dot(column[..1], column[1..]));
        var gradient = tape.GetGradient(a).ToMatrix();
        double expected = -0.5 / s;
        DifferentiationAssert.Close(expected, gradient[1, 0], 1e-14, 0, "[1, 0]");
        Assert.Equal(0, gradient[0, 1]);
        Assert.Equal(0, gradient[0, 0]);
        Assert.Equal(0, gradient[1, 1]);
    }

    [Fact]
    public void DecompositionsOfTheSameShapeKeepSeparateResults()
    {
        // 同じ大きさの分解を2つずつ記録し、一方の結果をもう一方の入力に使う。
        var random = new Random(1700);
        const int N = 6;
        var a1 = RandomMatrix(random, N, N);
        var a2 = RandomMatrix(random, N, N);
        var g = RandomMatrix(random, N, N);
        var c = RandomVector(random, N);
        ScalarNode Build(Tape tape, MatrixNode<double>[] m, VectorNode<double>[] v)
        {
            var qr1 = QrDecompositionNode.Compute(m[0]);
            var qr2 = QrDecompositionNode.Compute(m[1] + qr1.R);
            var svd1 = SingularValueDecompositionNode.Compute(m[1]);
            var svd2 = SingularValueDecompositionNode.Compute(qr2.Q * m[0]);
            var symmetric = m[0] * m[0].T;
            var eigen1 = SymmetricEigenDecompositionNode.Compute(symmetric);
            var eigen2 = SymmetricEigenDecompositionNode.Compute(svd1.U * m[1].T + symmetric);
            return Pair(tape, qr1.Q, g) + Pair(tape, qr2.R, g) + XMath.Dot(tape.Constant(c), svd1.Values) + XMath.Dot(tape.Constant(c), svd2.Values)
                + Pair(tape, svd2.U * svd2.V.T, g) + XMath.Dot(tape.Constant(c), eigen1.Values) + XMath.Dot(eigen2.Values, eigen1.Values)
                + Pair(tape, XMath.Hadamard(eigen2.Vectors, eigen2.Vectors), g);
        }

        AssertGradients([a1, a2], [], Build, 1e-6);

        // テープを使い回しても、前の記録の分解は値に影響しない（同じ大きさの分解を前の記録と逆の順に貸す）。
        var tape = new Tape();
        string expected = Fingerprint(tape, a1, a2, Build);
        for (int i = 0; i < 3; i++)
        {
            Fingerprint(tape, a2, a1, Build);
            Assert.Equal(expected, Fingerprint(tape, a1, a2, Build));
        }

        Assert.Equal(expected, Fingerprint(new Tape(), a1, a2, Build));
    }

    [Fact]
    public void ConstantInputsRecordNothing()
    {
        var random = new Random(1800);
        var tape = new Tape();
        var constant = tape.Constant(RandomMatrix(random, 5, 3));
        var qr = QrDecompositionNode.Compute(constant);
        var svd = SingularValueDecompositionNode.Compute(constant);
        var eigen = SymmetricEigenDecompositionNode.Compute(tape.Constant(SymmetricMatrix(random, 4, garbleUpper: false)));
        Assert.False(qr.Q.RequiresGradient);
        Assert.False(qr.R.RequiresGradient);
        Assert.False(svd.Values.RequiresGradient);
        Assert.False(svd.U.RequiresGradient);
        Assert.False(svd.V.RequiresGradient);
        Assert.False(eigen.Values.RequiresGradient);
        Assert.False(eigen.Vectors.RequiresGradient);
    }

    [Fact]
    public void EmptyMatricesDecomposeWithoutGradients()
    {
        foreach (var (m, n) in new[] { (0, 0), (0, 3), (3, 0) })
        {
            var tape = new Tape();
            var a = tape.Leaf(new Matrix<double>(m, n));
            var qr = QrDecompositionNode.Compute(a);
            var svd = SingularValueDecompositionNode.Compute(a);
            int k = Math.Min(m, n);
            Assert.Equal((m, k), (qr.Q.Rows, qr.Q.Columns));
            Assert.Equal((k, n), (qr.R.Rows, qr.R.Columns));
            Assert.Equal(k, svd.Values.Length);
            Assert.Equal((m, k), (svd.U.Rows, svd.U.Columns));
            Assert.Equal((n, k), (svd.V.Rows, svd.V.Columns));
            ScalarNode loss = XMath.Sum(XMath.ColumnSums(qr.Q)) + XMath.Sum(XMath.ColumnSums(qr.R)) + XMath.Sum(svd.Values) + XMath.Sum(XMath.ColumnSums(svd.U));
            if (m == n)
            {
                var eigen = SymmetricEigenDecompositionNode.Compute(a);
                Assert.Equal(0, eigen.Order);
                loss += XMath.Sum(eigen.Values);
            }

            tape.Backward(loss);
            Assert.Equal(0, loss.Value);
            Assert.Equal((m, n), (tape.GetGradient(a).Rows, tape.GetGradient(a).Columns));
        }
    }

    [Fact]
    public void InvalidArgumentsAreRejected()
    {
        var tape = new Tape();
        var rectangular = tape.Leaf(new Matrix<double>(3, 2));
        DifferentiationAssert.Throws<ArgumentException>(() => SymmetricEigenDecompositionNode.Compute(rectangular), "symmetric eigen of a rectangular matrix");
        DifferentiationAssert.Throws<InvalidOperationException>(() => _ = default(QrDecompositionNode).Q, "an uninitialized QR node");
        DifferentiationAssert.Throws<InvalidOperationException>(() => _ = default(SymmetricEigenDecompositionNode).Order, "an uninitialized eigen node");
        DifferentiationAssert.Throws<InvalidOperationException>(() => _ = default(SingularValueDecompositionNode).V, "an uninitialized SVD node");
        var qr = QrDecompositionNode.Compute(rectangular);
        var svd = SingularValueDecompositionNode.Compute(rectangular);
        var identity = tape.Leaf(Identity(3));
        var eigen = SymmetricEigenDecompositionNode.Compute(identity);
        tape.Backward(XMath.Sum(svd.Values) + XMath.Sum(eigen.Values) + XMath.Sum(XMath.ColumnSums(qr.R)));
        DifferentiationAssert.Throws<InvalidOperationException>(() => QrDecompositionNode.Compute(rectangular), "recording after the backward pass");
        DifferentiationAssert.Throws<InvalidOperationException>(() => SingularValueDecompositionNode.Compute(rectangular), "recording after the backward pass");
        DifferentiationAssert.Throws<InvalidOperationException>(() => SymmetricEigenDecompositionNode.Compute(identity), "recording after the backward pass");
        tape.Reset();
        DifferentiationAssert.Throws<InvalidOperationException>(() => _ = qr.R, "a decomposition before Reset");
        DifferentiationAssert.Throws<InvalidOperationException>(() => _ = svd.U, "a decomposition before Reset");
        DifferentiationAssert.Throws<InvalidOperationException>(() => _ = eigen.Vectors, "a decomposition before Reset");
    }

    // 対称行列 (M + Mᵀ)/2（固有値は互いに離れる）。garbleUpper なら、コアが読まない上三角部分を無関係な値にする。
    internal static Matrix<double> SymmetricMatrix(Random random, int n, bool garbleUpper)
    {
        var m = RandomMatrix(random, n, n);
        var a = new Matrix<double>(n, n);
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < n; i++)
            {
                a[i, j] = garbleUpper && i < j ? 3 * random.NextDouble() - 1.5 : 0.5 * (m[i, j] + m[j, i]);
            }
        }

        return a;
    }

    private static Matrix<double> Hilbert(int m, int n)
    {
        var a = new Matrix<double>(m, n);
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < m; i++)
            {
                a[i, j] = 1.0 / (i + j + 1);
            }
        }

        return a;
    }

    private static Matrix<double> Identity(int n)
    {
        var a = new Matrix<double>(n, n);
        for (int i = 0; i < n; i++)
        {
            a[i, i] = 1;
        }

        return a;
    }

    private static string Fingerprint(Tape tape, Matrix<double> first, Matrix<double> second,
        Func<Tape, MatrixNode<double>[], VectorNode<double>[], ScalarNode> build)
    {
        tape.Reset();
        var leaves = new[] { tape.Leaf(first), tape.Leaf(second) };
        var loss = build(tape, leaves, []);
        tape.Backward(loss);
        return BitConverter.DoubleToInt64Bits(loss.Value).ToString("X16") + ";"
            + Convert.ToHexString(System.Runtime.InteropServices.MemoryMarshal.AsBytes(tape.GetGradient(leaves[0]).ToMatrix().AsSpan())) + ";"
            + Convert.ToHexString(System.Runtime.InteropServices.MemoryMarshal.AsBytes(tape.GetGradient(leaves[1]).ToMatrix().AsSpan()));
    }

    private static void AssertBits(ReadOnlySpan<double> expected, ReadOnlySpan<double> actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            DifferentiationAssert.EqualBits(expected[i], actual[i], $"[{i}]");
        }
    }
}
