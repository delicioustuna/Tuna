using Tuna.LinearAlgebra;
using static Tuna.Tests.LinearAlgebra.LinearAlgebraTestUtilities;

namespace Tuna.Tests.LinearAlgebra;

// QR・対称固有値・特異値分解の同一オブジェクト再利用（Recompute）を検証する。
[Trait("LinearAlgebra", "Recompute")]
[Collection(nameof(ParallelismCollection))]
public class OrthogonalRecomputeTests
{
    // 非ブロック版（k ≤ 128）、ブロック版、並列パネル分解（300 次）、縦長・横長
    public static TheoryData<int, int> QrShapes => new() { { 0, 0 }, { 0, 3 }, { 3, 0 }, { 1, 1 }, { 5, 3 }, { 3, 5 }, { 40, 40 }, { 200, 130 }, { 130, 200 }, { 300, 300 } };

    // QL 法（n ≤ 25）、分割統治法、ブロック三重対角化（n > 128）、並列分割統治法（300 次）
    public static TheoryData<int, bool> EigenSizes
    {
        get
        {
            var data = new TheoryData<int, bool>();
            foreach (int n in new[] { 0, 1, 2, 25, 26, 65, 129, 300 })
            {
                data.Add(n, true);
                data.Add(n, false);
            }

            return data;
        }
    }

    // QR 法（n ≤ 25）、分割統治法、QR 先行の縦長（m ≥ 1.6n）、転置経路の横長、並列分割統治法（300 次）
    public static TheoryData<int, int, bool> SvdShapes
    {
        get
        {
            var data = new TheoryData<int, int, bool>();
            foreach (var (m, n) in new[] { (0, 3), (3, 0), (1, 1), (5, 3), (3, 5), (26, 26), (60, 30), (30, 60), (140, 130), (300, 300), (300, 150) })
            {
                data.Add(m, n, true);
                data.Add(m, n, false);
            }

            return data;
        }
    }

    // Recompute の結果が新規分解とビット単位で一致し、生成済みの Q・R が同一インスタンスのまま書き換わること
    [Theory]
    [MemberData(nameof(QrShapes))]
    public void QrRecomputeMatchesFreshDecomposition(int m, int n)
    {
        var second = Random(m, n, 2000 + m + n);
        var qr = Random(m, n, 1000 + m + n).Qr();
        var q = qr.Q;
        var r = qr.R;

        qr.Recompute(second);
        var fresh = second.Qr();
        Assert.Same(q, qr.Q);
        Assert.Same(r, qr.R);
        Assert.Equal(fresh.Q.ToArray(), q.ToArray());
        Assert.Equal(fresh.R.ToArray(), r.ToArray());
        if (m >= n)
        {
            var b = Random(m, 1, 3000 + m).AsSpan().ToArray();
            Assert.Equal(fresh.Solve(b), qr.Solve(b));
        }

        // 元の行列は変えない
        Assert.Equal(Random(m, n, 2000 + m + n).ToArray(), second.ToArray());
    }

    // 因子のみ計算した分解では Q・R を遅延生成し、R の対角より下を書き換えても再計算時に 0 に戻ること
    [Fact]
    public void QrRecomputeFormsOnlyWhatWasRead()
    {
        var first = Random(50, 30, 11);
        var second = Random(50, 30, 12);
        var qr = QrDecomposition.ComputeFactors(first);
        qr.Recompute(second);
        var fresh = second.Qr();
        Assert.Equal(fresh.Q.ToArray(), qr.Q.ToArray());

        var r = qr.R;
        r.AsSpan().Fill(7);
        qr.Recompute(first);
        Assert.Same(r, qr.R);
        Assert.Equal(first.Qr().R.ToArray(), r.ToArray());
    }

    /// <summary>オーバーフローする大きさに近い列は、新しく分解する場合と同じく縮めてやり直す</summary>
    [Fact]
    public void QrRecomputeRescalesColumnsNearOverflow()
    {
        var large = Random(40, 6, 21);
        for (int i = 0; i < 40; i++)
        {
            large[i, 2] *= 1e307;
        }

        var qr = Random(40, 6, 22).Qr();
        qr.Recompute(large);
        var fresh = large.Qr();
        Assert.All(qr.R.AsSpan().ToArray(), value => Assert.True(double.IsFinite(value)));
        Assert.Equal(fresh.Q.ToArray(), qr.Q.ToArray());
        Assert.Equal(fresh.R.ToArray(), qr.R.ToArray());
    }

    /// <summary>入力はビュー（部分行列・転置）でもよく、overwriteInput で渡した領域そのものはその場で分解する。作ってある Q・R も入力にできる。形状が違う、一部だけ重なる入力は例外</summary>
    [Fact]
    public void QrRecomputeAcceptsViewsAndItsOwnStorage()
    {
        int m = 70, n = 45;
        var a = Random(m, n, 31);
        var expected = a.Qr();
        var qr = Random(m, n, 32).Qr();

        var padded = new Matrix<double>(m + 3, n + 5);
        a.AsMatrixSpan().CopyTo(padded[2..(m + 2), 4..(n + 4)]);
        qr.Recompute(padded[2..(m + 2), 4..(n + 4)]);
        Assert.Equal(expected.R.ToArray(), qr.R.ToArray());

        var wide = Random(n, m, 33);
        qr.Recompute(wide.T);
        Assert.Equal(wide.T.ToMatrix().Qr().Q.ToArray(), qr.Q.ToArray());

        var storage = Random(m, n, 34);
        var inPlace = QrDecomposition.Compute(storage, overwriteInput: true);
        a.AsMatrixSpan().CopyTo(storage);
        inPlace.Recompute(storage);
        Assert.Equal(expected.Q.ToArray(), inPlace.Q.ToArray());
        Assert.Equal(expected.R.ToArray(), inPlace.R.ToArray());

        var before = qr.R.ToArray();
        Assert.Throws<ArgumentException>(() => qr.Recompute(Random(m + 1, n, 35)));
        Assert.Throws<ArgumentException>(() => qr.Recompute(Random(m, n - 1, 36)));
        Assert.Equal(before, qr.R.ToArray());

        // 作ってある Q・R を入力にしてもよい（先に分解の領域へ写すので、新しく分解した結果と同じ。その行列には新しい値が書かれる）
        var own = Random(20, 20, 38).Qr();
        var ownQ = own.Q;
        var fromQ = ownQ.AsMatrixSpan().ToMatrix().Qr();
        own.Recompute(ownQ);
        Assert.Equal(fromQ.Q.ToArray(), ownQ.ToArray());
        Assert.Equal(fromQ.R.ToArray(), own.R.ToArray());
        var ownR = own.R;
        var fromR = ownR.AsMatrixSpan().ToMatrix().Qr();
        own.Recompute(ownR);
        Assert.Equal(fromR.R.ToArray(), ownR.ToArray());
        Assert.Equal(fromR.Q.ToArray(), own.Q.ToArray());

        // 正方行列の領域の転置は、領域と一部だけ重なる
        var square = Random(20, 20, 37);
        var squareQr = QrDecomposition.Compute(square, overwriteInput: true);
        var factors = square.ToArray();
        Assert.Throws<ArgumentException>(() => squareQr.Recompute(square.T));
        Assert.Equal(factors, square.ToArray());
    }

    /// <summary>対称行列の固有値分解を分解し直した結果は新しく分解した結果とビット単位で同じで、固有値の配列と固有ベクトルの行列は同じまま</summary>
    [Theory]
    [MemberData(nameof(EigenSizes))]
    public void SymmetricEigenRecomputeMatchesFreshDecomposition(int n, bool vectors)
    {
        var second = RandomSymmetric(n, 4000 + n);
        var eig = SymmetricEigenDecomposition.Compute(RandomSymmetric(n, 3000 + n), vectors);
        var values = eig.Values;
        var v = eig.Vectors;

        eig.Recompute(second);
        var fresh = SymmetricEigenDecomposition.Compute(second, vectors);
        Assert.Same(values, eig.Values);
        Assert.Same(v, eig.Vectors);
        Assert.Equal(fresh.Values, values);
        Assert.Equal(vectors, v is not null);
        if (vectors)
        {
            Assert.Equal(fresh.Vectors!.ToArray(), v!.ToArray());
        }

        Assert.Equal(RandomSymmetric(n, 4000 + n).ToArray(), second.ToArray());
    }

    /// <summary>入力はビューでもよく、下三角だけを読む。固有ベクトルの行列そのものを入力にしても、先に写してから分解する</summary>
    [Fact]
    public void SymmetricEigenRecomputeAcceptsViewsAndItsOwnResult()
    {
        int n = 70;
        var a = RandomSymmetric(n, 41);
        var eig = RandomSymmetric(n, 42).SymmetricEigen();

        var padded = new Matrix<double>(n + 3, n + 5);
        a.AsMatrixSpan().CopyTo(padded[2..(n + 2), 4..(n + 4)]);
        eig.Recompute(padded[2..(n + 2), 4..(n + 4)]);
        Assert.Equal(a.SymmetricEigen().Values, eig.Values);

        // 転置のビューの下三角は元の行列の上三角
        var general = Random(n, n, 43);
        eig.Recompute(general.T);
        var fromUpper = general.T.ToMatrix().SymmetricEigen();
        Assert.Equal(fromUpper.Values, eig.Values);
        Assert.Equal(fromUpper.Vectors!.ToArray(), eig.Vectors!.ToArray());

        var vectors = eig.Vectors!;
        var expected = vectors.Clone().SymmetricEigen();
        eig.Recompute(vectors);
        Assert.Equal(expected.Values, eig.Values);
        Assert.Equal(expected.Vectors!.ToArray(), vectors.ToArray());

        Assert.Throws<ArgumentException>(() => eig.Recompute(RandomSymmetric(n + 1, 44)));
        Assert.Throws<ArgumentException>(() => eig.Recompute(Random(n, n + 1, 45)));
        Assert.Equal(expected.Values, eig.Values);
    }

    /// <summary>特異値分解を分解し直した結果は新しく分解した結果とビット単位で同じで、特異値の配列と U・V は同じまま</summary>
    [Theory]
    [MemberData(nameof(SvdShapes))]
    public void SvdRecomputeMatchesFreshDecomposition(int m, int n, bool vectors)
    {
        var second = Random(m, n, 6000 + m + n);
        var svd = SingularValueDecomposition.Compute(Random(m, n, 5000 + m + n), vectors);
        var values = svd.Values;
        var u = svd.U;
        var v = svd.V;

        svd.Recompute(second);
        var fresh = SingularValueDecomposition.Compute(second, vectors);
        Assert.Same(values, svd.Values);
        Assert.Same(u, svd.U);
        Assert.Same(v, svd.V);
        Assert.Equal(fresh.Values, values);
        Assert.Equal(fresh.Rank, svd.Rank);
        Assert.Equal(vectors, u is not null);
        if (vectors)
        {
            Assert.Equal(fresh.U!.ToArray(), u!.ToArray());
            Assert.Equal(fresh.V!.ToArray(), v!.ToArray());
        }

        Assert.Equal(Random(m, n, 6000 + m + n).ToArray(), second.ToArray());
    }

    /// <summary>入力はビューでもよい。U そのものを入力にしても、先に写してから分解する。形状が違えば例外</summary>
    [Theory]
    [InlineData(40, 40)]
    [InlineData(70, 30)]
    [InlineData(30, 70)]
    public void SvdRecomputeAcceptsViewsAndItsOwnResult(int m, int n)
    {
        var a = Random(m, n, 51);
        var svd = Random(m, n, 52).Svd();

        var padded = new Matrix<double>(m + 3, n + 5);
        a.AsMatrixSpan().CopyTo(padded[2..(m + 2), 4..(n + 4)]);
        svd.Recompute(padded[2..(m + 2), 4..(n + 4)]);
        var expected = a.Svd();
        Assert.Equal(expected.Values, svd.Values);
        Assert.Equal(expected.U!.ToArray(), svd.U!.ToArray());

        var transposed = Random(n, m, 53);
        svd.Recompute(transposed.T);
        Assert.Equal(transposed.T.ToMatrix().Svd().V!.ToArray(), svd.V!.ToArray());

        if (m == n)
        {
            var own = svd.U!;
            var fromOwn = own.Clone().Svd();
            svd.Recompute(own);
            Assert.Equal(fromOwn.Values, svd.Values);
            Assert.Equal(fromOwn.U!.ToArray(), own.ToArray());
        }

        var before = svd.Values.ToArray();
        Assert.Throws<ArgumentException>(() => svd.Recompute(Random(m + 1, n, 54)));
        Assert.Throws<ArgumentException>(() => svd.Recompute(Random(m, n + 1, 55)));
        Assert.Equal(before, svd.Values);
    }

    /// <summary>二段階の縮約を使う設定でも、分解し直した結果は新しく分解した結果とビット単位で同じ</summary>
    [Theory]
    [InlineData(130, true)]
    [InlineData(130, false)]
    [InlineData(300, true)]
    public void RecomputeWithTwoStageReductionMatchesFreshDecomposition(int n, bool vectors)
    {
        var saved = LinearAlgebraSettings.TwoStageReduction;
        try
        {
            LinearAlgebraSettings.TwoStageReduction = TwoStageReduction.Always;
            var symmetric = RandomSymmetric(n, 61);
            var eig = SymmetricEigenDecomposition.Compute(RandomSymmetric(n, 62), vectors);
            eig.Recompute(symmetric);
            var freshEig = SymmetricEigenDecomposition.Compute(symmetric, vectors);
            Assert.Equal(freshEig.Values, eig.Values);
            Assert.Equal(freshEig.Vectors?.ToArray(), eig.Vectors?.ToArray());

            var general = Random(n + 20, n, 63);
            var svd = SingularValueDecomposition.Compute(Random(n + 20, n, 64), vectors);
            svd.Recompute(general);
            var freshSvd = SingularValueDecomposition.Compute(general, vectors);
            Assert.Equal(freshSvd.Values, svd.Values);
            Assert.Equal(freshSvd.U?.ToArray(), svd.U?.ToArray());
            Assert.Equal(freshSvd.V?.ToArray(), svd.V?.ToArray());
        }
        finally
        {
            LinearAlgebraSettings.TwoStageReduction = saved;
        }
    }
}
