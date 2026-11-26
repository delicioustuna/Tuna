using Tuna.LinearAlgebra;
using static Tuna.Tests.LinearAlgebra.LinearAlgebraTestUtilities;

namespace Tuna.Tests.LinearAlgebra;

// 二段階の縮約（密行列 → 帯幅 64 の帯行列 → 三重・二重対角行列）を Always で強制し、
// 一段階の縮約と同じ判定比で分解を確かめ、固有値・特異値が一段階の縮約の結果と一致することを確かめる。
// 帯幅をまたぐ大きさ（端数のパネル、最後のスイープが短い場合）と、バルジ追跡が並列になる大きさを含める。
[Trait("LinearAlgebra", "TwoStage")]
[Collection(nameof(ParallelismCollection))]
public class TwoStageReductionTests
{
    public static TheoryData<int> Sizes => new() { 129, 130, 193, 300 };

    /// <summary>対称行列の固有値分解（固有ベクトルあり・なし）</summary>
    [Theory]
    [MemberData(nameof(Sizes))]
    public void DecomposesSymmetricMatrix(int n)
    {
        var a = RandomSymmetric(n, 900 + n);
        var reference = Run(TwoStageReduction.Never, () => a.SymmetricEigen());
        var eig = Run(TwoStageReduction.Always, () => a.SymmetricEigen());
        var values = Run(TwoStageReduction.Always, () => SymmetricEigenDecomposition.Compute(a, computeVectors: false));
        Assert.Null(values.Vectors);
        double norm = Norm(a);
        AssertClose(reference.Values, eig.Values, norm, n);
        AssertClose(reference.Values, values.Values, norm, n);

        var v = eig.Vectors!;
        AssertRatio(Difference(MultiplyNaive(a, v), MultiplyNaive(v, Matrix<double>.FromDiagonal(eig.Values))), norm, n, "AV - VΛ");
        AssertRatio(OrthogonalityError(v), 1, n, "VᵀV - I");
    }

    /// <summary>特異値分解（正方・縦長・QR を先に行う縦長・横長、特異ベクトルあり・なし）</summary>
    [Theory]
    [InlineData(130, 130)]
    [InlineData(300, 300)]
    [InlineData(350, 300)]
    [InlineData(600, 300)]
    [InlineData(260, 330)]
    public void DecomposesGeneralMatrix(int m, int n)
    {
        var a = Random(m, n, 7 * m + n);
        var reference = Run(TwoStageReduction.Never, () => a.Svd());
        var svd = Run(TwoStageReduction.Always, () => a.Svd());
        var values = Run(TwoStageReduction.Always, () => SingularValueDecomposition.Compute(a, computeVectors: false));
        Assert.Null(values.U);
        double norm = Norm(a);
        int dim = Math.Max(m, n);
        AssertClose(reference.Values, svd.Values, norm, dim);
        AssertClose(reference.Values, values.Values, norm, dim);
        for (int i = 1; i < svd.Values.Length; i++)
        {
            Assert.True(svd.Values[i - 1] >= svd.Values[i] && svd.Values[i] >= 0);
        }

        var usv = MultiplyNaive(MultiplyNaive(svd.U!, Matrix<double>.FromDiagonal(svd.Values)), svd.V!, transposeB: true);
        AssertRatio(Difference(a, usv), norm, dim, "A - UΣVᵀ");
        AssertRatio(OrthogonalityError(svd.U!), 1, m, "UᵀU - I");
        AssertRatio(OrthogonalityError(svd.V!), 1, n, "VᵀV - I");
    }

    /// <summary>重複した固有値・特異値とランク落ち（帯行列の一部が0になる、反射の係数が0になる）</summary>
    [Fact]
    public void HandlesRankDeficientMatrix()
    {
        int n = 200;
        var spectrum = new double[n];
        for (int i = 0; i < n; i++)
        {
            spectrum[i] = i < n / 2 ? 0 : (i % 3) + 1;
        }

        var s = SymmetricWithSpectrum(spectrum, 77);
        var eig = Run(TwoStageReduction.Always, () => s.SymmetricEigen());
        Array.Sort(spectrum);
        AssertClose(spectrum, eig.Values, Norm(s), n);
        AssertRatio(OrthogonalityError(eig.Vectors!), 1, n, "VᵀV - I");

        var a = Random(220, 200, 78);
        for (int i = 0; i < a.Rows; i++)
        {
            a[i, 3] = 0;
            a[i, 150] = a[i, 10];
        }

        var svd = Run(TwoStageReduction.Always, () => a.Svd());
        var usv = MultiplyNaive(MultiplyNaive(svd.U!, Matrix<double>.FromDiagonal(svd.Values)), svd.V!, transposeB: true);
        AssertRatio(Difference(a, usv), Norm(a), 220, "A - UΣVᵀ");
        Assert.True(svd.Values[^1] <= 30 * 220 * Epsilon * svd.Values[0]);
        Assert.True(svd.Values[^2] <= 30 * 220 * Epsilon * svd.Values[0]);
    }

    /// <summary>
    /// バルジ追跡を複数のスレッドで進める大きさで、1スレッドと全コアの固有値・特異値が一致する（既定の Auto で二段階になる大きさ）
    /// </summary>
    [Fact]
    public void ParallelBulgeChasingMatchesSequential()
    {
        const int n = 2048;
        var s = RandomSymmetric(n, 5);
        var a = Random(n, n, 6);
        int saved = LinearAlgebraSettings.MaxDegreeOfParallelism;
        try
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = 1;
            var eigOne = SymmetricEigenDecomposition.Compute(s, computeVectors: false);
            var svdOne = SingularValueDecomposition.Compute(a, computeVectors: false);
            LinearAlgebraSettings.MaxDegreeOfParallelism = Math.Max(2, Environment.ProcessorCount);
            var eigMany = SymmetricEigenDecomposition.Compute(s, computeVectors: false);
            var svdMany = SingularValueDecomposition.Compute(a, computeVectors: false);
            var eigReference = Run(TwoStageReduction.Never, () => SymmetricEigenDecomposition.Compute(s, computeVectors: false));
            var svdReference = Run(TwoStageReduction.Never, () => SingularValueDecomposition.Compute(a, computeVectors: false));

            Assert.Equal(eigOne.Values, eigMany.Values);
            Assert.Equal(svdOne.Values, svdMany.Values);
            AssertClose(eigReference.Values, eigMany.Values, Norm(s), n);
            AssertClose(svdReference.Values, svdMany.Values, Norm(a), n);
        }
        finally
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = saved;
        }
    }

    /// <summary>既定は Auto で、範囲外の値は設定できない</summary>
    [Fact]
    public void ValidatesSetting()
    {
        Assert.Equal(TwoStageReduction.Auto, LinearAlgebraSettings.TwoStageReduction);
        Assert.Throws<ArgumentOutOfRangeException>(() => LinearAlgebraSettings.TwoStageReduction = (TwoStageReduction)3);
        Assert.Equal(TwoStageReduction.Auto, LinearAlgebraSettings.TwoStageReduction);
    }

    private static T Run<T>(TwoStageReduction mode, Func<T> compute)
    {
        var saved = LinearAlgebraSettings.TwoStageReduction;
        try
        {
            LinearAlgebraSettings.TwoStageReduction = mode;
            return compute();
        }
        finally
        {
            LinearAlgebraSettings.TwoStageReduction = saved;
        }
    }

    private static void AssertClose(double[] expected, double[] actual, double scale, int n)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            AssertRatio(Math.Abs(expected[i] - actual[i]), scale, n, $"value[{i}]");
        }
    }
}
