using System.Numerics;
using System.Runtime.InteropServices;
using Tuna.LinearAlgebra;

namespace Tuna.Tests.LinearAlgebra;

/// <summary>公開 GEMM の領域・検査・計算を省略する場合の契約を、両精度で同じ条件で確かめる</summary>
public abstract class GemmContractTests<T> where T : unmanaged, IFloatingPointIeee754<T>
{
    protected abstract void Multiply(
        BlasTranspose transA, BlasTranspose transB, int m, int n, int k,
        T alpha, ReadOnlySpan<T> a, int lda, ReadOnlySpan<T> b, int ldb, T beta, Span<T> c, int ldc);

    /// <summary>ずれた先頭・最小長の領域・列の余りを使い、入力と出力の外側が変わらないことも確かめる</summary>
    [Theory]
    [InlineData(3, 7, 19)]
    [InlineData(49, 13, 257)]
    [InlineData(289, 9, 19)]
    [InlineData(7, 3073, 83)]
    public void ProductPreservesInputsAndMemoryOutsideOutput(int m, int n, int k)
    {
        foreach (var transA in Enum.GetValues<BlasTranspose>())
        {
            foreach (var transB in Enum.GetValues<BlasTranspose>())
            {
                var (rowsA, columnsA) = Shape(transA, m, k);
                var (rowsB, columnsB) = Shape(transB, k, n);
                int lda = rowsA + 2, ldb = rowsB + 3, ldc = m + 4;
                int lengthA = Extent(rowsA, columnsA, lda), lengthB = Extent(rowsB, columnsB, ldb), lengthC = Extent(m, n, ldc);
                var a = Matrix(rowsA, columnsA, lda, 5);
                var b = Matrix(rowsB, columnsB, ldb, 7);
                var c = Matrix(m, n, ldc, 11);
                var savedA = (T[])a.Clone();
                var savedB = (T[])b.Clone();
                var savedC = (T[])c.Clone();

                Multiply(transA, transB, m, n, k, Number(-0.5), a.AsSpan(5, lengthA), lda,
                    b.AsSpan(7, lengthB), ldb, Number(-0.25), c.AsSpan(11, lengthC), ldc);

                VerifyProduct(transA, transB, m, n, k, -0.5, savedA, 5, lda, savedB, 7, ldb, -0.25, savedC, 11, ldc, c);
                SameBits(savedA, a);
                SameBits(savedB, b);
            }
        }
    }

    /// <summary>α = ±0 または k = 0 なら、非有限の入力を使わずに C だけを処理する。密な配置と余りのある配置を含む</summary>
    [Theory]
    [InlineData(0, double.NaN, 0.0)]
    [InlineData(0, double.PositiveInfinity, 1.0)]
    [InlineData(0, double.NegativeInfinity, -2.0)]
    [InlineData(17, -0.0, -0.0)]
    [InlineData(17, -0.0, 1.0)]
    [InlineData(17, 0.0, -2.0)]
    public void InactiveProductOnlyScalesLogicalOutput(int k, double alpha, double beta)
    {
        // 単精度でも C の β 倍が並列化されうる大きさ。β = 1 では -0 と NaN のビットも保つ。
        const int m = 259, n = 257, offset = 3;
        T[] zero = [T.Zero];
        int savedThreads = LinearAlgebraSettings.MaxDegreeOfParallelism;
        try
        {
            foreach (int threads in new[] { 1, Environment.ProcessorCount })
            {
                LinearAlgebraSettings.MaxDegreeOfParallelism = threads;
                foreach (int extra in new[] { 0, 3 })
                {
                    foreach (var transA in Enum.GetValues<BlasTranspose>())
                    {
                        foreach (var transB in Enum.GetValues<BlasTranspose>())
                        {
                            var (rowsA, columnsA) = Shape(transA, m, k);
                            var (rowsB, columnsB) = Shape(transB, k, n);
                            int lda = Math.Max(1, rowsA), ldb = Math.Max(1, rowsB), ldc = m + extra;
                            var a = Enumerable.Repeat(T.NaN, Extent(rowsA, columnsA, lda)).ToArray();
                            var b = Enumerable.Repeat(T.PositiveInfinity, Extent(rowsB, columnsB, ldb)).ToArray();
                            var c = Matrix(m, n, ldc, offset);
                            T[] values = [T.NaN, T.PositiveInfinity, T.NegativeInfinity, -T.Zero, T.Epsilon, Number(4)];
                            for (int j = 0; j < n; j++)
                            {
                                for (int i = 0; i < m; i++)
                                {
                                    c[offset + i + j * ldc] = values[(i + j) % values.Length];
                                }
                            }

                            var saved = (T[])c.Clone();
                            Multiply(transA, transB, m, n, k, Number(alpha), a, lda, b, ldb, Number(beta), c.AsSpan(offset), ldc);
                            for (int index = 0; index < c.Length; index++)
                            {
                                int position = index - offset;
                                bool active = position >= 0 && position / ldc < n && position % ldc < m;
                                if (!active || beta == 1)
                                {
                                    SameBits(saved.AsSpan(index, 1), c.AsSpan(index, 1));
                                }
                                else if (beta == 0)
                                {
                                    SameBits(zero, c.AsSpan(index, 1));
                                }
                                else
                                {
                                    Assert.Equal(Number(beta) * saved[index], c[index]);
                                }
                            }
                        }
                    }
                }
            }
        }
        finally
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = savedThreads;
        }
    }

    /// <summary>出力に要素がなければ、共有する配列に NaN の係数を渡しても配列を変えない</summary>
    [Theory]
    [InlineData(0, 4, 3)]
    [InlineData(4, 0, 3)]
    [InlineData(0, 0, int.MaxValue)]
    public void EmptyOutputLeavesBackingStorageUntouched(int m, int n, int k)
    {
        foreach (var transA in Enum.GetValues<BlasTranspose>())
        {
            foreach (var transB in Enum.GetValues<BlasTranspose>())
            {
                var data = Enumerable.Repeat(-T.Zero, 64).ToArray();
                data[7] = T.NaN;
                var saved = (T[])data.Clone();
                Multiply(transA, transB, m, n, k, T.NaN, data, Math.Max(1, Shape(transA, m, k).Rows),
                    data, Math.Max(1, Shape(transB, k, n).Rows), T.NaN, data.AsSpan(5), Math.Max(1, m));
                SameBits(saved, data);
            }
        }
    }

    /// <summary>検査の失敗は引数名を返し、β = 0 であっても入力や出力を先に書き換えない</summary>
    [Theory]
    [InlineData("m")]
    [InlineData("n")]
    [InlineData("k")]
    [InlineData("lda")]
    [InlineData("ldb")]
    [InlineData("ldc")]
    [InlineData("a")]
    [InlineData("b")]
    [InlineData("c")]
    public void InvalidArgumentsLeaveEveryBufferUntouched(string parameter)
    {
        foreach (var transA in Enum.GetValues<BlasTranspose>())
        {
            foreach (var transB in Enum.GetValues<BlasTranspose>())
            {
                int m = 2, n = 3, k = 4;
                var (rowsA, columnsA) = Shape(transA, m, k);
                var (rowsB, columnsB) = Shape(transB, k, n);
                int lda = rowsA + 1, ldb = rowsB + 1, ldc = m + 1;
                var a = Matrix(rowsA, columnsA, lda, 0);
                var b = Matrix(rowsB, columnsB, ldb, 0);
                var c = Matrix(m, n, ldc, 0);
                var savedA = (T[])a.Clone();
                var savedB = (T[])b.Clone();
                var savedC = (T[])c.Clone();
                int lengthA = Extent(rowsA, columnsA, lda), lengthB = Extent(rowsB, columnsB, ldb), lengthC = Extent(m, n, ldc);
                switch (parameter)
                {
                    case "m":
                        m = -1;
                        break;
                    case "n":
                        n = -1;
                        break;
                    case "k":
                        k = -1;
                        break;
                    case "lda":
                        lda = rowsA - 1;
                        break;
                    case "ldb":
                        ldb = rowsB - 1;
                        break;
                    case "ldc":
                        ldc = m - 1;
                        break;
                    case "a":
                        lengthA--;
                        break;
                    case "b":
                        lengthB--;
                        break;
                    case "c":
                        lengthC--;
                        break;
                }

                Action call = () => Multiply(transA, transB, m, n, k, T.One, a.AsSpan(0, lengthA), lda,
                    b.AsSpan(0, lengthB), ldb, T.Zero, c.AsSpan(0, lengthC), ldc);
                ArgumentException error = parameter is "a" or "b" or "c"
                    ? Assert.Throws<ArgumentException>(call)
                    : Assert.Throws<ArgumentOutOfRangeException>(call);
                Assert.Equal(parameter, error.ParamName);
                SameBits(savedA, a);
                SameBits(savedB, b);
                SameBits(savedC, c);
            }
        }
    }

    /// <summary>必要な領域長が int を超える場合も、折り返さずに該当する領域の長さ不足を報告する</summary>
    [Theory]
    [InlineData("a")]
    [InlineData("b")]
    [InlineData("c")]
    public void RequiredSpanLengthDoesNotWrapAround(string parameter)
    {
        var a = new T[4];
        var b = new T[4];
        var c = new T[4];
        var error = Assert.Throws<ArgumentException>(() => Multiply(BlasTranspose.None, BlasTranspose.None, 2, 2, 2,
            T.Zero, a, parameter == "a" ? int.MaxValue : 2, b, parameter == "b" ? int.MaxValue : 2,
            T.One, c, parameter == "c" ? int.MaxValue : 2));
        Assert.Equal(parameter, error.ParamName);
    }

    /// <summary>異なる先頭と列の間隔の重なりを、実際の要素位置の集合と比べる。余りだけを共有する場合は計算できる</summary>
    [Theory]
    [InlineData(false, -0.5)]
    [InlineData(true, -0.5)]
    [InlineData(false, -0.0)]
    [InlineData(true, -0.0)]
    public void OverlapChecksMatchElementIntersection(bool shareB, double alpha)
    {
        const int m = 2, n = 3, k = 4, inputOffset = 2;
        double beta = alpha == 0 ? 1 : 0.25;
        foreach (var transA in Enum.GetValues<BlasTranspose>())
        {
            foreach (var transB in Enum.GetValues<BlasTranspose>())
            {
                var (rowsA, columnsA) = Shape(transA, m, k);
                var (rowsB, columnsB) = Shape(transB, k, n);
                for (int extra = 0; extra < 3; extra++)
                {
                    int lda = rowsA + extra, ldb = rowsB + extra;
                    var inputPositions = Positions(shareB ? rowsB : rowsA, shareB ? columnsB : columnsA, shareB ? ldb : lda, inputOffset);
                    for (int ldc = m; ldc <= m + 2; ldc++)
                    {
                        for (int outputOffset = 0; outputOffset < 16; outputOffset++)
                        {
                            var shared = Enumerable.Range(0, 64).Select(i => Number((i % 7 + 1) / 4.0)).ToArray();
                            var other = (T[])shared.Clone();
                            var saved = (T[])shared.Clone();
                            var savedOther = (T[])other.Clone();
                            var a = shareB ? other : shared;
                            var b = shareB ? shared : other;
                            int offsetA = shareB ? 0 : inputOffset, offsetB = shareB ? inputOffset : 0;
                            Action call = () => Multiply(transA, transB, m, n, k, Number(alpha), a.AsSpan(offsetA), lda,
                                b.AsSpan(offsetB), ldb, Number(beta), shared.AsSpan(outputOffset), ldc);
                            if (inputPositions.Overlaps(Positions(m, n, ldc, outputOffset)))
                            {
                                Assert.Equal("c", Assert.Throws<ArgumentException>(call).ParamName);
                                SameBits(saved, shared);
                            }
                            else
                            {
                                call();
                                VerifyProduct(transA, transB, m, n, k, alpha, shareB ? savedOther : saved, offsetA, lda,
                                    shareB ? saved : savedOther, offsetB, ldb, beta, saved, outputOffset, ldc, shared);
                            }

                            SameBits(savedOther, other);
                        }
                    }
                }
            }
        }
    }

    /// <summary>A と B の同じ領域を、転置の各組み合わせで読める</summary>
    [Theory]
    [InlineData(3)]
    [InlineData(25)]
    public void InputsMayShareStorage(int size)
    {
        foreach (var transA in Enum.GetValues<BlasTranspose>())
        {
            foreach (var transB in Enum.GetValues<BlasTranspose>())
            {
                int ld = size + 2;
                var input = Matrix(size, size, ld, 5);
                var savedInput = (T[])input.Clone();
                var c = Matrix(size, size, ld, 7);
                var savedC = (T[])c.Clone();
                Multiply(transA, transB, size, size, size, Number(-0.5), input.AsSpan(5), ld,
                    input.AsSpan(5), ld, Number(0.25), c.AsSpan(7), ld);
                VerifyProduct(transA, transB, size, size, size, -0.5, savedInput, 5, ld, savedInput, 5, ld, 0.25, savedC, 7, ld, c);
                SameBits(savedInput, input);
            }
        }
    }

    private static (int Rows, int Columns) Shape(BlasTranspose trans, int rows, int columns) =>
        trans == BlasTranspose.None ? (rows, columns) : (columns, rows);

    private static int Extent(int rows, int columns, int ld) => rows == 0 || columns == 0 ? 0 : (columns - 1) * ld + rows;

    private static T Number(double value) => T.CreateChecked(value);

    private static T[] Matrix(int rows, int columns, int ld, int offset)
    {
        var data = Enumerable.Repeat(T.NaN, offset + Extent(rows, columns, ld) + 5).ToArray();
        for (int j = 0; j < columns; j++)
        {
            for (int i = 0; i < rows; i++)
            {
                data[offset + i + j * ld] = Number(((3 * i + 5 * j) % 13 - 6) / 4.0);
            }
        }

        return data;
    }

    private static HashSet<int> Positions(int rows, int columns, int ld, int offset)
    {
        var result = new HashSet<int>();
        for (int j = 0; j < columns; j++)
        {
            for (int i = 0; i < rows; i++)
            {
                result.Add(offset + i + j * ld);
            }
        }

        return result;
    }

    /// <summary>小さい2進有理数なので積和も両精度で正確に表せる。許容誤差を設けず倍精度の三重ループと比較する</summary>
    private static void VerifyProduct(
        BlasTranspose transA, BlasTranspose transB, int m, int n, int k, double alpha,
        T[] a, int offsetA, int lda, T[] b, int offsetB, int ldb, double beta, T[] originalC, int offsetC, int ldc, T[] actual)
    {
        var active = Positions(m, n, ldc, offsetC);
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < m; i++)
            {
                double sum = 0;
                for (int p = 0; p < k; p++)
                {
                    int indexA = offsetA + (transA == BlasTranspose.None ? i + p * lda : p + i * lda);
                    int indexB = offsetB + (transB == BlasTranspose.None ? p + j * ldb : j + p * ldb);
                    sum += double.CreateChecked(a[indexA]) * double.CreateChecked(b[indexB]);
                }

                int index = offsetC + i + j * ldc;
                Assert.Equal(Number(alpha * sum + beta * double.CreateChecked(originalC[index])), actual[index]);
            }
        }

        for (int index = 0; index < actual.Length; index++)
        {
            if (!active.Contains(index))
            {
                SameBits(originalC.AsSpan(index, 1), actual.AsSpan(index, 1));
            }
        }
    }

    private static void SameBits(ReadOnlySpan<T> expected, ReadOnlySpan<T> actual) =>
        Assert.True(MemoryMarshal.AsBytes(expected).SequenceEqual(MemoryMarshal.AsBytes(actual)), "値のビット表現が変わった");
}

[Collection(nameof(ParallelismCollection))]
public sealed class DoubleGemmContractTests : GemmContractTests<double>
{
    protected override void Multiply(
        BlasTranspose transA, BlasTranspose transB, int m, int n, int k,
        double alpha, ReadOnlySpan<double> a, int lda, ReadOnlySpan<double> b, int ldb, double beta, Span<double> c, int ldc) =>
        Blas.Gemm(transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

[Collection(nameof(ParallelismCollection))]
public sealed class SingleGemmContractTests : GemmContractTests<float>
{
    protected override void Multiply(
        BlasTranspose transA, BlasTranspose transB, int m, int n, int k,
        float alpha, ReadOnlySpan<float> a, int lda, ReadOnlySpan<float> b, int ldb, float beta, Span<float> c, int ldc) =>
        Blas.Gemm(transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
