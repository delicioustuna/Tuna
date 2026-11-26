using System.Numerics;
using Tuna.LinearAlgebra;
using static Tuna.Tests.LinearAlgebra.LinearAlgebraTestUtilities;

namespace Tuna.Tests.LinearAlgebra;

/// <summary>
/// 単精度の行列積（パッキングした単精度の GEMM）と複素数の行列積（実部と虚部を分けた実行列の積）の動作確認。
/// 参照は倍精度・複素数の単純な三重ループで求め、誤差は積和の絶対値の和に対する丸め誤差の上界で判定する
/// </summary>
[Trait("LinearAlgebra", "Matrix")]
public class SingleAndComplexProductTests
{
    /// <summary>単精度の丸めの単位の2倍（<c>2⁻²³</c>）</summary>
    private const double SingleEpsilon = 1.1920928955078125e-7;

    public static TheoryData<int, int, int> SingleSizes => new()
    {
        { 5, 300, 3 },      // m·k が小さく、列ごとの GEMV
        { 4, 9, 145 },
        { 16, 8, 40 },
        { 17, 9, 33 },
        { 33, 31, 257 },    // 端のパネル（幅 48 未満）と KC（256）をまたぐ k
        { 48, 8, 256 },
        { 49, 13, 513 },
        { 97, 70, 300 },
        { 300, 9, 5 },
        { 300, 300, 40 },   // 並列に計算されうる大きさ
    };

    public static TheoryData<int, int, int> ComplexSizes => new()
    {
        { 8, 8, 8 },        // 小さい積は列ごとの GEMV
        { 40, 41, 90 },
        { 65, 33, 129 },
        { 129, 70, 300 },
        { 4200, 130, 130 }, // 作業配列を C の列と内積方向のブロックに分ける
    };

    /// <summary>単精度の積が倍精度の参照と丸め誤差の範囲で一致する（転置の4通り）</summary>
    [Theory]
    [MemberData(nameof(SingleSizes))]
    public void SingleMatchesDoubleReference(int m, int n, int k)
    {
        var a = RandomSingle(m, k, 11);
        var b = RandomSingle(k, n, 12);
        var at = XMath.Transpose(a);
        var bt = XMath.Transpose(b);
        AssertSingle(a, b, XMath.Multiply(a, b), "AB");
        AssertSingle(a, b, XMath.Multiply(at.T, b), "(Aᵀ)ᵀB");
        AssertSingle(a, b, XMath.Multiply(a, bt.T), "A(Bᵀ)ᵀ");
        AssertSingle(a, b, XMath.Multiply(at.T, bt.T), "(Aᵀ)ᵀ(Bᵀ)ᵀ");
    }

    /// <summary>複素数の積が複素数のままの参照と丸め誤差の範囲で一致する（転置の4通り）</summary>
    [Theory]
    [MemberData(nameof(ComplexSizes))]
    public void ComplexMatchesReference(int m, int n, int k)
    {
        var a = RandomComplex(m, k, 21);
        var b = RandomComplex(k, n, 22);
        var at = XMath.Transpose(a);
        var bt = XMath.Transpose(b);
        AssertComplex(a, b, XMath.Multiply(a, b), "AB");
        AssertComplex(a, b, XMath.Multiply(at.T, b), "(Aᵀ)ᵀB");
        AssertComplex(a, b, XMath.Multiply(a, bt.T), "A(Bᵀ)ᵀ");
        AssertComplex(a, b, XMath.Multiply(at.T, bt.T), "(Aᵀ)ᵀ(Bᵀ)ᵀ");
    }

    /// <summary>部分行列のビュー（列の間隔が行数より大きい）を読み書きし、出力のビューの外は書き換えない</summary>
    [Fact]
    public void ProductsUseLeadingDimensionsOfViews()
    {
        var a = RandomSingle(80, 90, 31)[3..60, 5..70];
        var b = RandomSingle(100, 50, 32)[7..72, 1..41];
        var c = new Matrix<float>(70, 50);
        c.AsSpan().Fill(7f);
        XMath.Multiply(a, b, c[2..59, 4..44]);
        AssertSingle(a, b, c[2..59, 4..44], "single view");
        AssertUntouched(c, 2..59, 4..44, 7f);

        var x = RandomComplex(70, 80, 33)[3..60, 5..70];
        var y = RandomComplex(90, 50, 34)[7..72, 1..41];
        var z = new Matrix<Complex>(70, 50);
        z.AsSpan().Fill(new Complex(7, -7));
        XMath.Multiply(x, y, z[2..59, 4..44]);
        AssertComplex(x, y, z[2..59, 4..44], "complex view");
        AssertUntouched(z, 2..59, 4..44, new Complex(7, -7));
    }

    /// <summary>NaN・無限大は複素数の演算と同じく伝わる（積和に入れば結果は NaN か無限大）</summary>
    [Fact]
    public void NonFiniteValuesPropagate()
    {
        var a = RandomSingle(64, 64, 41);
        var b = RandomSingle(64, 64, 42);
        a[5, 7] = float.NaN;
        b[9, 3] = float.PositiveInfinity;
        var c = XMath.Multiply(a, b);
        for (int j = 0; j < 64; j++)
        {
            Assert.True(float.IsNaN(c[5, j]), $"row 5, column {j}");
        }

        for (int i = 0; i < 64; i++)
        {
            Assert.True(i == 5 ? float.IsNaN(c[i, 3]) : float.IsInfinity(c[i, 3]), $"row {i}, column 3");
        }

        Assert.True(float.IsFinite(c[6, 4]));

        var x = RandomComplex(64, 64, 43);
        var y = RandomComplex(64, 64, 44);
        x[5, 7] = new Complex(double.NaN, 0);
        var z = XMath.Multiply(x, y);
        for (int j = 0; j < 64; j++)
        {
            Assert.True(double.IsNaN(z[5, j].Real) && double.IsNaN(z[5, j].Imaginary), $"row 5, column {j}");
        }

        Assert.True(double.IsFinite(z[6, 4].Real) && double.IsFinite(z[6, 4].Imaginary));
    }

    /// <summary>
    /// 実部と虚部の寄与を別々に足すと打ち消す前に溢れる入力でも、複素数の積和を順に足すのと同じく有限になる。
    /// <c>a_p = (s, σ_p s)</c>・<c>b_p = (s, σ_p s)</c>（<c>σ_p = (-1)^p</c>、<c>s² = 2^1020</c>）の積は <c>2σ_p s²·i</c> で、
    /// 複素数の部分和は <c>2s²·i</c> と 0 を繰り返して結果は 0 だが、<c>Σ Re a·Re b = k·s²</c> は溢れる
    /// </summary>
    [Fact]
    public void ComplexProductDoesNotOverflowBeforeCancellation()
    {
        const int m = 64, n = 64, k = 64;
        var a = RandomComplex(m, k, 51);
        var b = RandomComplex(k, n, 52);
        double s = Math.ScaleB(1, 510);
        for (int p = 0; p < k; p++)
        {
            double sign = p % 2 == 0 ? 1 : -1;
            a[3, p] = new Complex(s, sign * s);
            b[p, 5] = new Complex(s, sign * s);
        }

        var c = XMath.Multiply(a, b);
        Assert.Equal(Complex.Zero, c[3, 5]);
        AssertComplex(a, b, c, "near overflow");
    }

    // すべての要素がオーバーフロー直前でも複素積和と一致することを検証する（各複素積は ±1e308·i、部分和は有限で結果は 0）。
    [Fact]
    public void ComplexProductDoesNotOverflowWhenEveryProductIsNearOverflow()
    {
        var a = new Matrix<Complex>(64, 64);
        var b = new Matrix<Complex>(64, 64);
        for (int i = 0; i < 64; i++)
        {
            for (int j = 0; j < 64; j++)
            {
                a[i, j] = new Complex(5e307, j % 2 == 0 ? 5e307 : -5e307);
                b[i, j] = new Complex(1, i % 2 == 0 ? 1 : -1);
            }
        }

        Complex reference = 0;
        for (int p = 0; p < 64; p++)
        {
            reference += a[0, p] * b[p, 0];
        }

        Assert.Equal(Complex.Zero, reference);

        var product = XMath.Multiply(a, b);
        for (int i = 0; i < 64; i++)
        {
            for (int j = 0; j < 64; j++)
            {
                Assert.Equal(reference, product[i, j]);
            }
        }
    }

    // 共通行列を左から掛けるバッチ積（ブロードキャスト）が各行列の積とビット単位で一致することを検証する。
    // 複素数は列数で経路（列ごとの GEMV か実行列積か）が切り替わるため、横結合時に経路が変わるサイズも含める。
    [Theory]
    [InlineData(64, 4, 64, 8)]      // 各行列は GEMV（n < 8）、横に並べると実行列の積の大きさ
    [InlineData(32, 8, 32, 64)]     // 各行列は GEMV（m·n·k < 128·32·32）、横に並べると実行列の積の大きさ
    [InlineData(64, 32, 64, 3)]     // どちらも実行列の積
    [InlineData(5, 3, 7, 4)]        // どちらも GEMV
    [InlineData(32, 8, 64, 16)]     // 各行列は GEMV、横に並べると n = 128 で実行列の積に切り替わるサイズ
    public void BroadcastProductsMatchEachMatrix(int m, int n, int k, int count)
    {
        var a = RandomComplex(m, k, 61);
        var b = Tensor<Complex>.FromColumnMajor([k, n, count], RandomComplex(k, n * count, 62).AsSpan());
        var products = XMath.Multiply(a.Broadcast(count), b);
        for (int q = 0; q < count; q++)
        {
            Assert.Equal(XMath.Multiply(a, b.GetMatrix(q)).ToArray(), products.GetMatrix(q).ToArray());
        }

        var x = RandomSingle(m, k, 63);
        var y = Tensor<float>.FromColumnMajor([k, n, count], RandomSingle(k, n * count, 64).AsSpan());
        var singles = XMath.Multiply(x.Broadcast(count), y);
        for (int q = 0; q < count; q++)
        {
            Assert.Equal(XMath.Multiply(x, y.GetMatrix(q)).ToArray(), singles.GetMatrix(q).ToArray());
        }
    }

    private static Matrix<float> RandomSingle(int rows, int columns, int seed)
    {
        var random = new Random(seed);
        var m = new Matrix<float>(rows, columns);
        var data = m.AsSpan();
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = (float)(2 * random.NextDouble() - 1);
        }

        return m;
    }

    private static Matrix<Complex> RandomComplex(int rows, int columns, int seed)
    {
        var random = new Random(seed);
        var m = new Matrix<Complex>(rows, columns);
        var data = m.AsSpan();
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = new Complex(2 * random.NextDouble() - 1, 2 * random.NextDouble() - 1);
        }

        return m;
    }

    /// <summary>各要素の差が <c>(k + 1)·2⁻²³·Σ|a||b|</c>（単精度の積和の誤差の上界の2倍）以内</summary>
    private static void AssertSingle(ReadOnlyMatrixSpan<float> a, ReadOnlyMatrixSpan<float> b, ReadOnlyMatrixSpan<float> c, string label)
    {
        int k = a.Columns;
        Assert.Equal(a.Rows, c.Rows);
        Assert.Equal(b.Columns, c.Columns);
        for (int j = 0; j < c.Columns; j++)
        {
            for (int i = 0; i < c.Rows; i++)
            {
                double sum = 0, bound = 0;
                for (int p = 0; p < k; p++)
                {
                    double t = (double)a[i, p] * b[p, j];
                    sum += t;
                    bound += Math.Abs(t);
                }

                Assert.True(Math.Abs(c[i, j] - sum) <= (k + 1) * SingleEpsilon * bound, $"{label}[{i}, {j}]: {c[i, j]} vs {sum}");
            }
        }
    }

    /// <summary>各要素の差が <c>4(k + 2)·ε·Σ(|Re a| + |Im a|)(|Re b| + |Im b|)</c> 以内（参照の誤差を含む）</summary>
    private static void AssertComplex(ReadOnlyMatrixSpan<Complex> a, ReadOnlyMatrixSpan<Complex> b, ReadOnlyMatrixSpan<Complex> c, string label)
    {
        int k = a.Columns;
        Assert.Equal(a.Rows, c.Rows);
        Assert.Equal(b.Columns, c.Columns);
        for (int j = 0; j < c.Columns; j++)
        {
            for (int i = 0; i < c.Rows; i++)
            {
                Complex sum = Complex.Zero;
                double bound = 0;
                for (int p = 0; p < k; p++)
                {
                    Complex x = a[i, p], y = b[p, j];
                    sum += x * y;
                    bound += (Math.Abs(x.Real) + Math.Abs(x.Imaginary)) * (Math.Abs(y.Real) + Math.Abs(y.Imaginary));
                }

                Assert.True((c[i, j] - sum).Magnitude <= 4 * (k + 2) * Epsilon * bound, $"{label}[{i}, {j}]: {c[i, j]} vs {sum}");
            }
        }
    }

    private static void AssertUntouched<T>(Matrix<T> c, Range rows, Range columns, T sentinel)
        where T : unmanaged, INumberBase<T>
    {
        var (r0, rn) = rows.GetOffsetAndLength(c.Rows);
        var (c0, cn) = columns.GetOffsetAndLength(c.Columns);
        for (int j = 0; j < c.Columns; j++)
        {
            for (int i = 0; i < c.Rows; i++)
            {
                bool inside = i >= r0 && i < r0 + rn && j >= c0 && j < c0 + cn;
                if (!inside)
                {
                    Assert.Equal(sentinel, c[i, j]);
                }
            }
        }
    }
}
