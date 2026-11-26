using System.Numerics;
using Tuna.LinearAlgebra;

namespace Tuna.Tests.LinearAlgebra;

/// <summary>
/// 線形代数テストの共通処理。残差や直交性はライブラリの行列積を使わず、単純な三重ループで独立に計算する。
/// </summary>
/// <remarks>
/// 合否は LAPACK のテストと同じ「残差 / (‖A‖·max(m, n)·ε)」の比で判定する。LAPACK の既定の閾値は 30。
/// </remarks>
internal static class LinearAlgebraTestUtilities
{
    internal const double Epsilon = 2.220446049250313e-16;

    /// <summary>LAPACK のテストプログラムの既定の閾値</summary>
    internal const double Threshold = 30;

    /// <summary>次数の一覧（ブロック幅 32/64 とマイクロカーネル 16×8 の境界をまたぐ値を含む）</summary>
    public static TheoryData<int> Sizes => new() { 1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 31, 33, 63, 65, 100, 129, 200 };

    /// <summary>一様乱数 [-1, 1) の行列（固定シード）</summary>
    internal static Matrix<double> Random(int rows, int columns, int seed)
    {
        var random = new Random(seed);
        var m = new Matrix<double>(rows, columns);
        var data = m.AsSpan();
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = 2 * random.NextDouble() - 1;
        }

        return m;
    }

    /// <summary>乱数の対称行列</summary>
    internal static Matrix<double> RandomSymmetric(int n, int seed)
    {
        var a = Random(n, n, seed);
        for (int j = 0; j < n; j++)
        {
            for (int i = j + 1; i < n; i++)
            {
                a[j, i] = a[i, j];
            }
        }

        return a;
    }

    /// <summary>乱数の対称正定値行列 <c>BᵀB + nI</c></summary>
    internal static Matrix<double> RandomPositiveDefinite(int n, int seed)
    {
        var b = Random(n, n, seed);
        var a = MultiplyNaive(b, b, transposeA: true);
        for (int i = 0; i < n; i++)
        {
            a[i, i] += n;
        }

        return a;
    }

    /// <summary>乱数の直交行列（乱数行列のQR分解の Q）</summary>
    internal static Matrix<double> RandomOrthogonal(int n, int seed) => Random(n, n, seed).Qr().Q;

    /// <summary>直交行列で相似変換して固有値を指定した対称行列 <c>QΛQᵀ</c></summary>
    internal static Matrix<double> SymmetricWithSpectrum(double[] eigenvalues, int seed)
    {
        int n = eigenvalues.Length;
        var q = RandomOrthogonal(n, seed);
        var a = new Matrix<double>(n, n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                double sum = 0;
                for (int k = 0; k < n; k++)
                {
                    sum += q[i, k] * eigenvalues[k] * q[j, k];
                }

                a[i, j] = sum;
                a[j, i] = sum;
            }
        }

        return a;
    }

    /// <summary>単純な三重ループによる行列積 <c>op(A)·op(B)</c></summary>
    internal static Matrix<double> MultiplyNaive(Matrix<double> a, Matrix<double> b, bool transposeA = false, bool transposeB = false)
    {
        int m = transposeA ? a.Columns : a.Rows;
        int k = transposeA ? a.Rows : a.Columns;
        int n = transposeB ? b.Rows : b.Columns;
        var c = new Matrix<double>(m, n);
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double sum = 0;
                for (int p = 0; p < k; p++)
                {
                    double aip = transposeA ? a[p, i] : a[i, p];
                    double bpj = transposeB ? b[j, p] : b[p, j];
                    sum += aip * bpj;
                }

                c[i, j] = sum;
            }
        }

        return c;
    }

    /// <summary>フロベニウスノルム（単純な実装）</summary>
    internal static double Norm(Matrix<double> a)
    {
        double scale = 0;
        for (int j = 0; j < a.Columns; j++)
        {
            for (int i = 0; i < a.Rows; i++)
            {
                scale = Math.Max(scale, Math.Abs(a[i, j]));
            }
        }

        if (scale == 0)
        {
            return 0;
        }

        double sum = 0;
        for (int j = 0; j < a.Columns; j++)
        {
            for (int i = 0; i < a.Rows; i++)
            {
                double t = a[i, j] / scale;
                sum += t * t;
            }
        }

        return scale * Math.Sqrt(sum);
    }

    /// <summary><c>‖A - B‖_F</c></summary>
    internal static double Difference(Matrix<double> a, Matrix<double> b)
    {
        Assert.Equal(a.Rows, b.Rows);
        Assert.Equal(a.Columns, b.Columns);
        var d = new Matrix<double>(a.Rows, a.Columns);
        for (int j = 0; j < a.Columns; j++)
        {
            for (int i = 0; i < a.Rows; i++)
            {
                d[i, j] = a[i, j] - b[i, j];
            }
        }

        return Norm(d);
    }

    /// <summary>直交性の誤差 <c>‖QᵀQ - I‖_F</c></summary>
    internal static double OrthogonalityError(Matrix<double> q)
    {
        var qtq = MultiplyNaive(q, q, transposeA: true);
        for (int i = 0; i < qtq.Rows; i++)
        {
            qtq[i, i] -= 1;
        }

        return Norm(qtq);
    }

    /// <summary>
    /// LAPACK 式の判定比 <c>residual / (scale·dim·ε)</c>（<c>scale = 0</c> なら 1 とみなす）。
    /// 非正規数の行列では表現できる精度そのものが落ちるので、分母は非正規数の刻み <c>dim·2⁻¹⁰⁷⁴</c> を下限とする
    /// </summary>
    internal static double Ratio(double residual, double scale, int dimension)
    {
        if (scale == 0)
        {
            scale = 1;
        }

        int dim = Math.Max(dimension, 1);
        return residual / Math.Max(scale * dim * Epsilon, dim * double.Epsilon);
    }

    /// <summary>判定比が閾値以下であることを確認する</summary>
    internal static void AssertRatio(double residual, double scale, int dimension, string what)
    {
        double ratio = Ratio(residual, scale, dimension);
        Assert.True(ratio <= Threshold, $"{what}: ratio {ratio:G4} exceeds {Threshold} (residual {residual:G4}, scale {scale:G4}, n {dimension})");
    }

    /// <summary>行列の下三角を除いた部分（上三角の外側）が厳密に0であることを確認する</summary>
    internal static void AssertUpperTriangular(Matrix<double> r)
    {
        for (int j = 0; j < r.Columns; j++)
        {
            for (int i = j + 1; i < r.Rows; i++)
            {
                Assert.Equal(0.0, r[i, j]);
            }
        }
    }

    /// <summary>行列の上三角を除いた部分が厳密に0であることを確認する</summary>
    internal static void AssertLowerTriangular(Matrix<double> l)
    {
        for (int j = 0; j < l.Columns; j++)
        {
            for (int i = 0; i < Math.Min(j, l.Rows); i++)
            {
                Assert.Equal(0.0, l[i, j]);
            }
        }
    }

    /// <summary>複素固有対の残差 <c>‖AV - VD‖_F</c>（固有ベクトルは複素数で組み立てる）</summary>
    internal static double EigenResidual(Matrix<double> a, EigenDecomposition evd)
    {
        int n = a.Rows;
        double sum = 0;
        for (int j = 0; j < n; j++)
        {
            Complex[] v = evd.GetVector(j);
            Complex lambda = evd.Values[j];
            for (int i = 0; i < n; i++)
            {
                Complex av = Complex.Zero;
                for (int k = 0; k < n; k++)
                {
                    av += a[i, k] * v[k];
                }

                Complex r = av - lambda * v[i];
                sum += r.Real * r.Real + r.Imaginary * r.Imaginary;
            }
        }

        return Math.Sqrt(sum);
    }
}
