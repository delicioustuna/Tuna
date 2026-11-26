using System.Numerics;

namespace Tuna.LinearAlgebra;

/// <summary>
/// Represents the eigendecomposition <c>AV = VD</c> of a general real square matrix.
/// <para>一般の実正方行列の固有値分解 <c>AV = VD</c> を表します。</para>
/// </summary>
/// <remarks>
/// <para>Eigenvectors are stored in <see cref="Vectors"/> using the LAPACK real packed convention: if eigenvalue <c>j</c> is real, column <c>j</c> is its eigenvector; if <c>λⱼ = a + bi</c> and <c>λⱼ₊₁ = a - bi</c> (<c>b &gt; 0</c>) form a complex conjugate pair, columns <c>j</c> and <c>j + 1</c> store the real and imaginary parts so that <c>vⱼ = V[:, j] + i·V[:, j + 1]</c> and <c>vⱼ₊₁ = V[:, j] - i·V[:, j + 1]</c>. Each eigenvector is normalized to unit Euclidean norm with its largest-magnitude component real.</para>
/// <para>固有ベクトルは LAPACK の実表現形式で <see cref="Vectors"/> に格納されます。第 <c>j</c> 固有値が実数なら第 <c>j</c> 列が対応する固有ベクトルであり、複素共役対 <c>λⱼ = a + bi</c>・<c>λⱼ₊₁ = a - bi</c>（<c>b &gt; 0</c>）では第 <c>j</c> 列が実部、第 <c>j + 1</c> 列が虚部を表します。各固有ベクトルはユークリッドノルム 1 に正規化され、複素ベクトルは絶対値最大の成分が実数になるよう位相を揃えています。</para>
/// </remarks>
/// <seealso href="https://www.netlib.org/lapack/explore-html/d4/d68/group__geev.html"/>
public sealed class EigenDecomposition
{
    private EigenDecomposition(Complex[] values, Matrix<double>? vectors)
    {
        Values = values;
        Vectors = vectors;
    }

    /// <summary>
    /// Gets the computed complex eigenvalues (complex conjugate pairs appear consecutively with the positive imaginary part first).
    /// <para>計算された複素固有値を取得します（複素共役対は虚部が正のものが先に連続して並びます）。</para>
    /// </summary>
    public Complex[] Values { get; }

    /// <summary>
    /// Gets the real-packed eigenvector matrix, or <see langword="null"/> if eigenvectors were not computed.
    /// <para>実形式で格納された固有ベクトル行列を取得します（固有ベクトルを計算しなかった場合は <see langword="null"/>）。</para>
    /// </summary>
    public Matrix<double>? Vectors { get; }

    /// <summary>
    /// Extracts the complex eigenvector corresponding to the eigenvalue at <paramref name="index"/>.
    /// <para><paramref name="index"/> 番目の固有値に対応する複素固有ベクトルを取り出します。</para>
    /// </summary>
    /// <param name="index">The zero-based eigenvalue index. / 0 から始まる固有値インデックス。</param>
    /// <returns>A new complex array of length <c>n</c> containing the eigenvector. / 固有ベクトルを格納した長さ <c>n</c> の新しい複素数配列。</returns>
    /// <exception cref="InvalidOperationException">Eigenvectors were not computed. / 固有ベクトルが計算されていない場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="index"/> is out of range. / <paramref name="index"/> が範囲外の場合。</exception>
    public Complex[] GetVector(int index)
    {
        if (Vectors is null)
        {
            throw new InvalidOperationException("Eigenvectors were not computed.");
        }

        int n = Vectors.Rows;
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(index, n);
        var v = Vectors.Data;
        var result = new Complex[n];
        double im = Values[index].Imaginary;
        if (im == 0)
        {
            for (int i = 0; i < n; i++)
            {
                result[i] = v[i + index * n];
            }
        }
        else
        {
            int re = im > 0 ? index : index - 1;
            double sign = im > 0 ? 1 : -1;
            for (int i = 0; i < n; i++)
            {
                result[i] = new Complex(v[i + re * n], sign * v[i + (re + 1) * n]);
            }
        }

        return result;
    }

    // Empirical crossover below which parallelization overhead outweighs gains.
    private const int SequentialBelow = 160;

    /// <summary>
    /// Computes the eigendecomposition of a general real square matrix.
    /// <para>一般の実正方行列の固有値分解を計算します。</para>
    /// </summary>
    /// <param name="a">The input square matrix (unmodified when <paramref name="overwriteInput"/> is <see langword="false"/>). / 入力正方行列（<paramref name="overwriteInput"/> が <see langword="false"/> の場合は変更されません）。</param>
    /// <param name="computeVectors"><see langword="true"/> to also compute right eigenvectors; <see langword="false"/> to compute eigenvalues only. / 右固有ベクトルも計算する場合は <see langword="true"/>、固有値のみの場合は <see langword="false"/>。</param>
    /// <param name="overwriteInput"><see langword="true"/> to reuse <paramref name="a"/>'s backing array as workspace (destroying its contents); <see langword="false"/> to copy <paramref name="a"/>. / <paramref name="a"/> の格納領域をワークスペースとして再利用する（元の内容は破壊されます）場合は <see langword="true"/>。</param>
    /// <returns>The computed eigendecomposition. / 固有値分解結果。</returns>
    /// <exception cref="ArgumentException"><paramref name="a"/> is not square. / <paramref name="a"/> が正方行列でない場合。</exception>
    public static EigenDecomposition Compute(Matrix<double> a, bool computeVectors = true, bool overwriteInput = false)
    {
        if (!a.IsSquare)
        {
            throw new ArgumentException("The matrix must be square.", nameof(a));
        }

        int n = a.Rows;
        var values = new Complex[n];
        if (n == 0)
        {
            return new EigenDecomposition(values, computeVectors ? new Matrix<double>(0, 0) : null);
        }

        return ComputeCore(n, overwriteInput ? a.Data : (double[])a.Data.Clone(), computeVectors);
    }

    /// <summary>
    /// Computes the eigendecomposition of a general real square matrix view.
    /// <para>一般の実正方行列ビューの固有値分解を計算します。</para>
    /// </summary>
    /// <param name="a">The input square matrix view (not modified). / 入力正方行列ビュー（変更されません）。</param>
    /// <param name="computeVectors"><see langword="true"/> to also compute right eigenvectors; <see langword="false"/> to compute eigenvalues only. / 右固有ベクトルも計算する場合は <see langword="true"/>、固有値のみの場合は <see langword="false"/>。</param>
    /// <returns>The computed eigendecomposition. / 固有値分解結果。</returns>
    /// <exception cref="ArgumentException"><paramref name="a"/> is not square. / <paramref name="a"/> が正方行列でない場合。</exception>
    public static EigenDecomposition Compute(ReadOnlyMatrixSpan<double> a, bool computeVectors = true)
    {
        if (a.Rows != a.Columns)
        {
            throw new ArgumentException("The matrix must be square.", nameof(a));
        }

        int n = a.Rows;
        if (n == 0)
        {
            return new EigenDecomposition([], computeVectors ? new Matrix<double>(0, 0) : null);
        }

        var h = GC.AllocateUninitializedArray<double>(checked(n * n));
        StridedKernels.CopyIntoFactorStorage(a, h);
        return ComputeCore(n, h, computeVectors);
    }

    private static EigenDecomposition ComputeCore(int n, double[] h, bool computeVectors)
    {
        var values = new Complex[n];
        using var sequential = Parallelism.Sequential(n < SequentialBelow);
        double scale = Scaling.ScaleGeneral(n, n, h, n);
        var balance = new double[n];
        HessenbergReduction.Balance(n, h, n, balance, out int ilo, out int ihi);

        var tau = new double[n];
        HessenbergReduction.Reduce(n, ilo, ihi, h, n, tau);
        double[]? z = null;
        if (computeVectors)
        {
            z = new double[n * n];
            HessenbergReduction.FormQ(n, ilo, ihi, h, n, tau, z, n);
        }

        // Zero out the Householder reflectors stored below the first subdiagonal to form the Hessenberg matrix.
        for (int j = 0; j < n - 2; j++)
        {
            h.AsSpan(j + 2 + j * n, n - j - 2).Clear();
        }

        var wr = new double[n];
        var wi = new double[n];
        bool ok = HessenbergQr.Solve(n, ilo, ihi, h, wr, wi, z ?? Span<double>.Empty, computeVectors);
        if (!ok)
        {
            wr.AsSpan().Fill(double.NaN);
            wi.AsSpan().Fill(double.NaN);
            z?.AsSpan().Fill(double.NaN);
        }

        double inv = 1.0 / scale;
        for (int i = 0; i < n; i++)
        {
            values[i] = new Complex(wr[i] * inv, wi[i] * inv);
        }

        if (z is null)
        {
            return new EigenDecomposition(values, null);
        }

        if (ok)
        {
            HessenbergReduction.BackTransform(n, n, balance, ilo, ihi, z, n);
            Normalize(n, wi, z);
        }

        return new EigenDecomposition(values, Matrix<double>.Wrap(n, n, z));
    }

    // Normalizes eigenvectors following LAPACK dgeev conventions.
    private static void Normalize(int n, ReadOnlySpan<double> wi, Span<double> z)
    {
        for (int j = 0; j < n; j++)
        {
            if (wi[j] == 0)
            {
                var col = z.Slice(j * n, n);
                double norm = Level1Kernels.Norm2(col);
                if (norm > 0)
                {
                    Level1Kernels.Scale(1.0 / norm, col);
                }
            }
            else if (wi[j] > 0 && j + 1 < n)
            {
                var re = z.Slice(j * n, n);
                var im = z.Slice((j + 1) * n, n);
                double norm = Householder.Hypot(Level1Kernels.Norm2(re), Level1Kernels.Norm2(im));
                if (norm > 0)
                {
                    Level1Kernels.Scale(1.0 / norm, re);
                    Level1Kernels.Scale(1.0 / norm, im);
                }

                // Rotate phase so the largest-magnitude component is real.
                int k = 0;
                double max = -1;
                for (int i = 0; i < n; i++)
                {
                    double m2 = re[i] * re[i] + im[i] * im[i];
                    if (m2 > max)
                    {
                        max = m2;
                        k = i;
                    }
                }

                double r = Householder.Hypot(re[k], im[k]);
                if (r > 0)
                {
                    double cs = re[k] / r;
                    double sn = im[k] / r;

                    // (re + i·im)·(cs - i·sn)
                    Level1Kernels.Rotate(re, im, cs, sn);
                    im[k] = 0;
                }

                j++;
            }
        }
    }
}
