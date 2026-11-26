using Tuna.LinearAlgebra;

namespace Tuna.Probabilistic.Inference;

// 正定値共分散行列 Σ（対角または密行列）とそのコレスキー因子 L（Σ = LLᵀ、対角の場合は √Σᵢᵢ）を保持する。
// ランダムウォーク・メトロポリス法の提案分布およびハミルトニアン法の逆計量 M⁻¹ で共有する。
internal sealed class CovarianceShape
{
    private readonly int n;
    private readonly double[] covariance;   // 密行列なら n × n の列優先、対角なら長さ n
    private readonly double[] factor;       // 密行列なら下三角 L（列優先）、対角なら √Σᵢᵢ

    public CovarianceShape(int dimension, bool dense)
    {
        n = dimension;
        IsDense = dense;
        covariance = new double[dense ? n * n : n];
        factor = new double[dense ? n * n : n];
        for (int i = 0; i < n; i++)
        {
            covariance[dense ? i * n + i : i] = 1;
            factor[dense ? i * n + i : i] = 1;
        }
    }

    public bool IsDense { get; }

    // Σ を更新する（正定値でなければ false を返し、元の状態を保持する）
    public bool TrySet(ReadOnlySpan<double> values)
    {
        if (!IsDense)
        {
            foreach (double v in values)
            {
                if (!(v > 0 && v < double.PositiveInfinity))
                {
                    return false;
                }
            }

            values.CopyTo(covariance);
            for (int i = 0; i < n; i++)
            {
                factor[i] = Math.Sqrt(covariance[i]);
            }

            return true;
        }

        foreach (double v in values)
        {
            if (!double.IsFinite(v))
            {
                return false;
            }
        }

        var matrix = Matrix<double>.FromColumnMajor(n, n, values);
        if (!CholeskyDecomposition.TryCompute(matrix, out CholeskyDecomposition cholesky))
        {
            return false;
        }

        values.CopyTo(covariance);
        Matrix<double> l = cholesky.L;
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < n; i++)
            {
                factor[j * n + i] = i >= j ? l[i, j] : 0;
            }
        }

        return true;
    }

    public void WriteDense(Span<double> destination)
    {
        if (IsDense)
        {
            covariance.CopyTo(destination);
            return;
        }

        destination[..(n * n)].Clear();
        for (int i = 0; i < n; i++)
        {
            destination[i * n + i] = covariance[i];
        }
    }

    // y = L·z（z ~ N(0, I) のとき y ~ N(0, Σ)）
    public void MultiplyFactor(ReadOnlySpan<double> z, Span<double> y)
    {
        if (!IsDense)
        {
            for (int i = 0; i < n; i++)
            {
                y[i] = factor[i] * z[i];
            }

            return;
        }

        for (int i = 0; i < n; i++)
        {
            double sum = 0;
            for (int j = 0; j <= i; j++)
            {
                sum += factor[j * n + i] * z[j];
            }

            y[i] = sum;
        }
    }

    // z = L⁻¹·y（‖z‖² = yᵀΣ⁻¹y）
    public void SolveFactor(ReadOnlySpan<double> y, Span<double> z)
    {
        if (!IsDense)
        {
            for (int i = 0; i < n; i++)
            {
                z[i] = y[i] / factor[i];
            }

            return;
        }

        // Lz = y を前進代入で解く
        for (int i = 0; i < n; i++)
        {
            double sum = y[i];
            for (int j = 0; j < i; j++)
            {
                sum -= factor[j * n + i] * z[j];
            }

            z[i] = sum / factor[i * n + i];
        }
    }

    // p = L⁻ᵀ·z（z ~ N(0, I) のとき p ~ N(0, Σ⁻¹)）
    public void SolveFactorTranspose(ReadOnlySpan<double> z, Span<double> p)
    {
        if (!IsDense)
        {
            for (int i = 0; i < n; i++)
            {
                p[i] = z[i] / factor[i];
            }

            return;
        }

        // Lᵀp = z を後退代入で解く（Lᵀ の (i, j) 成分は L の (j, i) 成分）
        for (int i = n - 1; i >= 0; i--)
        {
            double sum = z[i];
            for (int j = i + 1; j < n; j++)
            {
                sum -= factor[i * n + j] * p[j];
            }

            p[i] = sum / factor[i * n + i];
        }
    }

    // v = Σ·p（ハミルトニアン法の速度 M⁻¹p）
    public void Multiply(ReadOnlySpan<double> p, Span<double> v)
    {
        if (!IsDense)
        {
            for (int i = 0; i < n; i++)
            {
                v[i] = covariance[i] * p[i];
            }

            return;
        }

        for (int i = 0; i < n; i++)
        {
            double sum = 0;
            for (int j = 0; j < n; j++)
            {
                sum += covariance[j * n + i] * p[j];
            }

            v[i] = sum;
        }
    }
}
