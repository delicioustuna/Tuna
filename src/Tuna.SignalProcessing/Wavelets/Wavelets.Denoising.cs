using System.Buffers;

namespace Tuna.SignalProcessing;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む。
using Tuna.LinearAlgebra;

public static partial class Wavelets
{
    // Standard normal 75th percentile Phi^-1(0.75); MAD of N(0, sigma^2) equals sigma * Phi^-1(0.75).
    private const double NormalQuartile = 0.6744897501960817;

    // IEEE 754 binary64 machine epsilon (2^-52).
    private const double MachineEpsilon = 2.220446049250313e-16;

    /// <summary>
    /// Denoises a 1D signal by wavelet thresholding using BayesShrink and soft thresholding.
    /// <para>BayesShrink と軟判定しきい値処理により、1 次元信号のノイズを除去します。</para>
    /// </summary>
    /// <param name="x">The finite input signal to denoise. / ノイズ除去対象の有限な入力信号。</param>
    /// <param name="wavelet">The discrete wavelet to use. / 使用する離散ウェーブレット。</param>
    /// <returns>A newly allocated vector containing the denoised signal with the same length as <paramref name="x"/>. / <paramref name="x"/> と同じ長さのノイズ除去済みベクトル。</returns>
    /// <remarks>
    /// <para>
    /// Decomposes the signal to <c>max(MaxLevel(x.Length, wavelet) - 3, 1)</c> levels with <see cref="WaveletBoundary.Symmetric"/>,
    /// estimates noise standard deviation from the finest detail coefficients via <see cref="EstimateNoise(ReadOnlyVectorSpan{double})"/>,
    /// shrinks each detail level using <see cref="BayesThreshold(ReadOnlyVectorSpan{double}, double)"/> while leaving approximation coefficients unchanged,
    /// and reconstructs to the original length without clamping the output range.
    /// </para>
    /// <para>
    /// 対称境界拡張で <c>max(MaxLevel(x.Length, wavelet) - 3, 1)</c> 段まで分解し、最も細かい段の詳細係数から <see cref="EstimateNoise(ReadOnlyVectorSpan{double})"/> でノイズ標準偏差を推定します。
    /// 各段の詳細係数を <see cref="BayesThreshold(ReadOnlyVectorSpan{double}, double)"/> で縮小（近似係数は保持）したうえで元の長さに再構成します。出力値の範囲切り詰めは行いません。
    /// </para>
    /// </remarks>
    public static Vector<double> Denoise(ReadOnlyVectorSpan<double> x, DiscreteWavelet wavelet) =>
        Denoise(x, wavelet, WaveletThresholdRule.Bayes, WaveletShrinkage.Soft);

    /// <summary>
    /// Denoises a 1D signal by wavelet thresholding using the specified threshold rule and shrinkage function.
    /// <para>指定したしきい値ルールと縮小方式を用いて、1 次元信号のノイズを除去します。</para>
    /// </summary>
    /// <param name="x">The finite input signal to denoise. / ノイズ除去対象の有限な入力信号。</param>
    /// <param name="wavelet">The discrete wavelet to use. / 使用する離散ウェーブレット。</param>
    /// <param name="rule">The rule used to compute detail thresholds. / 詳細係数のしきい値を決定するルール。</param>
    /// <param name="shrinkage">The shrinkage mode applied to detail coefficients. / 詳細係数に適用する縮小方式。</param>
    /// <returns>A newly allocated vector containing the denoised signal with the same length as <paramref name="x"/>. / <paramref name="x"/> と同じ長さのノイズ除去済みベクトル。</returns>
    public static Vector<double> Denoise(ReadOnlyVectorSpan<double> x, DiscreteWavelet wavelet, WaveletThresholdRule rule, WaveletShrinkage shrinkage)
    {
        ArgumentNullException.ThrowIfNull(wavelet);
        return Denoise(x, wavelet, rule, shrinkage, DefaultDenoiseLevels(MaxLevel(x.Length, wavelet)));
    }

    /// <summary>
    /// Denoises a 1D signal by wavelet thresholding using the specified threshold rule, shrinkage function, and decomposition depth.
    /// <para>指定したしきい値ルール、縮小方式、および分解段数を用いて、1 次元信号のノイズを除去します。</para>
    /// </summary>
    /// <param name="x">The finite input signal to denoise. / ノイズ除去対象の有限な入力信号。</param>
    /// <param name="wavelet">The discrete wavelet to use. / 使用する離散ウェーブレット。</param>
    /// <param name="rule">The rule used to compute detail thresholds. / 詳細係数のしきい値を決定するルール。</param>
    /// <param name="shrinkage">The shrinkage mode applied to detail coefficients. / 詳細係数に適用する縮小方式。</param>
    /// <param name="levels">The number of decomposition levels, at least 1. / 1 以上の分解段数。</param>
    /// <returns>A newly allocated vector containing the denoised signal with the same length as <paramref name="x"/>. / <paramref name="x"/> と同じ長さのノイズ除去済みベクトル。</returns>
    public static Vector<double> Denoise(
        ReadOnlyVectorSpan<double> x, DiscreteWavelet wavelet, WaveletThresholdRule rule, WaveletShrinkage shrinkage, int levels)
    {
        var decomposition = DecomposeForDenoising(x, wavelet, rule, shrinkage, levels);
        ShrinkDetails(decomposition, rule, shrinkage, EstimateNoise(decomposition.Detail(1)), x.Length);
        return Reconstruct(decomposition);
    }

    /// <summary>
    /// Denoises a 1D signal by wavelet thresholding using the specified parameters and a known noise standard deviation.
    /// <para>既知のノイズ標準偏差と指定した設定を用いて、1 次元信号のノイズを除去します。</para>
    /// </summary>
    /// <param name="x">The finite input signal to denoise. / ノイズ除去対象の有限な入力信号。</param>
    /// <param name="wavelet">The discrete wavelet to use. / 使用する離散ウェーブレット。</param>
    /// <param name="rule">The rule used to compute detail thresholds. / 詳細係数のしきい値を決定するルール。</param>
    /// <param name="shrinkage">The shrinkage mode applied to detail coefficients. / 詳細係数に適用する縮小方式。</param>
    /// <param name="levels">The number of decomposition levels, at least 1. / 1 以上の分解段数。</param>
    /// <param name="noiseLevel">The known non-negative finite noise standard deviation sigma. / 0 以上の有限なノイズ標準偏差 σ。</param>
    /// <returns>A newly allocated vector containing the denoised signal with the same length as <paramref name="x"/>. / <paramref name="x"/> と同じ長さのノイズ除去済みベクトル。</returns>
    public static Vector<double> Denoise(
        ReadOnlyVectorSpan<double> x, DiscreteWavelet wavelet, WaveletThresholdRule rule, WaveletShrinkage shrinkage, int levels, double noiseLevel)
    {
        RequireNoiseLevel(noiseLevel);
        var decomposition = DecomposeForDenoising(x, wavelet, rule, shrinkage, levels);
        ShrinkDetails(decomposition, rule, shrinkage, noiseLevel, x.Length);
        return Reconstruct(decomposition);
    }

    /// <summary>
    /// Denoises a 2D signal matrix by wavelet thresholding using BayesShrink and soft thresholding.
    /// <para>BayesShrink と軟判定しきい値処理により、2 次元信号行列のノイズを除去します。</para>
    /// </summary>
    /// <param name="x">The finite input matrix to denoise. / ノイズ除去対象の有限な入力行列。</param>
    /// <param name="wavelet">The discrete wavelet to use. / 使用する離散ウェーブレット。</param>
    /// <returns>A newly allocated matrix containing the denoised signal with the same shape as <paramref name="x"/>. / <paramref name="x"/> と同じ形状のノイズ除去済み行列。</returns>
    /// <remarks>
    /// <para>
    /// Follows the same pipeline as <see cref="Denoise(ReadOnlyVectorSpan{double}, DiscreteWavelet)"/> using <see cref="Decompose2D(ReadOnlyMatrixSpan{double}, DiscreteWavelet, int, WaveletBoundary)"/>.
    /// Defaults to <c>max(MaxLevel(x.Rows, x.Columns, wavelet) - 3, 1)</c> levels, estimates sigma from the finest diagonal subband,
    /// and thresholds each horizontal, vertical, and diagonal subband separately (with <c>N = x.Rows * x.Columns</c> for <see cref="WaveletThresholdRule.Universal"/>).
    /// </para>
    /// <para>
    /// <see cref="Decompose2D(ReadOnlyMatrixSpan{double}, DiscreteWavelet, int, WaveletBoundary)"/> を用いて <see cref="Denoise(ReadOnlyVectorSpan{double}, DiscreteWavelet)"/> と同じ手順で処理します。
    /// 既定段数は <c>max(MaxLevel(x.Rows, x.Columns, wavelet) - 3, 1)</c>、σ は最も細かい段の対角詳細係数から推定し、
    /// 水平・垂直・対角の各サブバンドに対して個別にしきい値を適用します（<see cref="WaveletThresholdRule.Universal"/> の N は行列の全要素数）。
    /// </para>
    /// </remarks>
    public static Matrix<double> Denoise2D(ReadOnlyMatrixSpan<double> x, DiscreteWavelet wavelet) =>
        Denoise2D(x, wavelet, WaveletThresholdRule.Bayes, WaveletShrinkage.Soft);

    /// <summary>
    /// Denoises a 2D signal matrix by wavelet thresholding using the specified threshold rule and shrinkage function.
    /// <para>指定したしきい値ルールと縮小方式を用いて、2 次元信号行列のノイズを除去します。</para>
    /// </summary>
    /// <param name="x">The finite input matrix to denoise. / ノイズ除去対象の有限な入力行列。</param>
    /// <param name="wavelet">The discrete wavelet to use. / 使用する離散ウェーブレット。</param>
    /// <param name="rule">The rule used to compute subband thresholds. / サブバンドのしきい値を決定するルール。</param>
    /// <param name="shrinkage">The shrinkage mode applied to detail coefficients. / 詳細係数に適用する縮小方式。</param>
    /// <returns>A newly allocated matrix containing the denoised signal with the same shape as <paramref name="x"/>. / <paramref name="x"/> と同じ形状のノイズ除去済み行列。</returns>
    public static Matrix<double> Denoise2D(ReadOnlyMatrixSpan<double> x, DiscreteWavelet wavelet, WaveletThresholdRule rule, WaveletShrinkage shrinkage)
    {
        ArgumentNullException.ThrowIfNull(wavelet);
        return Denoise2D(x, wavelet, rule, shrinkage, DefaultDenoiseLevels(MaxLevel(x.Rows, x.Columns, wavelet)));
    }

    /// <summary>
    /// Denoises a 2D signal matrix by wavelet thresholding using the specified threshold rule, shrinkage function, and decomposition depth.
    /// <para>指定したしきい値ルール、縮小方式、および分解段数を用いて、2 次元信号行列のノイズを除去します。</para>
    /// </summary>
    /// <param name="x">The finite input matrix to denoise. / ノイズ除去対象の有限な入力行列。</param>
    /// <param name="wavelet">The discrete wavelet to use. / 使用する離散ウェーブレット。</param>
    /// <param name="rule">The rule used to compute subband thresholds. / サブバンドのしきい値を決定するルール。</param>
    /// <param name="shrinkage">The shrinkage mode applied to detail coefficients. / 詳細係数に適用する縮小方式。</param>
    /// <param name="levels">The number of decomposition levels, at least 1. / 1 以上の分解段数。</param>
    /// <returns>A newly allocated matrix containing the denoised signal with the same shape as <paramref name="x"/>. / <paramref name="x"/> と同じ形状のノイズ除去済み行列。</returns>
    public static Matrix<double> Denoise2D(
        ReadOnlyMatrixSpan<double> x, DiscreteWavelet wavelet, WaveletThresholdRule rule, WaveletShrinkage shrinkage, int levels)
    {
        var decomposition = DecomposeForDenoising2D(x, wavelet, rule, shrinkage, levels);
        ShrinkDetails(decomposition, rule, shrinkage, EstimateNoise(decomposition.Detail(1).Diagonal), (long)x.Rows * x.Columns);
        return Reconstruct(decomposition);
    }

    /// <summary>
    /// Denoises a 2D signal matrix by wavelet thresholding using the specified parameters and a known noise standard deviation.
    /// <para>既知のノイズ標準偏差と指定した設定を用いて、2 次元信号行列のノイズを除去します。</para>
    /// </summary>
    /// <param name="x">The finite input matrix to denoise. / ノイズ除去対象の有限な入力行列。</param>
    /// <param name="wavelet">The discrete wavelet to use. / 使用する離散ウェーブレット。</param>
    /// <param name="rule">The rule used to compute subband thresholds. / サブバンドのしきい値を決定するルール。</param>
    /// <param name="shrinkage">The shrinkage mode applied to detail coefficients. / 詳細係数に適用する縮小方式。</param>
    /// <param name="levels">The number of decomposition levels, at least 1. / 1 以上の分解段数。</param>
    /// <param name="noiseLevel">The known non-negative finite noise standard deviation sigma. / 0 以上の有限なノイズ標準偏差 σ。</param>
    /// <returns>A newly allocated matrix containing the denoised signal with the same shape as <paramref name="x"/>. / <paramref name="x"/> と同じ形状のノイズ除去済み行列。</returns>
    public static Matrix<double> Denoise2D(
        ReadOnlyMatrixSpan<double> x, DiscreteWavelet wavelet, WaveletThresholdRule rule, WaveletShrinkage shrinkage, int levels, double noiseLevel)
    {
        RequireNoiseLevel(noiseLevel);
        var decomposition = DecomposeForDenoising2D(x, wavelet, rule, shrinkage, levels);
        ShrinkDetails(decomposition, rule, shrinkage, noiseLevel, (long)x.Rows * x.Columns);
        return Reconstruct(decomposition);
    }

    /// <summary>
    /// Estimates the noise standard deviation from a 1D detail coefficient vector using the median absolute deviation (MAD).
    /// <para>中央絶対偏差（MAD）を用いて、1 次元詳細係数ベクトルからノイズの標準偏差を推定します。</para>
    /// </summary>
    /// <param name="detail">The detail coefficients (must not contain <see cref="double.NaN"/>). / 詳細係数（<see cref="double.NaN"/> を含まないこと）。</param>
    /// <returns>The median of the nonzero coefficient magnitudes divided by the standard normal 0.75 quantile, or 0 if all coefficients are zero. / 0 以外の係数の絶対値の中央値を標準正規分布の 0.75 分位点で割った値（すべて 0 の場合は 0）。</returns>
    public static double EstimateNoise(ReadOnlyVectorSpan<double> detail)
    {
        double[] magnitudes = ArrayPool<double>.Shared.Rent(Math.Max(detail.Length, 1));
        int count = 0;
        for (int i = 0; i < detail.Length; i++)
        {
            count = AddMagnitude(detail[i], magnitudes, count, nameof(detail));
        }

        double result = MedianOverQuartile(magnitudes, count);
        ArrayPool<double>.Shared.Return(magnitudes);
        return result;
    }

    /// <summary>
    /// Estimates the noise standard deviation from a 2D detail coefficient matrix using the median absolute deviation (MAD).
    /// <para>中央絶対偏差（MAD）を用いて、2 次元詳細係数行列からノイズの標準偏差を推定します。</para>
    /// </summary>
    /// <param name="detail">The detail coefficient matrix (must not contain <see cref="double.NaN"/>). / 詳細係数行列（<see cref="double.NaN"/> を含まないこと）。</param>
    /// <returns>The median of the nonzero coefficient magnitudes divided by the standard normal 0.75 quantile, or 0 if all coefficients are zero. / 0 以外の係数の絶対値の中央値を標準正規分布の 0.75 分位点で割った値（すべて 0 の場合は 0）。</returns>
    public static double EstimateNoise(ReadOnlyMatrixSpan<double> detail)
    {
        double[] magnitudes = ArrayPool<double>.Shared.Rent(Math.Max(detail.Rows * detail.Columns, 1));
        int count = 0;
        for (int column = 0; column < detail.Columns; column++)
        {
            for (int row = 0; row < detail.Rows; row++)
            {
                count = AddMagnitude(detail[row, column], magnitudes, count, nameof(detail));
            }
        }

        double result = MedianOverQuartile(magnitudes, count);
        ArrayPool<double>.Shared.Return(magnitudes);
        return result;
    }

    /// <summary>
    /// Computes the universal (VisuShrink) threshold <c>sigma * sqrt(2 * ln(N))</c>.
    /// <para>普遍しきい値（VisuShrink）<c>sigma * sqrt(2 * ln(N))</c> を計算します。</para>
    /// </summary>
    /// <param name="noiseLevel">The non-negative finite noise standard deviation sigma. / 0 以上の有限なノイズ標準偏差 σ。</param>
    /// <param name="count">The number of signal samples N, at least 1. / 1 以上の信号サンプル数 N。</param>
    /// <returns>The universal threshold value. / 計算された普遍しきい値。</returns>
    public static double UniversalThreshold(double noiseLevel, long count)
    {
        RequireNoiseLevel(noiseLevel);
        ArgumentOutOfRangeException.ThrowIfLessThan(count, 1);
        return noiseLevel * Math.Sqrt(2.0 * Math.Log(count));
    }

    /// <summary>
    /// Computes the BayesShrink threshold <c>sigma^2 / sqrt(max(mean(d_i^2) - sigma^2, eps))</c> for a 1D detail vector, where <c>eps = 2^-52</c>.
    /// <para>1 次元詳細係数ベクトルに対する BayesShrink のしきい値 <c>sigma^2 / sqrt(max(mean(d_i^2) - sigma^2, eps))</c>（<c>eps = 2^-52</c>）を計算します。</para>
    /// </summary>
    /// <param name="detail">The non-empty detail coefficients. / 空でない詳細係数。</param>
    /// <param name="noiseLevel">The non-negative finite noise standard deviation sigma. / 0 以上の有限なノイズ標準偏差 σ。</param>
    /// <returns>The BayesShrink threshold value. / 計算された BayesShrink のしきい値。</returns>
    public static double BayesThreshold(ReadOnlyVectorSpan<double> detail, double noiseLevel)
    {
        RequireNoiseLevel(noiseLevel);
        if (detail.Length == 0)
        {
            throw new ArgumentException("The coefficients must not be empty.", nameof(detail));
        }

        double sum = 0.0;
        for (int i = 0; i < detail.Length; i++)
        {
            sum += detail[i] * detail[i];
        }

        return BayesThreshold(sum / detail.Length, noiseLevel);
    }

    /// <summary>
    /// Computes the BayesShrink threshold for a 2D detail coefficient matrix, accumulating squared values in column-major order.
    /// <para>2 次元詳細係数行列に対する BayesShrink のしきい値を計算します（二乗和は列優先の順に集計します）。</para>
    /// </summary>
    /// <param name="detail">The non-empty detail coefficient matrix. / 空でない詳細係数行列。</param>
    /// <param name="noiseLevel">The non-negative finite noise standard deviation sigma. / 0 以上の有限なノイズ標準偏差 σ。</param>
    /// <returns>The BayesShrink threshold value. / 計算された BayesShrink のしきい値。</returns>
    public static double BayesThreshold(ReadOnlyMatrixSpan<double> detail, double noiseLevel)
    {
        RequireNoiseLevel(noiseLevel);
        if (detail.Rows == 0 || detail.Columns == 0)
        {
            throw new ArgumentException("The coefficients must not be empty.", nameof(detail));
        }

        double sum = 0.0;
        for (int column = 0; column < detail.Columns; column++)
        {
            for (int row = 0; row < detail.Rows; row++)
            {
                sum += detail[row, column] * detail[row, column];
            }
        }

        return BayesThreshold(sum / ((double)detail.Rows * detail.Columns), noiseLevel);
    }

    /// <summary>
    /// Applies thresholding to a vector of wavelet coefficients in place.
    /// <para>ウェーブレット係数のベクトルに対してインプレースでしきい値処理を適用します。</para>
    /// </summary>
    /// <param name="coefficients">The coefficients to shrink in place; <see cref="double.NaN"/> values are preserved. / インプレースで縮小する係数（<see cref="double.NaN"/> はそのまま保持されます）。</param>
    /// <param name="threshold">The non-negative, non-<see cref="double.NaN"/> threshold value. / 0 以上かつ <see cref="double.NaN"/> でないしきい値。</param>
    /// <param name="shrinkage">The shrinkage mode to apply. / 適用する縮小方式。</param>
    public static void Shrink(VectorSpan<double> coefficients, double threshold, WaveletShrinkage shrinkage)
    {
        RequireShrinkage(shrinkage);
        if (shrinkage == WaveletShrinkage.Soft)
        {
            XMath.InPlace.SoftThreshold(coefficients, threshold);
            return;
        }

        RequireThreshold(threshold);
        for (int i = 0; i < coefficients.Length; i++)
        {
            if (Math.Abs(coefficients[i]) < threshold)
            {
                coefficients[i] = 0.0;
            }
        }
    }

    /// <summary>
    /// Applies thresholding to a matrix of wavelet coefficients in place.
    /// <para>ウェーブレット係数の行列に対してインプレースでしきい値処理を適用します。</para>
    /// </summary>
    /// <param name="coefficients">The coefficient matrix to shrink in place; <see cref="double.NaN"/> values are preserved. / インプレースで縮小する係数行列（<see cref="double.NaN"/> はそのまま保持されます）。</param>
    /// <param name="threshold">The non-negative, non-<see cref="double.NaN"/> threshold value. / 0 以上かつ <see cref="double.NaN"/> でないしきい値。</param>
    /// <param name="shrinkage">The shrinkage mode to apply. / 適用する縮小方式。</param>
    public static void Shrink(MatrixSpan<double> coefficients, double threshold, WaveletShrinkage shrinkage)
    {
        RequireShrinkage(shrinkage);
        if (shrinkage == WaveletShrinkage.Soft)
        {
            XMath.InPlace.SoftThreshold(coefficients, threshold);
            return;
        }

        RequireThreshold(threshold);
        for (int column = 0; column < coefficients.Columns; column++)
        {
            for (int row = 0; row < coefficients.Rows; row++)
            {
                if (Math.Abs(coefficients[row, column]) < threshold)
                {
                    coefficients[row, column] = 0.0;
                }
            }
        }
    }

    private static int DefaultDenoiseLevels(int maxLevel) => Math.Max(maxLevel - 3, 1);

    private static WaveletDecomposition DecomposeForDenoising(
        ReadOnlyVectorSpan<double> x, DiscreteWavelet wavelet, WaveletThresholdRule rule, WaveletShrinkage shrinkage, int levels)
    {
        ArgumentNullException.ThrowIfNull(wavelet);
        RequireRule(rule);
        RequireShrinkage(shrinkage);
        ArgumentOutOfRangeException.ThrowIfLessThan(levels, 1);
        for (int i = 0; i < x.Length; i++)
        {
            if (!double.IsFinite(x[i]))
            {
                throw new ArgumentException("The signal must not contain NaN or infinity.", nameof(x));
            }
        }

        return Decompose(x, wavelet, levels, WaveletBoundary.Symmetric);
    }

    private static WaveletDecomposition2D DecomposeForDenoising2D(
        ReadOnlyMatrixSpan<double> x, DiscreteWavelet wavelet, WaveletThresholdRule rule, WaveletShrinkage shrinkage, int levels)
    {
        ArgumentNullException.ThrowIfNull(wavelet);
        RequireRule(rule);
        RequireShrinkage(shrinkage);
        ArgumentOutOfRangeException.ThrowIfLessThan(levels, 1);
        for (int column = 0; column < x.Columns; column++)
        {
            for (int row = 0; row < x.Rows; row++)
            {
                if (!double.IsFinite(x[row, column]))
                {
                    throw new ArgumentException("The matrix must not contain NaN or infinity.", nameof(x));
                }
            }
        }

        return Decompose2D(x, wavelet, levels, WaveletBoundary.Symmetric);
    }

    private static void ShrinkDetails(
        WaveletDecomposition decomposition, WaveletThresholdRule rule, WaveletShrinkage shrinkage, double noiseLevel, long count)
    {
        double universal = UniversalThreshold(noiseLevel, count);
        foreach (var detail in decomposition.Details)
        {
            Shrink(detail, rule == WaveletThresholdRule.Universal ? universal : BayesThreshold(detail, noiseLevel), shrinkage);
        }
    }

    private static void ShrinkDetails(
        WaveletDecomposition2D decomposition, WaveletThresholdRule rule, WaveletShrinkage shrinkage, double noiseLevel, long count)
    {
        double universal = UniversalThreshold(noiseLevel, count);
        foreach (var (horizontal, vertical, diagonal) in decomposition.Details)
        {
            foreach (var detail in (ReadOnlySpan<Matrix<double>>)[horizontal, vertical, diagonal])
            {
                Shrink(detail, rule == WaveletThresholdRule.Universal ? universal : BayesThreshold(detail, noiseLevel), shrinkage);
            }
        }
    }

    private static double BayesThreshold(double meanSquare, double noiseLevel)
    {
        double variance = noiseLevel * noiseLevel;
        return variance / Math.Sqrt(Math.Max(meanSquare - variance, MachineEpsilon));
    }

    private static int AddMagnitude(double value, double[] magnitudes, int count, string name)
    {
        if (double.IsNaN(value))
        {
            throw new ArgumentException("The coefficients must not contain NaN.", name);
        }

        if (value != 0.0)
        {
            magnitudes[count++] = Math.Abs(value);
        }

        return count;
    }

    private static double MedianOverQuartile(double[] magnitudes, int count)
    {
        if (count == 0)
        {
            return 0.0;
        }

        var values = magnitudes.AsSpan(0, count);
        int middle = count / 2;
        double upper = Select(values, middle);
        double median = upper;
        if ((count & 1) == 0)
        {
            // After partitioning around middle, max(values[0..middle]) is the (middle - 1)-th order statistic.
            double lower = values[0];
            for (int i = 1; i < middle; i++)
            {
                lower = Math.Max(lower, values[i]);
            }

            median = (lower + upper) / 2.0;
        }

        return median / NormalQuartile;
    }

    // Introselect (median-of-3 Hoare partitioning with sort fallback) guaranteeing O(n log n) worst-case time; requires no NaN.
    private static double Select(Span<double> values, int k)
    {
        int left = 0;
        int right = values.Length - 1;
        int budget = 2 * (32 - int.LeadingZeroCount(values.Length)) + 8;
        while (left < right)
        {
            if (budget-- == 0)
            {
                values[left..(right + 1)].Sort();
                break;
            }

            double a = values[left], b = values[left + (right - left) / 2], c = values[right];
            double pivot = Math.Max(Math.Min(a, b), Math.Min(Math.Max(a, b), c));
            int i = left;
            int j = right;
            while (i <= j)
            {
                while (values[i] < pivot)
                {
                    i++;
                }

                while (values[j] > pivot)
                {
                    j--;
                }

                if (i <= j)
                {
                    (values[i], values[j]) = (values[j], values[i]);
                    i++;
                    j--;
                }
            }

            // Invariant: values[left..=j] <= pivot, values[i..=right] >= pivot, elements in between equal pivot.
            if (k <= j)
            {
                right = j;
            }
            else if (k >= i)
            {
                left = i;
            }
            else
            {
                break;
            }
        }

        return values[k];
    }

    private static void RequireNoiseLevel(double noiseLevel)
    {
        if (!(noiseLevel >= 0.0) || double.IsPositiveInfinity(noiseLevel))
        {
            throw new ArgumentOutOfRangeException(nameof(noiseLevel), noiseLevel, "The noise level must be a finite, non-negative number.");
        }
    }

    private static void RequireThreshold(double threshold)
    {
        if (!(threshold >= 0.0))
        {
            throw new ArgumentOutOfRangeException(nameof(threshold), threshold, "The threshold must be non-negative and not NaN.");
        }
    }

    private static void RequireRule(WaveletThresholdRule rule)
    {
        if ((uint)rule > (uint)WaveletThresholdRule.Universal)
        {
            throw new ArgumentOutOfRangeException(nameof(rule));
        }
    }

    private static void RequireShrinkage(WaveletShrinkage shrinkage)
    {
        if ((uint)shrinkage > (uint)WaveletShrinkage.Hard)
        {
            throw new ArgumentOutOfRangeException(nameof(shrinkage));
        }
    }
}
