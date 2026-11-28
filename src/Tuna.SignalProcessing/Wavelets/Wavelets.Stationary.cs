using System.Buffers;
using System.Runtime.InteropServices;

namespace Tuna.SignalProcessing;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む。
using Tuna.LinearAlgebra;

public static partial class Wavelets
{
    /// <summary>
    /// Computes the undecimated stationary wavelet transform (SWT) of a signal.
    /// <para>入力信号の定常ウェーブレット変換（SWT）を計算します。</para>
    /// </summary>
    /// <param name="x">The input signal of length n &gt;= 1. / 長さ n &gt;= 1 の入力信号。</param>
    /// <param name="wavelet">The discrete wavelet to use (orthogonal or biorthogonal). / 使用する離散ウェーブレット（直交・双直交のいずれも可）。</param>
    /// <param name="levels">The number of decomposition levels, from 0 to <c>floor(log2(n))</c>. / 0 以上 <c>floor(log2(n))</c> 以下の分解段数。</param>
    /// <returns>
    /// A <c>(levels + 1) x n</c> matrix whose row 0 holds the coarsest approximation and subsequent rows hold detail coefficients ordered from coarsest to finest level. /
    /// 行 0 に最も粗い近似係数、後続の行に粗い段から細かい段への詳細係数を並べた <c>(levels + 1) x n</c> の行列。
    /// </returns>
    /// <remarks>
    /// <para>
    /// At level j (with stride <c>s = 2^(j - 1)</c> and filter length L), circular filtering computes
    /// <c>a_j[t] = Sum_l lo_l * a_(j-1)[(t + s * (L/2 - l)) mod n]</c> and <c>d_j[t] = Sum_l hi_l * a_(j-1)[(t + s * (L/2 - l)) mod n]</c>,
    /// where <c>lo</c> and <c>hi</c> are <see cref="DiscreteWavelet.DecompositionLow"/> and <see cref="DiscreteWavelet.DecompositionHigh"/>.
    /// Because coefficients are not decimated, every level has length n and the transform is circularly shift-equivariant.
    /// </para>
    /// <para>
    /// Row 0 is the level-<paramref name="levels"/> approximation and row <c>levels + 1 - j</c> is the level-j detail (matching <see cref="WaveletDecomposition"/> ordering).
    /// Each column corresponds to a single sample time. Any length <c>n &gt;= 1</c> is supported without requiring divisibility by <c>2^levels</c>.
    /// </para>
    /// <para>
    /// 第 j 段（間隔 <c>s = 2^(j - 1)</c>、フィルタ長 L）では、周期 n の巡回畳み込みにより
    /// <c>a_j[t] = Sum_l lo_l * a_(j-1)[(t + s * (L/2 - l)) mod n]</c>、<c>d_j[t] = Sum_l hi_l * a_(j-1)[(t + s * (L/2 - l)) mod n]</c> を計算します
    /// （<c>lo</c>・<c>hi</c> は <see cref="DiscreteWavelet.DecompositionLow"/> と <see cref="DiscreteWavelet.DecompositionHigh"/>）。
    /// 間引きを行わないため各段の長さは n であり、入力の巡回シフトに対して係数も同様にシフトします。
    /// </para>
    /// <para>
    /// 行 0 が第 <paramref name="levels"/> 段の近似係数、行 <c>levels + 1 - j</c> が第 j 段の詳細係数（<see cref="WaveletDecomposition"/> と同じ粗い段から細かい段の順）です。
    /// 各列が 1 サンプル時刻に対応します。<c>n</c> が <c>2^levels</c> で割り切れる必要はなく、任意の長さ <c>n &gt;= 1</c> を受け付けます。
    /// </para>
    /// </remarks>
    public static Matrix<double> Stationary(ReadOnlyVectorSpan<double> x, DiscreteWavelet wavelet, int levels)
    {
        RequireStationary(x.Length, wavelet, levels, orthogonal: false);
        var result = Matrix<double>.Uninitialized(levels + 1, x.Length);
        StationaryCore(x, wavelet, levels, maximalOverlap: false, result);
        return result;
    }

    /// <summary>
    /// Computes the undecimated stationary wavelet transform (SWT) and writes the coefficients into <paramref name="destination"/>.
    /// <para>定常ウェーブレット変換（SWT）を計算し、係数を <paramref name="destination"/> へ書き込みます。</para>
    /// </summary>
    /// <param name="x">The input signal of length n &gt;= 1. / 長さ n &gt;= 1 の入力信号。</param>
    /// <param name="wavelet">The discrete wavelet to use (orthogonal or biorthogonal). / 使用する離散ウェーブレット（直交・双直交のいずれも可）。</param>
    /// <param name="destination">The destination matrix with n columns and <c>levels + 1</c> rows. May alias <paramref name="x"/>. / 列数 n、行数 <c>levels + 1</c> の出力先行列。<paramref name="x"/> と重なっても構いません。</param>
    /// <returns><paramref name="destination"/> containing the coarsest approximation in row 0 and details ordered from coarsest to finest level in subsequent rows. / 行 0 に最も粗い近似係数、後続の行に粗い段から細かい段への詳細係数を書き込んだ <paramref name="destination"/>。</returns>
    public static MatrixSpan<double> Stationary(ReadOnlyVectorSpan<double> x, DiscreteWavelet wavelet, MatrixSpan<double> destination)
    {
        RequireStationaryDestination(x.Length, destination);
        RequireStationary(x.Length, wavelet, destination.Rows - 1, orthogonal: false);
        StationaryCore(x, wavelet, destination.Rows - 1, maximalOverlap: false, destination);
        return destination;
    }

    /// <summary>
    /// Reconstructs a signal from its stationary wavelet transform (SWT) coefficients.
    /// <para>定常ウェーブレット変換（SWT）の係数行列から信号を再構成します。</para>
    /// </summary>
    /// <param name="coefficients">The SWT coefficient matrix produced by <see cref="Stationary(ReadOnlyVectorSpan{double}, DiscreteWavelet, int)"/>. / <see cref="Stationary(ReadOnlyVectorSpan{double}, DiscreteWavelet, int)"/> で得た係数行列。</param>
    /// <param name="wavelet">The discrete wavelet used for the forward transform. / 順変換に使用した離散ウェーブレット。</param>
    /// <returns>A newly allocated vector containing the reconstructed signal, with length equal to the column count of <paramref name="coefficients"/>. / 係数行列の列数と同じ長さの再構成信号を格納した新しいベクトル。</returns>
    /// <remarks>
    /// <para>
    /// Each level computes <c>a_(j-1)[u] = Sum_l (lo_l * a_j[u + s * (L/2 - 1 - l)] + hi_l * d_j[u + s * (L/2 - 1 - l)]) / 2</c>
    /// using <see cref="DiscreteWavelet.ReconstructionLow"/> and <see cref="DiscreteWavelet.ReconstructionHigh"/>, averaging both even- and odd-phase inverse shifts.
    /// </para>
    /// <para>
    /// 各段で <see cref="DiscreteWavelet.ReconstructionLow"/> と <see cref="DiscreteWavelet.ReconstructionHigh"/> を用い、
    /// <c>a_(j-1)[u] = Sum_l (lo_l * a_j[u + s * (L/2 - 1 - l)] + hi_l * d_j[u + s * (L/2 - 1 - l)]) / 2</c> により偶奇位相の逆変換を平均して元の信号を復元します。
    /// </para>
    /// </remarks>
    public static Vector<double> InverseStationary(ReadOnlyMatrixSpan<double> coefficients, DiscreteWavelet wavelet)
    {
        RequireStationaryCoefficients(coefficients, wavelet, orthogonal: false);
        var result = Vector<double>.Uninitialized(coefficients.Columns);
        InverseStationaryCore(coefficients, wavelet, maximalOverlap: false, result);
        return result;
    }

    /// <summary>
    /// Reconstructs a signal from its stationary wavelet transform (SWT) coefficients into <paramref name="destination"/>.
    /// <para>定常ウェーブレット変換（SWT）の係数行列から信号を再構成し、<paramref name="destination"/> へ書き込みます。</para>
    /// </summary>
    /// <param name="coefficients">The SWT coefficient matrix. / 定常ウェーブレット変換の係数行列。</param>
    /// <param name="wavelet">The discrete wavelet used for the forward transform. / 順変換に使用した離散ウェーブレット。</param>
    /// <param name="destination">The destination span with length equal to the column count of <paramref name="coefficients"/>. May alias <paramref name="coefficients"/>. / 係数行列の列数と同じ長さの出力先スパン。入力と重なっても構いません。</param>
    /// <returns><paramref name="destination"/> containing the reconstructed signal. / 再構成した信号を書き込んだ <paramref name="destination"/>。</returns>
    public static VectorSpan<double> InverseStationary(
        ReadOnlyMatrixSpan<double> coefficients, DiscreteWavelet wavelet, VectorSpan<double> destination)
    {
        RequireStationaryCoefficients(coefficients, wavelet, orthogonal: false);
        RequireLength(coefficients.Columns, destination.Length, nameof(destination));
        InverseStationaryCore(coefficients, wavelet, maximalOverlap: false, destination);
        return destination;
    }

    /// <summary>
    /// Computes the maximal overlap discrete wavelet transform (MODWT) of a signal.
    /// <para>入力信号の最大重複離散ウェーブレット変換（MODWT）を計算します。</para>
    /// </summary>
    /// <param name="x">The input signal of length n &gt;= 1. / 長さ n &gt;= 1 の入力信号。</param>
    /// <param name="wavelet">The orthogonal discrete wavelet to use. / 使用する直交離散ウェーブレット。</param>
    /// <param name="levels">The number of decomposition levels, from 0 to <c>floor(log2(n))</c>. / 0 以上 <c>floor(log2(n))</c> 以下の分解段数。</param>
    /// <returns>
    /// A <c>(levels + 1) x n</c> matrix whose row 0 holds the coarsest scaling coefficients and subsequent rows hold wavelet coefficients ordered from coarsest to finest level. /
    /// 行 0 に最も粗いスケーリング係数、後続の行に粗い段から細かい段へのウェーブレット係数を並べた <c>(levels + 1) x n</c> の行列。
    /// </returns>
    /// <remarks>
    /// <para>
    /// Uses scaled reconstruction filters <c>g_tilde = g / sqrt(2)</c> and <c>h_tilde = h / sqrt(2)</c> (Percival–Walden convention, where <c>h_l = (-1)^l * g_(L-1-l)</c>)
    /// in the circular pyramid <c>W_j[t] = Sum_l h_tilde_l * V_(j-1)[(t - 2^(j-1) * l) mod n]</c> and <c>V_j[t] = Sum_l g_tilde_l * V_(j-1)[(t - 2^(j-1) * l) mod n]</c>.
    /// </para>
    /// <para>
    /// Only orthogonal wavelets are accepted. Energy is preserved across bands: <c>||x||^2 = Sum_j ||W_j||^2 + ||V_levels||^2</c>.
    /// Row ordering matches <see cref="Stationary(ReadOnlyVectorSpan{double}, DiscreteWavelet, int)"/>, while filter scaling and tap direction follow MODWT conventions.
    /// </para>
    /// <para>
    /// 再構成フィルタを <c>sqrt(2)</c> で割った <c>g_tilde = g / sqrt(2)</c>・<c>h_tilde = h / sqrt(2)</c>（Percival–Walden の定義）を用い、
    /// 周期 n の巡回ピラミッド <c>W_j[t] = Sum_l h_tilde_l * V_(j-1)[(t - 2^(j-1) * l) mod n]</c>、<c>V_j[t] = Sum_l g_tilde_l * V_(j-1)[(t - 2^(j-1) * l) mod n]</c> を計算します。
    /// </para>
    /// <para>
    /// 直交ウェーブレットのみ受け付け、エネルギー保存 <c>||x||^2 = Sum_j ||W_j||^2 + ||V_levels||^2</c> を満たします。
    /// 行の並びは <see cref="Stationary(ReadOnlyVectorSpan{double}, DiscreteWavelet, int)"/> と同じですが、フィルタのスケールと時刻の向きが異なります。
    /// </para>
    /// </remarks>
    public static Matrix<double> MaximalOverlap(ReadOnlyVectorSpan<double> x, DiscreteWavelet wavelet, int levels)
    {
        RequireStationary(x.Length, wavelet, levels, orthogonal: true);
        var result = Matrix<double>.Uninitialized(levels + 1, x.Length);
        StationaryCore(x, wavelet, levels, maximalOverlap: true, result);
        return result;
    }

    /// <summary>
    /// Computes the maximal overlap discrete wavelet transform (MODWT) and writes the coefficients into <paramref name="destination"/>.
    /// <para>最大重複離散ウェーブレット変換（MODWT）を計算し、係数を <paramref name="destination"/> へ書き込みます。</para>
    /// </summary>
    /// <param name="x">The input signal of length n &gt;= 1. / 長さ n &gt;= 1 の入力信号。</param>
    /// <param name="wavelet">The orthogonal discrete wavelet to use. / 使用する直交離散ウェーブレット。</param>
    /// <param name="destination">The destination matrix with n columns and <c>levels + 1</c> rows. May alias <paramref name="x"/>. / 列数 n、行数 <c>levels + 1</c> の出力先行列。<paramref name="x"/> と重なっても構いません。</param>
    /// <returns><paramref name="destination"/> containing the coarsest approximation in row 0 and details ordered from coarsest to finest level in subsequent rows. / 行 0 に最も粗い近似係数、後続の行に粗い段から細かい段への詳細係数を書き込んだ <paramref name="destination"/>。</returns>
    public static MatrixSpan<double> MaximalOverlap(ReadOnlyVectorSpan<double> x, DiscreteWavelet wavelet, MatrixSpan<double> destination)
    {
        RequireStationaryDestination(x.Length, destination);
        RequireStationary(x.Length, wavelet, destination.Rows - 1, orthogonal: true);
        StationaryCore(x, wavelet, destination.Rows - 1, maximalOverlap: true, destination);
        return destination;
    }

    /// <summary>
    /// Reconstructs a signal from its maximal overlap discrete wavelet transform (MODWT) coefficients.
    /// <para>最大重複離散ウェーブレット変換（MODWT）の係数行列から信号を再構成します。</para>
    /// </summary>
    /// <param name="coefficients">The MODWT coefficient matrix produced by <see cref="MaximalOverlap(ReadOnlyVectorSpan{double}, DiscreteWavelet, int)"/>. / <see cref="MaximalOverlap(ReadOnlyVectorSpan{double}, DiscreteWavelet, int)"/> で得た係数行列。</param>
    /// <param name="wavelet">The orthogonal discrete wavelet used for the forward transform. / 順変換に使用した直交離散ウェーブレット。</param>
    /// <returns>A newly allocated vector containing the reconstructed signal, with length equal to the column count of <paramref name="coefficients"/>. / 係数行列の列数と同じ長さの再構成信号を格納した新しいベクトル。</returns>
    public static Vector<double> InverseMaximalOverlap(ReadOnlyMatrixSpan<double> coefficients, DiscreteWavelet wavelet)
    {
        RequireStationaryCoefficients(coefficients, wavelet, orthogonal: true);
        var result = Vector<double>.Uninitialized(coefficients.Columns);
        InverseStationaryCore(coefficients, wavelet, maximalOverlap: true, result);
        return result;
    }

    /// <summary>
    /// Reconstructs a signal from its maximal overlap discrete wavelet transform (MODWT) coefficients into <paramref name="destination"/>.
    /// <para>最大重複離散ウェーブレット変換（MODWT）の係数行列から信号を再構成し、<paramref name="destination"/> へ書き込みます。</para>
    /// </summary>
    /// <param name="coefficients">The MODWT coefficient matrix. / MODWT の係数行列。</param>
    /// <param name="wavelet">The orthogonal discrete wavelet used for the forward transform. / 順変換に使用した直交離散ウェーブレット。</param>
    /// <param name="destination">The destination span with length equal to the column count of <paramref name="coefficients"/>. May alias <paramref name="coefficients"/>. / 係数行列の列数と同じ長さの出力先スパン。入力と重なっても構いません。</param>
    /// <returns><paramref name="destination"/> containing the reconstructed signal. / 再構成した信号を書き込んだ <paramref name="destination"/>。</returns>
    public static VectorSpan<double> InverseMaximalOverlap(
        ReadOnlyMatrixSpan<double> coefficients, DiscreteWavelet wavelet, VectorSpan<double> destination)
    {
        RequireStationaryCoefficients(coefficients, wavelet, orthogonal: true);
        RequireLength(coefficients.Columns, destination.Length, nameof(destination));
        InverseStationaryCore(coefficients, wavelet, maximalOverlap: true, destination);
        return destination;
    }

    /// <summary>
    /// Computes the MODWT multiresolution analysis (MRA) components of a signal.
    /// <para>MODWT による多重解像度解析（MRA）の成分を計算します。</para>
    /// </summary>
    /// <param name="x">The input signal of length n &gt;= 1. / 長さ n &gt;= 1 の入力信号。</param>
    /// <param name="wavelet">The orthogonal discrete wavelet to use. / 使用する直交離散ウェーブレット。</param>
    /// <param name="levels">The number of decomposition levels, from 0 to <c>floor(log2(n))</c>. / 0 以上 <c>floor(log2(n))</c> 以下の分解段数。</param>
    /// <returns>A <c>(levels + 1) x n</c> matrix whose row 0 holds the smooth component and subsequent rows hold detail components ordered from coarsest to finest level. / 行 0 に平滑成分、後続の行に粗い段から細かい段への詳細成分を並べた <c>(levels + 1) x n</c> の行列。</returns>
    /// <remarks>
    /// <para>
    /// Each detail component <c>D_j</c> is the inverse MODWT retaining only <c>W_j</c>, and the smooth component <c>S_levels</c> retains only <c>V_levels</c>.
    /// All components are zero-phase aligned with the input signal and sum back to <paramref name="x"/>.
    /// </para>
    /// <para>
    /// 各詳細成分 <c>D_j</c> は <c>W_j</c> のみを残した逆 MODWT、平滑成分 <c>S_levels</c> は <c>V_levels</c> のみを残した逆 MODWT です。
    /// 各成分は入力信号と位相ずれなく時刻が揃い、すべての行の和は元の信号 <paramref name="x"/> に一致します。
    /// </para>
    /// </remarks>
    public static Matrix<double> MaximalOverlapMultiresolution(ReadOnlyVectorSpan<double> x, DiscreteWavelet wavelet, int levels)
    {
        RequireStationary(x.Length, wavelet, levels, orthogonal: true);
        int n = x.Length;
        int size = (levels + 1) * n;
        var (low, high) = MaximalOverlapFilters(wavelet);
        double[] buffer = ArrayPool<double>.Shared.Rent(size + n);
        var rows = buffer.AsSpan(0, size);
        var next = buffer.AsSpan(size, n);
        StationaryRows(x, wavelet, levels, maximalOverlap: true, rows);
        for (int row = 0; row <= levels; row++)
        {
            // Synthesize row 0 (V_J) with g_tilde only; row r >= 1 (W_j, j = J + 1 - r) starts with h_tilde then uses g_tilde.
            int level = row == 0 ? levels : levels + 1 - row;
            var current = rows.Slice(row * n, n);
            for (int j = level; j >= 1; j--)
            {
                long step = 1L << (j - 1);
                var filter = row != 0 && j == level ? high : low;
                next.Clear();
                CircularSynthesis(current, filter, default, default, 0, step, next);
                next.CopyTo(current);
            }
        }

        var result = Matrix<double>.Uninitialized(levels + 1, n);
        CopyRows(rows, levels + 1, n, result);
        ArrayPool<double>.Shared.Return(buffer);
        return result;
    }

    private static void StationaryCore(
        ReadOnlyVectorSpan<double> x, DiscreteWavelet wavelet, int levels, bool maximalOverlap, MatrixSpan<double> destination)
    {
        int n = x.Length;
        double[] buffer = ArrayPool<double>.Shared.Rent((levels + 1) * n);
        var rows = buffer.AsSpan(0, (levels + 1) * n);
        StationaryRows(x, wavelet, levels, maximalOverlap, rows);
        CopyRows(rows, levels + 1, n, destination);
        ArrayPool<double>.Shared.Return(buffer);
    }

    private static void StationaryRows(ReadOnlyVectorSpan<double> x, DiscreteWavelet wavelet, int levels, bool maximalOverlap, Span<double> rows)
    {
        int n = x.Length;
        int f = wavelet.FilterLength;
        var (low, high) = maximalOverlap ? MaximalOverlapFilters(wavelet) : (wavelet.DecompositionLow.ToArray(), wavelet.DecompositionHigh.ToArray());
        double[] buffer = ArrayPool<double>.Shared.Rent(2 * n);
        var approximation = buffer.AsSpan(0, n);
        var next = buffer.AsSpan(n, n);
        for (int t = 0; t < n; t++)
        {
            approximation[t] = x[t];
        }

        for (int j = 1; j <= levels; j++)
        {
            // SWT accesses x[t + s * (L/2 - l)]; MODWT accesses x[t - s * l].
            long step = 1L << (j - 1);
            long offset = maximalOverlap ? 0 : step * (f / 2);
            CircularAnalysis(approximation, low, high, offset, -step, next, rows.Slice((levels + 1 - j) * n, n));
            next.CopyTo(approximation);
        }

        approximation.CopyTo(rows[..n]);
        ArrayPool<double>.Shared.Return(buffer);
    }

    private static void InverseStationaryCore(
        ReadOnlyMatrixSpan<double> coefficients, DiscreteWavelet wavelet, bool maximalOverlap, VectorSpan<double> destination)
    {
        int levels = coefficients.Rows - 1;
        int n = coefficients.Columns;
        int f = wavelet.FilterLength;
        double[] low, high;
        if (maximalOverlap)
        {
            (low, high) = MaximalOverlapFilters(wavelet);
        }
        else
        {
            // Scaling by 0.5 adjusts only the exponent and is exact in IEEE 754 floating-point arithmetic.
            low = wavelet.ReconstructionLow.ToArray();
            high = wavelet.ReconstructionHigh.ToArray();
            for (int l = 0; l < f; l++)
            {
                low[l] *= 0.5;
                high[l] *= 0.5;
            }
        }

        int size = (levels + 1) * n;
        double[] buffer = ArrayPool<double>.Shared.Rent(size + 2 * n);
        var rows = buffer.AsSpan(0, size);
        var approximation = buffer.AsSpan(size, n);
        var next = buffer.AsSpan(size + n, n);
        CopyToRows(coefficients, rows);
        rows[..n].CopyTo(approximation);
        for (int j = levels; j >= 1; j--)
        {
            // SWT accesses a[u + s * (L/2 - 1 - l)] (transpose of analysis); MODWT accesses a[u + s * l].
            long step = 1L << (j - 1);
            var detail = rows.Slice((levels + 1 - j) * n, n);
            next.Clear();
            if (maximalOverlap)
            {
                CircularSynthesis(detail, high, approximation, low, 0, step, next);
            }
            else
            {
                CircularSynthesis(approximation, low, detail, high, step * (f / 2 - 1), -step, next);
            }

            next.CopyTo(approximation);
        }

        for (int t = 0; t < n; t++)
        {
            destination[t] = approximation[t];
        }

        ArrayPool<double>.Shared.Return(buffer);
    }

    private static void CopyRows(ReadOnlySpan<double> rows, int count, int n, MatrixSpan<double> destination)
    {
        if (destination.TryGetDense(out var dense))
        {
            for (int t = 0; t < n; t++)
            {
                var column = dense.Slice(t * count, count);
                for (int row = 0; row < column.Length; row++)
                {
                    column[row] = rows[row * n + t];
                }
            }

            return;
        }

        for (int t = 0; t < n; t++)
        {
            var column = destination.Column(t);
            for (int row = 0; row < count; row++)
            {
                column[row] = rows[row * n + t];
            }
        }
    }

    private static void CopyToRows(ReadOnlyMatrixSpan<double> source, Span<double> rows)
    {
        int count = source.Rows;
        int n = source.Columns;
        if (source.TryGetDense(out var dense))
        {
            for (int t = 0; t < n; t++)
            {
                var column = dense.Slice(t * count, count);
                for (int row = 0; row < column.Length; row++)
                {
                    rows[row * n + t] = column[row];
                }
            }

            return;
        }

        for (int t = 0; t < n; t++)
        {
            var column = source.Column(t);
            for (int row = 0; row < count; row++)
            {
                rows[row * n + t] = column[row];
            }
        }
    }

    private static void CircularAnalysis(
        ReadOnlySpan<double> x, ReadOnlySpan<double> low, ReadOnlySpan<double> high, long offset, long stride, Span<double> a, Span<double> d)
    {
        int n = x.Length;
        a.Clear();
        d.Clear();
        for (int t0 = 0; t0 < n; t0 += Block)
        {
            int count = Math.Min(Block, n - t0);
            for (int l = 0; l < low.Length; l++)
            {
                // Split circular wrap-around at n - r into at most two contiguous slices.
                int r = (int)LongModulo(offset + stride * l + t0, n);
                int head = Math.Min(count, n - r);
                MultiplyAdd(x.Slice(r, head), low[l], high[l], a.Slice(t0, head), d.Slice(t0, head));
                if (head < count)
                {
                    MultiplyAdd(x[..(count - head)], low[l], high[l], a.Slice(t0 + head, count - head), d.Slice(t0 + head, count - head));
                }
            }
        }
    }

    private static void CircularSynthesis(
        ReadOnlySpan<double> first, ReadOnlySpan<double> firstFilter, ReadOnlySpan<double> second, ReadOnlySpan<double> secondFilter,
        long offset, long stride, Span<double> output)
    {
        int n = output.Length;
        for (int t0 = 0; t0 < n; t0 += Block)
        {
            int count = Math.Min(Block, n - t0);
            for (int l = 0; l < firstFilter.Length; l++)
            {
                int r = (int)LongModulo(offset + stride * l + t0, n);
                int head = Math.Min(count, n - r);
                Accumulate(output.Slice(t0, head), first.Slice(r, head), firstFilter[l], second.IsEmpty ? default : second.Slice(r, head), second.IsEmpty ? 0 : secondFilter[l]);
                if (head < count)
                {
                    int tail = count - head;
                    Accumulate(output.Slice(t0 + head, tail), first[..tail], firstFilter[l], second.IsEmpty ? default : second[..tail], second.IsEmpty ? 0 : secondFilter[l]);
                }
            }
        }
    }

    private static void Accumulate(Span<double> output, ReadOnlySpan<double> a, double factor, ReadOnlySpan<double> b, double other)
    {
        if (b.IsEmpty)
        {
            AddProduct(output, a, factor);
        }
        else
        {
            AddProducts(output, a, factor, b, other);
        }
    }

    // Separate multiply and add rounding guarantees bit-identical results across SIMD widths and scalar tail.
    private static void AddProduct(Span<double> output, ReadOnlySpan<double> a, double factor)
    {
        int i = 0;
        int width = System.Numerics.Vector<double>.Count;
        if (System.Numerics.Vector.IsHardwareAccelerated && output.Length >= width)
        {
            var v = new System.Numerics.Vector<double>(factor);
            ref double ro = ref MemoryMarshal.GetReference(output);
            ref double ra = ref MemoryMarshal.GetReference(a);
            for (; i <= output.Length - width; i += width)
            {
                var sum = System.Numerics.Vector.LoadUnsafe(ref ro, (nuint)i) + System.Numerics.Vector.LoadUnsafe(ref ra, (nuint)i) * v;
                System.Numerics.Vector.StoreUnsafe(sum, ref ro, (nuint)i);
            }
        }

        for (; i < output.Length; i++)
        {
            output[i] += a[i] * factor;
        }
    }

    private static (double[] Low, double[] High) MaximalOverlapFilters(DiscreteWavelet wavelet)
    {
        double root = XMath.Constants.Sqrt2;
        var low = wavelet.ReconstructionLow.ToArray();
        var high = wavelet.ReconstructionHigh.ToArray();
        for (int l = 0; l < low.Length; l++)
        {
            low[l] /= root;
            high[l] /= root;
        }

        return (low, high);
    }

    private static long LongModulo(long value, int modulus)
    {
        long r = value % modulus;
        return r < 0 ? r + modulus : r;
    }

    private static int StationaryMaxLevel(int length) => length < 1 ? 0 : 31 - int.LeadingZeroCount(length);

    private static void RequireStationary(int length, DiscreteWavelet wavelet, int levels, bool orthogonal)
    {
        ArgumentNullException.ThrowIfNull(wavelet);
        if (length == 0)
        {
            throw new ArgumentException("The signal must not be empty.", "x");
        }

        if (orthogonal && !wavelet.IsOrthogonal)
        {
            throw new ArgumentException($"The maximal overlap transform needs an orthogonal wavelet, but {wavelet.Name} is biorthogonal.", nameof(wavelet));
        }

        ArgumentOutOfRangeException.ThrowIfNegative(levels);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(levels, StationaryMaxLevel(length));
    }

    private static void RequireStationaryDestination(int length, MatrixSpan<double> destination)
    {
        if (destination.Rows < 1 || destination.Columns != length)
        {
            throw new ArgumentException(
                $"The destination must have at least one row and {length} columns, but it is {destination.Rows} x {destination.Columns}.",
                nameof(destination));
        }
    }

    private static void RequireStationaryCoefficients(ReadOnlyMatrixSpan<double> coefficients, DiscreteWavelet wavelet, bool orthogonal)
    {
        if (coefficients.Rows < 1 || coefficients.Columns < 1)
        {
            throw new ArgumentException("The coefficients must have at least one row and one column.", nameof(coefficients));
        }

        RequireStationary(coefficients.Columns, wavelet, coefficients.Rows - 1, orthogonal);
    }
}
