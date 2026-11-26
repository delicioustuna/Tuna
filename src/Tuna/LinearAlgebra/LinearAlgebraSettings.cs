namespace Tuna.LinearAlgebra;

/// <summary>
/// Provides global execution settings for linear algebra decompositions and kernels.
/// <para>線形代数演算および行列分解のグローバルな実行設定を提供します。</para>
/// </summary>
public static class LinearAlgebraSettings
{
    private static int maxDegreeOfParallelism = CpuTopology.PhysicalCores;

    private static int twoStageReduction = (int)TwoStageReduction.Auto;

    /// <summary>
    /// Gets or sets the maximum number of threads used by a single linear algebra operation.
    /// <para>1 回の線形代数演算で使用する最大スレッド数を取得または設定します。</para>
    /// </summary>
    /// <value>The maximum degree of parallelism. Defaults to the number of physical CPU cores available to the process (or <see cref="Environment.ProcessorCount"/> if unavailable); set to 1 to force single-threaded execution. / 最大並列度。既定値はプロセスが利用可能な物理コア数（取得できない場合は <see cref="Environment.ProcessorCount"/>）。1 を設定すると常に単一スレッドで実行します。</value>
    /// <remarks>
    /// <para>The effective thread count is capped at <see cref="Environment.ProcessorCount"/> observed at process startup. Small matrices run single-threaded regardless of this setting, and concurrent callers that cannot acquire worker threads fall back to single-threaded execution.</para>
    /// <para>Because parallel reduction order affects floating-point rounding in decompositions such as Cholesky, eigendecomposition, and SVD, set an explicit value if bit-for-bit reproducibility across machines with different core counts is required.</para>
    /// <para>実際に使用されるスレッド数はプロセス起動時の <see cref="Environment.ProcessorCount"/> が上限となります。小さな行列は設定にかかわらず単一スレッドで実行され、複数スレッドから同時に呼び出されて作業スレッドを確保できない場合も単一スレッドにフォールバックします。</para>
    /// <para>コレスキー分解・固有値分解・特異値分解の結果は並列数によって丸め誤差の範囲で変化するため、コア数の異なる環境間でビット単位の再現性が必要な場合は同じ値を明示的に設定してください。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">The value is less than 1. / 設定値が 1 未満の場合。</exception>
    public static int MaxDegreeOfParallelism
    {
        get => Volatile.Read(ref maxDegreeOfParallelism);
        set
        {
            ArgumentOutOfRangeException.ThrowIfLessThan(value, 1);
            Volatile.Write(ref maxDegreeOfParallelism, value);
        }
    }

    /// <summary>
    /// Gets or sets the policy controlling two-stage reduction (dense to band to tridiagonal or bidiagonal) in symmetric eigendecomposition and SVD.
    /// <para>対称行列の固有値分解および特異値分解で、二段階縮約（密行列から帯行列を経由して三重対角・二重対角行列へ縮約）を使用する条件を取得または設定します。</para>
    /// </summary>
    /// <value>The two-stage reduction policy. Defaults to <see cref="TwoStageReduction.Auto"/>. / 二段階縮約の適用ポリシー。既定値は <see cref="TwoStageReduction.Auto"/>。</value>
    /// <remarks>
    /// <para>Two-stage reduction relies heavily on cache-friendly matrix multiplications and accelerates large matrices, at the cost of an extra back-transformation step when eigenvectors or singular vectors are requested. With <see cref="TwoStageReduction.Auto"/>, it is enabled only when vectors are not computed and the matrix order is at least 2048.</para>
    /// <para>二段階縮約はキャッシュ効率の高い行列積を中心に構成されるため大きな行列で高速ですが、固有ベクトルや特異ベクトルを求める場合は帯行列からの逆変換コストが追加されます。<see cref="TwoStageReduction.Auto"/> では、ベクトルを求めないかつ次数が 2048 以上の場合にのみ使用します。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">The value is not a defined <see cref="Tuna.LinearAlgebra.TwoStageReduction"/> member. / 定義されていない <see cref="Tuna.LinearAlgebra.TwoStageReduction"/> の値が指定された場合。</exception>
    public static TwoStageReduction TwoStageReduction
    {
        get => (TwoStageReduction)Volatile.Read(ref twoStageReduction);
        set
        {
            if (!Enum.IsDefined(value))
            {
                throw new ArgumentOutOfRangeException(nameof(value));
            }

            Volatile.Write(ref twoStageReduction, (int)value);
        }
    }

    internal static bool UseTwoStage(int n, int bandwidth, bool vectors) => TwoStageReduction switch
    {
        TwoStageReduction.Never => false,
        TwoStageReduction.Always => n > 2 * bandwidth,
        _ => !vectors && n >= TwoStageThreshold,
    };

    internal const int TwoStageThreshold = 2048;
}

/// <summary>
/// Specifies when two-stage reduction is used by <see cref="LinearAlgebraSettings.TwoStageReduction"/>.
/// <para><see cref="LinearAlgebraSettings.TwoStageReduction"/> で二段階縮約を使用する条件を指定します。</para>
/// </summary>
public enum TwoStageReduction
{
    /// <summary>
    /// Uses two-stage reduction only when vectors are not computed and the matrix order is at least 2048.
    /// <para>固有ベクトルまたは特異ベクトルを求めない、次数 2048 以上の行列でのみ使用します。</para>
    /// </summary>
    Auto,

    /// <summary>
    /// Disables two-stage reduction and always uses single-stage reduction.
    /// <para>二段階縮約を使用せず、常に一段階縮約を実行します。</para>
    /// </summary>
    Never,

    /// <summary>
    /// Always uses two-stage reduction whenever the matrix order exceeds twice the band width, including when vectors are computed.
    /// <para>ベクトルを求める場合も含め、次数が帯幅の 2 倍を超える行列で常に二段階縮約を使用します。</para>
    /// </summary>
    Always,
}
