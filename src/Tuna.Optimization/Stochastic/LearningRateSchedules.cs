namespace Tuna.Optimization.Stochastic;

/// <summary>
/// Represents a learning-rate schedule that returns a multiplier for the base learning rate at each update step.
/// <para>各更新ステップにおいて基準学習率に掛ける倍率を返す学習率スケジュールを表します。</para>
/// </summary>
/// <remarks>
/// <para>Implementations should be stateless functions of <c>step</c> (returning the same value for the same step). Optimizers invoke <see cref="Multiplier"/> once before each update.</para>
/// <para>実装は <c>step</c> のみで決まるステートレスな関数にしてください（同じ <c>step</c> に対して常に同じ値を返します）。オプティマイザーは各更新の前に1回呼び出します。</para>
/// </remarks>
public interface ILearningRateSchedule
{
    /// <summary>
    /// Computes the learning-rate multiplier for the specified update step.
    /// <para>指定した更新ステップにおける学習率の倍率を計算します。</para>
    /// </summary>
    /// <param name="step">The zero-based update step index. / 0 から始まる更新ステップ番号。</param>
    /// <returns>A non-negative finite multiplier applied to the base learning rate. / 基準学習率に掛ける 0 以上の有限の倍率。</returns>
    double Multiplier(long step);
}

/// <summary>
/// Implements a step decay learning-rate schedule <c>γ^⌊t / s⌋</c>.
/// <para>一定間隔ごとに倍率を掛けるステップ減衰スケジュール <c>γ^⌊t / s⌋</c> を表します。</para>
/// </summary>
public sealed class StepDecaySchedule : ILearningRateSchedule
{
    /// <summary>
    /// Initializes a step decay schedule.
    /// <para>ステップ減衰スケジュールを初期化します。</para>
    /// </summary>
    /// <param name="stepSize">The number of steps <c>s</c> between decays (at least 1). / 倍率を更新する間隔 <c>s</c>（1 以上）。</param>
    /// <param name="factor">The multiplicative factor γ applied at each interval (in <c>(0, 1]</c>). / 間隔ごとに掛ける倍率 γ（<c>(0, 1]</c> の範囲）。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="stepSize"/> is less than 1, or <paramref name="factor"/> is not in <c>(0, 1]</c>. / <paramref name="stepSize"/> が 1 未満、または <paramref name="factor"/> が <c>(0, 1]</c> の範囲外の場合。</exception>
    public StepDecaySchedule(long stepSize, double factor)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(stepSize, 1L);
        if (!(factor > 0 && factor <= 1))
        {
            throw new ArgumentOutOfRangeException(nameof(factor), "The factor must lie in (0, 1].");
        }

        StepSize = stepSize;
        Factor = factor;
    }

    /// <summary>
    /// Gets the number of steps between multiplier drops.
    /// <para>倍率を減衰させる間隔を取得します。</para>
    /// </summary>
    public long StepSize { get; }

    /// <summary>
    /// Gets the decay factor applied at each interval.
    /// <para>間隔ごとに掛ける倍率を取得します。</para>
    /// </summary>
    public double Factor { get; }

    /// <inheritdoc/>
    public double Multiplier(long step) => Math.Pow(Factor, Math.Max(step, 0) / StepSize);
}

/// <summary>
/// Implements a continuous exponential decay learning-rate schedule <c>r^(t / s)</c>.
/// <para>連続的な指数減衰スケジュール <c>r^(t / s)</c> を表します（<c>s</c> 回の更新で倍率 <c>r</c>）。</para>
/// </summary>
public sealed class ExponentialDecaySchedule : ILearningRateSchedule
{
    /// <summary>
    /// Initializes an exponential decay schedule.
    /// <para>指数減衰スケジュールを初期化します。</para>
    /// </summary>
    /// <param name="rate">The decay factor <c>r</c> reached after <paramref name="decaySteps"/> updates (in <c>(0, 1]</c>). / <paramref name="decaySteps"/> 回の更新で掛かる倍率 <c>r</c>（<c>(0, 1]</c> の範囲）。</param>
    /// <param name="decaySteps">The number of steps <c>s</c> over which <paramref name="rate"/> applies (at least 1; default is 1). / 倍率 <c>r</c> に対応する更新回数 <c>s</c>（1 以上、既定値は 1）。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="rate"/> is not in <c>(0, 1]</c>, or <paramref name="decaySteps"/> is less than 1. / <paramref name="rate"/> が <c>(0, 1]</c> の範囲外、または <paramref name="decaySteps"/> が 1 未満の場合。</exception>
    public ExponentialDecaySchedule(double rate, long decaySteps = 1)
    {
        if (!(rate > 0 && rate <= 1))
        {
            throw new ArgumentOutOfRangeException(nameof(rate), "The rate must lie in (0, 1].");
        }

        ArgumentOutOfRangeException.ThrowIfLessThan(decaySteps, 1L);
        Rate = rate;
        DecaySteps = decaySteps;
    }

    /// <summary>
    /// Gets the decay multiplier applied per <see cref="DecaySteps"/> updates.
    /// <para><see cref="DecaySteps"/> 回の更新ごとに掛かる倍率を取得します。</para>
    /// </summary>
    public double Rate { get; }

    /// <summary>
    /// Gets the number of updates corresponding to <see cref="Rate"/>.
    /// <para>倍率 <see cref="Rate"/> に対応する更新回数を取得します。</para>
    /// </summary>
    public long DecaySteps { get; }

    /// <inheritdoc/>
    public double Multiplier(long step) => Math.Pow(Rate, (double)Math.Max(step, 0) / DecaySteps);
}

/// <summary>
/// Implements an inverse-time decay learning-rate schedule <c>1 / (1 + λt)</c>.
/// <para>時間に反比例して減衰する学習率スケジュール <c>1 / (1 + λt)</c> を表します。</para>
/// </summary>
public sealed class InverseTimeDecaySchedule : ILearningRateSchedule
{
    /// <summary>
    /// Initializes an inverse-time decay schedule.
    /// <para>逆時間減衰スケジュールを初期化します。</para>
    /// </summary>
    /// <param name="decay">The decay coefficient λ (non-negative and finite). / 減衰係数 λ（0 以上の有限値）。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="decay"/> is negative or non-finite. / <paramref name="decay"/> が負または有限値でない場合。</exception>
    public InverseTimeDecaySchedule(double decay)
    {
        if (!(decay >= 0 && double.IsFinite(decay)))
        {
            throw new ArgumentOutOfRangeException(nameof(decay), "The decay must be non-negative and finite.");
        }

        Decay = decay;
    }

    /// <summary>
    /// Gets the decay coefficient λ.
    /// <para>減衰係数 λ を取得します。</para>
    /// </summary>
    public double Decay { get; }

    /// <inheritdoc/>
    public double Multiplier(long step) => 1.0 / (1.0 + Decay * Math.Max(step, 0));
}

/// <summary>
/// Implements a cosine annealing learning-rate schedule, optionally with periodic restarts.
/// <para>余弦曲線に沿って学習率を減衰させるコサインアニーリングスケジュールを表します（周期的リスタートにも対応）。</para>
/// </summary>
/// <remarks>
/// <para>At step <c>t</c>, the multiplier is <c>m + (1 − m)(1 + cos(πt / T)) / 2</c>, where <c>T</c> is <see cref="Period"/> and <c>m</c> is <see cref="MinimumMultiplier"/>. After <c>T</c> steps, the multiplier resets to 1 if <see cref="Restart"/> is <see langword="true"/>, or stays at <c>m</c> otherwise.</para>
/// <para>ステップ <c>t</c> での倍率は <c>m + (1 − m)(1 + cos(πt / T)) / 2</c>（<c>T</c> は <see cref="Period"/>、<c>m</c> は <see cref="MinimumMultiplier"/>）です。<c>T</c> ステップ後は、<see cref="Restart"/> が <see langword="true"/> なら 1 に戻り、それ以外は <c>m</c> を維持します。</para>
/// </remarks>
public sealed class CosineAnnealingSchedule : ILearningRateSchedule
{
    /// <summary>
    /// Initializes a cosine annealing schedule.
    /// <para>コサインアニーリングスケジュールを初期化します。</para>
    /// </summary>
    /// <param name="period">The number of steps <c>T</c> in one half-cosine decay period (at least 1). / 減衰周期のステップ数 <c>T</c>（1 以上）。</param>
    /// <param name="minimumMultiplier">The minimum multiplier <c>m</c> in <c>[0, 1]</c> (default is 0). / 最小倍率 <c>m</c>（<c>[0, 1]</c> の範囲、既定値は 0）。</param>
    /// <param name="restart"><see langword="true"/> to restart to 1 every <paramref name="period"/> steps; <see langword="false"/> (the default) to clamp at <paramref name="minimumMultiplier"/>. / <paramref name="period"/> ステップごとに倍率を 1 に戻す場合は <see langword="true"/>、<paramref name="minimumMultiplier"/> に固定する場合は <see langword="false"/>（既定値）。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="period"/> is less than 1, or <paramref name="minimumMultiplier"/> is not in <c>[0, 1]</c>. / <paramref name="period"/> が 1 未満、または <paramref name="minimumMultiplier"/> が <c>[0, 1]</c> の範囲外の場合。</exception>
    public CosineAnnealingSchedule(long period, double minimumMultiplier = 0, bool restart = false)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(period, 1L);
        if (!(minimumMultiplier >= 0 && minimumMultiplier <= 1))
        {
            throw new ArgumentOutOfRangeException(nameof(minimumMultiplier), "The minimum multiplier must lie in [0, 1].");
        }

        Period = period;
        MinimumMultiplier = minimumMultiplier;
        Restart = restart;
    }

    /// <summary>
    /// Gets the decay period in update steps.
    /// <para>減衰周期のステップ数を取得します。</para>
    /// </summary>
    public long Period { get; }

    /// <summary>
    /// Gets the minimum learning-rate multiplier.
    /// <para>最小倍率を取得します。</para>
    /// </summary>
    public double MinimumMultiplier { get; }

    /// <summary>
    /// Gets a value indicating whether the schedule restarts every <see cref="Period"/> steps.
    /// <para><see cref="Period"/> ステップごとに倍率を 1 に戻すかどうかを取得します。</para>
    /// </summary>
    public bool Restart { get; }

    /// <inheritdoc/>
    public double Multiplier(long step)
    {
        step = Math.Max(step, 0);
        long t = Restart ? step % Period : Math.Min(step, Period);
        return MinimumMultiplier + (1 - MinimumMultiplier) * 0.5 * (1 + Math.Cos(Math.PI * t / Period));
    }
}

/// <summary>
/// Implements linear warmup <c>(t + 1) / W</c> over the first <c>W</c> steps, followed by an optional schedule.
/// <para>最初の <c>W</c> ステップで倍率を <c>(t + 1) / W</c> と線形に 1 まで立ち上げ、その後は指定したスケジュールに従うウォームアップスケジュールを表します。</para>
/// </summary>
public sealed class LinearWarmupSchedule : ILearningRateSchedule
{
    /// <summary>
    /// Initializes a linear warmup schedule.
    /// <para>線形ウォームアップスケジュールを初期化します。</para>
    /// </summary>
    /// <param name="warmupSteps">The number of warmup steps <c>W</c> (at least 1). / ウォームアップのステップ数 <c>W</c>（1 以上）。</param>
    /// <param name="after">Optional schedule to invoke after warmup (with step index shifted to start at 0), or <see langword="null"/> for constant multiplier 1. / ウォームアップ後に適用するスケジュール（ステップ番号は 0 から数え直されます）。<see langword="null"/> の場合は倍率 1 を維持します。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="warmupSteps"/> is less than 1. / <paramref name="warmupSteps"/> が 1 未満の場合。</exception>
    public LinearWarmupSchedule(long warmupSteps, ILearningRateSchedule? after = null)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(warmupSteps, 1L);
        WarmupSteps = warmupSteps;
        After = after;
    }

    /// <summary>
    /// Gets the number of linear warmup steps.
    /// <para>ウォームアップのステップ数を取得します。</para>
    /// </summary>
    public long WarmupSteps { get; }

    /// <summary>
    /// Gets the schedule applied after warmup completes, or <see langword="null"/> for constant multiplier 1.
    /// <para>ウォームアップ完了後に適用されるスケジュール（未指定時は <see langword="null"/>）を取得します。</para>
    /// </summary>
    public ILearningRateSchedule? After { get; }

    /// <inheritdoc/>
    public double Multiplier(long step)
    {
        step = Math.Max(step, 0);
        if (step < WarmupSteps)
        {
            return (double)(step + 1) / WarmupSteps;
        }

        return After?.Multiplier(step - WarmupSteps) ?? 1.0;
    }
}
