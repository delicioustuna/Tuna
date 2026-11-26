namespace Tuna.Optimization.Stochastic;

/// <summary>
/// Provides common configuration options for stochastic gradient optimizers.
/// <para>確率的勾配法オプティマイザーに共通する設定オプションの基底クラスです。</para>
/// </summary>
/// <remarks>
/// <para>Option values are validated by the optimizer constructor.</para>
/// <para>各設定値の妥当性はオプティマイザーのコンストラクターで検証されます。</para>
/// </remarks>
public abstract class StochasticGradientOptions
{
    private protected StochasticGradientOptions(double learningRate)
    {
        LearningRate = learningRate;
    }

    /// <summary>
    /// Gets the base learning rate η.
    /// <para>基準学習率 η を取得します。</para>
    /// </summary>
    /// <value>A positive finite number. Each update multiplies this value by <see cref="Schedule"/> if specified. / 正の有限値。各更新では <see cref="Schedule"/> の倍率が掛けられます。</value>
    public double LearningRate { get; init; }

    /// <summary>
    /// Gets the optional learning-rate schedule.
    /// <para>学習率スケジュールを取得します。</para>
    /// </summary>
    /// <value>A schedule instance, or <see langword="null"/> (the default) for a constant learning rate. / 学習率スケジュール、または一定の学習率を使用する <see langword="null"/>（既定値）。</value>
    public ILearningRateSchedule? Schedule { get; init; }

    /// <summary>
    /// Gets the weight decay coefficient λ.
    /// <para>重み減衰係数 λ を取得します。</para>
    /// </summary>
    /// <value>A non-negative finite number. The default is 0. Applied as L2 regularization (<c>g + λx</c>) when <see cref="DecoupledWeightDecay"/> is <see langword="false"/>, or directly as <c>x ← (1 − ηλ)x</c> when <see langword="true"/>. / 0 以上の有限値。既定値は 0。<see cref="DecoupledWeightDecay"/> が <see langword="false"/> の場合は勾配に <c>λx</c> を加算し、<see langword="true"/> の場合は更新とは独立に <c>x ← (1 − ηλ)x</c> を適用します。</value>
    public double WeightDecay { get; init; }

    /// <summary>
    /// Gets a value indicating whether weight decay is decoupled from the gradient update (as in AdamW).
    /// <para>重み減衰を勾配更新から切り離して適用するかどうか（AdamW 方式）を取得します。</para>
    /// </summary>
    /// <value><see langword="false"/> (the default) to add <c>λx</c> to the gradient; <see langword="true"/> to multiply parameters by <c>1 − ηλ</c> separately from the gradient step. / 勾配に <c>λx</c> を加算する場合は <see langword="false"/>（既定値）、勾配更新とは別にパラメーターへ <c>1 − ηλ</c> を掛ける場合は <see langword="true"/>。</value>
    public bool DecoupledWeightDecay { get; init; }

    /// <summary>
    /// Gets the optional maximum Euclidean norm for gradient clipping.
    /// <para>勾配クリッピングにおける L2 ノルム上限を取得します。</para>
    /// </summary>
    /// <value>A positive value, or <see langword="null"/> (the default) to disable clipping. Gradients exceeding this norm are rescaled to this norm before weight decay is applied. / 正の値、またはクリッピングを無効にする <see langword="null"/>（既定値）。上限を超える勾配は重み減衰の適用前に向きを保ってこのノルムへスケーリングされます。</value>
    public double? GradientClipNorm { get; init; }
}

/// <summary>
/// Configures options for <see cref="StochasticGradientDescent"/> (default learning rate 0.01).
/// <para><see cref="StochasticGradientDescent"/> の設定オプションを表します（基準学習率の既定値は 0.01）。</para>
/// </summary>
/// <remarks>
/// <para>Option values are validated by the optimizer constructor.</para>
/// <para>設定値はオプティマイザーのコンストラクターで検証されます。</para>
/// </remarks>
public sealed class StochasticGradientDescentOptions : StochasticGradientOptions
{
    /// <summary>
    /// Initializes a new instance with default SGD options.
    /// <para>既定の SGD 設定でインスタンスを初期化します。</para>
    /// </summary>
    public StochasticGradientDescentOptions() : base(0.01)
    {
    }

    /// <summary>
    /// Gets the momentum coefficient μ.
    /// <para>モーメンタム係数 μ を取得します。</para>
    /// </summary>
    /// <value>A value in <c>[0, 1)</c>. The default is 0 (no momentum). / <c>[0, 1)</c> の値。既定値は 0（モーメンタムなし）。</value>
    public double Momentum { get; init; }

    /// <summary>
    /// Gets a value indicating whether Nesterov momentum is enabled.
    /// <para>Nesterov モーメンタムを使用するかどうかを取得します。</para>
    /// </summary>
    /// <value><see langword="true"/> to use Nesterov momentum when <see cref="Momentum"/> is positive; otherwise, <see langword="false"/> (the default). / <see cref="Momentum"/> が正のときに Nesterov モーメンタムを使用する場合は <see langword="true"/>、それ以外は <see langword="false"/>（既定値）。</value>
    public bool Nesterov { get; init; }
}

/// <summary>
/// Configures options for <see cref="AdaGrad"/> (default learning rate 0.01).
/// <para><see cref="AdaGrad"/> の設定オプションを表します（基準学習率の既定値は 0.01）。</para>
/// </summary>
/// <remarks>
/// <para>Option values are validated by the optimizer constructor.</para>
/// <para>設定値はオプティマイザーのコンストラクターで検証されます。</para>
/// </remarks>
public sealed class AdaGradOptions : StochasticGradientOptions
{
    /// <summary>
    /// Initializes a new instance with default AdaGrad options.
    /// <para>既定の AdaGrad 設定でインスタンスを初期化します。</para>
    /// </summary>
    public AdaGradOptions() : base(0.01)
    {
    }

    /// <summary>
    /// Gets the numerical stability constant ε added to the denominator.
    /// <para>分母に加算する数値安定化定数 ε を取得します。</para>
    /// </summary>
    /// <value>A positive finite number. The default is <c>1e-10</c>. / 正の有限値。既定値は <c>1e-10</c>。</value>
    public double Epsilon { get; init; } = 1e-10;

    /// <summary>
    /// Gets the initial value of the squared-gradient accumulator.
    /// <para>勾配二乗和アキュムレーターの初期値を取得します。</para>
    /// </summary>
    /// <value>A non-negative finite number. The default is 0. / 0 以上の有限値。既定値は 0。</value>
    public double InitialAccumulator { get; init; }
}

/// <summary>
/// Configures options for <see cref="RmsProp"/> (default learning rate 0.001).
/// <para><see cref="RmsProp"/> の設定オプションを表します（基準学習率の既定値は 0.001）。</para>
/// </summary>
/// <remarks>
/// <para>Option values are validated by the optimizer constructor.</para>
/// <para>設定値はオプティマイザーのコンストラクターで検証されます。</para>
/// </remarks>
public sealed class RmsPropOptions : StochasticGradientOptions
{
    /// <summary>
    /// Initializes a new instance with default RMSProp options.
    /// <para>既定の RMSProp 設定でインスタンスを初期化します。</para>
    /// </summary>
    public RmsPropOptions() : base(0.001)
    {
    }

    /// <summary>
    /// Gets the exponential moving average decay factor ρ for squared gradients.
    /// <para>勾配二乗の指数移動平均係数 ρ を取得します。</para>
    /// </summary>
    /// <value>A value in <c>[0, 1)</c>. The default is 0.9. / <c>[0, 1)</c> の値。既定値は 0.9。</value>
    public double Decay { get; init; } = 0.9;

    /// <summary>
    /// Gets the numerical stability constant ε added to the denominator.
    /// <para>分母に加算する数値安定化定数 ε を取得します。</para>
    /// </summary>
    /// <value>A positive finite number. The default is <c>1e-8</c>. / 正の有限値。既定値は <c>1e-8</c>。</value>
    public double Epsilon { get; init; } = 1e-8;

    /// <summary>
    /// Gets the momentum coefficient μ.
    /// <para>モーメンタム係数 μ を取得します。</para>
    /// </summary>
    /// <value>A value in <c>[0, 1)</c>. The default is 0 (no momentum). / <c>[0, 1)</c> の値。既定値は 0（モーメンタムなし）。</value>
    public double Momentum { get; init; }

    /// <summary>
    /// Gets a value indicating whether to use centered RMSProp (normalizing by the estimated variance <c>s − m²</c>).
    /// <para>勾配の移動平均を引いた分散推定値 <c>s − m²</c> で正規化する中心化 RMSProp を使用するかどうかを取得します。</para>
    /// </summary>
    /// <value><see langword="true"/> to use centered RMSProp (Graves 2013); otherwise, <see langword="false"/> (the default). / 中心化 RMSProp（Graves 2013）を使用する場合は <see langword="true"/>、それ以外は <see langword="false"/>（既定値）。</value>
    public bool Centered { get; init; }
}

/// <summary>
/// Configures options for <see cref="Adam"/> (default learning rate 0.001).
/// <para><see cref="Adam"/> の設定オプションを表します（基準学習率の既定値は 0.001）。</para>
/// </summary>
/// <remarks>
/// <para>Option values are validated by the optimizer constructor.</para>
/// <para>設定値はオプティマイザーのコンストラクターで検証されます。</para>
/// </remarks>
public sealed class AdamOptions : StochasticGradientOptions
{
    /// <summary>
    /// Initializes a new instance with default Adam options.
    /// <para>既定の Adam 設定でインスタンスを初期化します。</para>
    /// </summary>
    public AdamOptions() : base(0.001)
    {
    }

    /// <summary>
    /// Gets the exponential decay rate β₁ for the first-moment estimate.
    /// <para>1次モーメント推定の指数移動平均係数 β₁ を取得します。</para>
    /// </summary>
    /// <value>A value in <c>[0, 1)</c>. The default is 0.9. / <c>[0, 1)</c> の値。既定値は 0.9。</value>
    public double Beta1 { get; init; } = 0.9;

    /// <summary>
    /// Gets the exponential decay rate β₂ for the second-moment estimate.
    /// <para>2次モーメント推定の指数移動平均係数 β₂ を取得します。</para>
    /// </summary>
    /// <value>A value in <c>[0, 1)</c>. The default is 0.999. / <c>[0, 1)</c> の値。既定値は 0.999。</value>
    public double Beta2 { get; init; } = 0.999;

    /// <summary>
    /// Gets the numerical stability constant ε added to the denominator.
    /// <para>分母に加算する数値安定化定数 ε を取得します。</para>
    /// </summary>
    /// <value>A positive finite number. The default is <c>1e-8</c>. / 正の有限値。既定値は <c>1e-8</c>。</value>
    public double Epsilon { get; init; } = 1e-8;

    /// <summary>
    /// Gets a value indicating whether to use the AMSGrad variant (maintaining the running maximum of the second moment).
    /// <para>2次モーメントのこれまでの最大値を分母に使用する AMSGrad バリアントを有効にするかどうかを取得します。</para>
    /// </summary>
    /// <value><see langword="true"/> to enable AMSGrad (Reddi, Kale &amp; Kumar 2018); otherwise, <see langword="false"/> (the default). / AMSGrad（Reddi, Kale &amp; Kumar 2018）を使用する場合は <see langword="true"/>、それ以外は <see langword="false"/>（既定値）。</value>
    public bool Amsgrad { get; init; }
}

/// <summary>
/// Configures options for <see cref="NAdam"/> (default learning rate 0.002).
/// <para><see cref="NAdam"/> の設定オプションを表します（基準学習率の既定値は 0.002）。</para>
/// </summary>
/// <remarks>
/// <para>Option values are validated by the optimizer constructor.</para>
/// <para>設定値はオプティマイザーのコンストラクターで検証されます。</para>
/// </remarks>
public sealed class NAdamOptions : StochasticGradientOptions
{
    /// <summary>
    /// Initializes a new instance with default NAdam options.
    /// <para>既定の NAdam 設定でインスタンスを初期化します。</para>
    /// </summary>
    public NAdamOptions() : base(0.002)
    {
    }

    /// <summary>
    /// Gets the exponential decay rate β₁ for the first-moment estimate.
    /// <para>1次モーメント推定の指数移動平均係数 β₁ を取得します。</para>
    /// </summary>
    /// <value>A value in <c>[0, 1)</c>. The default is 0.9. / <c>[0, 1)</c> の値。既定値は 0.9。</value>
    public double Beta1 { get; init; } = 0.9;

    /// <summary>
    /// Gets the exponential decay rate β₂ for the second-moment estimate.
    /// <para>2次モーメント推定の指数移動平均係数 β₂ を取得します。</para>
    /// </summary>
    /// <value>A value in <c>[0, 1)</c>. The default is 0.999. / <c>[0, 1)</c> の値。既定値は 0.999。</value>
    public double Beta2 { get; init; } = 0.999;

    /// <summary>
    /// Gets the numerical stability constant ε added to the denominator.
    /// <para>分母に加算する数値安定化定数 ε を取得します。</para>
    /// </summary>
    /// <value>A positive finite number. The default is <c>1e-8</c>. / 正の有限値。既定値は <c>1e-8</c>。</value>
    public double Epsilon { get; init; } = 1e-8;
}

/// <summary>
/// Configures options for <see cref="AdaMax"/> (default learning rate 0.002).
/// <para><see cref="AdaMax"/> の設定オプションを表します（基準学習率の既定値は 0.002）。</para>
/// </summary>
/// <remarks>
/// <para>Option values are validated by the optimizer constructor.</para>
/// <para>設定値はオプティマイザーのコンストラクターで検証されます。</para>
/// </remarks>
public sealed class AdaMaxOptions : StochasticGradientOptions
{
    /// <summary>
    /// Initializes a new instance with default AdaMax options.
    /// <para>既定の AdaMax 設定でインスタンスを初期化します。</para>
    /// </summary>
    public AdaMaxOptions() : base(0.002)
    {
    }

    /// <summary>
    /// Gets the exponential decay rate β₁ for the first-moment estimate.
    /// <para>1次モーメント推定の指数移動平均係数 β₁ を取得します。</para>
    /// </summary>
    /// <value>A value in <c>[0, 1)</c>. The default is 0.9. / <c>[0, 1)</c> の値。既定値は 0.9。</value>
    public double Beta1 { get; init; } = 0.9;

    /// <summary>
    /// Gets the exponential decay rate β₂ for the infinity-norm estimate.
    /// <para>無限大ノルム推定の指数減衰係数 β₂ を取得します。</para>
    /// </summary>
    /// <value>A value in <c>[0, 1)</c>. The default is 0.999. / <c>[0, 1)</c> の値。既定値は 0.999。</value>
    public double Beta2 { get; init; } = 0.999;
}
