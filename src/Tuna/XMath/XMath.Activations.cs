using System.Numerics;
using System.Runtime.CompilerServices;

namespace Tuna;

public static partial class XMath
{
    internal const double SeluAlpha = 1.6732632423543772848170429916717;
    internal const double SeluScale = 1.0507009873554804934193349852946;

    // 負の裾の指数シフト量 c（整数なので x + c に丸め誤差が生じない）と e⁻ᶜ。
    internal const double TailExpShift = 64;
    internal const double TailExpUnshift = 1.603810890548638e-28;

    // この値未満では e⁻ˣ がオーバーフローし eˣ が非正規化数になるため、x·e^(x+c)·e⁻ᶜ で正規化数の範囲に保ってから丸める。
    internal const double SiluTailBelow = -709;

    // この値未満では erfc(-x/√2) が非正規化数に近づき |x|/2 倍で誤差が拡大するため、erfcx による裾の式へ切り替える。
    private const double GeluTailBelow = -37.5;

    // この値未満では真の値の絶対値が 2⁻¹⁰⁷⁵ 未満となり -0 に丸まる（x² のオーバーフロー前に打ち切る）。
    private const double GeluUnderflowBelow = -39;

    /// <summary>
    /// Computes the rectified linear unit <c>max(x, 0)</c>.
    /// <para>ReLU 関数 <c>max(x, 0)</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The numeric type. / 数値型。</typeparam>
    /// <param name="x">The input value. / 入力値。</param>
    /// <returns><c>max(x, 0)</c>, propagating <c>NaN</c>. / <c>max(x, 0)</c>（<c>NaN</c> はそのまま伝播します）。</returns>
    [ElementwiseOperation(OperatorName = "Relu", OperatorContainer = "Operators")]
    [MethodImpl(MethodImplOptions.AggressiveInlining)] // SIMD ループの端数処理でインライン展開を確実にする
    public static T Relu<T>(T x)
        where T : unmanaged, INumber<T> =>
        x > T.Zero || T.IsNaN(x) ? x : T.Zero;

    /// <summary>
    /// Computes the leaky rectified linear unit (<paramref name="x"/> if <c>x &gt; 0</c>, otherwise <c>slope · x</c>).
    /// <para>Leaky ReLU（<c>x &gt; 0</c> のとき <paramref name="x"/>、それ以外は <c>slope · x</c>）を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The numeric type. / 数値型。</typeparam>
    /// <param name="x">The input value. / 入力値。</param>
    /// <param name="slope">The slope applied to non-positive inputs (commonly <c>0.01</c>). / 非正の入力に掛ける傾き（一般的には <c>0.01</c>）。</param>
    /// <returns>The Leaky ReLU value. / Leaky ReLU の値。</returns>
    [ElementwiseOperation(OperatorName = "LeakyRelu", OperatorContainer = "Operators")]
    public static T LeakyRelu<T>(T x, T slope)
        where T : unmanaged, INumber<T> =>
        x > T.Zero ? x : unchecked(x * slope);

    /// <summary>
    /// Computes the exponential linear unit (<paramref name="x"/> if <c>x &gt; 0</c>, otherwise <c>alpha · (eˣ - 1)</c>).
    /// <para>ELU（<c>x &gt; 0</c> のとき <paramref name="x"/>、それ以外は <c>alpha · (eˣ - 1)</c>）を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
    /// <param name="x">The input value. / 入力値。</param>
    /// <param name="alpha">The scale factor for non-positive inputs. / 非正の入力に掛ける係数。</param>
    /// <returns>The ELU value. / ELU の値。</returns>
    [ElementwiseOperation(OperatorName = "Elu", OperatorContainer = "Operators")]
    public static T Elu<T>(T x, T alpha)
        where T : unmanaged, IFloatingPointIeee754<T> =>
        x > T.Zero ? x : T.CreateTruncating(double.CreateTruncating(alpha) * XMath.Special.Expm1(double.CreateTruncating(x)));

    /// <summary>
    /// Computes the scaled exponential linear unit <c>λ · ELU(x; α)</c> using the Klambauer et al. (2017) constants.
    /// <para>Klambauer ら (2017) の定数を用いて SELU <c>λ · ELU(x; α)</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
    /// <param name="x">The input value. / 入力値。</param>
    /// <returns>The SELU value. / SELU の値。</returns>
    [ElementwiseOperation(OperatorName = "Selu", OperatorContainer = "Operators")]
    public static T Selu<T>(T x)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        double v = double.CreateTruncating(x);
        return T.CreateTruncating(SeluScale * (v > 0 ? v : SeluAlpha * XMath.Special.Expm1(v)));
    }

    /// <summary>
    /// Computes the exact Gaussian error linear unit <c>x · Φ(x) = (x / 2) · erfc(-x / √2)</c>.
    /// <para>定義どおりの GELU <c>x · Φ(x) = (x / 2) · erfc(-x / √2)</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
    /// <param name="x">The input value. / 入力値。</param>
    /// <returns>The exact GELU value. / 定義どおりの GELU の値。</returns>
    /// <remarks>
    /// <para>For the <c>tanh</c>-based approximation, use <see cref="Gelu{T}(T, bool)"/>.</para>
    /// <para><c>tanh</c> による近似式を使う場合は <see cref="Gelu{T}(T, bool)"/> を使用してください。</para>
    /// </remarks>
    [ElementwiseOperation(OperatorName = "Gelu", OperatorContainer = "Operators")]
    public static T Gelu<T>(T x)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        double v = double.CreateTruncating(x);
        // 真の値が -0 へ丸まる範囲（x = -∞ を含む）。裾の式で x² がオーバーフローして 0·∞ になる前に返す。
        if (v < GeluUnderflowBelow)
        {
            return T.CreateTruncating(-0.0);
        }

        return T.CreateTruncating(v < GeluTailBelow ? GeluNegativeTail(v) : 0.5 * v * XMath.Special.Erfc(-v * Constants.OneOverSqrt2));
    }

    // 負の裾の GELU: (x/2)·erfcx(-x/√2)·e^(-x²/2)。x² = hi + lo に誤差なく分割し、シフト付きで正規化数のまま評価してから最後に非正規化数へ丸める。
    private static double GeluNegativeTail(double v)
    {
        double hi = v * v;
        double lo = Math.FusedMultiplyAdd(v, v, -hi);
        double gaussian = Math.Exp(TailExpShift - 0.5 * hi) * (1 - 0.5 * lo);
        return 0.5 * v * XMath.Special.Erfcx(-v * Constants.OneOverSqrt2) * gaussian * TailExpUnshift;
    }

    /// <summary>
    /// Computes the sigmoid linear unit (SiLU / Swish) <c>x · σ(x) = x / (1 + e⁻ˣ)</c>.
    /// <para>SiLU（Swish）関数 <c>x · σ(x) = x / (1 + e⁻ˣ)</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
    /// <param name="x">The input value. / 入力値。</param>
    /// <returns>The SiLU value. / SiLU の値。</returns>
    [ElementwiseOperation(OperatorName = "Silu", OperatorContainer = "Operators")]
    public static T Silu<T>(T x)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        double v = double.CreateTruncating(x);
        // x = -∞ では -∞ / ∞ になるため極限値 -0 を返す。
        return T.CreateTruncating(
            double.IsNegativeInfinity(v) ? -0.0
            : v < SiluTailBelow ? v * Math.Exp(v + TailExpShift) * TailExpUnshift
            : v / (1 + Math.Exp(-v)));
    }

    /// <summary>
    /// Computes the softplus function <c>ln(1 + eˣ)</c> without intermediate overflow.
    /// <para>途中計算でのオーバーフローを回避して Softplus 関数 <c>ln(1 + eˣ)</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
    /// <param name="x">The input value. / 入力値。</param>
    /// <returns>The softplus value. / Softplus の値。</returns>
    [ElementwiseOperation(OperatorName = "Softplus", OperatorContainer = "Operators")]
    public static T Softplus<T>(T x)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        return T.CreateTruncating(SoftplusOf(double.CreateTruncating(x)));
    }

    private static double SoftplusOf(double v) => v > 0 ? v + XMath.Special.Log1p(Math.Exp(-v)) : XMath.Special.Log1p(Math.Exp(v));
}
