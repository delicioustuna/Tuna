namespace Tuna;

/// <summary>
/// Provides scalar elementary functions and numerical operations for vectors, matrices, and tensors.
/// <para>スカラーの初等関数およびベクトル・行列・テンソル向けの数値演算を提供します。</para>
/// </summary>
public static partial class XMath
{
    /// <summary>
    /// Provides in-place operations that overwrite their input buffer.
    /// <para>入力バッファをインプレースで上書きする演算を提供します。</para>
    /// </summary>
    /// <remarks>
    /// <para>
    /// Results match calling the corresponding <c>destination</c> overload with the input as the destination.
    /// Operations whose output cannot safely overwrite their input (such as matrix multiplication) are excluded.
    /// For container arithmetic, compound assignment operators (such as <c>x += y</c> and <c>x *= a</c>) can also be used.
    /// </para>
    /// <para>
    /// 結果は、対応する <c>destination</c> オーバーロードに入力と同じ領域を渡した場合と一致します。
    /// 行列積のようにインプレース更新ができない演算は含まれません。
    /// コンテナ同士の四則演算には複合代入演算子（<c>x += y</c>、<c>x *= a</c> など）も使用できます。
    /// </para>
    /// </remarks>
    public static partial class InPlace
    {
    }
}
