namespace Tuna;

/// <summary>
/// Generates elementwise vector and matrix overloads and a corresponding operator struct for a scalar method.
/// <para>スカラーメソッドに対して、ベクトル・行列向けの要素ごとのオーバーロードと対応する演算子構造体を生成します。</para>
/// </summary>
/// <remarks>
/// <para>
/// Apply this attribute to a <c>static E Name(...)</c> method inside a non-generic <c>partial</c> class, where <c>E</c> is the element type (a concrete type such as <see cref="double"/> or a method type parameter).
/// By default, the first parameter is iterated elementwise; use <see cref="Over"/> to select 1 to 3 parameters of type <c>E</c>. All other parameters are passed identically to every element.
/// </para>
/// <para>
/// Generates allocating and <c>destination</c> overloads for vectors and matrices (preserving parameter order), an <c>InPlace</c> nested class for single-input operations, an operator struct compatible with <c>XMath.Map</c> or <c>XMath.Zip</c>, and type-inference overloads when <c>E</c> is generic.
/// Multiple attributes may be applied to the same method with distinct <see cref="OperatorName"/> values.
/// </para>
/// <para>
/// 型引数を持たない <c>partial</c> クラス内の <c>static E Name(...)</c> メソッドに適用します（<c>E</c> は <see cref="double"/> などの具体型、またはメソッドの型引数）。
/// 既定では第1引数を要素ごとに反復します。<see cref="Over"/> で型 <c>E</c> の引数を1〜3個指定でき、残りの引数は全要素に共通の値として渡されます。
/// </para>
/// <para>
/// ベクトル・行列向けに結果を新規確保するオーバーロードと <c>destination</c> を受け取るオーバーロード（引数の順序を保持）、単一入力向けの入れ子クラス <c>InPlace</c>、<c>XMath.Map</c> または <c>XMath.Zip</c> に渡せる演算子構造体、および <c>E</c> がジェネリックな場合の型推論オーバーロードを生成します。
/// 同じメソッドに複数の属性を付ける場合は、それぞれ異なる <see cref="OperatorName"/> を指定してください。
/// </para>
/// <para>When the compilation references Tuna.Differentiation, specifying <see cref="Derivative"/> or <see cref="PartialDerivatives"/> generates node overloads for differentiation. Extra parameters are treated as constants.</para>
/// <para>Tuna.Differentiation を参照するプロジェクトでは、<see cref="Derivative"/> または <see cref="PartialDerivatives"/> を指定すると自動微分ノードを受け取るオーバーロードも生成します。追加の引数は定数として扱います。
/// </para>
/// <example>
/// <code>
/// [ElementwiseOperation(Over = [nameof(x)])]
/// public static double BesselJ(double nu, double x) => ...;
///
/// [ElementwiseOperation(Over = [nameof(a), nameof(x)], OperatorName = "RegularizedGammaP", OperatorContainer = "Operators")]
/// public static double RegularizedGammaP(double a, double x) => ...;
/// </code>
/// </example>
/// </remarks>
[System.AttributeUsage(System.AttributeTargets.Method, AllowMultiple = true)]
public sealed class ElementwiseOperationAttribute : System.Attribute
{
    /// <summary>
    /// Gets or sets the names of 1 to 3 parameters to iterate elementwise.
    /// <para>要素ごとに反復する 1〜3 個の引数名を取得または設定します。</para>
    /// </summary>
    /// <remarks>
    /// <para>When omitted, only the first parameter is iterated elementwise.</para>
    /// <para>省略した場合は第 1 引数のみを要素ごとに反復します。</para>
    /// </remarks>
    public string[]? Over { get; set; }

    /// <summary>
    /// Gets or sets the derivative method name for a single-input operation.
    /// <para>単一入力の演算に対応する導関数のメソッド名を取得または設定します。</para>
    /// </summary>
    /// <remarks>
    /// <para>The static method must be declared in the same type and accept the original arguments followed by the forward result.</para>
    /// <para>同じ型に宣言した静的メソッドを指定します。元の引数に続けて順伝播の結果を受け取ります。</para>
    /// </remarks>
    public string? Derivative { get; set; }

    /// <summary>
    /// Gets or sets the partial derivative method names for two or three inputs in <see cref="Over"/> order.
    /// <para>2 個または 3 個の入力に対応する偏微分のメソッド名を <see cref="Over"/> の指定順で取得または設定します。</para>
    /// </summary>
    /// <remarks>
    /// <para>Each static method must be declared in the same type and accept the original arguments followed by the forward result.</para>
    /// <para>同じ型に宣言した静的メソッドを指定します。各メソッドは元の引数に続けて順伝播の結果を受け取ります。</para>
    /// </remarks>
    public string[]? PartialDerivatives { get; set; }

    /// <summary>
    /// Gets or sets the name of the generated operator struct.
    /// <para>生成する演算子構造体の名前を取得または設定します。</para>
    /// </summary>
    /// <remarks>
    /// <para>Defaults to the method name followed by <c>Operator</c>. Specify distinct names when decorating overloaded methods.</para>
    /// <para>既定は「メソッド名 + Operator」です。同名のオーバーロードに適用する場合は、それぞれ異なる名前を指定してください。</para>
    /// </remarks>
    public string? OperatorName { get; set; }

    /// <summary>
    /// Gets or sets the name of a nested static partial class in which to place the generated operator struct.
    /// <para>生成する演算子構造体を配置する入れ子の <c>public static partial</c> クラス名を取得または設定します。</para>
    /// </summary>
    /// <remarks>
    /// <para>When omitted, the operator struct is generated directly in the declaring type.</para>
    /// <para>省略した場合は、メソッドを宣言する型に直接生成されます。</para>
    /// </remarks>
    public string? OperatorContainer { get; set; }
}

/// <summary>
/// Generates pairwise matrix overloads and a corresponding operator struct for a two-vector scalar method.
/// <para>2つのベクトルからスカラーを返すメソッドに対して、行列の列ペアごとのオーバーロードと対応する演算子構造体を生成します。</para>
/// </summary>
/// <remarks>
/// <para>
/// Apply this attribute to a <c>static E Name(ReadOnlyVectorSpan&lt;E&gt; x, ReadOnlyVectorSpan&lt;E&gt; y, ...)</c> method inside a non-generic <c>partial</c> class.
/// Generates allocating and <c>destination</c> matrix overloads computing <c>result[i, j] = Name(x.Column(i), y.Column(j), ...)</c>, along with an operator struct for <c>XMath.Pairwise</c>.
/// </para>
/// <para>
/// 型引数を持たない <c>partial</c> クラス内の <c>static E Name(ReadOnlyVectorSpan&lt;E&gt; x, ReadOnlyVectorSpan&lt;E&gt; y, ...)</c> メソッドに適用します。
/// <c>result[i, j] = Name(x.Column(i), y.Column(j), ...)</c> を計算する新規確保・<c>destination</c> オーバーロードと、<c>XMath.Pairwise</c> に渡せる演算子構造体を生成します。
/// </para>
/// </remarks>
[System.AttributeUsage(System.AttributeTargets.Method)]
public sealed class PairwiseOperationAttribute : System.Attribute
{
    /// <summary>
    /// Gets or sets the name of the generated operator struct.
    /// <para>生成する演算子構造体の名前を取得または設定します。</para>
    /// </summary>
    /// <remarks>
    /// <para>Defaults to the method name followed by <c>Operator</c>.</para>
    /// <para>既定は「メソッド名 + Operator」です。</para>
    /// </remarks>
    public string? OperatorName { get; set; }

    /// <summary>
    /// Gets or sets the name of a nested static partial class in which to place the generated operator struct.
    /// <para>生成する演算子構造体を配置する入れ子クラス名を取得または設定します。</para>
    /// </summary>
    /// <remarks>
    /// <para>When omitted, the operator struct is generated directly in the declaring type.</para>
    /// <para>省略した場合は、メソッドを宣言する型に直接生成されます。</para>
    /// </remarks>
    public string? OperatorContainer { get; set; }
}

/// <summary>
/// Generates overloads that accept containers and writable views so generic element types can be inferred at call sites.
/// <para>ビューを受け取るジェネリックメソッドに対して、コンテナや書き込み可能ビューから要素型を推論できるオーバーロードを生成します。</para>
/// </summary>
/// <remarks>
/// <para>
/// Because C# type inference does not consider user-defined implicit conversions, calling <c>F&lt;T&gt;(ReadOnlyVectorSpan&lt;T&gt;)</c> with a <c>Vector&lt;double&gt;</c> cannot infer <c>T</c>.
/// This attribute generates companion overloads taking container types (<c>Vector</c>, <c>Matrix</c>, <c>Tensor</c>) and writable view types, forwarding to the base method.
/// </para>
/// <para>
/// C# の型推論はユーザー定義の暗黙変換を考慮しないため、<c>F&lt;T&gt;(ReadOnlyVectorSpan&lt;T&gt;)</c> に <c>Vector&lt;double&gt;</c> を渡しても <c>T</c> を推論できません。
/// この属性は、コンテナ型（<c>Vector</c>、<c>Matrix</c>、<c>Tensor</c>）や書き込み可能ビューを受け取って元のメソッドへ転送するオーバーロードを生成します。
/// </para>
/// </remarks>
[System.AttributeUsage(System.AttributeTargets.Method)]
public sealed class InferenceOverloadsAttribute : System.Attribute
{
}

/// <summary>
/// Generates overloads with the first type parameter fixed to each specified element type.
/// <para>最初の型引数（要素型）を指定した各型に固定したオーバーロードを生成します。</para>
/// </summary>
/// <remarks>
/// <para>
/// Because C# does not support partial type-argument inference, these overloads allow callers to specify only the remaining type arguments (for example, <c>Map&lt;Softplus&gt;(x)</c>).
/// </para>
/// <para>
/// C# では一部の型引数だけを推論できないため、この属性により残りの型引数のみを指定して呼び出せるオーバーロード（例: <c>Map&lt;Softplus&gt;(x)</c>）を生成します。
/// </para>
/// </remarks>
[System.AttributeUsage(System.AttributeTargets.Method)]
public sealed class FixedElementTypeOverloadsAttribute : System.Attribute
{
    /// <summary>
    /// Initializes a new instance of the <see cref="FixedElementTypeOverloadsAttribute"/> class with the element types to fix.
    /// <para>固定する要素型の一覧を指定して <see cref="FixedElementTypeOverloadsAttribute"/> クラスの新しいインスタンスを初期化します。</para>
    /// </summary>
    /// <param name="elementTypes">The element types for which to generate fixed overloads. / 固定オーバーロードを生成する要素型の一覧。</param>
    public FixedElementTypeOverloadsAttribute(params System.Type[] elementTypes) => ElementTypes = elementTypes;

    /// <summary>
    /// Gets the element types for which fixed overloads are generated.
    /// <para>固定オーバーロードを生成する要素型の一覧を取得します。</para>
    /// </summary>
    public System.Type[] ElementTypes { get; }
}
