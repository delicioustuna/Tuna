# `XMath` の使い方

`Tuna.XMath` は、Tuna の数値計算における統一的な入口となる静的クラスです。
スカラーの初等関数から、ベクトル・行列・テンソルに対する要素別演算、縮約、行列積、連立一次方程式、ソート、ユーザー定義演算子までを一貫したインターフェースで呼び出せます。

コア `Tuna` の数式を短く書きたい場合は `using static Tuna.XMath;` を宣言できます。

```csharp
using Tuna;
using Tuna.LinearAlgebra;
using static Tuna.XMath;
```

> **拡張パッケージを併用する場合の注意**
> `Tuna.Differentiation`・`Tuna.SignalProcessing`・`Tuna.NeuralNetwork` が C# の拡張ブロック（`extension(XMath)`）で追加するメンバー（自動微分ノード版の `Sum` / `Exp`、`Convolve`、`Linear` / `LayerNorm` など）は、`using static Tuna.XMath;` ではインポートされません。`using static` のまま呼び出すと `CS0103` や、コア `XMath` と同名のメソッドで `CS0411`（型引数を推論できない）が発生します。拡張パッケージを使うファイルでは `XMath.Sum(...)`・`XMath.Convolve(...)`・`XMath.Linear(...)` のように `XMath.` 修飾を付けて呼び出してください。

## 3 つの呼び出し形式

`XMath` のほぼすべての配列・行列演算は、同じメソッド名で次の 3 形式を用意しています。

| 形式 | 呼び出し例 | 挙動 |
| --- | --- | --- |
| **結果を新規割り当て** | `Vector<double> y = Exp(x);` | 新しいコンテナを作成して返す。読みやすさ・プロトタイピング向き |
| **`destination` を指定** | `Exp(x, buffer);` | 末尾引数の `destination` に結果を書き込み、`destination` をそのまま返す |
| **インプレース更新** | `InPlace.Exp(x);` | 入力 `x` 自身の要素を直接書き換える（`void`） |

`InPlace` は入れ子クラス `XMath.InPlace` に分離されているため、`using static Tuna.XMath;` を使っても非破壊のオーバーロードと取り違えることはありません。
また、行列積のように入力と出力を直接重ねられない演算には `InPlace` オーバーロードはありません。

```csharp
var x = new Vector<double>([0.0, 1.0, 2.0]);

// 1. 新規割り当て
Vector<double> y = Exp(x);

// 2. destination 指定（既存バッファを再利用）
var buffer = new Vector<double>(x.Length);
Add(x, 1.0, buffer);

// 3. インプレース更新
InPlace.Exp(x);
InPlace.Multiply(x, 2.0);
```

## コンテナとビューの混在

`XMath` の入力は読み取り専用ビュー（`ReadOnlyVectorSpan<T>` / `ReadOnlyMatrixSpan<T>`）、出力先は書き込み可能なビューまたはコンテナを受け取ります。
コンテナ（`Vector<T>`, `Matrix<T>`）とビュー（`A.T`, `A.Row(0)`, `A[0..2, ..]` など）はそのまま渡すことができ、転置やスライスのためにコピーは発生しません。

```csharp
var A = Matrix<double>.FromRowMajor(3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9]);

Vector<double> diagSqrt = Sqrt(A.Diagonal());
Vector<double> prod = Multiply(A[0..2, ..], A.Column(0));

// 出力先にもビューを指定できる（第 0 行に第 2 行を加算）
Add(A.Row(0), A.Row(2), A.Row(0));
```

## 明示的なブロードキャスト

Tuna では意図しない形状変換を防ぐため、**暗黙のブロードキャストは行いません**。
ベクトルを行列のすべての列または行に適用する場合は、`Broadcast` / `BroadcastRows` でストライド 0 のゼロコピービューを作成して渡します。

```csharp
// 3 次元のデータ点が 4 つ（3 × 4 行列、1 列 = 1 点）
var X = Matrix<double>.FromRowMajor(3, 4,
[
    1,  2,  3,  4,
   10, 20, 30, 40,
    0,  0,  1,  1,
]);

// 各行の平均（長さ 3）を全列にブロードキャストして中心化
Vector<double> mean = RowMeans(X);
Subtract(X, mean.Broadcast(X.Columns), X);

// 長さ 4 のベクトルを全行にブロードキャストして列ごとにスケーリング
var scale = new Vector<double>([1.0, 0.5, 0.25, 0.125]);
Matrix<double> scaled = Hadamard(X, scale.BroadcastRows(X.Rows));
```

同様に、`Matrix<T>.Broadcast(count)` を使うと 1 つの行列を `ReadOnlyTensorSpan<T>` としてバッチ演算へ渡せます。

## 主な演算カテゴリ

- **初等関数・数値安定関数**: `Exp`, `Log`, `Sqrt`, `Pow`, `Log1p`, `Expm1`, `Log1mExp`, `LogAddExp`, `Hypot` など
- **活性化関数・Softmax**: `Relu`, `LeakyRelu`, `Elu`, `Selu`, `Silu`, `Gelu`, `Logistic`, `Tanh`, `Softplus`, `Softmax`, `LogSoftmax`, `ColumnSoftmax`, `ColumnLogSoftmax`
- **縮約・統計量**: `Sum`, `Mean`, `Variance`, `StandardDeviation`, `Min`, `Max`, `Norm`, `Norm1`, `NormInfinity`, `LogSumExp`, `Trace`, `ColumnSums`, `RowMeans` など（`ignoreNaN: true` オプション付きオーバーロードあり）
- **ソート・順序統計量**: `Sort`, `ArgSort`, `Median`, `Quantile`, `IndexOfMax`, `IndexOfMin`
- **積・距離**: `Dot`, `Cross`, `Multiply`, `Hadamard`, `Outer`, `Transpose`, `Distance`, `SquaredDistance`
- **複素数**: `Real`, `Imaginary`, `Magnitude`, `Phase`
- **連立一次方程式・対数行列式**: `Solve`, `Inverse`, `SolveLowerTriangular`, `SolveUpperTriangular`, `CholeskyLogDeterminant`

全メソッドの網羅的な一覧は [Tuna API リファレンス](api/Tuna.md#static-class-xmath) を参照してください。

## ユーザー定義演算子と Source Generator

任意の要素別演算や列ペア演算を中間配列なしで実行する方法が 2 つあります。

### 方法 A: `[ElementwiseOperation]` / `[PairwiseOperation]` 属性（推奨）

`partial` クラスの `static` メソッドに属性を付けると、`Tuna.Generators`（ビルド時 Source Generator）がベクトル・行列オーバーロード、`destination` オーバーロード、`InPlace`、および演算子構造体を自動生成します。

```csharp
internal static partial class CustomMath
{
    // 要素別演算（第 1 引数 x をベクトル/行列化、delta は全要素で共有）
    [ElementwiseOperation]
    public static double Huber(double x, double delta) =>
        Math.Abs(x) <= delta ? 0.5 * x * x : delta * (Math.Abs(x) - 0.5 * delta);

    // 2 引数を要素ごとに適用する場合は Over で指定
    [ElementwiseOperation(Over = [nameof(a), nameof(b)])]
    public static double Hypot2(double a, double b) => Math.Sqrt(a * a + b * b);

    // 列ペアごとの演算（K[i, j] = Gaussian(X.Column(i), Y.Column(j), lengthScale)）
    [PairwiseOperation]
    public static double Gaussian(ReadOnlyVectorSpan<double> x, ReadOnlyVectorSpan<double> y, double lengthScale)
    {
        double sq = 0;
        for (int i = 0; i < x.Length; i++)
        {
            double d = x[i] - y[i];
            sq += d * d;
        }

        return Math.Exp(-sq / (2 * lengthScale * lengthScale));
    }
}

// 生成されたオーバーロードの呼び出し
Vector<double> loss = CustomMath.Huber(x, delta: 1.0);
CustomMath.InPlace.Huber(x, delta: 1.0);
Matrix<double> K = CustomMath.Gaussian(X, X, lengthScale: 1.0);
```

### 方法 B: 演算子構造体を `XMath.Map` / `Zip` / `Pairwise` に渡す

状態のない演算は `IUnaryOperator<T>`（静的抽象メンバー）を型引数で、状態を持つ演算は `IStatefulUnaryOperator<T>` を値で渡します。

```csharp
readonly struct Cube : IUnaryOperator<double>
{
    public static double Invoke(double x) => x * x * x;
}

readonly struct Shift(double offset) : IStatefulUnaryOperator<double>
{
    public double Invoke(double x) => x + offset;
}

Vector<double> cubed = XMath.Map<Cube>(x);
Vector<double> shifted = XMath.Map(x, new Shift(0.5));
```

## ユーザー定義演算の自動微分

### 要素別演算の導関数

`[ElementwiseOperation]` に導関数を指定すると、`Tuna.Differentiation` を参照するプロジェクトで `ScalarNode`・`VectorNode<T>`・`MatrixNode<T>` のオーバーロードも生成されます。
NuGet の `Tuna` パッケージにはソースジェネレーターが含まれます。ソースをプロジェクト参照する場合は `Tuna.Generators` もアナライザーとして参照してください。

```csharp
public static partial class MyFunctions
{
    [ElementwiseOperation(Derivative = nameof(SoftsignDerivative))]
    public static double Softsign(double x) => x / (1 + Math.Abs(x));

    private static double SoftsignDerivative(double x, double value)
    {
        double t = 1 - Math.Abs(value);
        return t * t;
    }

    [ElementwiseOperation(Over = [nameof(x), nameof(y)],
        PartialDerivatives = [nameof(PartialX), nameof(PartialY)])]
    public static double WeightedSum(double x, double scale, double y) => scale * x + y;

    private static double PartialX(double x, double scale, double y, double value) => scale;
    private static double PartialY(double x, double scale, double y, double value) => 1;
}
```

導関数は元の関数の引数を同じ順で受け取り、最後に順伝播の計算結果を受け取ります。入力が 2 個または 3 個のときは、`PartialDerivatives` に偏微分のメソッド名を `Over` の指定順で並べます。
`Over` に含まれない引数（例の `scale`）は全要素で共通の定数として渡されます。配列入力は同じ形状にしてください。
スカラーの局所微分は、`ScalarNode.Record` と同様に記録時に評価します。配列の局所微分は逆伝播時に随伴が 0 以外の要素でのみ評価し、局所微分が 0 の要素の寄与は省きます。
型引数で要素型を受け取る関数は、配列オーバーロードでは `float`・`double`、`ScalarNode` オーバーロードでは `double` で評価します。`float` 専用の関数の `ScalarNode` オーバーロードでは、ノードの値を `float` に丸めて評価し、結果と局所微分を `double` に拡張します。

### 配列全体の演算を記録する

`IDifferentiableOperation<T>` を実装して `tape.Apply` に渡すと、既存のカーネルを 1 つの演算として記録できます。呼び出しの最後に出力の長さ、または行数と列数を指定します。

```csharp
var tape = new Tape();
var x = tape.Leaf(new Vector<double>([1, 2, 3]));
var y = tape.Apply(new Sine(), [x], x.Length);
tape.Backward(XMath.Sum(y));   // x の各要素の勾配は cos(x)

readonly struct Sine : IDifferentiableOperation<double>
{
    public void Forward(DifferentiableOperationContext<double> context, MatrixSpan<double> destination)
    {
        var input = context[0];
        Span<double> saved = context.AllocateSaved(input.Rows * input.Columns);
        for (int column = 0; column < input.Columns; column++)
        {
            for (int row = 0; row < input.Rows; row++)
            {
                (double sin, double cos) = Math.SinCos(input[row, column]);
                destination[row, column] = sin;
                saved[column * input.Rows + row] = cos;
            }
        }
    }

    public void PullBack(DifferentiableOperationContext<double> context,
        ReadOnlyMatrixSpan<double> output, ReadOnlyMatrixSpan<double> outputGradient)
    {
        if (context.RequiresGradient(0))
        {
            var dx = context.GetGradient(0);
            var saved = context.GetSaved(0);
            for (int column = 0; column < dx.Columns; column++)
            {
                for (int row = 0; row < dx.Rows; row++)
                {
                    dx[row, column] += saved[column * dx.Rows + row] * outputGradient[row, column];
                }
            }
        }
    }
}
```

`Forward` は出力先の全要素を書き込みます。`AllocateSaved(length)` で確保したテープ所有の領域は、`PullBack` から確保順のインデックスを `GetSaved(index)` に渡して読み取れます。インデックスは各 `Apply` で 0 から始まります。各ビューはコールバックの実行中のみ有効で、保存領域は `Reset()` で再利用されます。入力がすべて定数の場合は、`Forward` の終了時に保存領域を回収します。

`PullBack` は `RequiresGradient(index)` で勾配が必要な入力を判定し、`GetGradient(index)` のビューに寄与を加算します。同じノードや重なるビューを複数回渡した場合も、入力ごとの寄与が加算されます。ブロードキャスト入力への寄与は、コールバック終了後に集約されます。

ベクトルはコールバック内では 1 列の行列として扱われます。行列入力と混在させる場合は `x.AsColumnMatrix()` を使い、微分対象のスカラーを配列演算に渡す場合は長さ 1 のベクトルにします。`VectorNode<T>.Record` と `MatrixNode<T>.Record` は入力からテープを特定します。入力がすべて定数の場合は順伝播だけを実行します。

`Forward` の途中で例外が発生した場合、その演算の出力と保存領域は破棄され、同じテープで続けて記録できます。`PullBack` の途中で例外が発生した場合は、再利用の前に `Reset()` が必要です。すでにパラメータに加算された勾配は残るため、`ParameterSet<T>.ZeroGradients()` または `Parameter<T>.ZeroGradient()` でクリアしてから再評価してください。

## 入力と出力先の重なり（エイリアシング）

- **要素別演算**: 出力先が入力と完全に同一の位置を指す場合（例: `Exp(x, x)`）はインプレース更新として許容されます。部分的に重なるスライス（例: `Negate(x[0..3], x[1..4])`）は暗黙の一時コピーを行わず `ArgumentException` をスローします。
- **積・転置・縮約**: 出力先と入力が一部でも重なる場合はすべて `ArgumentException` になります。
