# `XMath.Special` の使い方

`Tuna.XMath.Special` は、ガンマ関数・ベッセル関数・誤差関数・不完全ガンマ / ベータ関数・超幾何関数・楕円積分・直交多項式などの特殊関数をスカラーの静的メソッドとして提供します。

`XMath.Special.LogGamma(x)` のように呼び出せるほか、`using static Tuna.XMath.Special;` を使うと直接呼び出せます（`using static Tuna.XMath;` を使っている場合は `Special.LogGamma(x)` と記述できます）。

```csharp
using System.Numerics;
using static Tuna.XMath.Special;
```

## ライブラリ固有の仕様

### 1. 定義域外・極での振る舞い（例外ではなく IEEE 754 特殊値）

実数オーバーロードでは、IEEE 754 および `System.Math` の慣習に従い、定義域外の入力や極に対して例外をスローせず、関数に応じて `double.NaN`・`double.PositiveInfinity`・`double.NegativeInfinity` を返します（整数のインデックスや次数が範囲外となる一部の関数は `ArgumentOutOfRangeException` をスローします）。

### 2. 対数スケール版・補関数版・スケーリング版の命名規則

大きな引数や確率分布の極端な裾では、関数値が `double` の範囲を超えてオーバーフローまたは 0 にアンダーフローします。Tuna では用途に応じて次のバリアントを用意しています。

| 接頭辞 / 接尾辞 | 目的 | 例 |
| --- | --- | --- |
| `Log*` | 値の自然対数を直接計算（オーバーフロー・アンダーフロー回避） | `LogGamma`, `LogBinomial`, `LogBesselI`, `LogRegularizedGammaQ`, `NormalLogSf` |
| `*Sf` / `*Q` / `Erfc` | 上側確率・補関数 `1 - F(x)` を桁落ちなく直接計算 | `NormalSf`, `RegularizedGammaQ`, `Erfc` |
| `*Scaled` | 指数増大・減衰因子を除いたスケーリング値 | `BesselIScaled`, `BesselKScaled`, `AiryAiScaled`, `Erfcx` |
| `Inverse*` | 逆関数（分位点など） | `InverseErf`, `InverseNormalCdf`, `InverseNormalLogSf`, `InverseRegularizedGammaP` |
| `*Zero` | 関数の零点 | `BesselJZero`, `BesselYZero`, `AiryAiZero` |

```csharp
// 通常の Gamma(200) は PositiveInfinity にオーバーフローするが、対数版なら有限値が得られる
double logG = LogGamma(200);        // ≈ 857.934

// 1 - Erf(6) は桁落ちで 0 になるため、Erfc を使う
double tail = Erfc(6);              // ≈ 2.152e-17

// 標準正規分布の深い裾: NormalSf(40) は 0 にアンダーフローするが、対数版と逆関数なら精度を保てる
double logP = NormalLogSf(40);      // ≈ -804.608
double z = InverseNormalLogSf(logP); // ≈ 40.0

// 変形ベッセル関数 I: 大きな引数ではスケーリング版または対数版を使う
double scaled = BesselIScaled(1, 800); // e^(-x) * I_1(x)
double logI = LogBesselI(1, 800);
```

### 3. 複素数オーバーロード

ガンマ関数・ポリガンマ関数・ベッセル関数・誤差関数 / Faddeeva 関数・ゼータ関数・超幾何関数など主要な関数には `System.Numerics.Complex` を受け取るオーバーロードがあります。

```csharp
Complex g = Gamma(new Complex(0.5, 1.0));
Complex w = Faddeeva(new Complex(1.0, 1.0));
```

## 主な関数カテゴリ

- **ガンマ・ベータ・組み合わせ**: `Gamma`, `LogGamma`, `Digamma`, `Polygamma`, `Beta`, `LogBeta`, `RegularizedGammaP/Q`, `RegularizedIncompleteBeta`, `Binomial`, `Factorial`, `BarnesG`, `MultivariateGamma`
- **誤差関数・正規分布**: `Erf`, `Erfc`, `Erfcx`, `Faddeeva`, `NormalPdf`, `NormalCdf`, `NormalSf`, `NormalLogCdf`, `NormalLogSf`, `InverseNormal*`, `OwensT`, `BivariateNormalCdf`
- **ベッセル・エアリー・ストルーヴェ・ケルビン**: `BesselJ/Y/I/K`, `SphericalBessel*`, `ModifiedSphericalBessel*`, `AiryAi/Bi`, `ScorerGi/Hi`, `StruveH/L`, `Kelvin*`
- **超幾何・直交多項式**: `Hypergeometric0F1/1F1/2F1/U`, `LegendreP/Q`, `HermiteH`, `LaguerreL`, `ChebyshevT/U`, `JacobiP`, `GegenbauerC`
- **楕円・積分・その他**: `EllipticK/E/F/Pi`, `JacobiSn/Cn/Dn`, `JacobiTheta*`, `ExpIntEi/E1`, `SinIntegral`, `CosIntegral`, `FresnelS/C`, `Zeta`, `Polylogarithm`, `LambertW0/Wm1`, `WrightOmega`, `Wigner3j/6j/9j`, `Mathieu*`, `MittagLeffler`

全関数の一覧は [Tuna API リファレンス (`XMath.Special`)](api/Tuna.md#static-class-xmathspecial) を参照してください。

## ベクトル・行列への適用

`XMath.Special` の各メソッドはスカラー関数です。ベクトルや行列にまとめて適用したい場合は、次のいずれかを使います。

1. **演算子構造体を作って `XMath.Map` / `XMath.Zip` に渡す**（中間配列を作らず高速）
   ```csharp
   readonly struct LogGammaOp : IUnaryOperator<double>
   {
       public static double Invoke(double x) => XMath.Special.LogGamma(x);
   }

   readonly struct BesselJOp(double order) : IStatefulUnaryOperator<double>
   {
       public double Invoke(double x) => XMath.Special.BesselJ(order, x);
   }

   Vector<double> lg = XMath.Map<LogGammaOp>(x);
   Vector<double> j1 = XMath.Map(x, new BesselJOp(1.0));
   ```
2. **自分の `partial` クラスに `[ElementwiseOperation]` を付けてラッパーを自動生成する**（繰り返し使う場合。詳しくは [`XMath` の使い方](xmath.md#ユーザー定義演算子と-source-generator) を参照）
