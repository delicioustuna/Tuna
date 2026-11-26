# 特殊関数の実装方針

## 1. 定義域外・極・特殊値のポリシー

- 実数オーバーロードでは、IEEE 754 および `System.Math` に準拠し、定義域外・極・不定形に対して例外をスローせず、`double.NaN`・`double.PositiveInfinity`・`double.NegativeInfinity` を返します。
- 整数の次数・インデックスが仕様上の範囲外となる一部の関数（例: 零点番号 $k \le 0$ など）のみ `ArgumentOutOfRangeException` をスローします。
- 関数値が `double` の表現範囲を超える・アンダーフローする領域では、利用者が対数スケールやスケーリング済み値を選べるよう、`Log*`・`*Scaled`・`*Sf`/`*Q` バリアントを揃えています。

## 2. `XMath.Special` の配置と静的初期化順

- **入れ子クラスへの配置**: 特殊関数は 200 種を超えるスカラー専用 API であるため、入口を `XMath` に統一しつつ配列対応演算が並ぶ `XMath` 直下の IntelliSense を圧迫しないよう、`XMath.InPlace` / `XMath.Operators` と揃えたネスト静的クラス `XMath.Special` に配置しています。`Tuna.Differentiation` のスカラーノード向け特殊関数も `[EditorBrowsable(EditorBrowsableState.Never)]` 付きの `XMathSpecialExtensions`（`extension(XMath.Special)`）から同じ入口へ拡張しています。
- **ファイル配置と静的初期化順**: 実装ファイルは `tools/SpecialFunctionTests` の関数群ごとの依存解析マッピングを維持するため、`src/Tuna/SpecialFunctions/<関数群>.cs` の `partial class` 構成を保っています。C# の仕様上、複数ファイルにまたがる静的フィールド初期化子の実行順序は規定されないため、他の係数表に依存して構築される係数表（`LogGamma1pCoefficients`, `EulerMaclaurinCoefficients`, 倍々精度 Stirling 係数など）は、すべてのフィールド初期化子の後に実行される静的コンストラクタ `static Special()` 内で初期化しています。

## 3. 数値精度の確保手法

全関数を通じて倍精度（`double`）の有効桁を広く保つため、以下の手法を組み合わせています。

- **倍々精度（double-double）演算**: 指数部の大きな相殺や高次漸化式（ガンマ関数の Stirling 展開、大次数ベッセル関数の位相・指数補正、合流超幾何関数 $U(a, b, x)$ の下向き漸化式など）で、倍精度の中間丸め誤差が最終結果の相対誤差へ直結する箇所に限定して使用しています。
- **引数縮小の誤差補償**: $\sin(\pi x)$ や $2\pi$ での還元（`PiReduction.cs`）および二項係数の反射公式では、引数差の丸め誤差項（hi/lo 分割）を保持して極や零点近傍の桁落ちを防いでいます。
- **仮数・指数分離**: 大次数変形ベッセル関数 $I_\nu(x), K_\nu(x)$ など、スケーリング値 $e^{-x} I_\nu(x)$ がアンダーフローしても $I_\nu(x)$ 自体は倍精度範囲内に収まる領域では、内部的に仮数と 2 の冪指数に分離して計算し、最終段階で合成します。
- **アルゴリズムの領域分割**:
  - 不完全ベータ関数: DiDonato & Morris (Algorithm 708) に基づき、$\lambda = a - (a+b)x$ の符号とパラメータ範囲に応じて級数・連分数（BFRAC）・漸近展開を切り替えます。
  - ベッセル関数: 次数・引数に応じて級数展開、修正 Lentz 法による連分数、Debye 展開（DLMF 10.19.3）、Airy 型一様漸近展開（DLMF 10.20.4–10.20.5、$z = 1$ 近傍の $0/0$ を回避する変換付き）を切り替えます。
  - Faddeeva 関数: 虚部 $y$ の大きさに応じて極補正項を省略した正項和へ切り替え、桁落ちを回避します。
  - クーロン波動関数: Barnett の COULFG と同様に修正 Lentz 法で連分数比 $F'/F$ を求め、分母符号の反転回数から符号を決定します。
- **自動生成テーブル**: 誤差関数の有理近似係数（`ErrorFunction.Generated.cs`）や対数ガンマの負の零点近傍展開係数（`LogGammaNegativeZeroExpansions.Generated.cs`）は専用ツールで高精度計算してソース生成しています。
