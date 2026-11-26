# 自動微分の設計（`Tuna.Differentiation`）

## 1. テープ構造とライフサイクル

- **リバースモード Wengert テープ（`Tape`）**: スカラー演算は CSR 形式の辺配列（`edgeEnds`, `inputIndices`, `derivatives`, `adjoints`）に、配列演算は要素型別の `ArrayStore<float>` / `ArrayStore<double>` に記録します。各 `ArrayRecord` は記録時点のスカラー `nodeCount` を `Position` として保持しており、`Tape.Backward` はスカラーノード `Position - 1` を処理する直前に該当する配列の逆伝播を挟み込むことで、独立したトポロジカルソートなしにスカラーと配列の混在グラフを処理します。
- **世代番号による無効化**: `Tape` は `Reset()` ごとにインクリメントされる 32-bit の `generation` を持ち、すべてのノード（`ScalarNode`, `VectorNode<T>`, `MatrixNode<T>`, 分解ノード）に記録します。演算時およびビュー参照時に世代番号を検証し、`Reset()` 後の古いノードの誤用を防ぎます。
- **単一パス・非スレッドセーフ設計**: 1 回の記録に対して `Backward` を高々 1 回呼び出してから `Reset()` する前提です。順伝播と逆伝播が重ならないため、損失関数や逆二次形式の順伝播で逆伝播用スクラッチバッファを安全に再利用できます。
- **ゼロコピー `ArrayView` と演算ディスパッチ**: `ArrayView` は連続配列・部分行列・転置・ベクトルブロードキャスト（`Repeat`）をコピーなしで表現します。`ArrayStore.ElementwiseBackward` は `ArrayOperation` 列挙型の連続配置（単項 → 活性化 → 二項 → スカラー演算 → 縮約・Softmax → 損失 → 線形代数）を前提にディスパッチします。
- **`XMath` / `XMath.Special` 拡張としての公開**: スカラー・ベクトル・行列・テンソルノードの演算は単一の `partial class XMathExtensions`（`extension(XMath)`）に集約し、スカラー特殊関数は `XMathSpecialExtensions`（`extension(XMath.Special)`）に配置しています。いずれも `[EditorBrowsable(EditorBrowsableState.Never)]` を付与し、コア演算と同じ `XMath.*` / `XMath.Special.*` の入口から統一的に呼び出せる構成としています。

## 2. メモリプーリングと保持上限

- **スロットキャッシュと長さ別スペア（`ArrayStore<T>`）**: プール済みの値・勾配配列をスロット番号に紐付けたまま `Reset()` をまたいで再利用し、定常状態の反復実行でアロケーションゼロを実現します（形状が変わらない場合は `reviewRetention` が走査を省略）。形状が変わった配列は長さ別の `spares` スタックへ退避します。
- **LRU による保持上限（`RetentionFactor = 3`）**: `Reset()` 時に未使用配列の総要素数が `3 * peakBorrowed` を超えると、記録 `epoch` による LRU 順（$O(k \log k)$）で解放します。係数 3 とすることで、2 種類の形状を交互に評価するパターン（最大 $2\times$ ピーク）に余裕を持たせつつ、無制限なメモリ膨張を防いでいます（`DecompositionPool` とは別集計）。
- **単調増加スクラッチバッファ**: `ArrayStore<T>.scratch[0..8]` はブロードキャスト逆伝播で長さ $n$ のベクトルから $n^2$ 要素へ展開されるケースがあるため `RetentionFactor` 対象外とし、ピークサイズまで単調増加させます。排他的なステージ間ではスクラッチスロットを共有します（`BlockSecond == LeftDerivative`, `DecompositionFirst == SolveAdjoint`）。
- **初回書き込みの上書き最適化（`BeginGradient`）**: プール済み勾配バッファに対する初回寄与がスロット全体を覆う場合（`view.StorageLength == gradient.Length`）はゼロ初期化を省略します。一方、`Parameter<T>`（`ExternalGradientFlag`）は常に加算蓄積します。
- **外部コンテナ・分解因子のゼロコピー参照**: `AddConstant` は `Vector<T>` / `Matrix<T>` を、`AddParameter` は `Parameter<T>` の値・勾配バッファを、`AddFactor` は分解因子の格納領域をコピーせずに直接参照し、`Reset()` 時にライブ参照配列をクリアして GC を妨げないようにします。
- **OOM 安全な拡張（`Grow()`）**: 並列なスロットメタデータ配列はローカル変数へすべてリサイズしてからフィールドへ反映し、`OutOfMemoryException` 時に配列長の不整合が残らないようにしています。

## 3. ビット一致決定性と IEEE 754・混合精度ルール

- **順伝播のコア一致**: 順伝播は `XMath` の `destination` オーバーロードおよびコア分解 API へ委譲し、非微分呼び出しとビット単位で一致させています。転置葉ノード（`X.T`）は `XMath` とビット一致させるため元の格納向きを保持し、`Tape.Gradient` は常に論理的な列優先順で書き出します。
- **命令セット幅・スレッド数に依存しない逆伝播**:
  - `ElementwiseKernels<T>`・`ElementwiseDerivatives`・`ActivationDerivatives`・`LossKernels<T>` は FMA を意図的に使わず、SIMD 経路とスカラー端数経路で同一の演算式・選択式を用いて命令セット幅に依存しないビット一致を保っています。
  - `ArrayStore.AddScaled` も `Blas.Axpy` ではなく乗算と加算を分離した `ElementwiseKernels<T>.AddScaled` を使用しています。
  - 小バッチのホットループには `MethodImplOptions.AggressiveOptimization` を付与して Tier-0 JIT の初回遅延を回避しています。
  - 線形代数逆伝播のブロック幅（`CholeskyBlock = 64`）は行列次数のみに依存し、スレッド数に依存しません。
- **ゼロ寄与のスキップと IEEE 754 伝播**:
  - 到達不能なレコード（`!Touched`）とゼロのスカラー随伴値（`float` 入力に対し `0f` へ丸まる `double` 随伴値を含む）は逆伝播をスキップします。
  - 要素演算では `MaskedProduct` / `MaskedScale` により、随伴値または局所導関数が厳密に 0 なら寄与を加算せず、`0 * Infinity` や `0 * NaN` による汚染を防ぎます（`ScalarNode` と一貫）。
  - 縮約・Softmax・損失・線形代数は到達レコード内で IEEE 754 演算に従います。`Blas.Ger` や非転置 `Blas.Gemv` は乗数が 0 の列をスキップし得るため、BLAS 呼び出し前にゼロ要素を事前走査（`NextZero` / `FirstSkippedOuterColumn`）し、非有限要素を含む列についてのみ `0 * non-finite = NaN` を復元します。
- **混合精度（`float` 配列と `double` スカラー）**: `float` 配列からのスカラー出力は `double` へ昇格します。`float` 配列とスカラーの演算は順・逆伝播ともスカラーを `float` へ丸め、`float` 平均の逆伝播は `g / n` を `double` で除算してから `float` へ 1 回だけ丸めます。`float` の `Pow` 逆伝播は `x^y` を `double` で再計算してから `XMathExtensions` の冪導関数を呼びます。

## 4. 数値安定性の工夫

- **Logistic / Tanh / SiLU の裾**: $t = e^{-|x|}$ を用いて $t / (1 + t)^2$ や $(2t / (1 + t^2))^2$ として評価し、$\pm 1$ 付近の桁落ちや早期アンダーフローを回避します。SiLU では $(1 + x) + t$ と結合し、導関数零点（$x \approx -1.2785$）付近の $[-2, -0.5]$ で $1 + x$ を厳密に評価します。
- **負の裾の指数シフト**: ELU・SELU・SiLU・GELU・GELU-tanh は負の裾で指数を整数 `TailShift = 64`（ELU の大 $|\alpha| > e^{27}$ ではさらに $k = \min(\lceil \ln|\alpha| \rceil, 704 / 64)$）だけシフトしてから最後に $e^{-64}$ を掛けることで、非正規化数丸めを高々 1 回に抑えています。
- **倍々精度の引数評価（`ActivationDerivatives`）**: `Gelu` は FMA で $x^2 = \mathrm{hi} + \mathrm{lo}$ に分割し、`GeluTanh` は $z = k(x + c x^3)$ を倍々精度 `(z, zLow)` で保持して大 $|z|$ での $e^{-|z|}$ の相対誤差爆発を防いでいます。
- **`Log1mExp` / `LogAddExp` / `LogSubExp` / `LogSumExp`**: $-\ln 2$ を境界に `exp(x)` と `expm1(-x)` を切り替え、`LogSumExp` 系では TwoDiff 誤差補償（`DifferenceError`）と $-708$ 未満での半指数分割を行います。最大値が $+\infty$ または全要素が $-\infty$ の場合は `LimitWeights` で極限重みを与え、`SoftmaxWeights` / `LossWeights` は NaN 検出により必要列のみを補正します。
- **`Pow` / `Divide` / `Reciprocal`**: `PowerDerivativeMagnitude` は仮数と指数を分離（`ILogB` / `ScaleB`）し、オーバーフロー・アンダーフロー時は $(x^{y/4})^4$ にフォールバックします。`DenominatorDerivatives` は $-(z/y)$ と指数分離フォールバックを用い、`Reciprocal` は $-y^2$ を用いて $x^2$ のオーバーフローを回避します。
- **損失関数**: `SoftmaxCrossEntropy` は $L = \mathrm{ColumnLogSoftmax}(Z)$ を計算して $P - \mathrm{onehot}(t)$ または $L$ を勾配不要スロットにキャッシュし、逆伝播では $P = \exp(L)$ から直接復元（NaN 列のみ `LimitWeights` で補正）します。`LogisticCrossEntropy` は $\mathrm{softplus}(-|\eta|)$ と $\mathrm{logistic}(-|\eta|)$ に符号選択を組み合わせて $0 \times \infty$ と裾の桁落ちを防いでいます。

## 5. 線形代数・行列分解の逆伝播（`double` 専用）

- **インプレース再利用プール（`DecompositionPool`）**: `LuDecomposition`, `CholeskyDecomposition`, `QrDecomposition`, `SymmetricEigenDecomposition`, `SingularValueDecomposition` を `(kind, rows, cols)` キーでプールし、`Recompute` / `TryRecompute` でインプレース更新します（`3 * peak` 超過分は LRU 解放）。多出力分解（QR・対称固有値・SVD）は複数因子スロットを 1 つのテープレコードへ束ね、到達した出力（`Touched`）からのみ勾配を計算します。
- **対称・三角行列の扱い**: `Cholesky` と `SymmetricEigen` は下三角のみを読み取り、勾配も下三角へ集約します（$i > j$ の $dA_{ij}$ が対称摂動微分 $P_{ij} + P_{ji}$、上三角は 0）。三角求解は参照される三角領域のみに勾配を加算します。
- **解析的逆伝播とオーバーフロー回避（`LinearAlgebraBackward`）**:
  - **LU**: $U = A^{-\top} G$, $dB \mathrel{+}= U$, $dA \mathrel{-}= U X^\top$（転置求解時は入れ替え）。`LogAbsDeterminant` は $ds \cdot A^{-\top}$（特異行列では NaN）。
  - **Cholesky**: 因子逆伝播は転置上三角作業領域 $W = \mathrm{tril}(dL)^\top$ 上で逆順左参照ブロック版 Murray (2016) 法を実行し、すべての三角求解をインプレース左求解に統一します。対角成分 $\tfrac{1}{2}(S + S^\top)_{jj}$ は $2 S_{jj}$ のオーバーフローを避けるため $S^\top$ から直接読み取ります。`CholeskyLogDeterminant` と `CholeskyInverseQuadraticForm` では `(2 * adjoint) * v` ではなく `2 * (adjoint * v)` の順で掛けます。
  - **QR**: Thin QR 微分（Liao et al. 2019）。横長行列（$m < n$）では $R = [X \mid V]$, $A = [\cdot \mid Y]$ に分割し、$dR$ の狭義下三角成分は無視します。
  - **SymmetricEigen**: Giles (2008) の公式を用い、$C_{ij} = 0$ をスキップします。有限固有値差がオーバーフローする場合は `HalvedQuotient`（$0.5 c / (0.5 \lambda_j - 0.5 \lambda_i)$）を使用します。
  - **SVD**: Townsend (2016) の Thin SVD 微分を用い、長方形行列の直交補空間射影を $N'$ へ折り込みます。$(s_j - s_i)(s_j + s_i)$ や分子が非正規化・オーバーフロー領域に入る場合は `ScaledSingularValueQuotients` / `SplitProduct` で仮数と 2 進指数（`ILogB` / `ScaleB`）を分離し、中間オーバーフローによる商の破綻を防ぎます。
