# API リファレンス一覧

各アセンブリの公開型・メンバー一覧です。`tools/ApiDocGenerator` によってビルド済みのアセンブリと XML ドキュメントから機械生成されます。

| アセンブリ | 公開型数 | 名前空間 |
| --- | ---: | --- |
| [Tuna](Tuna.md) | 55 | `Tuna`, `Tuna.LinearAlgebra`, `Tuna.QuasiRandom`, `Tuna.RandomSource` |
| [Tuna.Statistics](Tuna.Statistics.md) | 42 | `Tuna.Statistics.Distributions` |
| [Tuna.SignalProcessing](Tuna.SignalProcessing.md) | 17 | `Tuna.SignalProcessing` |
| [Tuna.Optimization](Tuna.Optimization.md) | 80 | `Tuna.Optimization`, `Tuna.Optimization.Evolutionary`, `Tuna.Optimization.Global`, `Tuna.Optimization.Gradient`, `Tuna.Optimization.Proximal`, `Tuna.Optimization.Stochastic` |
| [Tuna.Integration](Tuna.Integration.md) | 10 | `Tuna.Integration` |
| [Tuna.Differentiation](Tuna.Differentiation.md) | 16 | `Tuna.Differentiation` |
| [Tuna.Probabilistic](Tuna.Probabilistic.md) | 43 | `Tuna.Probabilistic.Inference` |
| [Tuna.NeuralNetwork](Tuna.NeuralNetwork.md) | 12 | `Tuna.NeuralNetwork`, `Tuna.NeuralNetwork.IO` |

## 再生成コマンド

```bash
dotnet build Tuna.slnx -c Release --no-restore
dotnet run --project tools/ApiDocGenerator -- --configuration Release
```
