namespace Tuna.Tests.SignalProcessing;

/// <summary>
/// <see cref="LinearAlgebra.LinearAlgebraSettings.MaxDegreeOfParallelism"/> を書き換えるテストをまとめ、他のテストと同時に走らせない。
/// コレクションはテストのアセンブリごとなので、コアのテスト（Tuna.Tests の ParallelismTests.cs）と同じ名前で定義する。
/// </summary>
[CollectionDefinition(nameof(ParallelismCollection), DisableParallelization = true)]
public class ParallelismCollection;
