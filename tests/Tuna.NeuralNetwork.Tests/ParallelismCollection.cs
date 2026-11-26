namespace Tuna.NeuralNetwork.Tests;

/// <summary>
/// <see cref="LinearAlgebra.LinearAlgebraSettings.MaxDegreeOfParallelism"/> を書き換えるテストをまとめ、他のテストと同時に走らせない。
/// </summary>
[CollectionDefinition(nameof(ParallelismCollection), DisableParallelization = true)]
public class ParallelismCollection;
