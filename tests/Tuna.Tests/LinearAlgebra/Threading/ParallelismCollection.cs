using Tuna.LinearAlgebra;

namespace Tuna.Tests.LinearAlgebra;

/// <summary>
/// <see cref="LinearAlgebraSettings.MaxDegreeOfParallelism"/> を書き換えるテストは他と同時に走らせない
/// </summary>
[CollectionDefinition(nameof(ParallelismCollection), DisableParallelization = true)]
public class ParallelismCollection;
