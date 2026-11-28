using System.Collections;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Tuna.Differentiation;
using Tuna.Differentiation.Tests;
using Tuna.LinearAlgebra;

namespace Tuna.Tests.Differentiation;

/// <summary>
/// 形を変え続ける記録で、テープが持つ配列の量が1回の記録の量の定数倍に収まること
/// </summary>
/// <remarks>
/// テープの回収前後のヒープ使用量を比較するため、他のテストと同時に実行しない。
/// </remarks>
[Collection(nameof(ParallelismCollection))]
public class ArrayRetentionTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void PointwiseOperationsReuseTapeStorageAfterReset(bool pointwise)
    {
        CheckPointwiseReuse<float>(pointwise);
        CheckPointwiseReuse<double>(pointwise);
    }

    private static void CheckPointwiseReuse<T>(bool pointwise) where T : unmanaged, System.Numerics.IFloatingPointIeee754<T>
    {
        var data = Vector<T>.Generate(64, static i => T.CreateChecked(i));
        var tape = new Tape();
        Assert.Throws<ArithmeticException>(() => Iterate(fail: true));
        for (int i = 0; i < 64; i++)
        {
            Iterate(fail: false);
        }

        AssertNoAllocation(() =>
        {
            for (int i = 0; i < 32; i++)
            {
                Iterate(fail: false);
            }
        });

        void Iterate(bool fail)
        {
            tape.Reset();
            var leaf = tape.Leaf(data);
            tape.Backward(PointwiseOperationTests.RecordRetainedModel(tape, leaf, tape.Leaf(1), pointwise, fail));
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void FailedPointwiseOperationsKeepStorageBoundedBeforeReset(bool pointwise)
    {
        var data = RandomVector(pointwise ? 192 : 16_384);
        long clean = RetainedBytes(tape => Record(tape, changeShape: false));
        long failing = RetainedBytes(tape => Record(tape, changeShape: true));
        Assert.True(failing <= 5 * Math.Max(0, clean) + 16_384, $"failed recordings retained {failing} B, clean recording {clean} B");

        void Record(Tape tape, bool changeShape)
        {
            var leaf = tape.Leaf(data);
            var scale = tape.Leaf(1);
            if (changeShape)
            {
                for (int count = data.Length - 160; count <= data.Length; count++)
                {
                    var slice = leaf[..count];
                    Assert.Throws<ArithmeticException>(() => PointwiseOperationTests.RecordRetainedModel(tape, slice, scale, pointwise, fail: true));
                }
            }

            tape.Backward(PointwiseOperationTests.RecordRetainedModel(tape, leaf, scale, pointwise, fail: false));
            Assert.Equal(1, tape.GetGradient(leaf)[^1]);
        }
    }

    [Fact]
    public void DiscardedOperationsReleaseEmptySparesWithinOneRecording()
    {
        var store = new ArrayStore<double>();
        for (int length = 1024; length <= 1280; length++)
        {
            int slot = store.AddPooled(length, 1, requiresGradient: false);
            store.DiscardOperationSlots(slot);
        }

        const BindingFlags fields = BindingFlags.Instance | BindingFlags.NonPublic;
        var storeType = typeof(ArrayStore<double>);
        var spares = Assert.IsAssignableFrom<IDictionary>(storeType.GetField("spares", fields)!.GetValue(store));
        var emptySpares = Assert.IsAssignableFrom<ICollection>(storeType.GetField("emptySpares", fields)!.GetValue(store));
        var trimOrder = Assert.IsAssignableFrom<Array>(storeType.GetField("trimOrder", fields)!.GetValue(store));
        // The 3 × 1280 element limit fits at most three buffers of length 1024 or more after trimming.
        Assert.InRange(spares.Count, 0, 3);
        Assert.Empty(emptySpares);
        Assert.InRange(trimOrder.Length, 0, 4);
    }

    [Fact]
    public void CholeskyLogDeterminantReusesTapeStorageAfterReset()
    {
        var matrix = SymmetricPositiveDefinite(6);
        var tape = new Tape();
        for (int i = 0; i < 64; i++)
        {
            Iterate();
        }

        AssertNoAllocation(() =>
        {
            for (int i = 0; i < 32; i++)
            {
                Iterate();
            }
        });

        void Iterate()
        {
            tape.Reset();
            var a = tape.Leaf(matrix);
            tape.Backward(XMath.CholeskyLogDeterminant(a));
            _ = tape.GetGradient(a);
        }
    }

    [Fact]
    public void ConstantCholeskyLogDeterminantReusesWorkspaceDuringRecording()
    {
        var tape = new Tape();
        var leaf = tape.Leaf(Matrix<double>.FromDiagonal([4, 16]));
        var constant = tape.Constant(Matrix<double>.FromDiagonal([9, 25]));
        ScalarNode loss = XMath.CholeskyLogDeterminant(leaf);
        ScalarNode value = XMath.CholeskyLogDeterminant(constant);
        Assert.False(value.RequiresGradient);
        Assert.Equal(XMath.CholeskyLogDeterminant(constant.Value), value.Value);

        EvaluateConstants();
        AssertNoAllocation(EvaluateConstants);
        tape.Backward(loss);

        var gradient = tape.GetGradient(leaf);
        Assert.Equal(0.25, gradient[0, 0]);
        Assert.Equal(0.0625, gradient[1, 1]);

        void EvaluateConstants()
        {
            for (int i = 0; i < 64; i++)
            {
                _ = XMath.CholeskyLogDeterminant(constant);
            }
        }
    }

    [Fact]
    public void CrossAndTraceReuseTapeStorageAfterReset()
    {
        CheckCrossAndTraceAllocation<double>();
        CheckCrossAndTraceAllocation<float>();
    }

    private static void CheckCrossAndTraceAllocation<T>() where T : unmanaged, System.Numerics.IFloatingPointIeee754<T>
    {
        var u = Vector<T>.Generate(3, i => T.CreateChecked(i + 1));
        var v = Vector<T>.Generate(3, i => T.CreateChecked(2 - i));
        var a = Matrix<T>.Generate(4, 5, (i, j) => T.CreateChecked(i - j));
        var tape = new Tape();
        for (int i = 0; i < 64; i++)
        {
            Iterate();
        }

        AssertNoAllocation(() =>
        {
            for (int i = 0; i < 32; i++)
            {
                Iterate();
            }
        });

        void Iterate()
        {
            tape.Reset();
            var x = tape.Leaf(u);
            var y = tape.Leaf(v);
            var matrix = tape.Leaf(a);
            tape.Backward(XMath.Sum(XMath.Cross(x, y)) + XMath.Trace(matrix));
            _ = tape.GetGradient(x);
            _ = tape.GetGradient(y);
            _ = tape.GetGradient(matrix);
        }
    }

    [Fact]
    public void OuterProductReusesTapeStorageAfterReset()
    {
        CheckOuterProductAllocation<double>();
        CheckOuterProductAllocation<float>();
    }

    [Fact]
    public void WedgeProductReusesTapeStorageAfterReset()
    {
        CheckWedgeAllocation<double>();
        CheckWedgeAllocation<float>();
    }

    private static void CheckWedgeAllocation<T>() where T : unmanaged, System.Numerics.IFloatingPointIeee754<T>
    {
        const int n = 19;
        var u = Vector<T>.Generate(n, i => T.CreateChecked(i - 9));
        var v = Vector<T>.Generate(n, i => T.CreateChecked(6 - i));
        var weights = Matrix<T>.Generate(n, n, (i, j) => T.CreateChecked(i - j));
        var tape = new Tape();
        for (int i = 0; i < 64; i++)
        {
            Iterate();
        }

        AssertNoAllocation(() =>
        {
            for (int i = 0; i < 32; i++)
            {
                Iterate();
            }
        });

        void Iterate()
        {
            tape.Reset();
            var x = tape.Leaf(u);
            var y = tape.Leaf(v);
            var wedge = XMath.Wedge(x, y);
            tape.Backward(XMath.Sum(XMath.ColumnSums(XMath.Hadamard(wedge, tape.Constant(weights)))));
            _ = tape.GetGradient(x);
            _ = tape.GetGradient(y);
        }
    }

    [Theory]
    [InlineData(9, 6, 0)]
    [InlineData(40, 60, 0)]
    [InlineData(4, 5, -400)]
    public void SvdSolvesAndPseudoInverseAllocateNothingAfterReset(int m, int n, int exponent)
    {
        var a = RandomRectangle(m, n);
        foreach (ref double value in a.AsSpan())
        {
            value = Math.ScaleB(value, exponent);
        }

        var b = RandomVector(m);
        var rhs = RandomRectangle(m, 9);
        var tape = new Tape();
        for (int i = 0; i < 32; i++)
        {
            Iterate();
        }

        AssertNoAllocation(() =>
        {
            for (int i = 0; i < 32; i++)
            {
                Iterate();
            }
        });

        void Iterate()
        {
            tape.Reset();
            var coefficient = tape.Leaf(a);
            var single = tape.Leaf(b);
            var multiple = tape.Leaf(rhs);
            var svd = SingularValueDecompositionNode.Compute(coefficient);
            ScalarNode loss = XMath.Sum(svd.Solve(single))
                + XMath.Sum(XMath.ColumnSums(svd.Solve(multiple)))
                + XMath.Sum(XMath.ColumnSums(svd.PseudoInverse()));
            tape.Backward(loss);
            _ = tape.GetGradient(coefficient);
            _ = tape.GetGradient(single);
            _ = tape.GetGradient(multiple);
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TanhReusesTapeStorageAfterReset(bool approximate)
    {
        CheckTanhAllocation<double>(approximate);
        CheckTanhAllocation<float>(approximate);
    }

    private static void CheckTanhAllocation<T>(bool approximate) where T : unmanaged, System.Numerics.IFloatingPointIeee754<T>
    {
        var input = Matrix<T>.Generate(13, 17, (i, j) => T.CreateChecked(i - j));
        var tape = new Tape();
        for (int i = 0; i < 64; i++)
        {
            Iterate();
        }

        AssertNoAllocation(() =>
        {
            for (int i = 0; i < 32; i++)
            {
                Iterate();
            }
        });

        void Iterate()
        {
            tape.Reset();
            var x = tape.Leaf(input);
            tape.Backward(XMath.Sum(XMath.ColumnSums(XMath.Tanh(x[1.., 1..].T, approximate))));
            _ = tape.GetGradient(x);
        }
    }

    private static void CheckOuterProductAllocation<T>() where T : unmanaged, System.Numerics.IFloatingPointIeee754<T>
    {
        var u = Vector<T>.Generate(19, i => T.CreateChecked(i - 9));
        var v = Vector<T>.Generate(13, i => T.CreateChecked(6 - i));
        var tape = new Tape();
        for (int i = 0; i < 64; i++)
        {
            Iterate();
        }

        AssertNoAllocation(() =>
        {
            for (int i = 0; i < 32; i++)
            {
                Iterate();
            }
        });

        void Iterate()
        {
            tape.Reset();
            var x = tape.Leaf(u);
            var y = tape.Leaf(v);
            tape.Backward(XMath.Sum(XMath.ColumnSums(XMath.Outer(x, y))));
            _ = tape.GetGradient(x);
            _ = tape.GetGradient(y);
        }
    }

    [Fact]
    public void CustomOperationSavedValuesAndBroadcastGradientsShareBoundedStorage()
    {
        const int columns = 64;
        const int largest = 240;
        long sameShape = RetainedBytes(tape =>
        {
            var x = RandomVector(largest);
            for (int i = 0; i < 3; i++)
            {
                CustomOperationTests.RecordBroadcastSine(tape, x, columns);
            }

            tape.Reset();
        });
        long growing = RetainedBytes(tape =>
        {
            for (int n = 96; n <= largest; n++)
            {
                CustomOperationTests.RecordBroadcastSine(tape, RandomVector(n), columns);
            }

            tape.Reset();
        });

        Assert.True(growing <= 5 * sameShape, $"changing custom-operation shapes retained {growing} B, the largest alone {sameShape} B");
    }

    [Fact]
    public void FailedCustomOperationsDoNotKeepTheirSavedSlotsLive()
    {
        var x = RandomVector(256);
        long clean = RetainedBytes(tape =>
        {
            CustomOperationTests.RecordBroadcastSine(tape, x, 64);
            tape.Reset();
        });
        long failing = RetainedBytes(tape =>
        {
            CustomOperationTests.RecordBroadcastSine(tape, x, 64, failures: 20);
            tape.Reset();
        });

        // A discarded output and saved derivative buffer may be cached, but each failure must reuse their slots.
        Assert.True(failing <= clean + 2 * x.Length * 64 * sizeof(double), $"failed operations retained {failing} B, clean recording {clean} B");
    }

    [Fact]
    public void ConstantCustomOperationsReleaseSavedSlotsWithoutLosingOutputs()
    {
        const int length = 16_384;
        const int repeats = 32;
        var values = new LinearAlgebra.Vector<double>([1]);
        long retained = RetainedBytes(tape =>
        {
            var operation = new AddOneWithSavedWorkspace(length);
            var first = tape.Apply(operation, [tape.Constant(values)], 1);
            var output = first;
            for (int i = 1; i < repeats; i++)
            {
                output = tape.Apply(operation, [output], 1);
            }

            Assert.False(output.RequiresGradient);
            Assert.Equal(2.0, first.Value[0]);
            Assert.Equal(1.0 + repeats, output.Value[0]);
        });

        Assert.True(retained <= 4L * length * sizeof(double), $"constant custom operations retained {retained} B");
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void CustomOperationSavedValuesAndBroadcastGradientsAllocateNothingAfterWarmup(int degree)
    {
        int savedDegree = LinearAlgebraSettings.MaxDegreeOfParallelism;
        try
        {
            if (degree != 0)
            {
                LinearAlgebraSettings.MaxDegreeOfParallelism = degree;
            }

            var tape = new Tape();
            var values = RandomVector(256);
            for (int i = 0; i < 32; i++)
            {
                CustomOperationTests.RecordBroadcastSine(tape, values, 64);
            }

            GC.Collect(2, GCCollectionMode.Forced, blocking: true);
            GC.WaitForPendingFinalizers();
            AssertNoAllocation(() =>
            {
                for (int i = 0; i < 8; i++)
                {
                    CustomOperationTests.RecordBroadcastSine(tape, values, 64);
                }
            });
        }
        finally
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = savedDegree;
        }
    }

    [Fact]
    public void GrowingShapesKeepRetainedStorageBounded()
    {
        // 予備の配列は長さが一致したときだけ使い回すので、上限がなければ n = 1〜120 のすべての長さが残る（同じ形の約 30 倍）。
        // 配列は1回の記録の最大の3倍まで残す。一時領域などの分を足しても5倍に収まる。
        long sameShape = RetainedBytes(tape =>
        {
            for (int i = 0; i < 3; i++)
            {
                SquareModel(tape, RandomMatrix(120));
            }
        });
        long growing = RetainedBytes(tape =>
        {
            for (int n = 1; n <= 120; n++)
            {
                SquareModel(tape, RandomMatrix(n));
            }
        });

        Assert.True(growing <= 5 * sameShape, $"growing shapes retained {growing} B, the largest shape alone {sameShape} B");
    }

    [Fact]
    public void ShorterRecordingsReleaseArraysOfUnusedSlots()
    {
        // 長い配列のスロットの番号が記録ごとに1つずつ前へずれる。使わなくなった後ろのスロットの配列を返さなければ、
        // 記録ごとに長い配列を確保し、後ろのスロットに残し続ける（同じ記録の約 30 倍）。
        var data = RandomVector(3600);
        var start = new LinearAlgebra.Vector<double>([0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7, -0.8]);
        long sameShape = RetainedBytes(tape =>
        {
            for (int i = 0; i < 3; i++)
            {
                ChainModel(tape, start, data, 100);
            }
        });
        long shrinking = RetainedBytes(tape =>
        {
            for (int steps = 100; steps >= 1; steps--)
            {
                ChainModel(tape, start, data, steps);
            }
        });

        Assert.True(shrinking <= 5 * sameShape, $"shrinking recordings retained {shrinking} B, the longest alone {sameShape} B");
    }

    [Fact]
    public void AlternatingRecordingsOfDifferentLengthsAllocateNothing()
    {
        var data = RandomVector(3600);
        var start = new LinearAlgebra.Vector<double>([0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7, -0.8]);
        var tape = new Tape();
        for (int i = 0; i < 3; i++)
        {
            ChainModel(tape, start, data, 10, fingerprint: false);
            ChainModel(tape, start, data, 50, fingerprint: false);
        }

        AssertNoAllocation(() =>
        {
            for (int i = 0; i < 5; i++)
            {
                ChainModel(tape, start, data, 10, fingerprint: false);
                ChainModel(tape, start, data, 50, fingerprint: false);
            }
        });
    }

    [Fact]
    public void CyclingLengthsThatEmptyAndRefillSparesAllocateNothing()
    {
        // 1周の中で予備がいったん空になり、後の記録で同じ長さの配列がまた返る。空の予備の入れ物を捨てると、返すたびに作り直す。
        (int First, int Second)[] cycle = [(100, 110), (100, 120), (101, 130)];
        var vectors = new Dictionary<int, LinearAlgebra.Vector<double>>();
        foreach (int length in new[] { 100, 101, 110, 120, 130 })
        {
            vectors[length] = RandomVector(length);
        }

        var tape = new Tape();
        void RecordCycle()
        {
            foreach (var (first, second) in cycle)
            {
                tape.Reset();
                var a = tape.Leaf(vectors[first]);
                var b = tape.Leaf(vectors[second]);
                tape.Backward(XMath.Sum(a) + XMath.Sum(b));
            }
        }

        for (int i = 0; i < 3; i++)
        {
            RecordCycle();
        }

        AssertNoAllocation(() =>
        {
            for (int i = 0; i < 5; i++)
            {
                RecordCycle();
            }
        });
    }

    [Fact]
    public void ValuesAndGradientsDoNotDependOnReleasedStorage()
    {
        var square = RandomMatrix(50);
        var data = RandomVector(3600);
        var start = new LinearAlgebra.Vector<double>([0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7, -0.8]);
        string expectedSquare = SquareModel(new Tape(), square);
        string expectedChain = ChainModel(new Tape(), start, data, 30);

        var tape = new Tape();
        for (int n = 1; n <= 120; n++)
        {
            SquareModel(tape, RandomMatrix(n));
        }

        Assert.Equal(expectedSquare, SquareModel(tape, square));
        for (int steps = 100; steps >= 1; steps--)
        {
            ChainModel(tape, start, data, steps);
        }

        Assert.Equal(expectedChain, ChainModel(tape, start, data, 30));
        Assert.Equal(expectedSquare, SquareModel(tape, square));
    }

    [Fact]
    public void GrowingDecompositionOrdersKeepRetainedStorageBounded()
    {
        // 分解のオブジェクトは種類と次数が一致したときだけ使い回すので、上限がなければ n = 1〜100 のすべての次数の
        // Cholesky 分解と LU 分解が残る（同じ次数の約 30 倍）。分解も1回の記録の最大の3倍まで残す。
        long sameShape = RetainedBytes(tape =>
        {
            for (int i = 0; i < 3; i++)
            {
                DecompositionModel(tape, SymmetricPositiveDefinite(100));
            }
        });
        long growing = RetainedBytes(tape =>
        {
            for (int n = 1; n <= 100; n++)
            {
                DecompositionModel(tape, SymmetricPositiveDefinite(n));
            }
        });

        Assert.True(growing <= 5 * sameShape, $"growing orders retained {growing} B, the largest order alone {sameShape} B");
    }

    [Fact]
    public void AlternatingDecompositionOrdersAllocateNothing()
    {
        var small = SymmetricPositiveDefinite(10);
        var large = SymmetricPositiveDefinite(40);
        var tape = new Tape();
        for (int i = 0; i < 3; i++)
        {
            DecompositionModel(tape, small, fingerprint: false);
            DecompositionModel(tape, large, fingerprint: false);
        }

        AssertNoAllocation(() =>
        {
            for (int i = 0; i < 5; i++)
            {
                DecompositionModel(tape, small, fingerprint: false);
                DecompositionModel(tape, large, fingerprint: false);
            }
        });
    }

    [Fact]
    public void ValuesAndGradientsDoNotDependOnReleasedDecompositions()
    {
        var matrix = SymmetricPositiveDefinite(50);
        string expected = DecompositionModel(new Tape(), matrix);
        var tape = new Tape();
        for (int n = 1; n <= 100; n++)
        {
            DecompositionModel(tape, SymmetricPositiveDefinite(n));
        }

        Assert.Equal(expected, DecompositionModel(tape, matrix));
        for (int n = 100; n >= 1; n--)
        {
            DecompositionModel(tape, SymmetricPositiveDefinite(n));
        }

        Assert.Equal(expected, DecompositionModel(tape, matrix));
    }

    [Fact]
    public void FailedSolveRecordingsReturnTheBorrowedDecompositions()
    {
        // 逆伝播を終えたテープへの XMath.Solve は、LU 分解を借りる前に例外にする（借りた後だと、分解が返らずに次の記録まで残る）。
        var a = SymmetricPositiveDefinite(40);
        var b = new LinearAlgebra.Vector<double>(40);
        b[0] = 1;
        void Record(Tape tape, int failures)
        {
            tape.Reset();
            var leaf = tape.Leaf(a);
            var right = tape.Constant(b);
            tape.Backward(XMath.Sum(XMath.Solve(leaf, right)));
            for (int i = 0; i < failures; i++)
            {
                Assert.Throws<InvalidOperationException>(() => XMath.Solve(leaf, right));
                Assert.Throws<InvalidOperationException>(() => XMath.Solve(leaf, tape.Constant(a)));
            }
        }

        long clean = RetainedBytes(tape =>
        {
            Record(tape, 0);
            Record(tape, 0);
        });
        long failing = RetainedBytes(tape =>
        {
            Record(tape, 10);
            Record(tape, 0);
        });

        // 返らなかった LU 分解（40 次で 12,800 B）が 20 個残ると 256,000 B 増える。
        Assert.True(failing <= clean + 40 * 40 * sizeof(double), $"failed recordings retained {failing} B, clean recordings {clean} B");
    }

    [Fact]
    public void ConstantSolvesAllocateNothing()
    {
        // 記録しない XMath.Solve は分解をすぐ返すが、借りていた間の量を保持量の上限に数えるので、次の記録でも同じ分解を使い回す。
        var a = SymmetricPositiveDefinite(40);
        var b = new LinearAlgebra.Vector<double>(40);
        var bs = new Matrix<double>(40, 3);
        b[1] = 1;
        bs[2, 1] = 1;
        var tape = new Tape();
        void Record()
        {
            tape.Reset();
            var coefficient = tape.Constant(a);
            XMath.Solve(coefficient, tape.Constant(b));
            XMath.Solve(coefficient, tape.Constant(bs));
        }

        for (int i = 0; i < 3; i++)
        {
            Record();
        }

        AssertNoAllocation(() =>
        {
            for (int i = 0; i < 5; i++)
            {
                Record();
            }
        });
    }

    [Fact]
    public void BroadcastTriangularCoefficientsAllocateNothing()
    {
        // ブロードキャストの係数（L[i, j] = v[i]）は、コアの三角求解が呼ぶたびに写す配置なので、テープの一時領域に写して解く。
        var random = new Random(40);
        var v = new LinearAlgebra.Vector<double>(40);
        var x = new LinearAlgebra.Vector<double>(40);
        var xs = new Matrix<double>(40, 3);
        for (int i = 0; i < 40; i++)
        {
            v[i] = 1 + random.NextDouble();
            x[i] = random.NextDouble() - 0.5;
            xs[i, i % 3] = random.NextDouble() - 0.5;
        }

        var tape = new Tape();
        void Record()
        {
            tape.Reset();
            var coefficient = tape.Leaf(v).Broadcast(40);
            var right = tape.Leaf(x);
            var rights = tape.Leaf(xs);
            ScalarNode loss = XMath.Sum(XMath.SolveLowerTriangular(coefficient, right)) + XMath.Sum(XMath.SolveUpperTriangular(coefficient.T, right))
                + XMath.Sum(XMath.ColumnSums(XMath.SolveLowerTriangular(coefficient, rights))) + XMath.Sum(XMath.ColumnSums(XMath.SolveUpperTriangular(coefficient.T, rights)));
            tape.Backward(loss);
        }

        for (int i = 0; i < 3; i++)
        {
            Record();
        }

        AssertNoAllocation(() =>
        {
            for (int i = 0; i < 5; i++)
            {
                Record();
            }
        });
    }

    [Fact]
    public void GrowingOrthogonalDecompositionShapesKeepRetainedStorageBounded()
    {
        // QR 分解・特異値分解は行数と列数の組、対称固有値分解は次数が一致したときだけ使い回す。
        long sameShape = RetainedBytes(tape =>
        {
            for (int i = 0; i < 3; i++)
            {
                OrthogonalModel(tape, RandomRectangle(60, 40), RandomMatrix(60));
            }
        });
        long growing = RetainedBytes(tape =>
        {
            for (int n = 1; n <= 40; n++)
            {
                OrthogonalModel(tape, RandomRectangle(60, n), RandomMatrix(n + 20));
                OrthogonalModel(tape, RandomRectangle(n + 20, 40), RandomMatrix(60));
            }
        });

        Assert.True(growing <= 5 * sameShape, $"growing shapes retained {growing} B, the largest shape alone {sameShape} B");
    }

    [Fact]
    public void AlternatingOrthogonalDecompositionShapesAllocateNothing()
    {
        var tall = RandomRectangle(12, 7);
        var wide = RandomRectangle(7, 12);
        var small = RandomMatrix(7);
        var large = RandomMatrix(12);
        var tape = new Tape();
        for (int i = 0; i < 3; i++)
        {
            OrthogonalModel(tape, tall, small, fingerprint: false);
            OrthogonalModel(tape, wide, large, fingerprint: false);
        }

        AssertNoAllocation(() =>
        {
            for (int i = 0; i < 5; i++)
            {
                OrthogonalModel(tape, tall, small, fingerprint: false);
                OrthogonalModel(tape, wide, large, fingerprint: false);
            }
        });
    }

    [Fact]
    public void ValuesAndGradientsDoNotDependOnReleasedOrthogonalDecompositions()
    {
        var general = RandomRectangle(30, 20);
        var symmetric = RandomMatrix(20);
        string expected = OrthogonalModel(new Tape(), general, symmetric);
        var tape = new Tape();
        for (int n = 1; n <= 40; n++)
        {
            OrthogonalModel(tape, RandomRectangle(30, n), RandomMatrix(n));
        }

        Assert.Equal(expected, OrthogonalModel(tape, general, symmetric));
    }

    // 一般の行列の QR 分解と特異値分解、正方行列（下三角部分を対称行列として読む）の固有値分解を記録して逆伝播し、値と葉の勾配のビット列を返す。
    private static string OrthogonalModel(Tape tape, Matrix<double> general, Matrix<double> symmetric, bool fingerprint = true)
    {
        tape.Reset();
        var a = tape.Leaf(general);
        var s = tape.Leaf(symmetric);
        var qr = QrDecompositionNode.Compute(a);
        var svd = SingularValueDecompositionNode.Compute(a);
        var eigen = SymmetricEigenDecompositionNode.Compute(s);
        // 値と勾配のビット列を比べるだけなので、結果は符号の選び方によるまま使う。
        ScalarNode loss = XMath.Sum(XMath.ColumnSums(qr.Q)) + XMath.Sum(XMath.ColumnSums(qr.R)) + XMath.Sum(svd.Values)
            + XMath.Sum(XMath.ColumnSums(svd.U)) + XMath.Sum(XMath.ColumnSums(svd.V)) + XMath.Sum(eigen.Values) + XMath.Sum(XMath.ColumnSums(eigen.Vectors));
        tape.Backward(loss);
        return fingerprint
            ? Fingerprint(loss.Value, tape.GetGradient(a).ToMatrix().AsSpan()) + Fingerprint(0, tape.GetGradient(s).ToMatrix().AsSpan())
            : "";
    }

    // 同じ行列の Cholesky 分解と LU 分解を2つずつ記録して逆伝播し、値と葉の勾配のビット列を返す。
    private static string DecompositionModel(Tape tape, Matrix<double> values, bool fingerprint = true)
    {
        tape.Reset();
        var a = tape.Leaf(values);
        var first = CholeskyDecompositionNode.Compute(a);
        var second = CholeskyDecompositionNode.Compute(a.T);
        var lu = LuDecompositionNode.Compute(a);
        var b = first.L * second.Solve(XMath.ColumnSums(a));
        ScalarNode loss = first.LogDeterminant + lu.LogAbsDeterminant + first.InverseQuadraticForm(b) + XMath.Sum(XMath.Solve(a, b));
        tape.Backward(loss);
        return fingerprint ? Fingerprint(loss.Value, tape.GetGradient(a).ToMatrix().AsSpan()) : "";
    }

    // 行列積・定数倍・Exp・列和・総和を記録して逆伝播し、値と葉の勾配のビット列を返す。
    private static string SquareModel(Tape tape, Matrix<double> values, bool fingerprint = true)
    {
        tape.Reset();
        var x = tape.Leaf(values);
        ScalarNode loss = XMath.Sum(XMath.ColumnSums(XMath.Exp(x * x * 1e-3)));
        tape.Backward(loss);
        return fingerprint ? Fingerprint(loss.Value, tape.GetGradient(x).ToMatrix().AsSpan()) : "";
    }

    // 短いベクトルに Logistic を steps 回かけた後に、長いベクトルの要素演算を記録の最後に置く。長い配列のスロットは steps で決まる。
    // fingerprint が false なら、確保を数えられるよう値と勾配を読まない。
    private static string ChainModel(Tape tape, LinearAlgebra.Vector<double> start, LinearAlgebra.Vector<double> data, int steps, bool fingerprint = true)
    {
        tape.Reset();
        var h = tape.Leaf(start);
        for (int i = 0; i < steps; i++)
        {
            h = XMath.Logistic(h);
        }

        var x = tape.Leaf(data);
        ScalarNode loss = XMath.Sum(h) + XMath.Sum(XMath.Exp(x * 1e-3));
        tape.Backward(loss);
        return fingerprint ? Fingerprint(loss.Value, tape.GetGradient(x).ToArray()) : "";
    }

    private static string Fingerprint(double value, ReadOnlySpan<double> gradient)
        => BitConverter.DoubleToInt64Bits(value).ToString("X16") + ";" + Convert.ToHexString(MemoryMarshal.AsBytes(gradient));

    private static long RetainedBytes(Action<Tape> action)
    {
        // 計測の間に共有プールが配列を解放すると差分が大きくなるため、3回の最小値を使う。
        long minimum = long.MaxValue;
        for (int sample = 0; sample < 3; sample++)
        {
            long withTape = MeasureWithLiveTape(action);
            long withoutTape = SettledHeapBytes();
            minimum = Math.Min(minimum, withTape - withoutTape);
        }

        return minimum;
    }

    private readonly struct AddOneWithSavedWorkspace(int length) : IDifferentiableOperation<double>
    {
        public void Forward(DifferentiableOperationContext<double> context, MatrixSpan<double> destination)
        {
            var workspace = context.AllocateSaved(length);
            workspace[^1] = context[0][0, 0];
            destination[0, 0] = workspace[^1] + 1;
        }

        public void PullBack(DifferentiableOperationContext<double> context, ReadOnlyMatrixSpan<double> output, ReadOnlyMatrixSpan<double> outputGradient)
            => throw new InvalidOperationException("Constant inputs do not require a pullback.");
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static long MeasureWithLiveTape(Action<Tape> action)
    {
        var tape = new Tape();
        action(tape);
        long bytes = SettledHeapBytes();
        GC.KeepAlive(tape);
        return bytes;
    }

    private static long SettledHeapBytes()
    {
        // プールがファイナライザーで解放した配列は、その次の GC で回収される。
        for (int cycle = 0; cycle < 3; cycle++)
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
        }

        GC.Collect();
        return GC.GetTotalMemory(forceFullCollection: false);
    }

    // 計画 3.7 の測り方：数える直前に gen0 の GC を入れ、窓を3回まで測って、どれかが 0 なら確保しないとみなす。
    private static void AssertNoAllocation(Action action)
    {
        long allocated = 0;
        for (int window = 0; window < 3; window++)
        {
            GC.Collect(0, GCCollectionMode.Forced, blocking: true);
            long before = GC.GetAllocatedBytesForCurrentThread();
            action();
            allocated = GC.GetAllocatedBytesForCurrentThread() - before;
            if (allocated == 0)
            {
                return;
            }
        }

        Assert.Equal(0, allocated);
    }

    private static LinearAlgebra.Vector<double> RandomVector(int length)
    {
        var random = new Random(length);
        var vector = new LinearAlgebra.Vector<double>(length);
        for (int i = 0; i < length; i++)
        {
            vector[i] = random.NextDouble() - 0.5;
        }

        return vector;
    }

    // A = MMᵀ/n + I（M は RandomMatrix(n)）。
    private static Matrix<double> SymmetricPositiveDefinite(int order)
    {
        var m = RandomMatrix(order);
        var a = XMath.Multiply(m, m.T);
        for (int j = 0; j < order; j++)
        {
            for (int i = 0; i < order; i++)
            {
                a[i, j] = a[i, j] / order + (i == j ? 1 : 0);
            }
        }

        return a;
    }

    private static Matrix<double> RandomRectangle(int rows, int columns)
    {
        var random = new Random(rows * 1000 + columns);
        var matrix = new Matrix<double>(rows, columns);
        Span<double> values = matrix.AsSpan();
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = random.NextDouble() - 0.5;
        }

        return matrix;
    }

    private static Matrix<double> RandomMatrix(int order)
    {
        var random = new Random(order);
        var matrix = new Matrix<double>(order, order);
        Span<double> values = matrix.AsSpan();
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = random.NextDouble() - 0.5;
        }

        return matrix;
    }
}
