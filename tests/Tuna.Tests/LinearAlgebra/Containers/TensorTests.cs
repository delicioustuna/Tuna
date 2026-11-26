using static Tuna.Tests.LinearAlgebra.LinearAlgebraTestUtilities;
// System.Numerics.Tensors にも同名の TensorSpan・Tensor があるので、名前空間の別名で区別する
using SystemTensors = System.Numerics.Tensors;

namespace Tuna.Tests.LinearAlgebra;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む
using Tuna.LinearAlgebra;

// Tensor（3階。先頭2軸が行列、第3軸が行列の番号）の格納・ビュー・演算・分解。
// 行列ごとの結果は、同じ行列を2階の API で計算した結果とビット単位で比べる（行列ごとに同じカーネルを呼ぶため）。
[Trait("LinearAlgebra", "Batch")]
public class TensorTests
{
    private readonly struct Square : IUnaryOperator<double>
    {
        public static double Invoke(double x) => x * x;
    }

    private readonly struct Shift(double offset) : IStatefulUnaryOperator<double>
    {
        public double Invoke(double x) => x + offset;
    }

    private readonly struct Hypot : IBinaryOperator<double>
    {
        public static double Invoke(double x, double y) => Math.Sqrt(x * x + y * y);
    }

    /// <summary>要素 (i, j, k) が i + 10j + 100k のテンソル</summary>
    private static Tensor<double> Indexed(int rows, int columns, int count)
    {
        var b = new Tensor<double>(rows, columns, count);
        for (int k = 0; k < count; k++)
        {
            for (int j = 0; j < columns; j++)
            {
                for (int i = 0; i < rows; i++)
                {
                    b[i, j, k] = i + 10 * j + 100 * k;
                }
            }
        }

        return b;
    }

    private static Tensor<double> RandomBatch(int rows, int columns, int count, int seed) =>
        Tensor<double>.FromColumnMajor([rows, columns, count], Random(rows, columns * count, seed).AsSpan());

    private static int[] Shape(ReadOnlySpan<nint> lengths) => lengths.ToArray().Select(length => (int)length).ToArray();

    private static Tensor<double> PositiveDefiniteBatch(int n, int count, int seed)
    {
        var matrices = new Matrix<double>[count];
        for (int k = 0; k < count; k++)
        {
            matrices[k] = RandomPositiveDefinite(n, seed + k);
        }

        return Tensor<double>.FromMatrices(matrices);
    }

    [Fact]
    public void StoresMatricesConsecutivelyInColumnMajorOrder()
    {
        var b = Tensor<double>.FromColumnMajor([2, 3, 4], Enumerable.Range(0, 24).Select(i => (double)i).ToArray());
        Assert.Equal(new[] { 2, 3, 4 }, Shape(b.Lengths));
        Assert.Equal(new[] { 1, 2, 6 }, Shape(b.Strides));
        Assert.Equal((3, 4, (nint)24, false), (b.Rank, b.Matrices.Count, b.FlattenedLength, b.IsEmpty));
        Assert.Equal(1 + 2 * 2 + 6 * 3, b[1, 2, 3]);
        Assert.Equal(new double[,] { { 6, 8, 10 }, { 7, 9, 11 } }, b.GetMatrix(1).ToArray());
        Assert.Equal(new double[] { 8, 9 }, b.GetVector(1, 1).ToArray());
        Assert.Equal("Tensor<Double> [2, 3, 4]", b.ToString());

        var array = b.ToArray();
        Assert.Equal(new[] { 2, 3, 4 }, new[] { array.GetLength(0), array.GetLength(1), array.GetLength(2) });
        Assert.Equal(b[1, 0, 2], array[1, 0, 2]);
        Assert.Equal(b.AsSpan().ToArray(), new Tensor<double>(array).AsSpan().ToArray());

        // 各行列のビューは領域を共有する
        b.GetMatrix(2)[0, 1] = -1;
        Assert.Equal(-1.0, b[0, 1, 2]);
        Assert.Throws<ArgumentOutOfRangeException>(() => b[0, 0, 4]);
        Assert.Throws<ArgumentOutOfRangeException>(() => b.GetMatrix(4));
    }

    [Fact]
    public void BuildsFromMatricesAndIdentity()
    {
        var a = Random(3, 2, 1);
        var c = Random(3, 2, 2);
        var batch = Tensor<double>.FromMatrices(a, c);
        Assert.Equal(a.ToArray(), batch.GetMatrix(0).ToArray());
        Assert.Equal(c.ToArray(), batch.GetMatrix(1).ToArray());
        Assert.Throws<ArgumentException>(() => Tensor<double>.FromMatrices(a, Random(2, 3, 3)));
        Assert.Throws<ArgumentException>(() => Tensor<double>.FromMatrices());

        var identity = Tensor<double>.Identity(3, 2);
        Assert.Equal(Matrix<double>.Identity(3).ToArray(), identity.GetMatrix(1).ToArray());
    }

    [Fact]
    public void SlicesTransposesAndDiagonalsShareStorage()
    {
        var b = Indexed(3, 4, 5);
        var slice = b[.., .., 1..3];
        Assert.Equal(new[] { 3, 4, 2 }, Shape(slice.Lengths));
        Assert.Equal(2, slice.Matrices.Count);
        Assert.Equal(b[2, 3, 2], slice[2, 3, 1]);

        var block = b[1.., 2..4, 3..];
        Assert.Equal(new[] { 2, 2, 2 }, Shape(block.Lengths));
        Assert.Equal(b[2, 3, 4], block[1, 1, 1]);

        var t = b.T;
        Assert.Equal(new[] { 4, 3, 5 }, Shape(t.Lengths));
        Assert.Equal(b[2, 1, 3], t[1, 2, 3]);

        var square = Indexed(3, 3, 4);
        var diagonals = square.Diagonals();
        Assert.Equal((3, 4), (diagonals.Rows, diagonals.Columns));
        Assert.Equal(square[2, 2, 3], diagonals[2, 3]);
        diagonals[1, 2] = -5;
        Assert.Equal(-5.0, square[1, 1, 2]);

        // 長方形の行列は min(Lengths[0], Lengths[1]) 個の対角成分
        Assert.Equal(3, b.Diagonals().Rows);
    }

    [Fact]
    public void AsMatrixPlacesMatricesSideBySide()
    {
        var b = Indexed(2, 3, 4);
        var wide = b.AsMatrix();
        Assert.Equal((2, 12), (wide.Rows, wide.Columns));
        Assert.Equal(b[1, 2, 3], wide[1, 3 * 3 + 2]);

        // 行の範囲を切り出しても横に並べられる。転置したテンソルは列の間隔が合わないので例外
        Assert.Equal(b[1, 0, 1], b[1..2, .., ..].AsMatrix()[0, 3]);
        Assert.Throws<InvalidOperationException>(() => b.T.AsMatrix());
    }

    [Fact]
    public void AsMatrixRejectsColumnCountsBeyondInt32()
    {
        // 横に並べた列数 65536·65536 は int に収まらない。要素のない形状でも、1要素のブロードキャストでも、形状を黙って変えない
        var empty = new Tensor<double>(0, 65536, 65536);
        Assert.Throws<InvalidOperationException>(() => empty.AsMatrix());
        Assert.Throws<InvalidOperationException>(() => empty.AsTensorSpan().AsReadOnly().AsMatrix());
        var one = new Vector<double>([42]);
        Assert.Equal(4294967296, (long)one.Broadcast(65536).Broadcast(65536).FlattenedLength);
        Assert.Throws<InvalidOperationException>(() => one.Broadcast(65536).Broadcast(65536).AsMatrix());

        // 行列にできない形状でも、要素ごとの演算は行列ごとに計算する
        var sum = XMath.Add(empty, empty);
        Assert.Equal(new[] { 0, 65536, 65536 }, Shape(sum.Lengths));
    }

    [Fact]
    public void MatricesAndVectorsBecomeBroadcastAndUnsqueezedViews()
    {
        var m = Random(3, 3, 4);
        var broadcast = m.Broadcast(5);
        Assert.Equal(new[] { 3, 3, 5 }, Shape(broadcast.Lengths));
        Assert.Equal(0, broadcast.Strides[2]);
        Assert.Equal(m[2, 1], broadcast[2, 1, 4]);

        // 同じ行列と各行列の積（ガウス過程で共通の K⁻¹ を候補ごとの行列に掛けるなど）
        var others = RandomBatch(3, 2, 5, 5);
        var products = XMath.Multiply(broadcast, others);
        for (int k = 0; k < 5; k++)
        {
            Assert.Equal(XMath.Multiply(m, others.GetMatrix(k)).ToArray(), products.GetMatrix(k).ToArray());
        }

        // n × m の行列を n × 1 の行列 m 個として扱う（書き込みは元の行列に届く）
        var columns = Random(4, 3, 6);
        var perColumn = columns.Unsqueeze(1);
        Assert.Equal(new[] { 4, 1, 3 }, Shape(perColumn.Lengths));
        Assert.Equal(columns.Columns, perColumn.Matrices.Count);
        Assert.Equal(columns.Column(1).ToArray(), perColumn.GetVector(0, 1).ToArray());
        perColumn[3, 0, 2] = 42;
        Assert.Equal(42.0, columns[3, 2]);

        // 他の位置にも長さ 1 の軸を差し込める（先頭なら各列を横にした 1 × n の行列、末尾なら行列1個のテンソル）
        var rows = columns.Unsqueeze(0);
        Assert.Equal(new[] { 1, 4, 3 }, Shape(rows.Lengths));
        Assert.Equal(columns.Column(2).ToArray(), rows.GetMatrix(2).Row(0).ToArray());
        var single = columns.AsMatrixSpan().AsReadOnly().Unsqueeze(2);
        Assert.Equal(new[] { 4, 3, 1 }, Shape(single.Lengths));
        Assert.Equal(columns.ToArray(), single.GetMatrix(0).ToArray());
        Assert.Throws<ArgumentOutOfRangeException>(() => columns.Unsqueeze(3));
        Assert.Throws<ArgumentOutOfRangeException>(() => columns.Unsqueeze(-1));

        // ベクトルのブロードキャストは列として並べた行列（ベクトルは列ベクトル）。行として並べる場合は BroadcastRows
        var v = new Vector<double>([1, 2, 3]);
        var repeated = v.Broadcast(5);
        Assert.Equal((3, 5), (repeated.Rows, repeated.Columns));
        Assert.Equal(v.ToArray(), repeated.Column(4).ToArray());
        Assert.Equal(v.ToArray(), v.BroadcastRows(4).Row(3).ToArray());
    }

    [Fact]
    public void WritableViewsMustBeInjective()
    {
        var data = new double[12];
        Assert.Throws<ArgumentException>(() => new TensorSpan<double>(data, [2, 2, 3], [1, 2, 0]));
        Assert.Throws<ArgumentException>(() => new TensorSpan<double>(data, [2, 2, 2], [1, 1, 4]));
        var readOnly = new ReadOnlyTensorSpan<double>(data, [2, 2, 3], [1, 2, 0]);
        Assert.Equal(3, readOnly.Matrices.Count);
        // 形状とストライドの要素数の違いは引数の誤り
        Assert.Throws<ArgumentException>(() => new TensorSpan<double>(data, [2, 2, 3], [1, 2]));
    }

    [Fact]
    public void MatricesAndTensorsAcceptTheSameWritableLayouts()
    {
        // テンソルの各行列の配置（第0列が {0, 2, 4}、第1列が {5, 7, 9}）は、行列のビューとしても作り直せる
        double[] values = Enumerable.Range(0, 10).Select(i => (double)i).ToArray();
        var tensor = new TensorSpan<double>(values, [3, 2, 1], [2, 5, 0]);
        var matrix = tensor.GetMatrix(0);
        var again = new MatrixSpan<double>(matrix.Data, matrix.Lengths, matrix.Strides);
        Assert.Equal(9.0, again[2, 1]);
        again[1, 1] = -7;
        Assert.Equal(-7.0, values[7]);

        // 行列の書き込み可能なビューは、行列 1 個のテンソルと同じ配置だけを受け付ける
        var data = new double[64];
        for (int rows = 1; rows <= 4; rows++)
        {
            for (int columns = 1; columns <= 4; columns++)
            {
                for (int rowStride = 0; rowStride <= 6; rowStride++)
                {
                    for (int columnStride = 0; columnStride <= 6; columnStride++)
                    {
                        bool asMatrix = Accepts(() => new MatrixSpan<double>(data, [rows, columns], [rowStride, columnStride]));
                        bool asTensor = Accepts(() => new TensorSpan<double>(data, [rows, columns, 1], [rowStride, columnStride, 0]));
                        Assert.True(asMatrix == asTensor, $"[{rows}, {columns}] strides [{rowStride}, {columnStride}]");
                    }
                }
            }
        }

        static bool Accepts(Action create)
        {
            try
            {
                create();
                return true;
            }
            catch (ArgumentException)
            {
                return false;
            }
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(4)]
    [InlineData(5)]
    public void UnsupportedRanksAreRejectedOnEveryCreationPath(int rank)
    {
        nint[] lengths = Enumerable.Repeat((nint)2, rank).ToArray();
        nint[] strides = Enumerable.Range(0, rank).Select(i => (nint)1 << i).ToArray();
        var data = new double[64];
        void AssertRank(Action create)
        {
            var ex = Assert.Throws<NotSupportedException>(create);
            Assert.Contains($"rank {rank} was requested", ex.Message);
        }

        AssertRank(() => _ = new Tensor<double>(lengths));
        AssertRank(() => Tensor<double>.FromColumnMajor(lengths, data));
        AssertRank(() => _ = new ReadOnlyTensorSpan<double>(data, lengths));
        AssertRank(() => _ = new TensorSpan<double>(data, lengths));
        AssertRank(() => _ = new ReadOnlyTensorSpan<double>(data, lengths, strides));
        AssertRank(() => _ = new TensorSpan<double>(data, lengths, strides));
        AssertRank(() => _ = ReadOnlyTensorSpan<double>.Create(in data[0], lengths, strides));
        AssertRank(() => _ = TensorSpan<double>.Create(ref data[0], lengths, strides));

        // 形状とストライドの要素数の違いや負の長さは、階数の非対応ではなく引数の誤り
        Assert.Throws<ArgumentException>(() => _ = new TensorSpan<double>(data, lengths, [1, 2, 4]));
        Assert.Throws<ArgumentOutOfRangeException>(() => _ = new Tensor<double>(2, -1, 3));
    }

    [Fact]
    public void GetMatrixFixesTheThirdAxisOfEveryView()
    {
        var b = Indexed(2, 3, 4);
        ReadOnlyTensorSpan<double> readOnly = b;
        TensorSpan<double> writable = b;
        Assert.Equal(b[0, 2, 3], readOnly.GetMatrix(3)[0, 2]);
        writable.GetMatrix(2)[1, 1] = -3;
        Assert.Equal(-3.0, b[1, 1, 2]);

        // 転置・部分ビューでも第3軸（行列の番号）を固定する
        Assert.Equal(b[1, 2, 3], b.T.GetMatrix(3)[2, 1]);
        Assert.Equal(b[0, 1, 2], b[.., .., 1..3].GetMatrix(1)[0, 1]);
        Assert.Throws<ArgumentOutOfRangeException>(() => ((ReadOnlyTensorSpan<double>)b).GetMatrix(4));
        Assert.Throws<ArgumentOutOfRangeException>(() => b.AsTensorSpan().GetMatrix(-1));
    }

    /// <summary>GetVector(column, index) は第 index 行列の第 column 列（列 = ベクトル）で、GetMatrix(index).Column(column) と同じ</summary>
    [Fact]
    public void GetVectorIsAColumnOfAMatrix()
    {
        var b = Indexed(2, 3, 4);
        ReadOnlyTensorSpan<double> readOnly = b;
        TensorSpan<double> writable = b;
        for (int k = 0; k < 4; k++)
        {
            for (int j = 0; j < 3; j++)
            {
                Assert.Equal(b.GetMatrix(k).Column(j).ToArray(), readOnly.GetVector(j, k).ToArray());
                Assert.Equal(b.GetMatrix(k).Column(j).ToArray(), b.GetVector(j, k).ToArray());
            }
        }

        // 書き込みは元の領域に届く
        writable.GetVector(2, 1)[0] = -4;
        Assert.Equal(-4.0, b[0, 2, 1]);
        b.GetVector(0, 3)[1] = -6;
        Assert.Equal(-6.0, b[1, 0, 3]);

        // 転置ビューの列は元の行
        Assert.Equal(b.GetMatrix(2).Row(1).ToArray(), b.T.GetVector(1, 2).ToArray());

        Assert.Throws<ArgumentOutOfRangeException>(() => b.GetVector(3, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => b.GetVector(0, 4));
        Assert.Throws<ArgumentOutOfRangeException>(() => ((ReadOnlyTensorSpan<double>)b).GetVector(-1, 0));
    }

    /// <summary>Matrices は行列の数（Count）と、通し番号での取り出し・列挙</summary>
    [Fact]
    public void MatricesCountsIndexesAndEnumeratesMatrices()
    {
        var b = Indexed(2, 3, 4);
        ReadOnlyTensorSpan<double> readOnly = b;
        Assert.Equal(4, b.Matrices.Count);
        Assert.Equal(4, readOnly.Matrices.Count);
        Assert.Equal(b.GetMatrix(3).ToArray(), readOnly.Matrices[3].ToArray());
        Assert.Throws<ArgumentOutOfRangeException>(() => { _ = b.Matrices[4]; });
        Assert.Throws<ArgumentOutOfRangeException>(() => { _ = ((ReadOnlyTensorSpan<double>)b).Matrices[-1]; });

        int count = 0;
        foreach (var matrix in readOnly.Matrices)
        {
            Assert.Equal(b.GetMatrix(count).ToArray(), matrix.ToArray());
            count++;
        }

        Assert.Equal(4, count);

        // 書き込み可能なビューの列挙は元の領域に書く
        foreach (var matrix in b.Matrices)
        {
            matrix[0, 0] = -1;
        }

        Assert.All(Enumerable.Range(0, 4), k => Assert.Equal(-1.0, b[0, 0, k]));

        // 部分・転置・ブロードキャストのビューでも同じ
        Assert.Equal(2, b[.., .., 1..3].Matrices.Count);
        Assert.Equal(b.GetMatrix(1).ToArray(), b.T.Matrices[1].T.ToArray());
        Assert.Equal(5, Random(2, 2, 1).Broadcast(5).Matrices.Count);
        Assert.Equal(0, new Tensor<double>(2, 2, 0).Matrices.Count);
        foreach (var matrix in new Tensor<double>(2, 2, 0).Matrices)
        {
            Assert.Fail($"An empty tensor enumerated a {matrix.Rows} × {matrix.Columns} matrix.");
        }
    }

    [Theory]
    [InlineData(0, 0, 3)]
    [InlineData(0, 2, 3)]
    [InlineData(2, 0, 3)]
    [InlineData(0, 2, 0)]
    [InlineData(0, 2, 1)]
    [InlineData(2, 3, 0)]
    public void EmptyBatchesHaveDiagonalsAndRoundTripThroughTheirLayout(int rows, int columns, int count)
    {
        // 要素のない配置では異なる添字が同じ要素を指すことはないので、書き込み可能なビューとして扱える
        var b = new Tensor<double>(rows, columns, count);
        var diagonals = b.Diagonals();
        Assert.Equal((Math.Min(rows, columns), count), (diagonals.Rows, diagonals.Columns));

        TensorSpan<double> view = b;
        var again = new TensorSpan<double>(view.Data, view.Lengths, view.Strides);
        Assert.Equal(new[] { rows, columns, count }, Shape(again.Lengths));
        Assert.Equal(rows == 0 || columns == 0 || count == 0, again.IsEmpty);

        var identity = Tensor<double>.Identity(0, count);
        Assert.Equal(new[] { 0, 0, count }, Shape(identity.Lengths));
    }

    [Fact]
    public void ViewsConvertToTensorSpansWithoutCopying()
    {
        var b = Indexed(3, 4, 2);
        TensorSpan<double> view = b;
        var tensor = new SystemTensors.TensorSpan<double>(view.Data, view.Lengths, view.Strides);
        Assert.Equal(b[2, 1, 1], tensor[[2, 1, 1]]);
        tensor[[0, 3, 1]] = -7;
        Assert.Equal(-7.0, b[0, 3, 1]);

        var t = b.T;
        var transposed = new SystemTensors.ReadOnlyTensorSpan<double>(t.Data, t.Lengths, t.Strides);
        Assert.Equal(b[1, 2, 0], transposed[[2, 1, 0]]);

        // 行優先の TensorSpan（最後の添字が最も速く変わる）もストライド付きのビューになる
        double[] values = Enumerable.Range(0, 24).Select(i => (double)i).ToArray();
        var rowMajor = new SystemTensors.TensorSpan<double>(values, [2, 3, 4], []);
        var fromTensor = TensorSpan<double>.Create(ref rowMajor.GetPinnableReference(), rowMajor.Lengths, rowMajor.Strides);
        Assert.Equal(1 * 12 + 2 * 4 + 3, fromTensor[1, 2, 3]);
    }

    [Fact]
    public void ElementwiseOperationsMatchMatrixOperations()
    {
        var x = RandomBatch(4, 3, 5, 10);
        var y = RandomBatch(3, 4, 5, 11);
        var yt = y.T;  // 横に並べられない配置（行列ごとの経路）
        var sum = x + yt;
        var difference = XMath.Subtract(x, yt);
        var hadamard = XMath.Hadamard(x, yt);
        var negated = -x;
        var scaled = 2.5 * x;
        var squared = XMath.Map<double, Square>(x);
        var shifted = XMath.Map(x, new Shift(3));
        var hypot = XMath.Zip<double, Hypot>(x, yt);
        for (int k = 0; k < 5; k++)
        {
            Assert.Equal(XMath.Add(x.GetMatrix(k), yt.GetMatrix(k)).ToArray(), sum.GetMatrix(k).ToArray());
            Assert.Equal(XMath.Subtract(x.GetMatrix(k), yt.GetMatrix(k)).ToArray(), difference.GetMatrix(k).ToArray());
            Assert.Equal(XMath.Hadamard(x.GetMatrix(k), yt.GetMatrix(k)).ToArray(), hadamard.GetMatrix(k).ToArray());
            Assert.Equal(XMath.Negate(x.GetMatrix(k)).ToArray(), negated.GetMatrix(k).ToArray());
            Assert.Equal(XMath.Multiply(x.GetMatrix(k), 2.5).ToArray(), scaled.GetMatrix(k).ToArray());
            Assert.Equal(XMath.Map<double, Square>(x.GetMatrix(k)).ToArray(), squared.GetMatrix(k).ToArray());
            Assert.Equal(XMath.Map(x.GetMatrix(k), new Shift(3)).ToArray(), shifted.GetMatrix(k).ToArray());
            Assert.Equal(XMath.Zip<double, Hypot>(x.GetMatrix(k), yt.GetMatrix(k)).ToArray(), hypot.GetMatrix(k).ToArray());
        }

        // 複合代入とインプレース演算
        var z = x.Clone();
        z += yt;
        Assert.Equal(sum.ToArray(), z.ToArray());
        z *= 2;
        XMath.InPlace.Map<double, Square>(z);
        Assert.Equal(4 * sum.GetMatrix(0)[1, 1] * sum.GetMatrix(0)[1, 1], z[1, 1, 0], 1e-15);
    }

    [Fact]
    public void ElementwiseOperationsRejectPartialOverlap()
    {
        var b = RandomBatch(2, 2, 4, 12);
        var other = RandomBatch(2, 2, 3, 13);

        // 行列の添字をずらした出力先は部分的な重なり（同じ添字の要素だけを見ても検出できない）
        Assert.Throws<ArgumentException>(() => XMath.Add(b[.., .., 0..3], other, b[.., .., 1..4]));
        Assert.Throws<ArgumentException>(() => XMath.Add(b[.., .., 0..3], other, b.T[.., .., 0..3]));
        Assert.Throws<ArgumentException>(() => XMath.Add(b, RandomBatch(2, 2, 3, 14)));

        // 完全に同じ領域なら許す
        var expected = XMath.Add(b[.., .., 0..3], other);
        XMath.Add(b[.., .., 0..3], other, b[.., .., 0..3]);
        Assert.Equal(expected.ToArray(), b[.., .., 0..3].ToArray());
    }

    [Fact]
    public void MultiplyAndTransposeWorkPerMatrix()
    {
        var a = RandomBatch(4, 6, 7, 20);
        var b = RandomBatch(6, 3, 7, 21);
        var c = a * b;
        var at = XMath.Transpose(a);
        for (int k = 0; k < 7; k++)
        {
            Assert.Equal(XMath.Multiply(a.GetMatrix(k), b.GetMatrix(k)).ToArray(), c.GetMatrix(k).ToArray());
            Assert.Equal(XMath.Transpose(a.GetMatrix(k)).ToArray(), at.GetMatrix(k).ToArray());
        }

        // 転置ビューとの積（転置した配置の行列積は積和の順序が変わるので、2階の同じ呼び出しと比べる）
        var ct = XMath.Multiply(at.T, b);
        for (int k = 0; k < 7; k++)
        {
            Assert.Equal(XMath.Multiply(at.T.GetMatrix(k), b.GetMatrix(k)).ToArray(), ct.GetMatrix(k).ToArray());
        }

        Assert.Throws<ArgumentException>(() => XMath.Multiply(a, a));
        var square = RandomBatch(3, 3, 4, 22);
        Assert.Throws<ArgumentException>(() => XMath.Multiply(square[.., .., 0..3], square[.., .., 0..3], square[.., .., 1..4]));
        Assert.Throws<ArgumentException>(() => XMath.Transpose(square, square.T));
    }

    [Theory]
    [InlineData(5, 3, 9)]
    [InlineData(40, 10, 6)]
    [InlineData(150, 10, 3)]
    public void TriangularAndGeneralSolvesMatchSingleMatrixResults(int n, int nrhs, int count)
    {
        var a = RandomBatch(n, n, count, 30 + n);
        var b = RandomBatch(n, nrhs, count, 40 + n);
        var lower = XMath.SolveLowerTriangular(a, b);
        var upper = XMath.SolveUpperTriangular(a.T, b);
        var general = XMath.Solve(a, b);
        for (int k = 0; k < count; k++)
        {
            Assert.Equal(XMath.SolveLowerTriangular(a.GetMatrix(k), b.GetMatrix(k)).ToArray(), lower.GetMatrix(k).ToArray());
            Assert.Equal(XMath.SolveUpperTriangular(a.T.GetMatrix(k), b.GetMatrix(k)).ToArray(), upper.GetMatrix(k).ToArray());
            Assert.Equal(XMath.Solve(a.GetMatrix(k), b.GetMatrix(k)).ToArray(), general.GetMatrix(k).ToArray());
        }

        // 右辺が1列ならベクトルの右辺の経路（2階のベクトル版と同じ計算）
        var column = b[.., 0..1, ..];
        var lowerVector = XMath.SolveLowerTriangular<double>(a, column);
        var generalVector = XMath.Solve(a, column);
        for (int k = 0; k < count; k++)
        {
            Assert.Equal(XMath.SolveLowerTriangular(a.GetMatrix(k), b.GetMatrix(k).Column(0)).ToArray(), lowerVector.GetMatrix(k).Column(0).ToArray());
            Assert.Equal(XMath.Solve(a.GetMatrix(k), b.GetMatrix(k).Column(0)).ToArray(), generalVector.GetMatrix(k).Column(0).ToArray());
        }

        // インプレースで解く
        var inPlace = b.Clone();
        XMath.Solve(a, inPlace, inPlace);
        Assert.Equal(general.ToArray(), inPlace.ToArray());
    }

    [Theory]
    [InlineData(5, 1, 9)]
    [InlineData(5, 3, 9)]
    [InlineData(150, 1, 3)]
    [InlineData(150, 10, 3)]
    public void BroadcastCoefficientGivesSameResultsAsSeparateMatrices(int n, int nrhs, int count)
    {
        // 共通の係数（ブロードキャストビュー）は1回だけ分解するが、各行列の結果は行列ごとに分解した場合と同じ
        var m = Random(n, n, 60 + n);
        var b = RandomBatch(n, nrhs, count, 70 + n);
        var broadcast = XMath.Solve(m.Broadcast(count), b);
        var separate = XMath.Solve(m.Broadcast(count).ToTensor(), b);
        Assert.Equal(separate.ToArray(), broadcast.ToArray());
        for (int k = 0; k < count; k++)
        {
            if (nrhs == 1)
            {
                Assert.Equal(XMath.Solve(m, b.GetMatrix(k).Column(0)).ToArray(), broadcast.GetMatrix(k).Column(0).ToArray());
            }
            else
            {
                Assert.Equal(XMath.Solve(m, b.GetMatrix(k)).ToArray(), broadcast.GetMatrix(k).ToArray());
            }
        }

        var singular = new Matrix<double>(n, n);
        var ex = Assert.Throws<ArgumentException>(() => XMath.Solve(singular.Broadcast(count), b));
        Assert.Contains("index 0", ex.Message);
    }

    [Fact]
    public void BatchSolveReportsFirstSingularMatrix()
    {
        var a = RandomBatch(3, 3, 5, 50);
        a.GetMatrix(3).Fill(0);
        a.GetMatrix(4).Fill(0);
        var ex = Assert.Throws<ArgumentException>(() => XMath.Solve(a, RandomBatch(3, 1, 5, 51)));
        Assert.Contains("index 3", ex.Message);
    }

    [Theory]
    [InlineData(1, 4)]
    [InlineData(5, 50)]
    [InlineData(40, 9)]
    [InlineData(130, 3)]
    public void BatchCholeskyMatchesSingleMatrixDecompositions(int n, int count)
    {
        var a = PositiveDefiniteBatch(n, count, 60 + n);
        var chol = BatchCholeskyDecomposition.Compute(a);
        Assert.Equal((n, count, 0), (chol.Order, chol.BatchCount, chol.FailureCount));
        var b = Random(n, count, 70 + n);
        var rhs = RandomBatch(n, 3, count, 80 + n);
        var x = chol.Solve(b);
        var xs = chol.Solve(rhs);
        var inverses = chol.Inverse();
        var logDets = chol.LogDeterminants();
        for (int k = 0; k < count; k++)
        {
            var single = a.GetMatrix(k).ToMatrix().Cholesky();
            Assert.Equal(single.L.ToArray(), chol.L.GetMatrix(k).ToArray());
            Assert.Equal(single.LogDeterminant, logDets[k]);
            Assert.Equal(single.Solve(b.Column(k)).ToArray(), x.Column(k).ToArray());
            Assert.Equal(single.Solve(rhs.GetMatrix(k)).ToArray(), xs.GetMatrix(k).ToArray());
            Assert.Equal(single.Inverse().ToArray(), inverses.GetMatrix(k).ToArray());
            Assert.True(chol.IsPositiveDefinite(k));
        }
    }

    [Fact]
    public void BatchCholeskyMarksMatricesThatAreNotPositiveDefinite()
    {
        var a = PositiveDefiniteBatch(4, 5, 90);
        a[1, 1, 3] = -100;
        a[2, 2, 4] = double.NaN;
        Assert.False(BatchCholeskyDecomposition.TryCompute(a, out var chol));
        Assert.Equal(2, chol.FailureCount);
        Assert.Equal([true, true, true, false, false], Enumerable.Range(0, 5).Select(chol.IsPositiveDefinite));
        Assert.All(chol.L.GetMatrix(3).ToMatrix().AsSpan().ToArray(), v => Assert.True(double.IsNaN(v)));
        var logDets = chol.LogDeterminants();
        Assert.True(double.IsNaN(logDets[3]));
        Assert.Equal(a.GetMatrix(0).ToMatrix().Cholesky().LogDeterminant, logDets[0]);
        var x = chol.Solve(Random(4, 5, 91));
        Assert.True(double.IsNaN(x[0, 4]));
        Assert.False(double.IsNaN(x[0, 2]));

        var ex = Assert.Throws<ArgumentException>(() => BatchCholeskyDecomposition.Compute(a));
        Assert.Contains("index 3", ex.Message);
        Assert.Throws<ArgumentOutOfRangeException>(() => chol.IsPositiveDefinite(5));
    }

    [Fact]
    public void BatchCholeskyCanOverwriteInput()
    {
        var a = PositiveDefiniteBatch(6, 4, 100);
        var expected = BatchCholeskyDecomposition.Compute(a).L;
        var chol = BatchCholeskyDecomposition.Compute(a, a);
        Assert.Same(a, chol.L);
        Assert.Equal(expected.ToArray(), a.ToArray());

        var b = PositiveDefiniteBatch(6, 4, 110);
        var shifted = new Tensor<double>(6, 6, 5);
        b.AsTensorSpan().CopyTo(shifted[.., .., 1..5]);
        Assert.Throws<ArgumentException>(() => BatchCholeskyDecomposition.Compute(shifted[.., .., 1..5], Tensor<double>.FromMatrices(
            shifted.GetMatrix(0).ToMatrix(), shifted.GetMatrix(1).ToMatrix(), shifted.GetMatrix(2).ToMatrix())));

        // 出力先が L と重なる求解は例外
        Assert.Throws<ArgumentException>(() => chol.Solve(Random(6, 4, 111).Unsqueeze(1), chol.L[.., 0..1, ..]));
    }

    /// <summary>ガウス過程の典型的な使い方: 候補ごとの K + σ²I を一度に分解し、対数周辺尤度を求める</summary>
    [Fact]
    public void ComputesLogMarginalLikelihoodsForSeveralNoiseLevels()
    {
        int n = 30;
        double[] noise = [1e-2, 1e-1, 1];
        var points = Random(1, n, 120);
        var kernel = new Matrix<double>(n, n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double d = points[0, i] - points[0, j];
                kernel[i, j] = Math.Exp(-0.5 * d * d / 0.09);
            }
        }

        var y = Random(n, 1, 121);
        var k = kernel.Broadcast(noise.Length).ToTensor();
        var diagonals = k.Diagonals();
        for (int c = 0; c < noise.Length; c++)
        {
            for (int i = 0; i < n; i++)
            {
                diagonals[i, c] += noise[c];
            }
        }

        var chol = BatchCholeskyDecomposition.Compute(k);
        var ys = y.Broadcast(noise.Length);
        var alpha = chol.Solve(ys);
        var logDets = chol.LogDeterminants();
        for (int c = 0; c < noise.Length; c++)
        {
            double fit = 0;
            for (int i = 0; i < n; i++)
            {
                fit += y[i, 0] * alpha[i, 0, c];
            }

            double lml = -0.5 * fit - 0.5 * logDets[c] - 0.5 * n * Math.Log(2 * Math.PI);
            var single = k.GetMatrix(c).ToMatrix().Cholesky();
            var a = single.Solve(y.Column(0).ToArray());
            double expected = -0.5 * a.Zip(y.Column(0).ToArray(), (p, q) => p * q).Sum() - 0.5 * single.LogDeterminant - 0.5 * n * Math.Log(2 * Math.PI);
            Assert.Equal(expected, lml, Math.Abs(expected) * 1e-12);
        }
    }
}
