using CsCheck;
using Tuna.LinearAlgebra;

namespace Tuna.Tests.Properties;

[Trait("Category", "Property")]
public class LinearAlgebraViewPropertyTests
{
    private const double Sentinel = -999999.25;

    [Fact]
    public void MatrixSpan_SliceTransposeCopyToAndOutOfViewIntegrity()
    {
        var gen = Gen.Select(
            Gen.Int[0, 12],
            Gen.Int[0, 12],
            Gen.Int[1, 4],
            Gen.Int[0, 4],
            Gen.Int[1, 4],
            Gen.Int[0, 4],
            Gen.Int[0, 1000]);

        gen.Sample((rows, cols, rowStride, colGap, dstRowStride, dstColGap, sliceSeed) =>
        {
            var rng = new Random(sliceSeed);
            int r0 = rows == 0 ? 0 : rng.Next(0, rows);
            int r1 = rows == 0 ? 0 : rng.Next(r0, rows + 1);
            int c0 = cols == 0 ? 0 : rng.Next(0, cols);
            int c1 = cols == 0 ? 0 : rng.Next(c0, cols + 1);

            int colStride = Math.Max(1, rows * rowStride) + colGap;
            int srcBufLen = Math.Max(1, rows * rowStride + cols * colStride + 8);
            var srcBuf = new double[srcBufLen];
            Array.Fill(srcBuf, Sentinel);

            for (int j = 0; j < cols; j++)
            {
                for (int i = 0; i < rows; i++)
                {
                    srcBuf[i * rowStride + j * colStride] = (i + 1) * 100.0 + (j + 1);
                }
            }

            var view = new MatrixSpan<double>(srcBuf, [rows, cols], [rowStride, colStride]);
            var sub = view[r0..r1, c0..c1];
            var subT = sub.T;
            var subTT = subT.T;
            int sr = r1 - r0;
            int sc = c1 - c0;

            Assert.Equal(sr, sub.Rows);
            Assert.Equal(sc, sub.Columns);
            Assert.Equal(sc, subT.Rows);
            Assert.Equal(sr, subT.Columns);

            for (int i = 0; i < sr; i++)
            {
                for (int j = 0; j < sc; j++)
                {
                    double expected = (r0 + i + 1) * 100.0 + (c0 + j + 1);
                    Assert.Equal(expected, sub[i, j]);
                    Assert.Equal(expected, subT[j, i]);
                    Assert.Equal(expected, subTT[i, j]);
                }
            }

            // 転置ビューを別の非連続ストライド先へ CopyTo し、ビュー外領域が破壊されないことを確認する
            int dstColStride = Math.Max(1, sc * dstRowStride) + dstColGap;
            int dstBufLen = Math.Max(1, sc * dstRowStride + sr * dstColStride + 8);
            var dstBuf = new double[dstBufLen];
            Array.Fill(dstBuf, Sentinel);

            var isDstElement = new bool[dstBufLen];
            for (int j = 0; j < sr; j++)
            {
                for (int i = 0; i < sc; i++)
                {
                    isDstElement[i * dstRowStride + j * dstColStride] = true;
                }
            }

            var dstView = new MatrixSpan<double>(dstBuf, [sc, sr], [dstRowStride, dstColStride]);
            subT.CopyTo(dstView);

            for (int i = 0; i < sc; i++)
            {
                for (int j = 0; j < sr; j++)
                {
                    Assert.Equal(subT[i, j], dstView[i, j]);
                }
            }

            for (int idx = 0; idx < dstBufLen; idx++)
            {
                if (!isDstElement[idx])
                {
                    Assert.Equal(Sentinel, dstBuf[idx]);
                }
            }

            var materialized = subT.ToMatrix();
            for (int i = 0; i < sc; i++)
            {
                for (int j = 0; j < sr; j++)
                {
                    Assert.Equal(dstView[i, j], materialized[i, j]);
                }
            }
        }, iter: 250, seed: "0123456789ab");
    }

    [Fact]
    public void TensorSpan_SliceTransposeCopyToAndOutOfViewIntegrity()
    {
        var gen = Gen.Select(
            Gen.Int[0, 6],
            Gen.Int[0, 6],
            Gen.Int[0, 5],
            Gen.Int[1, 3],
            Gen.Int[0, 3],
            Gen.Int[0, 3],
            Gen.Int[0, 1000]);

        gen.Sample((rows, cols, count, s0, g1, g2, sliceSeed) =>
        {
            var rng = new Random(sliceSeed);
            int r0 = rows == 0 ? 0 : rng.Next(0, rows);
            int r1 = rows == 0 ? 0 : rng.Next(r0, rows + 1);
            int c0 = cols == 0 ? 0 : rng.Next(0, cols);
            int c1 = cols == 0 ? 0 : rng.Next(c0, cols + 1);
            int k0 = count == 0 ? 0 : rng.Next(0, count);
            int k1 = count == 0 ? 0 : rng.Next(k0, count + 1);

            int s1 = Math.Max(1, rows * s0) + g1;
            int s2 = Math.Max(1, cols * s1) + g2;
            int srcBufLen = Math.Max(1, rows * s0 + cols * s1 + count * s2 + 8);
            var srcBuf = new double[srcBufLen];
            Array.Fill(srcBuf, Sentinel);

            for (int k = 0; k < count; k++)
            {
                for (int j = 0; j < cols; j++)
                {
                    for (int i = 0; i < rows; i++)
                    {
                        srcBuf[i * s0 + j * s1 + k * s2] = (k + 1) * 10000.0 + (i + 1) * 100.0 + (j + 1);
                    }
                }
            }

            var view = new TensorSpan<double>(srcBuf, [rows, cols, count], [s0, s1, s2]);
            var sub = view[r0..r1, c0..c1, k0..k1];
            var subT = sub.T;
            int sr = r1 - r0;
            int sc = c1 - c0;
            int sk = k1 - k0;

            Assert.Equal(sr, (int)sub.Lengths[0]);
            Assert.Equal(sc, (int)sub.Lengths[1]);
            Assert.Equal(sk, (int)sub.Lengths[2]);

            for (int k = 0; k < sk; k++)
            {
                var sliceMat = sub.GetMatrix(k);
                for (int i = 0; i < sr; i++)
                {
                    for (int j = 0; j < sc; j++)
                    {
                        double expected = (k0 + k + 1) * 10000.0 + (r0 + i + 1) * 100.0 + (c0 + j + 1);
                        Assert.Equal(expected, sub[i, j, k]);
                        Assert.Equal(expected, sliceMat[i, j]);
                        Assert.Equal(expected, subT[j, i, k]);
                    }
                }
            }

            // 非連続な出力先 TensorSpan への CopyTo とビュー外センチネル非破壊を確認する
            int d0 = 2;
            int d1 = Math.Max(1, sc * d0) + 1;
            int d2 = Math.Max(1, sr * d1) + 2;
            int dstBufLen = Math.Max(1, sc * d0 + sr * d1 + sk * d2 + 8);
            var dstBuf = new double[dstBufLen];
            Array.Fill(dstBuf, Sentinel);

            var isDstElement = new bool[dstBufLen];
            for (int k = 0; k < sk; k++)
            {
                for (int j = 0; j < sr; j++)
                {
                    for (int i = 0; i < sc; i++)
                    {
                        isDstElement[i * d0 + j * d1 + k * d2] = true;
                    }
                }
            }

            var dstView = new TensorSpan<double>(dstBuf, [sc, sr, sk], [d0, d1, d2]);
            subT.CopyTo(dstView);

            var materialized = subT.ToTensor();
            for (int k = 0; k < sk; k++)
            {
                for (int j = 0; j < sr; j++)
                {
                    for (int i = 0; i < sc; i++)
                    {
                        Assert.Equal(subT[i, j, k], dstView[i, j, k]);
                        Assert.Equal(dstView[i, j, k], materialized[i, j, k]);
                    }
                }
            }

            for (int idx = 0; idx < dstBufLen; idx++)
            {
                if (!isDstElement[idx])
                {
                    Assert.Equal(Sentinel, dstBuf[idx]);
                }
            }
        }, iter: 200, seed: "123456789abc");
    }

    [Fact]
    public void MatrixSpan_BroadcastAndUnsqueeze_PreserveElements()
    {
        var gen = Gen.Select(Gen.Int[0, 8], Gen.Int[0, 8], Gen.Int[0, 5], Gen.Int[0, 2]);

        gen.Sample((rows, cols, broadcastCount, unsqueezeAxis) =>
        {
            var mat = Matrix<double>.Generate(rows, cols, (i, j) => (i + 1) * 10.0 + j);
            var broadcasted = mat.AsMatrixSpan().Broadcast(broadcastCount);

            Assert.Equal(rows, (int)broadcasted.Lengths[0]);
            Assert.Equal(cols, (int)broadcasted.Lengths[1]);
            Assert.Equal(broadcastCount, (int)broadcasted.Lengths[2]);

            for (int k = 0; k < broadcastCount; k++)
            {
                for (int j = 0; j < cols; j++)
                {
                    for (int i = 0; i < rows; i++)
                    {
                        Assert.Equal(mat[i, j], broadcasted[i, j, k]);
                    }
                }
            }

            var unsqueezed = mat.AsMatrixSpan().Unsqueeze(unsqueezeAxis);
            if (unsqueezeAxis == 0)
            {
                Assert.Equal(1, (int)unsqueezed.Lengths[0]);
                Assert.Equal(rows, (int)unsqueezed.Lengths[1]);
                Assert.Equal(cols, (int)unsqueezed.Lengths[2]);
                for (int j = 0; j < cols; j++)
                {
                    for (int i = 0; i < rows; i++)
                    {
                        Assert.Equal(mat[i, j], unsqueezed[0, i, j]);
                    }
                }
            }
            else if (unsqueezeAxis == 1)
            {
                Assert.Equal(rows, (int)unsqueezed.Lengths[0]);
                Assert.Equal(1, (int)unsqueezed.Lengths[1]);
                Assert.Equal(cols, (int)unsqueezed.Lengths[2]);
                for (int j = 0; j < cols; j++)
                {
                    for (int i = 0; i < rows; i++)
                    {
                        Assert.Equal(mat[i, j], unsqueezed[i, 0, j]);
                    }
                }
            }
            else
            {
                Assert.Equal(rows, (int)unsqueezed.Lengths[0]);
                Assert.Equal(cols, (int)unsqueezed.Lengths[1]);
                Assert.Equal(1, (int)unsqueezed.Lengths[2]);
                for (int j = 0; j < cols; j++)
                {
                    for (int i = 0; i < rows; i++)
                    {
                        Assert.Equal(mat[i, j], unsqueezed[i, j, 0]);
                    }
                }
            }
        }, iter: 150, seed: "23456789abcd");
    }

    [Fact]
    public void ElementwiseOperations_OverloadsAndStridedAliasing_MatchScalarReference()
    {
        var gen = Gen.Select(
            Gen.Int[0, 130],
            Gen.Int[1, 4],
            Gen.Int[1, 4],
            Gen.Int[0, 10000]);

        gen.Sample((length, srcStride, dstStride, dataSeed) =>
        {
            var rng = new Random(dataSeed);
            var xBuf = new double[Math.Max(1, length * srcStride + 4)];
            var yBuf = new double[Math.Max(1, length * srcStride + 4)];
            var dstBuf = new double[Math.Max(1, length * dstStride + 4)];
            Array.Fill(xBuf, Sentinel);
            Array.Fill(yBuf, Sentinel);
            Array.Fill(dstBuf, Sentinel);

            for (int i = 0; i < length; i++)
            {
                xBuf[i * srcStride] = rng.NextDouble() * 6.0 - 3.0;
                yBuf[i * srcStride] = rng.NextDouble() * 6.0 - 3.0;
            }

            var xSpan = new VectorSpan<double>(xBuf, [length], [srcStride]);
            var ySpan = new VectorSpan<double>(yBuf, [length], [srcStride]);
            var dstSpan = new VectorSpan<double>(dstBuf, [length], [dstStride]);

            // 新規確保版と destination 版（非連続ストライド）の一致、およびビュー外センチネル非破壊
            var allocSum = XMath.Add<double>(xSpan, ySpan);
            XMath.Add<double>(xSpan, ySpan, dstSpan);

            for (int i = 0; i < length; i++)
            {
                double expected = xSpan[i] + ySpan[i];
                Assert.Equal(expected, allocSum[i]);
                Assert.Equal(expected, dstSpan[i]);
            }

            for (int idx = 0; idx < dstBuf.Length; idx++)
            {
                bool inView = idx % dstStride == 0 && idx / dstStride < length;
                if (!inView)
                {
                    Assert.Equal(Sentinel, dstBuf[idx]);
                }
            }

            // インプレースエイリアシング（出力先に入力と同じビューを渡す）の一致
            var absAlloc = XMath.Abs<double>(xSpan);
            XMath.Abs<double>(xSpan, xSpan);
            for (int i = 0; i < length; i++)
            {
                Assert.Equal(absAlloc[i], xSpan[i]);
            }

            for (int idx = 0; idx < xBuf.Length; idx++)
            {
                bool inView = idx % srcStride == 0 && idx / srcStride < length;
                if (!inView)
                {
                    Assert.Equal(Sentinel, xBuf[idx]);
                }
            }
        }, iter: 200, seed: "3456789abcde");
    }
}
