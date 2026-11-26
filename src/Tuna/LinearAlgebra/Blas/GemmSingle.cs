using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Tuna.LinearAlgebra;

// 単精度行列積 C <- C + α*op(A)*op(B)（BLAS sgemm 相当、β = 1 固定）。
// Gemm と同じ BLIS 型構成（AVX-512: 48x8, AVX2/128-bit: 16x6）で単精度積和を行い、スレッド数によらずビット一致する。
internal static unsafe class GemmSingle
{
    private static readonly bool Use512 = Gemm.Use512;

    private static readonly bool UseSimd = Use512 || (Vector256.IsHardwareAccelerated && Avx.IsSupported) || Vector128.IsHardwareAccelerated;
    private static readonly int MR = Use512 ? 48 : 16;
    private static readonly int NR = Use512 ? 8 : 6;
    private const int KC = 256;  // A マイクロパネル MR×KC のバイト数が倍精度と同等になる内積ブロック長
    private const int MC = 288;  // パック済み A (MC×KC) が L2 に収まる行ブロック上限
    private const int NC = 3072;
    private const long SmallMK = 24 * 24;
    private const double Grain = 8 * Parallelism.Grain; // 単精度は倍精度の約 2 倍速いため Gemm の 2 倍
    private const int ParallelPackB = 16 * 1024;

    // パッキング経路を使う規模か判定する（false の場合は小規模直接積または列ごと GEMV を使用）。
    internal static bool Prefers(int m, int k) => m >= 4 && (long)m * k > SmallMK;

    // C <- C + α*op(A)*op(B)（op(A): m×k, op(B): k×n, C: m×n）。
    internal static void Multiply(
        bool transA, bool transB, int m, int n, int k, float alpha,
        ReadOnlySpan<float> a, int lda, ReadOnlySpan<float> b, int ldb, Span<float> c, int ldc)
    {
        if (m <= 0 || n <= 0 || k <= 0 || alpha == 0)
        {
            return;
        }

        if (!UseSimd || !Prefers(m, k))
        {
            MultiplySmall(transA, transB, m, n, k, alpha, a, lda, b, ldb, c, ldc);
            return;
        }

        int threads = Parallelism.Degree((double)m * n * k, Grain);
        fixed (float* pa = a)
        fixed (float* pb = b)
        fixed (float* pc = c)
        {
            new Packed(transA, transB, m, n, k, alpha, pa, lda, pb, ldb, pc, ldc).Run(threads);
        }
    }

    private static int RoundUp(int x, int m) => (x + m - 1) / m * m;

    private struct Packed
    {
        private readonly bool transA;
        private readonly bool transB;
        private readonly int m;
        private readonly int n;
        private readonly int k;
        private readonly float alpha;
        private readonly float* a;
        private readonly int lda;
        private readonly float* b;
        private readonly int ldb;
        private readonly float* c;
        private readonly int ldc;

        // 現在のブロック（jc, pc）と作業の分け方
        private float* packB;
        private int jc;
        private int nc;
        private int pc;
        private int kc;
        private int rowTasks;
        private int colTasks;
        private int packTasks;
        private int aBufferLength;

        internal Packed(bool transA, bool transB, int m, int n, int k, float alpha, float* a, int lda, float* b, int ldb, float* c, int ldc)
        {
            this.transA = transA;
            this.transB = transB;
            this.m = m;
            this.n = n;
            this.k = k;
            this.alpha = alpha;
            this.a = a;
            this.lda = lda;
            this.b = b;
            this.ldb = ldb;
            this.c = c;
            this.ldc = ldc;
        }

        internal void Run(int threads)
        {
            int mr = MR;
            int nr = NR;
            int rowPanels = (m + mr - 1) / mr;
            int perTask = MC / mr;
            if (threads <= 1)
            {
                rowTasks = (rowPanels + perTask - 1) / perTask;
            }
            else
            {
                int rounds = (rowPanels + threads * perTask - 1) / (threads * perTask);
                rowTasks = Math.Min(rowPanels, threads * rounds);
            }

            int ncMax = Math.Min(NC, RoundUp(n, nr));
            int kcMax = Math.Min(KC, k);
            // 作業と最後の短い内積ブロックによらず同じ区分から借り、動的な分担が変わっても確保しない。
            aBufferLength = checked((rowPanels + rowTasks - 1) / rowTasks * mr * kcMax);
            float[] bBuffer = WorkArrayPool<float>.Rent(checked(ncMax * kcMax));
            try
            {
                fixed (float* pb = bBuffer)
                fixed (Packed* state = &this)
                {
                    packB = pb;
                    var packWork = new PackBWork(state);
                    var computeWork = new ComputeWork(state);

                    // 処理した列数・内積方向の長さだけ進める（jc += NC・pc += KC は、n・k が int の上限に近いと最後のブロックの後に負へ折り返す）
                    for (jc = 0; jc < n; jc += nc)
                    {
                        nc = Math.Min(NC, n - jc);
                        int colPanels = (nc + nr - 1) / nr;
                        colTasks = threads <= 1 || rowTasks >= threads ? 1 : Math.Max(1, Math.Min((threads + rowTasks - 1) / rowTasks, colPanels / 2));
                        for (pc = 0; pc < k; pc += kc)
                        {
                            kc = Math.Min(KC, k - pc);
                            packTasks = threads <= 1 || nc * kc < ParallelPackB ? 1 : Math.Min(threads, colPanels);
                            if (packTasks > 1)
                            {
                                Parallelism.For(packTasks, threads, ref packWork);
                            }
                            else
                            {
                                PackB(transB, b, ldb, pc, jc, kc, nc, nr, packB);
                            }

                            int tasks = rowTasks * colTasks;
                            if (threads > 1 && tasks > 1)
                            {
                                Parallelism.For(tasks, threads, ref computeWork);
                            }
                            else
                            {
                                for (int t = 0; t < tasks; t++)
                                {
                                    ComputeTask(t);
                                }
                            }
                        }
                    }
                }
            }
            finally
            {
                WorkArrayPool<float>.Return(bBuffer);
            }
        }

        internal void PackBTask(int t)
        {
            int nr = NR;
            int panels = (nc + nr - 1) / nr;
            int q0 = (int)((long)panels * t / packTasks);
            int q1 = (int)((long)panels * (t + 1) / packTasks);
            if (q1 > q0)
            {
                int s = q0 * nr;
                PackB(transB, b, ldb, pc, jc + s, kc, Math.Min(nc, q1 * nr) - s, nr, packB + (nint)s * kc);
            }
        }

        internal void ComputeTask(int t)
        {
            int mr = MR;
            int nr = NR;
            int rowPanels = (m + mr - 1) / mr;
            int colPanels = (nc + nr - 1) / nr;
            int ri = t % rowTasks;
            int ci = t / rowTasks;
            // 行のパネルの数と範囲の番号の積は行数が数百万で int を超えるので、範囲の境界は 64 ビットで計算する
            int p0 = (int)((long)rowPanels * ri / rowTasks);
            int p1 = (int)((long)rowPanels * (ri + 1) / rowTasks);
            int q0 = (int)((long)colPanels * ci / colTasks);
            int q1 = (int)((long)colPanels * (ci + 1) / colTasks);
            if (p1 <= p0 || q1 <= q0)
            {
                return;
            }

            int i0 = p0 * mr;
            int rows = Math.Min(m, p1 * mr) - i0;
            int j0 = q0 * nr;
            int cols = Math.Min(nc, q1 * nr) - j0;
            float[] aBuffer = WorkArrayPool<float>.Rent(aBufferLength);
            try
            {
                fixed (float* pa = aBuffer)
                {
                    PackA(transA, a, lda, i0, pc, rows, kc, mr, alpha, pa);
                    MacroKernel(rows, cols, kc, mr, nr, pa, packB + (nint)j0 * kc, c + i0 + (nint)(jc + j0) * ldc, ldc);
                }
            }
            finally
            {
                WorkArrayPool<float>.Return(aBuffer);
            }
        }
    }

    private readonly struct PackBWork(Packed* state) : IParallelWork
    {
        public void Invoke(int index) => state->PackBTask(index);
    }

    private readonly struct ComputeWork(Packed* state) : IParallelWork
    {
        public void Invoke(int index) => state->ComputeTask(index);
    }

    private static void MultiplySmall(
        bool transA, bool transB, int m, int n, int k, float alpha,
        ReadOnlySpan<float> a, int lda, ReadOnlySpan<float> b, int ldb, Span<float> c, int ldc)
    {
        if (!transA)
        {
            for (int j = 0; j < n; j++)
            {
                var cj = c.Slice(j * ldc, m);
                for (int p = 0; p < k; p++)
                {
                    float bpj = transB ? b[j + p * ldb] : b[p + j * ldb];
                    VectorOps.Axpy(alpha * bpj, a.Slice(p * lda, m), cj);
                }
            }

            return;
        }

        if (!transB)
        {
            for (int j = 0; j < n; j++)
            {
                var bj = b.Slice(j * ldb, k);
                for (int i = 0; i < m; i++)
                {
                    c[i + j * ldc] += alpha * VectorOps.Dot(a.Slice(i * lda, k), bj);
                }
            }

            return;
        }

        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < m; i++)
            {
                float sum = 0;
                for (int p = 0; p < k; p++)
                {
                    sum += a[p + i * lda] * b[j + p * ldb];
                }

                c[i + j * ldc] += alpha * sum;
            }
        }
    }

    // op(A)[ic..ic+mc, pc..pc+kc] を mr 行パネルにパックし α を掛ける（AVX-512 末尾パネルは 16 の倍数幅）。
    private static void PackA(bool transA, float* a, int lda, int ic, int pc, int mc, int kc, int mr, float alpha, float* dest)
    {
        for (int r = 0; r < mc; r += mr)
        {
            int rows = Math.Min(mr, mc - r);
            int width = PanelWidth(rows, mr);
            float* panel = dest + (nint)r * kc;
            if (!transA)
            {
                float* src = a + ic + r + (nint)pc * lda;
                if (rows == width)
                {
                    // 幅は 16 の倍数（AVX2 では MR = 16）なので 256 ビットのベクトルで割り切れる
                    var va = Vector256.Create(alpha);
                    for (int p = 0; p < kc; p++, src += lda, panel += width)
                    {
                        for (int ii = 0; ii < width; ii += 8)
                        {
                            (Vector256.Load(src + ii) * va).Store(panel + ii);
                        }
                    }
                }
                else
                {
                    for (int p = 0; p < kc; p++, src += lda, panel += width)
                    {
                        int ii = 0;
                        for (; ii < rows; ii++)
                        {
                            panel[ii] = alpha * src[ii];
                        }

                        for (; ii < width; ii++)
                        {
                            panel[ii] = 0;
                        }
                    }
                }
            }
            else
            {
                // op(A)[i, p] = a[p + i·lda]: 8行 × 8列ずつレジスタ内で転置する（AVX の命令がなければ下の1行ずつの経路だけを使う）
                float* src = a + pc + (nint)(ic + r) * lda;
                var va = Vector256.Create(alpha);
                int ii = 0;
                for (; Avx.IsSupported && ii + 8 <= rows; ii += 8)
                {
                    float* s = src + (nint)ii * lda;
                    int p = 0;
                    for (; p + 8 <= kc; p += 8)
                    {
                        Transpose8(s + p, lda, panel + (nint)p * width + ii, width, va);
                    }

                    for (; p < kc; p++)
                    {
                        for (int g = 0; g < 8; g++)
                        {
                            panel[(nint)p * width + ii + g] = alpha * s[p + (nint)g * lda];
                        }
                    }
                }

                for (; ii < width; ii++)
                {
                    if (ii < rows)
                    {
                        float* s = src + (nint)ii * lda;
                        for (int p = 0; p < kc; p++)
                        {
                            panel[(nint)p * width + ii] = alpha * s[p];
                        }
                    }
                    else
                    {
                        for (int p = 0; p < kc; p++)
                        {
                            panel[(nint)p * width + ii] = 0;
                        }
                    }
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int PanelWidth(int rows, int mr) => Use512 ? Math.Min(mr, (rows + 15) & ~15) : mr;

    // op(B)[pc..pc+kc, jc..jc+nc] を nr 列パネルにパックする（端数は 0 埋め）。
    private static void PackB(bool transB, float* b, int ldb, int pc, int jc, int kc, int nc, int nr, float* dest)
    {
        for (int s = 0; s < nc; s += nr)
        {
            int cols = Math.Min(nr, nc - s);
            float* panel = dest + (nint)s * kc;
            if (!transB)
            {
                // op(B)[p, j] = b[p + j·ldb]
                float* src = b + pc + (nint)(jc + s) * ldb;
                int p = 0;
                if (cols == 8)
                {
                    var one = Vector256.Create(1f);
                    for (; p + 8 <= kc; p += 8)
                    {
                        Transpose8(src + p, ldb, panel + (nint)p * 8, 8, one);
                    }
                }

                for (; p < kc; p++)
                {
                    int jj = 0;
                    for (; jj < cols; jj++)
                    {
                        panel[(nint)p * nr + jj] = src[p + (nint)jj * ldb];
                    }

                    for (; jj < nr; jj++)
                    {
                        panel[(nint)p * nr + jj] = 0;
                    }
                }
            }
            else
            {
                // op(B)[p, j] = b[j + p·ldb]
                float* src = b + jc + s + (nint)pc * ldb;
                for (int p = 0; p < kc; p++, src += ldb, panel += nr)
                {
                    if (cols == 8)
                    {
                        Vector256.Load(src).Store(panel);
                        continue;
                    }

                    int jj = 0;
                    for (; jj < cols; jj++)
                    {
                        panel[jj] = src[jj];
                    }

                    for (; jj < nr; jj++)
                    {
                        panel[jj] = 0;
                    }
                }
            }
        }
    }

    // 8×8 レジスタ内転置＋スケーリング。
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Transpose8(float* src, int lds, float* dest, int ldd, Vector256<float> scale)
    {
        nint ls = lds;
        var r0 = Vector256.Load(src);
        var r1 = Vector256.Load(src + ls);
        var r2 = Vector256.Load(src + 2 * ls);
        var r3 = Vector256.Load(src + 3 * ls);
        var r4 = Vector256.Load(src + 4 * ls);
        var r5 = Vector256.Load(src + 5 * ls);
        var r6 = Vector256.Load(src + 6 * ls);
        var r7 = Vector256.Load(src + 7 * ls);

        // 2×2 の要素単位（128 ビットの半分ごと）
        var t0 = Avx.UnpackLow(r0, r1);
        var t1 = Avx.UnpackHigh(r0, r1);
        var t2 = Avx.UnpackLow(r2, r3);
        var t3 = Avx.UnpackHigh(r2, r3);
        var t4 = Avx.UnpackLow(r4, r5);
        var t5 = Avx.UnpackHigh(r4, r5);
        var t6 = Avx.UnpackLow(r6, r7);
        var t7 = Avx.UnpackHigh(r6, r7);

        // 4×4 の要素単位（128 ビットの半分ごと）
        var u0 = Avx.Shuffle(t0, t2, 0x44);
        var u1 = Avx.Shuffle(t0, t2, 0xEE);
        var u2 = Avx.Shuffle(t1, t3, 0x44);
        var u3 = Avx.Shuffle(t1, t3, 0xEE);
        var u4 = Avx.Shuffle(t4, t6, 0x44);
        var u5 = Avx.Shuffle(t4, t6, 0xEE);
        var u6 = Avx.Shuffle(t5, t7, 0x44);
        var u7 = Avx.Shuffle(t5, t7, 0xEE);

        // 128 ビット単位
        nint ld = ldd;
        (Avx.Permute2x128(u0, u4, 0x20) * scale).Store(dest);
        (Avx.Permute2x128(u1, u5, 0x20) * scale).Store(dest + ld);
        (Avx.Permute2x128(u2, u6, 0x20) * scale).Store(dest + 2 * ld);
        (Avx.Permute2x128(u3, u7, 0x20) * scale).Store(dest + 3 * ld);
        (Avx.Permute2x128(u0, u4, 0x31) * scale).Store(dest + 4 * ld);
        (Avx.Permute2x128(u1, u5, 0x31) * scale).Store(dest + 5 * ld);
        (Avx.Permute2x128(u2, u6, 0x31) * scale).Store(dest + 6 * ld);
        (Avx.Permute2x128(u3, u7, 0x31) * scale).Store(dest + 7 * ld);
    }

    private static void MacroKernel(int mc, int nc, int kc, int mr, int nr, float* packA, float* packB, float* c, int ldc)
    {
        float* tile = stackalloc float[48 * 8];
        nint ld = ldc;
        for (int jr = 0; jr < nc; jr += nr)
        {
            int cols = Math.Min(nr, nc - jr);
            float* pb = packB + (nint)jr * kc;
            for (int ir = 0; ir < mc; ir += mr)
            {
                int rows = Math.Min(mr, mc - ir);
                int width = PanelWidth(rows, mr);
                float* pa = packA + (nint)ir * kc;
                float* cc = c + ir + jr * ld;
                if (rows == width && cols == nr)
                {
                    MicroKernel(width, kc, pa, pb, cc, ld);
                }
                else
                {
                    new Span<float>(tile, width * nr).Clear();
                    MicroKernel(width, kc, pa, pb, tile, width);
                    for (int jj = 0; jj < cols; jj++)
                    {
                        float* dst = cc + jj * ld;
                        float* src = tile + jj * width;
                        for (int ii = 0; ii < rows; ii++)
                        {
                            dst[ii] += src[ii];
                        }
                    }
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void MicroKernel(int width, int kc, float* a, float* b, float* c, nint ldc)
    {
        if (!Use512)
        {
            if (Vector256.IsHardwareAccelerated)
            {
                Kernel16x6(kc, a, b, c, ldc);
            }
            else
            {
                Kernel16x6Narrow(kc, a, b, c, ldc);
            }
        }
        else if (width == 48)
        {
            Kernel48x8(kc, a, b, c, ldc);
        }
        else if (width == 32)
        {
            Kernel32x8(kc, a, b, c, ldc);
        }
        else
        {
            Kernel16x8(kc, a, b, c, ldc);
        }
    }

    // AVX-512 48×8 マイクロカーネル（24 アキュムレータ）。
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel48x8(int kc, float* a, float* b, float* c, nint ldc)
    {
        Vector512<float> c00 = default, c01 = default, c02 = default, c03 = default, c04 = default, c05 = default, c06 = default, c07 = default;
        Vector512<float> c10 = default, c11 = default, c12 = default, c13 = default, c14 = default, c15 = default, c16 = default, c17 = default;
        Vector512<float> c20 = default, c21 = default, c22 = default, c23 = default, c24 = default, c25 = default, c26 = default, c27 = default;
        for (int p = 0; p < kc; p++, a += 48, b += 8)
        {
            var a0 = Vector512.Load(a);
            var a1 = Vector512.Load(a + 16);
            var a2 = Vector512.Load(a + 32);
            var bv = Vector512.Create(b[0]);
            c00 = Vector512.FusedMultiplyAdd(a0, bv, c00);
            c10 = Vector512.FusedMultiplyAdd(a1, bv, c10);
            c20 = Vector512.FusedMultiplyAdd(a2, bv, c20);
            bv = Vector512.Create(b[1]);
            c01 = Vector512.FusedMultiplyAdd(a0, bv, c01);
            c11 = Vector512.FusedMultiplyAdd(a1, bv, c11);
            c21 = Vector512.FusedMultiplyAdd(a2, bv, c21);
            bv = Vector512.Create(b[2]);
            c02 = Vector512.FusedMultiplyAdd(a0, bv, c02);
            c12 = Vector512.FusedMultiplyAdd(a1, bv, c12);
            c22 = Vector512.FusedMultiplyAdd(a2, bv, c22);
            bv = Vector512.Create(b[3]);
            c03 = Vector512.FusedMultiplyAdd(a0, bv, c03);
            c13 = Vector512.FusedMultiplyAdd(a1, bv, c13);
            c23 = Vector512.FusedMultiplyAdd(a2, bv, c23);
            bv = Vector512.Create(b[4]);
            c04 = Vector512.FusedMultiplyAdd(a0, bv, c04);
            c14 = Vector512.FusedMultiplyAdd(a1, bv, c14);
            c24 = Vector512.FusedMultiplyAdd(a2, bv, c24);
            bv = Vector512.Create(b[5]);
            c05 = Vector512.FusedMultiplyAdd(a0, bv, c05);
            c15 = Vector512.FusedMultiplyAdd(a1, bv, c15);
            c25 = Vector512.FusedMultiplyAdd(a2, bv, c25);
            bv = Vector512.Create(b[6]);
            c06 = Vector512.FusedMultiplyAdd(a0, bv, c06);
            c16 = Vector512.FusedMultiplyAdd(a1, bv, c16);
            c26 = Vector512.FusedMultiplyAdd(a2, bv, c26);
            bv = Vector512.Create(b[7]);
            c07 = Vector512.FusedMultiplyAdd(a0, bv, c07);
            c17 = Vector512.FusedMultiplyAdd(a1, bv, c17);
            c27 = Vector512.FusedMultiplyAdd(a2, bv, c27);
        }

        Add3(c, c00, c10, c20);
        Add3(c += ldc, c01, c11, c21);
        Add3(c += ldc, c02, c12, c22);
        Add3(c += ldc, c03, c13, c23);
        Add3(c += ldc, c04, c14, c24);
        Add3(c += ldc, c05, c15, c25);
        Add3(c += ldc, c06, c16, c26);
        Add3(c += ldc, c07, c17, c27);
    }

    // AVX-512 32×8 マイクロカーネル（末尾パネル用）。
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel32x8(int kc, float* a, float* b, float* c, nint ldc)
    {
        Vector512<float> c00 = default, c01 = default, c02 = default, c03 = default, c04 = default, c05 = default, c06 = default, c07 = default;
        Vector512<float> c10 = default, c11 = default, c12 = default, c13 = default, c14 = default, c15 = default, c16 = default, c17 = default;
        for (int p = 0; p < kc; p++, a += 32, b += 8)
        {
            var a0 = Vector512.Load(a);
            var a1 = Vector512.Load(a + 16);
            var bv = Vector512.Create(b[0]);
            c00 = Vector512.FusedMultiplyAdd(a0, bv, c00);
            c10 = Vector512.FusedMultiplyAdd(a1, bv, c10);
            bv = Vector512.Create(b[1]);
            c01 = Vector512.FusedMultiplyAdd(a0, bv, c01);
            c11 = Vector512.FusedMultiplyAdd(a1, bv, c11);
            bv = Vector512.Create(b[2]);
            c02 = Vector512.FusedMultiplyAdd(a0, bv, c02);
            c12 = Vector512.FusedMultiplyAdd(a1, bv, c12);
            bv = Vector512.Create(b[3]);
            c03 = Vector512.FusedMultiplyAdd(a0, bv, c03);
            c13 = Vector512.FusedMultiplyAdd(a1, bv, c13);
            bv = Vector512.Create(b[4]);
            c04 = Vector512.FusedMultiplyAdd(a0, bv, c04);
            c14 = Vector512.FusedMultiplyAdd(a1, bv, c14);
            bv = Vector512.Create(b[5]);
            c05 = Vector512.FusedMultiplyAdd(a0, bv, c05);
            c15 = Vector512.FusedMultiplyAdd(a1, bv, c15);
            bv = Vector512.Create(b[6]);
            c06 = Vector512.FusedMultiplyAdd(a0, bv, c06);
            c16 = Vector512.FusedMultiplyAdd(a1, bv, c16);
            bv = Vector512.Create(b[7]);
            c07 = Vector512.FusedMultiplyAdd(a0, bv, c07);
            c17 = Vector512.FusedMultiplyAdd(a1, bv, c17);
        }

        Add2(c, c00, c10);
        Add2(c += ldc, c01, c11);
        Add2(c += ldc, c02, c12);
        Add2(c += ldc, c03, c13);
        Add2(c += ldc, c04, c14);
        Add2(c += ldc, c05, c15);
        Add2(c += ldc, c06, c16);
        Add2(c += ldc, c07, c17);
    }

    // AVX-512 16×8 マイクロカーネル（k を 2 系統に分割し依存鎖を短縮）。
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel16x8(int kc, float* a, float* b, float* c, nint ldc)
    {
        Vector512<float> c0 = default, c1 = default, c2 = default, c3 = default, c4 = default, c5 = default, c6 = default, c7 = default;
        Vector512<float> d0 = default, d1 = default, d2 = default, d3 = default, d4 = default, d5 = default, d6 = default, d7 = default;
        int p = 0;
        for (; p + 2 <= kc; p += 2, a += 32, b += 16)
        {
            var a0 = Vector512.Load(a);
            var a1 = Vector512.Load(a + 16);
            c0 = Vector512.FusedMultiplyAdd(a0, Vector512.Create(b[0]), c0);
            c1 = Vector512.FusedMultiplyAdd(a0, Vector512.Create(b[1]), c1);
            c2 = Vector512.FusedMultiplyAdd(a0, Vector512.Create(b[2]), c2);
            c3 = Vector512.FusedMultiplyAdd(a0, Vector512.Create(b[3]), c3);
            c4 = Vector512.FusedMultiplyAdd(a0, Vector512.Create(b[4]), c4);
            c5 = Vector512.FusedMultiplyAdd(a0, Vector512.Create(b[5]), c5);
            c6 = Vector512.FusedMultiplyAdd(a0, Vector512.Create(b[6]), c6);
            c7 = Vector512.FusedMultiplyAdd(a0, Vector512.Create(b[7]), c7);
            d0 = Vector512.FusedMultiplyAdd(a1, Vector512.Create(b[8]), d0);
            d1 = Vector512.FusedMultiplyAdd(a1, Vector512.Create(b[9]), d1);
            d2 = Vector512.FusedMultiplyAdd(a1, Vector512.Create(b[10]), d2);
            d3 = Vector512.FusedMultiplyAdd(a1, Vector512.Create(b[11]), d3);
            d4 = Vector512.FusedMultiplyAdd(a1, Vector512.Create(b[12]), d4);
            d5 = Vector512.FusedMultiplyAdd(a1, Vector512.Create(b[13]), d5);
            d6 = Vector512.FusedMultiplyAdd(a1, Vector512.Create(b[14]), d6);
            d7 = Vector512.FusedMultiplyAdd(a1, Vector512.Create(b[15]), d7);
        }

        if (p < kc)
        {
            var a0 = Vector512.Load(a);
            c0 = Vector512.FusedMultiplyAdd(a0, Vector512.Create(b[0]), c0);
            c1 = Vector512.FusedMultiplyAdd(a0, Vector512.Create(b[1]), c1);
            c2 = Vector512.FusedMultiplyAdd(a0, Vector512.Create(b[2]), c2);
            c3 = Vector512.FusedMultiplyAdd(a0, Vector512.Create(b[3]), c3);
            c4 = Vector512.FusedMultiplyAdd(a0, Vector512.Create(b[4]), c4);
            c5 = Vector512.FusedMultiplyAdd(a0, Vector512.Create(b[5]), c5);
            c6 = Vector512.FusedMultiplyAdd(a0, Vector512.Create(b[6]), c6);
            c7 = Vector512.FusedMultiplyAdd(a0, Vector512.Create(b[7]), c7);
        }

        Add1(c, c0 + d0);
        Add1(c += ldc, c1 + d1);
        Add1(c += ldc, c2 + d2);
        Add1(c += ldc, c3 + d3);
        Add1(c += ldc, c4 + d4);
        Add1(c += ldc, c5 + d5);
        Add1(c += ldc, c6 + d6);
        Add1(c += ldc, c7 + d7);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Add1(float* c, Vector512<float> x) => (Vector512.Load(c) + x).Store(c);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Add2(float* c, Vector512<float> x, Vector512<float> y)
    {
        (Vector512.Load(c) + x).Store(c);
        (Vector512.Load(c + 16) + y).Store(c + 16);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Add3(float* c, Vector512<float> x, Vector512<float> y, Vector512<float> z)
    {
        (Vector512.Load(c) + x).Store(c);
        (Vector512.Load(c + 16) + y).Store(c + 16);
        (Vector512.Load(c + 32) + z).Store(c + 32);
    }

    // AVX2 16×6 マイクロカーネル（12 アキュムレータ）。
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel16x6(int kc, float* a, float* b, float* c, nint ldc)
    {
        Vector256<float> c00 = default, c01 = default, c02 = default, c03 = default, c04 = default, c05 = default;
        Vector256<float> c10 = default, c11 = default, c12 = default, c13 = default, c14 = default, c15 = default;
        for (int p = 0; p < kc; p++, a += 16, b += 6)
        {
            var a0 = Vector256.Load(a);
            var a1 = Vector256.Load(a + 8);
            var bv = Vector256.Create(b[0]);
            c00 = Vector256.MultiplyAddEstimate(a0, bv, c00);
            c10 = Vector256.MultiplyAddEstimate(a1, bv, c10);
            bv = Vector256.Create(b[1]);
            c01 = Vector256.MultiplyAddEstimate(a0, bv, c01);
            c11 = Vector256.MultiplyAddEstimate(a1, bv, c11);
            bv = Vector256.Create(b[2]);
            c02 = Vector256.MultiplyAddEstimate(a0, bv, c02);
            c12 = Vector256.MultiplyAddEstimate(a1, bv, c12);
            bv = Vector256.Create(b[3]);
            c03 = Vector256.MultiplyAddEstimate(a0, bv, c03);
            c13 = Vector256.MultiplyAddEstimate(a1, bv, c13);
            bv = Vector256.Create(b[4]);
            c04 = Vector256.MultiplyAddEstimate(a0, bv, c04);
            c14 = Vector256.MultiplyAddEstimate(a1, bv, c14);
            bv = Vector256.Create(b[5]);
            c05 = Vector256.MultiplyAddEstimate(a0, bv, c05);
            c15 = Vector256.MultiplyAddEstimate(a1, bv, c15);
        }

        Add256(c, c00, c10);
        Add256(c += ldc, c01, c11);
        Add256(c += ldc, c02, c12);
        Add256(c += ldc, c03, c13);
        Add256(c += ldc, c04, c14);
        Add256(c += ldc, c05, c15);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Add256(float* c, Vector256<float> lo, Vector256<float> hi)
    {
        (Vector256.Load(c) + lo).Store(c);
        (Vector256.Load(c + 8) + hi).Store(c + 8);
    }

    // 128 ビット SIMD 用 16×6 マイクロカーネル（Kernel16x6 と同一タイル・同一積和順序）。
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel16x6Narrow(int kc, float* a, float* b, float* c, nint ldc)
    {
        Vector128<float> c00 = default, c01 = default, c02 = default, c03 = default, c04 = default, c05 = default;
        Vector128<float> c10 = default, c11 = default, c12 = default, c13 = default, c14 = default, c15 = default;
        Vector128<float> c20 = default, c21 = default, c22 = default, c23 = default, c24 = default, c25 = default;
        Vector128<float> c30 = default, c31 = default, c32 = default, c33 = default, c34 = default, c35 = default;
        for (int p = 0; p < kc; p++, a += 16, b += 6)
        {
            var a0 = Vector128.Load(a);
            var a1 = Vector128.Load(a + 4);
            var a2 = Vector128.Load(a + 8);
            var a3 = Vector128.Load(a + 12);
            var bv = Vector128.Create(b[0]);
            c00 = Vector128.MultiplyAddEstimate(a0, bv, c00);
            c10 = Vector128.MultiplyAddEstimate(a1, bv, c10);
            c20 = Vector128.MultiplyAddEstimate(a2, bv, c20);
            c30 = Vector128.MultiplyAddEstimate(a3, bv, c30);
            bv = Vector128.Create(b[1]);
            c01 = Vector128.MultiplyAddEstimate(a0, bv, c01);
            c11 = Vector128.MultiplyAddEstimate(a1, bv, c11);
            c21 = Vector128.MultiplyAddEstimate(a2, bv, c21);
            c31 = Vector128.MultiplyAddEstimate(a3, bv, c31);
            bv = Vector128.Create(b[2]);
            c02 = Vector128.MultiplyAddEstimate(a0, bv, c02);
            c12 = Vector128.MultiplyAddEstimate(a1, bv, c12);
            c22 = Vector128.MultiplyAddEstimate(a2, bv, c22);
            c32 = Vector128.MultiplyAddEstimate(a3, bv, c32);
            bv = Vector128.Create(b[3]);
            c03 = Vector128.MultiplyAddEstimate(a0, bv, c03);
            c13 = Vector128.MultiplyAddEstimate(a1, bv, c13);
            c23 = Vector128.MultiplyAddEstimate(a2, bv, c23);
            c33 = Vector128.MultiplyAddEstimate(a3, bv, c33);
            bv = Vector128.Create(b[4]);
            c04 = Vector128.MultiplyAddEstimate(a0, bv, c04);
            c14 = Vector128.MultiplyAddEstimate(a1, bv, c14);
            c24 = Vector128.MultiplyAddEstimate(a2, bv, c24);
            c34 = Vector128.MultiplyAddEstimate(a3, bv, c34);
            bv = Vector128.Create(b[5]);
            c05 = Vector128.MultiplyAddEstimate(a0, bv, c05);
            c15 = Vector128.MultiplyAddEstimate(a1, bv, c15);
            c25 = Vector128.MultiplyAddEstimate(a2, bv, c25);
            c35 = Vector128.MultiplyAddEstimate(a3, bv, c35);
        }

        Add128(c, c00, c10, c20, c30);
        Add128(c += ldc, c01, c11, c21, c31);
        Add128(c += ldc, c02, c12, c22, c32);
        Add128(c += ldc, c03, c13, c23, c33);
        Add128(c += ldc, c04, c14, c24, c34);
        Add128(c += ldc, c05, c15, c25, c35);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Add128(float* c, Vector128<float> v0, Vector128<float> v1, Vector128<float> v2, Vector128<float> v3)
    {
        (Vector128.Load(c) + v0).Store(c);
        (Vector128.Load(c + 4) + v1).Store(c + 4);
        (Vector128.Load(c + 8) + v2).Store(c + 8);
        (Vector128.Load(c + 12) + v3).Store(c + 12);
    }
}
