using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Tuna.LinearAlgebra;

// 倍精度行列積 C <- C + α*op(A)*op(B)（BLAS dgemm 相当、β = 1 固定）。
// GotoBLAS/BLIS 型のパッキングとマイクロカーネル（AVX-512: 24x8, AVX2/128-bit: 8x6）を用い、スレッド数や分割によらずビット一致する。
internal static unsafe class Gemm
{
    internal static readonly bool Use512 = Vector512.IsHardwareAccelerated && Avx512F.IsSupported;
    // 512/256/128 ビット SIMD のいずれかが有効ならパッキング＋マイクロカーネル経路を使う。
    private static readonly bool UseSimd = Use512 || Vector256.IsHardwareAccelerated || Vector128.IsHardwareAccelerated;

    private static readonly int MR = Use512 ? 24 : 8;
    private static readonly int NR = Use512 ? 8 : 6;
    private const int KC = 256;  // B マイクロパネル KC×NR が L1 に収まる内積方向ブロック長
    private const int MC = 144;  // A ブロック MC×KC が L2 に収まる行ブロック上限
    private const int NC = 3072;

    // m*k がこれ以下ならパッキングせず直接計算する（n に依存させないことで列分割時のビット一致を保つ）。
    private const long SmallMK = 24 * 24;
    private const double Grain = 4 * Parallelism.Grain; // キャッシュ間転送コストを見込んだ 1 スレッド最小積和数
    private const int ParallelPackB = 16 * 1024;

    // C <- C + α*op(A)*op(B)（op(A): m×k, op(B): k×n, C: m×n）。
    internal static void Multiply(
        bool transA, bool transB, int m, int n, int k, double alpha,
        ReadOnlySpan<double> a, int lda, ReadOnlySpan<double> b, int ldb, Span<double> c, int ldc)
    {
        if (m <= 0 || n <= 0 || k <= 0 || alpha == 0)
        {
            return;
        }

        if (!UseSimd || (long)m * k <= SmallMK || m < 4)
        {
            MultiplySmall(transA, transB, m, n, k, alpha, a, lda, b, ldb, c, ldc);
            return;
        }

        int threads = Parallelism.Degree((double)m * n * k, Grain);
        fixed (double* pa = a)
        fixed (double* pb = b)
        fixed (double* pc = c)
        {
            new Packed(transA, transB, m, n, k, alpha, pa, lda, pb, ldb, pc, ldc).Run(threads);
        }
    }

    // C <- C + α*A*B^T の下三角（行 >= 列）のみ計算する（SYRK 用）。対角に掛かるタイルは上三角側も書き換える。
    internal static void MultiplyLower(
        int m, int n, int k, double alpha, ReadOnlySpan<double> a, int lda, ReadOnlySpan<double> b, int ldb, Span<double> c, int ldc)
    {
        if (m <= 0 || n <= 0 || k <= 0 || alpha == 0)
        {
            return;
        }

        if (UseSimd && (long)m * k > SmallMK && m >= 4)
        {
            int threads = Parallelism.Degree((double)(m - n / 2.0) * n * k, Grain);
            fixed (double* pa = a)
            fixed (double* pb = b)
            fixed (double* pc = c)
            {
                new Packed(false, true, m, n, k, alpha, pa, lda, pb, ldb, pc, ldc, lower: true).Run(threads);
            }

            return;
        }

        // 列を幅 32 ずつ区切り、各区切りの対角より下の行だけを計算する
        for (int j = 0; j < n && j < m; j += 32)
        {
            int w = Math.Min(32, n - j);
            Multiply(false, true, m - j, w, k, alpha, a[j..], lda, b[j..], ldb, c[(j + j * ldc)..], ldc);
        }
    }

    // C <- C + α*A^T*B の下三角のみ計算する（対角に掛かるタイルは上三角側も書き換える）。
    internal static void MultiplyLowerTransposed(
        int m, int n, int k, double alpha, ReadOnlySpan<double> a, int lda, ReadOnlySpan<double> b, int ldb, Span<double> c, int ldc)
    {
        if (m <= 0 || n <= 0 || k <= 0 || alpha == 0)
        {
            return;
        }

        if (UseSimd && (long)m * k > SmallMK && m >= 4)
        {
            int threads = Parallelism.Degree((double)(m - n / 2.0) * n * k, Grain);
            fixed (double* pa = a)
            fixed (double* pb = b)
            fixed (double* pc = c)
            {
                new Packed(true, false, m, n, k, alpha, pa, lda, pb, ldb, pc, ldc, lower: true).Run(threads);
            }

            return;
        }

        for (int j = 0; j < n && j < m; j += 32)
        {
            int w = Math.Min(32, n - j);
            Multiply(true, false, m - j, w, k, alpha, a[(j * lda)..], lda, b[(j * ldb)..], ldb, c[(j + j * ldc)..], ldc);
        }
    }

    private static int RoundUp(int x, int m) => (x + m - 1) / m * m;

    // パッキング行列積 1 回分の共有状態（並列区間完了までスタック上に固定）。
    private struct Packed
    {
        private readonly bool transA;
        private readonly bool transB;
        private readonly int m;
        private readonly int n;
        private readonly int k;
        private readonly double alpha;
        private readonly double* a;
        private readonly int lda;
        private readonly double* b;
        private readonly int ldb;
        private readonly double* c;
        private readonly int ldc;

        private readonly bool lower; // 下三角（行 >= 列）に掛かるタイルのみ計算するか

        // 現在のブロック（jc, pc）と作業の分け方
        private double* packB;
        private int jc;
        private int nc;
        private int pc;
        private int kc;
        private int rowTasks;
        private int colTasks;
        private int packTasks;
        private int aBufferLength;

        internal Packed(bool transA, bool transB, int m, int n, int k, double alpha, double* a, int lda, double* b, int ldb, double* c, int ldc, bool lower = false)
        {
            this.lower = lower;
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
                // 各スレッドが同じ回数ずつ作業を受け持つように、行のパネルを均等に分ける
                int rounds = (rowPanels + threads * perTask - 1) / (threads * perTask);
                rowTasks = Math.Min(rowPanels, threads * rounds);
            }

            int ncMax = Math.Min(NC, RoundUp(n, nr));
            int kcMax = Math.Min(KC, k);
            // 作業と最後の短い内積ブロックによらず同じ区分から借り、動的な分担が変わっても確保しない。
            aBufferLength = checked((rowPanels + rowTasks - 1) / rowTasks * mr * kcMax);
            double[] bBuffer = WorkArrayPool<double>.Rent(checked(ncMax * kcMax));
            try
            {
                fixed (double* pb = bBuffer)
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

                        // 行のパネルが足りなければ列の範囲も分ける（1つの範囲は2パネル以上）
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
                WorkArrayPool<double>.Return(bBuffer);
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
            if (lower && i0 + rows <= jc + j0)
            {
                return;
            }

            double[] aBuffer = WorkArrayPool<double>.Rent(aBufferLength);
            try
            {
                fixed (double* pa = aBuffer)
                {
                    PackA(transA, a, lda, i0, pc, rows, kc, mr, alpha, pa);
                    MacroKernel(rows, cols, kc, mr, nr, pa, packB + (nint)j0 * kc, c + i0 + (nint)(jc + j0) * ldc, ldc, lower ? jc + j0 - i0 : int.MinValue);
                }
            }
            finally
            {
                WorkArrayPool<double>.Return(aBuffer);
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

    // 小規模行列および SIMD 非対応環境向けの非パッキング行列積。
    private static void MultiplySmall(
        bool transA, bool transB, int m, int n, int k, double alpha,
        ReadOnlySpan<double> a, int lda, ReadOnlySpan<double> b, int ldb, Span<double> c, int ldc)
    {
        if (!transA)
        {
            for (int j = 0; j < n; j++)
            {
                var cj = c.Slice(j * ldc, m);
                for (int p = 0; p < k; p++)
                {
                    double bpj = transB ? b[j + p * ldb] : b[p + j * ldb];
                    Level1Kernels.Axpy(alpha * bpj, a.Slice(p * lda, m), cj);
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
                    c[i + j * ldc] += alpha * Level1Kernels.Dot(a.Slice(i * lda, k), bj);
                }
            }

            return;
        }

        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < m; i++)
            {
                double sum = 0;
                for (int p = 0; p < k; p++)
                {
                    sum += a[p + i * lda] * b[j + p * ldb];
                }

                c[i + j * ldc] += alpha * sum;
            }
        }
    }

    // op(A)[ic..ic+mc, pc..pc+kc] を mr 行パネル（各 k で行連続）にパックし α を掛ける（AVX-512 末尾パネルは 8 の倍数幅）。
    private static void PackA(bool transA, double* a, int lda, int ic, int pc, int mc, int kc, int mr, double alpha, double* dest)
    {
        for (int r = 0; r < mc; r += mr)
        {
            int rows = Math.Min(mr, mc - r);
            int width = PanelWidth(rows, mr);
            double* panel = dest + (nint)r * kc;
            if (!transA)
            {
                double* src = a + ic + r + (nint)pc * lda;
                if (Use512 && rows == width)
                {
                    var va = Vector512.Create(alpha);
                    for (int p = 0; p < kc; p++, src += lda, panel += width)
                    {
                        for (int ii = 0; ii < width; ii += 8)
                        {
                            (Vector512.Load(src + ii) * va).Store(panel + ii);
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
                // op(A)[i, p] = a[p + i·lda]: 8行 × 8列ずつレジスタ内で転置する
                double* src = a + pc + (nint)(ic + r) * lda;
                int ii = 0;
                if (Use512)
                {
                    var va = Vector512.Create(alpha);
                    for (; ii + 8 <= rows; ii += 8)
                    {
                        double* s = src + (nint)ii * lda;
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
                }

                for (; ii < width; ii++)
                {
                    if (ii < rows)
                    {
                        double* s = src + (nint)ii * lda;
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
    private static int PanelWidth(int rows, int mr) => Use512 ? Math.Min(mr, (rows + 7) & ~7) : mr;

    // op(B)[pc..pc+kc, jc..jc+nc] を nr 列パネル（各 k で列連続）にパックする（端数は 0 埋め）。
    private static void PackB(bool transB, double* b, int ldb, int pc, int jc, int kc, int nc, int nr, double* dest)
    {
        for (int s = 0; s < nc; s += nr)
        {
            int cols = Math.Min(nr, nc - s);
            double* panel = dest + (nint)s * kc;
            if (!transB)
            {
                // op(B)[p, j] = b[p + j·ldb]
                double* src = b + pc + (nint)(jc + s) * ldb;
                int p = 0;
                if (Use512 && cols == 8)
                {
                    var one = Vector512.Create(1.0);
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
                double* src = b + jc + s + (nint)pc * ldb;
                for (int p = 0; p < kc; p++, src += ldb, panel += nr)
                {
                    int jj = 0;
                    if (Use512 && cols == 8)
                    {
                        Vector512.Load(src).Store(panel);
                        continue;
                    }

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
    internal static void Transpose8(double* src, int lds, double* dest, int ldd, Vector512<double> scale)
    {
        nint ls = lds;
        var r0 = Vector512.Load(src);
        var r1 = Vector512.Load(src + ls);
        var r2 = Vector512.Load(src + 2 * ls);
        var r3 = Vector512.Load(src + 3 * ls);
        var r4 = Vector512.Load(src + 4 * ls);
        var r5 = Vector512.Load(src + 5 * ls);
        var r6 = Vector512.Load(src + 6 * ls);
        var r7 = Vector512.Load(src + 7 * ls);

        // 2×2 の要素単位
        var t0 = Avx512F.UnpackLow(r0, r1);
        var t1 = Avx512F.UnpackHigh(r0, r1);
        var t2 = Avx512F.UnpackLow(r2, r3);
        var t3 = Avx512F.UnpackHigh(r2, r3);
        var t4 = Avx512F.UnpackLow(r4, r5);
        var t5 = Avx512F.UnpackHigh(r4, r5);
        var t6 = Avx512F.UnpackLow(r6, r7);
        var t7 = Avx512F.UnpackHigh(r6, r7);

        // 128 ビット単位
        var u0 = Avx512F.Shuffle4x128(t0, t2, 0x44);
        var u1 = Avx512F.Shuffle4x128(t0, t2, 0xEE);
        var u2 = Avx512F.Shuffle4x128(t1, t3, 0x44);
        var u3 = Avx512F.Shuffle4x128(t1, t3, 0xEE);
        var u4 = Avx512F.Shuffle4x128(t4, t6, 0x44);
        var u5 = Avx512F.Shuffle4x128(t4, t6, 0xEE);
        var u6 = Avx512F.Shuffle4x128(t5, t7, 0x44);
        var u7 = Avx512F.Shuffle4x128(t5, t7, 0xEE);

        nint ld = ldd;
        (Avx512F.Shuffle4x128(u0, u4, 0x88) * scale).Store(dest);
        (Avx512F.Shuffle4x128(u2, u6, 0x88) * scale).Store(dest + ld);
        (Avx512F.Shuffle4x128(u0, u4, 0xDD) * scale).Store(dest + 2 * ld);
        (Avx512F.Shuffle4x128(u2, u6, 0xDD) * scale).Store(dest + 3 * ld);
        (Avx512F.Shuffle4x128(u1, u5, 0x88) * scale).Store(dest + 4 * ld);
        (Avx512F.Shuffle4x128(u3, u7, 0x88) * scale).Store(dest + 5 * ld);
        (Avx512F.Shuffle4x128(u1, u5, 0xDD) * scale).Store(dest + 6 * ld);
        (Avx512F.Shuffle4x128(u3, u7, 0xDD) * scale).Store(dest + 7 * ld);
    }

    // パック済み A・B の積を C に加算する。i + rows <= j + diagonal（全体が対角より上）のタイルはスキップ（全計算時は int.MinValue）。
    private static void MacroKernel(int mc, int nc, int kc, int mr, int nr, double* packA, double* packB, double* c, int ldc, int diagonal = int.MinValue)
    {
        double* tile = stackalloc double[24 * 8];
        nint ld = ldc;
        for (int jr = 0; jr < nc; jr += nr)
        {
            int cols = Math.Min(nr, nc - jr);
            double* pb = packB + (nint)jr * kc;
            for (int ir = 0; ir < mc; ir += mr)
            {
                int rows = Math.Min(mr, mc - ir);
                if (ir + rows <= jr + diagonal)
                {
                    continue;
                }

                int width = PanelWidth(rows, mr);
                double* pa = packA + (nint)ir * kc;
                double* cc = c + ir + jr * ld;
                if (rows == width && cols == nr)
                {
                    MicroKernel(width, kc, pa, pb, cc, ld);
                }
                else
                {
                    new Span<double>(tile, width * nr).Clear();
                    MicroKernel(width, kc, pa, pb, tile, width);
                    for (int jj = 0; jj < cols; jj++)
                    {
                        double* dst = cc + jj * ld;
                        double* src = tile + jj * width;
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
    private static void MicroKernel(int width, int kc, double* a, double* b, double* c, nint ldc)
    {
        if (!Use512)
        {
            if (Vector256.IsHardwareAccelerated)
            {
                Kernel8x6(kc, a, b, c, ldc);
            }
            else
            {
                Kernel8x6Narrow(kc, a, b, c, ldc);
            }
        }
        else if (width == 24)
        {
            Kernel24x8(kc, a, b, c, ldc);
        }
        else if (width == 16)
        {
            Kernel16x8(kc, a, b, c, ldc);
        }
        else
        {
            Kernel8x8(kc, a, b, c, ldc);
        }
    }

    // AVX-512 24×8 マイクロカーネル（24 アキュムレータ）。
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel24x8(int kc, double* a, double* b, double* c, nint ldc)
    {
        Vector512<double> c00 = default, c01 = default, c02 = default, c03 = default, c04 = default, c05 = default, c06 = default, c07 = default;
        Vector512<double> c10 = default, c11 = default, c12 = default, c13 = default, c14 = default, c15 = default, c16 = default, c17 = default;
        Vector512<double> c20 = default, c21 = default, c22 = default, c23 = default, c24 = default, c25 = default, c26 = default, c27 = default;
        for (int p = 0; p < kc; p++, a += 24, b += 8)
        {
            var a0 = Vector512.Load(a);
            var a1 = Vector512.Load(a + 8);
            var a2 = Vector512.Load(a + 16);
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

    // AVX-512 16×8 マイクロカーネル（末尾パネル用）。
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel16x8(int kc, double* a, double* b, double* c, nint ldc)
    {
        Vector512<double> c00 = default, c01 = default, c02 = default, c03 = default, c04 = default, c05 = default, c06 = default, c07 = default;
        Vector512<double> c10 = default, c11 = default, c12 = default, c13 = default, c14 = default, c15 = default, c16 = default, c17 = default;
        for (int p = 0; p < kc; p++, a += 16, b += 8)
        {
            var a0 = Vector512.Load(a);
            var a1 = Vector512.Load(a + 8);
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

    // AVX-512 8×8 マイクロカーネル（k を 2 系統に分割し依存鎖を短縮）。
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel8x8(int kc, double* a, double* b, double* c, nint ldc)
    {
        Vector512<double> c0 = default, c1 = default, c2 = default, c3 = default, c4 = default, c5 = default, c6 = default, c7 = default;
        Vector512<double> d0 = default, d1 = default, d2 = default, d3 = default, d4 = default, d5 = default, d6 = default, d7 = default;
        int p = 0;
        for (; p + 2 <= kc; p += 2, a += 16, b += 16)
        {
            var a0 = Vector512.Load(a);
            var a1 = Vector512.Load(a + 8);
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
    private static void Add1(double* c, Vector512<double> x) => (Vector512.Load(c) + x).Store(c);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Add2(double* c, Vector512<double> x, Vector512<double> y)
    {
        (Vector512.Load(c) + x).Store(c);
        (Vector512.Load(c + 8) + y).Store(c + 8);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Add3(double* c, Vector512<double> x, Vector512<double> y, Vector512<double> z)
    {
        (Vector512.Load(c) + x).Store(c);
        (Vector512.Load(c + 8) + y).Store(c + 8);
        (Vector512.Load(c + 16) + z).Store(c + 16);
    }

    // AVX2 8×6 マイクロカーネル（12 アキュムレータ）。
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel8x6(int kc, double* a, double* b, double* c, nint ldc)
    {
        Vector256<double> c00 = default, c01 = default, c02 = default, c03 = default, c04 = default, c05 = default;
        Vector256<double> c10 = default, c11 = default, c12 = default, c13 = default, c14 = default, c15 = default;
        for (int p = 0; p < kc; p++, a += 8, b += 6)
        {
            var a0 = Vector256.Load(a);
            var a1 = Vector256.Load(a + 4);
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
    private static void Add256(double* c, Vector256<double> lo, Vector256<double> hi)
    {
        (Vector256.Load(c) + lo).Store(c);
        (Vector256.Load(c + 4) + hi).Store(c + 4);
    }

    // 128 ビット SIMD 用 8×6 マイクロカーネル（Kernel8x6 と同一タイル・同一積和順序）。
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel8x6Narrow(int kc, double* a, double* b, double* c, nint ldc)
    {
        Vector128<double> c00 = default, c01 = default, c02 = default, c03 = default, c04 = default, c05 = default;
        Vector128<double> c10 = default, c11 = default, c12 = default, c13 = default, c14 = default, c15 = default;
        Vector128<double> c20 = default, c21 = default, c22 = default, c23 = default, c24 = default, c25 = default;
        Vector128<double> c30 = default, c31 = default, c32 = default, c33 = default, c34 = default, c35 = default;
        for (int p = 0; p < kc; p++, a += 8, b += 6)
        {
            var a0 = Vector128.Load(a);
            var a1 = Vector128.Load(a + 2);
            var a2 = Vector128.Load(a + 4);
            var a3 = Vector128.Load(a + 6);
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
    private static void Add128(double* c, Vector128<double> v0, Vector128<double> v1, Vector128<double> v2, Vector128<double> v3)
    {
        (Vector128.Load(c) + v0).Store(c);
        (Vector128.Load(c + 2) + v1).Store(c + 2);
        (Vector128.Load(c + 4) + v2).Store(c + 4);
        (Vector128.Load(c + 6) + v3).Store(c + 6);
    }
}
