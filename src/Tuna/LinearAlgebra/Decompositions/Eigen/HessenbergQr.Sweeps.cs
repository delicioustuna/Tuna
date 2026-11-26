using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace Tuna.LinearAlgebra;

internal static partial class HessenbergQr
{
    // LAPACK dlaqr5 の簡略版
    // バルジは1つずつ追跡する
    /// <summary>
    /// 積極的早期収縮で得た固有値をシフトにして、二重シフトの掃引を繰り返す
    /// </summary>
    /// <returns>
    /// 掃引したら真。使えるシフトがなければ偽。
    /// </returns>
    private static bool MultishiftSweeps(
        Span<double> h, int ld, int l, int n, ReadOnlySpan<double> shifts, int available, int wanted,
        int rowEnd, int colStart, Span<double> z, int low, int high, bool vectors)
    {
        int nwOffset = shifts.Length / 2;
        int count = Math.Min(wanted, available);
        int start = available - count;
        if (start > 0 && start < available && shifts[nwOffset + start] < 0)
        {
            start++;
        }

        if (available - start < 2)
        {
            return false;
        }

        // シフトを2つずつ組にする（共役複素数の組はそのまま、実数は2つずつ）
        Span<double> sums = stackalloc double[available];
        Span<double> products = stackalloc double[available];
        int pairs = 0;
        double pendingReal = double.NaN;
        for (int i = start; i < available; i++)
        {
            double re = shifts[i];
            double im = shifts[nwOffset + i];
            if (im > 0 && i + 1 < available)
            {
                sums[pairs] = 2 * re;
                products[pairs] = re * re + im * im;
                pairs++;
                i++;
            }
            else if (double.IsNaN(pendingReal))
            {
                pendingReal = re;
            }
            else
            {
                sums[pairs] = pendingReal + re;
                products[pairs] = pendingReal * re;
                pairs++;
                pendingReal = double.NaN;
            }
        }

        if (!double.IsNaN(pendingReal))
        {
            sums[pairs] = 2 * pendingReal;
            products[pairs] = pendingReal * pendingReal;
            pairs++;
        }

        if (pairs >= 2 && n - l + 1 >= 3 * pairs + 6)
        {
            ChainSweep(h, ld, l, n, sums[..pairs], products[..pairs], rowEnd, colStart, z, low, high, vectors);
            return true;
        }

        for (int k = 0; k < pairs; k++)
        {
            double h00 = h[l + l * ld], h10 = h[l + 1 + l * ld], h01 = h[l + (l + 1) * ld], h11 = h[l + 1 + (l + 1) * ld];
            double h21 = h[l + 2 + (l + 1) * ld];
            double p = h00 * h00 + h01 * h10 - sums[k] * h00 + products[k];
            double q = h10 * (h00 + h11 - sums[k]);
            double r = h10 * h21;
            double scale = Math.Abs(p) + Math.Abs(q) + Math.Abs(r);
            if (scale == 0)
            {
                break;
            }

            p /= scale;
            q /= scale;
            r /= scale;
            for (int i = l + 2; i <= n; i++)
            {
                h[i + (i - 2) * ld] = 0.0;
                if (i > l + 2)
                {
                    h[i + (i - 3) * ld] = 0.0;
                }
            }

            Sweep(h, ld, l, l, n, p, q, r, rowEnd, colStart, z, low, high, vectors);
        }

        return true;
    }

    // 窓の中の追跡は逐次的にしか進められないので、並列化するときは先読み（lookahead）を行う。
    // 前の窓の U を窓の外へ適用する行列積のうち、次の窓に入る列だけを先に済ませ、
    // 残り（さらに右の列・上の行・Z）は次の窓の追跡と同時に他のスレッドで計算する。
    // 2つの処理が触る範囲は重ならない（窓は右下へ進むだけで、追跡は窓の中の行・列しか読み書きしない）
    // LAPACK dlaqr5 の考え方
    // 窓の中では変換をすぐに適用しながら直交行列 U に蓄積し、窓の外の行・列と Z には窓ごとに行列積でまとめて適用する。
    /// <summary>
    /// 複数のバルジを3行間隔で連ねて同時に追跡する掃引
    /// </summary>
    private static unsafe void ChainSweep(
        Span<double> h, int ld, int l, int n, ReadOnlySpan<double> sums, ReadOnlySpan<double> products,
        int rowEnd, int colStart, Span<double> z, int low, int high, bool vectors)
    {
        int bulges = sums.Length;
        int stepsPerChunk = 3 * bulges;
        int maxWindow = 3 * (bulges - 1) + stepsPerChunk + 5;
        var pos = new int[bulges];
        var dead = new bool[bulges];
        var uLo = new int[maxWindow];
        var uHi = new int[maxWindow];
        double[] shiftSums = sums.ToArray();
        double[] shiftProducts = products.ToArray();
        for (int j = 0; j < bulges; j++)
        {
            pos[j] = l - 3 * j;
        }

        // 既存の掃引の名残（副対角より2つ以上下）を消す
        for (int i = l + 2; i <= n; i++)
        {
            h[i + (i - 2) * ld] = 0.0;
            if (i > l + 2)
            {
                h[i + (i - 3) * ld] = 0.0;
            }
        }

        int zRows = vectors ? high - low + 1 : 0;
        int threads = Parallelism.Degree((double)maxWindow * maxWindow * (rowEnd - colStart + zRows), 4 * Parallelism.Grain);
        double[] uBuffer = WorkArrayPool<double>.Rent(checked(2 * maxWindow * maxWindow));
        try
        {
            fixed (double* ph = h)
            fixed (double* pz = z)
            {
                var sh = new SharedSpan(ph, h.Length);
                var sz = new SharedSpan(pz, z.Length);
                int cur = 0;
                int p0 = 0, p1 = -1, pw = 0;
                var pieces = new List<(int Kind, int Start, int Length)>();
                while (pos[bulges - 1] <= n - 1)
                {
                    // この区間で各バルジが進む範囲から窓 [w0, w1] を決める
                    int trailing = Math.Max(pos[bulges - 1], l);
                    int w0 = Math.Max(l, trailing - 1);
                    int lead = Math.Min(pos[0] + stepsPerChunk - 1, n - 1);
                    int w1 = Math.Min(n, lead + 3);
                    int w = w1 - w0 + 1;
                    int uOffset = cur * maxWindow * maxWindow;
                    var u = uBuffer.AsSpan(uOffset, w * w);
                    u.Clear();
                    for (int i = 0; i < w; i++)
                    {
                        u[i + i * w] = 1;
                        uLo[i] = i;
                        uHi[i] = i;
                    }

                    if (p1 < 0)
                    {
                        ChaseChunk(h, ld, l, n, w0, w1, stepsPerChunk, pos, dead, shiftSums, shiftProducts, u, uLo, uHi, rowEnd, colStart);
                    }
                    else
                    {
                        // 前の窓の U を、次の窓に入る列へ先に適用する
                        var up = uBuffer.AsSpan((1 - cur) * maxWindow * maxWindow, pw * pw);
                        int entering = Math.Min(w1, rowEnd - 1);
                        if (entering > p1)
                        {
                            ApplyPiece(h, z, ld, up, pw, p0, p1, low, colStart, (0, p1 + 1, entering - p1));
                        }

                        // 残りの窓の外への適用と、次の窓の追跡を同時に行う
                        MakePieces(pieces, pw, p0, Math.Max(p1, entering) + 1, rowEnd, colStart, low, zRows);
                        int prevOffset = (1 - cur) * maxWindow * maxWindow;
                        int q0 = p0, q1 = p1, qw = pw;
                        Parallelism.For(pieces.Count + 1, threads, t =>
                        {
                            if (t == 0)
                            {
                                ChaseChunk(
                                    sh.Span, ld, l, n, w0, w1, stepsPerChunk, pos, dead, shiftSums, shiftProducts,
                                    uBuffer.AsSpan(uOffset, w * w), uLo, uHi, rowEnd, colStart);
                            }
                            else
                            {
                                ApplyPiece(sh.Span, sz.Span, ld, uBuffer.AsSpan(prevOffset, qw * qw), qw, q0, q1, low, colStart, pieces[t - 1]);
                            }
                        });
                    }

                    p0 = w0;
                    p1 = w1;
                    pw = w;
                    cur = 1 - cur;
                }

                // 最後の窓の U を窓の外へ適用する
                if (p1 >= 0)
                {
                    MakePieces(pieces, pw, p0, p1 + 1, rowEnd, colStart, low, zRows);
                    int lastOffset = (1 - cur) * maxWindow * maxWindow;
                    int q0 = p0, q1 = p1, qw = pw;
                    Parallelism.For(pieces.Count, threads, t =>
                        ApplyPiece(sh.Span, sz.Span, ld, uBuffer.AsSpan(lastOffset, qw * qw), qw, q0, q1, low, colStart, pieces[t]));
                }
            }
        }
        finally
        {
            WorkArrayPool<double>.Return(uBuffer);
        }
    }

    // 変換は窓の中の行・列にだけ適用し、U（w × w）に蓄積する。
    /// <summary>
    /// 窓 <c>[w0, w1]</c> の中で各バルジを <paramref name="steps"/> 段進める
    /// </summary>
    private static void ChaseChunk(
        Span<double> h, int ld, int l, int n, int w0, int w1, int steps, int[] pos, bool[] dead,
        double[] sums, double[] products, Span<double> u, int[] uLo, int[] uHi, int rowEnd, int colStart)
    {
        int bulges = pos.Length;
        int w = w1 - w0 + 1;
        int rowTop = Math.Max(colStart, w0);
        for (int step = 0; step < steps; step++)
        {
            for (int j = 0; j < bulges; j++)
            {
                int k = pos[j]++;
                if (k < l || k > n - 1 || dead[j])
                {
                    continue;
                }

                bool notlast = k != n - 1;
                double p, q, r, x = 0;
                if (k == l)
                {
                    double h00 = h[l + l * ld], h10 = h[l + 1 + l * ld], h01 = h[l + (l + 1) * ld], h11 = h[l + 1 + (l + 1) * ld];
                    double h21 = h[l + 2 + (l + 1) * ld];
                    p = h00 * h00 + h01 * h10 - sums[j] * h00 + products[j];
                    q = h10 * (h00 + h11 - sums[j]);
                    r = h10 * h21;
                    double sc = Math.Abs(p) + Math.Abs(q) + Math.Abs(r);
                    if (sc == 0)
                    {
                        dead[j] = true;
                        continue;
                    }

                    p /= sc;
                    q /= sc;
                    r /= sc;
                }
                else
                {
                    p = h[k + (k - 1) * ld];
                    q = h[k + 1 + (k - 1) * ld];
                    r = notlast ? h[k + 2 + (k - 1) * ld] : 0.0;
                    x = Math.Abs(p) + Math.Abs(q) + Math.Abs(r);
                    if (x == 0)
                    {
                        continue;
                    }

                    p /= x;
                    q /= x;
                    r /= x;
                }

                double s = Math.Sqrt(p * p + q * q + r * r);
                if (p < 0)
                {
                    s = -s;
                }

                if (s == 0)
                {
                    continue;
                }

                if (k != l)
                {
                    h[k + (k - 1) * ld] = -s * x;
                    h[k + 1 + (k - 1) * ld] = 0;
                    if (notlast)
                    {
                        h[k + 2 + (k - 1) * ld] = 0;
                    }
                }

                p += s;
                double xx = p / s, yy = q / s, zz = r / s, qq = q / p, rr = r / p;

                // 行の変換（窓の中の列 k..w1）
                int jEnd = Math.Min(w1, rowEnd - 1);
                if (notlast)
                {
                    for (int c = k; c <= jEnd; c++)
                    {
                        int o = k + c * ld;
                        double h0 = h[o], h1 = h[o + 1], h2 = h[o + 2];
                        double t = h0 + qq * h1 + rr * h2;
                        h[o] = h0 - t * xx;
                        h[o + 1] = h1 - t * yy;
                        h[o + 2] = h2 - t * zz;
                    }
                }
                else
                {
                    for (int c = k; c <= jEnd; c++)
                    {
                        int o = k + c * ld;
                        double h0 = h[o], h1 = h[o + 1];
                        double t = h0 + qq * h1;
                        h[o] = h0 - t * xx;
                        h[o + 1] = h1 - t * yy;
                    }
                }

                // 列の変換（窓の中の行 rowTop..k+3）と U への蓄積
                int iEnd = Math.Min(n, k + 3);
                ApplyColumns(h.Slice(rowTop + k * ld), ld, iEnd - rowTop + 1, notlast, xx, yy, zz, qq, rr);

                // U の各列で非零になりうる行の範囲だけを更新する
                int uc = k - w0;
                int width = notlast ? 3 : 2;
                int lo = uLo[uc], hiRow = uHi[uc];
                for (int c = 1; c < width; c++)
                {
                    lo = Math.Min(lo, uLo[uc + c]);
                    hiRow = Math.Max(hiRow, uHi[uc + c]);
                }

                for (int c = 0; c < width; c++)
                {
                    uLo[uc + c] = lo;
                    uHi[uc + c] = hiRow;
                }

                ApplyColumns(u.Slice(lo + uc * w), w, hiRow - lo + 1, notlast, xx, yy, zz, qq, rr);
            }
        }
    }

    // 断片は (種類, 開始, 長さ) で、種類 0 は右の列 [rightStart, rowEnd)、1 は上の行 [colStart, w0)、2 は Z の行。
    /// <summary>
    /// 窓 <c>[w0, w1]</c> の <c>U</c> を窓の外へ適用する処理を、ほぼ同じ演算量の断片に分ける
    /// </summary>
    private static void MakePieces(
        List<(int Kind, int Start, int Length)> pieces, int w, int w0, int rightStart, int rowEnd, int colStart, int low, int zRows)
    {
        pieces.Clear();
        int right = rowEnd - rightStart;
        int above = w0 - colStart;
        long total = (long)Math.Max(right, 0) + Math.Max(above, 0) + zRows;
        if (total == 0)
        {
            return;
        }

        // 行列積の形を並列数によらず固定する。1断片を128列（行）以下に収める。
        const int size = 128;
        void Add(int kind, int start, int length)
        {
            for (int s = 0; s < length; s += size)
            {
                pieces.Add((kind, start + s, Math.Min(size, length - s)));
            }
        }

        Add(2, low, zRows);
        Add(1, colStart, Math.Max(above, 0));
        Add(0, rightStart, Math.Max(right, 0));
    }

    /// <summary>
    /// <see cref="MakePieces"/> の断片を1つ適用する（右の列は <c>H ← UᵀH</c>、上の行と <c>Z</c> は <c>X ← XU</c>）
    /// </summary>
    private static void ApplyPiece(
        Span<double> h, Span<double> z, int ld, ReadOnlySpan<double> u, int w, int w0, int w1, int low, int colStart,
        (int Kind, int Start, int Length) piece)
    {
        int len = piece.Length;
        if (len <= 0)
        {
            return;
        }

        double[] rented = WorkArrayPool<double>.Rent(checked(w * len));
        var tmp = rented.AsSpan(0, w * len);
        tmp.Clear();
        if (piece.Kind == 0)
        {
            var target = h[(w0 + piece.Start * ld)..];
            Gemm.Multiply(true, false, w, len, w, 1.0, u, w, target, ld, tmp, w);
            Level1Kernels.Copy(w, len, tmp, w, target, ld);
        }
        else
        {
            var target = (piece.Kind == 1 ? h : z)[(piece.Start + w0 * ld)..];
            Gemm.Multiply(false, false, len, w, w, 1.0, target, ld, u, w, tmp, len);
            Level1Kernels.Copy(len, w, tmp, len, target, ld);
        }

        WorkArrayPool<double>.Return(rented);
    }

    /// <summary>
    /// 連続した3列（または2列）に列の変換 <c>t = x·c₀ + y·c₁ + z·c₂, (c₀, c₁, c₂) -= t·(1, q, r)</c> を行う
    /// </summary>
    private static void ApplyColumns(Span<double> c, int ld, int rows, bool three, double xx, double yy, double zz, double qq, double rr)
    {
        var c0 = c[..rows];
        var c1 = c.Slice(ld, rows);
        if (three)
        {
            var c2 = c.Slice(2 * ld, rows);
            int i = 0;
            if (Vector256.IsHardwareAccelerated && rows >= 4)
            {
                ref double p0 = ref MemoryMarshal.GetReference(c0);
                ref double p1 = ref MemoryMarshal.GetReference(c1);
                ref double p2 = ref MemoryMarshal.GetReference(c2);
                var vx = Vector256.Create(xx);
                var vy = Vector256.Create(yy);
                var vz = Vector256.Create(zz);
                var vq = Vector256.Create(qq);
                var vr = Vector256.Create(rr);
                for (; i <= rows - 4; i += 4)
                {
                    var a0 = Vector256.LoadUnsafe(ref p0, (nuint)i);
                    var a1 = Vector256.LoadUnsafe(ref p1, (nuint)i);
                    var a2 = Vector256.LoadUnsafe(ref p2, (nuint)i);
                    var t = Vector256.MultiplyAddEstimate(vz, a2, Vector256.MultiplyAddEstimate(vy, a1, vx * a0));
                    (a0 - t).StoreUnsafe(ref p0, (nuint)i);
                    Vector256.MultiplyAddEstimate(-t, vq, a1).StoreUnsafe(ref p1, (nuint)i);
                    Vector256.MultiplyAddEstimate(-t, vr, a2).StoreUnsafe(ref p2, (nuint)i);
                }
            }

            for (; i < rows; i++)
            {
                double t = xx * c0[i] + yy * c1[i] + zz * c2[i];
                c0[i] -= t;
                c1[i] -= t * qq;
                c2[i] -= t * rr;
            }
        }
        else
        {
            for (int i = 0; i < rows; i++)
            {
                double t = xx * c0[i] + yy * c1[i];
                c0[i] -= t;
                c1[i] -= t * qq;
            }
        }
    }

    // 窓の外側への作用はまとめて遅延適用する。
    /// <summary>
    /// 一度に追跡するステップ数（窓の幅）
    /// </summary>
    private const int Window = 48;

    /// <summary>
    /// 1ステップの 3要素（最後は2要素）反射 <c>I - v·wᵀ</c> の係数
    /// </summary>
    private struct Reflector
    {
        public double X;
        public double Y;
        public double Z;
        public double Q;
        public double R;
        public bool Active;
        public bool Three;
    }

    // 窓（Window ステップ）ごとに対角付近だけを即時に更新し、窓より右の列・上の行・Z への作用は 窓の終わりにまとめて適用する。
    // 適用はベクトル化した「3列のずらし窓」で行い、各要素を窓ごとに1回だけ読み書きする。
    /// <summary>
    /// 行 l:n、列 m:n に対する二重シフト QR ステップ（バルジの追跡）
    /// </summary>
    private static void Sweep(
        Span<double> h, int ld, int l, int m, int n, double p, double q, double r,
        int rowEnd, int colStart, Span<double> z, int low, int high, bool vectors)
    {
        Span<Reflector> steps = stackalloc Reflector[Window];
        for (int kStart = m; kStart <= n - 1; kStart += Window)
        {
            int kEnd = Math.Min(n - 1, kStart + Window - 1);
            int hi = Math.Min(n, kEnd + 2);
            int count = kEnd - kStart + 1;
            for (int k = kStart; k <= kEnd; k++)
            {
                ref var st = ref steps[k - kStart];
                st.Active = false;
                bool notlast = k != n - 1;
                st.Three = notlast;
                double x = 0;
                if (k != m)
                {
                    p = h[k + (k - 1) * ld];
                    q = h[(k + 1) + (k - 1) * ld];
                    r = notlast ? h[(k + 2) + (k - 1) * ld] : 0.0;
                    x = Math.Abs(p) + Math.Abs(q) + Math.Abs(r);
                    if (x == 0.0)
                    {
                        continue;
                    }

                    p /= x;
                    q /= x;
                    r /= x;
                }

                double s = Math.Sqrt(p * p + q * q + r * r);
                if (p < 0)
                {
                    s = -s;
                }

                if (s == 0)
                {
                    continue;
                }

                if (k != m)
                {
                    h[k + (k - 1) * ld] = -s * x;
                }
                else if (l != m)
                {
                    h[k + (k - 1) * ld] = -h[k + (k - 1) * ld];
                }

                p += s;
                double xx = p / s;
                double yy = q / s;
                double zz = r / s;
                double qq = q / p;
                double rr = r / p;
                st.X = xx;
                st.Y = yy;
                st.Z = zz;
                st.Q = qq;
                st.R = rr;
                st.Active = true;

                // 窓の中の行の変換（列 k..hi）
                int jEnd = Math.Min(hi, rowEnd - 1);
                if (notlast)
                {
                    for (int j = k; j <= jEnd; j++)
                    {
                        int o = k + j * ld;
                        double h0 = h[o];
                        double h1 = h[o + 1];
                        double h2 = h[o + 2];
                        double t = h0 + qq * h1 + rr * h2;
                        h[o] = h0 - t * xx;
                        h[o + 1] = h1 - t * yy;
                        h[o + 2] = h2 - t * zz;
                    }
                }
                else
                {
                    for (int j = k; j <= jEnd; j++)
                    {
                        int o = k + j * ld;
                        double h0 = h[o];
                        double h1 = h[o + 1];
                        double t = h0 + qq * h1;
                        h[o] = h0 - t * xx;
                        h[o + 1] = h1 - t * yy;
                    }
                }

                // 窓の中の列の変換（行 kStart..k+3）
                int iStart = Math.Max(colStart, kStart);
                int iEnd = Math.Min(n, k + 3);
                int c0 = k * ld;
                int c1 = c0 + ld;
                if (notlast)
                {
                    int c2 = c1 + ld;
                    for (int i = iStart; i <= iEnd; i++)
                    {
                        double t = xx * h[i + c0] + yy * h[i + c1] + zz * h[i + c2];
                        h[i + c0] -= t;
                        h[i + c1] -= t * qq;
                        h[i + c2] -= t * rr;
                    }
                }
                else
                {
                    for (int i = iStart; i <= iEnd; i++)
                    {
                        double t = xx * h[i + c0] + yy * h[i + c1];
                        h[i + c0] -= t;
                        h[i + c1] -= t * qq;
                    }
                }
            }

            var active = steps[..count];

            // 窓より右の列への行の変換（行 kStart..hi を転置して列方向にベクトル化）
            if (rowEnd - 1 > hi)
            {
                ApplyRowsDelayed(h, ld, kStart, hi, hi + 1, rowEnd - 1, active);
            }

            // 窓より上の行への列の変換
            if (kStart > colStart)
            {
                ApplySequence(h[(colStart + kStart * ld)..], ld, kStart - colStart, hi - kStart + 1, active, rowRole: false);
            }

            // 変換の蓄積
            if (vectors)
            {
                ApplySequence(z[(low + kStart * ld)..], ld, high - low + 1, hi - kStart + 1, active, rowRole: false);
            }
        }
    }

    /// <summary>
    /// 行 <c>r0..r1</c>、列 <c>j0..j1</c> に行の変換列を適用する（16列ずつ転置してベクトル化）
    /// </summary>
    private static void ApplyRowsDelayed(Span<double> h, int ld, int r0, int r1, int j0, int j1, ReadOnlySpan<Reflector> steps)
    {
        const int Chunk = 16;
        int len = r1 - r0 + 1;
        Span<double> buffer = stackalloc double[Chunk * (Window + 3)];
        for (int jc = j0; jc <= j1; jc += Chunk)
        {
            int width = Math.Min(Chunk, j1 - jc + 1);
            for (int c = 0; c < width; c++)
            {
                var col = h.Slice(r0 + (jc + c) * ld, len);
                for (int t = 0; t < len; t++)
                {
                    buffer[t * Chunk + c] = col[t];
                }
            }

            ApplySequence(buffer, Chunk, width, len, steps, rowRole: true);

            for (int c = 0; c < width; c++)
            {
                var col = h.Slice(r0 + (jc + c) * ld, len);
                for (int t = 0; t < len; t++)
                {
                    col[t] = buffer[t * Chunk + c];
                }
            }
        }
    }

    // 列の変換は p = x·c₀ + y·c₁ + z·c₂, (c₀, c₁, c₂) -= p·(1, q, r)、 行の変換（転置した領域）は p = c₀ + q·c₁ + r·c₂, (c₀, c₁, c₂) -= p·(x, y, z)。
    /// <summary>
    /// 連続した <paramref name="rows"/> 行を持つ列 <c>0..columns-1</c>（列の間隔 <paramref name="ld"/>）に、 ステップ <c>t</c> の反射を列 <c>t, t+1, t+2</c> へ順に作用させる
    /// </summary>
    private static void ApplySequence(Span<double> data, int ld, int rows, int columns, ReadOnlySpan<Reflector> steps, bool rowRole)
    {
        int count = steps.Length;
        ref double d0 = ref MemoryMarshal.GetReference(data);
        int i = 0;
        if (Vector512.IsHardwareAccelerated)
        {
            for (; i <= rows - 8; i += 8)
            {
                ref double baseRef = ref Unsafe.Add(ref d0, i);
                var v0 = Vector512.LoadUnsafe(ref baseRef);
                var v1 = Vector512.LoadUnsafe(ref baseRef, (nuint)ld);
                var v2 = columns > 2 ? Vector512.LoadUnsafe(ref baseRef, (nuint)(2 * ld)) : Vector512<double>.Zero;
                for (int t = 0; t < count; t++)
                {
                    ref readonly var st = ref steps[t];
                    if (st.Active)
                    {
                        double a0, a1, a2, b0, b1, b2;
                        if (rowRole)
                        {
                            (a0, a1, a2, b0, b1, b2) = (1, st.Q, st.R, st.X, st.Y, st.Z);
                        }
                        else
                        {
                            (a0, a1, a2, b0, b1, b2) = (st.X, st.Y, st.Z, 1, st.Q, st.R);
                        }

                        if (st.Three)
                        {
                            var pv = Vector512.FusedMultiplyAdd(Vector512.Create(a2), v2, Vector512.FusedMultiplyAdd(Vector512.Create(a1), v1, Vector512.Create(a0) * v0));
                            v0 -= pv * Vector512.Create(b0);
                            v1 -= pv * Vector512.Create(b1);
                            v2 -= pv * Vector512.Create(b2);
                        }
                        else
                        {
                            var pv = Vector512.FusedMultiplyAdd(Vector512.Create(a1), v1, Vector512.Create(a0) * v0);
                            v0 -= pv * Vector512.Create(b0);
                            v1 -= pv * Vector512.Create(b1);
                        }
                    }

                    v0.StoreUnsafe(ref baseRef, (nuint)(t * ld));
                    v0 = v1;
                    v1 = v2;
                    v2 = t + 3 < columns ? Vector512.LoadUnsafe(ref baseRef, (nuint)((t + 3) * ld)) : Vector512<double>.Zero;
                }

                v0.StoreUnsafe(ref baseRef, (nuint)(count * ld));
                if (count + 1 < columns)
                {
                    v1.StoreUnsafe(ref baseRef, (nuint)((count + 1) * ld));
                }
            }
        }

        if (Vector256.IsHardwareAccelerated)
        {
            for (; i <= rows - 4; i += 4)
            {
                ref double baseRef = ref Unsafe.Add(ref d0, i);
                var v0 = Vector256.LoadUnsafe(ref baseRef);
                var v1 = Vector256.LoadUnsafe(ref baseRef, (nuint)ld);
                var v2 = columns > 2 ? Vector256.LoadUnsafe(ref baseRef, (nuint)(2 * ld)) : Vector256<double>.Zero;
                for (int t = 0; t < count; t++)
                {
                    ref readonly var st = ref steps[t];
                    if (st.Active)
                    {
                        double a0, a1, a2, b0, b1, b2;
                        if (rowRole)
                        {
                            (a0, a1, a2, b0, b1, b2) = (1, st.Q, st.R, st.X, st.Y, st.Z);
                        }
                        else
                        {
                            (a0, a1, a2, b0, b1, b2) = (st.X, st.Y, st.Z, 1, st.Q, st.R);
                        }

                        if (st.Three)
                        {
                            var pv = Vector256.Create(a0) * v0 + Vector256.Create(a1) * v1 + Vector256.Create(a2) * v2;
                            v0 -= pv * Vector256.Create(b0);
                            v1 -= pv * Vector256.Create(b1);
                            v2 -= pv * Vector256.Create(b2);
                        }
                        else
                        {
                            var pv = Vector256.Create(a0) * v0 + Vector256.Create(a1) * v1;
                            v0 -= pv * Vector256.Create(b0);
                            v1 -= pv * Vector256.Create(b1);
                        }
                    }

                    v0.StoreUnsafe(ref baseRef, (nuint)(t * ld));
                    v0 = v1;
                    v1 = v2;
                    v2 = t + 3 < columns ? Vector256.LoadUnsafe(ref baseRef, (nuint)((t + 3) * ld)) : Vector256<double>.Zero;
                }

                v0.StoreUnsafe(ref baseRef, (nuint)(count * ld));
                if (count + 1 < columns)
                {
                    v1.StoreUnsafe(ref baseRef, (nuint)((count + 1) * ld));
                }
            }
        }

        for (; i < rows; i++)
        {
            ref double baseRef = ref Unsafe.Add(ref d0, i);
            double v0 = baseRef;
            double v1 = Unsafe.Add(ref baseRef, ld);
            double v2 = columns > 2 ? Unsafe.Add(ref baseRef, 2 * ld) : 0;
            for (int t = 0; t < count; t++)
            {
                ref readonly var st = ref steps[t];
                if (st.Active)
                {
                    if (rowRole)
                    {
                        double pv = st.Three ? v0 + st.Q * v1 + st.R * v2 : v0 + st.Q * v1;
                        v0 -= pv * st.X;
                        v1 -= pv * st.Y;
                        if (st.Three)
                        {
                            v2 -= pv * st.Z;
                        }
                    }
                    else
                    {
                        double pv = st.Three ? st.X * v0 + st.Y * v1 + st.Z * v2 : st.X * v0 + st.Y * v1;
                        v0 -= pv;
                        v1 -= pv * st.Q;
                        if (st.Three)
                        {
                            v2 -= pv * st.R;
                        }
                    }
                }

                Unsafe.Add(ref baseRef, t * ld) = v0;
                v0 = v1;
                v1 = v2;
                v2 = t + 3 < columns ? Unsafe.Add(ref baseRef, (t + 3) * ld) : 0;
            }

            Unsafe.Add(ref baseRef, count * ld) = v0;
            if (count + 1 < columns)
            {
                Unsafe.Add(ref baseRef, (count + 1) * ld) = v1;
            }
        }
    }
}
