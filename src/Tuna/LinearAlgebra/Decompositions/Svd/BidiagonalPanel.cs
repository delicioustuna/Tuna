namespace Tuna.LinearAlgebra;

// 二重対角化の並列パネル分解（LAPACK dlabrd 相当）。owner-computes 列分担と Rent/Return によるバッファ再利用を行う。
internal sealed unsafe class BidiagonalPanel
{
    [ThreadStatic]
    private static BidiagonalPanel? cached;

    private readonly int nbMax;
    private readonly int maxM;
    private readonly int maxN;
    private readonly int maxParts;
    private ColumnTeam team;
    private readonly double* z;
    private readonly double* w;
    private readonly double* sPartial;
    private readonly double* t1;
    private readonly double* xr;
    private readonly double* rb;
    private readonly Action<int> region1;
    private readonly Action<int> region2;

    // 現在のパネル
    private double* a;
    private int lda;
    private int m;
    private double* y;
    private int ldy;
    private int i;
    private double tauqI;

    // 固定した作業領域（パネルの間で使い回す）
    private readonly double[] scratch;

    internal BidiagonalPanel(ColumnTeam team, int maxM, int maxN, int nb)
        : this(team, maxM, maxN, nb, team.Parts)
    {
    }

    private BidiagonalPanel(ColumnTeam team, int maxM, int maxN, int nb, int parts)
    {
        this.team = team;
        nbMax = nb;
        this.maxM = maxM;
        this.maxN = maxN;
        maxParts = parts;

        // Σ A[0:m, c] v_c の部分和は ColumnTeam の領域に作る（team は maxM 行で作ったもの）
        scratch = GC.AllocateUninitializedArray<double>(checked(2 * maxM + parts * nb + 2 * nb + maxN), pinned: true);
        fixed (double* p = scratch)
        {
            // 固定ヒープ（POH）の配列なので、ポインタは配列の寿命の間有効
            z = p;
            w = z + maxM;
            sPartial = w + maxM;
            t1 = sPartial + parts * nb;
            xr = t1 + nb;
            rb = xr + nb;
        }

        region1 = Region1;
        region2 = Region2;
    }

    // 入れ子で使うと内側は新しく作る。
    /// <summary>
    /// <see cref="BidiagonalPanel(ColumnTeam, int, int, int)"/> と同じものを、このスレッドで前に返したものを使い回して用意する
    /// </summary>
    internal static BidiagonalPanel Rent(ColumnTeam team, int maxM, int maxN, int nb)
    {
        var panel = cached;
        cached = null;
        if (panel is not null && panel.nbMax == nb && panel.maxM >= maxM && panel.maxN >= maxN && panel.maxParts >= team.Parts)
        {
            panel.team = team;
            return panel;
        }

        return panel is not null && panel.nbMax == nb
            ? new BidiagonalPanel(team, Math.Max(maxM, panel.maxM), Math.Max(maxN, panel.maxN), nb, Math.Max(team.Parts, panel.maxParts))
            : new BidiagonalPanel(team, maxM, maxN, nb);
    }

    /// <summary>
    /// <see cref="Rent"/> で用意したものを返す（以後使わないこと）
    /// </summary>
    internal static void Return(BidiagonalPanel panel)
    {
        // 返したパネルが ColumnTeam を参照し続けると、ColumnTeam の使い回しを大きく作り直した後も古いチームの固定配列が残るので、参照を外す（次の Rent で設定する）。
        panel.team = null!;
        cached = panel;
    }

    // team の分担は呼び出し元で ColumnTeam.Partition 済みであること。
    // dlabrd と同じ結果。
    /// <summary>
    /// <c>m × n</c> の先頭 <paramref name="nb"/> 行・列を縮約し、<c>X</c>（<c>m × nb</c>）と <c>Y</c>（<c>n × nb</c>）を求める
    /// </summary>
    internal void Reduce(
        int m, int n, int nb, double* a, int lda, Span<double> d, Span<double> e, Span<double> tauq, Span<double> taup,
        double* x, int ldx, double* y, int ldy)
    {
        this.a = a;
        this.lda = lda;
        this.m = m;
        this.y = y;
        this.ldy = ldy;
        int parts = team.Parts;
        for (int i = 0; i < nb; i++)
        {
            // A[i:m, i] ← A - A[i:m, 0:i] Y[i, 0:i]ᵀ - X[i:m, 0:i] A[0:i, i]（パネルの列は呼び出し元の担当）
            int mr = m - i;
            double* col = a + i + (nint)i * lda;
            var colSpan = new Span<double>(col, mr);
            for (int j = 0; j < i; j++)
            {
                Level1Kernels.Axpy2(
                    -y[i + j * ldy], new ReadOnlySpan<double>(a + i + (nint)j * lda, mr),
                    -a[j + (nint)i * lda], new ReadOnlySpan<double>(x + i + (nint)j * ldx, mr), colSpan);
            }

            double alpha = col[0];
            tauq[i] = Householder.Generate(ref alpha, colSpan[1..]);
            d[i] = alpha;
            if (i >= n - 1)
            {
                taup[i] = 0;
                continue;
            }

            col[0] = 1;

            // z = [-X[i:m, 0:i]ᵀu; u]、t1 = A[i:m, 0:i]ᵀu、xr = X[i, 0:i]
            for (int j = 0; j < i; j++)
            {
                z[j] = -Level1Kernels.Dot(new ReadOnlySpan<double>(x + i + (nint)j * ldx, mr), colSpan);
                t1[j] = Level1Kernels.Dot(new ReadOnlySpan<double>(a + i + (nint)j * lda, mr), colSpan);
                xr[j] = x[i + (nint)j * ldx];
            }

            colSpan.CopyTo(new Span<double>(z + i, mr));
            this.i = i;
            tauqI = tauq[i];
            team.For(region1);

            // 右の反射 P(i) で A[i, i+2:n] を消去
            int nr = n - i - 1;
            double beta = rb[0];
            taup[i] = Householder.Generate(ref beta, new Span<double>(rb + 1, nr - 1));
            e[i] = beta;
            rb[0] = 1;
            team.For(region2);

            // 部分和を区間の順に合計する
            team.AddPartials(w, m, i + 1, int.MaxValue, lower: false);
            var wSpan = new Span<double>(w, m);
            var s = new Span<double>(sPartial, i + 1);
            for (int t = 1; t < parts; t++)
            {
                Level1Kernels.Axpy(1.0, new ReadOnlySpan<double>(sPartial + t * nbMax, i + 1), s);
            }

            // X[i+1:m, i] = τp (A[i+1:m, i+1:n]v - A[i+1:m, 0:i+1] (Y[i+1:n, 0:i+1]ᵀv) - X[i+1:m, 0:i] (A[0:i, i+1:n]v))
            int mx = m - i - 1;
            var xi = new Span<double>(x + i + 1 + (nint)i * ldx, mx);
            wSpan.Slice(i + 1, mx).CopyTo(xi);
            for (int j = 0; j <= i; j++)
            {
                Level1Kernels.Axpy(-s[j], new ReadOnlySpan<double>(a + i + 1 + (nint)j * lda, mx), xi);
            }

            for (int j = 0; j < i; j++)
            {
                Level1Kernels.Axpy(-w[j], new ReadOnlySpan<double>(x + i + 1 + (nint)j * ldx, mx), xi);
            }

            Level1Kernels.Scale(taup[i], xi);
        }
    }

    /// <summary>
    /// 担当の列 c について <c>Y[c, i]</c> と、更新した行 i の値 <c>rb[c - i - 1]</c> を求める
    /// </summary>
    private void Region1(int t)
    {
        var (lo, hi) = team.Range(t, i + 1);
        if (hi <= lo)
        {
            return;
        }

        int cnt = hi - lo;
        double* cols = a + (nint)lo * lda;
        var yi = new Span<double>(y + lo + (nint)i * ldy, cnt);

        // Y[c, i] = τq (A[0:m, c]ᵀ z - Y[c, 0:i] t1) = τq (A[i:m, c]ᵀu - A[0:i, c]ᵀ(Xᵀu) - Y[c, 0:i] t1)
        MatrixVector.MultiplyTransposedSequential(
            m, cnt, new ReadOnlySpan<double>(cols, (cnt - 1) * lda + m), lda, new ReadOnlySpan<double>(z, m), yi);
        for (int j = 0; j < i; j++)
        {
            Level1Kernels.Axpy(-t1[j], new ReadOnlySpan<double>(y + lo + (nint)j * ldy, cnt), yi);
        }

        Level1Kernels.Scale(tauqI, yi);

        // rb[c] = A[i, c] - Σ_{j≤i} A[i, j] Y[c, j] - A[0:i, c]ᵀ X[i, 0:i]（A[i, i] は反射の先頭の1）
        var r = new Span<double>(rb + (lo - i - 1), cnt);
        for (int c = 0; c < cnt; c++)
        {
            r[c] = cols[i + (nint)c * lda];
        }

        for (int j = 0; j <= i; j++)
        {
            Level1Kernels.Axpy(-a[i + (nint)j * lda], new ReadOnlySpan<double>(y + lo + (nint)j * ldy, cnt), r);
        }

        if (i > 0)
        {
            var xs = new ReadOnlySpan<double>(xr, i);
            for (int c = 0; c < cnt; c++)
            {
                r[c] -= Level1Kernels.Dot(new ReadOnlySpan<double>(cols + (nint)c * lda, i), xs);
            }
        }
    }

    /// <summary>
    /// 担当の列について行 i を書き戻し、部分和 <c>Σ A[0:m, c] v_c</c> と <c>Σ Y[c, 0:i+1] v_c</c> を作る
    /// </summary>
    private void Region2(int t)
    {
        var (lo, hi) = team.Range(t, i + 1);
        var s = new Span<double>(sPartial + t * nbMax, i + 1);
        if (hi <= lo)
        {
            // 空の区間の w の部分和は AddPartials が飛ばすので、w（0 番）だけを0にする
            if (t == 0)
            {
                new Span<double>(w, m).Clear();
            }

            s.Clear();
            return;
        }

        var dest = new Span<double>(t == 0 ? w : team.Partial(t), m);

        int cnt = hi - lo;
        double* cols = a + (nint)lo * lda;
        var r = new ReadOnlySpan<double>(rb + (lo - i - 1), cnt);
        for (int c = 0; c < cnt; c++)
        {
            cols[i + (nint)c * lda] = r[c];
        }

        MatrixVector.MultiplySequential(m, cnt, new ReadOnlySpan<double>(cols, (cnt - 1) * lda + m), lda, r, dest);
        for (int j = 0; j <= i; j++)
        {
            s[j] = Level1Kernels.Dot(new ReadOnlySpan<double>(y + lo + (nint)j * ldy, cnt), r);
        }
    }
}
