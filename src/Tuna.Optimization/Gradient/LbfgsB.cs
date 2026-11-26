using Tuna.LinearAlgebra;

namespace Tuna.Optimization.Gradient;

/// <summary>
/// Minimizes a differentiable objective function subject to simple box bounds using the L-BFGS-B method.
/// <para>L-BFGS-B 法を用いて上下限のボックス制約付き微分可能な目的関数を最小化します。</para>
/// </summary>
/// <remarks>
/// <para>Bounds may include <see cref="double.NegativeInfinity"/> or <see cref="double.PositiveInfinity"/>; variables with equal lower and upper bounds are held fixed. The initial point is projected onto the box, and convergence is measured using the infinity norm of the projected gradient <c>x − P(x − ∇f)</c>. See <see cref="LineSearchMinimizer"/> for the evaluation workflow and termination criteria.</para>
/// <para>上下限には <see cref="double.NegativeInfinity"/> や <see cref="double.PositiveInfinity"/> を指定でき、下限と上限が等しい変数はその値に固定されます。初期点はボックス内に射影され、収束判定には射影勾配 <c>x − P(x − ∇f)</c> の最大ノルムを使用します。評価手順と終了条件の詳細は <see cref="LineSearchMinimizer"/> を参照してください。</para>
/// </remarks>
public sealed class LbfgsB : LineSearchMinimizer
{
    private static readonly double Epsilon = double.BitIncrement(1.0) - 1.0;

    private readonly double[] lower;
    private readonly double[] upper;
    private readonly CurvaturePairs pairs;
    private readonly int capacity;

    // Generalized Cauchy point and subspace minimization buffers (length n)
    private readonly double[] breakpoint;
    private readonly double[] pathDirection;
    private readonly double[] cauchy;
    private readonly double[] reduced;
    private readonly double[] subspaceStep;
    private readonly int[] heap;
    private readonly int[] free;
    private readonly bool[] isFree;

    // Compact BFGS representation buffers (lengths 2m, m×m, 2m×2m)
    private readonly double[] p;
    private readonly double[] c;
    private readonly double[] row;
    private readonly double[] mRow;
    private readonly double[] small;
    private readonly double[] small2;
    private readonly double[] middleDiagonal;
    private readonly double[] lowerTriangle;
    private readonly double[] tMatrix;
    private CholeskyDecomposition? tFactor;
    private double theta = 1;

    private int limitingIndex = -1;
    private double limitingBound;
    private double limitingStep;

    /// <summary>
    /// Initializes a new L-BFGS-B minimizer over the box <c>[lowerBounds[i], upperBounds[i]]</c>.
    /// <para>ボックス領域 <c>[lowerBounds[i], upperBounds[i]]</c> 上で最小化を行う L-BFGS-B オプティマイザーを初期化します。</para>
    /// </summary>
    /// <param name="lowerBounds">Lower bounds for each variable (may include <see cref="double.NegativeInfinity"/>). Pass an empty span to leave all lower bounds unbounded. / 各変数の下限（<see cref="double.NegativeInfinity"/> も指定可能）。空のスパンを渡すと全変数が −∞ となります。</param>
    /// <param name="upperBounds">Upper bounds for each variable (may include <see cref="double.PositiveInfinity"/>, at least the corresponding lower bound). Pass an empty span to leave all upper bounds unbounded. / 各変数の上限（<see cref="double.PositiveInfinity"/> も指定可能、対応する下限以上）。空のスパンを渡すと全変数が +∞ となります。</param>
    /// <param name="options">Optional solver configuration, or <see langword="null"/> for defaults. / オプション設定。<see langword="null"/> の場合は既定値を使用します。</param>
    /// <exception cref="ArgumentException">Both bound spans are empty, their lengths differ, a bound is <see cref="double.NaN"/>, a lower bound exceeds an upper bound, or the box is empty. / 上下限が両方とも空、長さが異なる、<see cref="double.NaN"/> を含む、下限が上限を超えている、または実行可能領域が空の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">An option value in <paramref name="options"/> is out of range. / <paramref name="options"/> の設定値が有効範囲外の場合。</exception>
    public LbfgsB(ReadOnlySpan<double> lowerBounds, ReadOnlySpan<double> upperBounds, LbfgsBOptions? options = null)
        : base(BoundsDimension(lowerBounds, upperBounds), options ??= new LbfgsBOptions())
    {
        Bounds.Validate(lowerBounds, upperBounds, n, allowEmpty: false, out double[]? lo, out double[]? hi);
        lower = lo!;
        upper = hi!;
        capacity = options.HistorySize;
        pairs = new CurvaturePairs(n, capacity, trackProducts: true);
        breakpoint = new double[n];
        pathDirection = new double[n];
        cauchy = new double[n];
        reduced = new double[n];
        subspaceStep = new double[n];
        heap = new int[n];
        free = new int[n];
        isFree = new bool[n];
        p = new double[2 * capacity];
        c = new double[2 * capacity];
        row = new double[2 * capacity];
        mRow = new double[2 * capacity];
        small = new double[2 * capacity];
        small2 = new double[2 * capacity];
        middleDiagonal = new double[capacity];
        lowerTriangle = new double[capacity * capacity];
        tMatrix = new double[capacity * capacity];
    }

    /// <summary>
    /// Gets the lower bounds for each variable.
    /// <para>各変数の下限を取得します。</para>
    /// </summary>
    public ReadOnlySpan<double> LowerBounds => lower;

    /// <summary>
    /// Gets the upper bounds for each variable.
    /// <para>各変数の上限を取得します。</para>
    /// </summary>
    public ReadOnlySpan<double> UpperBounds => upper;

    /// <summary>
    /// Gets the maximum number of curvature pairs retained in history.
    /// <para>履歴に保持する曲率ペアの最大数 <c>m</c> を取得します。</para>
    /// </summary>
    public int HistorySize => capacity;

    /// <summary>
    /// Gets the number of curvature pairs currently stored in history.
    /// <para>現在履歴に保持している曲率ペアの数を取得します。</para>
    /// </summary>
    public int HistoryCount => pairs.Count;

    private static int BoundsDimension(ReadOnlySpan<double> lowerBounds, ReadOnlySpan<double> upperBounds)
    {
        if (lowerBounds.IsEmpty && upperBounds.IsEmpty)
        {
            throw new ArgumentException("At least one of the bounds must be given.", nameof(lowerBounds));
        }

        if (!lowerBounds.IsEmpty && !upperBounds.IsEmpty && lowerBounds.Length != upperBounds.Length)
        {
            throw new ArgumentException("The bounds must have the same length.", nameof(upperBounds));
        }

        return Math.Max(lowerBounds.Length, upperBounds.Length);
    }

    private protected override bool HasHistory => pairs.Count > 0;

    private protected override void ClearHistory()
    {
        pairs.Clear();
        tFactor = null;
        theta = 1;
    }

    private protected override void Prepare(Span<double> point)
    {
        for (int i = 0; i < n; i++)
        {
            point[i] = Bounds.Clamp(point[i], lower[i], upper[i]);
        }
    }

    private protected override double ComputeGradientNorm()
    {
        double norm = 0;
        for (int i = 0; i < n; i++)
        {
            double projected = x[i] - Bounds.Clamp(x[i] - g[i], lower[i], upper[i]);
            norm = Math.Max(norm, Math.Abs(projected));
        }

        return norm;
    }

    private protected override void UpdateHistory(ReadOnlySpan<double> s, ReadOnlySpan<double> y)
    {
        double sy = Blas.Dot(s, y);
        double yy = Blas.Dot(y, y);
        if (sy > Epsilon * yy && double.IsFinite(sy) && double.IsFinite(yy))
        {
            pairs.Add(s, y, sy, yy);
        }
    }

    private protected override double MaxStep(ReadOnlySpan<double> d)
    {
        double step = UnboundedStep;
        limitingIndex = -1;
        for (int i = 0; i < n; i++)
        {
            double ratio;
            double bound;
            if (d[i] > 0 && upper[i] < double.PositiveInfinity)
            {
                bound = upper[i];
            }
            else if (d[i] < 0 && lower[i] > double.NegativeInfinity)
            {
                bound = lower[i];
            }
            else
            {
                continue;
            }

            ratio = (bound - x[i]) / d[i];
            if (ratio < step)
            {
                step = ratio;
                limitingIndex = i;
                limitingBound = bound;
            }
        }

        limitingStep = Math.Max(step, 0);
        return limitingStep;
    }

    private protected override void TrialPoint(double alpha, ReadOnlySpan<double> d, Span<double> destination)
    {
        for (int i = 0; i < n; i++)
        {
            destination[i] = Bounds.Clamp(x[i] + alpha * d[i], lower[i], upper[i]);
        }

        if (alpha == limitingStep && limitingIndex >= 0)
        {
            // 範囲に収まる最大の歩幅では、その歩幅を決めた変数をちょうど境界に置く
            destination[limitingIndex] = limitingBound;
        }
    }

    private protected override bool ComputeDirection(Span<double> d)
    {
        int m = pairs.Count;
        theta = m > 0 ? 1.0 / pairs.LatestScaling : 1.0;
        if (m > 0 && !FactorMiddle(m))
        {
            return false;
        }

        CauchyPoint(m);
        int freeCount = FreeVariables();
        if (freeCount > 0)
        {
            Subspace(m, freeCount);
        }

        for (int i = 0; i < n; i++)
        {
            d[i] = cauchy[i] - x[i];
            if (!double.IsFinite(d[i]))
            {
                return false;
            }
        }

        return true;
    }

    // ---------- コンパクト表現 ----------

    // D = diag(sₖᵀyₖ)、L（狭義下三角）= sⱼᵀyₖ（j > k）、T = θSᵀS + LD⁻¹Lᵀ のコレスキー分解
    private bool FactorMiddle(int m)
    {
        for (int k = 0; k < m; k++)
        {
            middleDiagonal[k] = pairs.SY(k);
            for (int j = 0; j < m; j++)
            {
                lowerTriangle[j + k * m] = j > k ? pairs.STY(j, k) : 0;
            }
        }

        for (int j = 0; j < m; j++)
        {
            for (int i = j; i < m; i++)
            {
                double sum = theta * pairs.SS(i, j);
                for (int k = 0; k < j; k++)
                {
                    sum += lowerTriangle[i + k * m] * lowerTriangle[j + k * m] / middleDiagonal[k];
                }

                tMatrix[i + j * m] = sum;
                tMatrix[j + i * m] = sum;
            }
        }

        var t = Matrix<double>.Wrap(m, m, tMatrix.AsSpan(0, m * m).ToArray());
        if (!CholeskyDecomposition.TryCompute(t, t, out CholeskyDecomposition factor))
        {
            tFactor = null;
            return false;
        }

        tFactor = factor;
        return true;
    }

    // result = M v（M = [[−D, Lᵀ], [L, θSᵀS]]⁻¹）。v と result は長さ 2m で重ならない
    private void MultiplyMiddle(int m, ReadOnlySpan<double> v, Span<double> result)
    {
        if (m == 0)
        {
            return;
        }

        Span<double> v1 = small2.AsSpan(0, m);
        Span<double> x1 = result[..m];
        Span<double> x2 = result.Slice(m, m);
        for (int k = 0; k < m; k++)
        {
            v1[k] = v[k] / middleDiagonal[k];
        }

        // a₂ = v₂ + LD⁻¹v₁、x₂ = T⁻¹a₂
        for (int j = 0; j < m; j++)
        {
            double sum = v[m + j];
            for (int k = 0; k < j; k++)
            {
                sum += lowerTriangle[j + k * m] * v1[k];
            }

            x2[j] = sum;
        }

        tFactor!.Solve(x2, x2);

        // x₁ = D⁻¹(Lᵀx₂ − v₁)
        for (int k = 0; k < m; k++)
        {
            double sum = -v[k];
            for (int j = k + 1; j < m; j++)
            {
                sum += lowerTriangle[j + k * m] * x2[j];
            }

            x1[k] = sum / middleDiagonal[k];
        }
    }

    // W の第 i 行 [y₀[i], …, y_{m−1}[i], θs₀[i], …, θs_{m−1}[i]]
    private void Row(int m, int i, Span<double> destination)
    {
        for (int k = 0; k < m; k++)
        {
            destination[k] = pairs.Y(k)[i];
            destination[m + k] = theta * pairs.S(k)[i];
        }
    }

    // (Wu)ᵢ
    private double RowDot(int m, int i, ReadOnlySpan<double> u)
    {
        double y = 0, s = 0;
        for (int k = 0; k < m; k++)
        {
            y += pairs.Y(k)[i] * u[k];
            s += pairs.S(k)[i] * u[m + k];
        }

        return y + theta * s;
    }

    // ---------- 一般化コーシー点 ----------

    private void CauchyPoint(int m)
    {
        int heapCount = 0;
        double f1 = 0;
        int moving = 0;
        for (int i = 0; i < n; i++)
        {
            double gi = g[i];
            double t;
            if (gi < 0)
            {
                t = upper[i] == double.PositiveInfinity ? double.PositiveInfinity : (x[i] - upper[i]) / gi;
            }
            else if (gi > 0)
            {
                t = lower[i] == double.NegativeInfinity ? double.PositiveInfinity : (x[i] - lower[i]) / gi;
            }
            else
            {
                t = double.PositiveInfinity;
            }

            breakpoint[i] = t;
            cauchy[i] = x[i];
            double di = t > 0 ? -gi : 0;
            pathDirection[i] = di;
            if (di != 0)
            {
                moving++;
                f1 -= di * di;
                if (t < double.PositiveInfinity)
                {
                    heap[heapCount++] = i;
                }
            }
        }

        Span<double> pm = p.AsSpan(0, 2 * m);
        Span<double> cm = c.AsSpan(0, 2 * m);
        cm.Clear();
        if (moving == 0)
        {
            return;
        }

        // p = Wᵀd
        for (int k = 0; k < m; k++)
        {
            pm[k] = Blas.Dot(pairs.Y(k), pathDirection);
            pm[m + k] = theta * Blas.Dot(pairs.S(k), pathDirection);
        }

        Span<double> mp = mRow.AsSpan(0, 2 * m);
        MultiplyMiddle(m, pm, mp);
        double f2 = -theta * f1 - Dot(pm, mp);
        double f2Floor = Epsilon * f2;
        double dtMin = -f1 / f2;
        double tOld = 0;

        for (int i = heapCount / 2 - 1; i >= 0; i--)
        {
            SiftDown(heap, heapCount, i);
        }

        Span<double> wb = row.AsSpan(0, 2 * m);
        Span<double> mwb = mRow.AsSpan(0, 2 * m);
        while (heapCount > 0)
        {
            int b = heap[0];
            double t = breakpoint[b];
            double dt = t - tOld;
            if (dtMin < dt)
            {
                break;
            }

            heap[0] = heap[--heapCount];
            SiftDown(heap, heapCount, 0);

            // 折れ点 t で変数 b が範囲に達する
            cauchy[b] = pathDirection[b] > 0 ? upper[b] : lower[b];
            double zb = cauchy[b] - x[b];
            double gb = g[b];
            for (int k = 0; k < 2 * m; k++)
            {
                cm[k] += dt * pm[k];
            }

            Row(m, b, wb);
            MultiplyMiddle(m, wb, mwb);
            f1 += dt * f2 + gb * gb + theta * gb * zb - gb * Dot(mwb, cm);
            f2 -= theta * gb * gb + 2 * gb * Dot(mwb, pm) + gb * gb * Dot(mwb, wb);
            for (int k = 0; k < 2 * m; k++)
            {
                pm[k] += gb * wb[k];
            }

            pathDirection[b] = 0;
            moving--;
            tOld = t;
            if (moving == 0)
            {
                return;
            }

            f2 = Math.Max(f2, f2Floor);
            dtMin = -f1 / f2;
        }

        dtMin = Math.Max(dtMin, 0);
        tOld += dtMin;
        for (int i = 0; i < n; i++)
        {
            if (pathDirection[i] != 0)
            {
                cauchy[i] = Bounds.Clamp(x[i] + tOld * pathDirection[i], lower[i], upper[i]);
            }
        }

        for (int k = 0; k < 2 * m; k++)
        {
            cm[k] += dtMin * pm[k];
        }
    }

    // 折れ点の小さい順（同じなら番号の小さい順）の二分ヒープ
    private void SiftDown(int[] items, int count, int index)
    {
        int item = items[index];
        while (true)
        {
            int child = 2 * index + 1;
            if (child >= count)
            {
                break;
            }

            if (child + 1 < count && Before(items[child + 1], items[child]))
            {
                child++;
            }

            if (!Before(items[child], item))
            {
                break;
            }

            items[index] = items[child];
            index = child;
        }

        items[index] = item;
    }

    private bool Before(int a, int b) => breakpoint[a] < breakpoint[b] || (breakpoint[a] == breakpoint[b] && a < b);

    private int FreeVariables()
    {
        int count = 0;
        for (int i = 0; i < n; i++)
        {
            bool isInterior = cauchy[i] > lower[i] && cauchy[i] < upper[i];
            isFree[i] = isInterior;
            if (isInterior)
            {
                free[count++] = i;
            }
        }

        return count;
    }

    // ---------- 部分空間の最小化 ----------

    private void Subspace(int m, int freeCount)
    {
        // 自由変数の縮約勾配 r̂ = Zᵀ(g + θ(xᶜ − x) − WMc)
        Span<double> mc = small.AsSpan(0, 2 * m);
        MultiplyMiddle(m, c.AsSpan(0, 2 * m), mc);
        for (int j = 0; j < freeCount; j++)
        {
            int i = free[j];
            reduced[i] = g[i] + theta * (cauchy[i] - x[i]) - (m > 0 ? RowDot(m, i, mc) : 0);
        }

        if (m == 0)
        {
            for (int j = 0; j < freeCount; j++)
            {
                int i = free[j];
                subspaceStep[i] = -reduced[i] / theta;
            }
        }
        else if (!SubspaceStep(m, freeCount))
        {
            return;
        }

        // Morales & Nocedal (2011): 射影した点が降下方向なら採る
        double slope = 0;
        for (int i = 0; i < n; i++)
        {
            double target = isFree[i] ? Bounds.Clamp(cauchy[i] + subspaceStep[i], lower[i], upper[i]) : cauchy[i];
            slope += (target - x[i]) * g[i];
        }

        if (slope < 0)
        {
            for (int j = 0; j < freeCount; j++)
            {
                int i = free[j];
                cauchy[i] = Bounds.Clamp(cauchy[i] + subspaceStep[i], lower[i], upper[i]);
            }

            return;
        }

        // そうでなければ範囲に収まるまで一様に縮める（Byrd et al. 1995 の 5.1 節）
        double alpha = 1;
        for (int j = 0; j < freeCount; j++)
        {
            int i = free[j];
            double di = subspaceStep[i];
            if (di > 0 && upper[i] < double.PositiveInfinity)
            {
                alpha = Math.Min(alpha, (upper[i] - cauchy[i]) / di);
            }
            else if (di < 0 && lower[i] > double.NegativeInfinity)
            {
                alpha = Math.Min(alpha, (lower[i] - cauchy[i]) / di);
            }
        }

        alpha = Math.Max(alpha, 0);
        for (int j = 0; j < freeCount; j++)
        {
            int i = free[j];
            cauchy[i] = Bounds.Clamp(cauchy[i] + alpha * subspaceStep[i], lower[i], upper[i]);
        }
    }

    // B̂⁻¹ = (1/θ)I + (1/θ²)U(K − UᵀU/θ)⁻¹Uᵀ（U = ZᵀW）で d̂ = −B̂⁻¹r̂。解けなければ偽（一般化コーシー点のまま）
    private bool SubspaceStep(int m, int freeCount)
    {
        int size = 2 * m;
        var system = new double[size * size];
        Span<double> rhs = small.AsSpan(0, size);
        rhs.Clear();
        Span<double> w = row.AsSpan(0, size);

        // UᵀU = Σ_{自由} wᵢwᵢᵀ。自由変数が半分を超えるなら WᵀW − Σ_{固定} wᵢwᵢᵀ のほうが安い
        bool subtract = freeCount > n / 2;
        if (subtract)
        {
            for (int k = 0; k < m; k++)
            {
                for (int j = 0; j < m; j++)
                {
                    system[j + k * size] = pairs.YTY(j, k);
                    system[(m + j) + k * size] = theta * pairs.STY(j, k);
                    system[j + (m + k) * size] = theta * pairs.STY(k, j);
                    system[(m + j) + (m + k) * size] = theta * theta * pairs.SS(j, k);
                }
            }
        }

        for (int i = 0; i < n; i++)
        {
            if (isFree[i] == subtract)
            {
                continue;
            }

            Row(m, i, w);
            double sign = subtract ? -1 : 1;
            for (int k = 0; k < size; k++)
            {
                double wk = sign * w[k];
                for (int j = 0; j < size; j++)
                {
                    system[j + k * size] += w[j] * wk;
                }
            }
        }

        for (int j = 0; j < freeCount; j++)
        {
            int i = free[j];
            Row(m, i, w);
            double r = reduced[i];
            for (int k = 0; k < size; k++)
            {
                rhs[k] += w[k] * r;
            }
        }

        // N = K − UᵀU/θ（K = [[−D, Lᵀ], [L, θSᵀS]]）
        double inverseTheta = 1.0 / theta;
        for (int k = 0; k < size; k++)
        {
            for (int j = 0; j < size; j++)
            {
                double kjk;
                if (j < m && k < m)
                {
                    kjk = j == k ? -middleDiagonal[k] : 0;
                }
                else if (j < m)
                {
                    kjk = lowerTriangle[(k - m) + j * m];             // Lᵀ[j, k − m] = L[k − m, j]
                }
                else if (k < m)
                {
                    kjk = lowerTriangle[(j - m) + k * m];             // L[j − m, k]
                }
                else
                {
                    kjk = theta * pairs.SS(j - m, k - m);
                }

                system[j + k * size] = kjk - inverseTheta * system[j + k * size];
            }
        }

        var lu = LuDecomposition.Compute(Matrix<double>.Wrap(size, size, system), overwriteInput: true);
        if (lu.IsSingular)
        {
            return false;
        }

        Span<double> solution = small2.AsSpan(0, size);
        lu.Solve(new ReadOnlyVectorSpan<double>(rhs), new VectorSpan<double>(solution));
        double inverseTheta2 = inverseTheta * inverseTheta;
        for (int j = 0; j < freeCount; j++)
        {
            int i = free[j];
            double step = -reduced[i] * inverseTheta - inverseTheta2 * RowDot(m, i, solution);
            if (!double.IsFinite(step))
            {
                return false;
            }

            subspaceStep[i] = step;
        }

        return true;
    }

    private static double Dot(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
    {
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            sum += a[i] * b[i];
        }

        return sum;
    }
}
