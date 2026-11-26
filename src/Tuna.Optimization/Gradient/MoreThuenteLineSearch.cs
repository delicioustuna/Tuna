namespace Tuna.Optimization.Gradient;

internal enum LineSearchStatus
{
    Continue,
    Converged,
    MaximumStep,
    MinimumStep,
    RoundingLimit,
}

// Moré–Thuente (1994) strong Wolfe line search.
// Stage 1 minimizes the auxiliary function ψ(α) = φ(α) − φ(0) − μαφ′(0); stage 2 switches to φ once ψ(α) ≤ 0 and ψ′(α) ≥ 0.
// Non-finite trial evaluations are treated as +∞ bracket endpoints (excluded from interpolation) and bisected toward stx.
internal sealed class MoreThuenteLineSearch
{
    private const double ExtrapolateLower = 1.1;
    private const double ExtrapolateUpper = 4.0;
    private const double RequiredShrink = 0.66;
    private const double RelativeWidthTolerance = 1e-12;

    private readonly double mu;
    private readonly double eta;

    private double finit;
    private double ginit;
    private double gtest;
    private double stx, fx, gx;      // Best bracket endpoint (true φ value and derivative)
    private double sty, fy, gy;      // Other bracket endpoint
    private bool yValid;             // Whether sty's value/derivative are finite and usable for interpolation
    private double stp;              // Current trial step
    private double stmin, stmax;     // Safeguarded bounds for the current step choice
    private double stpMin, stpMax;
    private bool bracketed;
    private int stage;
    private double width, previousWidth;

    public MoreThuenteLineSearch(double sufficientDecrease, double curvature)
    {
        mu = sufficientDecrease;
        eta = curvature;
    }

    public double Step => stp;

    public double BestStep => stx;

    public double BestValue => fx;

    public bool BestIsLatest { get; private set; }

    // Require strict decrease (fx < finit) so rounding of finit + stx * gtest == finit is not accepted as progress.
    public bool BestSatisfiesSufficientDecrease => stx > 0 && fx <= finit + stx * gtest && fx < finit;

    public void Start(double f0, double g0, double initialStep, double minStep, double maxStep)
    {
        finit = f0;
        ginit = g0;
        gtest = mu * g0;
        stx = sty = 0;
        fx = fy = f0;
        gx = gy = g0;
        yValid = true;
        bracketed = false;
        stage = 1;
        stpMin = minStep;
        stpMax = maxStep;
        width = maxStep - minStep;
        previousWidth = 2 * width;
        stp = Bounds.Clamp(initialStep, minStep, maxStep);
        stmin = 0;
        stmax = stp + ExtrapolateUpper * stp;
        BestIsLatest = false;
    }

    public LineSearchStatus Next(double f, double g)
    {
        BestIsLatest = false;
        double ftest = finit + stp * gtest;
        if (stage == 1 && f <= ftest && g >= gtest)
        {
            stage = 2;
        }

        if (f <= ftest && Math.Abs(g) <= -eta * ginit)
        {
            return LineSearchStatus.Converged;
        }

        if (stp == stpMax && f <= ftest && g <= gtest)
        {
            return LineSearchStatus.MaximumStep;
        }

        if (stp == stpMin && (f > ftest || g >= gtest))
        {
            return LineSearchStatus.MinimumStep;
        }

        // 次の試行点の範囲（外挿の幅）は、区間を更新する前の端点から決める
        if (bracketed)
        {
            stmin = Math.Min(stx, sty);
            stmax = Math.Max(stx, sty);
        }
        else
        {
            stmin = stp + ExtrapolateLower * (stp - stx);
            stmax = stp + ExtrapolateUpper * (stp - stx);
        }

        // 第1段階は ψ（定数の φ(0) は比較にも補間にも効かないので省く）で、第2段階は φ で考える
        double shift = stage == 1 ? gtest : 0;
        double fxw = fx - stx * shift, gxw = gx - shift;
        double fyw = fy - sty * shift, gyw = gy - shift;
        double fw = f - stp * shift, gw = g - shift;

        double next = ChooseStep(stx, fxw, gxw, sty, fyw, gyw, yValid, stp, fw, gw, ref bracketed, stmin, stmax);

        // 区間の更新（論文の更新規則。端点には真の φ の値を持つ）
        if (fw > fxw)
        {
            sty = stp;
            fy = f;
            gy = g;
            yValid = true;
        }
        else
        {
            if (gw * (stx - stp) < 0)
            {
                sty = stx;
                fy = fx;
                gy = gx;
                yValid = true;
            }

            stx = stp;
            fx = f;
            gx = g;
            BestIsLatest = true;
        }

        return Advance(next);
    }

    public LineSearchStatus RejectNonFinite()
    {
        BestIsLatest = false;
        bracketed = true;
        sty = stp;
        fy = double.PositiveInfinity;
        gy = double.NaN;
        yValid = false;
        return Advance(stx + 0.5 * (stp - stx));
    }

    private LineSearchStatus Advance(double next)
    {
        if (bracketed)
        {
            double span = Math.Abs(sty - stx);
            if (span >= RequiredShrink * previousWidth)
            {
                next = stx + 0.5 * (sty - stx);
            }

            previousWidth = width;
            width = span;
            stmin = Math.Min(stx, sty);
            stmax = Math.Max(stx, sty);
            if (!(next > stmin && next < stmax))
            {
                next = stx + 0.5 * (sty - stx);
            }

            if (stmax - stmin <= RelativeWidthTolerance * stmax || !(next > stmin && next < stmax))
            {
                return LineSearchStatus.RoundingLimit;
            }
        }

        next = Bounds.Clamp(next, stpMin, stpMax);
        if (next == stp)
        {
            return LineSearchStatus.RoundingLimit;
        }

        stp = next;
        return LineSearchStatus.Continue;
    }

    // Four trial-step selection cases of Moré & Thuente (1994).
    private static double ChooseStep(double a, double fa, double da, double b, double fb, double db, bool bValid,
        double c, double fc, double dc, ref bool bracketed, double lowest, double highest)
    {
        double toward = c - a;
        if (fc > fa)
        {
            // Case 1: Higher function value; minimum is bracketed between a and c.
            double cubic = CubicMinimizer(a, fa, da, c, fc, dc, extrapolate: false, out double cu) ? cu : 0.5 * (a + c);
            double quadratic = QuadraticMinimizer(a, fa, da, c, fc);
            bracketed = true;
            return Math.Abs(cubic - a) < Math.Abs(quadratic - a) ? cubic : cubic + 0.5 * (quadratic - cubic);
        }

        if (dc * Math.Sign(da) < 0)
        {
            // Case 2: Opposite-sign derivative; minimum is bracketed between a and c.
            double cubic = CubicMinimizer(a, fa, da, c, fc, dc, extrapolate: false, out double cu) ? cu : 0.5 * (a + c);
            double secant = Secant(a, da, c, dc);
            bracketed = true;
            return Math.Abs(cubic - c) > Math.Abs(secant - c) ? cubic : secant;
        }

        if (Math.Abs(dc) < Math.Abs(da))
        {
            // Case 3: Lower value, same-sign derivative with smaller magnitude.
            double cubic = CubicMinimizer(a, fa, da, c, fc, dc, extrapolate: true, out double cu) && (cu - c) * toward > 0
                ? cu
                : toward > 0 ? highest : lowest;
            double secant = Secant(a, da, c, dc);
            if (bracketed)
            {
                double step = Math.Abs(cubic - c) < Math.Abs(secant - c) ? cubic : secant;
                double limit = c + RequiredShrink * (b - c);
                return toward > 0 ? Math.Min(limit, step) : Math.Max(limit, step);
            }

            double far = Math.Abs(cubic - c) > Math.Abs(secant - c) ? cubic : secant;
            return Bounds.Clamp(far, lowest, highest);
        }

        // Case 4: Lower value, same-sign derivative with non-decreasing magnitude.
        if (bracketed)
        {
            return bValid && CubicMinimizer(c, fc, dc, b, fb, db, extrapolate: false, out double cu) ? cu : 0.5 * (c + b);
        }

        return toward > 0 ? highest : lowest;
    }

    // Cubic minimizer through (a, fa, da) and (b, fb, db) (Nocedal & Wright 2006, eq. 3.59).
    private static bool CubicMinimizer(double a, double fa, double da, double b, double fb, double db, bool extrapolate, out double x)
    {
        x = double.NaN;
        if (a == b)
        {
            return false;
        }

        double d1 = da + db - 3 * (fa - fb) / (a - b);
        double scale = Math.Max(Math.Abs(d1), Math.Max(Math.Abs(da), Math.Abs(db)));
        if (scale == 0 || !double.IsFinite(scale))
        {
            return false;
        }

        double discriminant = (d1 / scale) * (d1 / scale) - (da / scale) * (db / scale);
        if (discriminant < 0)
        {
            if (!extrapolate)
            {
                return false;
            }

            discriminant = 0;
        }

        double d2 = scale * Math.Sqrt(discriminant) * (b > a ? 1 : -1);
        double denominator = db - da + 2 * d2;
        if (denominator == 0)
        {
            return false;
        }

        x = b - (b - a) * ((db + d2 - d1) / denominator);
        return double.IsFinite(x);
    }

    private static double QuadraticMinimizer(double a, double fa, double da, double c, double fc) =>
        a + 0.5 * (c - a) * (da / ((fa - fc) / (c - a) + da));

    private static double Secant(double a, double da, double c, double dc) => c - dc * (c - a) / (dc - da);
}
