using Tuna.LinearAlgebra;

namespace Tuna.Optimization.Gradient;

// Ring buffer holding up to capacity recent curvature pairs (sₖ, yₖ) indexed oldest-first (0..Count-1).
// When trackProducts is true, Gram matrices sⱼᵀsₖ, sⱼᵀyₖ, yⱼᵀyₖ for L-BFGS-B's compact representation are updated in O(mn) per insertion.
internal sealed class CurvaturePairs(int dimension, int capacity, bool trackProducts)
{
    private readonly double[] s = new double[checked(dimension * capacity)];
    private readonly double[] y = new double[checked(dimension * capacity)];
    private readonly double[] sy = new double[capacity];
    private readonly double[] yy = new double[capacity];
    private readonly double[]? ss = trackProducts ? new double[capacity * capacity] : null;
    private readonly double[]? sTy = trackProducts ? new double[capacity * capacity] : null;
    private readonly double[]? yTy = trackProducts ? new double[capacity * capacity] : null;
    private int start;

    public int Count { get; private set; }

    public int Capacity => capacity;

    private int Slot(int k) => (start + k) % capacity;

    public ReadOnlySpan<double> S(int k) => s.AsSpan(Slot(k) * dimension, dimension);

    public ReadOnlySpan<double> Y(int k) => y.AsSpan(Slot(k) * dimension, dimension);

    public double SY(int k) => sy[Slot(k)];

    public double YY(int k) => yy[Slot(k)];

    public double SS(int j, int k) => ss![Slot(j) * capacity + Slot(k)];

    public double STY(int j, int k) => sTy![Slot(j) * capacity + Slot(k)];

    public double YTY(int j, int k) => yTy![Slot(j) * capacity + Slot(k)];

    // Initial inverse-Hessian scaling γ = sᵀy / yᵀy from the most recent pair (L-BFGS-B uses θ = 1/γ).
    public double LatestScaling => Count == 0 ? 1.0 : SY(Count - 1) / YY(Count - 1);

    public void Clear()
    {
        start = 0;
        Count = 0;
    }

    public void Add(ReadOnlySpan<double> sNew, ReadOnlySpan<double> yNew, double sNewY, double yNewY)
    {
        int slot;
        if (Count == capacity)
        {
            slot = start;
            start = (start + 1) % capacity;
        }
        else
        {
            slot = Slot(Count);
            Count++;
        }

        sNew.CopyTo(s.AsSpan(slot * dimension, dimension));
        yNew.CopyTo(y.AsSpan(slot * dimension, dimension));
        sy[slot] = sNewY;
        yy[slot] = yNewY;
        if (ss is null)
        {
            return;
        }

        for (int k = 0; k < Count; k++)
        {
            int other = Slot(k);
            ReadOnlySpan<double> sk = s.AsSpan(other * dimension, dimension);
            ReadOnlySpan<double> yk = y.AsSpan(other * dimension, dimension);
            double sNewSk = other == slot ? Blas.Dot(sNew, sNew) : Blas.Dot(sNew, sk);
            ss[slot * capacity + other] = sNewSk;
            ss[other * capacity + slot] = sNewSk;
            double yNewYk = other == slot ? yNewY : Blas.Dot(yNew, yk);
            yTy![slot * capacity + other] = yNewYk;
            yTy[other * capacity + slot] = yNewYk;
            sTy![slot * capacity + other] = other == slot ? sNewY : Blas.Dot(sNew, yk);
            sTy[other * capacity + slot] = other == slot ? sNewY : Blas.Dot(sk, yNew);
        }
    }
}
