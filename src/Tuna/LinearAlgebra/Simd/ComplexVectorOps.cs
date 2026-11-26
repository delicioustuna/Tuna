using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace Tuna.LinearAlgebra;

// Complex 連続領域向けの SIMD ベクトル演算（BLAS Level 1/2 内部カーネル）。
// 実部・虚部が交互に並ぶレイアウトを double ベクトルとして処理し、実部・虚部スワップとの 2 回の積和で複素乗算を行う。
internal static class ComplexVectorOps
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ref double AsDoubles(ReadOnlySpan<Complex> x) =>
        ref Unsafe.As<Complex, double>(ref MemoryMarshal.GetReference(x));

    // 隣り合う実部・虚部を入れ替える
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<double> Swap(Vector512<double> v) => Vector512.Shuffle(v, Vector512.Create(1L, 0, 3, 2, 5, 4, 7, 6));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<double> Swap(Vector256<double> v) => Vector256.Shuffle(v, Vector256.Create(1L, 0, 3, 2));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<double> Swap(Vector128<double> v) => Vector128.Shuffle(v, Vector128.Create(1L, 0));

    // 偶数番目（実部）が +1、奇数番目（虚部）が -1
    private static Vector512<double> Alternate512
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => Vector512.Create(1.0, -1, 1, -1, 1, -1, 1, -1);
    }

    private static Vector256<double> Alternate256
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => Vector256.Create(1.0, -1, 1, -1);
    }

    private static Vector128<double> Alternate128
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => Vector128.Create(1.0, -1);
    }

    // conjugateX が true なら zdotc、false なら zdotu。
    internal static Complex Dot(ReadOnlySpan<Complex> x, ReadOnlySpan<Complex> y, bool conjugateX)
    {
        int n = x.Length;
        int len = 2 * n;
        ref double px = ref AsDoubles(x);
        ref double py = ref AsDoubles(y);
        int i = 0;

        // p = Σ (xr·yr, xi·yi)、q = Σ (xr·yi, xi·yr) を要素ごとに貯める
        double pEven = 0, pOdd = 0, qEven = 0, qOdd = 0;
        if (Vector512.IsHardwareAccelerated && len >= 8)
        {
            var p0 = Vector512<double>.Zero;
            var p1 = Vector512<double>.Zero;
            var q0 = Vector512<double>.Zero;
            var q1 = Vector512<double>.Zero;
            for (; i <= len - 16; i += 16)
            {
                var x0 = Vector512.LoadUnsafe(ref px, (nuint)i);
                var y0 = Vector512.LoadUnsafe(ref py, (nuint)i);
                var x1 = Vector512.LoadUnsafe(ref px, (nuint)(i + 8));
                var y1 = Vector512.LoadUnsafe(ref py, (nuint)(i + 8));
                p0 = Vector512.FusedMultiplyAdd(x0, y0, p0);
                q0 = Vector512.FusedMultiplyAdd(x0, Swap(y0), q0);
                p1 = Vector512.FusedMultiplyAdd(x1, y1, p1);
                q1 = Vector512.FusedMultiplyAdd(x1, Swap(y1), q1);
            }

            for (; i <= len - 8; i += 8)
            {
                var x0 = Vector512.LoadUnsafe(ref px, (nuint)i);
                var y0 = Vector512.LoadUnsafe(ref py, (nuint)i);
                p0 = Vector512.FusedMultiplyAdd(x0, y0, p0);
                q0 = Vector512.FusedMultiplyAdd(x0, Swap(y0), q0);
            }

            var p = p0 + p1;
            var q = q0 + q1;
            double pSum = Vector512.Sum(p);
            double pAlt = Vector512.Sum(p * Alternate512);
            double qSum = Vector512.Sum(q);
            double qAlt = Vector512.Sum(q * Alternate512);
            pEven = 0.5 * (pSum + pAlt);
            pOdd = 0.5 * (pSum - pAlt);
            qEven = 0.5 * (qSum + qAlt);
            qOdd = 0.5 * (qSum - qAlt);
        }
        else if (Vector256.IsHardwareAccelerated && len >= 4)
        {
            var p0 = Vector256<double>.Zero;
            var q0 = Vector256<double>.Zero;
            for (; i <= len - 4; i += 4)
            {
                var x0 = Vector256.LoadUnsafe(ref px, (nuint)i);
                var y0 = Vector256.LoadUnsafe(ref py, (nuint)i);
                p0 = RealVectorOps.MulAdd(x0, y0, p0);
                q0 = RealVectorOps.MulAdd(x0, Swap(y0), q0);
            }

            pEven = p0.GetElement(0) + p0.GetElement(2);
            pOdd = p0.GetElement(1) + p0.GetElement(3);
            qEven = q0.GetElement(0) + q0.GetElement(2);
            qOdd = q0.GetElement(1) + q0.GetElement(3);
        }
        else if (Vector128.IsHardwareAccelerated && len >= 2)
        {
            var p0 = Vector128<double>.Zero;
            var q0 = Vector128<double>.Zero;
            for (; i <= len - 2; i += 2)
            {
                var x0 = Vector128.LoadUnsafe(ref px, (nuint)i);
                var y0 = Vector128.LoadUnsafe(ref py, (nuint)i);
                p0 = RealVectorOps.MulAdd(x0, y0, p0);
                q0 = RealVectorOps.MulAdd(x0, Swap(y0), q0);
            }

            pEven = p0.GetElement(0);
            pOdd = p0.GetElement(1);
            qEven = q0.GetElement(0);
            qOdd = q0.GetElement(1);
        }

        for (; i < len; i += 2)
        {
            double xr = Unsafe.Add(ref px, i);
            double xi = Unsafe.Add(ref px, i + 1);
            double yr = Unsafe.Add(ref py, i);
            double yi = Unsafe.Add(ref py, i + 1);
            pEven += xr * yr;
            pOdd += xi * yi;
            qEven += xr * yi;
            qOdd += xi * yr;
        }

        return conjugateX
            ? new Complex(pEven + pOdd, qEven - qOdd)
            : new Complex(pEven - pOdd, qEven + qOdd);
    }

    internal static void Axpy(Complex a, ReadOnlySpan<Complex> x, Span<Complex> y)
    {
        int len = 2 * x.Length;
        ref double px = ref AsDoubles(x);
        ref double py = ref AsDoubles(y);
        double ar = a.Real;
        double ai = a.Imaginary;
        int i = 0;

        // y += ar·x + (-ai, ai)·x'
        if (Vector512.IsHardwareAccelerated && len >= 8)
        {
            var vr = Vector512.Create(ar);
            var vi = Vector512.Create(-ai) * Alternate512;
            for (; i <= len - 8; i += 8)
            {
                var xv = Vector512.LoadUnsafe(ref px, (nuint)i);
                var yv = Vector512.LoadUnsafe(ref py, (nuint)i);
                Vector512.FusedMultiplyAdd(vr, xv, Vector512.FusedMultiplyAdd(vi, Swap(xv), yv)).StoreUnsafe(ref py, (nuint)i);
            }
        }
        else if (Vector256.IsHardwareAccelerated && len >= 4)
        {
            var vr = Vector256.Create(ar);
            var vi = Vector256.Create(-ai) * Alternate256;
            for (; i <= len - 4; i += 4)
            {
                var xv = Vector256.LoadUnsafe(ref px, (nuint)i);
                var yv = Vector256.LoadUnsafe(ref py, (nuint)i);
                RealVectorOps.MulAdd(vr, xv, RealVectorOps.MulAdd(vi, Swap(xv), yv)).StoreUnsafe(ref py, (nuint)i);
            }
        }
        else if (Vector128.IsHardwareAccelerated && len >= 2)
        {
            var vr = Vector128.Create(ar);
            var vi = Vector128.Create(-ai) * Alternate128;
            for (; i <= len - 2; i += 2)
            {
                var xv = Vector128.LoadUnsafe(ref px, (nuint)i);
                var yv = Vector128.LoadUnsafe(ref py, (nuint)i);
                RealVectorOps.MulAdd(vr, xv, RealVectorOps.MulAdd(vi, Swap(xv), yv)).StoreUnsafe(ref py, (nuint)i);
            }
        }

        for (; i < len; i += 2)
        {
            double xr = Unsafe.Add(ref px, i);
            double xi = Unsafe.Add(ref px, i + 1);
            Unsafe.Add(ref py, i) += (ar * xr) - (ai * xi);
            Unsafe.Add(ref py, i + 1) += (ar * xi) + (ai * xr);
        }
    }

    internal static void Scale(Complex a, Span<Complex> x)
    {
        int len = 2 * x.Length;
        ref double px = ref AsDoubles(x);
        double ar = a.Real;
        double ai = a.Imaginary;
        int i = 0;
        if (Vector512.IsHardwareAccelerated && len >= 8)
        {
            var vr = Vector512.Create(ar);
            var vi = Vector512.Create(-ai) * Alternate512;
            for (; i <= len - 8; i += 8)
            {
                var xv = Vector512.LoadUnsafe(ref px, (nuint)i);
                Vector512.FusedMultiplyAdd(vr, xv, vi * Swap(xv)).StoreUnsafe(ref px, (nuint)i);
            }
        }
        else if (Vector256.IsHardwareAccelerated && len >= 4)
        {
            var vr = Vector256.Create(ar);
            var vi = Vector256.Create(-ai) * Alternate256;
            for (; i <= len - 4; i += 4)
            {
                var xv = Vector256.LoadUnsafe(ref px, (nuint)i);
                RealVectorOps.MulAdd(vr, xv, vi * Swap(xv)).StoreUnsafe(ref px, (nuint)i);
            }
        }
        else if (Vector128.IsHardwareAccelerated && len >= 2)
        {
            var vr = Vector128.Create(ar);
            var vi = Vector128.Create(-ai) * Alternate128;
            for (; i <= len - 2; i += 2)
            {
                var xv = Vector128.LoadUnsafe(ref px, (nuint)i);
                RealVectorOps.MulAdd(vr, xv, vi * Swap(xv)).StoreUnsafe(ref px, (nuint)i);
            }
        }

        for (; i < len; i += 2)
        {
            double xr = Unsafe.Add(ref px, i);
            double xi = Unsafe.Add(ref px, i + 1);
            Unsafe.Add(ref px, i) = (ar * xr) - (ai * xi);
            Unsafe.Add(ref px, i + 1) = (ar * xi) + (ai * xr);
        }
    }

    // y += a * col と Σ op(col[i]) * x[i] を 1 パスで同時計算する（対称・エルミート行列ベクトル積用）。
    internal static Complex AxpyDot(Complex a, ReadOnlySpan<Complex> col, ReadOnlySpan<Complex> x, Span<Complex> y, bool conjugateColumn)
    {
        int len = 2 * col.Length;
        ref double pc = ref AsDoubles(col);
        ref double px = ref AsDoubles(x);
        ref double py = ref AsDoubles(y);
        double ar = a.Real;
        double ai = a.Imaginary;
        int i = 0;
        double pEven = 0, pOdd = 0, qEven = 0, qOdd = 0;
        if (Vector512.IsHardwareAccelerated && len >= 8)
        {
            var vr = Vector512.Create(ar);
            var vi = Vector512.Create(-ai) * Alternate512;
            var p = Vector512<double>.Zero;
            var q = Vector512<double>.Zero;
            for (; i <= len - 8; i += 8)
            {
                var cv = Vector512.LoadUnsafe(ref pc, (nuint)i);
                var cs = Swap(cv);
                var yv = Vector512.LoadUnsafe(ref py, (nuint)i);
                Vector512.FusedMultiplyAdd(vr, cv, Vector512.FusedMultiplyAdd(vi, cs, yv)).StoreUnsafe(ref py, (nuint)i);
                var xv = Vector512.LoadUnsafe(ref px, (nuint)i);
                p = Vector512.FusedMultiplyAdd(cv, xv, p);
                q = Vector512.FusedMultiplyAdd(cs, xv, q);
            }

            // p = (cr·xr, ci·xi)、q = (ci·xr, cr·xi)
            double pSum = Vector512.Sum(p);
            double pAlt = Vector512.Sum(p * Alternate512);
            double qSum = Vector512.Sum(q);
            double qAlt = Vector512.Sum(q * Alternate512);
            pEven = 0.5 * (pSum + pAlt);
            pOdd = 0.5 * (pSum - pAlt);
            qEven = 0.5 * (qSum + qAlt);
            qOdd = 0.5 * (qSum - qAlt);
        }
        else if (Vector256.IsHardwareAccelerated && len >= 4)
        {
            var vr = Vector256.Create(ar);
            var vi = Vector256.Create(-ai) * Alternate256;
            var p = Vector256<double>.Zero;
            var q = Vector256<double>.Zero;
            for (; i <= len - 4; i += 4)
            {
                var cv = Vector256.LoadUnsafe(ref pc, (nuint)i);
                var cs = Swap(cv);
                var yv = Vector256.LoadUnsafe(ref py, (nuint)i);
                RealVectorOps.MulAdd(vr, cv, RealVectorOps.MulAdd(vi, cs, yv)).StoreUnsafe(ref py, (nuint)i);
                var xv = Vector256.LoadUnsafe(ref px, (nuint)i);
                p = RealVectorOps.MulAdd(cv, xv, p);
                q = RealVectorOps.MulAdd(cs, xv, q);
            }

            pEven = p.GetElement(0) + p.GetElement(2);
            pOdd = p.GetElement(1) + p.GetElement(3);
            qEven = q.GetElement(0) + q.GetElement(2);
            qOdd = q.GetElement(1) + q.GetElement(3);
        }
        else if (Vector128.IsHardwareAccelerated && len >= 2)
        {
            var vr = Vector128.Create(ar);
            var vi = Vector128.Create(-ai) * Alternate128;
            var p = Vector128<double>.Zero;
            var q = Vector128<double>.Zero;
            for (; i <= len - 2; i += 2)
            {
                var cv = Vector128.LoadUnsafe(ref pc, (nuint)i);
                var cs = Swap(cv);
                var yv = Vector128.LoadUnsafe(ref py, (nuint)i);
                RealVectorOps.MulAdd(vr, cv, RealVectorOps.MulAdd(vi, cs, yv)).StoreUnsafe(ref py, (nuint)i);
                var xv = Vector128.LoadUnsafe(ref px, (nuint)i);
                p = RealVectorOps.MulAdd(cv, xv, p);
                q = RealVectorOps.MulAdd(cs, xv, q);
            }

            pEven = p.GetElement(0);
            pOdd = p.GetElement(1);
            qEven = q.GetElement(0);
            qOdd = q.GetElement(1);
        }

        for (; i < len; i += 2)
        {
            double cr = Unsafe.Add(ref pc, i);
            double ci = Unsafe.Add(ref pc, i + 1);
            double xr = Unsafe.Add(ref px, i);
            double xi = Unsafe.Add(ref px, i + 1);
            Unsafe.Add(ref py, i) += (ar * cr) - (ai * ci);
            Unsafe.Add(ref py, i + 1) += (ar * ci) + (ai * cr);
            pEven += cr * xr;
            pOdd += ci * xi;
            qEven += ci * xr;
            qOdd += cr * xi;
        }

        // col·x = (cr·xr - ci·xi, cr·xi + ci·xr)、conj(col)·x = (cr·xr + ci·xi, cr·xi - ci·xr)
        return conjugateColumn
            ? new Complex(pEven + pOdd, qOdd - qEven)
            : new Complex(pEven - pOdd, qOdd + qEven);
    }

    // 2 列分の AxpyDot を融合し、x と y のメモリアクセスを共有する（zhemv 用）。
    internal static unsafe void AxpyDot2(
        int n, Complex ta, Complex* ca, Complex tb, Complex* cb, Complex* x, Complex* y, bool conjugateColumn, out Complex sa, out Complex sb)
    {
        double* pa = (double*)ca, pb = (double*)cb, px = (double*)x, py = (double*)y;
        int len = 2 * n;
        int i = 0;
        double paE = 0, paO = 0, qaE = 0, qaO = 0, pbE = 0, pbO = 0, qbE = 0, qbO = 0;
        if (Vector512.IsHardwareAccelerated && len >= 8)
        {
            var alt = Alternate512;
            var ar = Vector512.Create(ta.Real);
            var ai = Vector512.Create(-ta.Imaginary) * alt;
            var br = Vector512.Create(tb.Real);
            var bi = Vector512.Create(-tb.Imaginary) * alt;
            Vector512<double> p1 = default, q1 = default, p2 = default, q2 = default;
            for (; i <= len - 8; i += 8)
            {
                var va = Vector512.Load(pa + i);
                var sa1 = Swap(va);
                var vb = Vector512.Load(pb + i);
                var sb1 = Swap(vb);
                var yv = Vector512.Load(py + i);
                yv = Vector512.FusedMultiplyAdd(ar, va, Vector512.FusedMultiplyAdd(ai, sa1, yv));
                Vector512.FusedMultiplyAdd(br, vb, Vector512.FusedMultiplyAdd(bi, sb1, yv)).Store(py + i);
                var xv = Vector512.Load(px + i);
                p1 = Vector512.FusedMultiplyAdd(va, xv, p1);
                q1 = Vector512.FusedMultiplyAdd(sa1, xv, q1);
                p2 = Vector512.FusedMultiplyAdd(vb, xv, p2);
                q2 = Vector512.FusedMultiplyAdd(sb1, xv, q2);
            }

            // 偶数番目（実部）と奇数番目（虚部）の和に分ける
            (paE, paO) = Split(Vector512.Sum(p1), Vector512.Sum(p1 * alt));
            (qaE, qaO) = Split(Vector512.Sum(q1), Vector512.Sum(q1 * alt));
            (pbE, pbO) = Split(Vector512.Sum(p2), Vector512.Sum(p2 * alt));
            (qbE, qbO) = Split(Vector512.Sum(q2), Vector512.Sum(q2 * alt));
        }
        else if (Vector256.IsHardwareAccelerated && len >= 4)
        {
            var alt = Alternate256;
            var ar = Vector256.Create(ta.Real);
            var ai = Vector256.Create(-ta.Imaginary) * alt;
            var br = Vector256.Create(tb.Real);
            var bi = Vector256.Create(-tb.Imaginary) * alt;
            Vector256<double> p1 = default, q1 = default, p2 = default, q2 = default;
            for (; i <= len - 4; i += 4)
            {
                var va = Vector256.Load(pa + i);
                var sa1 = Swap(va);
                var vb = Vector256.Load(pb + i);
                var sb1 = Swap(vb);
                var yv = Vector256.Load(py + i);
                yv = RealVectorOps.MulAdd(ar, va, RealVectorOps.MulAdd(ai, sa1, yv));
                RealVectorOps.MulAdd(br, vb, RealVectorOps.MulAdd(bi, sb1, yv)).Store(py + i);
                var xv = Vector256.Load(px + i);
                p1 = RealVectorOps.MulAdd(va, xv, p1);
                q1 = RealVectorOps.MulAdd(sa1, xv, q1);
                p2 = RealVectorOps.MulAdd(vb, xv, p2);
                q2 = RealVectorOps.MulAdd(sb1, xv, q2);
            }

            (paE, paO) = Split(Vector256.Sum(p1), Vector256.Sum(p1 * alt));
            (qaE, qaO) = Split(Vector256.Sum(q1), Vector256.Sum(q1 * alt));
            (pbE, pbO) = Split(Vector256.Sum(p2), Vector256.Sum(p2 * alt));
            (qbE, qbO) = Split(Vector256.Sum(q2), Vector256.Sum(q2 * alt));
        }

        for (; i < len; i += 2)
        {
            double xr = px[i], xi = px[i + 1];
            double cr = pa[i], ci = pa[i + 1];
            py[i] += (ta.Real * cr) - (ta.Imaginary * ci);
            py[i + 1] += (ta.Real * ci) + (ta.Imaginary * cr);
            paE += cr * xr;
            paO += ci * xi;
            qaE += ci * xr;
            qaO += cr * xi;
            cr = pb[i];
            ci = pb[i + 1];
            py[i] += (tb.Real * cr) - (tb.Imaginary * ci);
            py[i + 1] += (tb.Real * ci) + (tb.Imaginary * cr);
            pbE += cr * xr;
            pbO += ci * xi;
            qbE += ci * xr;
            qbO += cr * xi;
        }

        sa = Assemble(paE, paO, qaE, qaO, conjugateColumn);
        sb = Assemble(pbE, pbO, qbE, qbO, conjugateColumn);

        static (double Even, double Odd) Split(double sum, double alternating) => (0.5 * (sum + alternating), 0.5 * (sum - alternating));

        // col·x = (cr·xr - ci·xi, cr·xi + ci·xr)、conj(col)·x = (cr·xr + ci·xi, cr·xi - ci·xr)
        static Complex Assemble(double pE, double pO, double qE, double qO, bool conj) =>
            conj ? new Complex(pE + pO, qO - qE) : new Complex(pE - pO, qO + qE);
    }

    // Σ (|Re x[i]| + |Im x[i]|)（BLAS dzasum 相当）。
    internal static double SumAbs(ReadOnlySpan<Complex> x) =>
        RealVectorOps.SumAbs(MemoryMarshal.CreateReadOnlySpan(ref AsDoubles(x), 2 * x.Length));

    // |Re| + |Im| が最大の最初のインデックス（空なら -1、NaN があれば最初の NaN）。
    internal static int IndexOfMaxAbs(ReadOnlySpan<Complex> x)
    {
        if (x.IsEmpty)
        {
            return -1;
        }

        int best = 0;
        double max = -1;
        for (int i = 0; i < x.Length; i++)
        {
            double v = Math.Abs(x[i].Real) + Math.Abs(x[i].Imaginary);
            if (double.IsNaN(v))
            {
                return i;
            }

            if (v > max)
            {
                max = v;
                best = i;
            }
        }

        return best;
    }

    // 実数スカラー倍（BLAS zdscal 相当）。
    internal static void Scale(double a, Span<Complex> x) =>
        RealVectorOps.Scale(a, MemoryMarshal.CreateSpan(ref Unsafe.As<Complex, double>(ref MemoryMarshal.GetReference(x)), 2 * x.Length));

    // 実数係数による平面回転（BLAS zdrot 相当）。
    internal static void Rotate(Span<Complex> x, Span<Complex> y, double c, double s) =>
        RealVectorOps.Rotate(
            MemoryMarshal.CreateSpan(ref Unsafe.As<Complex, double>(ref MemoryMarshal.GetReference(x)), 2 * x.Length),
            MemoryMarshal.CreateSpan(ref Unsafe.As<Complex, double>(ref MemoryMarshal.GetReference(y)), 2 * x.Length),
            c,
            s);

    // 4 列の内積 r_k = Σ op(A[i, k]) * x[i]（x の読み込みと実部・虚部スワップを 4 列で共有）。
    internal static unsafe void Dot4(int n, Complex* a, nint lda, Complex* x, bool conjugate, Span<Complex> result)
    {
        double* c0 = (double*)a;
        double* c1 = (double*)(a + lda);
        double* c2 = (double*)(a + (2 * lda));
        double* c3 = (double*)(a + (3 * lda));
        double* px = (double*)x;
        int len = 2 * n;
        int i = 0;
        Complex r0 = default, r1 = default, r2 = default, r3 = default;

        // p = (cr·xr, ci·xi)、q = (cr·xi, ci·xr) を貯め、最後に符号付きの和で実部・虚部にする:
        // col·x = (Σ p·(+,-), Σ q)、conj(col)·x = (Σ p, Σ q·(+,-))
        if (Vector512.IsHardwareAccelerated && len >= 8)
        {
            var sp = conjugate ? Vector512<double>.One : Alternate512;
            var sq = conjugate ? Alternate512 : Vector512<double>.One;
            var p0 = Vector512<double>.Zero;
            var p1 = Vector512<double>.Zero;
            var p2 = Vector512<double>.Zero;
            var p3 = Vector512<double>.Zero;
            var q0 = Vector512<double>.Zero;
            var q1 = Vector512<double>.Zero;
            var q2 = Vector512<double>.Zero;
            var q3 = Vector512<double>.Zero;
            for (; i <= len - 8; i += 8)
            {
                var xv = Vector512.Load(px + i);
                var xs = Swap(xv);
                var v0 = Vector512.Load(c0 + i);
                var v1 = Vector512.Load(c1 + i);
                var v2 = Vector512.Load(c2 + i);
                var v3 = Vector512.Load(c3 + i);
                p0 = Vector512.FusedMultiplyAdd(v0, xv, p0);
                q0 = Vector512.FusedMultiplyAdd(v0, xs, q0);
                p1 = Vector512.FusedMultiplyAdd(v1, xv, p1);
                q1 = Vector512.FusedMultiplyAdd(v1, xs, q1);
                p2 = Vector512.FusedMultiplyAdd(v2, xv, p2);
                q2 = Vector512.FusedMultiplyAdd(v2, xs, q2);
                p3 = Vector512.FusedMultiplyAdd(v3, xv, p3);
                q3 = Vector512.FusedMultiplyAdd(v3, xs, q3);
            }

            r0 = new Complex(Vector512.Sum(p0 * sp), Vector512.Sum(q0 * sq));
            r1 = new Complex(Vector512.Sum(p1 * sp), Vector512.Sum(q1 * sq));
            r2 = new Complex(Vector512.Sum(p2 * sp), Vector512.Sum(q2 * sq));
            r3 = new Complex(Vector512.Sum(p3 * sp), Vector512.Sum(q3 * sq));
        }
        else if (Vector256.IsHardwareAccelerated && len >= 4)
        {
            var sp = conjugate ? Vector256<double>.One : Alternate256;
            var sq = conjugate ? Alternate256 : Vector256<double>.One;
            var p0 = Vector256<double>.Zero;
            var p1 = Vector256<double>.Zero;
            var p2 = Vector256<double>.Zero;
            var p3 = Vector256<double>.Zero;
            var q0 = Vector256<double>.Zero;
            var q1 = Vector256<double>.Zero;
            var q2 = Vector256<double>.Zero;
            var q3 = Vector256<double>.Zero;
            for (; i <= len - 4; i += 4)
            {
                var xv = Vector256.Load(px + i);
                var xs = Swap(xv);
                var v0 = Vector256.Load(c0 + i);
                var v1 = Vector256.Load(c1 + i);
                var v2 = Vector256.Load(c2 + i);
                var v3 = Vector256.Load(c3 + i);
                p0 = RealVectorOps.MulAdd(v0, xv, p0);
                q0 = RealVectorOps.MulAdd(v0, xs, q0);
                p1 = RealVectorOps.MulAdd(v1, xv, p1);
                q1 = RealVectorOps.MulAdd(v1, xs, q1);
                p2 = RealVectorOps.MulAdd(v2, xv, p2);
                q2 = RealVectorOps.MulAdd(v2, xs, q2);
                p3 = RealVectorOps.MulAdd(v3, xv, p3);
                q3 = RealVectorOps.MulAdd(v3, xs, q3);
            }

            r0 = new Complex(Vector256.Sum(p0 * sp), Vector256.Sum(q0 * sq));
            r1 = new Complex(Vector256.Sum(p1 * sp), Vector256.Sum(q1 * sq));
            r2 = new Complex(Vector256.Sum(p2 * sp), Vector256.Sum(q2 * sq));
            r3 = new Complex(Vector256.Sum(p3 * sp), Vector256.Sum(q3 * sq));
        }

        // 共役なら列の虚部の符号を反転する
        double g = conjugate ? -1 : 1;
        for (; i < len; i += 2)
        {
            double xr = px[i];
            double xi = px[i + 1];
            r0 += new Complex((c0[i] * xr) - (g * c0[i + 1] * xi), (c0[i] * xi) + (g * c0[i + 1] * xr));
            r1 += new Complex((c1[i] * xr) - (g * c1[i + 1] * xi), (c1[i] * xi) + (g * c1[i + 1] * xr));
            r2 += new Complex((c2[i] * xr) - (g * c2[i + 1] * xi), (c2[i] * xi) + (g * c2[i + 1] * xr));
            r3 += new Complex((c3[i] * xr) - (g * c3[i + 1] * xi), (c3[i] * xi) + (g * c3[i + 1] * xr));
        }

        result[0] = r0;
        result[1] = r1;
        result[2] = r2;
        result[3] = r3;
    }

    // y += Σ_{k=0..3} t_k * A[:, k]（y の読み書きを 4 列で共有）。
    internal static unsafe void AddColumns4(int n, Complex* a, nint lda, ReadOnlySpan<Complex> t, Complex* y)
    {
        double* c0 = (double*)a;
        double* c1 = (double*)(a + lda);
        double* c2 = (double*)(a + (2 * lda));
        double* c3 = (double*)(a + (3 * lda));
        double* py = (double*)y;
        int len = 2 * n;
        int i = 0;
        if (Vector512.IsHardwareAccelerated && len >= 8)
        {
            var r0 = Vector512.Create(t[0].Real);
            var r1 = Vector512.Create(t[1].Real);
            var r2 = Vector512.Create(t[2].Real);
            var r3 = Vector512.Create(t[3].Real);
            var i0 = Vector512.Create(-t[0].Imaginary) * Alternate512;
            var i1 = Vector512.Create(-t[1].Imaginary) * Alternate512;
            var i2 = Vector512.Create(-t[2].Imaginary) * Alternate512;
            var i3 = Vector512.Create(-t[3].Imaginary) * Alternate512;
            for (; i <= len - 8; i += 8)
            {
                var acc = Vector512.Load(py + i);
                var v0 = Vector512.Load(c0 + i);
                var v1 = Vector512.Load(c1 + i);
                var v2 = Vector512.Load(c2 + i);
                var v3 = Vector512.Load(c3 + i);
                acc = Vector512.FusedMultiplyAdd(r0, v0, acc);
                acc = Vector512.FusedMultiplyAdd(i0, Swap(v0), acc);
                acc = Vector512.FusedMultiplyAdd(r1, v1, acc);
                acc = Vector512.FusedMultiplyAdd(i1, Swap(v1), acc);
                acc = Vector512.FusedMultiplyAdd(r2, v2, acc);
                acc = Vector512.FusedMultiplyAdd(i2, Swap(v2), acc);
                acc = Vector512.FusedMultiplyAdd(r3, v3, acc);
                acc = Vector512.FusedMultiplyAdd(i3, Swap(v3), acc);
                acc.Store(py + i);
            }
        }
        else if (Vector256.IsHardwareAccelerated && len >= 4)
        {
            var r0 = Vector256.Create(t[0].Real);
            var r1 = Vector256.Create(t[1].Real);
            var r2 = Vector256.Create(t[2].Real);
            var r3 = Vector256.Create(t[3].Real);
            var i0 = Vector256.Create(-t[0].Imaginary) * Alternate256;
            var i1 = Vector256.Create(-t[1].Imaginary) * Alternate256;
            var i2 = Vector256.Create(-t[2].Imaginary) * Alternate256;
            var i3 = Vector256.Create(-t[3].Imaginary) * Alternate256;
            for (; i <= len - 4; i += 4)
            {
                var acc = Vector256.Load(py + i);
                var v0 = Vector256.Load(c0 + i);
                var v1 = Vector256.Load(c1 + i);
                var v2 = Vector256.Load(c2 + i);
                var v3 = Vector256.Load(c3 + i);
                acc = RealVectorOps.MulAdd(r0, v0, acc);
                acc = RealVectorOps.MulAdd(i0, Swap(v0), acc);
                acc = RealVectorOps.MulAdd(r1, v1, acc);
                acc = RealVectorOps.MulAdd(i1, Swap(v1), acc);
                acc = RealVectorOps.MulAdd(r2, v2, acc);
                acc = RealVectorOps.MulAdd(i2, Swap(v2), acc);
                acc = RealVectorOps.MulAdd(r3, v3, acc);
                acc = RealVectorOps.MulAdd(i3, Swap(v3), acc);
                acc.Store(py + i);
            }
        }

        for (; i < len; i += 2)
        {
            double yr = py[i];
            double yi = py[i + 1];
            yr += (t[0].Real * c0[i]) - (t[0].Imaginary * c0[i + 1]);
            yi += (t[0].Real * c0[i + 1]) + (t[0].Imaginary * c0[i]);
            yr += (t[1].Real * c1[i]) - (t[1].Imaginary * c1[i + 1]);
            yi += (t[1].Real * c1[i + 1]) + (t[1].Imaginary * c1[i]);
            yr += (t[2].Real * c2[i]) - (t[2].Imaginary * c2[i + 1]);
            yi += (t[2].Real * c2[i + 1]) + (t[2].Imaginary * c2[i]);
            yr += (t[3].Real * c3[i]) - (t[3].Imaginary * c3[i + 1]);
            yi += (t[3].Real * c3[i + 1]) + (t[3].Imaginary * c3[i]);
            py[i] = yr;
            py[i + 1] = yi;
        }
    }
}
