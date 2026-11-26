using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace Tuna.NeuralNetwork;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む。
using Tuna.LinearAlgebra;

internal static unsafe class Kernels
{
    // 中心化には Chan, Golub, LeVeque (1983, The American Statistician 37(3)) の shifted-data アルゴリズムを用いる。
    // 平均の近似 K = Σx/n からの残差で δ = Σ(x − K)/n と分散 (Σ(x − K)² − δ·Σ(x − K))/n を計算し、
    // 偏差を (x − K) − δ として平均 K + δ の丸め誤差を回避する。縮約幅は決定的な加算順序のため Vector256 に固定する。
    // stats が null でなければ、逆伝播で正規化した値を同じ式で再計算するため (K, δ, 1/√(σ² + ε)) を書き込む。
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void NormalizeColumn<T>(T* x, T* y, int n, T* scale, T* shift, T epsilon, bool center, T* stats = null)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (n == 0)
        {
            return;
        }

        int w = Vector256<T>.Count;
        T count = T.CreateTruncating(n);

        // 中心化する場合は偏差 (x − origin) − offset、しない場合は x そのもの（origin = offset = 0）
        T origin = T.Zero;
        int i;
        if (center)
        {
            T sum = T.Zero;
            i = 0;
            if (n >= w)
            {
                var acc = Vector256<T>.Zero;
                for (; i <= n - w; i += w)
                {
                    acc += Vector256.Load(x + i);
                }

                sum = Vector256.Sum(acc);
            }

            for (; i < n; i++)
            {
                sum += x[i];
            }

            origin = sum / count;
        }

        // origin からの差の和と二乗和
        T squares = T.Zero, deviations = T.Zero;
        var o = Vector256.Create(origin);
        i = 0;
        if (n >= w)
        {
            var acc = Vector256<T>.Zero;
            var total = Vector256<T>.Zero;
            for (; i <= n - w; i += w)
            {
                var d = Vector256.Load(x + i) - o;
                acc = Simd.Fma(d, d, acc);
                total += d;
            }

            squares = Vector256.Sum(acc);
            deviations = Vector256.Sum(total);
        }

        for (; i < n; i++)
        {
            T d = x[i] - origin;
            squares = T.FusedMultiplyAdd(d, d, squares);
            deviations += d;
        }

        T offset = center ? deviations / count : T.Zero;
        T variance = center ? T.Max(T.Zero, (squares - offset * deviations) / count) : squares / count;
        var m = Vector256.Create(offset);
        T inverse = T.One / T.Sqrt(variance + epsilon);
        if (stats != null)
        {
            stats[0] = origin;
            stats[1] = offset;
            stats[2] = inverse;
        }

        var r = Vector256.Create(inverse);
        i = 0;
        for (; i <= n - w; i += w)
        {
            var v = ((Vector256.Load(x + i) - o) - m) * r;
            if (scale != null)
            {
                v = shift != null ? Simd.Fma(v, Vector256.Load(scale + i), Vector256.Load(shift + i)) : v * Vector256.Load(scale + i);
            }
            else if (shift != null)
            {
                v += Vector256.Load(shift + i);
            }

            v.Store(y + i);
        }

        for (; i < n; i++)
        {
            T v = ((x[i] - origin) - offset) * inverse;
            if (scale != null)
            {
                v = shift != null ? T.FusedMultiplyAdd(v, scale[i], shift[i]) : v * scale[i];
            }
            else if (shift != null)
            {
                v += shift[i];
            }

            y[i] = v;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void PreluColumn<T>(T* x, T* y, int n, T* slopes, T slope)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        int i = 0;
        if (Simd.Wide)
        {
            var s = Vector512.Create(slope);
            for (; i <= n - Vector512<T>.Count; i += Vector512<T>.Count)
            {
                var v = Vector512.Load(x + i);
                var a = slopes != null ? Vector512.Load(slopes + i) : s;
                Vector512.ConditionalSelect(Vector512.LessThan(v, Vector512<T>.Zero), v * a, v).Store(y + i);
            }
        }

        {
            var s = Vector256.Create(slope);
            for (; i <= n - Vector256<T>.Count; i += Vector256<T>.Count)
            {
                var v = Vector256.Load(x + i);
                var a = slopes != null ? Vector256.Load(slopes + i) : s;
                Vector256.ConditionalSelect(Vector256.LessThan(v, Vector256<T>.Zero), v * a, v).Store(y + i);
            }
        }

        for (; i < n; i++)
        {
            T v = x[i];
            y[i] = v < T.Zero ? v * (slopes != null ? slopes[i] : slope) : v;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void AddColumn<T>(T* y, T* b, int n) where T : unmanaged, IFloatingPointIeee754<T>
    {
        int i = 0;
        if (Simd.Wide)
        {
            for (; i <= n - Vector512<T>.Count; i += Vector512<T>.Count)
            {
                (Vector512.Load(y + i) + Vector512.Load(b + i)).Store(y + i);
            }
        }

        for (; i <= n - Vector256<T>.Count; i += Vector256<T>.Count)
        {
            (Vector256.Load(y + i) + Vector256.Load(b + i)).Store(y + i);
        }

        for (; i < n; i++)
        {
            y[i] += b[i];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void MultiplyColumn<T>(T* y, T* z, int n) where T : unmanaged, IFloatingPointIeee754<T>
    {
        int i = 0;
        if (Simd.Wide)
        {
            for (; i <= n - Vector512<T>.Count; i += Vector512<T>.Count)
            {
                (Vector512.Load(y + i) * Vector512.Load(z + i)).Store(y + i);
            }
        }

        for (; i <= n - Vector256<T>.Count; i += Vector256<T>.Count)
        {
            (Vector256.Load(y + i) * Vector256.Load(z + i)).Store(y + i);
        }

        for (; i < n; i++)
        {
            y[i] *= z[i];
        }
    }

    // チャネルごとの因果畳み込みの1位置: y_d = bias_d + Σ_k w(d, k)·taps[k]_d（taps[k] == null は 0 としてスキップ）
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void DepthwiseTaps<T>(T** taps, int k, T* weight, int ldw, T* bias, T* y, int n)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        int i = 0;
        if (Simd.Wide)
        {
            for (; i <= n - Vector512<T>.Count; i += Vector512<T>.Count)
            {
                var acc = bias != null ? Vector512.Load(bias + i) : Vector512<T>.Zero;
                for (int j = 0; j < k; j++)
                {
                    if (taps[j] != null)
                    {
                        acc = Simd.Fma(Vector512.Load(weight + (nint)j * ldw + i), Vector512.Load(taps[j] + i), acc);
                    }
                }

                acc.Store(y + i);
            }
        }

        for (; i <= n - Vector256<T>.Count; i += Vector256<T>.Count)
        {
            var acc = bias != null ? Vector256.Load(bias + i) : Vector256<T>.Zero;
            for (int j = 0; j < k; j++)
            {
                if (taps[j] != null)
                {
                    acc = Simd.Fma(Vector256.Load(weight + (nint)j * ldw + i), Vector256.Load(taps[j] + i), acc);
                }
            }

            acc.Store(y + i);
        }

        for (; i < n; i++)
        {
            T acc = bias != null ? bias[i] : T.Zero;
            for (int j = 0; j < k; j++)
            {
                if (taps[j] != null)
                {
                    acc = T.FusedMultiplyAdd(weight[(nint)j * ldw + i], taps[j][i], acc);
                }
            }

            y[i] = acc;
        }
    }

    // 選択的状態空間モデルの1ステップ（チャネル d ∈ [d0, d1)）。work は長さ N·(d1 − d0) の作業領域。
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void ScanStep<T>(T* a, int lda, T* delta, T* x, T* b, T* c, T* skip, T* h, int ldh, T* y, T* work,
        int d0, int d1, int n, bool approximate)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        int len = d1 - d0;

        // work[k·len + (d − d0)] = exp(Δ_d·A(d, k))
        for (int k = 0; k < n; k++)
        {
            T* ak = a + (nint)k * lda;
            T* row = work + (nint)k * len - d0;
            int d = d0;
            if (Simd.Wide)
            {
                for (; d <= d1 - Vector512<T>.Count; d += Vector512<T>.Count)
                {
                    (Vector512.Load(delta + d) * Vector512.Load(ak + d)).Store(row + d);
                }
            }

            for (; d <= d1 - Vector256<T>.Count; d += Vector256<T>.Count)
            {
                (Vector256.Load(delta + d) * Vector256.Load(ak + d)).Store(row + d);
            }

            for (; d < d1; d++)
            {
                row[d] = delta[d] * ak[d];
            }
        }

        XMath.InPlace.Exp(new VectorSpan<T>(new Span<T>(work, n * len)), approximate);

        int i = d0;
        if (Simd.Wide)
        {
            for (; i <= d1 - Vector512<T>.Count; i += Vector512<T>.Count)
            {
                var xv = Vector512.Load(x + i);
                var dx = Vector512.Load(delta + i) * xv;
                var acc = skip != null ? Vector512.Load(skip + i) * xv : Vector512<T>.Zero;
                for (int k = 0; k < n; k++)
                {
                    T* hk = h + (nint)k * ldh + i;
                    var hv = Simd.Fma(Vector512.Load(hk), Vector512.Load(work + (nint)k * len + (i - d0)), dx * Vector512.Create(b[k]));
                    hv.Store(hk);
                    acc = Simd.Fma(hv, Vector512.Create(c[k]), acc);
                }

                acc.Store(y + i);
            }
        }

        for (; i <= d1 - Vector256<T>.Count; i += Vector256<T>.Count)
        {
            var xv = Vector256.Load(x + i);
            var dx = Vector256.Load(delta + i) * xv;
            var acc = skip != null ? Vector256.Load(skip + i) * xv : Vector256<T>.Zero;
            for (int k = 0; k < n; k++)
            {
                T* hk = h + (nint)k * ldh + i;
                var hv = Simd.Fma(Vector256.Load(hk), Vector256.Load(work + (nint)k * len + (i - d0)), dx * Vector256.Create(b[k]));
                hv.Store(hk);
                acc = Simd.Fma(hv, Vector256.Create(c[k]), acc);
            }

            acc.Store(y + i);
        }

        for (; i < d1; i++)
        {
            T xv = x[i];
            T dx = delta[i] * xv;
            T acc = skip != null ? skip[i] * xv : T.Zero;
            for (int k = 0; k < n; k++)
            {
                T* hk = h + (nint)k * ldh + i;
                T hv = T.FusedMultiplyAdd(*hk, work[(nint)k * len + (i - d0)], dx * b[k]);
                *hk = hv;
                acc = T.FusedMultiplyAdd(hv, c[k], acc);
            }

            y[i] = acc;
        }
    }
}
