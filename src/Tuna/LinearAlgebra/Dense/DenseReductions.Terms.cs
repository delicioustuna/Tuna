using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace Tuna.LinearAlgebra;

internal static partial class DenseReductions
{
    // ---------- 項と演算 ----------

    // 総和の index 番目の項（ベクトル版は index からの連続要素を返す）。
    private interface ITerm<T> where T : unmanaged
    {
        T Invoke(nint index);

        Vector128<T> Invoke128(nint index);

        Vector256<T> Invoke256(nint index);

        Vector512<T> Invoke512(nint index);
    }

    private readonly ref struct UnaryTerm<T, TKernel> : ITerm<T>
        where T : unmanaged, IFloatingPointIeee754<T>
        where TKernel : ITermKernel<T>
    {
        private readonly ref readonly T x;
        private readonly T parameter;

        public UnaryTerm(ref readonly T x, T parameter)
        {
            this.x = ref x;
            this.parameter = parameter;
        }

        public T Invoke(nint index) => TKernel.Invoke(Unsafe.Add(ref Unsafe.AsRef(in x), index), parameter);

        public Vector128<T> Invoke128(nint index) => TKernel.Invoke(Vector128.LoadUnsafe(in x, (nuint)index), Vector128.Create(parameter));

        public Vector256<T> Invoke256(nint index) => TKernel.Invoke(Vector256.LoadUnsafe(in x, (nuint)index), Vector256.Create(parameter));

        public Vector512<T> Invoke512(nint index) => TKernel.Invoke(Vector512.LoadUnsafe(in x, (nuint)index), Vector512.Create(parameter));
    }

    private readonly ref struct ProductTerm<T> : ITerm<T> where T : unmanaged, IFloatingPointIeee754<T>
    {
        private readonly ref readonly T x;
        private readonly ref readonly T y;

        public ProductTerm(ref readonly T x, ref readonly T y)
        {
            this.x = ref x;
            this.y = ref y;
        }

        public T Invoke(nint index) => Unsafe.Add(ref Unsafe.AsRef(in x), index) * Unsafe.Add(ref Unsafe.AsRef(in y), index);

        public Vector128<T> Invoke128(nint index) => Vector128.LoadUnsafe(in x, (nuint)index) * Vector128.LoadUnsafe(in y, (nuint)index);

        public Vector256<T> Invoke256(nint index) => Vector256.LoadUnsafe(in x, (nuint)index) * Vector256.LoadUnsafe(in y, (nuint)index);

        public Vector512<T> Invoke512(nint index) => Vector512.LoadUnsafe(in x, (nuint)index) * Vector512.LoadUnsafe(in y, (nuint)index);
    }

    // 総和の項 f(x, p)（スカラーとベクトルでビット一致すること）。
    private interface ITermKernel<T> where T : unmanaged
    {
        static abstract T Invoke(T x, T p);

        static abstract Vector128<T> Invoke(Vector128<T> x, Vector128<T> p);

        static abstract Vector256<T> Invoke(Vector256<T> x, Vector256<T> p);

        static abstract Vector512<T> Invoke(Vector512<T> x, Vector512<T> p);
    }

    private readonly struct ElementKernel<T> : ITermKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
    {
        public static T Invoke(T x, T p) => x;

        public static Vector128<T> Invoke(Vector128<T> x, Vector128<T> p) => x;

        public static Vector256<T> Invoke(Vector256<T> x, Vector256<T> p) => x;

        public static Vector512<T> Invoke(Vector512<T> x, Vector512<T> p) => x;
    }

    // |x|（符号ビットのみ消去するため NaN ペイロードを保持）。
    private readonly struct AbsoluteKernel<T> : ITermKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
    {
        public static T Invoke(T x, T p) => T.Abs(x);

        public static Vector128<T> Invoke(Vector128<T> x, Vector128<T> p) => Vector128.Abs(x);

        public static Vector256<T> Invoke(Vector256<T> x, Vector256<T> p) => Vector256.Abs(x);

        public static Vector512<T> Invoke(Vector512<T> x, Vector512<T> p) => Vector512.Abs(x);
    }

    private readonly struct IgnoreNaNKernel<T> : ITermKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
    {
        public static T Invoke(T x, T p) => T.IsNaN(x) ? T.Zero : x;

        public static Vector128<T> Invoke(Vector128<T> x, Vector128<T> p) => Vector128.ConditionalSelect(Vector128.Equals(x, x), x, Vector128<T>.Zero);

        public static Vector256<T> Invoke(Vector256<T> x, Vector256<T> p) => Vector256.ConditionalSelect(Vector256.Equals(x, x), x, Vector256<T>.Zero);

        public static Vector512<T> Invoke(Vector512<T> x, Vector512<T> p) => Vector512.ConditionalSelect(Vector512.Equals(x, x), x, Vector512<T>.Zero);
    }

    private readonly struct SquareKernel<T> : ITermKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
    {
        public static T Invoke(T x, T p) => x * x;

        public static Vector128<T> Invoke(Vector128<T> x, Vector128<T> p) => x * x;

        public static Vector256<T> Invoke(Vector256<T> x, Vector256<T> p) => x * x;

        public static Vector512<T> Invoke(Vector512<T> x, Vector512<T> p) => x * x;
    }

    private readonly struct DeviationKernel<T> : ITermKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
    {
        public static T Invoke(T x, T p) => x - p;

        public static Vector128<T> Invoke(Vector128<T> x, Vector128<T> p) => x - p;

        public static Vector256<T> Invoke(Vector256<T> x, Vector256<T> p) => x - p;

        public static Vector512<T> Invoke(Vector512<T> x, Vector512<T> p) => x - p;
    }

    private readonly struct SquaredDeviationKernel<T> : ITermKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
    {
        public static T Invoke(T x, T p)
        {
            T d = x - p;
            return d * d;
        }

        public static Vector128<T> Invoke(Vector128<T> x, Vector128<T> p)
        {
            var d = x - p;
            return d * d;
        }

        public static Vector256<T> Invoke(Vector256<T> x, Vector256<T> p)
        {
            var d = x - p;
            return d * d;
        }

        public static Vector512<T> Invoke(Vector512<T> x, Vector512<T> p)
        {
            var d = x - p;
            return d * d;
        }
    }

    private readonly struct ScaledKernel<T> : ITermKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
    {
        public static T Invoke(T x, T p) => x * p;

        public static Vector128<T> Invoke(Vector128<T> x, Vector128<T> p) => x * p;

        public static Vector256<T> Invoke(Vector256<T> x, Vector256<T> p) => x * p;

        public static Vector512<T> Invoke(Vector512<T> x, Vector512<T> p) => x * p;
    }

    private readonly struct ScaledSquareKernel<T> : ITermKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
    {
        public static T Invoke(T x, T p)
        {
            T s = x * p;
            return s * s;
        }

        public static Vector128<T> Invoke(Vector128<T> x, Vector128<T> p)
        {
            var s = x * p;
            return s * s;
        }

        public static Vector256<T> Invoke(Vector256<T> x, Vector256<T> p)
        {
            var s = x * p;
            return s * s;
        }

        public static Vector512<T> Invoke(Vector512<T> x, Vector512<T> p)
        {
            var s = x * p;
            return s * s;
        }
    }

    // 順序不変な畳み込みステップ f(accumulated, x)（スカラーとベクトルでビット一致すること）。
    private interface IFoldKernel<T> where T : unmanaged
    {
        static abstract T Invoke(T accumulated, T x);

        static abstract Vector128<T> Invoke(Vector128<T> accumulated, Vector128<T> x);

        static abstract Vector256<T> Invoke(Vector256<T> accumulated, Vector256<T> x);

        static abstract Vector512<T> Invoke(Vector512<T> accumulated, Vector512<T> x);
    }

    // T.Max（NaN 伝播、+0 > -0。Vector.Max と同一規則）。
    private readonly struct MaximumKernel<T> : IFoldKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
    {
        public static T Invoke(T accumulated, T x) => T.Max(accumulated, x);

        public static Vector128<T> Invoke(Vector128<T> accumulated, Vector128<T> x) => Vector128.Max(accumulated, x);

        public static Vector256<T> Invoke(Vector256<T> accumulated, Vector256<T> x) => Vector256.Max(accumulated, x);

        public static Vector512<T> Invoke(Vector512<T> accumulated, Vector512<T> x) => Vector512.Max(accumulated, x);
    }

    private readonly struct MinimumKernel<T> : IFoldKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
    {
        public static T Invoke(T accumulated, T x) => T.Min(accumulated, x);

        public static Vector128<T> Invoke(Vector128<T> accumulated, Vector128<T> x) => Vector128.Min(accumulated, x);

        public static Vector256<T> Invoke(Vector256<T> accumulated, Vector256<T> x) => Vector256.Min(accumulated, x);

        public static Vector512<T> Invoke(Vector512<T> accumulated, Vector512<T> x) => Vector512.Min(accumulated, x);
    }

    private readonly struct MaximumNumberKernel<T> : IFoldKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
    {
        public static T Invoke(T accumulated, T x) => T.MaxNumber(accumulated, x);

        public static Vector128<T> Invoke(Vector128<T> accumulated, Vector128<T> x) => Vector128.MaxNumber(accumulated, x);

        public static Vector256<T> Invoke(Vector256<T> accumulated, Vector256<T> x) => Vector256.MaxNumber(accumulated, x);

        public static Vector512<T> Invoke(Vector512<T> accumulated, Vector512<T> x) => Vector512.MaxNumber(accumulated, x);
    }

    private readonly struct MinimumNumberKernel<T> : IFoldKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
    {
        public static T Invoke(T accumulated, T x) => T.MinNumber(accumulated, x);

        public static Vector128<T> Invoke(Vector128<T> accumulated, Vector128<T> x) => Vector128.MinNumber(accumulated, x);

        public static Vector256<T> Invoke(Vector256<T> accumulated, Vector256<T> x) => Vector256.MinNumber(accumulated, x);

        public static Vector512<T> Invoke(Vector512<T> accumulated, Vector512<T> x) => Vector512.MinNumber(accumulated, x);
    }

    private readonly struct NormInfinityKernel<T> : IFoldKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
    {
        public static T Invoke(T accumulated, T x) => T.Max(accumulated, T.Abs(x));

        public static Vector128<T> Invoke(Vector128<T> accumulated, Vector128<T> x) => Vector128.Max(accumulated, Vector128.Abs(x));

        public static Vector256<T> Invoke(Vector256<T> accumulated, Vector256<T> x) => Vector256.Max(accumulated, Vector256.Abs(x));

        public static Vector512<T> Invoke(Vector512<T> accumulated, Vector512<T> x) => Vector512.Max(accumulated, Vector512.Abs(x));
    }

    // StridedKernels.LargestMagnitude と同じ規則（NaN は qNaN 化せず符号ビットのみ消去して残す）。
    private readonly struct LargestMagnitudeKernel<T> : IFoldKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
    {
        public static T Invoke(T accumulated, T x)
        {
            T magnitude = T.Abs(x);
            return T.IsNaN(magnitude) || magnitude > accumulated ? magnitude : accumulated;
        }

        public static Vector128<T> Invoke(Vector128<T> accumulated, Vector128<T> x)
        {
            var magnitude = Vector128.Abs(x);
            return Vector128.ConditionalSelect(Vector128.IsNaN(magnitude) | Vector128.GreaterThan(magnitude, accumulated), magnitude, accumulated);
        }

        public static Vector256<T> Invoke(Vector256<T> accumulated, Vector256<T> x)
        {
            var magnitude = Vector256.Abs(x);
            return Vector256.ConditionalSelect(Vector256.IsNaN(magnitude) | Vector256.GreaterThan(magnitude, accumulated), magnitude, accumulated);
        }

        public static Vector512<T> Invoke(Vector512<T> accumulated, Vector512<T> x)
        {
            var magnitude = Vector512.Abs(x);
            return Vector512.ConditionalSelect(Vector512.IsNaN(magnitude) | Vector512.GreaterThan(magnitude, accumulated), magnitude, accumulated);
        }
    }
}
