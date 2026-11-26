using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Tuna.LinearAlgebra;

internal static partial class DenseReductions
{
    // ---------- 和 ----------

    private static bool TrySumReal<T, TOperator>(ref readonly T x, int length, T parameter, out T sum) where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (typeof(TOperator) == typeof(FirstOperator<T>))
        {
            sum = SumUnary<T, ElementKernel<T>>(in x, length, parameter);
        }
        else if (typeof(TOperator) == typeof(AbsoluteOperator<T>))
        {
            sum = SumUnary<T, AbsoluteKernel<T>>(in x, length, parameter);
        }
        else if (typeof(TOperator) == typeof(SquareOperator<T>))
        {
            sum = SumUnary<T, SquareKernel<T>>(in x, length, parameter);
        }
        else if (typeof(TOperator) == typeof(IgnoreNaNOperator<T>))
        {
            sum = SumUnary<T, IgnoreNaNKernel<T>>(in x, length, parameter);
        }
        else if (typeof(TOperator) == typeof(DeviationOperator<T>))
        {
            sum = SumUnary<T, DeviationKernel<T>>(in x, length, parameter);
        }
        else if (typeof(TOperator) == typeof(SquaredDeviationOperator<T>))
        {
            sum = SumUnary<T, SquaredDeviationKernel<T>>(in x, length, parameter);
        }
        else if (typeof(TOperator) == typeof(ScaledOperator<T>))
        {
            sum = SumUnary<T, ScaledKernel<T>>(in x, length, parameter);
        }
        else if (typeof(TOperator) == typeof(ScaledSquareByFactorOperator<T>))
        {
            sum = SumUnary<T, ScaledSquareKernel<T>>(in x, length, parameter);
        }
        else
        {
            sum = T.Zero;
            return false;
        }

        return true;
    }

    [SkipLocalsInit]
    private static T Sum<T, TTerm>(int length, TTerm term)
        where T : unmanaged, IFloatingPointIeee754<T>
        where TTerm : ITerm<T>, allows ref struct
    {
        var tree = new SumTree<T>(stackalloc T[SumTree<T>.Capacity]);
        AddBlocks(ref tree, length, term);
        return tree.Result();
    }

    // 項 0..length-1 をブロック単位で加算し、各ブロック和を順に tree へ渡す（先頭はブロック境界）。
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void AddBlocks<T, TTerm>(ref SumTree<T> tree, int length, scoped TTerm term)
        where T : unmanaged, IFloatingPointIeee754<T>
        where TTerm : ITerm<T>, allows ref struct
    {
        nint n = length, i = 0;
        if (typeof(T) == typeof(double))
        {
            if (Vector512.IsHardwareAccelerated && Avx512F.IsSupported)
            {
                // 2 組の 8 ブロックを交互に足して、加算の待ち時間を隠す
                for (; n - i >= 16 * Block; i += 16 * Block)
                {
                    var first = Vector512<double>.Zero;
                    var second = Vector512<double>.Zero;
                    for (nint c = 0; c < Block; c += 8)
                    {
                        first = AddTransposed8<T, TTerm>(term, i + c, first);
                        second = AddTransposed8<T, TTerm>(term, i + 8 * Block + c, second);
                    }

                    tree.Push(first.As<double, T>());
                    tree.Push(second.As<double, T>());
                }

                for (; n - i >= 8 * Block; i += 8 * Block)
                {
                    var sums = Vector512<double>.Zero;
                    for (nint c = 0; c < Block; c += 8)
                    {
                        sums = AddTransposed8<T, TTerm>(term, i + c, sums);
                    }

                    tree.Push(sums.As<double, T>());
                }
            }
            else if (Avx.IsSupported)
            {
                for (; n - i >= 8 * Block; i += 8 * Block)
                {
                    var first = Vector256<double>.Zero;
                    var second = Vector256<double>.Zero;
                    for (nint c = 0; c < Block; c += 4)
                    {
                        first = AddTransposed4<T, TTerm>(term, i + c, first);
                        second = AddTransposed4<T, TTerm>(term, i + 4 * Block + c, second);
                    }

                    tree.Push(first.As<double, T>());
                    tree.Push(second.As<double, T>());
                }

                for (; n - i >= 4 * Block; i += 4 * Block)
                {
                    var sums = Vector256<double>.Zero;
                    for (nint c = 0; c < Block; c += 4)
                    {
                        sums = AddTransposed4<T, TTerm>(term, i + c, sums);
                    }

                    tree.Push(sums.As<double, T>());
                }
            }
            else if (Sse2.IsSupported)
            {
                for (; n - i >= 4 * Block; i += 4 * Block)
                {
                    var first = Vector128<double>.Zero;
                    var second = Vector128<double>.Zero;
                    for (nint c = 0; c < Block; c += 2)
                    {
                        first = AddTransposed2<T, TTerm>(term, i + c, first);
                        second = AddTransposed2<T, TTerm>(term, i + 2 * Block + c, second);
                    }

                    tree.Push(first.As<double, T>());
                    tree.Push(second.As<double, T>());
                }
            }
        }
        else if (typeof(T) == typeof(float))
        {
            if (Avx.IsSupported)
            {
                for (; n - i >= 16 * Block; i += 16 * Block)
                {
                    var first = Vector256<float>.Zero;
                    var second = Vector256<float>.Zero;
                    for (nint c = 0; c < Block; c += 8)
                    {
                        first = AddTransposed8Single<T, TTerm>(term, i + c, first);
                        second = AddTransposed8Single<T, TTerm>(term, i + 8 * Block + c, second);
                    }

                    tree.Push(first.As<float, T>());
                    tree.Push(second.As<float, T>());
                }

                for (; n - i >= 8 * Block; i += 8 * Block)
                {
                    var sums = Vector256<float>.Zero;
                    for (nint c = 0; c < Block; c += 8)
                    {
                        sums = AddTransposed8Single<T, TTerm>(term, i + c, sums);
                    }

                    tree.Push(sums.As<float, T>());
                }
            }
            else if (Sse.IsSupported)
            {
                for (; n - i >= 8 * Block; i += 8 * Block)
                {
                    var first = Vector128<float>.Zero;
                    var second = Vector128<float>.Zero;
                    for (nint c = 0; c < Block; c += 4)
                    {
                        first = AddTransposed4Single<T, TTerm>(term, i + c, first);
                        second = AddTransposed4Single<T, TTerm>(term, i + 4 * Block + c, second);
                    }

                    tree.Push(first.As<float, T>());
                    tree.Push(second.As<float, T>());
                }
            }
        }

        // 残りのブロック。4 つのブロックを交互に足す（命令の並列性のため。各ブロックの中の順序は変わらない）
        for (; n - i >= 4 * Block; i += 4 * Block)
        {
            T s0 = T.Zero, s1 = T.Zero, s2 = T.Zero, s3 = T.Zero;
            for (nint k = i; k < i + Block; k++)
            {
                s0 += term.Invoke(k);
                s1 += term.Invoke(k + Block);
                s2 += term.Invoke(k + 2 * Block);
                s3 += term.Invoke(k + 3 * Block);
            }

            tree.Push(s0);
            tree.Push(s1);
            tree.Push(s2);
            tree.Push(s3);
        }

        while (i < n)
        {
            nint end = n - i <= Block ? n : i + Block;
            T sum = T.Zero;
            for (; i < end; i++)
            {
                sum += term.Invoke(i);
            }

            tree.Push(sum);
        }
    }

    // 8 つのブロック（先頭は offset、offset + Block、…）から 8 要素ずつ読み、8×8 を転置して、各ブロックの i 番目の項を i の順に sums に足す
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<double> AddTransposed8<T, TTerm>(TTerm term, nint offset, Vector512<double> sums)
        where T : unmanaged
        where TTerm : ITerm<T>, allows ref struct
    {
        var r0 = term.Invoke512(offset).AsDouble();
        var r1 = term.Invoke512(offset + Block).AsDouble();
        var r2 = term.Invoke512(offset + 2 * Block).AsDouble();
        var r3 = term.Invoke512(offset + 3 * Block).AsDouble();
        var r4 = term.Invoke512(offset + 4 * Block).AsDouble();
        var r5 = term.Invoke512(offset + 5 * Block).AsDouble();
        var r6 = term.Invoke512(offset + 6 * Block).AsDouble();
        var r7 = term.Invoke512(offset + 7 * Block).AsDouble();

        // 隣り合う2行の偶数番目・奇数番目
        var p0 = Avx512F.UnpackLow(r0, r1);
        var p1 = Avx512F.UnpackHigh(r0, r1);
        var p2 = Avx512F.UnpackLow(r2, r3);
        var p3 = Avx512F.UnpackHigh(r2, r3);
        var p4 = Avx512F.UnpackLow(r4, r5);
        var p5 = Avx512F.UnpackHigh(r4, r5);
        var p6 = Avx512F.UnpackLow(r6, r7);
        var p7 = Avx512F.UnpackHigh(r6, r7);

        // 4行ずつ: q0 は要素 0・4、q1 は 1・5、q2 は 2・6、q3 は 3・7（q4〜q7 は後半の4行）
        var low = Vector512.Create(0L, 1, 8, 9, 4, 5, 12, 13);
        var high = Vector512.Create(2L, 3, 10, 11, 6, 7, 14, 15);
        var q0 = Avx512F.PermuteVar8x64x2(p0, low, p2);
        var q1 = Avx512F.PermuteVar8x64x2(p1, low, p3);
        var q2 = Avx512F.PermuteVar8x64x2(p0, high, p2);
        var q3 = Avx512F.PermuteVar8x64x2(p1, high, p3);
        var q4 = Avx512F.PermuteVar8x64x2(p4, low, p6);
        var q5 = Avx512F.PermuteVar8x64x2(p5, low, p7);
        var q6 = Avx512F.PermuteVar8x64x2(p4, high, p6);
        var q7 = Avx512F.PermuteVar8x64x2(p5, high, p7);

        // 8行: 要素 0〜3 は前半、4〜7 は後半の 256 ビットから
        var first = Vector512.Create(0L, 1, 2, 3, 8, 9, 10, 11);
        var second = Vector512.Create(4L, 5, 6, 7, 12, 13, 14, 15);
        sums = Avx512F.Add(sums, Avx512F.PermuteVar8x64x2(q0, first, q4));
        sums = Avx512F.Add(sums, Avx512F.PermuteVar8x64x2(q1, first, q5));
        sums = Avx512F.Add(sums, Avx512F.PermuteVar8x64x2(q2, first, q6));
        sums = Avx512F.Add(sums, Avx512F.PermuteVar8x64x2(q3, first, q7));
        sums = Avx512F.Add(sums, Avx512F.PermuteVar8x64x2(q0, second, q4));
        sums = Avx512F.Add(sums, Avx512F.PermuteVar8x64x2(q1, second, q5));
        sums = Avx512F.Add(sums, Avx512F.PermuteVar8x64x2(q2, second, q6));
        return Avx512F.Add(sums, Avx512F.PermuteVar8x64x2(q3, second, q7));
    }

    // 4 つのブロックから 4 要素ずつ読み、4×4 を転置して足す（double、AVX）
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<double> AddTransposed4<T, TTerm>(TTerm term, nint offset, Vector256<double> sums)
        where T : unmanaged
        where TTerm : ITerm<T>, allows ref struct
    {
        var r0 = term.Invoke256(offset).AsDouble();
        var r1 = term.Invoke256(offset + Block).AsDouble();
        var r2 = term.Invoke256(offset + 2 * Block).AsDouble();
        var r3 = term.Invoke256(offset + 3 * Block).AsDouble();

        // t0 = [r0₀ r1₀ r0₂ r1₂]、t1 = [r0₁ r1₁ r0₃ r1₃]（t2・t3 は r2・r3）
        var t0 = Avx.UnpackLow(r0, r1);
        var t1 = Avx.UnpackHigh(r0, r1);
        var t2 = Avx.UnpackLow(r2, r3);
        var t3 = Avx.UnpackHigh(r2, r3);
        sums = Avx.Add(sums, Avx.Permute2x128(t0, t2, 0x20));
        sums = Avx.Add(sums, Avx.Permute2x128(t1, t3, 0x20));
        sums = Avx.Add(sums, Avx.Permute2x128(t0, t2, 0x31));
        return Avx.Add(sums, Avx.Permute2x128(t1, t3, 0x31));
    }

    // 2 つのブロックから 2 要素ずつ読み、2×2 を転置して足す（double、SSE2）
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<double> AddTransposed2<T, TTerm>(TTerm term, nint offset, Vector128<double> sums)
        where T : unmanaged
        where TTerm : ITerm<T>, allows ref struct
    {
        var r0 = term.Invoke128(offset).AsDouble();
        var r1 = term.Invoke128(offset + Block).AsDouble();
        sums = Sse2.Add(sums, Sse2.UnpackLow(r0, r1));
        return Sse2.Add(sums, Sse2.UnpackHigh(r0, r1));
    }

    // 8 つのブロックから 8 要素ずつ読み、8×8 を転置して足す（float、AVX）
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> AddTransposed8Single<T, TTerm>(TTerm term, nint offset, Vector256<float> sums)
        where T : unmanaged
        where TTerm : ITerm<T>, allows ref struct
    {
        var r0 = term.Invoke256(offset).AsSingle();
        var r1 = term.Invoke256(offset + Block).AsSingle();
        var r2 = term.Invoke256(offset + 2 * Block).AsSingle();
        var r3 = term.Invoke256(offset + 3 * Block).AsSingle();
        var r4 = term.Invoke256(offset + 4 * Block).AsSingle();
        var r5 = term.Invoke256(offset + 5 * Block).AsSingle();
        var r6 = term.Invoke256(offset + 6 * Block).AsSingle();
        var r7 = term.Invoke256(offset + 7 * Block).AsSingle();

        // 128 ビットごとに: t0 = [r0₀ r1₀ r0₁ r1₁ | r0₄ r1₄ r0₅ r1₅]、t1 = [r0₂ r1₂ r0₃ r1₃ | r0₆ r1₆ r0₇ r1₇]
        var t0 = Avx.UnpackLow(r0, r1);
        var t1 = Avx.UnpackHigh(r0, r1);
        var t2 = Avx.UnpackLow(r2, r3);
        var t3 = Avx.UnpackHigh(r2, r3);
        var t4 = Avx.UnpackLow(r4, r5);
        var t5 = Avx.UnpackHigh(r4, r5);
        var t6 = Avx.UnpackLow(r6, r7);
        var t7 = Avx.UnpackHigh(r6, r7);

        // u0 = [r0₀ r1₀ r2₀ r3₀ | r0₄ r1₄ r2₄ r3₄]、u1 は要素 1・5、u2 は 2・6、u3 は 3・7（u4〜u7 は後半の4行）
        var u0 = Avx.Shuffle(t0, t2, 0x44);
        var u1 = Avx.Shuffle(t0, t2, 0xEE);
        var u2 = Avx.Shuffle(t1, t3, 0x44);
        var u3 = Avx.Shuffle(t1, t3, 0xEE);
        var u4 = Avx.Shuffle(t4, t6, 0x44);
        var u5 = Avx.Shuffle(t4, t6, 0xEE);
        var u6 = Avx.Shuffle(t5, t7, 0x44);
        var u7 = Avx.Shuffle(t5, t7, 0xEE);

        sums = Avx.Add(sums, Avx.Permute2x128(u0, u4, 0x20));
        sums = Avx.Add(sums, Avx.Permute2x128(u1, u5, 0x20));
        sums = Avx.Add(sums, Avx.Permute2x128(u2, u6, 0x20));
        sums = Avx.Add(sums, Avx.Permute2x128(u3, u7, 0x20));
        sums = Avx.Add(sums, Avx.Permute2x128(u0, u4, 0x31));
        sums = Avx.Add(sums, Avx.Permute2x128(u1, u5, 0x31));
        sums = Avx.Add(sums, Avx.Permute2x128(u2, u6, 0x31));
        return Avx.Add(sums, Avx.Permute2x128(u3, u7, 0x31));
    }

    // 4 つのブロックから 4 要素ずつ読み、4×4 を転置して足す（float、SSE）
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> AddTransposed4Single<T, TTerm>(TTerm term, nint offset, Vector128<float> sums)
        where T : unmanaged
        where TTerm : ITerm<T>, allows ref struct
    {
        var r0 = term.Invoke128(offset).AsSingle();
        var r1 = term.Invoke128(offset + Block).AsSingle();
        var r2 = term.Invoke128(offset + 2 * Block).AsSingle();
        var r3 = term.Invoke128(offset + 3 * Block).AsSingle();

        // t0 = [r0₀ r1₀ r0₁ r1₁]、t1 = [r2₀ r3₀ r2₁ r3₁]、t2 = [r0₂ r1₂ r0₃ r1₃]、t3 = [r2₂ r3₂ r2₃ r3₃]
        var t0 = Sse.UnpackLow(r0, r1);
        var t1 = Sse.UnpackLow(r2, r3);
        var t2 = Sse.UnpackHigh(r0, r1);
        var t3 = Sse.UnpackHigh(r2, r3);
        sums = Sse.Add(sums, Sse.MoveLowToHigh(t0, t1));
        sums = Sse.Add(sums, Sse.MoveHighToLow(t1, t0));
        sums = Sse.Add(sums, Sse.MoveLowToHigh(t2, t3));
        return Sse.Add(sums, Sse.MoveHighToLow(t3, t2));
    }

    // ブロック和を受け取り、PairwiseSum と同じ繰り上がり順序で結合する加算木。
    private ref struct SumTree<T> where T : unmanaged, INumberBase<T>
    {
        // ブロックの数は 2³¹ 未満なので、部分和の段数は 31 以下
        internal const int Capacity = 32;

        private readonly Span<T> partial;
        private int depth;
        private int block;

        public SumTree(Span<T> partial)
        {
            this.partial = partial;
            depth = 0;
            block = 0;
        }

        public void Push(T sum)
        {
            // ブロック番号の末尾に並ぶ 1 の数だけ、同じ大きさの部分和と結合する（前の部分和を左に置く）
            for (int k = block; (k & 1) != 0; k >>= 1)
            {
                sum = partial[--depth] + sum;
            }

            partial[depth++] = sum;
            block++;
        }

        // 連続する 2^level ブロックの部分和を結合する（投入済みブロック数が 2^level の倍数であること）。
        public void Push(T sum, int level)
        {
            for (int k = block >> level; (k & 1) != 0; k >>= 1)
            {
                sum = partial[--depth] + sum;
            }

            partial[depth++] = sum;
            block += 1 << level;
        }

        public void Push(Vector512<T> sums)
        {
            for (int k = 0; k < Vector512<T>.Count; k++)
            {
                Push(sums.GetElement(k));
            }
        }

        public void Push(Vector256<T> sums)
        {
            for (int k = 0; k < Vector256<T>.Count; k++)
            {
                Push(sums.GetElement(k));
            }
        }

        public void Push(Vector128<T> sums)
        {
            for (int k = 0; k < Vector128<T>.Count; k++)
            {
                Push(sums.GetElement(k));
            }
        }

        public readonly T Result()
        {
            if (depth == 0)
            {
                return T.Zero;
            }

            int d = depth;
            T result = partial[--d];
            while (d > 0)
            {
                result = partial[--d] + result;
            }

            return result;
        }
    }
}
