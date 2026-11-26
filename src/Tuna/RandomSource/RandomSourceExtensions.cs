namespace Tuna.RandomSource;

/// <summary>
/// Provides extension methods for uniformly shuffling and generating random permutations with an <see cref="IRandomSource"/>.
/// <para><see cref="IRandomSource"/> を用いて有限列を一様にシャッフルおよび順列生成するための拡張メソッドを提供します。</para>
/// </summary>
/// <remarks>
/// <para>For subclasses of <see cref="RandomGenerator"/> (such as <see cref="Philox4x32Random"/> and <see cref="Pcg64DxsmRandom"/>), the entire operation executes under a single lock acquisition while producing the exact same outputs and random consumption as repeated single-element calls.</para>
/// <para>If an exception occurs during execution (for example, stream exhaustion), partial swaps and random state advances already performed are not rolled back.</para>
/// <para><see cref="RandomGenerator"/> の派生型（<see cref="Philox4x32Random"/> や <see cref="Pcg64DxsmRandom"/> など）では、呼び出し全体でロック取得を 1 回にまとめつつ、単発呼び出しを繰り返した場合と同一の結果および乱数消費になります。</para>
/// <para>処理の途中で例外が発生した場合（乱数ストリームの枯渇など）、それまでに行われた要素の入れ替えおよび乱数状態の進行はロールバックされません。</para>
/// </remarks>
public static class RandomSourceExtensions
{
    extension(IRandomSource random)
    {
        /// <summary>
        /// Randomly shuffles the elements of a span in place into a uniformly distributed permutation.
        /// <para>スパンの要素をインプレースで一様な順序にシャッフルします。</para>
        /// </summary>
        /// <typeparam name="T">The type of elements in <paramref name="values"/>. / <paramref name="values"/> の要素の型。</typeparam>
        /// <param name="values">The span of elements to shuffle in place. / インプレースでシャッフルする要素のスパン。</param>
        /// <remarks>
        /// <para>Uses the Fisher–Yates algorithm, invoking <see cref="IRandomSource.NextInt32(int)"/> with <c>i + 1</c> for <c>i = values.Length - 1</c> down to <c>1</c>. No random numbers are consumed when <c>values.Length &lt;= 1</c>.</para>
        /// <para>Fisher–Yates 法を使用し、<c>i = values.Length - 1</c> から <c>1</c> まで降順に <see cref="IRandomSource.NextInt32(int)"/> を引数 <c>i + 1</c> で呼び出します。<c>values.Length &lt;= 1</c> の場合は乱数を消費しません。</para>
        /// </remarks>
        /// <exception cref="ArgumentNullException">The random source is <see langword="null"/>. / 乱数源が <see langword="null"/> の場合。</exception>
        public void Shuffle<T>(Span<T> values)
        {
            ArgumentNullException.ThrowIfNull(random);
            if (random is RandomGenerator generator)
            {
                generator.ShuffleSynchronized(values);
                return;
            }

            for (int i = values.Length - 1; i > 0; i--)
            {
                int j = random.NextInt32(i + 1);
                (values[i], values[j]) = (values[j], values[i]);
            }
        }

        /// <summary>
        /// Writes a uniformly random permutation of the integers from <c>0</c> to <c>destination.Length - 1</c> into the specified span.
        /// <para><c>0</c> から <c>destination.Length - 1</c> までの整数の一様な順列を指定したスパンに書き込みます。</para>
        /// </summary>
        /// <param name="destination">The span to fill with the random permutation. / 順列を書き込む出力先スパン。</param>
        /// <remarks>
        /// <para>Initializes <paramref name="destination"/> with <c>0, 1, …, destination.Length - 1</c> and shuffles it using <see cref="Shuffle{T}(IRandomSource, Span{T})"/>, consuming the exact same random values.</para>
        /// <para><paramref name="destination"/> を <c>0, 1, …, destination.Length - 1</c> で初期化した後に <see cref="Shuffle{T}(IRandomSource, Span{T})"/> でシャッフルします。乱数の消費は <see cref="Shuffle{T}(IRandomSource, Span{T})"/> と同一です。</para>
        /// </remarks>
        /// <exception cref="ArgumentNullException">The random source is <see langword="null"/>. / 乱数源が <see langword="null"/> の場合。</exception>
        public void Permutation(Span<int> destination)
        {
            ArgumentNullException.ThrowIfNull(random);
            for (int k = 0; k < destination.Length; k++)
            {
                destination[k] = k;
            }

            random.Shuffle(destination);
        }
    }
}
