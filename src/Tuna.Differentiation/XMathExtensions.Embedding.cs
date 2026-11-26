using System.Numerics;

namespace Tuna.Differentiation;

public static partial class XMathExtensions
{
    extension(XMath)
    {
        /// <summary>
        /// Gathers the indexed columns of an embedding table and records their gradients.
        /// <para>埋め込みテーブルから指定された列を集め、その勾配を記録します。</para>
        /// </summary>
        /// <typeparam name="T">The element type, either float or double. / 要素型。float または double。</typeparam>
        /// <param name="table">The matrix node whose columns store the embeddings. / 各列に埋め込みベクトルを格納した行列ノード。</param>
        /// <param name="indices">The zero-based column indices in output order. / 出力順に並べた 0 始まりの列インデックス。</param>
        /// <returns>A matrix node with one column per index. / インデックスごとに 1 列を持つ行列ノード。</returns>
        /// <remarks>
        /// <para>Repeated indices accumulate gradient contributions in index order. The index span can be changed after this method returns.</para>
        /// <para>インデックスが重複する場合は、指定順に勾配の寄与を加算します。呼び出し後にインデックスのスパンを変更できます。</para>
        /// </remarks>
        /// <exception cref="ArgumentOutOfRangeException">An index is outside the columns of <paramref name="table"/>. / インデックスが <paramref name="table"/> の列の範囲外にある場合。</exception>
        /// <exception cref="InvalidOperationException"><paramref name="table"/> is invalid or recording requires a reset. / <paramref name="table"/> が無効、または記録にリセットが必要な場合。</exception>
        public static MatrixNode<T> Embedding<T>(MatrixNode<T> table, ReadOnlySpan<int> indices)
            where T : unmanaged, IFloatingPointIeee754<T>
        {
            var tape = table.Owner();
            var store = tape.Store<T>();
            bool requiresGradient = store.RequiresGradient(table.View.Slot);
            tape.EnsureCanApply(requiresGradient);
            foreach (int index in indices)
            {
                if ((uint)index >= (uint)table.Columns)
                {
                    throw new ArgumentOutOfRangeException(nameof(indices), "An embedding index is outside the table.");
                }
            }

            int slot = store.AddPooled(table.Rows, indices.Length, requiresGradient);
            if (table.Rows != 0 && indices.Length != 0)
            {
                var source = table.Value;
                var destination = store.WritableMatrix(slot);
                for (int i = 0; i < indices.Length; i++)
                {
                    source.Column(indices[i]).CopyTo(destination.Column(i));
                }

                if (requiresGradient)
                {
                    int saved = store.SaveEmbeddingIndices(indices);
                    tape.AppendArrayRecord<T>(ArrayOperation.Embedding, slot, table.View, default, 0, saved);
                }
            }

            return new(tape, ArrayView.Whole(slot, table.Rows, indices.Length));
        }
    }
}
