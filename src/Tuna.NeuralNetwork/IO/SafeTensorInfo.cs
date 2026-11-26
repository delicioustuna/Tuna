namespace Tuna.NeuralNetwork.IO;

/// <summary>
/// Describes a validated tensor entry in a safetensors file.
/// <para>safetensors ファイル内の検証済みテンソル情報を表します。</para>
/// </summary>
/// <remarks>
/// <para>Tensor data in the file is stored in little-endian byte order and C-contiguous (row-major) layout.</para>
/// <para>ファイル内のテンソルデータはリトルエンディアンかつ C 順（行優先）で格納されています。</para>
/// </remarks>
public sealed class SafeTensorInfo
{
    internal SafeTensorInfo(string name, SafeTensorDataType dataType, ulong[] shape, ulong elementCount, long dataOffset, long byteLength)
    {
        Name = name;
        DataType = dataType;
        Shape = Array.AsReadOnly(shape);
        ElementCount = elementCount;
        DataOffset = dataOffset;
        ByteLength = byteLength;
    }

    /// <summary>
    /// Gets the tensor name.
    /// <para>テンソル名を取得します。</para>
    /// </summary>
    public string Name { get; }
    /// <summary>
    /// Gets the stored element data type.
    /// <para>保存されている要素型を取得します。</para>
    /// </summary>
    public SafeTensorDataType DataType { get; }
    /// <summary>
    /// Gets the dimension lengths of the tensor.
    /// <para>テンソルの各軸の長さを取得します。</para>
    /// </summary>
    /// <value>A read-only list of axis lengths; empty for a scalar, or containing 0 for an empty tensor. / 各軸の長さを格納した読み取り専用リスト。空の場合はスカラー、0 の軸を含む場合は空テンソルを表します。</value>
    public IReadOnlyList<ulong> Shape { get; }
    /// <summary>
    /// Gets the total number of elements in the tensor.
    /// <para>テンソルの総要素数を取得します。</para>
    /// </summary>
    public ulong ElementCount { get; }
    /// <summary>
    /// Gets the byte offset from the start of the file's data region.
    /// <para>ファイルのデータ領域先頭からのバイトオフセットを取得します。</para>
    /// </summary>
    public long DataOffset { get; }
    /// <summary>
    /// Gets the stored byte length of the tensor data.
    /// <para>テンソルデータの保存バイト数を取得します。</para>
    /// </summary>
    public long ByteLength { get; }
}
