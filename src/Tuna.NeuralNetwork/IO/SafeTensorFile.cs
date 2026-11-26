using System.Buffers;
using System.Buffers.Binary;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Tuna.NeuralNetwork.IO;

/// <summary>
/// Reads tensors and metadata from a safetensors file or stream.
/// <para>safetensors ファイルまたはストリームからテンソルおよびメタデータを読み取ります。</para>
/// </summary>
/// <remarks>
/// <para>The region from the stream's initial position to its end is treated as a single file; do not manipulate the underlying stream externally while this instance is in use. Methods returning arrays allocate independent copies. <see cref="Tensors"/> and <see cref="Metadata"/> remain accessible after <see cref="Dispose"/>. Concurrent reads and disposal on the same instance are serialized.</para>
/// <para>ストリームの開始位置から末尾までを 1 ファイルとして扱います。使用中は外部から同じストリームを操作しないでください。配列を返すメソッドは独立したコピーを新規確保します。<see cref="Tensors"/> と <see cref="Metadata"/> は <see cref="Dispose"/> 後も参照できます。同一インスタンスに対する読み取りと破棄はスレッドセーフに直列化されます。</para>
/// </remarks>
public sealed class SafeTensorFile : IDisposable
{
    // safetensors 公式実装と同一のヘッダーサイズ上限
    /// <summary>
    /// The maximum allowed header size in bytes (100,000,000).
    /// <para>許容されるヘッダーサイズの最大値（100,000,000 バイト）。</para>
    /// </summary>
    public static readonly int MaxHeaderLength = 100_000_000;
    private const int BufferSize = 64 * 1024;
    private readonly Stream stream;
    private readonly bool leaveOpen;
    private readonly long dataStart;
    private readonly object gate = new();
    private bool disposed;

    private SafeTensorFile(Stream stream, bool leaveOpen)
    {
        (dataStart, Tensors, Metadata) = SafeTensorHeader.Read(stream);
        this.stream = stream;
        this.leaveOpen = leaveOpen;
    }

    /// <summary>
    /// Gets the validated tensor entries contained in the file.
    /// <para>ファイルに含まれる検証済みテンソル情報の一覧を取得します。</para>
    /// </summary>
    /// <value>A read-only dictionary keyed by case-sensitive tensor name. / 大文字と小文字を区別するテンソル名をキーとする読み取り専用辞書。</value>
    public IReadOnlyDictionary<string, SafeTensorInfo> Tensors { get; }
    /// <summary>
    /// Gets the file-level metadata from the <c>__metadata__</c> header section.
    /// <para>ヘッダーの <c>__metadata__</c> セクションに含まれるメタデータを取得します。</para>
    /// </summary>
    /// <value>A read-only string-to-string dictionary. / 文字列のキーと値を保持する読み取り専用辞書。</value>
    public IReadOnlyDictionary<string, string> Metadata { get; }

    /// <summary>
    /// Opens a safetensors file at the specified path and validates its header and data layout.
    /// <para>指定したパスの safetensors ファイルを開き、ヘッダーとデータ配置を検証します。</para>
    /// </summary>
    /// <param name="path">The path to the safetensors file. / safetensors ファイルのパス。</param>
    /// <returns>A validated <see cref="SafeTensorFile"/> instance that owns and disposes the underlying file stream. / ヘッダーとデータ配置を検証済みの <see cref="SafeTensorFile"/> インスタンス（<see cref="Dispose"/> 時に入力ファイルを閉じます）。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="path"/> is <see langword="null"/>. / <paramref name="path"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="path"/> is empty. / <paramref name="path"/> が空文字列の場合。</exception>
    /// <exception cref="InvalidDataException">The header or data layout is invalid. / ヘッダーまたはデータ領域の配置が不正な場合。</exception>
    /// <exception cref="NotSupportedException">A tensor declares an unsupported data type. / 未対応のデータ型が宣言されている場合。</exception>
    public static SafeTensorFile Open(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        return Open(new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read));
    }

    /// <summary>
    /// Opens a safetensors file from a readable, seekable stream and validates its header and data layout.
    /// <para>読み取りおよびシーク可能なストリームから safetensors ファイルを開き、ヘッダーとデータ配置を検証します。</para>
    /// </summary>
    /// <param name="stream">A readable and seekable input stream. / 読み取りとシークが可能な入力ストリーム。</param>
    /// <param name="leaveOpen"><see langword="true"/> to leave <paramref name="stream"/> open after disposal or on open failure; <see langword="false"/> to dispose it. / 破棄時またはヘッダー解析失敗時に入力ストリームを開いたままにする場合は <see langword="true"/>、閉じる場合は <see langword="false"/>。</param>
    /// <returns>A validated <see cref="SafeTensorFile"/> instance. / ヘッダーとデータ配置を検証済みの <see cref="SafeTensorFile"/> インスタンス。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="stream"/> is <see langword="null"/>. / <paramref name="stream"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="stream"/> does not support reading or seeking. / <paramref name="stream"/> が読み取りまたはシークに対応していない場合。</exception>
    /// <exception cref="InvalidDataException">The header or data layout is invalid. / ヘッダーまたはデータ領域の配置が不正な場合。</exception>
    /// <exception cref="NotSupportedException">A tensor declares an unsupported data type. / 未対応のデータ型が宣言されている場合。</exception>
    public static SafeTensorFile Open(Stream stream, bool leaveOpen = false)
    {
        ArgumentNullException.ThrowIfNull(stream);
        if (!stream.CanRead || !stream.CanSeek)
        {
            throw new ArgumentException("A readable, seekable stream is required.", nameof(stream));
        }

        try
        {
            return new SafeTensorFile(stream, leaveOpen);
        }
        catch
        {
            if (!leaveOpen)
            {
                stream.Dispose();
            }

            throw;
        }
    }

    /// <summary>
    /// Retrieves the metadata entry for the tensor with the specified name.
    /// <para>指定した名前のテンソル情報を取得します。</para>
    /// </summary>
    /// <param name="name">The case-sensitive tensor name. / 大文字と小文字を区別するテンソル名。</param>
    /// <returns>The <see cref="SafeTensorInfo"/> describing the tensor's shape, data type, and byte location. / テンソルの形状、保存型、データ位置などを表す <see cref="SafeTensorInfo"/>。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="name"/> is <see langword="null"/>. / <paramref name="name"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="KeyNotFoundException">No tensor with <paramref name="name"/> exists in the file. / 指定した名前のテンソルがファイルに存在しない場合。</exception>
    public SafeTensorInfo GetTensorInfo(string name)
    {
        ArgumentNullException.ThrowIfNull(name);
        return Tensors[name];
    }

    /// <summary>
    /// Reads a tensor into a new row-major array without type conversion.
    /// <para>型変換を行わずに、保存時の要素型でテンソルを新しい行優先配列として読み取ります。</para>
    /// </summary>
    /// <typeparam name="T">The exact unmanaged element type matching the tensor's stored data type (integer types, <see cref="Half"/>, <see cref="float"/>, or <see cref="double"/>). / テンソルの保存型と一致する要素型（整数型、<see cref="Half"/>、<see cref="float"/>、または <see cref="double"/>）。</typeparam>
    /// <param name="name">The case-sensitive tensor name. / 大文字と小文字を区別するテンソル名。</param>
    /// <returns>A new row-major array containing the tensor elements. / テンソルの要素を格納した新しい行優先配列。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="name"/> is <see langword="null"/>. / <paramref name="name"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="KeyNotFoundException">No tensor with <paramref name="name"/> exists in the file. / 指定した名前のテンソルがファイルに存在しない場合。</exception>
    /// <exception cref="NotSupportedException"><typeparamref name="T"/> is unsupported or does not match the stored data type, or the element count exceeds <see cref="Array.MaxLength"/>. / <typeparamref name="T"/> が未対応または保存型と一致しない場合、あるいは要素数が <see cref="Array.MaxLength"/> を超える場合。</exception>
    /// <exception cref="ObjectDisposedException">The file has been disposed. / ファイルが既に破棄されている場合。</exception>
    public T[] Read<T>(string name) where T : unmanaged
    {
        lock (gate)
        {
            ThrowIfDisposed();
            var info = GetTensorInfo(name);
            if (info.DataType != SafeTensorTypes.For<T>())
            {
                throw new NotSupportedException($"Tensor '{name}' has dtype {info.DataType}; it cannot be read as {typeof(T).Name} without conversion.");
            }

            var result = GC.AllocateUninitializedArray<T>(ArrayLength(info.ElementCount));
            stream.Position = checked(dataStart + info.DataOffset);
            int width = Unsafe.SizeOf<T>(), chunkElements = BufferSize / width;
            for (int i = 0; i < result.Length;)
            {
                int count = Math.Min(chunkElements, result.Length - i);
                var bytes = MemoryMarshal.AsBytes(result.AsSpan(i, count));
                stream.ReadExactly(bytes);
                if (!BitConverter.IsLittleEndian && width > 1)
                {
                    for (int j = 0; j < bytes.Length; j += width)
                    {
                        bytes.Slice(j, width).Reverse();
                    }
                }

                i += count;
            }

            return result;
        }
    }

    /// <summary>
    /// Reads a floating-point tensor (<c>F32</c>, <c>F16</c>, or <c>BF16</c>) into a new <see cref="float"/> array.
    /// <para>浮動小数点テンソル（<c>F32</c>、<c>F16</c>、または <c>BF16</c>）を単精度浮動小数点数の新しい配列として読み取ります。</para>
    /// </summary>
    /// <param name="name">The case-sensitive tensor name. / 大文字と小文字を区別するテンソル名。</param>
    /// <returns>A new row-major <see cref="float"/> array. / 行優先の新しい <see cref="float"/> 配列。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="name"/> is <see langword="null"/>. / <paramref name="name"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="KeyNotFoundException">No tensor with <paramref name="name"/> exists in the file. / 指定した名前のテンソルがファイルに存在しない場合。</exception>
    /// <exception cref="NotSupportedException">The stored data type cannot be converted to <see cref="float"/>, or the element count exceeds <see cref="Array.MaxLength"/>. / 保存型を <see cref="float"/> に変換できない場合、または要素数が <see cref="Array.MaxLength"/> を超える場合。</exception>
    /// <exception cref="ObjectDisposedException">The file has been disposed. / ファイルが既に破棄されている場合。</exception>
    public float[] ReadSingle(string name) => ReadFloating<float>(name);

    /// <summary>
    /// Reads a floating-point tensor (<c>F64</c>, <c>F32</c>, <c>F16</c>, or <c>BF16</c>) into a new <see cref="double"/> array.
    /// <para>浮動小数点テンソル（<c>F64</c>、<c>F32</c>、<c>F16</c>、または <c>BF16</c>）を倍精度浮動小数点数の新しい配列として読み取ります。</para>
    /// </summary>
    /// <param name="name">The case-sensitive tensor name. / 大文字と小文字を区別するテンソル名。</param>
    /// <returns>A new row-major <see cref="double"/> array. / 行優先の新しい <see cref="double"/> 配列。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="name"/> is <see langword="null"/>. / <paramref name="name"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="KeyNotFoundException">No tensor with <paramref name="name"/> exists in the file. / 指定した名前のテンソルがファイルに存在しない場合。</exception>
    /// <exception cref="NotSupportedException">The stored data type cannot be converted to <see cref="double"/>, or the element count exceeds <see cref="Array.MaxLength"/>. / 保存型を <see cref="double"/> に変換できない場合、または要素数が <see cref="Array.MaxLength"/> を超える場合。</exception>
    /// <exception cref="ObjectDisposedException">The file has been disposed. / ファイルが既に破棄されている場合。</exception>
    public double[] ReadDouble(string name) => ReadFloating<double>(name);

    /// <summary>
    /// Reads a floating-point tensor into a new array of the specified floating-point type, converting from narrower formats as needed.
    /// <para>必要に応じて低精度形式から変換しながら、テンソルを指定した浮動小数点型の新しい配列として読み取ります。</para>
    /// </summary>
    /// <typeparam name="T">The target floating-point type (<see cref="float"/> or <see cref="double"/>). / 読み取り先の浮動小数点型（<see cref="float"/> または <see cref="double"/>）。</typeparam>
    /// <param name="name">The case-sensitive tensor name. / 大文字と小文字を区別するテンソル名。</param>
    /// <returns>A new row-major array. Supports <c>F32</c>, <c>F16</c>, and <c>BF16</c> for both types, and <c>F64</c> when <typeparamref name="T"/> is <see cref="double"/>. / 行優先の新しい配列。<c>F32</c>、<c>F16</c>、<c>BF16</c> に加え、<typeparamref name="T"/> が <see cref="double"/> の場合は <c>F64</c> も読み取れます。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="name"/> is <see langword="null"/>. / <paramref name="name"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="KeyNotFoundException">No tensor with <paramref name="name"/> exists in the file. / 指定した名前のテンソルがファイルに存在しない場合。</exception>
    /// <exception cref="NotSupportedException"><typeparamref name="T"/> is neither <see cref="float"/> nor <see cref="double"/>, the stored data type is not convertible to <typeparamref name="T"/>, or the element count exceeds <see cref="Array.MaxLength"/>. / <typeparamref name="T"/> が <see cref="float"/> でも <see cref="double"/> でもない場合、保存型から <typeparamref name="T"/> への変換が未対応の場合、あるいは要素数が <see cref="Array.MaxLength"/> を超える場合。</exception>
    /// <exception cref="ObjectDisposedException">The file has been disposed. / ファイルが既に破棄されている場合。</exception>
    public T[] ReadFloating<T>(string name) where T : unmanaged, IFloatingPointIeee754<T>
    {
        lock (gate)
        {
            ThrowIfDisposed();
            if (typeof(T) != typeof(float) && typeof(T) != typeof(double))
            {
                throw new NotSupportedException("ReadFloating supports float and double.");
            }

            var info = GetTensorInfo(name);
            if (info.DataType == SafeTensorTypes.For<T>())
            {
                return Read<T>(name);
            }

            if (info.DataType is not (SafeTensorDataType.Float16 or SafeTensorDataType.BFloat16 or SafeTensorDataType.Float32))
            {
                throw new NotSupportedException($"The conversion from {info.DataType} to {typeof(T).Name} for tensor '{name}' is not supported.");
            }

            var result = GC.AllocateUninitializedArray<T>(ArrayLength(info.ElementCount));
            if (result.Length == 0)
            {
                return result;
            }

            int width = info.DataType == SafeTensorDataType.Float32 ? 4 : 2;
            var buffer = ArrayPool<byte>.Shared.Rent(BufferSize);
            try
            {
                stream.Position = checked(dataStart + info.DataOffset);
                for (int i = 0; i < result.Length;)
                {
                    int count = Math.Min(BufferSize / width, result.Length - i);
                    stream.ReadExactly(buffer.AsSpan(0, count * width));
                    for (int j = 0; j < count; j++)
                    {
                        float value;
                        if (width == 4)
                        {
                            value = BinaryPrimitives.ReadSingleLittleEndian(buffer.AsSpan(j * 4, 4));
                        }
                        else
                        {
                            ushort bits = BinaryPrimitives.ReadUInt16LittleEndian(buffer.AsSpan(j * 2, 2));
                            value = info.DataType == SafeTensorDataType.Float16
                                ? (float)BitConverter.UInt16BitsToHalf(bits)
                                : BitConverter.Int32BitsToSingle((int)((uint)bits << 16));
                        }

                        result[i + j] = T.CreateChecked(value);
                    }

                    i += count;
                }

                return result;
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(buffer);
            }
        }
    }

    /// <summary>
    /// Reads the raw little-endian bytes of a tensor into a new array.
    /// <para>テンソルの生バイト列（リトルエンディアン）を新しい配列として読み取ります。</para>
    /// </summary>
    /// <param name="name">The case-sensitive tensor name. / 大文字と小文字を区別するテンソル名。</param>
    /// <returns>A new byte array containing the raw tensor payload. / 生のテンソルデータを格納した新しいバイト配列。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="name"/> is <see langword="null"/>. / <paramref name="name"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="KeyNotFoundException">No tensor with <paramref name="name"/> exists in the file. / 指定した名前のテンソルがファイルに存在しない場合。</exception>
    /// <exception cref="NotSupportedException">The tensor's byte length exceeds <see cref="Array.MaxLength"/>. / データのバイト数が <see cref="Array.MaxLength"/> を超える場合。</exception>
    /// <exception cref="ObjectDisposedException">The file has been disposed. / ファイルが既に破棄されている場合。</exception>
    public byte[] ReadBytes(string name)
    {
        lock (gate)
        {
            ThrowIfDisposed();
            var info = GetTensorInfo(name);
            var result = GC.AllocateUninitializedArray<byte>(ArrayLength((ulong)info.ByteLength));
            stream.Position = checked(dataStart + info.DataOffset);
            stream.ReadExactly(result);
            return result;
        }
    }

    /// <summary>
    /// Copies the raw little-endian bytes of a tensor to the current position of <paramref name="destination"/>.
    /// <para>テンソルの生バイト列（リトルエンディアン）を <paramref name="destination"/> の現在位置へコピーします。</para>
    /// </summary>
    /// <param name="name">The case-sensitive tensor name. / 大文字と小文字を区別するテンソル名。</param>
    /// <param name="destination">A writable destination stream distinct from the underlying input stream. / 入力ストリームとは別の書き込み可能な出力先ストリーム。</param>
    /// <remarks>
    /// <para>Only the raw tensor payload is written (without a safetensors header), and <paramref name="destination"/> is left open.</para>
    /// <para>ヘッダーを含まないテンソル本体のバイト列のみを書き込み、<paramref name="destination"/> は開いたままにします。</para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="name"/> or <paramref name="destination"/> is <see langword="null"/>. / <paramref name="name"/> または <paramref name="destination"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="destination"/> is not writable or is the same instance as the input stream. / <paramref name="destination"/> が書き込み不可、または入力ストリームと同一インスタンスの場合。</exception>
    /// <exception cref="KeyNotFoundException">No tensor with <paramref name="name"/> exists in the file. / 指定した名前のテンソルがファイルに存在しない場合。</exception>
    /// <exception cref="ObjectDisposedException">The file has been disposed. / ファイルが既に破棄されている場合。</exception>
    public void CopyTo(string name, Stream destination)
    {
        ArgumentNullException.ThrowIfNull(destination);
        if (!destination.CanWrite || ReferenceEquals(stream, destination))
        {
            throw new ArgumentException("A writable stream distinct from the input is required.", nameof(destination));
        }

        lock (gate)
        {
            ThrowIfDisposed();
            var info = GetTensorInfo(name);
            if (info.ByteLength == 0)
            {
                return;
            }

            var buffer = ArrayPool<byte>.Shared.Rent(BufferSize);
            try
            {
                stream.Position = checked(dataStart + info.DataOffset);
                long remaining = info.ByteLength;
                while (remaining > 0)
                {
                    int count = (int)Math.Min(BufferSize, remaining);
                    stream.ReadExactly(buffer.AsSpan(0, count));
                    destination.Write(buffer.AsSpan(0, count));
                    remaining -= count;
                }
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(buffer);
            }
        }
    }

    /// <summary>
    /// Releases the underlying stream unless opened with <c>leaveOpen</c> set to <see langword="true"/>.
    /// <para>入力ストリームを解放します（<c>leaveOpen</c> に <see langword="true"/> を指定して開いた場合を除く）。</para>
    /// </summary>
    public void Dispose()
    {
        lock (gate)
        {
            if (disposed)
            {
                return;
            }

            disposed = true;
            if (!leaveOpen)
            {
                stream.Dispose();
            }
        }
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(disposed, this);

    private static int ArrayLength(ulong count)
    {
        if (count > (ulong)Array.MaxLength)
        {
            throw new NotSupportedException("The tensor exceeds Array.MaxLength. Inspect its metadata or use CopyTo for raw data.");
        }

        return (int)count;
    }
}
