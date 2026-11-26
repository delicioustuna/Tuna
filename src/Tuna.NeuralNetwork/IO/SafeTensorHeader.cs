using System.Buffers.Binary;
using System.Collections.ObjectModel;
using System.Text;
using System.Text.Json;

namespace Tuna.NeuralNetwork.IO;

internal static class SafeTensorHeader
{
    private static readonly UTF8Encoding StrictUtf8 = new(false, true);
    private const int MaxDepth = 128; // serde_json の既定の再帰上限に合わせる。

    internal static (long DataStart, IReadOnlyDictionary<string, SafeTensorInfo> Tensors, IReadOnlyDictionary<string, string> Metadata) Read(Stream stream)
    {
        try
        {
            long origin = stream.Position, length = stream.Length;
            if (origin < 0 || origin > length || length - origin < 8)
            {
                throw new InvalidDataException("A safetensors file must contain an eight-byte header length.");
            }

            Span<byte> prefix = stackalloc byte[8];
            stream.ReadExactly(prefix);
            ulong n = BinaryPrimitives.ReadUInt64LittleEndian(prefix);
            if (n > (ulong)SafeTensorFile.MaxHeaderLength)
            {
                throw new InvalidDataException($"The safetensors header exceeds {SafeTensorFile.MaxHeaderLength} bytes.");
            }

            if (n == 0 || n > (ulong)(length - origin - 8))
            {
                throw new InvalidDataException("The safetensors header length is outside the file.");
            }

            var bytes = new byte[(int)n];
            stream.ReadExactly(bytes);
            if (bytes[0] != (byte)'{')
            {
                throw new InvalidDataException("The safetensors header must begin with '{'.");
            }

            _ = StrictUtf8.GetCharCount(bytes); // 無視する追加フィールド内も含めて不正な UTF-8 を拒否する。
            ValidateEscapedStrings(bytes);
            using var document = JsonDocument.Parse(bytes, new JsonDocumentOptions { AllowDuplicateProperties = false, MaxDepth = MaxDepth });
            long dataStart = checked(origin + 8 + (long)n), dataLength = length - dataStart;
            var tensors = new Dictionary<string, SafeTensorInfo>(StringComparer.Ordinal);
            var metadata = new Dictionary<string, string>(StringComparer.Ordinal);
            foreach (var entry in document.RootElement.EnumerateObject())
            {
                if (entry.Name == "__metadata__")
                {
                    if (entry.Value.ValueKind == JsonValueKind.Null)
                    {
                        continue; // 公式実装と同様に null はメタデータなしとして扱う。
                    }

                    if (entry.Value.ValueKind != JsonValueKind.Object)
                    {
                        throw new InvalidDataException("safetensors metadata must be a string-to-string map.");
                    }

                    foreach (var field in entry.Value.EnumerateObject())
                    {
                        if (field.Value.ValueKind != JsonValueKind.String)
                        {
                            throw new InvalidDataException("safetensors metadata values must be strings.");
                        }

                        metadata.Add(field.Name, field.Value.GetString()!);
                    }

                    continue;
                }

                JsonElement tensor = entry.Value;
                var dtype = Field(tensor, "dtype", JsonValueKind.String, entry.Name);
                SafeTensorDataType type = SafeTensorTypes.Parse(dtype.GetString());
                var dimensions = Field(tensor, "shape", JsonValueKind.Array, entry.Name);
                var offsets = Field(tensor, "data_offsets", JsonValueKind.Array, entry.Name);
                if (offsets.GetArrayLength() != 2)
                {
                    throw new InvalidDataException($"Tensor '{entry.Name}' must have exactly two data offsets.");
                }

                ulong begin = Unsigned(offsets[0], entry.Name), end = Unsigned(offsets[1], entry.Name);
                if (begin > end || end > (ulong)dataLength)
                {
                    throw new InvalidDataException($"Tensor '{entry.Name}' has data offsets outside the data region.");
                }

                var shape = new ulong[dimensions.GetArrayLength()];
                ulong count = 1;
                for (int i = 0; i < shape.Length; i++)
                {
                    shape[i] = Unsigned(dimensions[i], entry.Name);
                    count = checked(count * shape[i]);
                }

                ulong bits = checked(count * (ulong)SafeTensorTypes.BitWidth(type));
                if (bits % 8 != 0 || bits / 8 != end - begin)
                {
                    throw new InvalidDataException($"Tensor '{entry.Name}' has inconsistent shape, dtype, or byte length.");
                }

                tensors.Add(entry.Name, new SafeTensorInfo(entry.Name, type, shape, count, (long)begin, (long)(end - begin)));
            }

            long cursor = 0;
            // 空の範囲を、同じ先頭位置の非空の範囲より前に検証する。JSON の列挙順は問わない。
            foreach (var tensor in tensors.Values.OrderBy(t => t.DataOffset).ThenBy(t => t.ByteLength))
            {
                if (tensor.DataOffset != cursor)
                {
                    throw new InvalidDataException($"Tensor '{tensor.Name}' overlaps another tensor or leaves a gap in the data region.");
                }

                cursor = checked(cursor + tensor.ByteLength);
            }

            if (cursor != dataLength)
            {
                throw new InvalidDataException("The safetensors data region is not entirely covered by tensors.");
            }

            return (dataStart, new ReadOnlyDictionary<string, SafeTensorInfo>(tensors), new ReadOnlyDictionary<string, string>(metadata));
        }
        // JsonDocument は、不完全な UTF-16 エスケープの展開時には InvalidOperationException を投げる。
        catch (Exception ex) when (ex is JsonException or DecoderFallbackException or InvalidOperationException or OverflowException or EndOfStreamException)
        {
            throw new InvalidDataException("Invalid safetensors header: " + ex.Message, ex);
        }
    }

    private static void ValidateEscapedStrings(ReadOnlySpan<byte> bytes)
    {
        var reader = new Utf8JsonReader(bytes, new JsonReaderOptions { MaxDepth = MaxDepth });
        while (reader.Read())
        {
            // JsonDocument は参照しない文字列のエスケープを展開しないため、未知のフィールドの値も検証する。
            if ((reader.TokenType is JsonTokenType.String or JsonTokenType.PropertyName) && reader.ValueIsEscaped)
            {
                _ = reader.GetString();
            }
        }
    }

    private static JsonElement Field(JsonElement value, string name, JsonValueKind kind, string tensor)
    {
        if (value.ValueKind != JsonValueKind.Object || !value.TryGetProperty(name, out var field) || field.ValueKind != kind)
        {
            throw new InvalidDataException($"Tensor '{tensor}' requires a '{name}' field of kind {kind}.");
        }

        return field;
    }

    private static ulong Unsigned(JsonElement value, string tensor)
    {
        if (value.ValueKind != JsonValueKind.Number || !value.TryGetUInt64(out ulong result))
        {
            throw new InvalidDataException($"Tensor '{tensor}' requires nonnegative 64-bit integer dimensions and offsets.");
        }

        return result;
    }
}
