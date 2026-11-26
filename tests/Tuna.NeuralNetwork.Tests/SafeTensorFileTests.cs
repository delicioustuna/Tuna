using System.Buffers.Binary;
using System.IO.Compression;
using System.Text;
using Tuna.NeuralNetwork.IO;

namespace Tuna.NeuralNetwork.Tests;

public class SafeTensorFileTests
{
    [Fact]
    public void OfficialWriterFixtureIsReadableWithoutThirdPartyRuntimeDependencies()
    {
        using var file = SafeTensorFile.Open(Path.Combine(AppContext.BaseDirectory, "Data", "safetensors-reference.safetensors"));
        Assert.Equal(22, file.Tensors.Count);
        Assert.Equal("Tuna synthetic interoperability fixture", file.Metadata["origin"]);
        Assert.Equal(new float[] { 1, -2, 0.25f, 9 }, file.Read<float>("matrix"));
        Assert.Equal(new double[] { -3.5, double.PositiveInfinity, -0.0, double.Epsilon }, file.Read<double>("double"));
        Assert.Equal(long.MinValue, BitConverter.DoubleToInt64Bits(file.Read<double>("double")[2]));
        Assert.Equal(new sbyte[] { -128, 127 }, file.Read<sbyte>("i8"));
        Assert.Equal(new byte[] { 0, 255 }, file.Read<byte>("u8"));
        Assert.Equal(new short[] { short.MinValue, short.MaxValue }, file.Read<short>("i16"));
        Assert.Equal(new ushort[] { 0, ushort.MaxValue }, file.Read<ushort>("u16"));
        Assert.Equal(new int[] { int.MinValue, int.MaxValue }, file.Read<int>("i32"));
        Assert.Equal(new uint[] { 0, uint.MaxValue }, file.Read<uint>("u32"));
        Assert.Equal(new long[] { long.MinValue, long.MaxValue }, file.Read<long>("i64"));
        Assert.Equal(new ulong[] { 0, ulong.MaxValue }, file.Read<ulong>("u64"));
        Assert.Equal(new ushort[] { 0x3c00, 0x8000, 0x7c00, 0xfc00, 1, 0x7e42 }, file.Read<Half>("half").Select(BitConverter.HalfToUInt16Bits));
        foreach (string name in new[] { "half", "bf16" })
        {
            var values = file.ReadSingle(name);
            Assert.Equal(1f, values[0]);
            Assert.Equal(int.MinValue, BitConverter.SingleToInt32Bits(values[1]));
            Assert.Equal(float.PositiveInfinity, values[2]);
            Assert.Equal(float.NegativeInfinity, values[3]);
            Assert.Equal(name == "half" ? (float)BitConverter.UInt16BitsToHalf(1) : BitConverter.Int32BitsToSingle(1 << 16), values[4]);
            Assert.True(float.IsNaN(values[5]));
        }

        Assert.Equal(new byte[] { 0, 1 }, file.ReadBytes("bool"));
        Assert.Equal(new byte[] { 0, 0, 128, 191, 0, 0, 0, 64 }, file.ReadBytes("complex"));
        Assert.Equal(new byte[] { 0x38, 0xb8 }, file.ReadBytes("fp8"));
        foreach (var (name, dtype, expected) in new[]
        {
            ("fp4", SafeTensorDataType.Float4, new byte[] { 0x21 }),
            ("fp8_e5m2", SafeTensorDataType.Float8E5M2, new byte[] { 0x3c, 0xbc }),
            ("fp8_e8m0", SafeTensorDataType.Float8E8M0, new byte[] { 0x7f, 0x80 }),
            ("fp8_e4m3fnuz", SafeTensorDataType.Float8E4M3Fnuz, new byte[] { 0x40, 0xc0 }),
            ("fp8_e5m2fnuz", SafeTensorDataType.Float8E5M2Fnuz, new byte[] { 0x40, 0xc0 }),
        })
        {
            Assert.Equal(dtype, file.Tensors[name].DataType);
            Assert.Equal(expected, file.ReadBytes(name));
        }

        Assert.Equal(new float[] { 12 }, file.ReadSingle("scalar"));
        Assert.Empty(file.ReadSingle("empty"));
    }

    [Fact]
    public void MixedTypesAreLazyAndValuesKeepRowMajorOrder()
    {
        const string header = """
            {"counter":{"dtype":"I64","shape":[],"data_offsets":[16,24]},
             "層.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},
             "__metadata__":{"format":"pt","説明":"合成データ"}}
            """;
        var bytes = FileBytes(header, 24);
        int start = bytes.Length - 24;
        for (int i = 0; i < 4; i++)
        {
            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(start + i * 4), i + 1);
        }

        BinaryPrimitives.WriteInt64LittleEndian(bytes.AsSpan(start + 16), 1234567890123);
        using var input = new VirtualStream(bytes.Length, bytes);
        using var file = SafeTensorFile.Open(input, leaveOpen: true);
        Assert.Equal(start, input.BytesRead);
        Assert.Equal("合成データ", file.Metadata["説明"]);
        Assert.Equal(new ulong[] { 2, 2 }, file.Tensors["層.weight"].Shape);
        Assert.Equal(SafeTensorDataType.Int64, file.Tensors["counter"].DataType);
        Assert.Equal(new float[] { 1, 2, 3, 4 }, file.Read<float>("層.weight"));
        Assert.Equal(new long[] { 1234567890123 }, file.Read<long>("counter"));
        Assert.Throws<NotSupportedException>(() => file.Read<double>("層.weight"));
        Assert.Throws<KeyNotFoundException>(() => file.ReadSingle("missing"));
    }

    [Fact]
    public void ReturnedArraysAndMetadataHaveIndependentOwnership()
    {
        using var input = new MemoryStream(FileBytes("""{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}""", 4));
        using var file = SafeTensorFile.Open(input);
        var values = file.ReadSingle("x");
        values[0] = 100;
        Assert.Equal(0, file.ReadSingle("x")[0]);
        Assert.Throws<NotSupportedException>(() => ((IDictionary<string, SafeTensorInfo>)file.Tensors).Clear());
        Assert.Throws<NotSupportedException>(() => ((IDictionary<string, string>)file.Metadata).Add("x", "y"));
        Assert.Throws<NotSupportedException>(() => ((IList<ulong>)file.Tensors["x"].Shape)[0] = 99);
    }

    [Theory]
    [InlineData("""{"__metadata__":null}""", 0)]
    [InlineData("""{"__metadata__":null,"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}""", 4)]
    public void NullMetadataMeansAbsentMetadata(string header, int dataLength)
    {
        using var file = SafeTensorFile.Open(new MemoryStream(FileBytes(header, dataLength)));
        Assert.Empty(file.Metadata);
        if (dataLength == 0)
        {
            Assert.Empty(file.Tensors);
        }
        else
        {
            Assert.Equal(new float[] { 0 }, file.ReadSingle("x"));
        }
    }

    [Fact]
    public void EscapedUnicodePairsAndLiteralBackslashesAreAcceptedInUnknownFields()
    {
        const string header = """{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4],"future":{"pair":"\ud83d\ude00","literal":"\\ud800"}}}""";
        using var file = SafeTensorFile.Open(new MemoryStream(FileBytes(header, 4)));
        Assert.Equal(new float[] { 0 }, file.ReadSingle("x"));
    }

    [Theory]
    [InlineData(70, true)]
    [InlineData(200, false)]
    public void UnknownNestedValuesUseOfficialJsonDepthLimit(int depth, bool valid)
    {
        string nested = new string('[', depth) + "0" + new string(']', depth);
        string header = """{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4],"future":NESTED}}""".Replace("NESTED", nested);
        using var input = new MemoryStream(FileBytes(header, 4));
        if (valid)
        {
            using var file = SafeTensorFile.Open(input);
            Assert.Single(file.ReadSingle("x"));
        }
        else
        {
            Assert.Throws<InvalidDataException>(() => SafeTensorFile.Open(input));
        }
    }

    [Fact]
    public void FloatingPointConversionsWidenAndPreserveNativeDoubleBits()
    {
        int[] bits = [0, int.MinValue, 1, 0x7f7fffff, 0x7f800000, unchecked((int)0xff800000), 0x7fc01234];
        var bytes = FileBytes("""{"x":{"dtype":"F32","shape":[7],"data_offsets":[0,28]}}""", 28);
        for (int i = 0; i < bits.Length; i++)
        {
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(bytes.Length - 28 + 4 * i), bits[i]);
        }

        using var file = SafeTensorFile.Open(new MemoryStream(bytes));
        var actual = file.ReadDouble("x");
        for (int i = 0; i < bits.Length; i++)
        {
            double expected = BitConverter.Int32BitsToSingle(bits[i]);
            if (double.IsNaN(expected))
            {
                Assert.True(double.IsNaN(actual[i]));
            }
            else
            {
                Assert.Equal(BitConverter.DoubleToInt64Bits(expected), BitConverter.DoubleToInt64Bits(actual[i]));
            }
        }

        Assert.Equal(actual, file.ReadFloating<double>("x"));
        Assert.Equal(bits, file.ReadFloating<float>("x").Select(BitConverter.SingleToInt32Bits));
        Assert.Throws<NotSupportedException>(() => file.ReadFloating<Half>("x"));

        long[] doubleBits = [0x7ff8123456789abc, long.MinValue, 1];
        var doubles = FileBytes("""{"x":{"dtype":"F64","shape":[3],"data_offsets":[0,24]}}""", 24);
        for (int i = 0; i < doubleBits.Length; i++)
        {
            BinaryPrimitives.WriteInt64LittleEndian(doubles.AsSpan(doubles.Length - 24 + 8 * i), doubleBits[i]);
        }

        using var native = SafeTensorFile.Open(new MemoryStream(doubles));
        Assert.Equal(doubleBits, native.ReadDouble("x").Select(BitConverter.DoubleToInt64Bits));
        Assert.Throws<NotSupportedException>(() => native.ReadSingle("x"));
    }

    [Theory]
    [InlineData("F16")]
    [InlineData("BF16")]
    public void AllSixteenBitFloatingPatternsCanBeWidenedToDouble(string dtype)
    {
        const int count = 65536;
        var bytes = FileBytes($$$"""{"x":{"dtype":"{{{dtype}}}","shape":[65536],"data_offsets":[0,131072]}}""", count * 2);
        for (int i = 0; i < count; i++)
        {
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(bytes.Length - count * 2 + i * 2), (ushort)i);
        }

        using var file = SafeTensorFile.Open(new MemoryStream(bytes));
        var actual = file.ReadFloating<double>("x");
        Assert.Equal(count, actual.Length);
        for (int i = 0; i < count; i++)
        {
            double expected = dtype == "F16" ? (float)BitConverter.UInt16BitsToHalf((ushort)i) : BitConverter.Int32BitsToSingle(i << 16);
            if (double.IsNaN(expected))
            {
                Assert.True(double.IsNaN(actual[i]));
            }
            else
            {
                Assert.Equal(BitConverter.DoubleToInt64Bits(expected), BitConverter.DoubleToInt64Bits(actual[i]));
            }
        }
    }

    [Fact]
    public void SingleToDoubleConversionCrossesBufferBoundaries()
    {
        const int count = 20001;
        var bytes = FileBytes($$$"""{"x":{"dtype":"F32","shape":[{{{count}}}],"data_offsets":[0,{{{count * 4}}}]}}""", count * 4);
        var expected = new double[count];
        for (int i = 0; i < count; i++)
        {
            float value = (i - 10000) / 13f;
            expected[i] = value;
            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(bytes.Length - count * 4 + i * 4), value);
        }

        using var file = SafeTensorFile.Open(new MemoryStream(bytes));
        Assert.Equal(expected, file.ReadDouble("x"));
    }

    [Fact]
    public void ScalarsEmptyTensorsAndUnorderedEqualOffsetsAreValid()
    {
        const string header = """
            {"x":{"dtype":"F32","shape":[],"data_offsets":[0,4]},
             "empty":{"dtype":"F32","shape":[0,3],"data_offsets":[0,0]},
             "largeEmpty":{"dtype":"F32","shape":[18446744073709551615,0],"data_offsets":[4,4]}}
            """;
        using var file = SafeTensorFile.Open(new MemoryStream(FileBytes(header, 4)));
        Assert.Empty(file.Tensors["x"].Shape);
        Assert.Equal(1UL, file.Tensors["x"].ElementCount);
        Assert.Single(file.Read<float>("x"));
        Assert.Empty(file.ReadSingle("empty"));
        Assert.Empty(file.ReadBytes("largeEmpty"));
        Assert.Equal(ulong.MaxValue, file.Tensors["largeEmpty"].Shape[0]);
        using var empty = SafeTensorFile.Open(new MemoryStream(FileBytes("{}   ", 0)));
        Assert.Empty(empty.Tensors);
    }

    [Fact]
    public void FloatBitsArePreservedIncludingNaNsAndNegativeZero()
    {
        int[] bits = [0x7fc01234, 0x7f800000, unchecked((int)0xff800000), int.MinValue, 1];
        var bytes = FileBytes("""{"x":{"dtype":"F32","shape":[5],"data_offsets":[0,20]}}""", 20);
        for (int i = 0; i < bits.Length; i++)
        {
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(bytes.Length - 20 + 4 * i), bits[i]);
        }

        using var file = SafeTensorFile.Open(new MemoryStream(bytes));
        Assert.Equal(bits, file.ReadSingle("x").Select(BitConverter.SingleToInt32Bits));
        Assert.Equal(bytes[^20..], file.ReadBytes("x"));
    }

    [Theory]
    [InlineData("F16", 0x3c00, 0x8000, 0x7c00, 1f)]
    [InlineData("BF16", 0x3f80, 0x8000, 0x7f80, 1f)]
    public void HalfPrecisionTypesCanBeConverted(string dtype, int a, int b, int c, float expected)
    {
        var bytes = FileBytes($$$"""{"x":{"dtype":"{{{dtype}}}","shape":[3],"data_offsets":[0,6]}}""", 6);
        for (int i = 0; i < 3; i++)
        {
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(bytes.Length - 6 + 2 * i), (ushort)new[] { a, b, c }[i]);
        }

        using var file = SafeTensorFile.Open(new MemoryStream(bytes));
        var actual = file.ReadSingle("x");
        Assert.Equal(expected, actual[0]);
        Assert.Equal(int.MinValue, BitConverter.SingleToInt32Bits(actual[1]));
        Assert.Equal(float.PositiveInfinity, actual[2]);
        if (dtype == "F16")
        {
            Assert.Equal((Half)1, file.Read<Half>("x")[0]);
        }
        else
        {
            Assert.Throws<NotSupportedException>(() => file.Read<Half>("x"));
        }
    }

    [Theory]
    [InlineData("BOOL", SafeTensorDataType.Boolean, 1, 8)]
    [InlineData("F4", SafeTensorDataType.Float4, 2, 4)]
    [InlineData("F6_E2M3", SafeTensorDataType.Float6E2M3, 4, 6)]
    [InlineData("F6_E3M2", SafeTensorDataType.Float6E3M2, 4, 6)]
    [InlineData("F8_E5M2", SafeTensorDataType.Float8E5M2, 1, 8)]
    [InlineData("F8_E4M3", SafeTensorDataType.Float8E4M3, 1, 8)]
    [InlineData("F8_E8M0", SafeTensorDataType.Float8E8M0, 1, 8)]
    [InlineData("F8_E4M3FNUZ", SafeTensorDataType.Float8E4M3Fnuz, 1, 8)]
    [InlineData("F8_E5M2FNUZ", SafeTensorDataType.Float8E5M2Fnuz, 1, 8)]
    [InlineData("C64", SafeTensorDataType.Complex64, 1, 64)]
    public void RawOnlyTypesAreValidatedAndReadable(string dtype, SafeTensorDataType type, int elements, int width)
    {
        int length = elements * width / 8;
        var bytes = FileBytes($$$"""{"x":{"dtype":"{{{dtype}}}","shape":[{{{elements}}}],"data_offsets":[0,{{{length}}}]}}""", length);
        bytes[^1] = 123;
        using var file = SafeTensorFile.Open(new MemoryStream(bytes));
        Assert.Equal(type, file.Tensors["x"].DataType);
        Assert.Equal(bytes[^length..], file.ReadBytes("x"));
        using var destination = new MemoryStream();
        destination.WriteByte(99);
        file.CopyTo("x", destination);
        Assert.Equal(new byte[] { 99 }.Concat(bytes[^length..]), destination.ToArray());
        Assert.True(destination.CanWrite);
        Assert.Throws<NotSupportedException>(() => file.ReadSingle("x"));
        Assert.Throws<NotSupportedException>(() => file.Read<bool>("x"));
    }

    public static IEnumerable<object[]> InvalidHeaders()
    {
        string[] headers =
        [
            "[]", "null", "{}", " {\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}",
            """{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}""",
            """{"x":{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}}""",
            """{"x":{"dtype":"F32","shape":[18446744073709551616],"data_offsets":[0,4]}}""",
            """{"x":{"dtype":"F32","shape":[2147483647,2147483647,2147483647],"data_offsets":[0,4]}}""",
            """{"x":{"dtype":"F32","shape":[1152921504606846976],"data_offsets":[0,4]}}""",
            """{"x":{"dtype":"F32","shape":[0],"data_offsets":[0,4]}}""",
            """{"x":{"dtype":"F32","shape":[],"data_offsets":[0,0]}}""",
            """{"x":{"dtype":"F32","shape":[1.0],"data_offsets":[0,4]}}""",
            """{"x":{"dtype":"F32","shape":[true],"data_offsets":[0,4]}}""",
            """{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,3]}}""",
            """{"x":{"dtype":"F32","shape":[1],"data_offsets":[-4,0]}}""",
            """{"x":{"dtype":"F32","shape":[1],"data_offsets":[1,0]}}""",
            """{"x":{"dtype":"F32","shape":[1],"data_offsets":[4294967296,4294967300]}}""",
            """{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,8]}}""",
            """{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4,4]}}""",
            """{"x":{"dtype":"F32","shape":[1],"data_offsets":[0]}}""",
            """{"x":{"dtype":"F32","shape":[1],"data_offsets":[0.0,4]}}""",
            """{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,0]},"y":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}""",
            """{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"y":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}""",
            """{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}""",
            """{"x":{"dtype":"F32","dtype":"F32","shape":[1],"data_offsets":[0,4]}}""",
            """{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"\u0078":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}""",
            """{"__metadata__":{"n":1},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}""",
            """{"__metadata__":[],"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}""",
            """{"__metadata__":{"x":"a","x":"b"},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}""",
            """{"__metadata__":{"x":"a","\u0078":"b"},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}""",
            """{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4],"future":{"x":1,"x":2}}}""",
            """{"\ud800":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}""",
            """{"__metadata__":{"x":"\ud800"},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}""",
            """{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4],"note":"\ud800"}}""",
            """{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4],"note":[{"text":"\udc00"}]}}""",
            """{"x":["F32",[1],[0,4]]}""",
            """{"x":{"shape":[1],"data_offsets":[0,4]}}""",
            """{"x":{"dtype":"F32","data_offsets":[0,4]}}""",
            """{"x":{"dtype":"F32","shape":[1]}}""",
            """{"x":{"dtype":32,"shape":[1],"data_offsets":[0,4]}}""",
            """{"x":{"dtype":"F4","shape":[1],"data_offsets":[0,1]}}""",
            """{"x":{"dtype":"F6_E2M3","shape":[1],"data_offsets":[0,1]}}""",
        ];
        return headers.Select(h => new object[] { h });
    }

    [Theory]
    [MemberData(nameof(InvalidHeaders))]
    public void MalformedFilesAreRejectedBeforeReadingPayload(string header)
    {
        var bytes = FileBytes(header, 4);
        using var input = new VirtualStream(bytes.Length, bytes);
        Assert.Throws<InvalidDataException>(() => SafeTensorFile.Open(input, leaveOpen: true));
        Assert.True(input.BytesRead <= bytes.Length - 4);
    }

    [Theory]
    [InlineData("""{"x":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}""", 8)]
    [InlineData("""{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[8,12]}}""", 12)]
    [InlineData("""{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}""", 8)]
    public void GapsAndTrailingDataAreRejected(string header, int dataLength)
    {
        var bytes = FileBytes(header, dataLength);
        Assert.Throws<InvalidDataException>(() => SafeTensorFile.Open(new MemoryStream(bytes)));
    }

    [Theory]
    [InlineData("F16", 0x3c00)]
    [InlineData("BF16", 0x3f80)]
    public void ConversionAndConcurrentReadsCrossBufferBoundaries(string dtype, int oneBits)
    {
        const int count = 40001;
        var bytes = FileBytes($$$"""{"x":{"dtype":"{{{dtype}}}","shape":[{{{count}}}],"data_offsets":[0,{{{count * 2}}}]}}""", count * 2);
        for (int i = bytes.Length - count * 2; i < bytes.Length; i += 2)
        {
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(i), (ushort)oneBits);
        }

        using var file = SafeTensorFile.Open(new MemoryStream(bytes));
        Parallel.For(0, 4, _ =>
        {
            var actual = file.ReadSingle("x");
            Assert.Equal(count, actual.Length);
            Assert.All(actual, x => Assert.Equal(1f, x));
            Assert.Equal(bytes[^(count * 2)..], file.ReadBytes("x"));
        });
    }

    [Fact]
    public void TypedReadCrossesSeveralBuffersWithoutChangingIntegerBits()
    {
        const int count = 17001;
        var bytes = FileBytes($$$"""{"x":{"dtype":"U64","shape":[{{{count}}}],"data_offsets":[0,{{{count * 8}}}]}}""", count * 8);
        var expected = Enumerable.Range(0, count).Select(i => ((ulong)i << 32) | (uint)~i).ToArray();
        for (int i = 0; i < count; i++)
        {
            BinaryPrimitives.WriteUInt64LittleEndian(bytes.AsSpan(bytes.Length - count * 8 + i * 8), expected[i]);
        }

        using var file = SafeTensorFile.Open(new MemoryStream(bytes));
        Assert.Equal(expected, file.Read<ulong>("x"));
    }

    [Fact]
    public void InvalidUtf8IsNotReplacedEvenInUnknownFields()
    {
        var bytes = FileBytes("""{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4],"future":"!"}}""", 4);
        bytes[Array.IndexOf(bytes, (byte)'!')] = 0xff;
        Assert.Throws<InvalidDataException>(() => SafeTensorFile.Open(new MemoryStream(bytes)));
    }

    [Fact]
    public void UnknownDtypeIsAnExplicitUnsupportedFeature()
    {
        Assert.Throws<NotSupportedException>(() => SafeTensorFile.Open(new MemoryStream(FileBytes("""{"x":{"dtype":"FUTURE","shape":[1],"data_offsets":[0,4]}}""", 4))));
    }

    [Theory]
    [InlineData(0UL)]
    [InlineData(100000001UL)]
    [InlineData(4294967296UL + 60)]
    [InlineData(9223372036854775808UL + 60)]
    [InlineData(ulong.MaxValue)]
    public void HeaderLengthIsUnsignedBoundedAndNeverTruncated(ulong length)
    {
        var prefix = new byte[8];
        BinaryPrimitives.WriteUInt64LittleEndian(prefix, length);
        using var input = new VirtualStream(long.MaxValue, prefix);
        Assert.Throws<InvalidDataException>(() => SafeTensorFile.Open(input, leaveOpen: true));
        Assert.Equal(8, input.BytesRead);
    }

    [Fact]
    public void ExactOfficialHeaderLimitIsAccepted()
    {
        byte[] prefix = FileBytes("{}", 0);
        BinaryPrimitives.WriteUInt64LittleEndian(prefix, (ulong)SafeTensorFile.MaxHeaderLength);
        using var input = new VirtualStream(8L + SafeTensorFile.MaxHeaderLength, prefix, spaceUntil: 8L + SafeTensorFile.MaxHeaderLength);
        using var file = SafeTensorFile.Open(input);
        Assert.Empty(file.Tensors);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(7)]
    [InlineData(8)]
    public void TruncatedPrefixOrHeaderIsInvalid(int length)
    {
        byte[] bytes = new byte[length];
        if (length == 8)
        {
            BinaryPrimitives.WriteUInt64LittleEndian(bytes, 100);
        }

        Assert.Throws<InvalidDataException>(() => SafeTensorFile.Open(new MemoryStream(bytes)));
    }

    [Fact]
    public void LargeOffsetsUse64BitsAndDoNotReadUnrequestedTensors()
    {
        const long offset = (1L << 32) + 4;
        var prefix = FileBytes($$$"""{"large":{"dtype":"U8","shape":[{{{offset}}}],"data_offsets":[0,{{{offset}}}]},"tail":{"dtype":"F32","shape":[1],"data_offsets":[{{{offset}}},{{{offset + 4}}}]}}""", 0);
        using var input = new VirtualStream(prefix.Length + offset + 4, prefix, tail: new byte[] { 0, 0, 128, 63 });
        using var file = SafeTensorFile.Open(input);
        Assert.Equal(prefix.Length, input.BytesRead);
        Assert.Equal(new float[] { 1 }, file.ReadSingle("tail"));
        Assert.Equal(prefix.Length + 4, input.BytesRead);
        Assert.Throws<NotSupportedException>(() => file.ReadBytes("large"));
        Assert.Throws<NotSupportedException>(() => file.Read<byte>("large"));
        Assert.Equal(prefix.Length + 4, input.BytesRead);
    }

    [Fact]
    public void RawCopyCanExceedArrayMaxLengthWithBoundedBuffers()
    {
        long length = (long)Array.MaxLength + 1;
        var prefix = FileBytes($$$"""{"x":{"dtype":"U8","shape":[{{{length}}}],"data_offsets":[0,{{{length}}}]}}""", 0);
        using var input = new VirtualStream(prefix.Length + length, prefix);
        using var file = SafeTensorFile.Open(input);
        using var output = new CountingOutput();
        file.CopyTo("x", output);
        Assert.Equal(length, output.BytesWritten);
        Assert.InRange(output.LargestWrite, 1, 64 * 1024);
        Assert.Equal(prefix.Length + length, input.BytesRead);
    }

    [Fact]
    public void ShortReadsCurrentPositionAndLeaveOpenAreHandled()
    {
        var bytes = new byte[] { 99, 99, 99 }.Concat(FileBytes("""{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}""", 4)).ToArray();
        using var input = new VirtualStream(bytes.Length, bytes, maxRead: 3) { Position = 3 };
        var file = SafeTensorFile.Open(input, leaveOpen: true);
        Assert.Equal(0, file.ReadSingle("x")[0]);
        Assert.Throws<ArgumentException>(() => file.CopyTo("x", input));
        file.Dispose();
        file.Dispose();
        Assert.True(input.CanRead);
        Assert.Single(file.Tensors);
        Assert.Throws<ObjectDisposedException>(() => file.ReadSingle("x"));
        Assert.Throws<ObjectDisposedException>(() => file.Read<int>("x"));
        Assert.Throws<ObjectDisposedException>(() => file.ReadBytes("x"));
        Assert.Throws<ObjectDisposedException>(() => file.CopyTo("x", Stream.Null));
    }

    [Fact]
    public void OwnedStreamsAreClosedOnDisposeAndParsingFailure()
    {
        var input = new MemoryStream(FileBytes("{}", 0));
        SafeTensorFile.Open(input).Dispose();
        Assert.False(input.CanRead);
        input = new MemoryStream(FileBytes("{}", 4));
        Assert.Throws<InvalidDataException>(() => SafeTensorFile.Open(input));
        Assert.False(input.CanRead);
        using var unowned = new MemoryStream(FileBytes("{}", 4));
        Assert.Throws<InvalidDataException>(() => SafeTensorFile.Open(unowned, leaveOpen: true));
        Assert.True(unowned.CanRead);
        Assert.Throws<ArgumentException>(() => SafeTensorFile.Open(new CountingOutput()));
    }

    [Fact]
    public void MissingPathPreservesFileNotFoundException()
    {
        string path = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".safetensors");
        Assert.Throws<FileNotFoundException>(() => SafeTensorFile.Open(path));
    }

    [Fact]
    public void FailedPathParsingReleasesTheFileForExclusiveOverwriteAndDeletion()
    {
        string path = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".safetensors");
        try
        {
            File.WriteAllBytes(path, FileBytes("{}", 4));
            Assert.Throws<InvalidDataException>(() => SafeTensorFile.Open(path));
            using (var output = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None))
            {
                output.Write(FileBytes("{}", 0));
            }

            using (var valid = SafeTensorFile.Open(path))
            {
                Assert.Empty(valid.Tensors);
            }

            File.Delete(path);
            Assert.False(File.Exists(path));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void ReadableNonSeekableInputIsRejectedBeforeReading()
    {
        using var compressed = new GZipStream(new MemoryStream(), CompressionMode.Decompress);
        Assert.True(compressed.CanRead);
        Assert.False(compressed.CanSeek);
        Assert.Throws<ArgumentException>(() => SafeTensorFile.Open(compressed));
    }

    [Theory]
    [InlineData("F32", 4)]
    [InlineData("F16", 2)]
    [InlineData("BF16", 2)]
    [InlineData("F64", 8)]
    public void FloatingConversionsEnforceArrayMaxLengthBeforeReading(string dtype, int width)
    {
        long count = (long)Array.MaxLength + 1, size = count * width;
        var prefix = FileBytes($$$"""{"x":{"dtype":"{{{dtype}}}","shape":[{{{count}}}],"data_offsets":[0,{{{size}}}]}}""", 0);
        using var input = new VirtualStream(prefix.Length + size, prefix);
        using var file = SafeTensorFile.Open(input);
        Assert.Throws<NotSupportedException>(() => file.ReadSingle("x"));
        Assert.Throws<NotSupportedException>(() => file.ReadDouble("x"));
        Assert.Equal(prefix.Length, input.BytesRead);
    }

    [Theory]
    [InlineData("F32", 4)]
    [InlineData("F16", 2)]
    [InlineData("BF16", 2)]
    public void InputTruncationAfterOpenPreservesEndOfStreamException(string dtype, int width)
    {
        var bytes = FileBytes($$$"""{"x":{"dtype":"{{{dtype}}}","shape":[2],"data_offsets":[0,{{{width * 2}}}]}}""", width * 2);
        using var input = new MemoryStream(bytes);
        using var file = SafeTensorFile.Open(input, leaveOpen: true);
        input.SetLength(input.Length - width);
        Assert.Throws<EndOfStreamException>(() => file.ReadSingle("x"));
        Assert.Throws<EndOfStreamException>(() => file.ReadDouble("x"));
        using var destination = new MemoryStream();
        Assert.Throws<EndOfStreamException>(() => file.CopyTo("x", destination));
    }

    internal static byte[] FileBytes(string header, int dataLength)
    {
        byte[] json = Encoding.UTF8.GetBytes(header);
        var result = new byte[8 + json.Length + dataLength];
        BinaryPrimitives.WriteUInt64LittleEndian(result, (ulong)json.Length);
        json.CopyTo(result, 8);
        return result;
    }

    // 大容量のファイル位置と実ヘッダの上限を、大容量の実ファイルを作らずに検証する。
    private sealed class VirtualStream(long length, byte[] prefix, long spaceUntil = 0, byte[]? tail = null, int maxRead = int.MaxValue) : Stream
    {
        private bool disposed;
        public long BytesRead { get; private set; }
        public override long Position { get; set; }
        public override long Length => length;
        public override bool CanRead => !disposed;
        public override bool CanSeek => !disposed;
        public override bool CanWrite => false;
        public override int Read(byte[] buffer, int offset, int count) => Read(buffer.AsSpan(offset, count));
        public override int Read(Span<byte> buffer)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            int count = (int)Math.Min(Math.Min(buffer.Length, maxRead), length - Position);
            var target = buffer[..count];
            target.Clear();
            long fill = Math.Min(count, spaceUntil - Position);
            if (fill > 0)
            {
                target[..(int)fill].Fill((byte)' ');
            }

            Copy(prefix, 0, target);
            if (tail is not null)
            {
                Copy(tail, length - tail.Length, target);
            }

            Position += count;
            BytesRead += count;
            return count;
        }
        private void Copy(byte[] source, long at, Span<byte> target)
        {
            long begin = Math.Max(Position, at), end = Math.Min(Position + target.Length, at + source.Length);
            if (begin < end)
            {
                source.AsSpan((int)(begin - at), (int)(end - begin)).CopyTo(target[(int)(begin - Position)..]);
            }
        }
        protected override void Dispose(bool disposing)
        {
            disposed = true;
            base.Dispose(disposing);
        }
        public override long Seek(long offset, SeekOrigin origin) => Position = origin switch { SeekOrigin.Begin => offset, SeekOrigin.Current => Position + offset, _ => length + offset };
        public override void Flush() { }
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    }

    private sealed class CountingOutput : Stream
    {
        public long BytesWritten { get; private set; }
        public int LargestWrite { get; private set; }
        public override bool CanRead => false;
        public override bool CanSeek => false;
        public override bool CanWrite => true;
        public override long Length => BytesWritten;
        public override long Position { get => BytesWritten; set => throw new NotSupportedException(); }
        public override void Write(byte[] buffer, int offset, int count) => Write(buffer.AsSpan(offset, count));
        public override void Write(ReadOnlySpan<byte> buffer)
        {
            BytesWritten += buffer.Length;
            LargestWrite = Math.Max(LargestWrite, buffer.Length);
        }
        public override void Flush() { }
        public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException();
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
    }
}
