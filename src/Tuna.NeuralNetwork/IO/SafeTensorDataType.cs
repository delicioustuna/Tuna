namespace Tuna.NeuralNetwork.IO;

/// <summary>
/// Specifies the stored element data type of a tensor in a safetensors file.
/// <para>safetensors ファイルに保存されているテンソルの要素型を表します。</para>
/// </summary>
/// <remarks>
/// <para>This enum represents all data types defined by the safetensors format, independent of which types are supported for numerical conversion or computation.</para>
/// <para>この列挙型は safetensors 形式で定義された保存型を表し、演算や数値変換の対応範囲とは独立しています。</para>
/// </remarks>
public enum SafeTensorDataType
{
    /// <summary>
    /// 1-byte boolean (<c>BOOL</c>).
    /// <para>1 バイトの真偽値（<c>BOOL</c>）。</para>
    /// </summary>
    Boolean,
    /// <summary>
    /// 4-bit floating-point (<c>F4</c>).
    /// <para>4 ビット浮動小数点数（<c>F4</c>）。</para>
    /// </summary>
    Float4,
    /// <summary>
    /// 6-bit floating-point with 2 exponent bits and 3 mantissa bits (<c>F6_E2M3</c>).
    /// <para>6 ビット浮動小数点数（指数 2 ビット、仮数 3 ビット、<c>F6_E2M3</c>）。</para>
    /// </summary>
    Float6E2M3,
    /// <summary>
    /// 6-bit floating-point with 3 exponent bits and 2 mantissa bits (<c>F6_E3M2</c>).
    /// <para>6 ビット浮動小数点数（指数 3 ビット、仮数 2 ビット、<c>F6_E3M2</c>）。</para>
    /// </summary>
    Float6E3M2,
    /// <summary>
    /// Unsigned 8-bit integer (<c>U8</c>).
    /// <para>符号なし 8 ビット整数（<c>U8</c>）。</para>
    /// </summary>
    UInt8,
    /// <summary>
    /// Signed 8-bit integer (<c>I8</c>).
    /// <para>符号付き 8 ビット整数（<c>I8</c>）。</para>
    /// </summary>
    Int8,
    /// <summary>
    /// 8-bit floating-point in E5M2 format (<c>F8_E5M2</c>).
    /// <para>8 ビット浮動小数点数（E5M2 形式、<c>F8_E5M2</c>）。</para>
    /// </summary>
    Float8E5M2,
    /// <summary>
    /// 8-bit floating-point in E4M3 format (<c>F8_E4M3</c>).
    /// <para>8 ビット浮動小数点数（E4M3 形式、<c>F8_E4M3</c>）。</para>
    /// </summary>
    Float8E4M3,
    /// <summary>
    /// 8-bit exponent-only scale in E8M0 format (<c>F8_E8M0</c>).
    /// <para>8 ビットスケール値（E8M0 形式、<c>F8_E8M0</c>）。</para>
    /// </summary>
    Float8E8M0,
    /// <summary>
    /// 8-bit floating-point in E4M3FNUZ format (<c>F8_E4M3FNUZ</c>).
    /// <para>8 ビット浮動小数点数（E4M3FNUZ 形式、<c>F8_E4M3FNUZ</c>）。</para>
    /// </summary>
    Float8E4M3Fnuz,
    /// <summary>
    /// 8-bit floating-point in E5M2FNUZ format (<c>F8_E5M2FNUZ</c>).
    /// <para>8 ビット浮動小数点数（E5M2FNUZ 形式、<c>F8_E5M2FNUZ</c>）。</para>
    /// </summary>
    Float8E5M2Fnuz,
    /// <summary>
    /// Signed 16-bit integer (<c>I16</c>).
    /// <para>符号付き 16 ビット整数（<c>I16</c>）。</para>
    /// </summary>
    Int16,
    /// <summary>
    /// Unsigned 16-bit integer (<c>U16</c>).
    /// <para>符号なし 16 ビット整数（<c>U16</c>）。</para>
    /// </summary>
    UInt16,
    /// <summary>
    /// IEEE 754 binary16 floating-point (<c>F16</c>).
    /// <para>IEEE 754 binary16 浮動小数点数（<c>F16</c>）。</para>
    /// </summary>
    Float16,
    /// <summary>
    /// 16-bit brain floating-point (<c>BF16</c>).
    /// <para>bfloat16 浮動小数点数（<c>BF16</c>）。</para>
    /// </summary>
    BFloat16,
    /// <summary>
    /// Signed 32-bit integer (<c>I32</c>).
    /// <para>符号付き 32 ビット整数（<c>I32</c>）。</para>
    /// </summary>
    Int32,
    /// <summary>
    /// Unsigned 32-bit integer (<c>U32</c>).
    /// <para>符号なし 32 ビット整数（<c>U32</c>）。</para>
    /// </summary>
    UInt32,
    /// <summary>
    /// IEEE 754 binary32 floating-point (<c>F32</c>).
    /// <para>IEEE 754 binary32 浮動小数点数（<c>F32</c>）。</para>
    /// </summary>
    Float32,
    /// <summary>
    /// 64-bit complex number composed of two IEEE 754 binary32 components (<c>C64</c>).
    /// <para>実部と虚部がそれぞれ IEEE 754 binary32 の複素数（<c>C64</c>）。</para>
    /// </summary>
    Complex64,
    /// <summary>
    /// IEEE 754 binary64 floating-point (<c>F64</c>).
    /// <para>IEEE 754 binary64 浮動小数点数（<c>F64</c>）。</para>
    /// </summary>
    Float64,
    /// <summary>
    /// Signed 64-bit integer (<c>I64</c>).
    /// <para>符号付き 64 ビット整数（<c>I64</c>）。</para>
    /// </summary>
    Int64,
    /// <summary>
    /// Unsigned 64-bit integer (<c>U64</c>).
    /// <para>符号なし 64 ビット整数（<c>U64</c>）。</para>
    /// </summary>
    UInt64,
}

internal static class SafeTensorTypes
{
    internal static SafeTensorDataType Parse(string? name) => name switch
    {
        "BOOL" => SafeTensorDataType.Boolean,
        "F4" => SafeTensorDataType.Float4,
        "F6_E2M3" => SafeTensorDataType.Float6E2M3,
        "F6_E3M2" => SafeTensorDataType.Float6E3M2,
        "U8" => SafeTensorDataType.UInt8,
        "I8" => SafeTensorDataType.Int8,
        "F8_E5M2" => SafeTensorDataType.Float8E5M2,
        "F8_E4M3" => SafeTensorDataType.Float8E4M3,
        "F8_E8M0" => SafeTensorDataType.Float8E8M0,
        "F8_E4M3FNUZ" => SafeTensorDataType.Float8E4M3Fnuz,
        "F8_E5M2FNUZ" => SafeTensorDataType.Float8E5M2Fnuz,
        "I16" => SafeTensorDataType.Int16,
        "U16" => SafeTensorDataType.UInt16,
        "F16" => SafeTensorDataType.Float16,
        "BF16" => SafeTensorDataType.BFloat16,
        "I32" => SafeTensorDataType.Int32,
        "U32" => SafeTensorDataType.UInt32,
        "F32" => SafeTensorDataType.Float32,
        "C64" => SafeTensorDataType.Complex64,
        "F64" => SafeTensorDataType.Float64,
        "I64" => SafeTensorDataType.Int64,
        "U64" => SafeTensorDataType.UInt64,
        _ => throw new NotSupportedException($"The safetensors dtype '{name}' is not supported."),
    };

    internal static int BitWidth(SafeTensorDataType type) => type switch
    {
        SafeTensorDataType.Float4 => 4,
        SafeTensorDataType.Float6E2M3 or SafeTensorDataType.Float6E3M2 => 6,
        SafeTensorDataType.Int16 or SafeTensorDataType.UInt16 or SafeTensorDataType.Float16 or SafeTensorDataType.BFloat16 => 16,
        SafeTensorDataType.Int32 or SafeTensorDataType.UInt32 or SafeTensorDataType.Float32 => 32,
        SafeTensorDataType.Int64 or SafeTensorDataType.UInt64 or SafeTensorDataType.Float64 or SafeTensorDataType.Complex64 => 64,
        SafeTensorDataType.Boolean or SafeTensorDataType.UInt8 or SafeTensorDataType.Int8
            or SafeTensorDataType.Float8E5M2 or SafeTensorDataType.Float8E4M3 or SafeTensorDataType.Float8E8M0
            or SafeTensorDataType.Float8E4M3Fnuz or SafeTensorDataType.Float8E5M2Fnuz => 8,
        _ => throw new NotSupportedException($"The dtype {type} has no known bit width."),
    };

    internal static SafeTensorDataType For<T>() where T : unmanaged
    {
        if (typeof(T) == typeof(byte))
        {
            return SafeTensorDataType.UInt8;
        }

        if (typeof(T) == typeof(sbyte))
        {
            return SafeTensorDataType.Int8;
        }

        if (typeof(T) == typeof(short))
        {
            return SafeTensorDataType.Int16;
        }

        if (typeof(T) == typeof(ushort))
        {
            return SafeTensorDataType.UInt16;
        }

        if (typeof(T) == typeof(int))
        {
            return SafeTensorDataType.Int32;
        }

        if (typeof(T) == typeof(uint))
        {
            return SafeTensorDataType.UInt32;
        }

        if (typeof(T) == typeof(long))
        {
            return SafeTensorDataType.Int64;
        }

        if (typeof(T) == typeof(ulong))
        {
            return SafeTensorDataType.UInt64;
        }

        if (typeof(T) == typeof(Half))
        {
            return SafeTensorDataType.Float16;
        }

        if (typeof(T) == typeof(float))
        {
            return SafeTensorDataType.Float32;
        }

        if (typeof(T) == typeof(double))
        {
            return SafeTensorDataType.Float64;
        }

        throw new NotSupportedException($"The element type {typeof(T).Name} has no supported safetensors array representation. Use ReadBytes or CopyTo for raw data.");
    }
}
