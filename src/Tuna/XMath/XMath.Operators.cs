namespace Tuna;

// SIMD カーネルから生成済み演算子の状態フィールドを参照するための内部アクセサー。
public static partial class XMath
{
    public static partial class Operators
    {
        public readonly partial struct DivideScalar<T>
        {
            internal T Y => y;
        }

        public readonly partial struct MaxScalar<T>
        {
            internal T Y => y;
        }

        public readonly partial struct MinScalar<T>
        {
            internal T Y => y;
        }

        public readonly partial struct Clamp<T>
        {
            internal T Min => min;

            internal T Max => max;
        }

        public readonly partial struct LeakyRelu<T>
        {
            internal T Slope => slope;
        }

        public readonly partial struct Elu<T>
        {
            internal T Alpha => alpha;
        }
    }
}
