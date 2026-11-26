namespace Tuna.NeuralNetwork.Tests;

/// <summary>
/// 定義どおりに書いた倍精度の基準（配列は列優先 [チャネル × 位置]、重みは PyTorch と同じ行優先）。速さは考えない
/// </summary>
internal static class Reference
{
    public static double[] LayerNorm(double[] x, int channels, int positions, double[]? scale, double[]? shift, double epsilon, bool center = true)
    {
        var y = new double[x.Length];
        for (int t = 0; t < positions; t++)
        {
            double mean = 0;
            if (center)
            {
                for (int c = 0; c < channels; c++)
                {
                    mean += x[t * channels + c];
                }

                mean /= channels;
            }

            double variance = 0;
            for (int c = 0; c < channels; c++)
            {
                double d = x[t * channels + c] - mean;
                variance += d * d;
            }

            variance /= channels;
            double inverse = 1 / Math.Sqrt(variance + epsilon);
            for (int c = 0; c < channels; c++)
            {
                double v = (x[t * channels + c] - mean) * inverse;
                y[t * channels + c] = v * (scale?[c] ?? 1) + (shift?[c] ?? 0);
            }
        }

        return y;
    }

    public static double[] Prelu(double[] x, int channels, double[] slopes)
    {
        var y = new double[x.Length];
        for (int i = 0; i < x.Length; i++)
        {
            double a = slopes.Length == 1 ? slopes[0] : slopes[i % channels];
            y[i] = x[i] < 0 ? a * x[i] : x[i];
        }

        return y;
    }

    /// <summary>y = W·x + b（W は行優先 [outputs, inputs]）</summary>
    public static double[] Linear(double[] weight, double[] x, double[]? bias, int outputs, int inputs, int positions)
    {
        var y = new double[outputs * positions];
        for (int t = 0; t < positions; t++)
        {
            for (int o = 0; o < outputs; o++)
            {
                double s = bias?[o] ?? 0;
                for (int i = 0; i < inputs; i++)
                {
                    s += weight[o * inputs + i] * x[t * inputs + i];
                }

                y[t * outputs + o] = s;
            }
        }

        return y;
    }

    /// <summary>y(d, t) = b_d + Σ_k w(d, k)·x(d, t − K + 1 + k)（w は行優先 [D, K]）</summary>
    public static double[] CausalDepthwise(double[] x, double[] weight, double[]? bias, int channels, int positions, int taps)
    {
        var y = new double[channels * positions];
        for (int t = 0; t < positions; t++)
        {
            for (int d = 0; d < channels; d++)
            {
                double s = bias?[d] ?? 0;
                for (int k = 0; k < taps; k++)
                {
                    int source = t - taps + 1 + k;
                    if (source >= 0)
                    {
                        s += weight[d * taps + k] * x[source * channels + d];
                    }
                }

                y[t * channels + d] = s;
            }
        }

        return y;
    }

    /// <summary>
    /// 選択的走査（a は行優先 [D, N]、b・c は列優先 [N × L]、h は行優先 [D, N] で更新する）
    /// </summary>
    public static double[] SelectiveScan(double[] x, double[] delta, double[] a, double[] b, double[] c, double[]? skip,
        int channels, int states, int positions, double[] h)
    {
        var y = new double[channels * positions];
        for (int t = 0; t < positions; t++)
        {
            for (int d = 0; d < channels; d++)
            {
                double dt = delta[t * channels + d], xv = x[t * channels + d];
                double s = (skip?[d] ?? 0) * xv;
                for (int k = 0; k < states; k++)
                {
                    ref double hv = ref h[d * states + k];
                    hv = Math.Exp(dt * a[d * states + k]) * hv + dt * xv * b[t * states + k];
                    s += c[t * states + k] * hv;
                }

                y[t * channels + d] = s;
            }
        }

        return y;
    }

    /// <summary>
    /// PyTorch の nn.Conv2d と同じ定義（入力は [C, W, H] の列優先、重みは行優先 [Cout, Cin, KH, KW]）。出力も [Cout, W', H'] の列優先
    /// </summary>
    public static double[] Convolution2D(double[] x, int channels, int width, int height, double[] weight, double[]? bias,
        int outChannels, int kernelHeight, int kernelWidth, int strideHeight, int strideWidth, int dilationHeight, int dilationWidth,
        Padding2D padding, out int outWidth, out int outHeight)
    {
        outWidth = Math.Max(0, (width + padding.Left + padding.Right - dilationWidth * (kernelWidth - 1) - 1) / strideWidth + 1);
        outHeight = Math.Max(0, (height + padding.Top + padding.Bottom - dilationHeight * (kernelHeight - 1) - 1) / strideHeight + 1);
        if (width + padding.Left + padding.Right - dilationWidth * (kernelWidth - 1) - 1 < 0)
        {
            outWidth = 0;
        }

        if (height + padding.Top + padding.Bottom - dilationHeight * (kernelHeight - 1) - 1 < 0)
        {
            outHeight = 0;
        }

        var y = new double[outChannels * outWidth * outHeight];
        for (int oh = 0; oh < outHeight; oh++)
        {
            for (int ow = 0; ow < outWidth; ow++)
            {
                for (int o = 0; o < outChannels; o++)
                {
                    double s = bias?[o] ?? 0;
                    for (int c = 0; c < channels; c++)
                    {
                        for (int i = 0; i < kernelHeight; i++)
                        {
                            int ih = oh * strideHeight + i * dilationHeight - padding.Top;
                            if (ih < 0 || ih >= height)
                            {
                                continue;
                            }

                            for (int j = 0; j < kernelWidth; j++)
                            {
                                int iw = ow * strideWidth + j * dilationWidth - padding.Left;
                                if (iw < 0 || iw >= width)
                                {
                                    continue;
                                }

                                s += weight[((o * channels + c) * kernelHeight + i) * kernelWidth + j] * x[(ih * width + iw) * channels + c];
                            }
                        }
                    }

                    y[(oh * outWidth + ow) * outChannels + o] = s;
                }
            }
        }

        return y;
    }

    public static double Silu(double x) => x / (1 + Math.Exp(-x));

    public static double Softplus(double x) => x > 30 ? x : Math.Log(1 + Math.Exp(x));

    /// <summary>mamba_ssm の Mamba（状態 0 から1本の系列。重みは PyTorch の行優先）</summary>
    public static double[] Mamba(MambaWeights w, double[] u, int length)
    {
        var (model, inner, n, kernel, rank) = w.Dimensions;
        int projection = rank + 2 * n;
        var xz = Linear(w.InProjection, u, w.InBias, 2 * inner, model, length);
        var x = new double[inner * length];
        var z = new double[inner * length];
        for (int t = 0; t < length; t++)
        {
            Array.Copy(xz, t * 2 * inner, x, t * inner, inner);
            Array.Copy(xz, t * 2 * inner + inner, z, t * inner, inner);
        }

        var xc = CausalDepthwise(x, w.Convolution, w.ConvolutionBias, inner, length, kernel).Select(Silu).ToArray();
        var xdb = Linear(w.XProjection, xc, null, projection, inner, length);
        var dtIn = new double[rank * length];
        var b = new double[n * length];
        var c = new double[n * length];
        for (int t = 0; t < length; t++)
        {
            Array.Copy(xdb, t * projection, dtIn, t * rank, rank);
            Array.Copy(xdb, t * projection + rank, b, t * n, n);
            Array.Copy(xdb, t * projection + rank + n, c, t * n, n);
        }

        var delta = Linear(w.DeltaProjection, dtIn, w.DeltaBias, inner, rank, length).Select(Softplus).ToArray();
        var a = w.ALog.Select(v => -Math.Exp(v)).ToArray();
        var y = SelectiveScan(xc, delta, a, b, c, w.D, inner, n, length, new double[inner * n]);
        for (int i = 0; i < y.Length; i++)
        {
            y[i] *= Silu(z[i]);
        }

        return Linear(w.OutProjection, y, w.OutBias, model, inner, length);
    }
}

/// <summary>乱数で作った Mamba の重み（倍精度、PyTorch の並び）</summary>
internal sealed class MambaWeights
{
    public MambaWeights(MambaDimensions dimensions, int seed, bool projectionBiases = false)
    {
        Dimensions = dimensions;
        var (model, inner, n, kernel, rank) = dimensions;
        InProjection = TestData.Uniform(2 * inner * model, seed, 1 / Math.Sqrt(model));
        Convolution = TestData.Uniform(inner * kernel, seed + 1, 1 / Math.Sqrt(kernel));
        ConvolutionBias = TestData.Uniform(inner, seed + 2, 0.2);
        XProjection = TestData.Uniform((rank + 2 * n) * inner, seed + 3, 1 / Math.Sqrt(inner));
        DeltaProjection = TestData.Uniform(inner * rank, seed + 4, 1 / Math.Sqrt(rank));
        DeltaBias = TestData.Uniform(inner, seed + 5, -4, -1);
        ALog = TestData.Uniform(inner * n, seed + 6, 0, Math.Log(n));
        D = TestData.Uniform(inner, seed + 7, 0.5, 1.5);
        OutProjection = TestData.Uniform(model * inner, seed + 8, 1 / Math.Sqrt(inner));
        InBias = projectionBiases ? TestData.Uniform(2 * inner, seed + 9, 0.1) : null;
        OutBias = projectionBiases ? TestData.Uniform(model, seed + 10, 0.1) : null;
    }

    public MambaDimensions Dimensions { get; }
    public double[] InProjection { get; private set; }
    public double[] Convolution { get; private set; }
    public double[] ConvolutionBias { get; private set; }
    public double[] XProjection { get; private set; }
    public double[] DeltaProjection { get; private set; }
    public double[] DeltaBias { get; private set; }
    public double[] ALog { get; private set; }
    public double[] D { get; private set; }
    public double[] OutProjection { get; private set; }
    public double[]? InBias { get; private set; }
    public double[]? OutBias { get; private set; }

    /// <summary>要素型 T に丸めた重み（基準もこれで計算する）</summary>
    public MambaWeights RoundedTo<T>() where T : System.Numerics.INumberBase<T>
    {
        var copy = (MambaWeights)MemberwiseClone();
        copy.InProjection = TestData.Rounded<T>(InProjection);
        copy.Convolution = TestData.Rounded<T>(Convolution);
        copy.ConvolutionBias = TestData.Rounded<T>(ConvolutionBias);
        copy.XProjection = TestData.Rounded<T>(XProjection);
        copy.DeltaProjection = TestData.Rounded<T>(DeltaProjection);
        copy.DeltaBias = TestData.Rounded<T>(DeltaBias);
        copy.ALog = TestData.Rounded<T>(ALog);
        copy.D = TestData.Rounded<T>(D);
        copy.OutProjection = TestData.Rounded<T>(OutProjection);
        copy.InBias = InBias is null ? null : TestData.Rounded<T>(InBias);
        copy.OutBias = OutBias is null ? null : TestData.Rounded<T>(OutBias);
        return copy;
    }

    public MambaBlock<T> Create<T>() where T : unmanaged, System.Numerics.IFloatingPointIeee754<T> =>
        new(Dimensions, TestData.To<T>(InProjection), TestData.To<T>(Convolution), TestData.To<T>(ConvolutionBias),
            TestData.To<T>(XProjection), TestData.To<T>(DeltaProjection), TestData.To<T>(DeltaBias), TestData.To<T>(ALog),
            TestData.To<T>(D), TestData.To<T>(OutProjection),
            InBias is null ? default : TestData.To<T>(InBias), OutBias is null ? default : TestData.To<T>(OutBias));
}
