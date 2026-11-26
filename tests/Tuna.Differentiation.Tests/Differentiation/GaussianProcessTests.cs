using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.Tests.Differentiation;

/// <summary>ガウス過程回帰の対数周辺尤度の勾配（自動微分と、解析式 ½tr(W·∂C/∂θ)、W = ααᵀ − C⁻¹ の比較）</summary>
public class GaussianProcessTests
{
    [Theory]
    [InlineData(16)]
    [InlineData(128)]
    public void LogMarginalLikelihoodGradientMatchesTheTraceFormula(int observations)
    {
        var process = new GaussianProcessExample(observations);
        double[] expected = new double[3];
        double manual = process.Manual(expected);
        var tape = new Tape();
        double[] actual = new double[3];
        // 値は手で書いた計算（XMath と CholeskyDecomposition）と同じビット。
        DifferentiationAssert.EqualBits(manual, process.Automatic(tape, actual, lu: false));
        for (int k = 0; k < 3; k++)
        {
            DifferentiationAssert.Close(expected[k], actual[k], 1e-10, 1e-10 * observations, $"θ[{k}] (Cholesky)");
        }

        // 同じモデルを LU 分解（XMath.Solve と ln |det C|）で書いても同じ勾配になる。
        double viaLu = process.Automatic(tape, actual, lu: true);
        DifferentiationAssert.Close(manual, viaLu, 1e-12, 1e-12 * observations, "value (LU)");
        for (int k = 0; k < 3; k++)
        {
            DifferentiationAssert.Close(expected[k], actual[k], 1e-9, 1e-9 * observations, $"θ[{k}] (LU)");
        }
    }

    [Fact]
    public void RepeatedEvaluationsGiveTheSameBits()
    {
        var process = new GaussianProcessExample(40);
        var tape = new Tape();
        double[] first = new double[3], again = new double[3], other = new double[3];
        double value = process.Automatic(tape, first, lu: false);
        // LU 分解の記録をはさんでも、テープの使い回しは値と勾配のビットを変えない。
        for (int i = 0; i < 3; i++)
        {
            process.Automatic(tape, other, lu: true);
            DifferentiationAssert.EqualBits(value, process.Automatic(tape, again, lu: false));
            Assert.Equal(first, again);
        }
    }
}

/// <summary>
/// 1次元の入力のガウス過程回帰（二乗指数カーネル）。ハイパーパラメーター θ = (log ℓ, log σf², log σn²) の対数周辺尤度
/// <c>−½yᵀC⁻¹y − ½ln det C − (n/2)ln 2π</c>（<c>C = K + σn²I</c>）と、その勾配。
/// </summary>
internal sealed class GaussianProcessExample
{
    private readonly int observations;
    private readonly double[] parameters = [Math.Log(0.7), Math.Log(1.3), Math.Log(0.05)];
    private readonly Matrix<double> squaredDistances, identity, kernel, weighted, covariance, factor, inverse;
    private readonly LinearAlgebra.Vector<double> y, alpha, product;

    public GaussianProcessExample(int observations)
    {
        this.observations = observations;
        var random = new Random(observations + 7);
        var inputs = new double[observations];
        y = new LinearAlgebra.Vector<double>(observations);
        for (int i = 0; i < observations; i++)
        {
            inputs[i] = 5 * random.NextDouble();
            y[i] = Math.Sin(inputs[i]) + 0.2 * (random.NextDouble() - 0.5);
        }

        squaredDistances = new Matrix<double>(observations, observations);
        identity = new Matrix<double>(observations, observations);
        for (int j = 0; j < observations; j++)
        {
            identity[j, j] = 1;
            for (int i = 0; i < observations; i++)
            {
                double difference = inputs[i] - inputs[j];
                squaredDistances[i, j] = difference * difference;
            }
        }

        kernel = new Matrix<double>(observations, observations);
        weighted = new Matrix<double>(observations, observations);
        covariance = new Matrix<double>(observations, observations);
        factor = new Matrix<double>(observations, observations);
        inverse = new Matrix<double>(observations, observations);
        alpha = new LinearAlgebra.Vector<double>(observations);
        product = new LinearAlgebra.Vector<double>(observations);
    }

    public int Leaves => parameters.Length;

    // 自動微分。値を返し、勾配を gradient に書く。lu なら C⁻¹y を XMath.Solve で、ln det C を LU 分解で求める。
    public double Automatic(Tape tape, double[] gradient, bool lu)
    {
        tape.Reset();
        var logLength = tape.Leaf(parameters[0]);
        var logSignal = tape.Leaf(parameters[1]);
        var logNoise = tape.Leaf(parameters[2]);
        ScalarNode lengthScale = XMath.Exp(logLength), signal = XMath.Exp(logSignal), noise = XMath.Exp(logNoise);
        var k = XMath.Multiply(XMath.Exp(XMath.Multiply(tape.Constant(squaredDistances), -0.5 / (lengthScale * lengthScale))), signal);
        var c = k + XMath.Multiply(tape.Constant(identity), noise);
        var targets = tape.Constant(y);
        ScalarNode value;
        if (lu)
        {
            value = -0.5 * XMath.Dot(targets, XMath.Solve(c, targets)) - 0.5 * LuDecompositionNode.Compute(c).LogAbsDeterminant
                - 0.5 * observations * Math.Log(2 * Math.PI);
        }
        else
        {
            var cholesky = CholeskyDecompositionNode.Compute(c);
            value = -0.5 * XMath.Dot(targets, cholesky.Solve(targets)) - 0.5 * cholesky.LogDeterminant - 0.5 * observations * Math.Log(2 * Math.PI);
        }

        return tape.Gradient(value, new VectorSpan<double>(gradient));
    }

    // 解析式。∂C/∂log ℓ = K∘D²/ℓ²、∂C/∂log σf² = K、∂C/∂log σn² = σn²·I。½tr(W·S) = ½(αᵀSα − Σ C⁻¹∘S)（S は対称）。
    public double Manual(double[] result)
    {
        double lengthScale = Math.Exp(parameters[0]), signal = Math.Exp(parameters[1]), noise = Math.Exp(parameters[2]);
        XMath.Multiply(squaredDistances, -0.5 / (lengthScale * lengthScale), kernel);
        XMath.Exp(kernel, kernel);
        XMath.Multiply(kernel, signal, kernel);
        kernel.AsSpan().CopyTo(covariance.AsSpan());
        for (int i = 0; i < observations; i++)
        {
            covariance[i, i] += noise;
        }

        var cholesky = CholeskyDecomposition.Compute(covariance, factor);
        cholesky.Solve(y, alpha);
        double value = -0.5 * XMath.Dot(y, alpha) - 0.5 * cholesky.LogDeterminant - 0.5 * observations * Math.Log(2 * Math.PI);
        cholesky.Inverse(inverse);
        XMath.Hadamard(kernel, squaredDistances, weighted);
        result[0] = 0.5 * (QuadraticForm(weighted) - XMath.Dot(inverse.AsVector(), weighted.AsVector())) / (lengthScale * lengthScale);
        result[1] = 0.5 * (QuadraticForm(kernel) - XMath.Dot(inverse.AsVector(), kernel.AsVector()));
        double trace = 0;
        for (int i = 0; i < observations; i++)
        {
            trace += inverse[i, i];
        }

        result[2] = 0.5 * noise * (XMath.Dot(alpha, alpha) - trace);
        return value;
    }

    private double QuadraticForm(Matrix<double> a)
    {
        XMath.Multiply(a, alpha, product);
        return XMath.Dot(alpha, product);
    }
}
