using Tuna.Differentiation;
using Tuna.LinearAlgebra;
using Tuna.Statistics.Distributions;
using Xunit.Abstractions;
using static Tuna.Tests.Statistics.DistributionDifferentiationTests;

namespace Tuna.Tests.Statistics;

public class StudentTDistributionDifferentiationTests(ITestOutputHelper output)
{
    [Fact]
    public void StudentTAndCauchyMatchMpmath()
    {
        var tape = new Tape();
        double maxNu = 0, maxLocation = 0, maxScale = 0;
        foreach (var row in DistributionGradients.StudentT)
        {
            tape.Reset();
            var nu = tape.Leaf(row.Nu);
            var location = tape.Leaf(row.Location);
            var scale = tape.Leaf(row.Scale);
            var x = tape.Leaf(row.X);
            var result = StudentTDistribution.LogPdf(nu, location, scale, x);
            double expected = new StudentTDistribution(row.Nu, row.Location, row.Scale).LogPdf(row.X);
            var constant = StudentTDistribution.LogPdf(row.Nu, row.Location, row.Scale, row.X);
            AssertSameBits(expected, constant.Value);
            AssertSameBits(expected, result.Value);
            Assert.False(constant.RequiresGradient);
            tape.Backward(result);

            string context = $"LogPdf({row.Nu:R}, {row.Location:R}, {row.Scale:R}, {row.X:R})";
            maxNu = Math.Max(maxNu, CheckDerivative(row.DNu, tape.GetGradient(nu), $"ν, {context}"));
            maxLocation = Math.Max(maxLocation, CheckDerivative(row.DLocation, tape.GetGradient(location), $"μ, {context}"));
            maxScale = Math.Max(maxScale, CheckDerivative(row.DScale, tape.GetGradient(scale), $"σ, {context}"));
            CheckDerivative(row.DX, tape.GetGradient(x), $"x, {context}");

            if (row.Nu == 1)
            {
                tape.Reset();
                location = tape.Leaf(row.Location);
                scale = tape.Leaf(row.Scale);
                x = tape.Leaf(row.X);
                result = CauchyDistribution.LogPdf(location, scale, x);
                expected = new CauchyDistribution(row.Location, row.Scale).LogPdf(row.X);
                AssertSameBits(expected, result.Value);
                AssertSameBits(expected, CauchyDistribution.LogPdf(row.Location, row.Scale, row.X).Value);
                tape.Backward(result);
                CheckDerivative(row.DLocation, tape.GetGradient(location), $"Cauchy μ, {context}");
                CheckDerivative(row.DScale, tape.GetGradient(scale), $"Cauchy σ, {context}");
                CheckDerivative(row.DX, tape.GetGradient(x), $"Cauchy x, {context}");
            }
        }

        output.WriteLine($"Student t: {DistributionGradients.StudentT.Length} inputs, maximum relative error ν={maxNu:E3}, μ={maxLocation:E3}, σ={maxScale:E3}");
    }

    [Fact]
    public void BatchesMatchMpmath()
    {
        var tape = new Tape();
        double maximum = 0;
        foreach (var row in DistributionGradients.StudentTSums)
        {
            tape.Reset();
            ScalarNode nu = tape.Leaf(row.Nu), location = tape.Leaf(row.Location), scale = tape.Leaf(row.Scale);
            var strided = new double[2 * row.Observations.Length];
            for (int i = 0; i < row.Observations.Length; i++)
            {
                strided[2 * i] = row.Observations[i];
                strided[2 * i + 1] = double.NaN;
            }

            ReadOnlyVectorSpan<double> data = strided.Length == 0 ? default : new(strided, [row.Observations.Length], [2]);
            var result = StudentTDistribution.LogPdf(nu, location, scale, data);
            double expected = row.Observations.Aggregate(0.0, (sum, x) => sum + new StudentTDistribution(row.Nu, row.Location, row.Scale).LogPdf(x));
            AssertSameBits(expected, result.Value);
            AssertSameBits(expected, StudentTDistribution.LogPdf(row.Nu, row.Location, row.Scale, data).Value);
            Array.Fill(strided, double.NaN);
            tape.Backward(result);
            string context = $"LogPdf sum ({row.Nu:R}, {row.Location:R}, {row.Scale:R}, [{string.Join(", ", row.Observations)}])";
            maximum = Math.Max(maximum, CheckDerivative(row.DNu, tape.GetGradient(nu), $"ν, {context}"));
            maximum = Math.Max(maximum, CheckDerivative(row.DLocation, tape.GetGradient(location), $"μ, {context}"));
            maximum = Math.Max(maximum, CheckDerivative(row.DScale, tape.GetGradient(scale), $"σ, {context}"));

            if (row.Nu == 1)
            {
                tape.Reset();
                location = tape.Leaf(row.Location);
                scale = tape.Leaf(row.Scale);
                result = CauchyDistribution.LogPdf(location, scale, new ReadOnlyVectorSpan<double>(row.Observations));
                expected = row.Observations.Aggregate(0.0, (sum, x) => sum + new CauchyDistribution(row.Location, row.Scale).LogPdf(x));
                AssertSameBits(expected, result.Value);
                tape.Backward(result);
                CheckDerivative(row.DLocation, tape.GetGradient(location), $"Cauchy μ, {context}");
                CheckDerivative(row.DScale, tape.GetGradient(scale), $"Cauchy σ, {context}");
            }
        }

        output.WriteLine($"Student t sum: {DistributionGradients.StudentTSums.Length} vectors, maximum relative error {maximum:E3}");
    }

    [Theory]
    [InlineData(0.5, -35.0)]
    [InlineData(4.0, 2.5)]
    [InlineData(31.0, 16.0)]
    public void StudentTMatchesCentralDifferences(double nu, double observation)
    {
        double[] inputs = [nu, 0.3, 1.2, observation];
        var tape = new Tape();
        ScalarNode result = StudentTDistribution.LogPdf(tape.Leaf(inputs[0]), tape.Leaf(inputs[1]), tape.Leaf(inputs[2]), tape.Leaf(inputs[3]));
        Span<double> gradient = stackalloc double[4];
        tape.Gradient(result, new VectorSpan<double>(gradient));
        for (int i = 0; i < inputs.Length; i++)
        {
            int index = i;
            double input = inputs[i];
            double h = 1e-5 * Math.Max(1, Math.Abs(input));
            double coarse = (At(input + h) - At(input - h)) / (2 * h);
            double fine = (At(input + 0.5 * h) - At(input - 0.5 * h)) / h;
            double expected = (4 * fine - coarse) / 3;
            Assert.InRange(Math.Abs((gradient[index] - expected) / expected), 0, 3e-7);

            double At(double value) => new StudentTDistribution(
                index == 0 ? value : inputs[0], index == 1 ? value : inputs[1], index == 2 ? value : inputs[2])
                .LogPdf(index == 3 ? value : inputs[3]);
        }
    }

    [Fact]
    public void InfiniteDegreesOfFreedomUsesTheNormalLimit()
    {
        var data = new ReadOnlyVectorSpan<double>([-1.0, 0, 2]);
        var tape = new Tape();
        var location = tape.Leaf(1);
        var scale = tape.Leaf(2);
        var expected = NormalDistribution.LogPdf(location, scale, data);
        Span<double> reference = stackalloc double[2];
        double expectedValue = tape.Gradient(expected, new VectorSpan<double>(reference));

        tape.Reset();
        var nu = tape.Leaf(double.PositiveInfinity);
        var result = StudentTDistribution.LogPdf(nu, tape.Leaf(1), tape.Leaf(2), data);
        Span<double> gradient = stackalloc double[3];
        tape.Gradient(result, new VectorSpan<double>(gradient));
        AssertSameBits(expectedValue, result.Value);
        Assert.Equal(0, gradient[0]);
        AssertSameBits(reference[0], gradient[1]);
        AssertSameBits(reference[1], gradient[2]);
    }

    [Fact]
    public void DegreesOfFreedomBatchKeepsSubnormalContributions()
    {
        var tape = new Tape();
        var nu = tape.Leaf(1e162);
        var observations = new ReadOnlyVectorSpan<double>(new double[4096]);
        tape.Backward(StudentTDistribution.LogPdf(nu, 0, 1, observations));
        // 4096A′(10¹⁶²) ≈ 1024/10³²⁴、最小非正規化数の約207倍。
        Assert.Equal(207 * double.Epsilon, tape.GetGradient(nu));
    }

    [Fact]
    public void InfiniteObservationDeterminesTheBatchDegreesOfFreedomLimit()
    {
        var tape = new Tape();
        var nu = tape.Leaf(double.Epsilon);
        var observations = new ReadOnlyVectorSpan<double>([0, double.PositiveInfinity]);
        tape.Backward(StudentTDistribution.LogPdf(nu, 0, 1, observations));
        Assert.Equal(double.NegativeInfinity, tape.GetGradient(nu));
    }

    [Fact]
    public void BatchTapeDoesNotGrowWithTheNumberOfObservations()
    {
        var tape = new Tape();
        double[] data = new double[8192];
        Array.Fill(data, 1.25);
        Evaluate(1);
        long before = GC.GetAllocatedBytesForCurrentThread();
        Evaluate(data.Length);
        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);

        void Evaluate(int count)
        {
            tape.Reset();
            ScalarNode nu = tape.Leaf(4), location = tape.Leaf(0), scale = tape.Leaf(1);
            var view = new ReadOnlyVectorSpan<double>(data.AsSpan(0, count));
            tape.Backward(StudentTDistribution.LogPdf(nu, location, scale, view) + CauchyDistribution.LogPdf(location, scale, view));
        }
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.NegativeInfinity)]
    [InlineData(double.PositiveInfinity)]
    public void NonfiniteObservationsHaveConsistentGradients(double x)
    {
        var tape = new Tape();
        ScalarNode nu = tape.Leaf(4), location = tape.Leaf(1), scale = tape.Leaf(2), observation = tape.Leaf(x);
        var result = StudentTDistribution.LogPdf(nu, location, scale, observation);
        AssertSameBits(new StudentTDistribution(4, 1, 2).LogPdf(x), result.Value);
        tape.Backward(result);
        if (double.IsNaN(x))
        {
            Assert.True(double.IsNaN(tape.GetGradient(nu)));
            Assert.True(double.IsNaN(tape.GetGradient(location)));
            Assert.True(double.IsNaN(tape.GetGradient(scale)));
            Assert.True(double.IsNaN(tape.GetGradient(observation)));
            return;
        }

        Assert.Equal(double.NegativeInfinity, tape.GetGradient(nu));
        Assert.Equal(0, tape.GetGradient(location));
        Assert.Equal(2, tape.GetGradient(scale));
        Assert.Equal(0, tape.GetGradient(observation));
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.NegativeInfinity)]
    [InlineData(double.PositiveInfinity)]
    public void CauchyNonfiniteObservationsHaveConsistentGradients(double x)
    {
        var tape = new Tape();
        ScalarNode location = tape.Leaf(1), scale = tape.Leaf(2), observation = tape.Leaf(x);
        var result = CauchyDistribution.LogPdf(location, scale, observation);
        AssertSameBits(new CauchyDistribution(1, 2).LogPdf(x), result.Value);
        tape.Backward(result);
        if (double.IsNaN(x))
        {
            Assert.True(double.IsNaN(tape.GetGradient(location)));
            Assert.True(double.IsNaN(tape.GetGradient(scale)));
            Assert.True(double.IsNaN(tape.GetGradient(observation)));
            return;
        }

        Assert.Equal(0, tape.GetGradient(location));
        Assert.Equal(0.5, tape.GetGradient(scale));
        Assert.Equal(0, tape.GetGradient(observation));
    }

    [Fact]
    public void CauchyLocationDerivativeSurvivesStandardizationOverflow()
    {
        var tape = new Tape();
        var location = tape.Leaf(-1.6070603001241967e308);
        var scale = tape.Leaf(1.6e-322);
        var x = tape.Leaf(3.984818874183486e306);
        var result = CauchyDistribution.LogPdf(location, scale, x);
        Assert.Equal(double.NegativeInfinity, result.Value);
        tape.Backward(result);
        const double expected = 1.214396557866483e-308;
        CheckDerivative(expected, tape.GetGradient(location), "Cauchy location after standardization overflow");
        CheckDerivative(-expected, tape.GetGradient(x), "Cauchy variate after standardization overflow");
        Assert.Equal(double.PositiveInfinity, tape.GetGradient(scale));
    }
}
