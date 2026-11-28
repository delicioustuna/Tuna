using static Tuna.XMath.Constants;
using static Tuna.NumericConstants;

namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        // Debye の一様漸近展開による ln(e⁻ˣ I_ν(x)) (ν ≥ DebyeMinOrder, 0 < x < ∞; DLMF 10.41.3)。
        // R = √(ν² + x²), p = ν/R として ln I_ν(x) = R + ν ln(x/(ν + R)) - ln √(2πR) + ln Σ U_k(p)/νᵏ。
        // x ≥ 1.25ν では指数部 νη - x の桁落ちを避けるため BesselPhaseCorrection を用いる
        private static double LogBesselIScaledDebye(double nu, double x)
        {
            // ν や x が double.MaxValue に近いと R や R + x がオーバーフローするため、2 の冪 (1/4) でスケーリングして丸めを保ったまま比を計算する (ln R = ln sR - ln s)
            double s = Math.Max(nu, x) > DebyeHugeArgument ? 0.25 : 1;
            double sr = double.Hypot(s * nu, s * x);
            double p = s * nu / sr;
            double logR = s == 1 ? Math.Log(sr) : Math.Log(sr) + 2 * Ln2;
            if (nu / x <= 0.8)
            {
                var exponent = BesselPhaseCorrection(nu, x, alternating: true);
                double rest = Log1p(DebyeSeriesTail(p, nu)) - (LnSqrtTwoPi + 0.5 * logR);
                return -exponent.Hi + (rest - exponent.Lo);
            }

            // 大きな x での桁落ちを避けるため R - x = ν²/(R + x) とし、ν² や ν + R のオーバーフローを避けて比で計算する
            double rMinusX = nu * (s * nu / (sr + s * x));
            double ratio = s * x / sr / (1 + p);
            double logRatio = ratio >= MinNormal ? Math.Log(ratio) : Math.Log(x) - logR - Log1p(p);
            return rMinusX + nu * logRatio - (LnSqrtTwoPi + 0.5 * logR) + Log1p(DebyeSeriesTail(p, nu));
        }

        // max(ν, x) が 2¹⁰²⁰ を超えると R, R + x, 2ν + 1 がオーバーフローしうる (2¹⁰²⁰ 以下なら R₀ + R₁ ≤ 2√2·2¹⁰²⁰ < 2¹⁰²⁴)
        private const double DebyeHugeArgument = 1.1235582092889474e307;

        // Debye の一様漸近展開による比 I_(ν+1)(x)/I_ν(x) (ν ≥ DebyeMinOrder, 0 < x < ∞)。
        // 2 つの ln I の差を直接取ると O(x) の項が桁落ちするため、R₀ = √(ν² + x²), R₁ = √((ν+1)² + x²) として項ごとに解析的に差を計算する
        private static double BesselIRatioDebye(double nu, double x)
        {
            // R₀ + R₁ や 2ν + 1 のオーバーフローを避けるため、長さの次元を持つ量を 1/4 にスケーリングして比のみを作る
            double s = Math.Max(nu, x) > DebyeHugeArgument ? 0.25 : 1;
            double nu1 = nu + 1;
            double sr0 = double.Hypot(s * nu, s * x);
            double sr1 = double.Hypot(s * nu1, s * x);

            // R₁ - R₀ = (2ν + 1)/(R₀ + R₁)
            double dr = (s * nu + s * nu1) / (sr0 + sr1);

            // (ν + 1 + R₁)/(ν + R₀) = 1 + (1 + R₁ - R₀)/(ν + R₀)。ν + R₀ のオーバーフローを避けて R₀(1 + p₀) を使う
            double p0 = s * nu / sr0;
            double p1 = s * nu1 / sr1;
            double exponent = dr - nu * Log1p((1 + dr) * s / sr0 / (1 + p0)) - 0.5 * Log1p(dr * s / sr0)
                + (Log1p(DebyeSeriesTail(p1, nu1)) - Log1p(DebyeSeriesTail(p0, nu)));

            // x/(ν + 1 + R₁) ≤ 1 は x が非正規化数でもそのまま計算できる
            return s * x / sr1 / (1 + p1) * Math.Exp(exponent);
        }
    }
}
