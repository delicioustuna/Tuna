# Safetensors IO provenance

The C# reader in this directory is original Tuna implementation code, provided under the MIT license in the package's LICENSE.txt.
It was written from the public file format and the validation requirements of the reference implementation. No Rust/Python source code or binaries from the reference implementation are copied, translated verbatim, or bundled.

References consulted:

- Format: https://github.com/safetensors/safetensors/blob/e246a2560645b7525f5775669ed816eb57c5bcc8/README.md#format
- Validation and dtypes: https://github.com/safetensors/safetensors/blob/e246a2560645b7525f5775669ed816eb57c5bcc8/safetensors/src/tensor.rs
- The reference implementation's license (Apache-2.0): https://github.com/safetensors/safetensors/blob/e246a2560645b7525f5775669ed816eb57c5bcc8/LICENSE

The official Python package is used only as a development tool to generate an interoperability fixture from Tuna's own synthetic numeric constants. It is not a runtime dependency or part of the NuGet package. No NVIDIA model code, weights, audio, or model configuration is incorporated into this reader or the fixtures. Licenses of separately obtained model weights continue to apply to those weights.

The reader validates headers of up to 100,000,000 bytes, uses 64-bit stream positions, and checks shapes and dtype sizes with unsigned 64-bit arithmetic. Array-returning methods enforce Array.MaxLength; CopyTo streams raw data using a 64 KiB buffer. See docs/safetensors.md in the source repository for the API contracts and supported conversions.
