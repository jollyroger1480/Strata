/**
 * A short name for a downloaded optimized model: the family and the quantization ("Swift IQ2_XS", "Gyro-S", "IQ3_S"), made from the id the
 * installation gives it (the family tag in front of the lower-cased quantization). The long name stays available as a tooltip.
 */
const FAMILIES: [string, string][] = [['swift-', 'Swift'], ['coder-', 'Coder'], ['unsloth-', 'Unsloth'], ['gyro-gyro-', 'Gyro'], ['gyro-', 'Gyro'], ['nvfp4-', 'NVFP4'], ['qwen36-', 'Qwen3.6']];

export function nativeShortName(id: string): string {
  const lower = String(id ?? '').toLowerCase();
  for (const [prefix, label] of FAMILIES) {
    if (!lower.startsWith(prefix)) continue;
    const rest = lower.slice(prefix.length).toUpperCase();
    if (!rest) return label;
    return label === 'Gyro' ? `Gyro-${rest.replace(/^GYRO-/, '')}` : `${label} ${rest}`;
  }
  return lower.toUpperCase() || String(id);
}
