module.exports = (namedScores) => {
  const quality = Number(namedScores.translation_quality || 0);
  const script = Number(namedScores.target_script_and_format || 0);
  const reference = Number(namedScores.reference_form_diagnostic || 0);
  const strictMingrelian = Number(
    namedScores.strict_mingrelian_reference_required || 0,
  );
  const boundedQuality = Math.max(0, Math.min(1, quality));
  let score = boundedQuality;
  if (reference >= 1) {
    score = Math.max(score, 0.95);
  }
  if (strictMingrelian >= 1 && reference < 1) {
    score = Math.min(score, 0.89);
  }
  if (script < 1) {
    score = Math.min(score, 0.2);
  }
  const pass = script >= 1 && score >= 0.9;

  return {
    pass,
    score,
    reason: [
      `translation_quality=${boundedQuality.toFixed(3)}`,
      `script_format=${script.toFixed(3)}`,
      `reference_form=${reference.toFixed(3)}`,
      `strict_mingrelian_reference=${strictMingrelian.toFixed(3)}`,
      pass ? 'full-quality pass' : 'below full-quality threshold',
    ].join('; '),
  };
};
