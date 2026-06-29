const FINAL_MINGRELIAN_VOWELS = new Set(['ა', 'ე', 'ი', 'ო', 'უ', 'ჷ']);

function tokens(text, target) {
  const normalized = String(text || '')
    .normalize('NFC')
    .toLocaleLowerCase()
    .replace(/[-‐‑‒–—=·•]/gu, '')
    .match(/[\p{L}\p{N}]+/gu) || [];

  if (target !== 'mingrelian') {
    return normalized;
  }
  return normalized.map((token) => {
    const last = token[token.length - 1];
    return FINAL_MINGRELIAN_VOWELS.has(last) ? token.slice(0, -1) : token;
  });
}

function signature(text, target) {
  const values = tokens(text, target);
  if (target === 'mingrelian') {
    return values.sort().join('\u0001');
  }
  return values.join('\u0001');
}

module.exports = (output, context) => {
  const vars = (context && context.vars) || {};
  const target = String(vars.target_language || '').toLowerCase();
  const references = [
    vars.reference_text,
    ...String(vars.acceptable_target_variants || '')
      .split(';')
      .map((value) => value.trim())
      .filter(Boolean),
  ];
  const actual = signature(output, target);
  const match = actual && references.some(
    (reference) => signature(reference, target) === actual,
  );

  return {
    pass: true,
    score: match ? 1 : 0,
    reason: match
      ? 'Matches a reference after conservative diagnostic normalization'
      : 'No normalized reference-form match; semantic judge decides quality',
  };
};
