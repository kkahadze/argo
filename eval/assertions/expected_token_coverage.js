const FINAL_MINGRELIAN_VOWELS = new Set(['ა', 'ე', 'ი', 'ო', 'უ', 'ჷ']);

function tokens(text, target) {
  const values = String(text || '')
    .normalize('NFC')
    .toLocaleLowerCase()
    .replace(/[-‐‑‒–—=·•]/gu, '')
    .match(/[\p{L}\p{N}]+/gu) || [];

  if (target !== 'mingrelian') {
    return values;
  }
  return values.map((token) => {
    const last = token[token.length - 1];
    return FINAL_MINGRELIAN_VOWELS.has(last) ? token.slice(0, -1) : token;
  });
}

function coverage(expected, actual) {
  const remaining = new Map();
  for (const token of actual) {
    remaining.set(token, (remaining.get(token) || 0) + 1);
  }

  let matched = 0;
  for (const token of expected) {
    const count = remaining.get(token) || 0;
    if (count > 0) {
      matched += 1;
      remaining.set(token, count - 1);
    }
  }
  return expected.length ? matched / expected.length : 0;
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
  const actual = tokens(output, target);
  const scores = references
    .map((reference) => coverage(tokens(reference, target), actual))
    .filter(Number.isFinite);
  const score = scores.length ? Math.max(...scores) : 0;

  return {
    pass: true,
    score,
    reason: 'Expected-token coverage=' + score.toFixed(3),
  };
};
