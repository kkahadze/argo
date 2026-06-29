module.exports = (output, context) => {
  const text = String(output || '').trim();
  const vars = (context && context.vars) || {};
  const target = String(vars.target_language || '').toLowerCase();
  const lines = text.split(/\n+/).filter(Boolean);
  const mkhedruli = (text.match(/[ა-ჰჱ-ჿ]/g) || []).length;
  const latin = (text.match(/[A-Za-z]/g) || []).length;
  const cyrillic = (text.match(/[\u0400-\u04FF]/g) || []).length;
  const hasWrapper = /(?:^|\s)(?:translation|translated|output|answer|final)(?:\s|:|$)/i.test(text);

  let pass = false;
  if (target === 'mingrelian' || target === 'georgian') {
    pass = mkhedruli > 0 && latin === 0 && cyrillic === 0;
  } else if (target === 'english') {
    pass = latin > 0 && mkhedruli === 0 && cyrillic === 0;
  }
  pass = pass && lines.length <= 2 && !hasWrapper;

  return {
    pass,
    score: pass ? 1 : 0,
    reason: pass
      ? `Concise ${target} script/format output`
      : `Expected concise ${target} output. lines=${lines.length}, mkhedruli=${mkhedruli}, latin=${latin}, cyrillic=${cyrillic}, wrapper=${hasWrapper}`,
  };
};
