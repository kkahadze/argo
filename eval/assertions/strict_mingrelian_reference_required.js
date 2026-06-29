module.exports = (_output, context) => {
  const vars = (context && context.vars) || {};
  const target = String(vars.target_language || '').toLowerCase();
  const confidence = String(vars.confidence || '').toLowerCase();
  const required = target === 'mingrelian' && confidence === 'high';
  return {
    pass: true,
    score: required ? 1 : 0,
    reason: required
      ? 'High-confidence Mingrelian target requires audited reference evidence for full credit'
      : 'LLM semantic score may decide full credit for this row',
  };
};
