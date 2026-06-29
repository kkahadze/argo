const assert = require('assert');
const path = require('path');

const assertions = path.join(__dirname, '..', 'assertions');
const scoring = path.join(__dirname, '..', 'scoring');
const scriptAndFormat = require(path.join(assertions, 'target_script_and_format.js'));
const referenceForm = require(path.join(assertions, 'normalized_reference_form.js'));
const tokenCoverage = require(path.join(assertions, 'expected_token_coverage.js'));
const strictMingrelian = require(path.join(assertions, 'strict_mingrelian_reference_required.js'));
const qualityScore = require(path.join(scoring, 'mingrelian_translation_quality.js'));

function context(vars) {
  return { vars };
}

assert.equal(
  scriptAndFormat('მა ვორექჷ ქორთუ', context({ target_language: 'mingrelian' })).score,
  1,
);
assert.equal(
  scriptAndFormat('ma voreq kortu', context({ target_language: 'mingrelian' })).score,
  0,
);
assert.equal(
  scriptAndFormat('I am Georgian', context({ target_language: 'english' })).score,
  1,
);
assert.equal(
  scriptAndFormat('მე ქართველი ვარ', context({ target_language: 'english' })).score,
  0,
);

assert.equal(
  referenceForm(
    'თენა ოკოდჷ ის-ჷ-თი',
    context({
      target_language: 'mingrelian',
      reference_text: 'თენა ოკოდჷ ისჷთი',
      acceptable_target_variants: '',
    }),
  ).score,
  1,
);

assert.equal(
  tokenCoverage(
    'მა ვორექჷ ქორთუ',
    context({
      target_language: 'mingrelian',
      reference_text: 'მა ვორექი ქორთ',
      acceptable_target_variants: '',
    }),
  ).score,
  1,
);
assert.equal(
  tokenCoverage(
    'მა ქორთუ',
    context({
      target_language: 'mingrelian',
      reference_text: 'მა ვორექი ქორთ',
      acceptable_target_variants: '',
    }),
  ).score,
  2 / 3,
);
assert.equal(
  referenceForm(
    'მა ვორექჷ ქორთუ',
    context({
      target_language: 'mingrelian',
      reference_text: 'მა ვორექი ქორთ',
      acceptable_target_variants: '',
    }),
  ).score,
  1,
);

assert.equal(
  strictMingrelian('', context({ target_language: 'mingrelian', confidence: 'high' })).score,
  1,
);
assert.equal(
  strictMingrelian('', context({ target_language: 'english', confidence: 'high' })).score,
  0,
);

assert.equal(
  qualityScore({
    translation_quality: 0.96,
    target_script_and_format: 1,
    reference_form_diagnostic: 1,
  }).pass,
  true,
);
assert.equal(
  qualityScore({
    translation_quality: 0.96,
    target_script_and_format: 0,
    reference_form_diagnostic: 0,
  }).score,
  0.2,
);
assert.equal(
  qualityScore({
    translation_quality: 1,
    target_script_and_format: 1,
    reference_form_diagnostic: 0,
    strict_mingrelian_reference_required: 1,
  }).score,
  0.89,
);
assert.equal(
  qualityScore({
    translation_quality: 0.4,
    target_script_and_format: 1,
    reference_form_diagnostic: 1,
    strict_mingrelian_reference_required: 1,
  }).score,
  0.95,
);

console.log('mingrelian quality assertions: ok');
