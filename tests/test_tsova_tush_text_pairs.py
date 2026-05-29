import unittest

from src.tsova_tush.text_pair_extraction import (
    TextPairCandidate,
    iter_numbered_translation_pairs,
)


class TsovaTushTextPairExtractionTests(unittest.TestCase):
    def test_extracts_adjacent_numbered_bilingual_runs_with_multiline_items(self) -> None:
        fixture = """
        წკიპუინ ამბუ

        10. ცჰ>ა დაჰ> ღორას ო მაუმბარჩ,
        ო ხალხულრ, ო ბაცბილო.
        11. ლე, მაქ ხახენე დონნ.

        10

        წკიპოს ამბავი

        10. ერთი წავალ იმ ალაზნისთავში,
        იმ ხალხში, იმ თუშებში.
        11. შეჯდა ცხენზე და გადმოვიდა.
        """

        pairs = list(
            iter_numbered_translation_pairs(
                fixture,
                source_id="part1-fixture",
            )
        )

        self.assertEqual(
            pairs,
            [
                TextPairCandidate(
                    source_id="part1-fixture:run-001:item-010",
                    batsbi_text="ცჰ>ა დაჰ> ღორას ო მაუმბარჩ, ო ხალხულრ, ო ბაცბილო.",
                    georgian_translation=(
                        "ერთი წავალ იმ ალაზნისთავში, იმ ხალხში, იმ თუშებში."
                    ),
                    english_translation=None,
                    confidence=1.0,
                    notes="adjacent numbered runs with identical item sequence",
                ),
                TextPairCandidate(
                    source_id="part1-fixture:run-001:item-011",
                    batsbi_text="ლე, მაქ ხახენე დონნ.",
                    georgian_translation="შეჯდა ცხენზე და გადმოვიდა.",
                    english_translation=None,
                    confidence=1.0,
                    notes="adjacent numbered runs with identical item sequence",
                ),
            ],
        )

    def test_rejects_runs_without_identical_number_sequences(self) -> None:
        fixture = """
        10. ცჰ>ა დაჰ> ღორას.
        11. ლე, მაქ ხახენე.

        10. ერთი წავალ იმ ალაზნისთავში.
        12. ეს ნომერი აღარ ემთხვევა.
        """

        pairs = list(
            iter_numbered_translation_pairs(
                fixture,
                source_id="mismatch-fixture",
            )
        )

        self.assertEqual(pairs, [])

    def test_ignores_too_short_numbered_fragments(self) -> None:
        fixture = """
        10. ცჰ>ა დაჰ> ღორას.

        10. ერთი წავალ იმ ალაზნისთავში.
        """

        pairs = list(
            iter_numbered_translation_pairs(
                fixture,
                source_id="short-fixture",
            )
        )

        self.assertEqual(pairs, [])


if __name__ == "__main__":
    unittest.main()
