"""§3C.7 genetics learning — exact crosses, linkage, chi-square, lessons.

Pins the engine in nomorals/learn/genetics.py and the /learn genetics
front door in nomorals/learn/chat.py (which `nm learn genetics` also
drives). Every ratio below is hand-computed from first principles —
the tests pin the genetics, not the implementation.
"""

import unittest
from fractions import Fraction

from nomorals.learn.chat import control_learn
from nomorals.learn.genetics import (
    GeneticsError,
    _gametes,
    _gammq,
    alleles_for,
    chi_square,
    get_allele,
    get_trait,
    lesson,
    list_lessons,
    parse_genotype,
    parse_ratios,
    predict,
    score_prediction,
    simulate,
)


def cross(mother, father):
    return predict(parse_genotype(mother), parse_genotype(father))


class CatalogueTests(unittest.TestCase):
    def test_nine_traits_twenty_nine_alleles(self):
        from nomorals.learn.genetics import ALLELES, TRAITS

        self.assertEqual(len(TRAITS), 9)
        self.assertEqual(len(ALLELES), 29)

    def test_allele_codes_unique_across_traits(self):
        from nomorals.learn.genetics import ALLELES

        codes = [a.code for a in ALLELES.values()]
        self.assertEqual(len(codes), len(set(codes)))

    def test_unknown_trait_and_allele_refuse_by_name(self):
        with self.assertRaises(GeneticsError) as ctx:
            get_trait("NOPE")
        self.assertEqual(ctx.exception.code, "unknown_trait")
        with self.assertRaises(GeneticsError) as ctx:
            get_allele("ZZ")
        self.assertEqual(ctx.exception.code, "unknown_allele")

    def test_alleles_for_in_map_order(self):
        alleles = alleles_for("BC")
        self.assertEqual([a.code for a in alleles],
                         ["BL", "E", "Y", "T", "S"])
        # X first by chromosome string, then map position within each.
        self.assertEqual([a.chromo for a in alleles], [2, 2, "X", "X", "X"])


class ParseTests(unittest.TestCase):
    def test_parse_forms(self):
        geno = parse_genotype("f E/WT AR/WT")
        self.assertEqual(geno.sex, "f")
        self.assertEqual(sorted(geno.genes), ["AN", "BC"])
        self.assertEqual(parse_genotype("male W").sex, "m")
        self.assertEqual(parse_genotype("f WT").genes, {})

    def test_parse_rejects_by_name(self):
        for spec, code in (
            ("", "invalid_genotype"),
            ("x E/WT", "invalid_genotype"),
            ("f E/E/E", "invalid_genotype"),
            ("f ZZ", "unknown_allele"),
        ):
            with self.assertRaises(GeneticsError) as ctx:
                parse_genotype(spec)
            self.assertEqual(ctx.exception.code, code)

    def test_male_two_x_alleles_invalid(self):
        # A male cannot carry two X-linked alleles in ONE trait:
        # Sable and Tan are both Body Color, both X-linked.
        with self.assertRaises(GeneticsError) as ctx:
            parse_genotype("m S/T")
        self.assertEqual(ctx.exception.code, "invalid_genotype")

    def test_double_lethal_invalid(self):
        with self.assertRaises(GeneticsError) as ctx:
            parse_genotype("f SB/SB")
        self.assertEqual(ctx.exception.code, "invalid_genotype")

    def test_lethal_x_male_invalid(self):
        # No X-linked lethal allele exists in the catalogue, so the
        # rule is pinned on a standalone trait (never registered).
        from nomorals.learn.genetics import Allele, Genotype, Trait

        trait = Trait("XX", "Test")
        lethal_x = Allele("LX", "LethalX", trait, "X", 9.9, lethal=True)
        geno = Genotype("m", {"XX": (lethal_x,)})
        ok, _ = geno.valid()
        self.assertFalse(ok)


class ExactRatioTests(unittest.TestCase):
    def test_monohybrid_recessive_one_to_three(self):
        result = cross("f E/WT", "m E/WT")
        self.assertEqual(result.combined,
                         {"E": Fraction(1, 4), "WT": Fraction(3, 4)})

    def test_homozygous_recessive_x_wildtype_all_carriers(self):
        result = cross("f E/E", "m WT")
        self.assertEqual(result.combined, {"WT": Fraction(1)})

    def test_dominant_one_to_one(self):
        result = cross("f AR/WT", "m WT")
        self.assertEqual(result.combined,
                         {"AR": Fraction(1, 2), "WT": Fraction(1, 2)})

    def test_dihybrid_nine_three_three_one(self):
        # Two DOMINANT traits on different chromosomes.
        result = cross("f AR/WT L/WT", "m AR/WT L/WT")
        self.assertEqual(result.combined, {
            "AR+L": Fraction(9, 16),
            "AR": Fraction(3, 16),
            "L": Fraction(3, 16),
            "WT": Fraction(1, 16),
        })

    def test_parents_swap_freely(self):
        a = cross("f E/WT", "m E/E")
        b = cross("m E/E", "f E/WT")
        self.assertEqual(a.combined, b.combined)

    def test_same_sex_cross_refuses(self):
        with self.assertRaises(GeneticsError) as ctx:
            cross("f E/WT", "f E/WT")
        self.assertEqual(ctx.exception.code, "invalid_genotype")


class SexLinkedTests(unittest.TestCase):
    def test_criss_cross_white_eyes(self):
        result = cross("f W/W", "m WT")
        self.assertEqual(result.by_sex["f"], {"WT": Fraction(1)})
        self.assertEqual(result.by_sex["m"], {"W": Fraction(1)})
        self.assertEqual(result.combined,
                         {"WT": Fraction(1, 2), "W": Fraction(1, 2)})

    def test_carrier_mother_half_sons_white(self):
        result = cross("f W/WT", "m WT")
        self.assertEqual(result.by_sex["m"],
                         {"W": Fraction(1, 2), "WT": Fraction(1, 2)})
        self.assertEqual(result.by_sex["f"], {"WT": Fraction(1)})


class LinkageTests(unittest.TestCase):
    def test_female_gametes_haldane(self):
        # E (chr2 @70.7) and BW (chr2 @104.5): 33.8 cM apart.
        # r = 1/2 (1 - e^(-0.676)) ≈ 0.2457.
        gametes = _gametes(parse_genotype("f E/WT BW/WT"))
        keyed = {
            tuple(a.code if a else None for _, a in gamete): float(prob)
            for gamete, prob in gametes.items()
        }
        self.assertAlmostEqual(keyed[("E", "BW")], 0.377, places=2)
        self.assertAlmostEqual(keyed[(None, None)], 0.377, places=2)
        self.assertAlmostEqual(keyed[("E", None)], 0.123, places=2)
        self.assertAlmostEqual(keyed[(None, "BW")], 0.123, places=2)

    def test_male_achiasmy_no_recombination(self):
        # A Drosophila father deals only his two parental haplotypes.
        gametes = _gametes(parse_genotype("m E/WT BW/WT"))
        keyed = {
            tuple(a.code if a else None for _, a in gamete): float(prob)
            for gamete, prob in gametes.items()
        }
        self.assertEqual(set(keyed), {("E", "BW"), (None, None)})
        self.assertAlmostEqual(keyed[("E", "BW")], 0.5)
        self.assertAlmostEqual(keyed[(None, None)], 0.5)

    def test_linkage_practice_has_zero_recombinants(self):
        # Sex-swapped linkage cross: the father cannot recombine, so
        # the recombinant classes vanish entirely.
        result = cross("m L/WT E/WT", "f E/E")
        self.assertEqual(result.combined,
                         {"E+L": Fraction(1, 2), "WT": Fraction(1, 2)})

    def test_unlinked_traits_assort_independently(self):
        # E is recessive, so it never SHOWS against a wild-type
        # father — only AR (dominant) assorts visibly, 1:1.
        result = cross("f E/WT AR/WT", "m WT")
        self.assertEqual(result.combined, {
            "AR": Fraction(1, 2),
            "WT": Fraction(1, 2),
        })
        # ...while the GENOTYPE table still shows independent 1:1:1:1.
        mother = parse_genotype("f E/WT AR/WT")
        gametes = _gametes(mother)
        self.assertEqual(len(gametes), 4)
        for prob in gametes.values():
            self.assertEqual(prob, Fraction(1, 4))


class LethalTests(unittest.TestCase):
    def test_stubble_two_to_one_with_missing_quarter(self):
        result = cross("f SB/WT", "m SB/WT")
        self.assertEqual(result.combined,
                         {"SB": Fraction(2, 3), "WT": Fraction(1, 3)})
        self.assertEqual(result.inviable, Fraction(1, 4))

    def test_lethal_heterozygote_x_wildtype_clean(self):
        result = cross("f SB/WT", "m WT")
        self.assertEqual(result.combined,
                         {"SB": Fraction(1, 2), "WT": Fraction(1, 2)})
        self.assertEqual(result.inviable, Fraction(0))


class ChiSquareTests(unittest.TestCase):
    def test_gammq_known_values(self):
        self.assertAlmostEqual(_gammq(0.5, 0.0), 1.0)
        # chi2 = 3.841 on 1 df is the 5% point.
        self.assertAlmostEqual(_gammq(0.5, 3.841459 / 2), 0.05, places=2)

    def test_mendel_787_277(self):
        result = chi_square({"WT": 787, "E": 277},
                            {"WT": Fraction(3, 4), "E": Fraction(1, 4)})
        self.assertAlmostEqual(result.statistic, 0.6065, places=3)
        self.assertEqual(result.dof, 1)
        self.assertAlmostEqual(result.p_value, 0.436, places=2)
        self.assertIn("fits", result.verdict)

    def test_perfect_fit(self):
        result = chi_square({"WT": 90, "E": 30},
                            {"WT": Fraction(3, 4), "E": Fraction(1, 4)})
        self.assertAlmostEqual(result.statistic, 0.0)
        self.assertAlmostEqual(result.p_value, 1.0)

    def test_bad_fit(self):
        result = chi_square({"WT": 50, "E": 50},
                            {"WT": Fraction(3, 4), "E": Fraction(1, 4)})
        self.assertLess(result.p_value, 0.05)
        self.assertIn("does NOT fit", result.verdict)

    def test_observed_impossible_class_refuses(self):
        with self.assertRaises(GeneticsError) as ctx:
            chi_square({"WT": 90, "E": 30, "AR": 5},
                       {"WT": Fraction(3, 4), "E": Fraction(1, 4)})
        self.assertEqual(ctx.exception.code, "bad_ratio")


class PredictionTests(unittest.TestCase):
    def test_exact_prediction_scores_exact(self):
        expected = cross("f E/WT", "m E/WT").combined
        score = score_prediction({"WT": Fraction(3, 4), "E": Fraction(1, 4)},
                                 expected)
        self.assertTrue(score.exact)
        self.assertEqual(score.max_error, 0.0)

    def test_close_prediction(self):
        expected = cross("f E/WT", "m E/WT").combined
        score = score_prediction({"WT": Fraction(0.76), "E": Fraction(0.24)},
                                 expected)
        self.assertFalse(score.exact)
        self.assertLess(score.max_error, 0.02)

    def test_missing_class_is_full_error(self):
        expected = cross("f E/WT", "m E/WT").combined
        score = score_prediction({"WT": Fraction(1)}, expected)
        self.assertAlmostEqual(score.max_error, 0.25)

    def test_parse_ratios(self):
        parsed = parse_ratios("WT:3/4 E:1/4")
        self.assertEqual(parsed, {"WT": Fraction(3, 4), "E": Fraction(1, 4)})
        parsed = parse_ratios("E:0.5 WT:0.5")
        self.assertEqual(parsed, {"E": Fraction(1, 2), "WT": Fraction(1, 2)})
        with self.assertRaises(GeneticsError) as ctx:
            parse_ratios("WT:three")
        self.assertEqual(ctx.exception.code, "bad_ratio")


class SimulateTests(unittest.TestCase):
    def test_brood_reproducible_and_sized(self):
        mother = parse_genotype("f E/WT")
        father = parse_genotype("m E/WT")
        first = simulate(mother, father, n=200, seed=42)
        second = simulate(mother, father, n=200, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(sum(first.values()), 200)

    def test_brood_counts_inviable(self):
        mother = parse_genotype("f SB/WT")
        father = parse_genotype("m SB/WT")
        brood = simulate(mother, father, n=400, seed=1)
        self.assertIn("INVIABLE", brood)
        self.assertAlmostEqual(brood["INVIABLE"] / 400, 0.25, delta=0.08)


class LessonTests(unittest.TestCase):
    def test_six_lessons_in_order(self):
        lessons = list_lessons()
        self.assertEqual([entry["id"] for entry in lessons],
                         ["monohybrid", "dihybrid", "sexlinked",
                          "linkage", "lethal", "testcross"])

    def test_every_lesson_worked_and_practice_are_valid_crosses(self):
        for entry in list_lessons():
            worked = predict(parse_genotype(entry["worked"][0]),
                             parse_genotype(entry["worked"][1]))
            self.assertTrue(sum(worked.combined.values()) > 0)
            practice = predict(parse_genotype(entry["practice"][0]),
                               parse_genotype(entry["practice"][1]))
            self.assertTrue(sum(practice.combined.values()) > 0)

    def test_dihybrid_worked_is_nine_three_three_one(self):
        entry = lesson("dihybrid")
        result = predict(parse_genotype(entry["worked"][0]),
                         parse_genotype(entry["worked"][1]))
        self.assertEqual(result.combined, {
            "AR+L": Fraction(9, 16),
            "AR": Fraction(3, 16),
            "L": Fraction(3, 16),
            "WT": Fraction(1, 16),
        })

    def test_linkage_worked_recombinants_near_one_percent(self):
        entry = lesson("linkage")
        result = predict(parse_genotype(entry["worked"][0]),
                         parse_genotype(entry["worked"][1]))
        self.assertAlmostEqual(float(result.combined["E"]), 0.0064, places=2)
        self.assertAlmostEqual(float(result.combined["L"]), 0.0064, places=2)

    def test_unknown_lesson_refuses_by_name(self):
        with self.assertRaises(GeneticsError) as ctx:
            lesson("nope")
        self.assertEqual(ctx.exception.code, "unknown_trait")


class ChatTests(unittest.TestCase):
    KEY = "genetics-c7-test"

    def test_traits_lists_nine(self):
        out = control_learn("genetics traits", chat_key=self.KEY)
        self.assertIn("9, with their mutant alleles", out)
        self.assertIn("BC — Body Color (5 alleles)", out)

    def test_alleles_lists_trait(self):
        out = control_learn("genetics alleles BC", chat_key=self.KEY)
        self.assertIn("E — Ebony", out)
        self.assertIn("chr 2 @ 70.7", out)

    def test_cross_reports_exact_ratios(self):
        out = control_learn("genetics cross f E/WT x m E/WT",
                            chat_key=self.KEY)
        self.assertIn("1/4 (25.0%)", out)
        self.assertIn("3/4 (75.0%)", out)

    def test_cross_names_unknown_allele(self):
        out = control_learn("genetics cross f ZZ x m WT", chat_key=self.KEY)
        self.assertIn("unknown_allele", out)
        self.assertIn("ZZ", out)

    def test_sexlinked_cross_shows_by_sex(self):
        out = control_learn("genetics cross f W/W x m WT", chat_key=self.KEY)
        self.assertIn("daughters:", out)
        self.assertIn("sons:", out)
        self.assertIn("100.0%", out)

    def test_lethal_cross_names_inviable(self):
        out = control_learn("genetics cross f SB/WT x m SB/WT",
                            chat_key=self.KEY)
        self.assertIn("inviable conceptions: 1/4 (25.0%", out)

    def test_lesson_flow_predict_and_complete(self):
        key = self.KEY + "-lesson"
        out = control_learn("genetics lesson monohybrid", chat_key=key)
        self.assertIn("One trait", out)
        self.assertIn("practice:", out)
        out = control_learn("genetics predict WT:1/2 E:1/2", chat_key=key)
        self.assertIn("exact", out)
        self.assertIn("complete", out)

    def test_predict_wrong_is_off(self):
        key = self.KEY + "-wrong"
        control_learn("genetics lesson monohybrid", chat_key=key)
        out = control_learn("genetics predict WT:1 E:0", chat_key=key)
        self.assertIn("off", out)

    def test_chisq_against_last_cross(self):
        key = self.KEY + "-chisq"
        control_learn("genetics cross f E/WT x m E/WT", chat_key=key)
        out = control_learn("genetics chisq WT:90 E:30", chat_key=key)
        self.assertIn("chi-square: 0.00", out)
        self.assertIn("p = 1.000", out)

    def test_history_records_crosses(self):
        key = self.KEY + "-history"
        control_learn("genetics cross f E/WT x m E/WT", chat_key=key)
        out = control_learn("genetics history", chat_key=key)
        self.assertIn("f E/WT × m E/WT", out)

    def test_unknown_verb_shows_usage(self):
        out = control_learn("genetics frobnicate", chat_key=self.KEY)
        self.assertIn("/learn genetics cross", out)

    def test_learn_usage_mentions_genetics(self):
        out = control_learn("help", chat_key=self.KEY)
        self.assertIn("genetics", out)


class CliTests(unittest.TestCase):
    def test_nm_learn_genetics_traits(self):
        from types import SimpleNamespace

        from nomorals.cmdline.commands.learn import _cmd_learn

        args = SimpleNamespace(task=["genetics", "traits"])
        self.assertEqual(_cmd_learn(args, context=None), 0)

    def test_nm_learn_usage_mentions_genetics(self):
        import io
        from contextlib import redirect_stderr
        from types import SimpleNamespace

        from nomorals.cmdline.commands.learn import _cmd_learn

        err = io.StringIO()
        with redirect_stderr(err):
            self.assertEqual(_cmd_learn(SimpleNamespace(task=[]),
                                        context=None), 2)
        self.assertIn("genetics", err.getvalue())


if __name__ == "__main__":
    unittest.main()
