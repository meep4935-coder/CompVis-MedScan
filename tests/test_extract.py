import unittest
from datetime import date

from medscan.extract import extract, parse_dates
from medscan.vote import Voter


def page(*lines):
    """Fake OCR output: one item per line, stacked top to bottom."""
    return [([20, 40 * i, 600, 40 * i + 30], t, 0.95) for i, t in enumerate(lines)]


class TestExtract(unittest.TestCase):
    def test_sticker_all_agree(self):
        r = extract(page("TREMBLAY, Marie-Ève Anne", "NAM: TREM 8555 1512 Sexe: F",
                         "D.N.: 1985-05-15 (41 ans)", "Dossier: 1234567"))
        self.assertEqual((r.last.value, r.first.value, r.middle.value), ("Tremblay", "Marie-Ève", "Anne"))
        self.assertEqual(r.ramq.value, "TREM85551512")
        self.assertEqual(r.dob.value, date(1985, 5, 15))
        self.assertTrue(r.agree_name and r.agree_dob)
        self.assertEqual(r.sex, "F")

    def test_visit_date_is_not_dob(self):
        r = extract(page("Date : 2026-03-02", "Patient : Jean Roy", "NAM : ROYJ 0311 0207",
                         "DDN : 02/11/2003"))
        self.assertEqual(r.dob.value, date(2003, 11, 2))

    def test_ramq_letter_repaired_from_name(self):
        # OCR read J as 1 -> 'GIR1': the name GIRARD, Jean says it must be GIRJ
        r = extract(page("GIRARD, Jean Robert", "NAM: GIR1 3508 1718", "D.N.: 1935-08-17"))
        self.assertEqual(r.ramq.value, "GIRJ35081718")

    def test_family_name_order_from_ramq(self):
        # all capitals: 'LEMIEUX GENEVIEVE' could be either order; the RAMQ decides
        r = extract(page("LEMIEUX GENEVIÈVE", "RAMQ LEMG 5453 1886", "Naissance 18/03/1954"))
        self.assertEqual((r.last.value, r.first.value), ("Lemieux", "Geneviève"))

    def test_ambiguous_order_low_confidence(self):
        r = extract(page("GRENIER WILLIAM GUY", "APIXABAN 5 mg"))
        self.assertLess(r.last.conf, 0.5)

    def test_physician_and_headers_not_names(self):
        r = extract(page("Hôpital Fleury", "NOTE D'ÉVOLUTION", "Dre Sarah Boucher, permis 1-67688"))
        self.assertIsNone(r.last.value)

    def test_titles_and_period_for_comma(self):
        r = extract(page("M. Gaétan Pierre Bédard-Fournier", "BEDG67102931", "né le 1967-10-29"))
        self.assertEqual((r.last.value, r.first.value), ("Bédard-Fournier", "Gaétan"))
        r = extract(page("DUBÉ. Marie-Ève Anne", "NAM: DUBM84541713", "Né(e): 1984/04/17"))
        self.assertEqual((r.last.value, r.first.value), ("Dubé", "Marie-Ève"))

    def test_card_labels_in_pieces(self):
        items = [([33, 183, 113, 219], "NOM", 0.99), ([94, 186, 288, 224], "LEMIEUX", 0.99),
                 ([155, 240, 533, 290], "GENEVIÈVE LUCIE", 0.99), ([35, 242, 173, 280], "PRÉNOM", 0.99),
                 ([33, 312, 471, 363], "RAMQ LEMG 5453 1886", 0.97)]
        r = extract(items)
        self.assertEqual((r.last.value, r.first.value, r.middle.value), ("Lemieux", "Geneviève", "Lucie"))

    def test_names_stacked_on_two_lines(self):
        r = extract(page("TREMBLAY", "MARIE-ÈVE ANNE", "TREM 8555 1512", "1985-05-15"))
        self.assertEqual((r.last.value, r.first.value, r.middle.value), ("Tremblay", "Marie-Ève", "Anne"))
        self.assertTrue(r.agree_name)
        # given name above family name: the RAMQ letters (GAGL) decide the order
        r = extract(page("LUC", "GAGNON", "GAGL 7203 1408", "1972-03-14"))
        self.assertEqual((r.last.value, r.first.value), ("Gagnon", "Luc"))

    def test_health_card_one_line_first_then_last(self):
        # card layout described by the user: one line, given name(s) then family name, in capitals
        r = extract(page("MARIE ANNE TREMBLAY", "TREM 8555 1512", "1985-05-15"))
        self.assertEqual((r.last.value, r.first.value, r.middle.value), ("Tremblay", "Marie", "Anne"))
        r = extract(page("JEAN-FRANÇOIS CÔTÉ-LAVOIE", "COTJ 6011 0322", "1960-11-03"))
        self.assertEqual((r.last.value, r.first.value), ("Côté-Lavoie", "Jean-François"))
        self.assertTrue(r.agree_name and r.agree_dob)

    def test_stacked_without_ramq_is_low_confidence(self):
        r = extract(page("GAGNON", "LUC"))
        self.assertLess(r.last.conf, 0.5)

    def test_header_next_to_sticker_not_merged(self):
        items = [([51, 58, 600, 109], "CHU de Québec - Hôpital de l'Enfant-Jésus", 0.99),
                 ([620, 60, 856, 90], "Émilie Élise Cloutier", 0.98),
                 ([620, 95, 1054, 125], "NAM: CLOE 6857 1257 Sexe: F", 0.97)]
        self.assertEqual(extract(items).last.value, "Cloutier")

    def test_dates(self):
        self.assertIn(date(1939, 4, 28), [d for d, _ in parse_dates("Né(e): 28 avril 1939")])
        self.assertIn(date(1953, 12, 20), [d for d, _ in parse_dates("DDN 20-DÉC-1953")])
        self.assertIn(date(1985, 4, 12), [d for d, _ in parse_dates("19B5-O4-l2")])
        self.assertEqual(parse_dates("2031-01-01"), [])  # future


class TestVoter(unittest.TestCase):
    def setUp(self):
        self.good = page("TREMBLAY, Marie-Ève", "NAM: TREM 8555 1512", "D.N.: 1985-05-15")
        self.bad = page("TREMBLAY, Marie-Ève", "NAM: TREM 8555 1512", "D.N.: 1985-05-16")

    def test_needs_several_views(self):
        v = Voter(min_views=3)
        self.assertIsNone(v.add(extract(self.good)))
        self.assertIsNone(v.add(extract(self.good)))
        rec = v.add(extract(self.good))
        self.assertEqual(rec["ramq"], "TREM85551512")

    def test_disagreeing_fields_not_proposed(self):
        v = Voter(min_views=2)
        for _ in range(4):
            self.assertIsNone(v.add(extract(self.bad)))  # DOB contradicts the RAMQ

    def test_reject_waits_for_new_views(self):
        v = Voter(min_views=2, max_rejects=3)
        v.add(extract(self.good))
        self.assertIsNotNone(v.add(extract(self.good)))
        self.assertEqual(v.reject(), "reading")
        # the same evidence again must not bring back the rejected record
        for _ in range(3):
            self.assertIsNone(v.add(extract(self.good)))


if __name__ == "__main__":
    unittest.main()
