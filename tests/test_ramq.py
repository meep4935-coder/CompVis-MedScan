import unittest
from datetime import date

from medscan import ramq


class TestRamq(unittest.TestCase):
    def test_make_female(self):
        r = ramq.make_ramq("Tremblay", "Marie-Ève", date(1985, 5, 15), "F", 12)
        self.assertEqual(r, "TREM85551512")
        self.assertEqual(ramq.format_ramq(r), "TREM 8555 1512")

    def test_make_male_accents_and_compound(self):
        self.assertEqual(ramq.name_code("Côté-Lavoie", "Éric"), "COTE")
        self.assertEqual(ramq.name_code("D'Amour", "Luc"), "DAML")
        self.assertEqual(ramq.name_code("St-Pierre", "Jean"), "STPJ")
        self.assertEqual(ramq.name_code("Li", "Wei"), "LIXW")  # padded
        self.assertEqual(ramq.make_ramq("Roy", "Jean", date(2003, 11, 2), "M", 7), "ROYJ03110207")

    def test_find_and_fix(self):
        self.assertEqual(ramq.find_ramq("NAM: trem 8555 1512 exp 2027"), ["TREM85551512"])
        self.assertEqual(ramq.find_ramq("dossier 12345678"), [])
        self.assertEqual(ramq.fix_confusions("7REM 8S55 1S12"), "TREM85551512")
        self.assertEqual(ramq.fix_confusions("TRE0 8555 1512"), "TREO85551512")
        self.assertIsNone(ramq.fix_confusions("TREM 8555 151"))

    def test_valid_and_check(self):
        r = "TREM85551512"
        self.assertTrue(ramq.is_valid(r))
        self.assertFalse(ramq.is_valid("TREM85631512"))  # month 63-50 = 13
        self.assertFalse(ramq.is_valid("TREM85023012"))  # Feb 30
        self.assertEqual(ramq.check(r, "Tremblay", "Marie", date(1985, 5, 15), "F"), [])
        self.assertEqual(len(ramq.check(r, "Gagnon", "Marie", date(1985, 5, 16), "M")), 3)


if __name__ == "__main__":
    unittest.main()
