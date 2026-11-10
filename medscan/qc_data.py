"""
Fake but realistic Quebec patient data. Every patient generated here is FICTIONAL.

Hospitals are real Quebec institutions, used only as header text (no logos or
real layouts). Clinics and pharmacies are invented names.
"""

import random
from dataclasses import dataclass, field
from datetime import date, timedelta

from medscan.ramq import make_ramq

# Most common Quebec family names + Montreal diversity
LAST_NAMES = """Tremblay Gagnon Roy Côté Bouchard Gauthier Morin Lavoie Fortin Gagné Ouellet Pelletier
Bélanger Lévesque Bergeron Leblanc Paquette Girard Simard Boucher Caron Beaulieu Cloutier Dubé Poirier
Fournier Lapointe Leclerc Lefebvre Poulin Thibault St-Pierre Nadeau Martin Landry Martel Bédard Grenier
Lessard Bernier Richard Michaud Hébert Desjardins Couture Turcotte Lachance Parent Blais Gosselin Savard
Proulx Beaudoin Demers Perron Boudreau Charbonneau Mercier D'Amour Lemieux Vachon Allard Champagne
Nguyen Tran Singh Patel Haddad Khoury Joseph Jean-Baptiste Pierre-Louis Smith Wilson Li Wang Chen Kim
Garcia Rodriguez Ahmed Ali Cohen Rossi Papadopoulos Benali Diallo Mendoza O'Brien MacDonald""".split()

FIRST_F = """Marie Marie-Ève Julie Sophie Isabelle Catherine Geneviève Nathalie Chantal Mélanie Émilie
Camille Léa Chloé Florence Alice Rosalie Béatrice Jeanne Gabrielle Audrey Valérie Josée Lucie Ginette
Françoise Thérèse Anne-Marie Marie-Claude Fatima Amira Mei Priya Ana Sarah Olivia Charlotte""".split()
FIRST_M = """Jean Jean-François Pierre Michel Luc Marc-André Mathieu Simon Alexandre Gabriel Olivier
Félix Samuel William Thomas Louis Antoine Nicolas Étienne Maxime Guillaume Philippe Réjean Gaétan
Normand Yves Denis François Mohamed Karim Minh Raj David Daniel Noah Liam Hugo Benoît""".split()
MIDDLE_F = "Marie Anne Louise Claire Thérèse Élise Rose Lucie Simone Joséphine Hélène Jeanne".split()
MIDDLE_M = "Joseph Jean Paul André Pierre Charles Michel Henri Guy Louis Émile Robert".split()

# (full name, short name, city)
HOSPITALS = [
    ("Centre hospitalier de l'Université de Montréal", "CHUM", "Montréal"),
    ("Hôpital Maisonneuve-Rosemont", "HMR", "Montréal"),
    ("Hôpital général juif", "HGJ", "Montréal"),
    ("Centre universitaire de santé McGill", "CUSM", "Montréal"),
    ("CHU Sainte-Justine", "CHUSJ", "Montréal"),
    ("Hôpital du Sacré-Cœur-de-Montréal", "HSCM", "Montréal"),
    ("Hôpital de Verdun", "HV", "Montréal"),
    ("Hôpital Fleury", "HF", "Montréal"),
    ("Hôpital de la Cité-de-la-Santé", "CSL", "Laval"),
    ("Hôpital Charles-Le Moyne", "HCLM", "Longueuil"),
    ("Hôpital Pierre-Le Gardeur", "HPLG", "Terrebonne"),
    ("Hôpital régional de Saint-Jérôme", "HRSJ", "Saint-Jérôme"),
    ("CHU de Québec – Hôpital de l'Enfant-Jésus", "HEJ", "Québec"),
    ("CHU de Québec – L'Hôtel-Dieu de Québec", "HDQ", "Québec"),
    ("Institut universitaire de cardiologie et de pneumologie de Québec", "IUCPQ", "Québec"),
    ("CHUS – Hôpital Fleurimont", "CHUS", "Sherbrooke"),
    ("Hôpital de Chicoutimi", "HC", "Saguenay"),
    ("Centre hospitalier affilié universitaire régional", "CHAUR", "Trois-Rivières"),
    ("Hôpital de Gatineau", "HG", "Gatineau"),
    ("Hôpital régional de Rimouski", "HRR", "Rimouski"),
]

CLINICS = ["GMF du Plateau", "GMF des Rivières", "Clinique médicale Saint-Laurent", "GMF Les Érables",
           "Clinique médicale du Vieux-Port", "GMF Boréal", "Clinique familiale des Cèdres", "GMF Val-Fleuri"]
PHARMACIES = ["Pharmacie Lacroix & Fils", "Pharmacie du Quartier", "Pharmacie Santé Plus",
              "Pharmacie Rivard", "Pharmacie des Monts", "Pharmacie Beaulac", "Pharmacie L'Étoile"]
STREETS = ["rue Sherbrooke", "boul. Saint-Laurent", "rue Saint-Denis", "ch. Sainte-Foy", "rue King Ouest",
           "boul. des Laurentides", "rue Principale", "av. du Parc", "boul. Talbot", "rue Racine Est"]

MEDS = [("Atorvastatine", "20 mg", "1 comprimé au coucher"), ("Metformine", "500 mg", "1 comprimé 2 fois par jour"),
        ("Ramipril", "5 mg", "1 capsule 1 fois par jour"), ("Amlodipine", "5 mg", "1 comprimé le matin"),
        ("Lévothyroxine", "50 mcg", "1 comprimé à jeun"), ("Pantoprazole", "40 mg", "1 comprimé avant le déjeuner"),
        ("Apixaban", "5 mg", "1 comprimé 2 fois par jour"), ("Sertraline", "50 mg", "1 comprimé par jour"),
        ("Amoxicilline", "500 mg", "1 capsule 3 fois par jour x 10 jours"),
        ("Salbutamol", "100 mcg", "2 inhalations au besoin"), ("Hydrochlorothiazide", "25 mg", "1 comprimé le matin"),
        ("Rosuvastatine", "10 mg", "1 comprimé par jour"), ("Gabapentine", "300 mg", "1 capsule 3 fois par jour")]

MONTHS_FR = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre",
             "octobre", "novembre", "décembre"]
MONTHS_FR_ABBR = ["JAN", "FÉV", "MAR", "AVR", "MAI", "JUN", "JUL", "AOÛ", "SEP", "OCT", "NOV", "DÉC"]


@dataclass
class Patient:
    first: str
    middle: str  # may be ""
    last: str
    sex: str
    dob: date
    ramq: str
    dossier: str  # hospital record number (a distractor number, NOT the RAMQ)
    phone: str = field(default="")

    @property
    def full(self):
        return " ".join(p for p in (self.first, self.middle, self.last) if p)


def random_patient(rng=random):
    sex = rng.choice("FM")
    last = rng.choice(LAST_NAMES)
    if rng.random() < 0.12:  # compound family name
        last = f"{last}-{rng.choice(LAST_NAMES)}"
    first = rng.choice(FIRST_F if sex == "F" else FIRST_M)
    middle = rng.choice(MIDDLE_F if sex == "F" else MIDDLE_M) if rng.random() < 0.6 else ""
    dob = date(1925, 1, 1) + timedelta(days=rng.randint(0, 365 * 100))
    ramq = make_ramq(last, first, dob, sex, rng.randint(1, 99))
    dossier = str(rng.randint(100000, 9999999))
    phone = f"{rng.choice(['514', '438', '450', '418', '819'])}-555-{rng.randint(0, 9999):04d}"  # 555 = fictional
    return Patient(first, middle, last, sex, dob, ramq, dossier, phone)


def format_date(d: date, rng=random):
    """Dates the way they appear on Quebec documents (AAAA-MM-JJ is the official format)."""
    style = rng.choices(["iso", "dmy", "ymd_slash", "fr_long", "fr_abbr"], weights=[5, 3, 1, 1, 1])[0]
    if style == "iso":
        return d.strftime("%Y-%m-%d")
    if style == "dmy":
        return d.strftime("%d/%m/%Y")
    if style == "ymd_slash":
        return d.strftime("%Y/%m/%d")
    if style == "fr_long":
        return f"{d.day} {MONTHS_FR[d.month - 1]} {d.year}"
    return f"{d.day:02d}-{MONTHS_FR_ABBR[d.month - 1]}-{d.year}"


def format_name(p: Patient, rng=random):
    """'TREMBLAY, Marie-Ève Anne' | 'Marie-Ève Anne Tremblay' | 'TREMBLAY MARIE-ÈVE' ..."""
    given = " ".join(x for x in (p.first, p.middle) if x)
    style = rng.choices(["last_first", "natural", "upper_natural", "upper_last_first"], weights=[4, 3, 1, 2])[0]
    if style == "last_first":
        return f"{p.last.upper()}, {given}"
    if style == "natural":
        return p.full
    if style == "upper_natural":
        return p.full.upper()
    return f"{p.last.upper()} {given.upper()}"


def physician(rng=random):
    sex = rng.choice("FM")
    name = f"{rng.choice(FIRST_F if sex == 'F' else FIRST_M)} {rng.choice(LAST_NAMES)}"
    return f"Dr{'e' if sex == 'F' else ''} {name}", f"{rng.randint(1, 2)}-{rng.randint(10000, 99999)}"
