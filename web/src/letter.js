// The letter: the physician's letterhead with the accepted patient written in, ready to
// edit and print. Opened in its own tab by the scan page as letter.html?p=<id>; the id
// points to the patient saved on this device (store.js). No patient data is in the URL.
//
// Every text on the sheet can be edited. Edits are kept with this patient's letter only
// (so a reload keeps them); the physician's saved details change only through "My details".

import { getPatient, getProfile, savePatient, saveProfile } from "./store.js";

const $ = (id) => document.getElementById(id);
const id = new URLSearchParams(location.search).get("p");

const T = {
  fr: {
    print: "Imprimer", profile: "Mes coordonnées", reset: "Annuler mes modifications", other: "English",
    hint: "Tout le texte de la feuille se modifie : touchez-le et tapez. Les modifications ne valent que pour cette feuille.",
    setupTitle: "Vos coordonnées", setupNote: "Entrées une seule fois et gardées sur cet appareil seulement.",
    fDocName: "Nom du médecin", fLicence: "No de permis", fDocTitle: "Spécialité", fClinic: "Clinique ou hôpital",
    fAddr: "Adresse", fPhone: "Téléphone / télécopieur", fLogo: "Logo", fStamp: "Tampon / sceau",
    remove: "Retirer", cancel: "Annuler", save: "Enregistrer",
    resetAsk: "Remettre la feuille comme au départ ? Vos modifications de cette feuille seront perdues.",
    noPatient: "Aucun patient trouvé pour cette feuille : les champs du patient sont vides.",
    logo: "Logo", stamp: "Tampon",
    // the sheet's own words (editable on the sheet)
    title: "Ordonnance", lblLicence: "No de permis :", lblLast: "Nom", lblFirst: "Prénom", lblRamq: "NAM",
    lblDob: "Date de naissance", lblSex: "Sexe", lblDate: "Date", lblSign: "Signature", bodyHint: "Rx",
  },
  en: {
    print: "Print", profile: "My details", reset: "Undo my edits", other: "Français",
    hint: "Every text on the sheet can be edited: tap it and type. Edits apply to this sheet only.",
    setupTitle: "Your details", setupNote: "Entered once and kept on this device only.",
    fDocName: "Physician's name", fLicence: "Licence no.", fDocTitle: "Specialty", fClinic: "Clinic or hospital",
    fAddr: "Address", fPhone: "Phone / fax", fLogo: "Logo", fStamp: "Stamp / seal",
    remove: "Remove", cancel: "Cancel", save: "Save",
    resetAsk: "Put the sheet back as it started? Your edits to this sheet will be lost.",
    noPatient: "No patient found for this sheet: the patient fields are empty.",
    logo: "Logo", stamp: "Stamp",
    title: "Prescription", lblLicence: "Licence no.:", lblLast: "Last name", lblFirst: "First name", lblRamq: "RAMQ",
    lblDob: "Date of birth", lblSex: "Sex", lblDate: "Date", lblSign: "Signature", bodyHint: "Rx",
  },
};

let lang = "fr";
try { lang = localStorage.getItem("medscan.lang") === "en" ? "en" : "fr"; } catch {}

let profile = {};
let patient = null;
let draft = {};

const isoDay = (d) => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
const ramqText = (r) => (r ? `${r.slice(0, 4)} ${r.slice(4, 8)} ${r.slice(8)}` : "");

/** What each part of the sheet says before any edit. */
function defaults() {
  const t = T[lang], r = patient?.record || {};
  return {
    clinic: profile.clinic || "", clinicAddr: profile.clinicAddr || "", clinicPhone: profile.clinicPhone || "",
    docName: profile.docName || "", docTitle: profile.docTitle || "",
    lblLicence: t.lblLicence, licence: profile.licence || "",
    title: t.title,
    lblLast: t.lblLast, last: (r.last || "").toUpperCase(),
    lblFirst: t.lblFirst, first: [r.first, r.middle].filter(Boolean).join(" "),
    lblRamq: t.lblRamq, ramq: ramqText(r.ramq),
    lblDob: t.lblDob, dob: r.dob || "",
    lblSex: t.lblSex, sex: r.sex || "",
    lblDate: t.lblDate, date: isoDay(patient?.acceptedAt ? new Date(patient.acceptedAt) : new Date()),
    body: "",
    lblSign: t.lblSign, signName: profile.docName || "",
    lblLicence2: t.lblLicence, licence2: profile.licence || "",
  };
}

function render() {
  const d = defaults();
  for (const el of document.querySelectorAll("[data-k]")) {
    const k = el.dataset.k;
    if (document.activeElement !== el) el.innerText = draft[k] ?? d[k];
  }
  $("sheet").querySelector('[data-k="body"]').dataset.hint = T[lang].bodyHint;
  image($("logoSlot"), profile.logo, T[lang].logo);
  image($("stampSlot"), profile.stamp, T[lang].stamp);
  for (const el of document.querySelectorAll("[data-t]")) el.textContent = T[lang][el.dataset.t];
  $("lang").textContent = T[lang].other;
  document.documentElement.lang = lang;
  $("sheet").hidden = false;
}

function image(slot, src, label) {
  slot.replaceChildren();
  slot.classList.toggle("empty", !src);
  if (src) {
    const img = new Image();
    img.src = src;
    img.alt = label;
    slot.append(img);
    slot.classList.remove("slot");
  } else {
    slot.classList.add("slot");
    slot.textContent = label;
  }
}

// ------------------------------------------------------------------ editing the sheet

let saveTimer;
function saveDraftSoon() {
  if (!patient) return;
  clearTimeout(saveTimer);
  saveTimer = setTimeout(saveDraftNow, 400);
}
function saveDraftNow() {
  if (!patient || !saveTimer) return;
  clearTimeout(saveTimer);
  saveTimer = null;
  savePatient(id, { ...patient, draft }).catch(showError);
}
// don't lose the last keystrokes when the tab is closed or put away
addEventListener("pagehide", saveDraftNow);
document.addEventListener("visibilitychange", () => document.hidden && saveDraftNow());
document.addEventListener("focusout", saveDraftNow);

function makeEditable() {
  let plain = false;
  try {
    const probe = document.createElement("div");
    probe.contentEditable = "plaintext-only";
    plain = probe.contentEditable === "plaintext-only";
  } catch {}
  for (const el of document.querySelectorAll("[data-k]")) {
    el.contentEditable = plain ? "plaintext-only" : "true";
    el.spellcheck = el.dataset.k === "body";
    el.addEventListener("input", () => {
      const v = el.innerText.replace(/\n$/, "");
      if (v === defaults()[el.dataset.k]) delete draft[el.dataset.k];
      else draft[el.dataset.k] = v;
      saveDraftSoon();
    });
    // pasted text arrives as plain text, so the sheet keeps its own look
    el.addEventListener("paste", (e) => {
      if (plain) return;
      e.preventDefault();
      document.execCommand("insertText", false, e.clipboardData.getData("text/plain"));
    });
  }
}

// ------------------------------------------------------------------ physician's details

const pending = {};

/** A logo or stamp photo, made small enough to keep on the device (transparency kept). */
async function shrink(file, maxSide = 900) {
  const bmp = await createImageBitmap(file);
  const s = Math.min(1, maxSide / Math.max(bmp.width, bmp.height));
  const c = document.createElement("canvas");
  c.width = Math.round(bmp.width * s);
  c.height = Math.round(bmp.height * s);
  c.getContext("2d").drawImage(bmp, 0, 0, c.width, c.height);
  bmp.close();
  return c.toDataURL("image/png");
}

function preview(which, src) {
  $(`${which}Prev`).src = src || "";
  $(`${which}Prev`).hidden = !src;
}

function openSetup() {
  const f = $("setupForm");
  for (const k of ["docName", "licence", "docTitle", "clinic", "clinicAddr", "clinicPhone"]) f.elements[k].value = profile[k] || "";
  for (const w of ["logo", "stamp"]) {
    delete pending[w];
    $(`${w}File`).value = "";
    preview(w, profile[w]);
  }
  $("setup").showModal();
}

for (const w of ["logo", "stamp"]) {
  $(`${w}File`).onchange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    try {
      pending[w] = await shrink(file);
      preview(w, pending[w]);
    } catch {
      showError(lang === "fr" ? "Cette image n'a pas pu être ouverte." : "That picture could not be opened.");
    }
  };
  $(`${w}Clear`).onclick = () => {
    pending[w] = null;
    $(`${w}File`).value = "";
    preview(w, null);
  };
}

$("setupForm").onsubmit = async () => {
  const f = $("setupForm"), next = { ...profile };
  for (const k of ["docName", "licence", "docTitle", "clinic", "clinicAddr", "clinicPhone"]) next[k] = f.elements[k].value.trim();
  for (const w of ["logo", "stamp"]) if (w in pending) next[w] = pending[w];
  try {
    await saveProfile(next);
    profile = next;
    render();
  } catch (e) {
    showError(e);
  }
};
$("setupCancel").onclick = () => $("setup").close();

// ------------------------------------------------------------------ bar

$("print").onclick = () => {
  document.activeElement?.blur();
  window.print();
};
$("profileBtn").onclick = openSetup;
$("resetBtn").onclick = () => {
  if (!Object.keys(draft).length || !confirm(T[lang].resetAsk)) return;
  draft = {};
  saveDraftSoon();
  render();
};
$("lang").onclick = () => {
  lang = lang === "fr" ? "en" : "fr";
  try { localStorage.setItem("medscan.lang", lang); } catch {}
  render();
};

function showError(e) {
  $("error").textContent = String(e?.message || e);
}

// On a phone the Letter-size sheet is shrunk to the screen's width (printing is unaffected).
function fit() {
  $("sheet").style.zoom = String(Math.min(1, (document.documentElement.clientWidth - 32) / (8.5 * 96)));
}
addEventListener("resize", fit);
fit();

// ------------------------------------------------------------------ start

/** The scan page opens this tab first and saves the patient right after, so wait briefly. */
async function loadPatient() {
  if (!id) return null;
  for (let i = 0; i < 20; i++) {
    const p = await getPatient(id);
    if (p) return p;
    await new Promise((r) => setTimeout(r, 100));
  }
  return null;
}

async function start() {
  try {
    [profile, patient] = await Promise.all([getProfile().then((p) => p || {}), loadPatient()]);
  } catch (e) {
    showError(e);
  }
  draft = { ...(patient?.draft || {}) };
  if (!patient) showError(T[lang].noPatient);
  makeEditable();
  render();
  if (!profile.docName && !profile.licence && !profile.clinic) openSetup();
}

start();
