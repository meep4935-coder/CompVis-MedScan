# MedScan web app: design

Decided on 2026-11-13. This is the plan for the platform; the reading logic
(`medscan/`) keeps improving separately and reaches the web app without rewriting.

## What it does

1. **Scan:** the physician points the device's camera at a card, note, file or label.
   The app reads the patient's name, RAMQ and date of birth and shows a one-tap
   accept / reject.
2. **Fill:** the accepted patient is written into one or more PDF forms, ready to
   print, share or save.

The physician never types or corrects patient data.

## Decisions

| Question | Decision |
|---|---|
| Where the reading runs | Inside the browser, on the device. Nothing is uploaded. |
| Device | Scan and fill on the same device (phone, tablet or computer). |
| Capture | Live camera by default; "use a photo instead" as a backup. |
| Finding the document | Read the whole picture inside an on-screen frame. The outline detector is added later only if testers need tilted pages or several documents at once. |
| Reading rules | The same Python files as the desktop (`ramq.py`, `ctc.py`, `extract.py`, `vote.py`), run in the browser. One copy of the rules. |
| Template | PDF forms with type-in boxes (fillable PDFs). Flat or scanned forms come later. |
| Where forms come from | A small built-in starter set, plus forms the physician adds. |
| Matching boxes to fields | Automatic, by the boxes' names. Shown once per form for a yes / no. |
| Also filled in | Today's date, sex (from the RAMQ), and the physician's name, licence number and clinic (typed once in settings). |
| Patients kept | A saved list on the device, cleared by hand. On a new day the app asks once: "Clear N patients from earlier?" |
| Access | Anyone with the link. No accounts, no login. |
| Language | French by default, with a switch to English. |
| First test devices | iPhone / iPad and Android, plus the Mac. |
| First thing to build | A speed trial on a phone (below). |

## Screens

1. **Scan** (home)
   - Live camera with a frame to hold the document in.
   - One status line: "reading", "bring closer", "more light".
   - "Use a photo instead".
   - A small count of saved patients.
2. **Confirm**
   - The proposed patient: family name, given name, RAMQ, date of birth (marked
     "from NAM" when worked out from the RAMQ), sex.
   - Accept or reject. On reject the app keeps reading and never proposes the same
     wrong record again.
   - For a photo the app is not sure about, it says so and asks for another photo.
     It does not guess.
3. **Choose forms:** tick one or more forms, then "Fill".
4. **Filled form:** a preview, with Print / Share / Save, "another form for this
   patient" and "next patient".
5. **Patients:** the saved list, delete one, clear all.
6. **Forms**
   - Add a PDF. The app shows which boxes it will fill; the physician answers yes,
     or switches off any box that is wrong.
   - A PDF with no type-in boxes is refused with a clear message.
7. **Settings:** physician's details, language, "erase everything on this device".

## How it is built

The app is a set of plain files (a static site). There is no server of ours.

| Part | Runs as | Notes |
|---|---|---|
| Screens, camera | JavaScript in the page | Rear camera, 1920x1080 asked. |
| The two reading models | ONNX Runtime Web, in a background worker | The same model files as the desktop. Uses the graphics chip (WebGPU) when the browser has it. |
| Reading rules and voting | Our Python files, run by Pyodide in a background worker | Unchanged, so the tests still cover them. |
| PDF filling | pdf-lib (JavaScript) | Writes into the form's existing boxes. |
| Storage | The browser's own on-device storage (IndexedDB) | Forms, settings, the saved patients list. |
| Offline | A service worker caches everything | Can be added to the home screen. |

The code lives in a `web/` folder of this repo. The desktop `scan.py` stays as the
bench for developing and measuring the reading logic.

## Privacy rules for the web app

- No photo, text or patient record leaves the device. The page is served with a
  rule (Content-Security-Policy) that blocks it from connecting anywhere but its
  own files, so this can be checked, not just promised.
- No analytics, no error reporting, no scripts loaded from other sites.
- A filled PDF that the physician saves or shares is then in their hands: it sits
  in the device's files or wherever they sent it. The app says so.
- The repo and the starter forms contain only fictional patients.
- A starter form is included only if we are allowed to redistribute it.
- Testers check their clinic's privacy rules (Law 25) before using real documents.
- The page states that this is a pilot, not a certified medical product, and that
  the physician must compare the proposed record with the document before
  accepting.

## Build order

1. **Speed trial.** A bare page: camera, the reader running in the browser, the
   text it read and the time per read. Tried on an iPhone, an Android phone and
   the Mac, with fictional documents only. It answers:
   - How long is the first load, and how big is the download? (Measured: 46 MB of
     files, about 26 to 30 MB over the network once compressed.)
   - How long does one read take on a phone? Target: about 1.5 s or less, so a
     proposal appears within 5 to 8 s.
   - Does it stay within an iPhone's memory limits?

   **Status (2026-11-13):** the page is built (`web/`) and checked in a desktop
   browser, where it matches the desktop reader and takes 0.5 to 1.3 s per read.
   The image steps are already in JavaScript; only the rules run in Python (about
   40 ms per read). Not yet tried on a phone. If a phone is too slow, the options
   are the graphics-chip engine, a smaller picture for finding text, or fewer
   lines read per view.
2. **Live reading and confirm:** voting, accept / reject, photo backup.
3. **PDF filling:** forms list, automatic matching, the yes / no check.
4. **Patients list, settings, French / English, offline install.**
5. **Pilot:** the doctor friends try it with fictional documents first, then real
   ones under their clinic's rules.

## Open items

- Ask the doctor friends which 3 to 5 forms they fill most, and whether those are
  fillable PDFs.
- Date format per form (default YYYY-MM-DD).
- Publishing the page (GitHub Pages) makes it public: needs an explicit OK.
- Later: flat PDFs, the outline detector, several documents at once, Ontario.
