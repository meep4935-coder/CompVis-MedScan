// On-device storage (IndexedDB), shared by the scan page and the letter page.
// Nothing here leaves the device.
//
//   profile   one record, key "me": the physician's details, logo and stamp
//   patients  accepted patients, key = a random id; each may carry the edits made
//             to its letter ("draft")

const DB = "medscan";
const VERSION = 1;

let opened;
function db() {
  opened ??= new Promise((resolve, reject) => {
    const req = indexedDB.open(DB, VERSION);
    req.onupgradeneeded = () => {
      const d = req.result;
      if (!d.objectStoreNames.contains("profile")) d.createObjectStore("profile");
      if (!d.objectStoreNames.contains("patients")) d.createObjectStore("patients");
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return opened;
}

async function run(store, mode, fn) {
  const d = await db();
  return new Promise((resolve, reject) => {
    const tx = d.transaction(store, mode);
    const req = fn(tx.objectStore(store));
    tx.oncomplete = () => resolve(req?.result);
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error);
  });
}

export const getProfile = () => run("profile", "readonly", (s) => s.get("me"));
export const saveProfile = (p) => run("profile", "readwrite", (s) => s.put(p, "me"));
export const getPatient = (id) => run("patients", "readonly", (s) => s.get(id));
export const savePatient = (id, p) => run("patients", "readwrite", (s) => s.put(p, id));
export const countPatients = () => run("patients", "readonly", (s) => s.count());
