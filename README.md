# SonicSphere — real-time 360° binaural audio with measured HRIRs

SonicSphere renders a sound source anywhere around a virtual listener and plays it over ordinary stereo headphones. Directional cues come from **measured head-related impulse responses** (CIPIC HRTF Database). Directions between measurements are **interpolated** with a method chosen by a held-out experiment. Rendering runs **in real time in an AudioWorklet** using uniformly partitioned FFT convolution. The source can be clicked into place, dragged continuously, or flown along hand-drawn 3D trajectories. A Python/FastAPI backend validates the dataset, prepares filters, renders reference audio and benchmarks every algorithm. The browser engine is tested sample-for-sample against it.

```
Input audio → preprocessing → direction lookup → spatial interpolation → left/right convolution → stereo output
```

---

## Contents

1. [Quick start](#quick-start)
2. [Publish the website for free](#publish-the-website-for-free)
3. [Dataset: acquisition, format, licence](#dataset)
4. [Architecture](#architecture)
5. [Signal processing](#signal-processing)
6. [Using the application](#using-the-application)
7. [Testing and evaluation](#testing-and-evaluation)
8. [Measured results](#measured-results)
9. [Built-in sounds and licensing](#built-in-sounds-and-licensing)
10. [Configuration](#configuration)
11. [Project layout](#project-layout)
12. [Limitations](#limitations)
13. [Troubleshooting](#troubleshooting)

---

## Quick start

Requirements: **Python 3.11+**, **Node.js 18+** (tested with 22), a current Chromium-based browser or Firefox, and headphones.

### 1. Backend (from `backend/`)

Windows (PowerShell):

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m sonicsphere.download_dataset          # CIPIC subjects 003 008 021 040 048 165 (~11 MB)
python -m sonicsphere.generate_sounds           # the four built-in sounds (already in assets/sounds)
uvicorn sonicsphere.main:app --host 127.0.0.1 --port 8000
```

macOS / Linux:

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m sonicsphere.download_dataset
python -m sonicsphere.generate_sounds           # voice clip needs Windows TTS, see "Built-in sounds"
uvicorn sonicsphere.main:app --host 127.0.0.1 --port 8000
```

The API docs are then at <http://127.0.0.1:8000/docs>.

### 2. Frontend (from `frontend/`, in a second terminal)

```bash
cd frontend
npm install
npm run dev            # http://localhost:5173  (proxies /api to the backend on port 8000)
```

Open <http://localhost:5173>, put on headphones, press **Start audio engine** and then **Play**.

### 3. Tests and experiments

```bash
# backend/ (venv active)
pytest                                         # 141 tests
python -m sonicsphere.interpolation_eval       # held-out interpolation experiment → results/interpolation_eval.json
python -m sonicsphere.benchmark                # direct vs FFT benchmarks           → results/benchmark.json
python -m sonicsphere.export_golden            # golden vectors for the browser tests

# frontend/
npm test                                       # 23 tests: browser DSP vs Python golden vectors
npm run typecheck && npm run build             # strict TypeScript + production bundle
```

`npm test` needs the golden vectors, so run `python -m sonicsphere.export_golden` first.

---

## Publish the website for free

The website does not need a server. Everything it reads — ear filters, dataset details, sounds — is exported as static files into `frontend/public/data/` (12.5 MB), and all audio processing runs in the visitor's browser. Any static host works. The repository is set up for **GitHub Pages** (only a GitHub account is needed); Netlify and Cloudflare Pages also work.

### Steps (GitHub Pages)

1. **Create a free GitHub account** at <https://github.com/signup>. To keep your email address private in commits, open *Settings → Emails*, enable **Keep my email addresses private**, and copy the `…@users.noreply.github.com` address shown there.
2. **Create an empty repository**: <https://github.com/new>, name it e.g. `sonicsphere`, choose **Public** (free GitHub Pages needs a public repository), and do **not** add a README or licence.
3. **Commit and push the project** (PowerShell or Git Bash, in the project folder):

   ```bash
   git config user.name  "Your Name"
   git config user.email "12345678+yourname@users.noreply.github.com"   # the address from step 1
   git add -A
   git commit -m "SonicSphere website"
   git branch -M main
   git remote add origin https://github.com/<your-username>/sonicsphere.git
   git push -u origin main
   ```

   Git asks you to sign in to GitHub in the browser the first time.
4. **Turn on Pages**: in the repository open *Settings → Pages → Build and deployment → Source* and choose **GitHub Actions**.
5. Open the **Actions** tab. The workflow *Deploy website to GitHub Pages* builds and publishes the site in about 2–3 minutes; if it ran before Pages was enabled, press *Re-run all jobs*. Your site is then live at

   **`https://<your-username>.github.io/sonicsphere/`**

### Updating the live site

Edit, then `git add -A && git commit -m "…" && git push` — the site redeploys automatically. Only if you changed the ear-filter settings or the sounds, refresh the exported data first:

```bash
cd backend
python -m sonicsphere.export_static          # rewrites frontend/public/data/
```

### Adding your own "Come here" voice

The Windows text-to-speech clip is **not** published: its licence is unclear for public hosting, so `export_static` skips it and `.gitignore` keeps it out of the repository. To add a voice, record yourself saying "Come here!" (any phone recording, WAV/MP3), then:

```bash
cd backend
python -m sonicsphere.generate_sounds --voice-file path/to/my_voice.wav   # trims, filters, marks it publishable
python -m sonicsphere.export_static
```

and commit `frontend/public/data/` (remove the `assets/sounds/come_here.wav` line from `.gitignore` if you also want the processed file in the repository).

### Other free hosts

| Host | Settings |
|---|---|
| **Netlify** | "Add new site → Import from Git", pick the repository. `netlify.toml` already sets base `frontend`, command `npm run build`, publish `dist`. |
| **Cloudflare Pages** | "Create → Pages → Connect to Git". Root directory `frontend`, build command `npm run build`, output `dist`, environment variable `NODE_VERSION = 20`. |
| **Vercel** | "Add New Project", root directory `frontend`, framework preset *Vite*. |

On these hosts the site lives at the domain root, so no base path is needed. GitHub Pages project sites live under `/<repository-name>/`, which the workflow passes to the build as `VITE_BASE`.

### Checking a build locally

```bash
cd frontend
npm run build && npm run preview      # http://localhost:4173 — works with the Python backend switched off
```

`npm run dev` still uses the live Python backend (API mode); production builds read the static files. Override with `VITE_DATA_MODE=static|api`.

---

## Dataset

### Selection and verification

The **CIPIC HRTF Database** (Algazi, Duda, Thompson & Avendano, IEEE WASPAA 2001) was evaluated first and is used.

| Check | Finding |
|---|---|
| Acquisition | UC Davis' original CIPIC site now returns HTTP 410. The authoritative copy is the **SOFA conventions repository**: <https://sofacoustics.org/data/database/cipic/>, 45 subjects, one `subject_XXX.sofa` each (≈1.8 MB). |
| Licence | Embedded in every file: *"The Regents of the University of California hereby grant users permission to reproduce and/or use materials available therein for any purpose — educational, research or commercial. However, each reproduction of any part of the materials must include the copyright notice, if it is present. … if these materials are used in published research, this use should be acknowledged …"* The full text is returned by `GET /api/datasets/cipic`, and the page footer carries the copyright notice. The files are **downloaded by the user** and not redistributed in this repository. |
| Format | SOFA 1.0, convention `SimpleFreeFieldHRIR` (netCDF-4/HDF5), `DataType = FIR`: **time-domain HRIRs**. |
| Sampling rate / length | 44 100 Hz, **200 taps** (4.5 ms). |
| Directions | **1250** per subject = 25 lateral angles (−80°…80°) × 50 polar angles (−45°…230.625°, step 5.625°), 1 m distance. |
| Coordinates | `SourcePosition` is SOFA *spherical* (azimuth, elevation, metre). `ListenerView = [1,0,0]`, `ListenerUp = [0,0,1]`. Receiver 0 sits at y = +0.09 m (left ear). Verified: at azimuth 80° channel 0 carries 74× more energy and arrives 31 samples earlier. Converting back to interaural-polar coordinates reproduces CIPIC's native 25 × 50 grid exactly (tested). |
| Gap | No measurements for polar angles 230.6°…315°, a wedge below the head at every lateral angle (see gap fill below). |

A second dataset, **MIT KEMAR** (Gardner & Martin, 710 directions, 512 taps), is configured to show that nothing is hard-coded. It loads through the same adapter:

```bash
python -m sonicsphere.download_dataset --dataset mit_kemar --subjects normal_pinna large_pinna
python -m sonicsphere.download_dataset --all      # all 45 CIPIC subjects (~80 MB)
```

The downloader validates every file (HDF5 signature, SOFA convention, loadable by the adapter) and writes SHA-256 hashes to `data/<dataset>/MANIFEST.json`.

### Adapter (`dataset_loader.py`)

The adapter reads any SOFA `SimpleFreeFieldHRIR` file and:

- applies `Data.Delay`;
- rotates positions into the head frame if `ListenerView` / `ListenerUp` are not the defaults;
- orders ears by `ReceiverPosition` rather than by assumption;
- converts spherical or cartesian positions (degrees or radians);
- rejects NaN/Inf, wrong ranks or receiver counts, and mismatched position/IR counts, with explicit messages.

`SimpleFreeFieldHRTF` files (frequency domain) are converted with an inverse real FFT. The conversion checks uniform bin spacing, derives N (even or odd) from whether the last bin is Nyquist, and enforces conjugate symmetry by discarding any imaginary part of DC/Nyquist (reported).

---

## Architecture

```
Python backend (FastAPI · NumPy · SciPy)              Browser (React · TypeScript · Three.js · Web Audio)
────────────────────────────────────────              ─────────────────────────────────────────────────────
SOFA ─► dataset_loader ─► hrir_preprocessing                Main thread
        (validate, head frame)  (TOA, alignment)              ├ 3D scene (R3F): click / drag / draw / edit
     ─► hrir_interpolation (triangulation, gap rim)           ├ UI: experiences, live ear cues, how-it-works   
     ─► filter_pack  ═══ GET …/filters (binary, once) ══════► └ AudioEngine ── postMessage (positions, pose,
                                                                              trajectory, transport, settings)
convolution · audio_renderer (reference)                    AudioWorklet (audio thread, 128-frame blocks)
interpolation_eval · benchmark                                ├ source player, sample-accurate fades
POST /api/render  ◄─── A/B reference render                   ├ trajectory evaluated on the audio clock
GET  /api/analysis/holdout, /evaluation, /benchmarks          ├ world → head → (az, el) → cache key
                                                              ├ barycentric-aligned interpolation (+ gap fill)
                                                              ├ UPOLS / direct convolution + crossfade
                                                              └ peak limiter → stereo out
```

**The backend is never on the real-time path.** The browser downloads one filter pack per subject (5.4 MB for CIPIC, ETag-cached). It contains measured and onset-aligned HRIRs, TOAs, the triangulation with precomputed inverse matrices, and the gap boundary. Every direction change is then resolved inside the AudioWorklet. Trajectories are also evaluated there on the audio clock, so a dropped UI frame cannot make the sound stutter. The main thread sends only small control messages.

### Filter pack format

`"SSFP"` · uint32 version · uint32 header length · uint32 reserved · JSON header · 16-byte-aligned little-endian arrays:

| array | type, shape | content |
|---|---|---|
| `measured` | float32 (M, 2, L) | measured HRIRs, channel 0 = left, zero-padded to L |
| `aligned` | float32 (M, 2, L) | onset-aligned shapes (onset at `margin`) |
| `toa` | float32 (M, 2) | onsets in samples |
| `vertices` | float64 (M, 3) | unit directions, head frame |
| `azel` | float32 (M, 2) | SOFA azimuth/elevation |
| `triangles` | uint32 (T, 3) | spherical Delaunay triangles |
| `triInverse` | float64 (T, 9) | inverse vertex matrix per triangle |
| `triMaxEdge`, `triGap` | float32 / uint8 (T) | longest edge, unmeasured-gap flag |
| `boundaryEdges` | uint32 (E, 2) | rim of the measured region |

The header also carries fs, L, margin, block size, normalisation gain, snap/quantum angles, limiter settings and grid metadata. `?fs=48000` returns HRIRs resampled with a polyphase filter, used when a browser refuses a 44.1 kHz AudioContext.

### HTTP API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health`, `/api/config` | status, downloaded subjects, rendering configuration |
| GET | `/api/datasets` · `/api/datasets/{id}?subject=` | datasets; metadata, licence, conventions, grid |
| GET | `/api/datasets/{id}/subjects/{s}/directions` | measured directions (SOFA + interaural-polar) |
| GET | `/api/datasets/{id}/subjects/{s}/filters?fs=` | binary filter pack |
| POST | `/api/datasets/{id}/subjects/{s}/prepare` | preprocessing statistics |
| GET | `/api/hrir?azimuth=&elevation=&method=` | HRIR pair, weights, neighbours, spectrum |
| GET | `/api/analysis/holdout?azimuth=&elevation=` | leave-one-out comparison of all methods |
| GET/POST | `/api/evaluation/interpolation[/run]` | stored / fresh held-out evaluation |
| GET/POST | `/api/benchmarks[/run]` | stored / fresh Python benchmark |
| GET | `/api/sounds`, `/api/sounds/{id}` | built-in sounds with licensing |
| POST | `/api/render` (multipart `params` JSON + optional `file`) | reference binaural WAV; details in `X-Render-Info` |

All request and response bodies are Pydantic models (`schemas.py`). Dataset errors return 404 and audio errors 422, each with a descriptive message.

---

## Signal processing

### Coordinates (`coordinate_system.py`, `src/dsp/coordinates.ts`)

| Frame | Definition |
|---|---|
| World (3D scene) | right-handed, metres, **+Y up**. Head centre at (0, 1.15, 0). At yaw 0 the listener faces **−Z**, so world −X is their left. |
| Listener pose | yaw ψ about +Y, **positive = turn left**; pitch θ positive = nose up. Rows of R(ψ, θ): forward = (−sinψ cosθ, sinθ, −cosψ cosθ), left = (−cosψ, 0, sinψ), up = forward × left. |
| Head frame (SOFA) | x forward, y left, z up. |
| SOFA spherical | azimuth φ = atan2(y, x) mod 360° (0 = front, **90 = left**, 270 = right); elevation δ = atan2(z, √(x²+y²)). |
| Interaural-polar | lateral = asin(y) (+ left), polar = atan2(z, x) ∈ [−90°, 270°). |

For a source at **p**ₛ(t): **d**(t) = **p**ₛ(t) − **p**ₗ, then **d**_head = R(ψ, θ)**d**, then (φ, δ, r). Turning the listener changes **d**_head, so the perceived direction changes (tested).

### Convolution (`convolution.py`, `src/dsp/convolver.ts`)

- **Reference:** `numpy.convolve`, verified against the literal double sum. y_{L,R} = x ∗ h_{L,R}, output length N + L − 1.
- **FFT:** single-block, overlap-add, overlap-save, and **uniformly partitioned overlap-save (UPOLS)** with B = 128 (the Web Audio render quantum), FFT size 2B and P = L/B = 2 partitions. A frequency-domain delay line holds the input spectra. Because B equals the host block, no latency is added. All variants match direct convolution to < 10⁻¹⁰ relative error (tested for impulses, silence, 1-sample inputs and odd lengths).
- **Choice by measurement:** in Python, compiled NumPy direct convolution beats every FFT variant for 200-tap HRIRs. In JavaScript, UPOLS is 2.7–9× faster (see results). The browser therefore uses UPOLS by default. An earlier version benchmarked both kernels at start-up, but that froze the page for over 2 s (much longer on slow laptops), so it was removed. `src/dsp/benchmark.ts` remains available for measurement.

### Preprocessing (`hrir_preprocessing.py`)

- **Onset (TOA):** first crossing of −20 dB relative to the per-ear peak, on an 8× band-limited upsampled response with linear interpolation. Cross-checked against interaural cross-correlation ITDs (median difference < 40 µs).
- **Alignment:** h[n] = a[n − (τ − m)]. Shapes are shifted to a margin m = 40 samples with an FFT fractional delay, which is exactly invertible except for the real-valued Nyquist bin. The 40-sample margin was chosen by measurement: contralateral responses carry energy well before their −20 dB onset, and at m = 12 the round trip was only −21 dB worst case. At m = 40 it is **−76 dB median, −41 dB worst**.
- **Filter length:** L = ⌈(N + 32)/B⌉·B = 256 at 44.1/48 kHz. The worst-case truncated energy is reported (−154 dB for CIPIC).
- **Normalisation:** a dataset gain gives the loudest ear filter unit energy. Sources are normalised to −20 dBFS RMS with a −1 dBFS peak ceiling. An instant-attack/80 ms-release peak limiter guarantees |y| ≤ 0.98.

### Interpolation (`hrir_interpolation.py`, `src/dsp/interpolation.ts`)

1. **Nearest neighbour:** baseline.
2. **Inverse-distance weighting:** k = 4, p = 2.
3. **Spherical barycentric:** the convex hull of the unit direction vectors is a spherical Delaunay triangulation (2496 triangles for CIPIC). Weights are **g** = [v₁ v₂ v₃]⁻¹**p**, normalised, and continuous across edges. Azimuth wraparound is handled implicitly because everything is vector-based.

Each scheme is applied to **raw** HRIRs, to **onset-aligned** shapes with separately interpolated TOAs (delay re-inserted by fractional shift), or to **minimum-phase** shapes with TOAs.

- Directions within **0.1°** of a measurement use the **measured HRIR verbatim**. In an E2E test the rendered frontal ILD (−2.41 dB) matched the raw subject-165 data (−2.42 dB).
- **Unmeasured regions:** hull triangles with an edge longer than 30° are flagged as gaps (46 for CIPIC, all in the polar wedge 230.6°–315°). A gap direction is **filled along its cone of confusion**: the circle of constant lateral angle is intersected in closed form with the rim edges, and the HRIR is interpolated in polar angle between the two crossings. This keeps the ITD of the lateral angle (variation < 70 µs across the gap) and is continuous with the measured region. I tried clamping to the nearest rim point first. It jumped from the front rim to the back rim mid-gap (−0.8 dB NMSE between neighbouring 0.5° steps), so it is now only a fallback.
- **Filter grid and cache:** filters are computed on a **0.5°** (az, el) grid, below the ~1° minimum audible angle, with an LRU cache. A finer 0.1° grid rebuilt the filter in 97 % of blocks during movement; 0.5° rebuilds in about 25 %. If the audio thread's measured load exceeds 55 % of the deadline, the worklet coarsens the grid to 1° automatically ("economy mode") and returns to 0.5° when the load drops. The filter is always computed for the quantised direction, so Python and the browser produce identical filters.

### Transitions and trajectories

- **Crossfade:** the direction is evaluated once per 128-sample block. When the cache key changes, the block is convolved with the old *and* new filter (same input history) and crossfaded linearly, y = (1 − r)·y_old + r·y_new with r = (n+1)/B. Gains also ramp per block. Turning the crossfade off is available as a demonstration; the test suite shows the abrupt switch produces a sample jump more than 3× larger than any in normal playback.
- **Trajectories** (`trajectory.py`, `src/dsp/trajectory.ts`): a centripetal Catmull–Rom spline (α = 0.5) through the control points, resampled to 512 points uniformly by arc length, so speed is constant. Playback position is a phase ∈ [0, 1) driven by the audio clock, closed for loops and open otherwise.

---

## Using the application

The website is a single page built to let visitors *hear* and *understand* binaural rendering. There are no technical dashboards. The engine still chooses its methods from measurements in the background.

- **3D scene.** A seated listener wears headphones (cyan = left ear, pink = right ear). The glowing orb is the sound, and the dots show where real ear recordings were made. Click the sphere to place the sound, or drag the orb. Drag empty space to look around and scroll to zoom. The overlay says in words where the sound is ("behind-left, slightly above").
- **Listen.** Play, pause, stop, loop, volume and L/R level meters.
  - **3D sound on/off:** an instant, click-free A/B switch. Off sends the plain sound to both ears at matched loudness (the RMS ear-filter gain averaged over all directions), so the only thing that changes is the spatial processing.
  - **Room ambience:** soft, direction-independent small-room reflections (native ConvolverNode, decorrelated between the ears, ~10 dB below the direct sound at 1.5 m). They help sounds appear outside the head without disturbing the direct sound's binaural cues.
  - **Headphone check:** short bursts in the left ear only, then the right ear only, with the main mix muted. If both ears hear both, the system is playing mono (e.g. Windows "Mono audio" or spatial-sound enhancements are on) and no binaural renderer can work.
- **Try this.** One-click experiences, each with a "what to notice" line:
  - circle around your head;
  - whisper in your left ear;
  - right behind you, compared with straight in front;
  - above your head;
  - something flies past;
  - spiral up and down.
- **Choose a sound.** Four built-in sounds with waveform previews and a dry preview button. You can upload WAV/MP3 (up to 25 MB and 60 s); the file stays in the browser. For a stereo file you choose *one sound (mix L+R)* or *two sounds side by side*, and the page says how much of the stereo content a mix would remove.
- **Move the sound.**
  - **Position mode:** click or drag the sound.
  - **Draw mode:** drag on the sphere to draw a flight path.
  - **Edit mode:** drag the handles; click to add a point, shift-click a handle to remove it.
  - **Ready-made paths:** Orbit, Spiral, Over the head, Figure eight, Fly-by.
  - **Path controls:** Play/Pause/Stop/Replay/Clear, loop, and the time for one pass.
- **Turn the listener.** Turn left/right, look up/down, sound distance, and face forward. The sound stays put in the room, so turning the listener moves it around the head.
- **What your ears are hearing** (live, for the current direction):
  1. **arrival time:** a top-down view of which ear the sound reaches first, by how many milliseconds, and how many centimetres further it travels;
  2. **loudness:** L/R bars and the level difference in dB;
  3. **ear-shape colouring:** how each ear boosts and cuts pitches for this direction.

  Below these, the two actual signals sent to the headphones are drawn live. All values come from the filters the audio engine is using at that moment.
- **How binaural rendering works.** Four plain-language steps: a plain sound → real ear recordings → a filter for each ear → your brain does the rest. They are followed by a note on why generic ears do not suit everyone.
- **Dark/light mode** applies to the page, the 3D scene and the plots.

The backend's analysis endpoints (`/api/render` reference renders, `/api/analysis/holdout`, `/api/evaluation/interpolation`, `/api/benchmarks`) remain available for verification through <http://127.0.0.1:8000/docs>.

---

## Testing and evaluation

| Suite | What it checks |
|---|---|
| `tests/test_convolution.py` | direct = literal sum; all FFT methods = direct (<10⁻¹⁰); output length; impulse, silence, 1-sample inputs; L/R assignment; streaming convolvers |
| `tests/test_dataset.py` | CIPIC shape/fs/licence; native grid recovery; left ear = channel 0; synthetic SOFA files (receiver reordering, ListenerView rotation); malformed, NaN, missing and non-HDF5 files; HRTF→HRIR round trip and conjugate symmetry; MIT KEMAR through the same adapter |
| `tests/test_coordinates.py` | cardinal directions, round trips, wraparound, orthonormal basis, listener turning/pitch, quantisation |
| `tests/test_interpolation.py` | fractional shift and TOA accuracy; ITD vs cross-correlation; alignment round trip; sphere coverage; gap location; weights valid and measured exact; status regions; gap fill along the cone of confusion; wraparound continuity; smoothness along paths; convergence to measurements; TOA/ITD of interpolated filters; holes in the data; held-out validation |
| `tests/test_trajectory.py` | spline through points, uniform speed, C¹ smoothness, looping and clamping, monotonic listener-relative azimuth, degenerate inputs |
| `tests/test_audio.py` | decoding and limits; resampling; stereo policies; normalisation and silence; padding; limiter bound; static render = definition; ILD/ITD of lateral sources; block engine = reference; **crossfade removes switching clicks**; trajectory audio follows the path; stereo pair; transient and sustained inputs |
| `tests/test_api.py` | all endpoints: pack round trip and ETag, resampled packs, HRIR statuses, holdout, sounds, uploads, static and trajectory renders, validation errors |
| `frontend/src/test/golden.test.ts` | browser coordinates, fractional shift, **interpolation weights (1e-9) and filters**, trajectory, limiter, and the **full moving-source engine** (direct and UPOLS, 1 and 2 voices): identical cache keys in every block, identical update and crossfade counts, output within 1e-4 of the Python `BlockEngine` |
| `frontend/src/test/dsp.test.ts` | FFTs vs DFT definition, convolvers vs naïve convolution, coordinates, trajectory, stereo policy, normalisation, UTF-8 fallback |

**End-to-end (`frontend/e2e/e2e.mjs`, headless Chrome).** With the backend and `npm run dev` running:

```bash
cd frontend
npm i --no-save playwright-core@1.48.2
node e2e/e2e.mjs e2e-shots        # set CHROME=/path/to/chrome if it is not in the default location
```

The script drives the app through the real worklet and asserts 14 checks (all pass):

- start audio, play, move the source, play the Orbit trajectory;
- switch convolution method and interpolation during playback;
- upload a stereo white-noise WAV and reject a corrupt file;
- run the "Whisper in your left ear" experience and verify that the ear-cue panel explains the left-ear timing and loudness;
- 3D off gives no left/right difference (0.00 dB); 3D on with room ambience still gives > 6 dB at 90°;
- the headphone check sounds in one ear only;
- the audio thread's measured load stays below 50 %;
- toggle the theme.

Broadband ILD at the ears: **+16.4 dB at 90°, −17.0 dB at 270°, +11.6 dB at 45°, −13.7 dB at 315°**. During the orbit the worklet-reported azimuth advanced counter-clockwise in step with the phase. The run produced no console errors. This run also found a real bug (`TextDecoder` is missing in AudioWorkletGlobalScope), which is now fixed and covered by a unit test.

---

## Measured results

Measured on the development machine (Windows 10, Python 3.11.5, NumPy 2.4.6; Chrome 157). Full data: `backend/results/*.json`. Browser numbers come from `src/dsp/benchmark.ts` and from the worklet's own load measurement.

### Interpolation (6 CIPIC subjects, 10 % of directions withheld, seed 7)

| method | LSD dB | NMSE dB | \|ΔITD\| µs | \|ΔTOA\| µs | vertex consistency | |
|---|---|---|---|---|---|---|
| barycentric · minimum phase | 1.86 | −0.1 | 15.5 | 26.0 | +3 dB ✗ | rejected: does not converge to the measured HRIRs |
| **barycentric · onset-aligned** | **2.16** | **−13.1** | **13.2** | 12.5 | −49 dB ✓ | **selected** |
| IDW · onset-aligned | 2.20 | −11.6 | 15.0 | 12.2 | −49 dB ✓ | |
| nearest neighbour | 2.61 | −8.0 | 28.3 | 18.3 | exact | rejected: ITD |
| barycentric · raw | 4.10 | −11.9 | 20.6 | 17.5 | exact | rejected: ITD; comb filtering |
| IDW · raw | 4.87 | −10.7 | 21.4 | 18.5 | exact | rejected: ITD |

**Selection rule**, applied automatically: lowest mean LSD among methods with mean ITD error < 20 µs, 95th-percentile vertex-consistency NMSE < −30 dB, and query cost < 5 ms. Minimum phase has the lowest LSD, but it discards the measured phase. Because measured HRIRs are used verbatim at measurement points, it would cause a filter jump whenever a moving source passes one. With 25 % withheld, the ranking is the same.

### Convolution

| | direct | FFT (UPOLS) |
|---|---|---|
| Python, 1 s ⊛ 200 taps | **3.11 ms** | 3.36 ms (single FFT), 4.15 ms (overlap-add), 31.3 ms (UPOLS block loop) |
| Python, per 128-block, 2 ears × 2 filters | **40 µs** | 206 µs |
| Chrome, kernel per block, 2 ears × 2 filters | 476 µs | **52 µs** |
| Chrome, full engine, moving source, 1 voice | 600 µs (20.7 %) | **225 µs (7.7 %)** of the 2 903 µs deadline |
| Chrome, full engine, moving source, 2 voices | 1068 µs | **282 µs** |
| Chrome, static source, 1 voice | 238 µs | **37 µs** |

All FFT methods match direct convolution to ≈ −305 dB (float64 rounding). Uncached interpolation in Chrome costs 181 µs (aligned), 91 µs (raw) and 8 µs (nearest) per filter. That is why interpolation can run inside the audio callback once per block. Headless Chrome coarsens `performance.now()` to 100 µs, so the benchmark times batches of 10–20 blocks.

### Responsiveness (laptop lag fixes)

Measured in Chrome (headless) while a source orbits the listener:

| | before | after |
|---|---|---|
| page frozen when starting sound | 2204 ms (synchronous start-up benchmark) | ≈ 550–700 ms (loading the worklet and filters; no long tasks) |
| filter rebuilds during movement | 97 % of blocks (0.1° grid) | ≈ 25 % (0.5° grid) |
| audio-thread allocations per block | arrays, objects and strings | none (preallocated) |
| main-thread long tasks, CPU throttled 4× | 64 tasks, 3.4 s per 8 s | 0–4 tasks, ≤ 0.23 s |
| frame rate, normal speed | 56 fps | 60 fps |
| audio-thread load (measured in the worklet) | – | 6–7 % of the 2.9 ms deadline |

Changes made:
- the 3D view renders at ≤ 1.5× pixel density and adapts down to 1× when frames drop;
- shadows are rendered only when the listener turns;
- labels are GPU sprites instead of per-frame HTML;
- rendering pauses while the view is off screen;
- the ear-signal plot animates only while sound plays and is visible.

### Built-in sounds: directional content

A sound localises only if it has energy where the head creates cues; low frequencies diffract around the head. Broadband ILD at 90° through subject 165's filters:

| sound | before | after |
|---|---|---|
| helicopter | 2.7 dB (84 % of energy < 300 Hz) | **13.7 dB** (less rumble, brighter blade slap and rotor wash) |
| AK-47 | 7.2 dB | **12.2 dB** (smaller boom, stronger crack) |
| guitar | 7.9 dB | 7.9 dB |
| voice | 2.7 dB | **3.3 dB plus a 680 µs ITD** (the TTS voice is narrow-band, so speech lateralises mainly through arrival time) |

---

## Built-in sounds and licensing

| Sound | Source | Licence |
|---|---|---|
| Helicopter | original procedural synthesis (`generate_sounds.py`, seed 11): 11 Hz blade slap, tail rotor, turbine whine, rotor wash | CC0 1.0 |
| AK-47 firing | original synthesis of three automatic bursts at the AK-47's ~600 rounds/min cyclic rate: muzzle blast, mechanism, reflections. **Not a recording.** | CC0 1.0 |
| Guitar | original Karplus–Strong synthesis, Am–F–C–G arpeggios | CC0 1.0 |
| Voice: "Come here!" | rendered locally by `scripts/generate_voice.ps1` with the Windows `System.Speech` synthesizer (Microsoft Zira Desktop), high-passed at 170 Hz with a +9 dB presence shelf above 2.5 kHz | subject to the Windows licence terms; **not published** on the website or in the repository; add your own recording instead (see *Publish the website for free*) |

All sounds pass through the same renderer; nothing is per-sound. On non-Windows systems the voice step is skipped with instructions. Save any mono WAV as `assets/sounds/come_here.wav` and rerun `generate_sounds`.

---

## Configuration

`backend/config.example.toml` is the template. Copy it to `backend/sonicsphere.toml`, or point `SONICSPHERE_CONFIG` at a file.

- `[datasets.*]`: name, directory, file pattern, default subject, download URL, licence summary.
- `[rendering]`: block size, alignment margin, onset threshold, interpolation method, gap threshold, snap and quantum angles, normalisation, limiter, stereo-pair width.
- `[upload]`: limits.

Changing preprocessing parameters invalidates the filter-pack cache automatically.

Frontend: `SONICSPHERE_BACKEND=http://host:port npm run dev` changes the proxy target.

---

## Project layout

```
backend/
  sonicsphere/  config · dataset_loader · coordinate_system · hrir_preprocessing · hrir_interpolation
                convolution · audio_processing · audio_renderer · trajectory · filter_pack
                interpolation_eval · benchmark · export_golden · export_static · download_dataset · generate_sounds
                schemas · main (FastAPI)
  tests/        141 pytest tests
  results/      interpolation_eval.json, benchmark.json
  config.example.toml, requirements.txt
frontend/
  src/dsp/      fft · coordinates · filterPack · fracshift · interpolation · convolver · engine
                limiter · trajectory · sourcePrep · benchmark      (shared by worklet, UI and tests)
  src/audio/    spatializer.worklet.ts · AudioEngine.ts · protocol.ts
  src/scene/    Scene.tsx (interactions) · Listener.tsx (seated figure with headphones) · palette.ts
  src/components/ panels, MathLab, Docs, charts, scopes
  src/test/     golden.test.ts, dsp.test.ts
assets/sounds/  built-in sounds + manifest.json
scripts/        generate_voice.ps1
```

---

## Limitations

- **Generic HRIRs.** They are not your ears, so expect front/back confusion, elevation errors and in-head localisation for some listeners. Try several subjects.
- **Far-field responses (1 m).** Distance changes only the gain (1/r, clamped to −12…+6 dB). Near-field effects are not modelled.
- **No head tracking, room acoustics or headphone equalisation.**
- **CIPIC's gap** below the head is gap-filled, not measured; the UI flags it.
- **Unmeasured pole caps.** Directions within 10° of the interaural axis (beyond lateral ±80°) are interpolated from the 80° ring.
- **Live worklet timing.** Chrome does not expose `performance.now()` inside AudioWorkletGlobalScope, so live per-block timing cannot be measured there. The worklet therefore estimates its load from `Date.now()`; with 1 ms resolution the mean over many blocks is still unbiased. It reported 6–7 % of the deadline during movement on the development machine.
- **Cache size.** The 768-entry filter cache (~8 MB) is too small to hold a full slow orbit, so revisited directions are recomputed; at ≈225 µs/block this stays far inside the deadline.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Backend unreachable" banner | Start `uvicorn sonicsphere.main:app --port 8000` in `backend/` with the venv active. |
| "Subject … is not downloaded" | `python -m sonicsphere.download_dataset --subjects <id>` |
| Audio status "Suspended" / "Interrupted" | Click the status pill; browsers need a user gesture and suspend audio in background tabs. |
| Left and right swapped | Check headphone orientation. A source at 90° (LEFT label) must be louder in the left cup. |
| Crackling | Choose "FFT (UPOLS)", close heavy tabs, compare the benchmark's load % with the deadline. |
| Browser refuses 44.1 kHz | Handled automatically: filters are fetched resampled to the context rate. |
| `npm test` says golden vectors missing | `cd backend && python -m sonicsphere.export_golden` |
| MP3 upload fails | Decoding is done by the browser; convert to WAV if your browser lacks the codec. |
| HDF5 / h5py install error | Use Python 3.11+ 64-bit; `pip install --upgrade pip` first (wheels exist for Windows, macOS and Linux). |

---

HRIR data: CIPIC HRTF Database © The Regents of the University of California, used under the licence above. SOFA conversion by the Acoustics Research Institute (Majdak, Mihocic).
