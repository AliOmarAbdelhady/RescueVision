# Person 6 — Report Completion and Editorial Control

Owner: Marwan Mesbah · Reviewer of my work: Person 2 · I review: Person 2
Sources: Six-Person Documentation Plan (pp. 3, 9, 12, 15–18), Project I Requirements Guide, Graduation Project Guidelines (Word template).

---

## 1. What I own

| Part | What it contains | Can start now? |
|---|---|---|
| Master file | One Word/Google Doc built from the template's styles; all chapters merged into it | **Yes** |
| Cover page | Institution, college, department, degree, title, subtitle, 6 names, supervisor, month–year | Yes (fill only confirmed facts) |
| Declaration | Template wording unchanged; each student's name, reg. no., signature, date | Yes (collect details) |
| Dedication | Optional, no heading, centred | Optional |
| Acknowledgments | Real contributions only (supervisor, team, anyone who helped) | Draft now, finalise last |
| Abstract | ~200 words: problem → proposed solution → scope → contribution | **No — write last** |
| TOC, List of Figures, List of Tables, Acronyms | Auto-generated; acronym list kept as chapters arrive | Set up now, update last |
| **System Prototype** chapter | Captioned, honest description of real preliminary material | **Yes — main writing task** |
| Discussion, Conclusion & Future Work | With Person 1; links back to objectives | After chapters stabilise |
| References & Appendices | With Person 2: order, format, cross-check | Ongoing |
| Final checks | Formatting, numbering, terminology, page-by-page PDF review | End |
| Contribution record + unresolved-items list | Who wrote what; open questions for supervisor | Start now |

---

## 2. What I need from each person

| From | I need | Used for |
|---|---|---|
| Person 1 | Agreed one-paragraph project definition, objectives list, scope table, glossary, constraints | Abstract, Discussion, Conclusion, Acronyms |
| Person 2 | Reference list (IEEE, numbered by first citation), reading register | References, citation check, my review of P2 |
| Person 3 | User roles, feature list, assumptions/limitations | Prototype "what it does not cover", Discussion |
| Person 4 | Requirement IDs (FR-xx, NFR-xx) | Map each prototype figure to the requirements it touches |
| Person 5 | Diagrams + captions, list of figures; screen *illustrations* clearly labelled as designs | Figure numbering; keep illustrations separate from real prototype screenshots |
| Everyone | Name, registration no., signature, date; any real screenshots/results they have | Cover, Declaration, Prototype |

Send this table to the group early. Add a deadline for each item before it goes out.

---

## 3. System Prototype chapter — what actually exists in the repo

Rule from the plan: *show only genuine material, say what it demonstrates and what it does not.* Here's what's in `RescueVision` (branch `feature/mapping-routing`) right now:

| # | Evidence | File(s) | What it demonstrates | What it does NOT establish |
|---|---|---|---|---|
| P-1 | Before/after satellite mosaics of Kahramanmaraş city centre (2023 Türkiye earthquake), ~0.8 × 0.8 km at 0.5 m/px | `mapping/output/city_before.png`, `city_after.png` | Real post-disaster imagery can be collected and assembled into a 2D area view | Nothing about damage detection accuracy |
| P-2 | Interactive 2D map with BEFORE/AFTER toggle and legend | `mapping/output/satellite_city_map.html` (screenshot it) | The 2D area-view concept for responders | Entry/exit routes on it are **placeholders**, not computed routes |
| P-3 | Side-by-side synced comparison with a "generated" change overlay | `mapping/output/compare.html`, `damage_overlay.png` | Visual before/after comparison | The overlay is a **pixel-change baseline**, not a trained model output |
| P-4 | Process view: collect tiles → mosaic → map | `mapping/output/process.html` | The data-preparation pipeline | — |
| P-5 | Scene browser of all Maxar scenes for the area | `mapping/output/dataset.html` | Data availability and selection | — |
| P-6 | Web dashboard components (upload, image overlay viewer, damage summary cards, route map, report panel, similar cases) | `dashboard/src/components/` | Planned operator interface | Use only if it runs and you screenshot it working; otherwise it's not evidence |
| P-7 | Mobile field app + web ops app scaffold | `frontend/apps/mobile`, `frontend/apps/web` | Planned structure | Same: screenshot only if it runs |
| P-8 | Auto-generated emergency report template | `outputs/*/report.md` | Report format/template | Current samples show **0 buildings, "No damage map available"**. Don't present these as results |

**Gaps to flag (put them on the unresolved-items list, not in the prose):**
- `docs/experiments.md` lists all 11 model experiments (E1–E11) as **Pending**. No trained-model metrics exist in the repo. If a teammate ran the Kaggle notebooks, ask them for their numbers and screenshots.
- The ML pipeline code (`src/rescuevision/`, `tests/`) was removed in commit `f89a61f` and only exists in the initial commit `3c699f7`. Ask the team whether it's still part of the project.
- **Scope mismatch:** the plan describes 2D **and 3D** views, **person tracking** and **hazard markers**. The prototype covers 2D satellite before/after damage mapping only. State this plainly in the chapter, e.g. "3D views and person tracking are not yet prototyped". Agree the wording with Persons 1 and 3.
- **Image licence:** Maxar Open Data imagery must be cited as a source. Check its licence terms and caption every satellite figure "Source: Maxar Open Data Program [n]".

**Suggested chapter outline**

1. Purpose of the prototype (1 paragraph: what question it answers)
2. Data used (Maxar Open Data, event, area, dates, resolution) → Table: data summary
3. Prototype components, one subsection per evidence item P-1…P-5 (+ P-6/P-7 if working). Each has: intro sentence → figure → explanation → linked requirement IDs from Person 4
4. Relation to the proposed system (which features it touches, which it doesn't) → Table: feature vs. prototype coverage
5. Limitations of the prototype evidence
6. Next prototype steps (replace pixel-change overlay with model output, compute real routes, add 3D, etc.), which feed into Future Work

Figure numbering: chapter-hyphen (e.g. Figure 6-1). Caption below the figure, bold 10 pt, centred. Every figure must be mentioned in the text.

---

## 4. Discussion, Conclusion & Future Work (with Person 1)

The template says the conclusions "should not be very short". Structure:

- **Discussion:** for each objective from Person 1, explain how the report addresses it (requirement IDs, design figures, prototype evidence). Cover the value of the proposal, its limits (data quality, uncertainty, need for human judgement, privacy of person data), and how the design responds to the 1.7 constraints.
- **Conclusion:** restate the problem, what this Project I delivered (SRS + SDD + preliminary prototype), and what the evidence does and doesn't show. Don't claim outcomes that weren't measured.
- **Future Work (Project II):** train and evaluate the models (E1–E11), real route computation, 3D view, person tracking, field app, testing with real users.

---

## 5. Master file and formatting checklist

**Set up once:** A4, single-sided, margins L 4 cm / R 2 cm / T,B 2.54 cm. Build named styles; don't format paragraphs by hand.

| Element | Format |
|---|---|
| Body | Times New Roman 12, justified, 1.5 spacing, no first-line indent, 12 pt before |
| Chapter label | TNR italic 14, centred, expanded 3.5, 24 pt after, page break before |
| Heading 1 | TNR bold 16, ALL CAPS, centred, 12 before / 36 after, numbered level 1 |
| Heading 2 | TNR bold 14, ALL CAPS, left, 18 before / 12 after, numbered level 2 |
| Heading 3 | TNR bold 14, Title Case, left, 12 / 12, numbered level 3 |
| Heading 4 / 5 | Bold 13 / underlined 12, Title Case, 12 / 12 |
| Table text | TNR 12, left, single, 6 / 6 |
| Figure caption | Below figure, TNR bold 10, centred, 6 before / 24 after |
| Table title | Above table, TNR bold 12, left, 12 / 12 |
| Cover | Logo B&W 3.25 × 3.25 cm centred; Academy title bold 18; college bold 16; dept bold 16; degree 14; TITLE bold 16 CAPS; subtitle bold 16; names italic 14; month–year 10 in footer |

**Final page-by-page checks**
- [ ] Every required section from both guides is present (cover → references)
- [ ] TOC, figure list and table list regenerated after the last edit
- [ ] All figures and tables numbered by chapter, referenced in the text, with sources cited
- [ ] Design illustrations labelled as illustrations, never as screenshots
- [ ] Every acronym defined at first use *and* listed in the acronym list
- [ ] References in IEEE format, numbered by first citation, one per number, all cited in the text
- [ ] Same terms everywhere (roles, "assistance case", "hazard", "response status", etc.)
- [ ] Proposed/intended wording for anything that isn't built
- [ ] No contractions, serial comma used, 0 before decimals, consistent tense
- [ ] Exported PDF read page by page: page numbers, broken references, image quality

---

## 6. Reviewing Person 2 (clarity, structure, formatting)

- [ ] Organised by themes/questions, not a list of paper summaries
- [ ] Each theme ends with "what this means for RescueVision"
- [ ] Related-systems comparison table present; unknowns marked
- [ ] Gap statement supported by the cited sources
- [ ] Citation style and order correct

---

## 7. Questions for the supervisor (unresolved-items list)

1. Abstract length: 150–200 words (Project I guide) or 200–250 (template)? Draft at ~200 for now.
2. Does the 50-page minimum for the main chapters apply to Project I?
3. Final chapter arrangement and headings (merged SRS + SDD layout vs. template Chapters I–V)
4. Expected depth of the prototype evidence; is the mapping prototype enough?
5. Final project title and subtitle; department name; submission month and year
6. Submission files required (Word + PDF? printed copy?)

---

## 8. Writing style rules (my "humanizer" checklist)

Apply these to every text before it goes into the master:

- Write in a plain academic register. Prefer passive or "we" consistently, as the template allows.
- Vary sentence length. Avoid the same opening word across consecutive sentences.
- Delete filler and AI tells: "delve", "crucial", "pivotal", "seamless", "robust", "leverage", "in today's world", "it is important to note", "plays a vital role", "a testament to", "furthermore/moreover" at every paragraph start, triplet lists everywhere, em-dash chains.
- One idea per paragraph. Topic sentence → evidence/citation → link to RescueVision.
- Make concrete claims with numbers and citations. If there's no source, soften the claim or cut it.
- Keep hedging honest: "is intended to", "the prototype shows…, but does not establish…".
- Read each section aloud once. If a sentence sounds like a brochure, rewrite it.

---

## 9. Order of work

1. **Now:** build the master file with styles. Draft the cover, declaration, acknowledgments skeleton and acronym list. Send the "what I need" table to the group. Send the supervisor questions.
2. **Now:** write the System Prototype chapter from section 3 (screenshots of P-1…P-5, captions, limits).
3. **As chapters arrive:** merge them, normalise terms and tense, number figures and tables, and keep the acronym list and contribution record up to date.
4. **When chapters are stable:** write the Discussion, Conclusion & Future Work with Person 1, then the Abstract.
5. **End:** finalise references with Person 2, regenerate the lists, export the PDF and check it page by page. Get approval from all six.
