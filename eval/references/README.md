# Reference transcriptions

One folder per document, named after the PDF's stem; one file per page (`pNN.txt`, 1-based).

**Status: silver.** Transcribed by Claude from the page images, after it had already seen
Sarvam's OCR of page 6 (so p06 may be biased towards the OCR). Waiting for a human
spot-check. Fix a file in place if you find a mistake; the next `python -m eval.run` uses it.

Conventions:
- Transcribe what the student finally wrote: crossed-out words are left out, inserted
  corrections are put in place. The student's own spelling is kept (e.g. नीयत, संधारित).
- Printed parts of the answer sheet (प्रश्न क्र., the B S E margin letters, page-total
  boxes) and diagram labels are left out. The scorer strips the printed parts from the OCR too.
- Two-column tables: left column first, then the right column (Sarvam's reading order).
- Maths in plain Unicode: `ε = NBAω sin ωt`, `λ/d`, `n²h²ε0/πme²`.

Source: Board of Secondary Education, Madhya Pradesh, "Answer books of Meritorious Students,
Class 12th Exam-2025", roll no. 253140887, Physics (https://www.mpbse.nic.in/Top-stu-answer12-2025.html).
