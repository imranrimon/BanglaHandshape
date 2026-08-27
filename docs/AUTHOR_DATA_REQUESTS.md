# Author data requests — signer/participant groupings (E13, E17)

Both requests ask ONLY for an **anonymous** grouping (an integer participant id per
image). No names, consent forms, or identity documents — just the label that lets us
reproduce a signer-disjoint split. Send from your WVU address; CC your advisor.

Contacts to fill in:
- **RSBdSL38** — corresponding author of arXiv:2608.06252 ("Toward Deployable Bangla
  Sign Language Recognition…"); Mendeley 10.17632/tgvmb2jsdb.1 contributor; GitHub
  `saadbaust`. Find the email on the arXiv PDF first page.
- **BDSL49** — corresponding author of the BDSL49 (49-class BdSL) dataset paper;
  Mendeley dataset `k5yk4j8z8s` contributor page lists the contact.

---

## Email 1 — RSBdSL38 authors (E13)

**Subject:** Request: anonymous signer grouping for RSBdSL38 (reproducing your SI split)

Dear Dr. [Last name],

Congratulations on RSBdSL38 and the accompanying paper (arXiv:2608.06252) — the
expert validation and the signer-independent evaluation are exactly the kind of rigor
the field needs.

I am a graduate researcher at West Virginia University building a **signer-independent
still-image benchmark** for Bangla Sign Language handshape recognition, and I would
like to include RSBdSL38 as an external, signer-diverse source and to **faithfully
reproduce your signer-independent experiment** (the 6-held-out-signer setting).

The public release (Mendeley 10.17632/tgvmb2jsdb.1 and the `rsbdsl38-final` Kaggle
mirror) provides the class-organized `train/val/test` folders, but not the
participant labels; the signer-independent partition (`rsbdsl38-o-signer-m`) appears
to be private. Could you share **either**:

1. an anonymous CSV mapping each image filename → an integer participant id (1–36),
   **or**
2. the exact file lists you used for the signer-independent `train`/`val`/`test`
   folders,

whichever is easier? I only need the **anonymous grouping** — no participant names,
consent forms, or any identifiable information. It will be used solely to construct a
signer-disjoint split for evaluation.

I will of course cite RSBdSL38 and acknowledge your help, and I am happy to share our
results on your dataset with you. Thank you very much for considering this.

Best regards,
[Your name]
[Department / West Virginia University]
[email]

---

## Email 2 — BDSL49 authors (E17)

**Subject:** Request: anonymous participant IDs for BDSL49 Recognition images

Dear Dr. [Last name],

Thank you for releasing the BDSL49 dataset — I am using the Recognition task in a
**signer-independent** still-image handshape benchmark at West Virginia University.

To run a signer-disjoint evaluation I need to know which participant produced each
image. The released images are organized by class, and I could not find a participant
field in the filenames or image EXIF. Could you share an **anonymous** mapping from
image filename → an integer participant id (for the reported participants)? I do not
need any names or identifiable information — only the grouping, so that images from
the same person can be kept on the same side of a train/test split.

I will cite BDSL49 and acknowledge your assistance. Thank you for your time.

Best regards,
[Your name]
[Department / West Virginia University]
[email]

### BDSL49 response (received 2026-08-14)
Dr. Md. Rajibul Islam (BUBT) replied: the requested per-image participant mapping "wasn't included in the data curation process, so it is not available." **Outcome:** E17 is closed — BDSL49 cannot be made signer-independent without re-collection. Folded into the paper (BdSL47 = sole SI source). No further action.

### RSBdSL38 response (received 2026-08-13)
Saad Ahmed (BAUST) released **V2** (doi 10.17632/tgvmb2jsdb.2) with the exact signer-independent train/val/test file lists (images provided directly, no participant list). **Outcome:** E13-faithful DONE — faithful within-RSBdSL38 SI = 91.6+/-0.2% (vs 96.9% leaky provided split). Folded into the paper.
