# Release checklist / 上传前核对

The package is prepared locally. External deposition has not occurred.

1. Author supplied: XiaZhiyuan; ORCID 0009-0009-8897-1883; intended repository name SunGroupAI. Add affiliations and any additional creators. The GitHub account owner/URL and archive DOI are not yet known.
2. The author chose a submission-review package without selecting an open-source licence. Before a future public release, choose a software licence for your own code. A software licence does not automatically grant redistribution rights to third-party CIFs, text or embeddings.
3. Confirm the right to redistribute included processed inputs, embedded CIFs and text. If some input must remain separate, provide a stable data record and matching checksum rather than leaving a broken local path.
4. Reconcile the manuscript methods with the two dataset versions (180 supervised / 126 AL), model differences, numeric chemical-prior versus text modality, and the selection limitations documented in REPRODUCIBILITY.md.
5. Check the validation report. Full retraining is not certified; do not promise exact historical retraining until it has been independently checked.
6. Create a GitHub repository, upload the extracted package contents, and tag the checked release. Preserve the complete representation-level results alongside the selected figure contrasts.
7. Archive the version on Zenodo or another permanent software repository. Enter actual repository URL and version DOI into the metadata and manuscript only after assignment.
8. Provide a separate Data Availability statement for the processed inputs and raw-label source. Final paper title, author list, funding and target-journal requirements are NEED_DATA.

Official references checked during preparation:

- Code sharing and documentation: https://support.nature.com/en/support/solutions/articles/6000237619-software-and-code-sharing
- Example journal guidance on DOI repositories: https://www.nature.com/ncomms/editorial-policies/reporting-standards
- GitHub release archiving: https://help.zenodo.org/docs/github/archive-software/github-upload/

Suggested statement AFTER verified deposition:

The code for the supervised multimodal benchmark, retrospective active-learning simulations and figure-level analyses is available at [GITHUB_URL]. The accompanying processed inputs, fixed split files, reference outputs and reproduction instructions are archived as version [RELEASE_VERSION] at [ARCHIVE_DOI].

Before deposition, use this only as a draft. Do not submit a statement with placeholder links or an unassigned DOI.
