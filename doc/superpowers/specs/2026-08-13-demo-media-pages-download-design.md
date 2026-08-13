# GitHub Pages demo download design

## Goal

Clicking either DMA-BUF MP4 link in the repository README must start a Chrome
download named `16-input-grid-10s.mp4`. The solution must not create a GitHub
Release, add a gallery UI, or otherwise change the README's existing content.

## Published surface

GitHub Pages acts only as a static demo-media host. The dedicated `gh-pages`
branch publishes:

- files below `demos/*/docs/`, preserving their repository-relative paths;
- one reusable `download.html` bridge at the Pages site root.

The site does not publish the repository root, source code, or generated
documentation.

## Download bridge

The README links to the project Pages URL with the media path in a query
parameter, for example:

```text
https://amagimedia.github.io/avplumber/download.html?file=demos/dmabuf-browser/docs/16-input-grid-10s.mp4
```

On load, `download.html` validates that `file` is a relative path below
`demos/`, contains no `..` segment, and ends in `.mp4`. It resolves the file
against the `/avplumber/` project-site base, creates a same-origin anchor with
the `download` attribute set to the path's basename, and clicks it once.

The page retains a visible download link and a short status message. This is a
fallback for browsers that block the automatic click or users who revisit the
page. Invalid paths produce an error and never navigate to another origin.

## README change

Only the destinations of the existing clickable poster and the existing
“Watch or download” text link change. The poster, surrounding prose, and the
rest of `demos/dmabuf-browser/README.md` remain unchanged.

## Deployment

The published tree is assembled from `.github/pages/download.html` and files
below `demos/*/docs/`, then committed to the dedicated `gh-pages` branch.
Repository Pages uses its built-in branch deployment. This remains compatible
with organizations whose IP allow list blocks API access from GitHub-hosted
Actions runners. No GitHub Release is created.

## Validation

Before deployment:

1. Confirm the staging directory contains only `download.html` and
   `demos/*/docs/` content.
2. Serve it below a local `/avplumber/` base and verify that a Chrome click on
   the README destination downloads the MP4 with the expected filename.
3. Verify the downloaded file's SHA-256 matches the repository MP4.
4. Confirm invalid, absolute, and parent-traversal query paths are rejected.

After deployment, confirm the workflow succeeded, the Pages MP4 URL returns
successfully, and the README destination initiates the same download in
Chrome. The original repository-relative MP4 remains the source artifact.

## Success criteria

- One click from either existing README MP4 link initiates a Chrome download.
- The downloaded file is the committed 600-frame, 10-second MP4 and retains
  its `.mp4` filename.
- No GitHub Release or gallery page is created.
- Future MP4s under another `demos/<name>/docs/` directory can use the same
  bridge without a new downloader implementation.
