# Security Policy

## Supported Versions

Only the latest release on PyPI receives security updates.

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you believe you have found a security vulnerability in mosaicraft, please
report it privately. **Do not** open a public issue.

* Use [GitHub Security Advisories](https://github.com/hinanohart/mosaicraft/security/advisories/new) to report privately.
* Please include:
  * A clear description of the issue and its impact.
  * Steps to reproduce, ideally with a minimal example.
  * The version of mosaicraft and its dependencies.

We will acknowledge your report within a reasonable timeframe and work with
you on a fix and coordinated disclosure.

## Scope

mosaicraft is an image-processing library that reads and writes local image
files. The most relevant classes of issues are:

* Crashes or memory exhaustion triggered by malformed input images.
* Path traversal or arbitrary file write through user-supplied paths.
* Vulnerabilities in our dependency surface (NumPy, OpenCV, scikit-image,
  SciPy).

## Out of scope

* Issues affecting only third-party forks or vendored copies.
* Resource consumption from intentionally enormous inputs (the library does
  not promise constant-memory operation).
* Issues that require an attacker to already control the local filesystem
  or Python environment.
