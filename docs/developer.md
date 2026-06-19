# Developer Guide

This page is the entry point for contributor-facing engineering notes.

## Start here

- [Developer architecture](developer/architecture.md)
- [Service result pattern](developer/service_result_pattern.md)
- [Widget template](developer/widget_template.md)
- [Widget help (F1)](developer/widget_help.md)
- [Testing guide](developer/testing.md)

## Related project docs

- [Packaging notes](packaging.md)
- [Optional dependency guide](optional_dependencies.md)
- [Troubleshooting](troubleshooting.md)
- [Release process](release_process.md)

## Short version

The current package direction is:

- keep Orange widgets thin
- move chemistry and data logic into `chemcore`
- avoid silent failures
- prefer small, testable helper functions
- keep optional dependencies optional
- add targeted tests for every behavioral change
