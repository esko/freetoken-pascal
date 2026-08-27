# Security policy

## Supported versions

Only the latest `main` branch and the latest tagged release receive security fixes during initial development.

## Reporting

Do not open a public issue for a vulnerability that enables arbitrary code execution, unsafe model-file parsing, path traversal, network authentication bypass, or host/GPU memory corruption. Contact the repository owner privately through GitHub.

## Security boundaries

Model files and chat requests are untrusted inputs. Implementations must validate:

- GGUF metadata, tensor sizes, offsets, row strides, and quant identifiers;
- cache slot indices and expert IDs;
- host and device buffer bounds;
- file paths and mmap ranges;
- server payload sizes, concurrency, and cancellation;
- subprocess invocation and environment variables.

CUDA and C++ code must fail closed on unsupported shapes or types. Never return uninitialized buffers for an unknown quant type.
