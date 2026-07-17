# SPDX-License-Identifier: Apache-2.0
import os
import sys
import shutil
import tempfile
import platform
import logging
import subprocess
import hashlib
from pathlib import Path
from cffi import FFI

# Set up a logger
logger = logging.getLogger("vfhe.dynamic_extensions")

# Module level state storing the custom source files added by the user
_custom_c_files = []
_custom_cdef_files = []
_custom_cdef_strings = []
_temp_c_files = []

# List of callback functions that will be invoked upon successful compilation and library reloading.
# Each callable in this list is executed after the new CFFI library has been loaded.
REINITIALIZATION_REGISTRY = []


def register_reinitializer(func):
    """Register a callback function to be called after a new C library is compiled and loaded.

    The function will be called with (new_ffi, new_lib) as its arguments.
    """
    if func not in REINITIALIZATION_REGISTRY:
        REINITIALIZATION_REGISTRY.append(func)
    return func


def find_vfhe_root() -> Path:
    """Find the root directory of the vfhe project containing 'modules' and 'native/discovery.py'."""
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "modules").is_dir() and (current / "native" / "discovery.py").exists():
            return current
        current = current.parent
    raise RuntimeError(
        "vfhe root directory containing 'modules' and 'native/discovery.py' not found. "
        "Please ensure the vfhe source files are available."
    )


def _enable_asm_sources() -> None:
    """Register .S and .s files to be compiled using the compiler."""
    # First, import the standard modules to ensure they are loaded
    modnames = (
        "distutils.ccompiler",
        "distutils.unixccompiler",
        "distutils.compilers.C.base",
        "distutils.compilers.C.unix",
        "setuptools._distutils.ccompiler",
        "setuptools._distutils.unixccompiler",
        "setuptools._distutils.compilers.C.base",
        "setuptools._distutils.compilers.C.unix",
    )
    for modname in modnames:
        try:
            __import__(modname, fromlist=["*"])
        except Exception:
            pass

    # Patch every loaded distutils module class and module-level variables
    import sys
    for name, module in list(sys.modules.items()):
        if ("distutils" in name or "setuptools" in name) and module:
            for k, obj in list(vars(module).items()):
                if isinstance(obj, type) and hasattr(obj, "src_extensions"):
                    exts = obj.src_extensions
                    if isinstance(exts, list):
                        for e in (".S", ".s"):
                            if e not in exts:
                                exts.append(e)
                # Monkey-patch _make_out_path_exts if present on classes
                if isinstance(obj, type) and hasattr(obj, "_make_out_path_exts"):
                    if not getattr(obj, "_make_out_path_exts_patched", False):
                        orig_method = getattr(obj, "_make_out_path_exts")
                        def patched_method(cls, output_dir, strip_dir, src_name, extensions):
                            if extensions is not None:
                                if isinstance(extensions, dict):
                                    if ".S" not in extensions:
                                        extensions[".S"] = ".o"
                                    if ".s" not in extensions:
                                        extensions[".s"] = ".o"
                                elif isinstance(extensions, list):
                                    if ".S" not in extensions:
                                        extensions.append(".S")
                                    if ".s" not in extensions:
                                        extensions.append(".s")
                            return orig_method(output_dir, strip_dir, src_name, extensions)
                        setattr(obj, "_make_out_path_exts_patched", True)
                        setattr(obj, "_make_out_path_exts", classmethod(patched_method))

            exts = getattr(module, "src_extensions", None)
            if isinstance(exts, list):
                for e in (".S", ".s"):
                    if e not in exts:
                        exts.append(e)


def _host_has_avx512ifma() -> bool:
    """True if -march=native enables AVX-512 IFMA on this machine."""
    import shlex
    import sysconfig

    cc = os.environ.get("CC") or sysconfig.get_config_var("CC") or "cc"
    argv = [*shlex.split(cc), "-march=native", "-dM", "-E", "-x", "c", os.devnull]
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    except Exception:
        return False
    return out.returncode == 0 and "__AVX512IFMA__" in out.stdout


def add_c_file(path: str):
    """Add a C or assembly file (.c, .S) to be compiled with the library."""
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Source file not found: {abs_path}")
    if not abs_path.endswith((".c", ".S")):
        raise ValueError(f"Unsupported file type (expected .c or .S): {abs_path}")
    if abs_path not in _custom_c_files:
        _custom_c_files.append(abs_path)
        logger.info(f"Added source file: {abs_path}")


def add_cdef_file(path: str):
    """Add a CFFI declaration file (.cdef) to be processed with the library."""
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Declaration file not found: {abs_path}")
    if not abs_path.endswith(".cdef"):
        raise ValueError(f"Unsupported file type (expected .cdef): {abs_path}")
    if abs_path not in _custom_cdef_files:
        _custom_cdef_files.append(abs_path)
        logger.info(f"Added CFFI declaration file: {abs_path}")


def add_c_definitions(definitions: str):
    """Add CFFI declarations directly as a string."""
    _custom_cdef_strings.append(definitions)
    logger.info("Added CFFI declarations directly.")


def add_c_code(code: str):
    """Add a string of C code directly to be compiled with the library."""
    fd, path = tempfile.mkstemp(suffix=".c")
    try:
        with open(fd, "w") as f:
            f.write(code)
    except Exception:
        os.close(fd)
        try:
            os.unlink(path)
        except Exception:
            pass
        raise

    _custom_c_files.append(path)
    _temp_c_files.append(path)
    logger.info("Added C code string directly.")


def add_c_dir(path: str):
    """Add all C, assembly, and CDEF files in a directory to be compiled with the library."""
    abs_dir = os.path.abspath(path)
    if not os.path.isdir(abs_dir):
        raise NotADirectoryError(f"Directory not found: {abs_dir}")

    added_any = False
    for root, _, files in os.walk(abs_dir):
        for file in files:
            full_path = os.path.join(root, file)
            if file.endswith((".c", ".S")):
                add_c_file(full_path)
                added_any = True
            elif file.endswith(".cdef"):
                add_cdef_file(full_path)
                added_any = True
    if not added_any:
        logger.warning(f"No custom files (.c, .S, .cdef) found in directory: {abs_dir}")


def clear_extensions():
    """Clear all added custom files and clean up temporary resources."""
    _custom_c_files.clear()
    _custom_cdef_files.clear()
    _custom_cdef_strings.clear()

    # Clean up temporary C files
    for path in _temp_c_files:
        try:
            if os.path.exists(path):
                os.unlink(path)
        except Exception as e:
            logger.warning(f"Failed to delete temporary C file {path}: {e}")
    _temp_c_files.clear()
    logger.info("Cleared custom extension files.")


def get_added_files():
    """Return a list of added C/assembly files."""
    return list(_custom_c_files)


def update_cffi_references(new_ffi, new_lib):
    """Update all global ffi and lib references in imported vfhe modules."""
    import sys
    libvfhe_mod = sys.modules["vfhe.misc.libvfhe"]

    old_ffi = getattr(libvfhe_mod, "ffi")
    old_lib = getattr(libvfhe_mod, "lib")

    setattr(libvfhe_mod, "ffi", new_ffi)
    setattr(libvfhe_mod, "lib", new_lib)

    libvfhe_instance = getattr(libvfhe_mod, "libvfhe")
    setattr(libvfhe_instance, "ffi", new_ffi)
    setattr(libvfhe_instance, "lib", new_lib)

    for name, module in list(sys.modules.items()):
        if name.startswith("vfhe"):
            if hasattr(module, "lib") and getattr(module, "lib") is old_lib:
                setattr(module, "lib", new_lib)
            if hasattr(module, "ffi") and getattr(module, "ffi") is old_ffi:
                setattr(module, "ffi", new_ffi)


# --- Default Reinitializers ---


@register_reinitializer
def reinit_libvfhe(new_ffi, new_lib):
    """Update the fundamental libvfhe singleton."""
    from .libvfhe import libvfhe
    libvfhe.lib = new_lib
    libvfhe.ffi = new_ffi


@register_reinitializer
def reinit_ntt(new_ffi, new_lib):
    """Update NTT processor instance and flush conversion cache."""
    try:
        from vfhe.arith.ntt import NTT_Processor
        import vfhe.arith.ntt as ntt_mod
        if hasattr(ntt_mod, "NTT_processor_instance") and ntt_mod.NTT_processor_instance:
            ntt_mod.NTT_processor_instance.cleanup()
        ntt_mod.NTT_processor_instance = NTT_Processor()
    except Exception as e:
        logger.warning(f"Could not re-initialize NTT_processor_instance: {e}")


@register_reinitializer
def reinit_lwe(new_ffi, new_lib):
    """Re-initialize LWE library prototypes."""
    try:
        from vfhe.mlwe.lwe import LibLWE
        import vfhe.mlwe.lwe as lwe_mod
        lwe_mod.lib_lwe = LibLWE()
    except Exception as e:
        logger.warning(f"Could not re-initialize lib_lwe: {e}")


@register_reinitializer
def reinit_rlwe(new_ffi, new_lib):
    """Re-initialize MLWE library prototypes."""
    try:
        from vfhe.mlwe.mlwe import LibMLWE
        import vfhe.mlwe.mlwe as mlwe_mod
        mlwe_mod.lib_rlwe = LibMLWE()
    except Exception as e:
        logger.warning(f"Could not re-initialize lib_rlwe: {e}")


# --- Compilation and Loading logic ---


def compile(output_dir=None, extra_compile_args=None, extra_link_args=None):
    """Compiles the library together with the added extensions and updates the loaded library."""
    root = find_vfhe_root()
    sys.path.insert(0, str(root / "native"))
    try:
        import discovery
    finally:
        sys.path.pop(0)

    modules_dir = root / "modules"
    aliases = discovery.host_arch_aliases()

    # Locate default headers, sources, and include directories
    preamble_headers = []
    for c_dir in sorted(modules_dir.glob("*/c")):
        preamble_headers += sorted((c_dir / "include").glob("*.h"))

    vfhe_sources = [str(src) for src in discovery.module_sources(modules_dir, aliases)]
    vfhe_include_dirs = [str(inc) for inc in discovery.module_include_dirs(modules_dir)]

    # Add vendored BLAKE3
    blake3 = discovery.blake3_dir(root)
    vfhe_sources += [str(s) for s in discovery.blake3_sources(root)]
    vfhe_include_dirs.append(str(blake3))

    is_x86 = discovery.is_x86_host()
    coverage = os.environ.get("VFHE_COVERAGE") == "1"
    force_portable = os.environ.get("VFHE_PORTABLE") == "1" or coverage
    tune = (
        (not force_portable)
        and is_x86
        and sys.platform != "win32"
        and _host_has_avx512ifma()
    )

    # Establish compile arguments, link arguments, macros, and libraries
    if extra_compile_args is None:
        if sys.platform == "win32":
            compile_args = ["/O2", "/GL"]
        elif coverage:
            compile_args = ["-O0", "-g", "-std=gnu11", "--coverage"]
        else:
            compile_args = [
                "-O3",
                "-flto",
                "-std=gnu11",
                "-Wall",
                "-Wno-unused-function",
                "-Wno-unused-result",
                "-Wno-sign-compare",
                "-Wno-write-strings",
            ]
            if tune:
                compile_args += ["-march=native", "-funroll-all-loops"]
    else:
        compile_args = list(extra_compile_args)

    if extra_link_args is None:
        if sys.platform == "win32":
            link_args = ["/LTCG"]
        elif coverage:
            link_args = ["--coverage"]
        else:
            link_args = ["-flto"]
    else:
        link_args = list(extra_link_args)

    if sys.platform == "win32":
        libraries = []
    else:
        libraries = ["m"]

    # Select defines and extra sources
    if tune:
        define_macros = []
        vfhe_sources += [
            str(blake3 / f"blake3_{x}_x86-64_unix.S")
            for x in ("sse2", "sse41", "avx2", "avx512")
        ]
    else:
        define_macros = discovery.portable_macros(is_x86)

    # Combine with custom sources
    all_sources = sorted(set(vfhe_sources + _custom_c_files))
    custom_include_dirs = {os.path.dirname(f) for f in _custom_c_files}
    all_include_dirs = sorted(set(vfhe_include_dirs + list(custom_include_dirs)))

    # Determine dynamic module name using a hash of custom files
    hasher = hashlib.sha256()
    for f in sorted(_custom_c_files):
        hasher.update(f.encode("utf-8"))
        if os.path.exists(f):
            with open(f, "rb") as fp:
                hasher.update(fp.read())
    for f in sorted(_custom_cdef_files):
        hasher.update(f.encode("utf-8"))
        if os.path.exists(f):
            with open(f, "rb") as fp:
                hasher.update(fp.read())
    for s in _custom_cdef_strings:
        hasher.update(s.encode("utf-8"))
    module_name = f"_vfhe_custom_{hasher.hexdigest()[:16]}"

    # Configure CFFI builder
    ffi = FFI()

    # Read default cdef files
    for cdef in sorted(modules_dir.glob("*/python/cdef/*.cdef")):
        ffi.cdef(cdef.read_text())

    # Read custom cdef files
    custom_cdefs = []
    for cdef_path in _custom_cdef_files:
        with open(cdef_path, "r") as f:
            custom_cdefs.append(f.read())
    custom_cdefs.extend(_custom_cdef_strings)
    custom_cdef_content = "\n".join(custom_cdefs)
    if custom_cdef_content:
        ffi.cdef(custom_cdef_content)

    # Build the C source preamble
    preamble = "\n".join(f'#include "{h.name}"' for h in preamble_headers)
    if custom_cdef_content:
        preamble += "\n\n/* Custom declarations */\n" + custom_cdef_content

    ffi.set_source(
        module_name,
        preamble,
        sources=all_sources,
        include_dirs=all_include_dirs,
        define_macros=define_macros,
        libraries=libraries,
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    )

    # Default output directory: ~/.cache/vfhe
    if output_dir is None:
        output_dir = os.path.expanduser("~/.cache/vfhe")
    os.makedirs(output_dir, exist_ok=True)

    _enable_asm_sources()

    with tempfile.TemporaryDirectory() as build_temp:
        logger.info(f"Compiling custom library '{module_name}'...")
        compiled_file = ffi.compile(tmpdir=build_temp, verbose=True)

        if not compiled_file or not os.path.exists(compiled_file):
            raise RuntimeError("Compilation succeeded but no shared library was found in the build output.")

        dest_path = os.path.join(output_dir, os.path.basename(compiled_file))
        shutil.copy2(compiled_file, dest_path)
        logger.info(f"Custom library compiled and copied to: {dest_path}")

    # Add to sys.path and import
    if output_dir not in sys.path:
        sys.path.insert(0, output_dir)

    if module_name in sys.modules:
        del sys.modules[module_name]

    new_mod = __import__(module_name)
    new_ffi = new_mod.ffi
    new_lib = new_mod.lib

    # Update CFFI references globally in all imported modules
    update_cffi_references(new_ffi, new_lib)

    # Run registered reinitializers
    for reinitializer in REINITIALIZATION_REGISTRY:
        reinitializer(new_ffi, new_lib)

    logger.info("Successfully reloaded custom C library and updated all registered package singletons.")
    return dest_path


def create_headers(target_dir=None):
    """Create a file vfhe.h that contains includes to all other headers from the library.

    If target_dir is None, the directory from where the script was called is used.
    """
    if target_dir is None:
        target_dir = os.getcwd()

    target_dir = os.path.abspath(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    root = find_vfhe_root()
    modules_dir = root / "modules"

    # Get all header files in modules/*/c/include/*.h
    headers = []
    for include_dir in sorted(modules_dir.glob("*/c/include")):
        for h_file in sorted(include_dir.glob("*.h")):
            headers.append(h_file)

    if not headers:
        raise RuntimeError(f"No header files found under {modules_dir}/*/c/include")

    vfhe_h_path = os.path.join(target_dir, "vfhe.h")

    # Write the vfhe.h file containing absolute includes
    with open(vfhe_h_path, "w") as f:
        f.write("/* Automatically generated vfhe.h wrapper for compiling extensions */\n")
        f.write("#ifndef VFHE_H_WRAPPER\n")
        f.write("#define VFHE_H_WRAPPER\n\n")
        for h in headers:
            h_clean = os.path.abspath(h).replace("\\", "/")
            f.write(f'#include "{h_clean}"\n')
        f.write("\n#endif /* VFHE_H_WRAPPER */\n")

    logger.info(f"Created headers wrapper at: {vfhe_h_path}")
    print(f"Created header wrapper file: {vfhe_h_path}")


def cli_main():
    """CLI wrapper for create_headers."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Create a vfhe.h wrapper header containing absolute path includes to the installed vfhe library headers."
    )
    parser.add_argument(
        "target_dir",
        nargs="?",
        default=None,
        help="The target directory where vfhe.h will be created. Defaults to the current working directory.",
    )
    args = parser.parse_args()

    try:
        create_headers(args.target_dir)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    cli_main()
