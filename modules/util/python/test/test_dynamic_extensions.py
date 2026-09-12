# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0
import os
import subprocess
import sys
import tempfile
import textwrap

import pytest
from vfhe import dynamic_extensions, engine
from vfhe.dynamic_extensions import _build_module

# Compiles and reloads a live C extension; heavy enough for the complete suite.
# Skipped against a gcov build: a user module links the engine archive with the
# engine's own flags, and an instrumented archive would also demand gcov's
# runtime. The installed-package smoke test is this path's real check, which is
# why coverage omits dynamic_extensions altogether (pyproject.toml).
pytestmark = [
    pytest.mark.complete,
    pytest.mark.skipif(
        os.environ.get("VFHE_COVERAGE") == "1",
        reason="a user module cannot link a gcov-instrumented libvfhe",
    ),
]


@pytest.fixture
def restore_native_lib():
    """Undo the hot swap: later tests must run against _vfhe_native."""
    orig_ffi, orig_lib = engine.ffi, engine.lib
    yield
    dynamic_extensions.update_cffi_references(orig_ffi, orig_lib)
    for reinitializer in dynamic_extensions.REINITIALIZATION_REGISTRY:
        reinitializer(orig_ffi, orig_lib)


@pytest.mark.usefixtures("restore_native_lib")
def test_dynamic_extensions_flow():
    # 1. Create a temporary directory for our user C extension files
    with tempfile.TemporaryDirectory() as user_dir:
        # User C names the module headers it needs; there is no umbrella.
        c_code = """
#include <util.h>

uint64_t my_custom_add(uint64_t a, uint64_t b) {{
    return a + b;
}}

const char *engine_via_custom(void) {{
    return vfhe_engine_active();
}}
"""
        c_file_path = os.path.join(user_dir, "my_extension.c")
        with open(c_file_path, "w") as f:
            f.write(c_code)

        # 3. Write corresponding CFFI declarations
        cdef_code = """
uint64_t my_custom_add(uint64_t a, uint64_t b);
const char *engine_via_custom(void);
"""
        cdef_file_path = os.path.join(user_dir, "my_extension.cdef")
        with open(cdef_file_path, "w") as f:
            f.write(cdef_code)

        # 4. Clear registry and register our new custom files using the new API
        dynamic_extensions.clear_extensions()

        # Verify validation error behavior
        with pytest.raises(ValueError):
            dynamic_extensions.add_c_file(cdef_file_path)

        with pytest.raises(ValueError):
            dynamic_extensions.add_cdef_file(c_file_path)

        # Correctly register source and cdef files
        dynamic_extensions.add_c_file(c_file_path)
        dynamic_extensions.add_cdef_file(cdef_file_path)

        # Register inline definitions and inline C code
        inline_cdef = "uint64_t my_custom_inline_add(uint64_t a, uint64_t b);"
        inline_c_code = """
#include <stdint.h>
uint64_t my_custom_inline_add(uint64_t a, uint64_t b) {
    return a + b + 10;
}
"""
        dynamic_extensions.add_c_definitions(inline_cdef)
        dynamic_extensions.add_c_code(inline_c_code)

        # Assert files are added (including the generated temp file for inline C code)
        added_files = dynamic_extensions.get_added_files()
        assert c_file_path in added_files
        assert len(added_files) == 2  # c_file_path + temp c file

        # 5. Compile the extension together with the library into a custom output dir
        output_dir = os.path.join(user_dir, "out")
        dest_path = dynamic_extensions.compile(output_dir=output_dir)

        assert os.path.exists(dest_path)

        # 6. Verify that the library is reloaded and the new functions are callable
        # directly from engine.lib!
        assert hasattr(engine.lib, "my_custom_add")
        assert hasattr(engine.lib, "engine_via_custom")
        assert hasattr(engine.lib, "my_custom_inline_add")

        # Call the custom C function and verify logic
        res = engine.lib.my_custom_add(100, 250)
        assert res == 350

        # Call the function that invokes internal library functions
        active = engine.ffi.string(engine.lib.vfhe_engine_active())
        assert engine.ffi.string(engine.lib.engine_via_custom()) == active

        # Call the inline C code function and verify logic
        res_inline = engine.lib.my_custom_inline_add(20, 30)
        assert res_inline == 60

        # Clean up and assert temp files are deleted
        dynamic_extensions.clear_extensions()
        for f in dynamic_extensions.get_added_files():
            assert not os.path.exists(f)


def test_the_swap_does_not_need_the_engine_already_imported():
    """`update_cffi_references` installs the new handles on `vfhe.engine`, so
    it has to import it rather than find it.

    A caller that keeps its own handles lazy -- which is what to do when a
    swap can replace them -- reaches this call with `vfhe.engine` unimported.
    Reading `sys.modules` there raised `KeyError` from the one call meant to
    install its module, and a caller guarding the load with `except
    Exception` reads that as "this build is unusable" and recompiles on every
    run. Run in a subprocess, because nothing can un-import the engine here.
    """
    script = textwrap.dedent(
        """
        import sys
        from vfhe.dynamic_extensions import update_cffi_references

        assert "vfhe.engine" not in sys.modules, "the engine must start unimported"

        marker_ffi, marker_lib = object(), object()
        update_cffi_references(marker_ffi, marker_lib)

        import vfhe.engine
        assert vfhe.engine.lib is marker_lib, "the swap did not stick"
        assert vfhe.engine.ffi is marker_ffi
        assert vfhe.engine.libvfhe.lib is marker_lib
        print("ok")
        """
    )
    result = subprocess.run(  # noqa: S603 - this interpreter, a fixed script
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


@pytest.mark.usefixtures("restore_native_lib")
def test_the_module_name_carries_the_engine():
    """Two engines' builds of one set of sources are different modules.

    The public headers' types change under the engine's flags and its kernels
    need the ISA, so a name that omits the engine lets an output directory
    hold only whichever was compiled last -- and hands a caller that loads the
    cached module by name one its process cannot run.
    """
    with tempfile.TemporaryDirectory() as user_dir:
        source = os.path.join(user_dir, "named.c")
        with open(source, "w") as f:
            f.write("#include <util.h>\nuint64_t named(void) { return 1; }\n")
        dynamic_extensions.clear_extensions()
        dynamic_extensions.add_c_file(source)
        dynamic_extensions.add_c_definitions("uint64_t named(void);")

        digest = _build_module._inputs_hash()[:16]
        active = _build_module._active_engine()
        dest = dynamic_extensions.compile(output_dir=user_dir)

        name = os.path.basename(dest)
        assert name.startswith(f"_vfhe_custom_{active}_{digest}")
        # The name the other engine's build of these same sources would take
        # must not be this one -- which is what naming by the hash alone did.
        assert not name.startswith(f"_vfhe_custom_{digest}")
        dynamic_extensions.clear_extensions()
