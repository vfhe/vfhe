# SPDX-License-Identifier: Apache-2.0
import os
import sys
import tempfile
import shutil
import pytest

from vfhe.misc import libvfhe
from vfhe.misc import dynamic_extensions


def test_dynamic_extensions_flow():
    # 1. Create a temporary directory for our user C extension files
    with tempfile.TemporaryDirectory() as user_dir:
        # Create a target directory for vfhe.h
        header_dir = os.path.join(user_dir, "include")
        os.makedirs(header_dir, exist_ok=True)
        
        # Generate the vfhe.h wrapper
        dynamic_extensions.create_headers(header_dir)
        vfhe_h_path = os.path.join(header_dir, "vfhe.h")
        assert os.path.exists(vfhe_h_path)
        
        # Verify that vfhe.h contains at least some library headers
        with open(vfhe_h_path, "r") as f:
            vfhe_h_content = f.read()
        assert "arith.h" in vfhe_h_content
        
        vfhe_h_clean = vfhe_h_path.replace('\\', '/')
        # 2. Write custom C source code that calls an internal library function
        # (vfhe_build_is_portable)
        c_code = f"""
#include "{vfhe_h_clean}"

uint64_t my_custom_add(uint64_t a, uint64_t b) {{
    return a + b;
}}

int check_portable_via_custom(void) {{
    return vfhe_build_is_portable();
}}
"""
        c_file_path = os.path.join(user_dir, "my_extension.c")
        with open(c_file_path, "w") as f:
            f.write(c_code)
            
        # 3. Write corresponding CFFI declarations
        cdef_code = """
uint64_t my_custom_add(uint64_t a, uint64_t b);
int check_portable_via_custom(void);
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
        # directly from libvfhe.lib!
        assert hasattr(libvfhe.lib, "my_custom_add")
        assert hasattr(libvfhe.lib, "check_portable_via_custom")
        assert hasattr(libvfhe.lib, "my_custom_inline_add")
        
        # Call the custom C function and verify logic
        res = libvfhe.lib.my_custom_add(100, 250)
        assert res == 350
        
        # Call the function that invokes internal library functions
        is_portable = libvfhe.lib.vfhe_build_is_portable()
        res_portable = libvfhe.lib.check_portable_via_custom()
        assert res_portable == is_portable
        
        # Call the inline C code function and verify logic
        res_inline = libvfhe.lib.my_custom_inline_add(20, 30)
        assert res_inline == 60

        # Clean up and assert temp files are deleted
        dynamic_extensions.clear_extensions()
        for f in dynamic_extensions.get_added_files():
            assert not os.path.exists(f)
