import subprocess
import sys
import os

def build_rust_sandbox():
    """
    Compiles the zero-allocation Rust execution engine.
    Must be run before initializing the Eunoia Raiden training loop.
    """
    print("[Eunoia Raiden] Initializing bare-metal Rust compilation via Maturin...")
    dsl_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        # We enforce --release to guarantee opt-level=3 and LTO
        result = subprocess.run(
            [sys.executable, "-m", "maturin", "develop", "--release"],
            cwd=dsl_dir,
            check=True,
            capture_output=True,
            text=True
        )
        print("[Eunoia Raiden] Execution Sandbox built successfully.")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("[FATAL] FFI Boundary compilation failed. Do not proceed to training.")
        print(e.stderr)
        sys.exit(1)

if __name__ == "__main__":
    build_rust_sandbox()