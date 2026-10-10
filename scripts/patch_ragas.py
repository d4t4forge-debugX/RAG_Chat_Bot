import importlib.util
import os
import shutil
import subprocess
import sys

# the two import lines in ragas 0.3.9's llms/base.py that break on current langchain-community
OLD = (
    "from langchain_community.chat_models.vertexai import ChatVertexAI\n"
    "from langchain_community.llms import VertexAI\n"
)

# replacement: try the old path, then langchain_google_vertexai, then harmless stand-in classes
NEW = (
    "try:\n"
    "    from langchain_community.chat_models.vertexai import ChatVertexAI\n"
    "    from langchain_community.llms import VertexAI\n"
    "except ImportError:\n"
    "    try:\n"
    "        from langchain_google_vertexai import ChatVertexAI, VertexAI\n"
    "    except Exception:\n"
    "        class ChatVertexAI:\n"
    "            pass\n"
    "\n"
    "        class VertexAI:\n"
    "            pass\n"
)


# pure function: returns (new_source, status) where status is "patched", "already_patched" or "unexpected"
def apply_patch(source):
    if NEW in source:
        return source, "already_patched"
    if OLD in source:
        return source.replace(OLD, NEW, 1), "patched"
    return source, "unexpected"


# finds ragas's llms/base.py on disk WITHOUT importing ragas (importing it is exactly what crashes)
def locate_base_py():
    spec = importlib.util.find_spec("ragas")
    if spec is None or not spec.origin:
        return None
    return os.path.join(os.path.dirname(spec.origin), "llms", "base.py")


# deletes ragas's compiled-bytecode caches so a stale .pyc can't hide the patch
def clear_pycache(ragas_dir):
    for root, dirs, _ in os.walk(ragas_dir):
        for d in dirs:
            if d == "__pycache__":
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)


# applies the patch to the installed ragas (with a one-time backup), then checks that "import ragas" works
def main():
    path = locate_base_py()
    if path is None or not os.path.exists(path):
        print("ragas is not installed in this environment (pip install ragas==0.3.9 first).")
        sys.exit(1)

    with open(path) as f:
        source = f.read()

    new_source, status = apply_patch(source)

    if status == "unexpected":
        print(f"{path} does not contain the expected import lines; not touching it.")
        print("(Possibly a different ragas version. This patch targets ragas==0.3.9.)")
        sys.exit(1)

    if status == "patched":
        backup = path + ".orig"
        if not os.path.exists(backup):
            shutil.copy(path, backup)
        with open(path, "w") as f:
            f.write(new_source)
        clear_pycache(os.path.dirname(os.path.dirname(path)))
        print("Patched:", path)
    else:
        print("Already patched:", path)

    check = subprocess.run([sys.executable, "-c", "import ragas"], capture_output=True, text=True)
    if check.returncode == 0:
        print("Verified: 'import ragas' works.")
    else:
        print("'import ragas' still fails:")
        print(check.stderr[-600:])
        sys.exit(1)


if __name__ == "__main__":
    main()