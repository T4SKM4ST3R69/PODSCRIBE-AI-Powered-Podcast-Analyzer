import sys

print("=" * 60)
print("PODSCRIBE ENVIRONMENT TEST")
print("=" * 60)

# Test results storage
passed = []
failed = []

# 1. Test FFmpeg
print("\n1. Testing FFmpeg...")
try:
    import subprocess

    result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        version = result.stdout.split('version')[1].split()[0]
        print(f"   ✓ FFmpeg installed: {version}")
        passed.append("FFmpeg")
    else:
        print("   ✗ FFmpeg not found")
        failed.append("FFmpeg")
except Exception as e:
    print(f"   ✗ FFmpeg test failed: {e}")
    failed.append("FFmpeg")

# 2. Test PyTorch
print("\n2. Testing PyTorch...")
try:
    import torch

    print(f"   ✓ PyTorch version: {torch.__version__}")

    # Test CUDA
    if torch.cuda.is_available():
        print(f"   ✓ CUDA available: {torch.version.cuda}")
        print(f"   ✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   ✓ GPU Count: {torch.cuda.device_count()}")

        # Test actual tensor on GPU
        try:
            x = torch.rand(3, 3).cuda()
            print(f"   ✓ GPU tensor creation successful")
        except Exception as e:
            print(f"   ⚠ GPU tensor creation failed: {e}")
    else:
        print("   ⚠ CUDA not available (CPU only)")

    passed.append("PyTorch")
except ImportError as e:
    print(f"   ✗ PyTorch not installed: {e}")
    failed.append("PyTorch")
except Exception as e:
    print(f"   ✗ PyTorch test failed: {e}")
    failed.append("PyTorch")

# 3. Test Torchaudio
print("\n3. Testing Torchaudio...")
try:
    import torchaudio

    print(f"   ✓ Torchaudio version: {torchaudio.__version__}")
    passed.append("Torchaudio")
except ImportError:
    print("   ✗ Torchaudio not installed")
    failed.append("Torchaudio")

# 4. Test Torchvision
print("\n4. Testing Torchvision...")
try:
    import torchvision

    print(f"   ✓ Torchvision version: {torchvision.__version__}")
    passed.append("Torchvision")
except ImportError:
    print("   ✗ Torchvision not installed")
    failed.append("Torchvision")

# 5. Test WhisperX
print("\n5. Testing WhisperX...")
try:
    import whisperx

    print(f"   ✓ WhisperX installed")

    # Test if models can be loaded
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   ✓ WhisperX will use: {device}")
    except:
        pass

    passed.append("WhisperX")
except ImportError as e:
    print(f"   ✗ WhisperX not installed: {e}")
    failed.append("WhisperX")

# 6. Test Pyannote
print("\n6. Testing Pyannote Audio...")
try:
    import pyannote.audio

    print(f"   ✓ Pyannote Audio installed")
    passed.append("Pyannote")
except ImportError:
    print("   ✗ Pyannote Audio not installed")
    failed.append("Pyannote")

# 7. Test ChromaDB
print("\n7. Testing ChromaDB...")
try:
    import chromadb

    print(f"   ✓ ChromaDB version: {chromadb.__version__}")

    # Test basic functionality
    try:
        client = chromadb.Client()
        print(f"   ✓ ChromaDB client initialized")
    except Exception as e:
        print(f"   ⚠ ChromaDB client warning: {e}")

    passed.append("ChromaDB")
except ImportError:
    print("   ✗ ChromaDB not installed")
    failed.append("ChromaDB")

# 8. Test Sentence Transformers
print("\n8. Testing Sentence Transformers...")
try:
    import sentence_transformers

    print(f"   ✓ Sentence Transformers version: {sentence_transformers.__version__}")
    passed.append("Sentence Transformers")
except ImportError:
    print("   ✗ Sentence Transformers not installed")
    failed.append("Sentence Transformers")

# 9. Test MoviePy
print("\n9. Testing MoviePy...")
try:
    import moviepy

    print(f"   ✓ MoviePy installed")

    # Test if FFmpeg is accessible to MoviePy
    try:
        from moviepy.config import check

        check()
        print(f"   ✓ MoviePy can access FFmpeg")
    except:
        print(f"   ⚠ MoviePy FFmpeg check warning")

    passed.append("MoviePy")
except ImportError:
    print("   ✗ MoviePy not installed")
    failed.append("MoviePy")

# 10. Test Pydub
print("\n10. Testing Pydub...")
try:
    from pydub import AudioSegment

    print(f"   ✓ Pydub installed")

    # Check if FFmpeg is accessible
    from pydub.utils import which

    ffmpeg_path = which("ffmpeg")
    if ffmpeg_path:
        print(f"   ✓ Pydub can access FFmpeg")
    else:
        print(f"   ⚠ Pydub cannot find FFmpeg")

    passed.append("Pydub")
except ImportError:
    print("   ✗ Pydub not installed")
    failed.append("Pydub")

# 11. Test Streamlit
print("\n11. Testing Streamlit...")
try:
    import streamlit

    print(f"   ✓ Streamlit version: {streamlit.__version__}")
    passed.append("Streamlit")
except ImportError:
    print("   ✗ Streamlit not installed")
    failed.append("Streamlit")

# 12. Test Python-dotenv
print("\n12. Testing Python-dotenv...")
try:
    import dotenv

    print(f"   ✓ Python-dotenv installed")
    passed.append("Python-dotenv")
except ImportError:
    print("   ✗ Python-dotenv not installed")
    failed.append("Python-dotenv")

# 13. Test Pandas
print("\n13. Testing Pandas...")
try:
    import pandas

    print(f"   ✓ Pandas version: {pandas.__version__}")
    passed.append("Pandas")
except ImportError:
    print("   ✗ Pandas not installed")
    failed.append("Pandas")

# 14. Test NumPy
print("\n14. Testing NumPy...")
try:
    import numpy

    print(f"   ✓ NumPy version: {numpy.__version__}")
    passed.append("NumPy")
except ImportError:
    print("   ✗ NumPy not installed")
    failed.append("NumPy")

# 15. Test Transformers
print("\n15. Testing Transformers...")
try:
    import transformers

    print(f"   ✓ Transformers version: {transformers.__version__}")
    passed.append("Transformers")
except ImportError:
    print("   ✗ Transformers not installed")
    failed.append("Transformers")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"✓ Passed: {len(passed)}/15")
print(f"✗ Failed: {len(failed)}/15")

if failed:
    print(f"\nMissing dependencies: {', '.join(failed)}")
    print("\nTo install missing packages, run:")
    print("pip install " + " ".join(failed).lower().replace(" ", "-"))
else:
    print("\n🎉 All dependencies installed and working!")

print("=" * 60)
