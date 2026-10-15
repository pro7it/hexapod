from pathlib import Path
import pymeshlab

SOURCE = Path("stl")
DEST = Path("stl_decimate")

# Цільова кількість граней: приблизно 10–20% оригіналу.
TARGET_RATIO = 0.15
MIN_FACES = 200

DEST.mkdir(parents=True, exist_ok=True)

files = sorted(SOURCE.glob("*.stl"))

if not files:
    raise SystemExit(f"Не знайдено STL-файлів у {SOURCE.resolve()}")

print(f"Знайдено моделей: {len(files)}")

for src in files:
    try:
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(src))

        original_faces = ms.current_mesh().face_number()
        target_faces = max(MIN_FACES, int(original_faces * TARGET_RATIO))

        if original_faces <= MIN_FACES:
            print(f"[SKIP] {src.name}: лише {original_faces} faces")
            ms.save_current_mesh(str(DEST / src.name))
            continue

        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=target_faces,
            preservenormal=True,
            preservetopology=True,
        )

        final_faces = ms.current_mesh().face_number()
        output = DEST / src.name
        ms.save_current_mesh(str(output))

        reduction = 100 * (1 - final_faces / original_faces)
        print(
            f"[OK] {src.name}: {original_faces} -> {final_faces} "
            f"faces (зменшення на {reduction:.1f}%)"
        )

    except Exception as exc:
        print(f"[ERROR] {src.name}: {exc}")

print(f"\nГотово. Результати: {DEST.resolve()}")
