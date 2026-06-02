"""
Prepara os pares da base própria controlada para verificação 1:1.

Estrutura esperada em data/known_faces/:
  pessoa_01/
    img_01.jpg ... img_20.jpg
  pessoa_02/
  ...
  pessoa_10/

Saída:
  data/test_pairs/own_pairs_genuine.csv
  data/test_pairs/own_pairs_impostor.csv

Configuração:
  10 indivíduos x 20 imagens = 200 imagens
  Genuínos: C(20,2) x 10 = 1.900 pares
  Impostores: ~1.900 pares amostrados (balanceado)
"""

import os
import sys
import csv
import random
from itertools import combinations
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

OWN_BASE_DIR = Path(__file__).parent / "known_faces"
PAIRS_DIR = Path(__file__).parent / "test_pairs"
EXPECTED_PEOPLE = 10
EXPECTED_IMAGES = 20
RANDOM_SEED = 42


def load_own_base(base_dir: Path) -> dict[str, list[Path]]:
    people = {}
    if not base_dir.exists():
        return people
    for person_dir in sorted(base_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        images = sorted(
            p for p in person_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        if images:
            people[person_dir.name] = images
    return people


def validate(people: dict[str, list[Path]]):
    warnings = []
    if len(people) < EXPECTED_PEOPLE:
        warnings.append(
            f"[AVISO] Apenas {len(people)} pastas encontradas "
            f"(esperado: {EXPECTED_PEOPLE})"
        )
    for name, imgs in people.items():
        if len(imgs) < EXPECTED_IMAGES:
            warnings.append(
                f"[AVISO] {name}: {len(imgs)} imagens "
                f"(esperado: {EXPECTED_IMAGES})"
            )
    for w in warnings:
        print(w)
    return len(warnings) == 0


def generate_pairs(people: dict[str, list[Path]], seed: int):
    rng = random.Random(seed)
    person_names = list(people.keys())

    genuine_pairs = []
    for name in person_names:
        for img1, img2 in combinations(people[name], 2):
            genuine_pairs.append((str(img1), str(img2), 1))

    impostor_pairs = []
    target = len(genuine_pairs)
    impostor_set = set()
    attempts = 0
    while len(impostor_pairs) < target and attempts < target * 20:
        attempts += 1
        name_a, name_b = rng.sample(person_names, 2)
        img_a = rng.choice(people[name_a])
        img_b = rng.choice(people[name_b])
        key = (str(img_a), str(img_b))
        if key not in impostor_set:
            impostor_set.add(key)
            impostor_pairs.append((str(img_a), str(img_b), 0))

    return genuine_pairs, impostor_pairs


def save_csv(pairs: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["img1_path", "img2_path", "label"])
        writer.writerows(pairs)
    print(f"  Salvo: {path} ({len(pairs)} pares)")


def main():
    if not OWN_BASE_DIR.exists():
        print(f"[ERRO] Pasta não encontrada: {OWN_BASE_DIR}")
        print("Crie a estrutura:")
        print("  data/known_faces/pessoa_01/img_01.jpg ... img_20.jpg")
        print("  data/known_faces/pessoa_02/ ...")
        return

    print(f"Carregando base própria em {OWN_BASE_DIR}...")
    people = load_own_base(OWN_BASE_DIR)
    print(f"  Pessoas encontradas: {len(people)}")
    for name, imgs in people.items():
        print(f"    {name}: {len(imgs)} imagens")

    validate(people)

    print("\nGerando pares...")
    genuine, impostor = generate_pairs(people, RANDOM_SEED)

    print(f"  Pares genuínos: {len(genuine)}")
    print(f"  Pares impostores: {len(impostor)}")

    save_csv(genuine, PAIRS_DIR / "own_pairs_genuine.csv")
    save_csv(impostor, PAIRS_DIR / "own_pairs_impostor.csv")

    print("\nPreparação da base própria concluída.")


if __name__ == "__main__":
    main()
