"""
Prepara o dataset LFW para o protocolo de verificação 1:1.

Requisitos:
  - Baixar o LFW do Kaggle: https://www.kaggle.com/datasets/jessicali9530/lfw-dataset
  - Descompactar em data/lfw/ (deve conter a pasta lfw_funneled/)

Saída:
  data/test_pairs/lfw_pairs_genuine.csv
  data/test_pairs/lfw_pairs_impostor.csv

Configuração:
  LFW_PEOPLE = 200 (pessoas com 5+ imagens, seed fixa = 42)
  LFW_IMAGES_PER_PERSON = 5
  Pares genuínos: C(5,2) x 200 = 2.000
  Pares impostores: ~2.000 (amostrados aleatoriamente, balanceado)
"""

import os
import sys
import csv
import random
from itertools import combinations
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

LFW_DIR = Path(__file__).parent / "lfw" / "lfw-funneled"
PAIRS_DIR = Path(__file__).parent / "test_pairs"
IMAGES_PER_PERSON = 5
NUM_PEOPLE = 200
RANDOM_SEED = 42


def load_people(lfw_dir: Path, min_images: int) -> dict[str, list[Path]]:
    people = {}
    for person_dir in sorted(lfw_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        images = sorted(person_dir.glob("*.jpg"))
        if len(images) >= min_images:
            people[person_dir.name] = images
    return people


def generate_pairs(people: dict[str, list[Path]], images_per_person: int, seed: int):
    rng = random.Random(seed)
    person_names = list(people.keys())

    genuine_pairs = []
    for name in person_names:
        selected = rng.sample(people[name], images_per_person)
        for img1, img2 in combinations(selected, 2):
            genuine_pairs.append((str(img1), str(img2), 1))

    impostor_pairs = []
    target = len(genuine_pairs)
    attempts = 0
    impostor_set = set()
    while len(impostor_pairs) < target and attempts < target * 20:
        attempts += 1
        name_a, name_b = rng.sample(person_names, 2)
        img_a = rng.choice(people[name_a][:images_per_person])
        img_b = rng.choice(people[name_b][:images_per_person])
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
    if not LFW_DIR.exists():
        print(f"[ERRO] Pasta LFW não encontrada: {LFW_DIR}")
        print("Baixe em: https://www.kaggle.com/datasets/jessicali9530/lfw-dataset")
        print("Descompacte em data/lfw/ de forma que exista data/lfw/lfw_funneled/")
        return

    print(f"Carregando pessoas com >= {IMAGES_PER_PERSON} imagens em {LFW_DIR}...")
    people = load_people(LFW_DIR, IMAGES_PER_PERSON)
    print(f"  Encontradas: {len(people)} pessoas elegíveis")

    if len(people) < NUM_PEOPLE:
        print(f"[AVISO] Apenas {len(people)} pessoas com {IMAGES_PER_PERSON}+ imagens. "
              f"Usando todas.")
        num_people = len(people)
    else:
        num_people = NUM_PEOPLE

    rng = random.Random(RANDOM_SEED)
    selected_names = rng.sample(list(people.keys()), num_people)
    selected_people = {n: people[n] for n in selected_names}
    print(f"  Selecionadas: {num_people} pessoas (seed={RANDOM_SEED})")

    print("Gerando pares...")
    genuine, impostor = generate_pairs(selected_people, IMAGES_PER_PERSON, RANDOM_SEED)

    print(f"  Pares genuínos: {len(genuine)}")
    print(f"  Pares impostores: {len(impostor)}")

    save_csv(genuine, PAIRS_DIR / "lfw_pairs_genuine.csv")
    save_csv(impostor, PAIRS_DIR / "lfw_pairs_impostor.csv")

    print("\nPreparação do LFW concluída.")


if __name__ == "__main__":
    main()
